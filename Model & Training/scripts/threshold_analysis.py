"""
Best-threshold analysis for "is this filter invalid?" — every model config
trained across this project (23 individual trainings: every ratio split
and every CV fold, across all 4 label schemes).

Every training script so far decided valid-vs-invalid with a bare
`argmax()` on the model's logits — equivalent to an untuned, default 0.5
probability cutoff (see the conversation: no calibration, no ROC/PR
analysis was ever done). This script instead:

  1. Re-trains each config (same fixed model config as every other sweep:
     MobileNetV2, last block fine-tuned, lr=1e-3, 10 epochs, seed=42 —
     only the label scheme/ratio/fold varies) and captures full predicted
     PROBABILITIES on its test set, not just the argmax class.
  2. Collapses those probabilities to P(invalid) = sum of probability
     mass on any "invalid*" class (one-vs-rest; "invalid" is the
     actionable class throughout this project's docs — a missed invalid
     costs more than a false alarm).
  3. For CV configs, pools every fold's held-out predictions into one
     set (more data points for the curves than any single small split).
  4. Computes, per config: ROC curve + AUC, precision-recall curve +
     average precision, a calibration curve, and two candidate best
     thresholds — Youden's J (max TPR-FPR, balances both error types)
     and the F1-maximizing threshold from the PR curve (favors invalid
     recall/precision specifically, the class that matters most here).
  5. Saves per-config results to JSON and three comparison plots (ROC,
     PR, calibration — all configs overlaid) instead of ~50 separate
     small ones.

This is a long run: 23 trainings at 10 epochs each on CPU. Use --quick
(1 epoch, 3 representative configs only) to sanity-check the pipeline
before committing to the full run.

Keep this file and notebooks/threshold_analysis.ipynb in sync.

Usage:
  python threshold_analysis.py            # full run, all 23 configs
  python threshold_analysis.py --quick    # fast smoke test
"""

import argparse
import copy
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMPLE_DIR = os.path.join(TRAINING_DIR, "pictures", "simple version")
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version")
NOEMPTY_DIR = os.path.join(TRAINING_DIR, "pictures", "no-empty version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
OUT_DIR = os.path.join(TRAINING_DIR, "models", "threshold_analysis")

EXTENDED_LABEL_TO_FOLDER = {
    "valid_day": "תקין יום", "valid_night": "תקין לילה",
    "invalid_day": "לא תקין יום", "invalid_night": "לא תקין לילה", "empty": "מסנן ריק",
}
NOEMPTY_SPLIT_LABEL_TO_FOLDER = {
    "valid_day": "תקין יום", "valid_night": "תקין לילה",
    "invalid_day": "לא תקין יום", "invalid_night": "לא תקין לילה",
}

SEED = 42
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FIXED_CONFIG = {"backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 10}


# ---------------------------------------------------------------- data

def mask_timestamp(img):
    img = img.copy()
    w, h = img.size
    band_h = int(h * 0.10)
    ImageDraw.Draw(img).rectangle([0, 0, w, band_h], fill=(0, 0, 0))
    return img


def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.RandomRotation(6),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def lite_path_fn(row):
    return os.path.join(SIMPLE_DIR, row["label"], row["filename"])


def extended_path_fn(row):
    return os.path.join(EXTENDED_DIR, EXTENDED_LABEL_TO_FOLDER[row["extended_label"]], row["filename"])


def noempty_path_fn(row):
    return os.path.join(NOEMPTY_DIR, NOEMPTY_SPLIT_LABEL_TO_FOLDER[row["noempty_split_label"]], row["filename"])


class GypsumDataset(Dataset):
    def __init__(self, dataframe, path_fn, label_col, class_to_idx, transform):
        self.df = dataframe.reset_index(drop=True)
        self.path_fn = path_fn
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.path_fn(row)).convert("RGB")
        img = mask_timestamp(img)
        img = self.transform(img)
        return img, self.class_to_idx[row[self.label_col]]


# ---------------------------------------------------------------- model

def build_model(backbone_name, unfreeze_last_block, num_classes):
    if backbone_name != "mobilenet_v2":
        raise ValueError(f"unknown backbone: {backbone_name}")
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_last_block:
        for p in model.features[-1].parameters():
            p.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(dataframe, label_col, classes, class_to_idx):
    counts = dataframe[label_col].value_counts()
    total = len(dataframe)
    weights = torch.zeros(len(classes))
    for c in classes:
        count = counts.get(c, 0)
        weights[class_to_idx[c]] = total / (len(classes) * count) if count else 0.0
    return weights


def run_training(epochs, train_loader, val_loader, class_weights, device):
    model = build_model(FIXED_CONFIG["backbone"], FIXED_CONFIG["unfreeze_last_block"], len(class_weights)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=FIXED_CONFIG["lr"])

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        vloss = vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += criterion(out, y).item() * x.size(0)
                vn += x.size(0)
        val_loss = vloss / max(vn, 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_loss


def predict_probs(model, loader, device):
    """Full softmax probability vector per sample, not just argmax."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


# ---------------------------------------------------------------- one training run -> (p_invalid, is_invalid)

def run_split(df, label_col, path_fn, invalid_classes, train_df_full, test_df, epochs, device, sampler=False):
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    invalid_idx = [class_to_idx[c] for c in invalid_classes]

    train_df, val_df = train_test_split(
        train_df_full, test_size=0.15, stratify=train_df_full[label_col], random_state=SEED,
    )
    train_transform, eval_transform = make_transforms()
    train_ds = GypsumDataset(train_df, path_fn, label_col, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, path_fn, label_col, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, path_fn, label_col, class_to_idx, eval_transform)

    class_weights = compute_class_weights(train_df, label_col, classes, class_to_idx)

    if sampler:
        counts = train_df[label_col].value_counts()
        sample_weights = train_df[label_col].map(lambda c: 1.0 / counts[c]).values
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, best_val_loss = run_training(epochs, train_loader, val_loader, class_weights, device)

    probs, labels = predict_probs(model, test_loader, device)
    p_invalid = probs[:, invalid_idx].sum(axis=1)
    is_invalid = np.isin(labels, invalid_idx).astype(int)
    return p_invalid, is_invalid, best_val_loss


# ---------------------------------------------------------------- config registry

def build_registry(df):
    """Returns {config_name: {"kind": "ratio"|"cv", ...}} for all 23 runs,
    grouped into the 17 result rows used throughout this project."""
    reg = {}

    def ratio(name, label_col, path_fn, invalid_classes, split_col, sampler=False):
        reg[name] = dict(kind="ratio", label_col=label_col, path_fn=path_fn,
                          invalid_classes=invalid_classes, split_col=split_col, sampler=sampler)

    def cv(name, label_col, path_fn, invalid_classes, fold_col):
        k = df[fold_col].nunique()
        reg[name] = dict(kind="cv", label_col=label_col, path_fn=path_fn,
                          invalid_classes=invalid_classes, fold_col=fold_col, k=k)

    ratio("lite_80_20", "label", lite_path_fn, ["invalid"], "split")
    ratio("lite_75_25", "label", lite_path_fn, ["invalid"], "split_75")
    ratio("lite_70_30", "label", lite_path_fn, ["invalid"], "split_70")
    cv("lite_cv", "label", lite_path_fn, ["invalid"], "fold_lite")
    ratio("lite_80_20_sampler", "label", lite_path_fn, ["invalid"], "split", sampler=True)

    ratio("extended_80_20", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "split_extended")
    ratio("extended_75_25", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "split_extended_75")
    ratio("extended_70_30", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "split_extended_70")
    cv("extended_cv", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "fold_extended")

    ratio("collapsed_80_20", "noempty_label", noempty_path_fn, ["invalid"], "split_noempty")
    ratio("collapsed_75_25", "noempty_label", noempty_path_fn, ["invalid"], "split_noempty_75")
    ratio("collapsed_70_30", "noempty_label", noempty_path_fn, ["invalid"], "split_noempty_70")
    cv("collapsed_cv", "noempty_label", noempty_path_fn, ["invalid"], "fold_noempty")

    ratio("split_80_20", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "split_noempty4")
    ratio("split_75_25", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "split_noempty4_75")
    ratio("split_70_30", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "split_noempty4_70")
    cv("split_cv", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "fold_noempty4")

    return reg


def run_config(df, name, cfg, epochs, device):
    print(f"\n=== {name} ({cfg['kind']}) ===")
    if cfg["kind"] == "ratio":
        train_df_full = df[df[cfg["split_col"]] == "train"].reset_index(drop=True)
        test_df = df[df[cfg["split_col"]] == "test"].reset_index(drop=True)
        p_invalid, is_invalid, val_loss = run_split(
            df, cfg["label_col"], cfg["path_fn"], cfg["invalid_classes"],
            train_df_full, test_df, epochs, device, sampler=cfg["sampler"],
        )
        print(f"  n_test={len(test_df)}  val_loss={val_loss:.3f}")
    else:
        all_p, all_y = [], []
        for fold_i in range(cfg["k"]):
            print(f"  fold {fold_i + 1}/{cfg['k']}")
            test_df = df[df[cfg["fold_col"]] == str(fold_i)].reset_index(drop=True)
            train_df_full = df[df[cfg["fold_col"]] != str(fold_i)].reset_index(drop=True)
            p_invalid_f, is_invalid_f, val_loss = run_split(
                df, cfg["label_col"], cfg["path_fn"], cfg["invalid_classes"],
                train_df_full, test_df, epochs, device,
            )
            all_p.append(p_invalid_f)
            all_y.append(is_invalid_f)
        p_invalid = np.concatenate(all_p)
        is_invalid = np.concatenate(all_y)
        print(f"  pooled n_test={len(is_invalid)}")

    return p_invalid, is_invalid


# ---------------------------------------------------------------- threshold metrics

def analyze(name, p_invalid, is_invalid):
    n_pos, n_neg = int(is_invalid.sum()), int((1 - is_invalid).sum())
    if n_pos == 0 or n_neg == 0:
        print(f"  SKIPPING metrics for '{name}': only one class present in pooled test data (pos={n_pos}, neg={n_neg}).")
        return None

    fpr, tpr, roc_thresh = roc_curve(is_invalid, p_invalid)
    roc_auc = roc_auc_score(is_invalid, p_invalid)
    precision, recall, pr_thresh = precision_recall_curve(is_invalid, p_invalid)
    ap = average_precision_score(is_invalid, p_invalid)

    # Youden's J: maximize TPR - FPR.
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    youden_threshold = float(roc_thresh[best_j_idx]) if len(roc_thresh) else 0.5

    # F1-maximizing point on the PR curve (precision/recall arrays are one
    # longer than pr_thresh; drop the last point, which has no threshold).
    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12),
        0,
    )
    best_f1_idx = int(np.argmax(f1)) if len(f1) else 0
    f1_threshold = float(pr_thresh[best_f1_idx]) if len(pr_thresh) else 0.5

    frac_pos, mean_pred = calibration_curve(is_invalid, p_invalid, n_bins=5, strategy="uniform")

    print(f"  ROC AUC={roc_auc:.3f}  PR AP={ap:.3f}  "
          f"Youden threshold={youden_threshold:.3f} (TPR={tpr[best_j_idx]:.3f}, FPR={fpr[best_j_idx]:.3f})  "
          f"F1-optimal threshold={f1_threshold:.3f} (P={precision[best_f1_idx]:.3f}, R={recall[best_f1_idx]:.3f})")

    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "roc_auc": float(roc_auc), "pr_ap": float(ap),
        "youden_threshold": youden_threshold,
        "youden_tpr": float(tpr[best_j_idx]), "youden_fpr": float(fpr[best_j_idx]),
        "f1_threshold": f1_threshold,
        "f1_precision": float(precision[best_f1_idx]), "f1_recall": float(recall[best_f1_idx]),
        "default_argmax_threshold": 0.5,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "calibration_curve": {"mean_predicted": mean_pred.tolist(), "frac_positive": frac_pos.tolist()},
    }


def plot_overlays(results, out_dir):
    names = [n for n, r in results.items() if r is not None]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], color=cmap(i % 20),
                label=f"{name} (AUC={r['roc_auc']:.2f})", linewidth=1.3)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — P(invalid) vs. true invalid, all configs")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_overlay.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"], color=cmap(i % 20),
                label=f"{name} (AP={r['pr_ap']:.2f})", linewidth=1.3)
    ax.set_xlabel("Recall (invalid)"); ax.set_ylabel("Precision (invalid)")
    ax.set_title("Precision-Recall — invalid class, all configs")
    ax.legend(fontsize=6, loc="lower left", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_overlay.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfectly calibrated")
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["calibration_curve"]["mean_predicted"], r["calibration_curve"]["frac_positive"],
                marker="o", markersize=3, color=cmap(i % 20), label=name, linewidth=1.1)
    ax.set_xlabel("Mean predicted P(invalid)"); ax.set_ylabel("Observed fraction invalid")
    ax.set_title("Calibration — all configs")
    ax.legend(fontsize=6, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_overlay.png"), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                         help="1 epoch, 3 representative configs only, for a pipeline smoke test.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = pd.read_csv(SPLIT_CSV, dtype={
        "fold_lite": str, "fold_extended": str, "fold_noempty": str, "fold_noempty4": str,
    })

    registry = build_registry(df)
    epochs = 1 if args.quick else FIXED_CONFIG["epochs"]
    names_to_run = ["lite_80_20", "extended_cv", "split_80_20"] if args.quick else list(registry.keys())
    if args.quick:
        print(f"--quick: running only {names_to_run}, 1 epoch each (smoke test, not real results)")

    results = {}
    for name in names_to_run:
        p_invalid, is_invalid = run_config(df, name, registry[name], epochs, device)
        results[name] = analyze(name, p_invalid, is_invalid)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n=== Summary: recommended thresholds per config ===")
    header = f"{'config':22s} {'ROC AUC':>8s} {'PR AP':>7s} {'Youden thr':>11s} {'F1 thr':>8s}"
    print(header)
    for name in names_to_run:
        r = results[name]
        if r is None:
            print(f"{name:22s}  (skipped — one class only)")
            continue
        print(f"{name:22s} {r['roc_auc']:8.3f} {r['pr_ap']:7.3f} {r['youden_threshold']:11.3f} {r['f1_threshold']:8.3f}")

    valid_results = {k: v for k, v in results.items() if v is not None}
    if not args.quick and valid_results:
        plot_overlays(valid_results, OUT_DIR)
        print(f"Saved roc_overlay.png, pr_overlay.png, calibration_overlay.png to {OUT_DIR}")

    with open(os.path.join(OUT_DIR, "threshold_results.json"), "w") as f:
        json.dump({"config": FIXED_CONFIG, "quick": args.quick, "results": results}, f, indent=2)
    print(f"Saved {os.path.join(OUT_DIR, 'threshold_results.json')}")


if __name__ == "__main__":
    main()
