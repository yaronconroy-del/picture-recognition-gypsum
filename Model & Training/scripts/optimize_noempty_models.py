"""
Runs the same optimization sweep as optimize_models.py, but for the
"no-empty" dataset version (pictures/no-empty version/), where the 8
"empty filter" photos were manually redistributed into invalid_day /
invalid_night instead of being their own class (see
Evaluation & Docs/Project Workflow.md).

Two granularities, same as lite/extended:
  - collapsed (2-class): valid / invalid
  - split (4-class): valid_day / valid_night / invalid_day / invalid_night

For each: ratio sweep (80/20, 75/25, 70/30) and cross-validation. CV was
requested at k=4, but build_dataset_index.py's determine_k found the
scarcest group only supports 3-fold (collapsed) / 2-fold (split) — the
empty photos' timestamps mostly fall inside existing invalid sessions
rather than creating new independent ones, so merging them in didn't add
session diversity.

Same fixed config as every other sweep in this project, so only the label
scheme/ratio/CV varies.

Keep this file and notebooks/optimize_noempty_models.ipynb in sync.

Usage:
  python optimize_noempty_models.py            # full run
  python optimize_noempty_models.py --quick    # fast smoke test (1 epoch each)
"""

import argparse
import copy
import json
import os
import random
import statistics

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
NOEMPTY_DIR = os.path.join(TRAINING_DIR, "pictures", "no-empty version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
OUT_DIR = os.path.join(TRAINING_DIR, "models", "optimization_sweep_noempty")

NOEMPTY_SPLIT_LABEL_TO_FOLDER = {
    "valid_day": "תקין יום",
    "valid_night": "תקין לילה",
    "invalid_day": "לא תקין יום",
    "invalid_night": "לא תקין לילה",
}

SEED = 42
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_CLASSES = ["invalid", "valid"]  # the common 2-class scale everything is read on

FIXED_CONFIG = {"backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 10}


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


def path_fn(row):
    folder = NOEMPTY_SPLIT_LABEL_TO_FOLDER[row["noempty_split_label"]]
    return os.path.join(NOEMPTY_DIR, folder, row["filename"])


class GypsumDataset(Dataset):
    def __init__(self, dataframe, label_col, class_to_idx, transform):
        self.df = dataframe.reset_index(drop=True)
        self.label_col = label_col
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(path_fn(row)).convert("RGB")
        img = mask_timestamp(img)
        img = self.transform(img)
        return img, self.class_to_idx[row[self.label_col]]


def build_model(backbone_name, unfreeze_last_block, num_classes):
    if backbone_name == "mobilenet_v2":
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        model = torchvision.models.mobilenet_v2(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            for p in model.features[-1].parameters():
                p.requires_grad = True
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {backbone_name}")
    return model


def compute_class_weights(dataframe, label_col, classes, class_to_idx):
    counts = dataframe[label_col].value_counts()
    total = len(dataframe)
    weights = torch.zeros(len(classes))
    for c in classes:
        count = counts.get(c, 0)
        weights[class_to_idx[c]] = total / (len(classes) * count) if count else 0.0
    return weights


def run_training(epochs, train_loader, val_loader, class_weights, device, verbose=True):
    model = build_model(
        FIXED_CONFIG["backbone"], FIXED_CONFIG["unfreeze_last_block"], len(class_weights)
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=FIXED_CONFIG["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = running_correct = n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        train_loss, train_acc = running_loss / n, running_correct / n

        model.eval()
        vloss = vcorrect = vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                vloss += loss.item() * x.size(0)
                vcorrect += (out.argmax(1) == y).sum().item()
                vn += x.size(0)
        val_loss, val_acc = vloss / max(vn, 1), vcorrect / max(vn, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"    epoch {epoch + 1:2d}/{epochs}  "
                  f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return np.array(all_preds), np.array(all_labels)


def collapse_split_to_collapsed(label):
    return label.rsplit("_", 1)[0]


def target_scale_metrics(true_names, pred_names):
    report = classification_report(
        true_names, pred_names, labels=TARGET_CLASSES, target_names=TARGET_CLASSES,
        digits=3, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "invalid_recall": report.get("invalid", {}).get("recall", float("nan")),
    }


def train_one(df, scheme, label_col, train_df_full, test_df, epochs, device, verbose=True):
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df, val_df = train_test_split(
        train_df_full, test_size=0.15, stratify=train_df_full[label_col], random_state=SEED,
    )

    train_transform, eval_transform = make_transforms()
    train_ds = GypsumDataset(train_df, label_col, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, label_col, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, label_col, class_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_df, label_col, classes, class_to_idx)

    model, history, best_val_loss = run_training(
        epochs, train_loader, val_loader, class_weights, device, verbose=verbose,
    )

    y_pred, y_true = predict_all(model, test_loader, device)
    true_names = [classes[i] for i in y_true]
    pred_names = [classes[i] for i in y_pred]
    if scheme == "split":
        true_names = [collapse_split_to_collapsed(n) for n in true_names]
        pred_names = [collapse_split_to_collapsed(n) for n in pred_names]

    metrics = target_scale_metrics(true_names, pred_names)
    metrics["best_val_loss"] = best_val_loss
    metrics["n_test"] = len(test_df)
    metrics["n_train"] = len(train_df_full)
    return metrics


def run_ratio(df, scheme, label_col, split_col, epochs, device):
    print(f"\n=== no-empty {scheme} — ratio split ({split_col}) ===")
    train_df_full = df[df[split_col] == "train"].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)
    print(f"train={len(train_df_full)}  test={len(test_df)}")
    m = train_one(df, scheme, label_col, train_df_full, test_df, epochs, device)
    print(f"  -> accuracy={m['accuracy']:.3f}  val_loss={m['best_val_loss']:.3f}  "
          f"macro_f1={m['macro_f1']:.3f}  invalid_recall={m['invalid_recall']:.3f}")
    return m


def run_cv(df, scheme, label_col, fold_col, k, epochs, device):
    print(f"\n=== no-empty {scheme} — {k}-fold CV ({fold_col}) ===")
    fold_metrics = []
    for fold_i in range(k):
        print(f"  fold {fold_i + 1}/{k}")
        test_df = df[df[fold_col] == str(fold_i)].reset_index(drop=True)
        train_df_full = df[df[fold_col] != str(fold_i)].reset_index(drop=True)
        m = train_one(df, scheme, label_col, train_df_full, test_df, epochs, device, verbose=False)
        print(f"    -> accuracy={m['accuracy']:.3f}  invalid_recall={m['invalid_recall']:.3f}  (n_test={m['n_test']})")
        fold_metrics.append(m)

    agg = {}
    for key in ("accuracy", "macro_f1", "invalid_recall", "best_val_loss"):
        vals = [m[key] for m in fold_metrics]
        agg[key] = statistics.mean(vals)
        agg[f"{key}_std"] = statistics.pstdev(vals)
    agg["k"] = k
    print(f"  -> mean accuracy={agg['accuracy']:.3f} (+/-{agg['accuracy_std']:.3f})  "
          f"mean invalid_recall={agg['invalid_recall']:.3f} (+/-{agg['invalid_recall_std']:.3f})")
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 epoch per run, for a pipeline smoke test only.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epochs = 1 if args.quick else FIXED_CONFIG["epochs"]
    df = pd.read_csv(SPLIT_CSV, dtype={"fold_noempty": str, "fold_noempty4": str})

    results = {"config": FIXED_CONFIG, "quick_smoke_test": args.quick}

    results["collapsed_80_20"] = run_ratio(df, "collapsed", "noempty_label", "split_noempty", epochs, device)
    results["collapsed_75_25"] = run_ratio(df, "collapsed", "noempty_label", "split_noempty_75", epochs, device)
    results["collapsed_70_30"] = run_ratio(df, "collapsed", "noempty_label", "split_noempty_70", epochs, device)

    results["split_80_20"] = run_ratio(df, "split", "noempty_split_label", "split_noempty4", epochs, device)
    results["split_75_25"] = run_ratio(df, "split", "noempty_split_label", "split_noempty4_75", epochs, device)
    results["split_70_30"] = run_ratio(df, "split", "noempty_split_label", "split_noempty4_70", epochs, device)

    k_collapsed = df["fold_noempty"].nunique()
    k_split = df["fold_noempty4"].nunique()
    results["collapsed_cv"] = run_cv(df, "collapsed", "noempty_label", "fold_noempty", k_collapsed, epochs, device)
    results["split_cv"] = run_cv(df, "split", "noempty_split_label", "fold_noempty4", k_split, epochs, device)

    print("\n=== Summary (no-empty dataset, all read on the invalid/valid 2-class scale) ===")
    header = f"{'variant':22s} {'accuracy':>10s} {'val_loss':>10s} {'macro_f1':>10s} {'invalid_recall':>15s}"
    print(header)
    for name in [
        "collapsed_80_20", "collapsed_75_25", "collapsed_70_30", "collapsed_cv",
        "split_80_20", "split_75_25", "split_70_30", "split_cv",
    ]:
        m = results[name]
        print(f"{name:22s} {m['accuracy']:10.3f} {m['best_val_loss']:10.3f} {m['macro_f1']:10.3f} {m['invalid_recall']:15.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
