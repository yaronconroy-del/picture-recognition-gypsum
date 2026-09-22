"""
Optimization sweep: split ratio, cross-validation, and one oversampling
variant, for both the lite (3-class) and extended (5-class) schemes.

Reuses the already-committed 80/20 baseline results (from
compare_label_schemes.py) instead of re-training them, and adds:

  - 75/25 and 70/30 ratio splits, for both schemes (4 new runs) — see
    build_dataset_index.py's split_75/split_70/split_extended_75/
    split_extended_70 columns. The 'empty' class always keeps >=3 images
    in train regardless of ratio (build_dataset_index.py's MIN_TRAIN_FLOOR).
  - K-fold cross-validation, for both schemes (build_dataset_index.py's
    fold_lite [k=3] / fold_extended [k=2] columns — different k per scheme
    because the extended scheme's night classes only have 2 independent
    sessions each, too few for 3 folds). Reports mean +/- std across folds
    instead of trusting one train/test split.
  - A WeightedRandomSampler variant on the lite 80/20 split, on top of the
    existing class-weighted loss, to see whether oversampling the scarce
    'empty' class during training helps beyond loss weighting alone.

Every run uses the SAME fixed model config as the earlier comparison, so
split ratio / CV / sampling is the only thing that varies run to run.

Keep this file and notebooks/optimize_models.ipynb in sync.

Usage:
  python optimize_models.py            # full run
  python optimize_models.py --quick    # fast smoke test (1 epoch each)
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMPLE_DIR = os.path.join(TRAINING_DIR, "pictures", "simple version")
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
BASELINE_JSON = os.path.join(TRAINING_DIR, "models", "label_scheme_comparison", "comparison.json")
OUT_DIR = os.path.join(TRAINING_DIR, "models", "optimization_sweep")

EXTENDED_LABEL_TO_FOLDER = {
    "valid_day": "תקין יום",
    "valid_night": "תקין לילה",
    "invalid_day": "לא תקין יום",
    "invalid_night": "לא תקין לילה",
    "empty": "מסנן ריק",
}

SEED = 42
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same fixed config as compare_label_schemes.py, on purpose.
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


class GypsumDataset(Dataset):
    def __init__(self, dataframe, path_fn, label_fn, class_to_idx, transform):
        self.df = dataframe.reset_index(drop=True)
        self.path_fn = path_fn
        self.label_fn = label_fn
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.path_fn(row)).convert("RGB")
        img = mask_timestamp(img)
        img = self.transform(img)
        return img, self.class_to_idx[self.label_fn(row)]


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
    elif backbone_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        if unfreeze_last_block:
            for p in model.layer4.parameters():
                p.requires_grad = True
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
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


def run_training(backbone_name, unfreeze_last_block, lr, epochs, train_loader, val_loader,
                  class_weights, device, verbose=True):
    model = build_model(backbone_name, unfreeze_last_block, len(class_weights)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr)

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
            print(f"      epoch {epoch + 1:2d}/{epochs}  "
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


def collapse_extended_to_lite(label):
    return "empty" if label == "empty" else label.rsplit("_", 1)[0]


def lite_scale_metrics(true_names, pred_names, lite_classes):
    report = classification_report(
        true_names, pred_names, labels=lite_classes, target_names=lite_classes,
        digits=3, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "invalid_recall": report.get("invalid", {}).get("recall", float("nan")),
    }


def path_fn_for(scheme):
    if scheme == "lite":
        return lambda row: os.path.join(SIMPLE_DIR, row["label"], row["filename"])
    return lambda row: os.path.join(
        EXTENDED_DIR, EXTENDED_LABEL_TO_FOLDER[row["extended_label"]], row["filename"]
    )


def label_col_for(scheme):
    return "label" if scheme == "lite" else "extended_label"


def train_one(df, scheme, train_df_full, test_df, epochs, device, sampler=False, verbose=True):
    label_col = label_col_for(scheme)
    path_fn = path_fn_for(scheme)
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df, val_df = train_test_split(
        train_df_full, test_size=0.15, stratify=train_df_full[label_col], random_state=SEED,
    )

    train_transform, eval_transform = make_transforms()
    label_fn = lambda row: row[label_col]  # noqa: E731

    train_ds = GypsumDataset(train_df, path_fn, label_fn, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, path_fn, label_fn, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, path_fn, label_fn, class_to_idx, eval_transform)

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

    model, history, best_val_loss = run_training(
        FIXED_CONFIG["backbone"], FIXED_CONFIG["unfreeze_last_block"], FIXED_CONFIG["lr"], epochs,
        train_loader, val_loader, class_weights, device, verbose=verbose,
    )

    y_pred, y_true = predict_all(model, test_loader, device)
    true_names = [classes[i] for i in y_true]
    pred_names = [classes[i] for i in y_pred]

    if scheme == "extended":
        true_names = [collapse_extended_to_lite(n) for n in true_names]
        pred_names = [collapse_extended_to_lite(n) for n in pred_names]

    lite_classes = ["empty", "invalid", "valid"]
    metrics = lite_scale_metrics(true_names, pred_names, lite_classes)
    metrics["best_val_loss"] = best_val_loss
    metrics["n_test"] = len(test_df)
    metrics["n_train"] = len(train_df_full)
    return metrics


def run_ratio(df, scheme, split_col, epochs, device):
    print(f"\n=== {scheme} — ratio split ({split_col}) ===")
    train_df_full = df[df[split_col] == "train"].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)
    print(f"train={len(train_df_full)}  test={len(test_df)}")
    metrics = train_one(df, scheme, train_df_full, test_df, epochs, device)
    print(f"  -> accuracy={metrics['accuracy']:.3f}  val_loss={metrics['best_val_loss']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  invalid_recall={metrics['invalid_recall']:.3f}")
    return metrics


def run_cv(df, scheme, fold_col, k, epochs, device):
    print(f"\n=== {scheme} — {k}-fold CV ({fold_col}) ===")
    fold_metrics = []
    for fold_i in range(k):
        print(f"  fold {fold_i + 1}/{k}")
        test_df = df[df[fold_col] == str(fold_i)].reset_index(drop=True)
        train_df_full = df[df[fold_col] != str(fold_i)].reset_index(drop=True)
        m = train_one(df, scheme, train_df_full, test_df, epochs, device, verbose=False)
        print(f"    -> accuracy={m['accuracy']:.3f}  invalid_recall={m['invalid_recall']:.3f}  "
              f"(n_test={m['n_test']})")
        fold_metrics.append(m)

    agg = {}
    for key in ("accuracy", "macro_f1", "invalid_recall", "best_val_loss"):
        vals = [m[key] for m in fold_metrics]
        agg[key] = statistics.mean(vals)
        agg[f"{key}_std"] = statistics.pstdev(vals)
    agg["folds"] = fold_metrics
    print(f"  -> mean accuracy={agg['accuracy']:.3f} (+/-{agg['accuracy_std']:.3f})  "
          f"mean invalid_recall={agg['invalid_recall']:.3f} (+/-{agg['invalid_recall_std']:.3f})")
    return agg


def run_sampler_variant(df, epochs, device):
    print("\n=== lite — 80/20 split + WeightedRandomSampler (on top of class-weighted loss) ===")
    train_df_full = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    metrics = train_one(df, "lite", train_df_full, test_df, epochs, device, sampler=True)
    print(f"  -> accuracy={metrics['accuracy']:.3f}  val_loss={metrics['best_val_loss']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  invalid_recall={metrics['invalid_recall']:.3f}")
    return metrics


def load_baseline():
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    return {
        "lite_80_20": data["lite"]["summary"],
        "extended_80_20": data["extended"]["summary"],
    }


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
    df = pd.read_csv(SPLIT_CSV, dtype={"fold_lite": str, "fold_extended": str})

    results = {"config": FIXED_CONFIG, "quick_smoke_test": args.quick}

    baseline = load_baseline()
    results["lite_80_20"] = baseline["lite_80_20"]
    results["extended_80_20"] = baseline["extended_80_20"]
    print("\nLoaded existing 80/20 baselines (not re-trained):", baseline)

    results["lite_75_25"] = run_ratio(df, "lite", "split_75", epochs, device)
    results["lite_70_30"] = run_ratio(df, "lite", "split_70", epochs, device)
    results["extended_75_25"] = run_ratio(df, "extended", "split_extended_75", epochs, device)
    results["extended_70_30"] = run_ratio(df, "extended", "split_extended_70", epochs, device)

    results["lite_cv"] = run_cv(df, "lite", "fold_lite", 3, epochs, device)
    results["extended_cv"] = run_cv(df, "extended", "fold_extended", 2, epochs, device)

    results["lite_80_20_sampler"] = run_sampler_variant(df, epochs, device)

    print("\n=== Summary (all read on the lite 3-class scale) ===")
    header = f"{'variant':26s} {'accuracy':>10s} {'val_loss':>10s} {'macro_f1':>10s} {'invalid_recall':>15s}"
    print(header)
    for name in [
        "lite_80_20", "lite_75_25", "lite_70_30", "lite_cv", "lite_80_20_sampler",
        "extended_80_20", "extended_75_25", "extended_70_30", "extended_cv",
    ]:
        m = results[name]
        acc = m.get("accuracy", float("nan"))
        loss = m.get("best_val_loss", float("nan"))
        f1 = m.get("macro_f1", float("nan"))
        rec = m.get("invalid_recall", float("nan"))
        print(f"{name:26s} {acc:10.3f} {loss:10.3f} {f1:10.3f} {rec:15.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "optimization_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
