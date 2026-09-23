"""
Retrains the two leading candidates (extended_70_30, split_70_30) plus
their CV variants on dataset_split_corrected.csv -- the day/night-corrected
dataset (see build_dataset_index_corrected.py) -- using the exact same
fixed config as every other training in this project, so results are
directly comparable to the original (uncorrected) numbers.

Saves training curves, confusion matrices, and a threshold/ROC/PR/
calibration analysis (matching scripts/threshold_analysis.py's method) for
each. Real run, not a smoke test -- takes a while on CPU.

Usage: python retrain_corrected.py
"""

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
    ConfusionMatrixDisplay, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version (day-night corrected)")
NOEMPTY_DIR = os.path.join(TRAINING_DIR, "pictures", "no-empty version (day-night corrected)")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split_corrected.csv")
OUT_DIR = os.path.join(TRAINING_DIR, "models", "corrected_daynight")
os.makedirs(OUT_DIR, exist_ok=True)

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
EPOCHS = FIXED_CONFIG["epochs"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

df = pd.read_csv(SPLIT_CSV, dtype={"fold_extended": str, "fold_noempty4": str})
print(f"{len(df)} images loaded from {SPLIT_CSV}")


def mask_timestamp(img):
    img = img.copy()
    w, h = img.size
    band_h = int(h * 0.10)
    ImageDraw.Draw(img).rectangle([0, 0, w, band_h], fill=(0, 0, 0))
    return img


def make_transforms():
    from torchvision import transforms
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


def build_model(num_classes):
    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
    model = torchvision.models.mobilenet_v2(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
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


def run_training(epochs, train_loader, val_loader, class_weights, num_classes):
    model = build_model(num_classes).to(device)
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
        print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}  val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history, best_val_loss


def predict_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(y.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def collapse_label(scheme, label):
    if scheme == "extended":
        return label if label == "empty" else label.rsplit("_", 1)[0]
    if scheme == "split":
        return label.rsplit("_", 1)[0]
    return label


TARGET_CLASSES = {"extended": ["empty", "invalid", "valid"], "split": ["invalid", "valid"]}


def train_and_eval_one(scheme, label_col, path_fn, classes, class_to_idx, invalid_idx, train_df_full, test_df):
    train_df, val_df = train_test_split(
        train_df_full, test_size=0.15, stratify=train_df_full[label_col], random_state=SEED,
    )
    train_tf, eval_tf = make_transforms()
    train_ds = GypsumDataset(train_df, path_fn, label_col, class_to_idx, train_tf)
    val_ds = GypsumDataset(val_df, path_fn, label_col, class_to_idx, eval_tf)
    test_ds = GypsumDataset(test_df, path_fn, label_col, class_to_idx, eval_tf)

    class_weights = compute_class_weights(train_df, label_col, classes, class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model, history, best_val_loss = run_training(EPOCHS, train_loader, val_loader, class_weights, len(classes))

    probs, y_true_idx = predict_probs(model, test_loader)
    y_pred_idx = probs.argmax(1)
    true_names = [collapse_label(scheme, classes[i]) for i in y_true_idx]
    pred_names = [collapse_label(scheme, classes[i]) for i in y_pred_idx]
    target = TARGET_CLASSES[scheme]
    report = classification_report(true_names, pred_names, labels=target, target_names=target, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_names, pred_names, labels=target)
    metrics = {
        "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"],
        "invalid_recall": report.get("invalid", {}).get("recall", float("nan")),
        "best_val_loss": best_val_loss, "n_test": len(test_df),
    }
    p_invalid = probs[:, invalid_idx].sum(axis=1)
    is_invalid = np.isin(y_true_idx, invalid_idx).astype(int)
    return model, metrics, p_invalid, is_invalid, history, cm


def analyze_scores(scores, is_invalid):
    fpr, tpr, roc_thresh = roc_curve(is_invalid, scores)
    roc_auc = roc_auc_score(is_invalid, scores)
    precision, recall, pr_thresh = precision_recall_curve(is_invalid, scores)
    ap = average_precision_score(is_invalid, scores)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    youden_threshold = float(roc_thresh[best_j_idx]) if len(roc_thresh) else 0.5
    f1 = np.where((precision[:-1] + recall[:-1]) > 0, 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12), 0)
    best_f1_idx = int(np.argmax(f1)) if len(f1) else 0
    f1_threshold = float(pr_thresh[best_f1_idx]) if len(pr_thresh) else 0.5
    frac_pos, mean_pred = calibration_curve(is_invalid, scores, n_bins=5, strategy="uniform")
    return {
        "roc_auc": float(roc_auc), "pr_ap": float(ap),
        "youden_threshold": youden_threshold, "f1_threshold": f1_threshold,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "calibration_curve": {"mean_predicted": mean_pred.tolist(), "frac_positive": frac_pos.tolist()},
    }


def plot_result(name, subtitle, history, cm, target_labels, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    epochs_x = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs_x, history["train_loss"], marker="o", markersize=3, label="train")
    axes[0].plot(epochs_x, history["val_loss"], marker="o", markersize=3, label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=8)
    axes[1].plot(epochs_x, history["train_acc"], marker="o", markersize=3, label="train")
    axes[1].plot(epochs_x, history["val_acc"], marker="o", markersize=3, label="val")
    axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=8)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    disp = ConfusionMatrixDisplay(cm_norm, display_labels=target_labels)
    disp.plot(ax=axes[2], cmap="Oranges", colorbar=False, values_format=".2f")
    axes[2].set_title("confusion matrix (row-normalized)")
    fig.suptitle(f"{name} — {subtitle} (day/night corrected)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


results = {}

# ===================================================================
# extended_70_30 and split_70_30 (ratio configs, real training curves + heatmap)
# ===================================================================
for name, scheme, label_col, path_fn, invalid_classes, split_col in [
    ("extended_70_30", "extended", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "split_extended_70"),
    ("split_70_30", "split", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "split_noempty4_70"),
]:
    print(f"\n=== {name} ===")
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    invalid_idx = [class_to_idx[c] for c in invalid_classes]
    train_df_full = df[df[split_col] == "train"].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)
    model, metrics, p_inv, is_inv, history, cm = train_and_eval_one(
        scheme, label_col, path_fn, classes, class_to_idx, invalid_idx, train_df_full, test_df,
    )
    print(f"  n_test={len(test_df)} accuracy={metrics['accuracy']:.3f} invalid_recall={metrics['invalid_recall']:.3f}")
    plot_result(name, "70/30 split", history, cm, TARGET_CLASSES[scheme], os.path.join(OUT_DIR, f"result_{name}.png"))
    results[name] = {"metrics": metrics, "threshold": analyze_scores(p_inv, is_inv)}

    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"gypsum_classifier_{name}_corrected.pt"))

# ===================================================================
# extended_cv and split_cv (for threshold/ROC/PR pooled analysis)
# ===================================================================
for name, scheme, label_col, path_fn, invalid_classes, fold_col in [
    ("extended_cv", "extended", "extended_label", extended_path_fn, ["invalid_day", "invalid_night"], "fold_extended"),
    ("split_cv", "split", "noempty_split_label", noempty_path_fn, ["invalid_day", "invalid_night"], "fold_noempty4"),
]:
    print(f"\n=== {name} ===")
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    invalid_idx = [class_to_idx[c] for c in invalid_classes]
    k = df[fold_col].nunique()
    fold_metrics, all_p, all_y = [], [], []
    cm_sum = None
    for fold_i in range(k):
        print(f"  fold {fold_i+1}/{k}")
        test_df = df[df[fold_col] == str(fold_i)].reset_index(drop=True)
        train_df_full = df[df[fold_col] != str(fold_i)].reset_index(drop=True)
        model, m, p_inv, is_inv, history, cm = train_and_eval_one(
            scheme, label_col, path_fn, classes, class_to_idx, invalid_idx, train_df_full, test_df,
        )
        fold_metrics.append(m)
        all_p.append(p_inv); all_y.append(is_inv)
        cm_sum = cm if cm_sum is None else cm_sum + cm

    p_pool = np.concatenate(all_p)
    y_pool = np.concatenate(all_y)
    agg = {key: float(np.mean([fm[key] for fm in fold_metrics])) for key in ("accuracy", "macro_f1", "invalid_recall", "best_val_loss")}
    agg["accuracy_std"] = float(np.std([fm["accuracy"] for fm in fold_metrics]))
    agg["invalid_recall_std"] = float(np.std([fm["invalid_recall"] for fm in fold_metrics]))
    print(f"  pooled n_test={len(y_pool)} mean accuracy={agg['accuracy']:.3f} (+/-{agg['accuracy_std']:.3f})")
    results[name] = {"metrics": agg, "threshold": analyze_scores(p_pool, y_pool)}

    row_sums = cm_sum.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_sum.astype(float), row_sums, out=np.zeros_like(cm_sum, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm_norm, display_labels=TARGET_CLASSES[scheme]).plot(ax=ax, cmap="Oranges", colorbar=False, values_format=".2f")
    ax.set_title(f"{name} — pooled CV confusion matrix\n(day/night corrected)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"confusion_{name}.png"), dpi=130)
    plt.close(fig)

with open(os.path.join(OUT_DIR, "corrected_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved {os.path.join(OUT_DIR, 'corrected_results.json')}")

# ROC/PR overlay comparing all 4 corrected configs
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
for name, r in results.items():
    t = r["threshold"]
    ax.plot(t["roc_curve"]["fpr"], t["roc_curve"]["tpr"], label=f"{name} (AUC={t['roc_auc']:.3f})")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC — day/night corrected"); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "roc_corrected.png"), dpi=130)
plt.close(fig)

print("\nDone.")
for name, r in results.items():
    m, t = r["metrics"], r["threshold"]
    print(f"{name:16s} accuracy={m['accuracy']:.3f}  ROC AUC={t['roc_auc']:.3f}  PR AP={t['pr_ap']:.3f}  F1-thresh={t['f1_threshold']:.3f}")
