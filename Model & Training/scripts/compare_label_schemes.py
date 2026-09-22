"""
A/B test: does keeping day/night as separate classes (the "extended", 5-class
scheme) help or hurt, compared to the "lite" 3-class scheme (valid / invalid
/ empty, day+night collapsed) actually used for the main model?

Trains one model per scheme with the SAME fixed config (backbone, learning
rate, epochs) so label granularity is the only thing that differs between
the two runs. Each scheme uses its own leakage-safe train/test split (see
build_dataset_index.py) — the two test sets aren't the same images, so
treat this as a directional comparison, not a strict paired one.

The extended model's 5-class predictions are also collapsed back to 3
classes (valid_day/valid_night -> valid, etc.) so its performance can be
read on the same scale as the lite model's.

Keep this file and notebooks/compare_label_schemes.ipynb in sync.

Usage:
  python compare_label_schemes.py            # full run
  python compare_label_schemes.py --quick    # fast smoke test (1 epoch each)
"""

import argparse
import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMPLE_DIR = os.path.join(TRAINING_DIR, "pictures", "simple version")
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
OUT_DIR = os.path.join(TRAINING_DIR, "models", "label_scheme_comparison")

# extended_label -> its folder name under pictures/extended version/
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

# Held fixed across both runs on purpose — the whole point is to isolate
# the label-scheme variable, not to also search over hyperparameters here
# (that search already happened in train_and_evaluate.*).
FIXED_CONFIG = {"backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 10}


def mask_timestamp(img):
    """Paint over the top strip of the frame, where the camera burns in
    its date/time overlay, so the model can't key off it."""
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
    """Generic version: path_fn(row) builds the image path, label_fn(row)
    returns the class name, so this works for either scheme's folder layout."""

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
            print(f"    epoch {epoch + 1:2d}/{epochs}  "
                  f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, history


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
    return np.array(all_labels), np.array(all_preds)


def run_scheme(name, df, label_col, split_col, images_root, path_fn, epochs, device):
    print(f"\n=== {name} scheme (label column: {label_col}) ===")
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print("classes:", classes)

    full_train_df = df[df[split_col] == "train"].reset_index(drop=True)
    test_df = df[df[split_col] == "test"].reset_index(drop=True)
    train_df, val_df = train_test_split(
        full_train_df, test_size=0.15, stratify=full_train_df[label_col], random_state=SEED,
    )
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_transform, eval_transform = make_transforms()
    label_fn = lambda row: row[label_col]  # noqa: E731

    train_ds = GypsumDataset(train_df, path_fn, label_fn, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, path_fn, label_fn, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, path_fn, label_fn, class_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_df, label_col, classes, class_to_idx)
    print("class weights:", dict(zip(classes, class_weights.tolist())))

    model, history = run_training(
        FIXED_CONFIG["backbone"], FIXED_CONFIG["unfreeze_last_block"], FIXED_CONFIG["lr"], epochs,
        train_loader, val_loader, class_weights, device,
    )

    y_true, y_pred = predict_all(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=classes, digits=3, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0))

    return {
        "model": model,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "history": history,
        "test_df": test_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report,
    }


def collapse_extended_to_lite(label):
    return "empty" if label == "empty" else label.rsplit("_", 1)[0]


def summarize(name, y_true_names, y_pred_names, lite_classes):
    """accuracy / macro-F1 / invalid-recall on a lite (3-class) labeling."""
    report = classification_report(
        y_true_names, y_pred_names, labels=lite_classes, target_names=lite_classes,
        digits=3, output_dict=True, zero_division=0,
    )
    accuracy = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]
    invalid_recall = report.get("invalid", {}).get("recall", float("nan"))
    print(f"{name:28s} accuracy={accuracy:.3f}  macro_f1={macro_f1:.3f}  invalid_recall={invalid_recall:.3f}")
    return {"accuracy": accuracy, "macro_f1": macro_f1, "invalid_recall": invalid_recall}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 epoch each, for a pipeline smoke test only.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epochs = 1 if args.quick else FIXED_CONFIG["epochs"]
    df = pd.read_csv(SPLIT_CSV)

    lite_path_fn = lambda row: os.path.join(SIMPLE_DIR, row["label"], row["filename"])  # noqa: E731
    extended_path_fn = lambda row: os.path.join(  # noqa: E731
        EXTENDED_DIR, EXTENDED_LABEL_TO_FOLDER[row["extended_label"]], row["filename"]
    )

    lite = run_scheme("lite (3-class)", df, "label", "split", SIMPLE_DIR, lite_path_fn, epochs, device)
    extended = run_scheme(
        "extended (5-class)", df, "extended_label", "split_extended", EXTENDED_DIR, extended_path_fn,
        epochs, device,
    )

    lite_classes = lite["classes"]
    lite_true_names = [lite_classes[i] for i in lite["y_true"]]
    lite_pred_names = [lite_classes[i] for i in lite["y_pred"]]

    ext_classes = extended["classes"]
    ext_true_names = [collapse_extended_to_lite(ext_classes[i]) for i in extended["y_true"]]
    ext_pred_names = [collapse_extended_to_lite(ext_classes[i]) for i in extended["y_pred"]]

    print("\n=== Comparison (both read on the lite 3-class scale) ===")
    print(
        "Note: the two schemes have independently-built test sets (different "
        "images), so this is a directional comparison, not a strict paired one."
    )
    lite_summary = summarize("lite (native)", lite_true_names, lite_pred_names, lite_classes)
    extended_summary = summarize("extended (collapsed to lite)", ext_true_names, ext_pred_names, lite_classes)

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.quick:
        torch.save(lite["model"].state_dict(), os.path.join(OUT_DIR, "lite.pt"))
        torch.save(extended["model"].state_dict(), os.path.join(OUT_DIR, "extended.pt"))

    with open(os.path.join(OUT_DIR, "comparison.json"), "w") as f:
        json.dump({
            "config": FIXED_CONFIG,
            "quick_smoke_test": args.quick,
            "lite": {"classes": lite_classes, "report": lite["report"], "summary": lite_summary},
            "extended": {"classes": ext_classes, "report": extended["report"], "summary": extended_summary},
        }, f, indent=2)
    print(f"\nSaved {os.path.join(OUT_DIR, 'comparison.json')}")


if __name__ == "__main__":
    main()
