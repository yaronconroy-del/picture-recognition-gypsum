"""
Train the best-performing configuration found by the optimization sweep
(see Evaluation & Docs/Project Workflow.md §6): the extended (5-class)
label scheme, 70/30 split, MobileNetV2 with the last block fine-tuned.

Unlike optimize_models.py (which only measured this config as part of a
sweep and didn't keep the weights), this script trains it for real and
saves the model — a candidate to promote to the project's main model,
pending the open decision on whether to move off the lite 3-class scheme
(see Project Workflow.md §7).

Keep this file and notebooks/train_extended_70_30.ipynb in sync.

Usage:
  python train_extended_70_30.py            # full run
  python train_extended_70_30.py --quick    # fast smoke test (1 epoch)
"""

import argparse
import copy
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # headless: save plots to file instead of showing them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
MODELS_DIR = os.path.join(TRAINING_DIR, "models")

LABEL_COL = "extended_label"
SPLIT_COL = "split_extended_70"

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

# The winning config from the optimization sweep, held fixed.
CONFIG = {"backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 10}


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
    # No horizontal flip: the camera's framing (motor always upper-right)
    # isn't mirror-symmetric in real footage.
    return train_transform, eval_transform


class GypsumDataset(Dataset):
    def __init__(self, dataframe, class_to_idx, transform):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(EXTENDED_DIR, EXTENDED_LABEL_TO_FOLDER[row[LABEL_COL]], row["filename"])
        img = Image.open(path).convert("RGB")
        img = mask_timestamp(img)
        img = self.transform(img)
        return img, self.class_to_idx[row[LABEL_COL]]


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


def compute_class_weights(dataframe, classes, class_to_idx):
    counts = dataframe[LABEL_COL].value_counts()
    total = len(dataframe)
    weights = torch.zeros(len(classes))
    for c in classes:
        count = counts.get(c, 0)
        weights[class_to_idx[c]] = total / (len(classes) * count) if count else 0.0
    return weights


def run_training(model, epochs, train_loader, val_loader, class_weights, device, verbose=True):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=CONFIG["lr"])

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
            print(f"  epoch {epoch + 1:2d}/{epochs}  "
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


def collapse_to_lite(label):
    return "empty" if label == "empty" else label.rsplit("_", 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 epoch, for a pipeline smoke test only.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cpu":
        print("No GPU — a full run will be slow; use --quick to smoke-test, or run the notebook on Colab.")

    df = pd.read_csv(SPLIT_CSV)
    CLASSES = sorted(df[LABEL_COL].unique())
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print("classes:", CLASSES)
    print(df.groupby([LABEL_COL, SPLIT_COL]).size().unstack(fill_value=0))

    full_train_df = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    test_df = df[df[SPLIT_COL] == "test"].reset_index(drop=True)
    train_df, val_df = train_test_split(
        full_train_df, test_size=0.15, stratify=full_train_df[LABEL_COL], random_state=SEED,
    )
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_transform, eval_transform = make_transforms()
    train_ds = GypsumDataset(train_df, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, class_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_df, CLASSES, class_to_idx)
    print("class weights:", dict(zip(CLASSES, class_weights.tolist())))

    epochs = 1 if args.quick else CONFIG["epochs"]
    model = build_model(CONFIG["backbone"], CONFIG["unfreeze_last_block"], len(CLASSES)).to(device)
    model, history, best_val_loss = run_training(model, epochs, train_loader, val_loader, class_weights, device)

    os.makedirs(MODELS_DIR, exist_ok=True)
    if not args.quick:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(history["train_loss"], label="train")
        axes[0].plot(history["val_loss"], label="val")
        axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(history["train_acc"], label="train")
        axes[1].plot(history["val_acc"], label="val")
        axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
        plt.tight_layout()
        curves_path = os.path.join(MODELS_DIR, "extended_70_30_training_curves.png")
        plt.savefig(curves_path)
        print(f"Saved {curves_path}")

    y_pred, y_true = predict_all(model, test_loader, device)
    print("\n5-class (native) report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0))

    true_names_lite = [collapse_to_lite(CLASSES[i]) for i in y_true]
    pred_names_lite = [collapse_to_lite(CLASSES[i]) for i in y_pred]
    lite_classes = ["empty", "invalid", "valid"]
    print("\nCollapsed to lite 3-class scale (for comparison with the other models):")
    lite_report = classification_report(
        true_names_lite, pred_names_lite, labels=lite_classes, target_names=lite_classes,
        digits=3, output_dict=True, zero_division=0,
    )
    print(classification_report(
        true_names_lite, pred_names_lite, labels=lite_classes, target_names=lite_classes,
        digits=3, zero_division=0,
    ))

    if not args.quick:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Oranges", xticks_rotation=45)
        plt.title("Test confusion matrix — extended, 70/30")
        plt.tight_layout()
        cm_path = os.path.join(MODELS_DIR, "extended_70_30_confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Saved {cm_path}")

    model_path = os.path.join(MODELS_DIR, "gypsum_classifier_extended_70_30.pt")
    meta_path = os.path.join(MODELS_DIR, "gypsum_classifier_extended_70_30.json")
    torch.save(model.state_dict(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "config": CONFIG,
            "label_scheme": "extended",
            "split": "70_30",
            "classes": CLASSES,
            "best_val_loss": best_val_loss,
            "test_report_native": classification_report(
                y_true, y_pred, target_names=CLASSES, digits=3, output_dict=True, zero_division=0
            ),
            "test_report_lite_scale": lite_report,
        }, f, indent=2)
    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
