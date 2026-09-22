"""
Local/offline mirror of notebooks/train_and_evaluate.ipynb — same data
pipeline, model, training loop, optimization pass, and evaluation, without
the Colab-only setup (repo cloning, GPU runtime messaging, Drive downloads).

Meant to be run from a local clone of this repo — e.g. for a quick CPU
smoke test of the pipeline before spending Colab GPU time on it. For real
training, use the notebook on Colab's GPU; it's much faster.

Keep this file and notebooks/train_and_evaluate.ipynb in sync: whichever
one changes, mirror the change into the other.

Usage:
  python train_and_evaluate.py            # full run (matches the notebook)
  python train_and_evaluate.py --quick    # fast smoke test (1 config, 1 epoch)
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
IMAGES_ROOT = os.path.join(TRAINING_DIR, "pictures", "simple version")
SPLIT_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
MODELS_DIR = os.path.join(TRAINING_DIR, "models")

SEED = 42
BATCH_SIZE = 16
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CONFIGS = [
    {"name": "mobilenet_frozen_lr1e-3", "backbone": "mobilenet_v2", "unfreeze_last_block": False, "lr": 1e-3, "epochs": 8},
    {"name": "mobilenet_finetune_lr1e-3", "backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 8},
    {"name": "mobilenet_finetune_lr3e-4", "backbone": "mobilenet_v2", "unfreeze_last_block": True, "lr": 3e-4, "epochs": 8},
    {"name": "resnet18_finetune_lr1e-3", "backbone": "resnet18", "unfreeze_last_block": True, "lr": 1e-3, "epochs": 8},
]


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
    def __init__(self, dataframe, images_root, class_to_idx, transform):
        self.df = dataframe.reset_index(drop=True)
        self.images_root = images_root
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_root, row["label"], row["filename"])
        img = Image.open(path).convert("RGB")
        img = mask_timestamp(img)
        img = self.transform(img)
        return img, self.class_to_idx[row["label"]]


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


def compute_class_weights(dataframe, classes, class_to_idx):
    counts = dataframe["label"].value_counts()
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
    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast smoke test: 1 config, 1 epoch, skips saving plots. "
             "For checking the pipeline runs end to end, not for real results.",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cpu":
        print("No GPU — this will be slow for a full run; use --quick to smoke-test, "
              "or run the notebook on Colab for real training.")

    df = pd.read_csv(SPLIT_CSV)
    CLASSES = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    print("classes:", CLASSES)
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0))

    full_train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_df, val_df = train_test_split(
        full_train_df, test_size=0.15, stratify=full_train_df["label"], random_state=SEED,
    )
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_transform, eval_transform = make_transforms()
    train_ds = GypsumDataset(train_df, IMAGES_ROOT, class_to_idx, train_transform)
    val_ds = GypsumDataset(val_df, IMAGES_ROOT, class_to_idx, eval_transform)
    test_ds = GypsumDataset(test_df, IMAGES_ROOT, class_to_idx, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_df, CLASSES, class_to_idx)
    print("class weights:", dict(zip(CLASSES, class_weights.tolist())))

    configs = CONFIGS
    if args.quick:
        configs = [{**CONFIGS[0], "epochs": 1}]
        print("--quick: running 1 config, 1 epoch (smoke test only, not a real result)")

    results = []
    trained_models = {}
    histories = {}
    for cfg in configs:
        print(f"--- {cfg['name']} ---")
        model, history, best_val_loss = run_training(
            cfg["backbone"], cfg["unfreeze_last_block"], cfg["lr"], cfg["epochs"],
            train_loader, val_loader, class_weights, device,
        )
        trained_models[cfg["name"]] = model
        histories[cfg["name"]] = history
        results.append({**cfg, "best_val_loss": best_val_loss, "best_val_acc": max(history["val_acc"])})

    results_df = pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)
    print("\n", results_df, "\n", sep="")

    best_name = results_df.iloc[0]["name"]
    best_model = trained_models[best_name]
    best_history = histories[best_name]
    print("chosen config:", best_name)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not args.quick:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(best_history["train_loss"], label="train")
        axes[0].plot(best_history["val_loss"], label="val")
        axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
        axes[1].plot(best_history["train_acc"], label="train")
        axes[1].plot(best_history["val_acc"], label="val")
        axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
        plt.tight_layout()
        curves_path = os.path.join(MODELS_DIR, "training_curves.png")
        plt.savefig(curves_path)
        print(f"Saved {curves_path}")

    y_true, y_pred = predict_all(best_model, test_loader, device)
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

    invalid_idx = class_to_idx["invalid"]
    denom = max((y_true == invalid_idx).sum(), 1)
    invalid_recall = ((y_pred == invalid_idx) & (y_true == invalid_idx)).sum() / denom
    print(f"'invalid' recall: {invalid_recall:.3f}  "
          f"(a missed fault costs more than a false alarm, so this is the number to watch)")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    test_df_eval = test_df.reset_index(drop=True).copy()
    test_df_eval["pred"] = [CLASSES[p] for p in y_pred]
    test_df_eval["true"] = [CLASSES[t] for t in y_true]
    test_df_eval["correct"] = test_df_eval["pred"] == test_df_eval["true"]
    daynight_rows = test_df_eval[test_df_eval["daynight"] != "n/a"]
    if len(daynight_rows):
        print("\nday/night breakdown:")
        print(daynight_rows.groupby("daynight")["correct"].agg(["mean", "count"]))

    model_path = os.path.join(MODELS_DIR, "gypsum_classifier.pt")
    meta_path = os.path.join(MODELS_DIR, "gypsum_classifier.json")
    chosen_cfg = next(c for c in configs if c["name"] == best_name)
    torch.save(best_model.state_dict(), model_path)
    with open(meta_path, "w") as f:
        json.dump({
            "backbone": chosen_cfg["backbone"],
            "unfreeze_last_block": chosen_cfg["unfreeze_last_block"],
            "classes": CLASSES,
            "test_classification_report": classification_report(
                y_true, y_pred, target_names=CLASSES, digits=3, output_dict=True
            ),
        }, f, indent=2)
    print(f"Saved {model_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
