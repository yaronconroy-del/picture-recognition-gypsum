"""
Live demo: classify one gypsum filter photo as VALID or INVALID.

Loads the promoted model (no-empty split, 4-class — see
Evaluation & Docs/Project Workflow.md SS12) if it has been trained, and
falls back to the extended (5-class) runner-up otherwise. Preprocessing
(timestamp masking, resize/crop, normalization) and the model
architecture match training exactly (see train_extended_70_30.py /
notebooks/full_pipeline.ipynb) so the loaded weights behave the same
way here as they did during evaluation.

Usage:
  python predict.py path/to/photo.jpeg
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preference order: the promoted model first, then the runner-up.
CANDIDATES = ["gypsum_classifier_split_70_30", "gypsum_classifier_extended_70_30"]


def mask_timestamp(img):
    """Paint over the top strip of the frame, where the camera burns in
    its date/time overlay, so the model can't key off it."""
    img = img.copy()
    w, h = img.size
    band_h = int(h * 0.10)
    ImageDraw.Draw(img).rectangle([0, 0, w, band_h], fill=(0, 0, 0))
    return img


def eval_transform():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(num_classes):
    model = torchvision.models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def collapse_to_decision(class_name):
    """Map any native class name to the operator-facing call. Per the
    SS12 decision, an empty filter counts as invalid (not its own
    outcome) regardless of which model produced the prediction."""
    if class_name == "empty":
        return "invalid"
    return class_name.rsplit("_", 1)[0]  # valid_day/valid_night -> valid, etc.


def find_model():
    for name in CANDIDATES:
        pt_path = os.path.join(MODELS_DIR, f"{name}.pt")
        json_path = os.path.join(MODELS_DIR, f"{name}.json")
        if os.path.exists(pt_path) and os.path.exists(json_path):
            return pt_path, json_path
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to a filter photo (jpeg/png).")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"No such file: {args.image}")

    pt_path, json_path = find_model()
    if pt_path is None:
        sys.exit(
            "No trained model found in models/. Expected one of: "
            + ", ".join(f"{n}.pt" for n in CANDIDATES)
            + " — train one first (see Model & Training/CLAUDE.md)."
        )

    with open(json_path) as f:
        meta = json.load(f)
    classes = meta["classes"]
    scheme = meta.get("label_scheme", meta.get("scheme", "?"))
    split = meta.get("split", "70_30")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    img = Image.open(args.image).convert("RGB")
    img = mask_timestamp(img)
    x = eval_transform()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    pred_idx = probs.argmax().item()
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx].item()
    decision = collapse_to_decision(pred_class)

    print(f"Model:       {os.path.basename(pt_path)}  ({scheme} scheme, {split} split)")
    print(f"Image:       {args.image}")
    print(f"Prediction:  {pred_class}  ({confidence:.1%} confidence)")
    print(f"Decision:    {decision.upper()}")
    print()
    print("Full class probabilities:")
    for c, p in sorted(zip(classes, probs.tolist()), key=lambda t: -t[1]):
        print(f"  {c:<15} {p:.1%}")


if __name__ == "__main__":
    main()
