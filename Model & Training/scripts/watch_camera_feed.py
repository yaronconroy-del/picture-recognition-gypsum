"""
Simulated live-feed demo: watches a folder for new photos and classifies
each one as it appears, printing an operator-facing alert whenever a
prediction comes back INVALID.

Stands in for a real camera connection until the deployment questions in
Evaluation & Docs/Project Workflow.md SS14 are decided (where frames
actually come from — RTSP stream, folder sync, etc. — and what the
alerting channel should be). For the course demo: point this at a
folder and drop a new photo into it to simulate the camera producing a
new frame; it gets classified within a few seconds.

Reuses predict.py's model loading and preprocessing exactly, so a photo
classified here gets the same answer predict.py would give it.

Usage:
  python watch_camera_feed.py                # watches ../pictures/incoming
  python watch_camera_feed.py path/to/folder  # watches a custom folder
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from predict import build_model, collapse_to_decision, eval_transform, find_model, mask_timestamp  # noqa: E402

DEFAULT_WATCH_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "pictures", "incoming"))
POLL_SECONDS = 3
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_model():
    pt_path, json_path = find_model()
    if pt_path is None:
        sys.exit("No trained model found in models/ — see predict.py.")
    with open(json_path) as f:
        meta = json.load(f)
    classes = meta["classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    return model, classes, device, pt_path


def classify(model, classes, device, path):
    img = Image.open(path).convert("RGB")
    img = mask_timestamp(img)
    x = eval_transform()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_class = classes[pred_idx]
    confidence = probs[pred_idx].item()
    decision = collapse_to_decision(pred_class)
    return pred_class, confidence, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default=DEFAULT_WATCH_DIR, help="Folder to watch for new photos.")
    args = parser.parse_args()

    os.makedirs(args.folder, exist_ok=True)
    model, classes, device, pt_path = load_model()

    print(f"Watching: {args.folder}")
    print(f"Model:    {os.path.basename(pt_path)}")
    print("Drop a .jpg/.jpeg/.png file into that folder to simulate a new camera frame. Ctrl+C to stop.\n")

    seen = set(os.listdir(args.folder))
    try:
        while True:
            current = {f for f in os.listdir(args.folder) if f.lower().endswith(IMAGE_EXTS)}
            for fname in sorted(current - seen):
                path = os.path.join(args.folder, fname)
                time.sleep(0.2)  # let the file finish writing before reading it
                try:
                    pred_class, confidence, decision = classify(model, classes, device, path)
                except Exception as e:
                    print(f"[{fname}] could not read image ({e}), skipping")
                    seen.add(fname)
                    continue
                alert = "  <<< ALERT: INVALID" if decision == "invalid" else ""
                print(f"[{fname}] {pred_class} ({confidence:.1%}) -> {decision.upper()}{alert}")
                seen.add(fname)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
