"""
Build the train/test split for the "simple version" (3-class) dataset:
valid / invalid / empty.

Images from Model & Training/pictures/simple version/<label>/ are the
source of truth for filenames + labels. The rename manifest
(Evaluation & Docs/context/rename_manifest.csv) is used to recover each
photo's real original capture time, which lets the split group photos
into "bursts" (near-duplicate frames shot seconds apart) and keep a
whole burst in a single split — otherwise near-duplicate frames could
leak between train and test.

Each class only has a handful of genuinely independent shooting
sessions (median gap between consecutive photos is ~0s — they're
rapid bursts — with only 2-3 big jumps between sessions per class), so
there isn't enough independent data for a 3-way train/val/test split
without starving val or test. This script does an 80/20 train/test
split by session/burst instead; carve a validation slice out of the
train split at training time in the notebook if you need one for
comparing runs.

Output: Model & Training/dataset_split.csv with columns:
  filename, label, daynight, original_filename, captured_at, burst_id, split

Usage: python build_dataset_index.py
"""

import csv
import os
import random
import re
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMPLE_DIR = os.path.join(TRAINING_DIR, "pictures", "simple version")
MANIFEST = os.path.abspath(
    os.path.join(TRAINING_DIR, "..", "Evaluation & Docs", "context", "rename_manifest.csv")
)
OUTPUT = os.path.join(TRAINING_DIR, "dataset_split.csv")

LABELS = ["valid", "invalid", "empty"]

# Gap between two photos (by their real capture time) beyond which they
# count as separate "bursts" rather than the same near-duplicate group.
BURST_GAP = timedelta(seconds=120)

# Target split ratios, by image count within each class.
SPLIT_RATIOS = {"train": 0.80, "test": 0.20}

RANDOM_SEED = 42

TIMESTAMP_RE = re.compile(
    r"WhatsApp Image (\d{4}-\d{2}-\d{2}) at (\d{2})\.(\d{2})\.(\d{2})"
)


def daynight_from_class_folder(class_folder):
    if "יום" in class_folder:
        return "day"
    if "לילה" in class_folder:
        return "night"
    return "n/a"  # empty filter isn't split by day/night


def load_manifest():
    """new_filename -> (original_filename, class_folder)"""
    lookup = {}
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["new_filename"]] = (row["original_filename"], row["class_folder"])
    return lookup


def parse_captured_at(original_filename):
    m = TIMESTAMP_RE.search(original_filename)
    if not m:
        return None
    date_str, hh, mm, ss = m.groups()
    return datetime.strptime(f"{date_str} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def assign_bursts(records):
    """records: list of dicts with 'captured_at', sorted in place by time.
    Assigns a burst_id to each record (None-time records each get their own burst)."""
    with_time = [r for r in records if r["captured_at"] is not None]
    without_time = [r for r in records if r["captured_at"] is None]
    with_time.sort(key=lambda r: r["captured_at"])

    burst_id = 0
    prev_time = None
    for r in with_time:
        if prev_time is None or (r["captured_at"] - prev_time) > BURST_GAP:
            burst_id += 1
        r["burst_id"] = f"b{burst_id}"
        prev_time = r["captured_at"]

    for r in without_time:
        burst_id += 1
        r["burst_id"] = f"b{burst_id}"

    return with_time + without_time


def split_bursts(records, label):
    """Greedily assign whole bursts to train/val/test to approximate
    SPLIT_RATIOS by image count for this label."""
    bursts = {}
    for r in records:
        bursts.setdefault(r["burst_id"], []).append(r)

    burst_list = list(bursts.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(burst_list)
    # Largest bursts first so a single oversized burst doesn't blow the ratio.
    burst_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(records)
    targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: 0 for split in SPLIT_RATIOS}

    for _burst_id, burst_records in burst_list:
        # Assign to whichever split is furthest below its target (by count).
        deficit = {s: targets[s] - counts[s] for s in SPLIT_RATIOS}
        chosen = max(deficit, key=deficit.get)
        counts[chosen] += len(burst_records)
        for r in burst_records:
            r["split"] = chosen

    if total > 0 and counts["test"] == 0:
        print(
            f"  WARNING: '{label}' has too few bursts ({len(burst_list)}) to fill "
            f"every split — test={counts['test']}. "
            f"Results on this class will be noisy; treat with caution."
        )

    return counts


def main():
    manifest = load_manifest()
    all_records = []

    for label in LABELS:
        label_dir = os.path.join(SIMPLE_DIR, label)
        if not os.path.isdir(label_dir):
            raise RuntimeError(f"Missing expected folder: {label_dir}")

        records = []
        for fname in sorted(os.listdir(label_dir)):
            if not os.path.isfile(os.path.join(label_dir, fname)):
                continue
            original_filename, class_folder = manifest.get(fname, (None, None))
            captured_at = parse_captured_at(original_filename) if original_filename else None
            records.append(
                {
                    "filename": fname,
                    "label": label,
                    "daynight": daynight_from_class_folder(class_folder) if class_folder else "n/a",
                    "original_filename": original_filename or "",
                    "captured_at": captured_at,
                }
            )

        records = assign_bursts(records)
        counts = split_bursts(records, label)
        print(f"{label}: {len(records)} images -> {counts}")
        all_records.extend(records)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "label", "daynight", "original_filename", "captured_at", "burst_id", "split"]
        )
        for r in all_records:
            writer.writerow(
                [
                    r["filename"],
                    r["label"],
                    r["daynight"],
                    r["original_filename"],
                    r["captured_at"].isoformat() if r["captured_at"] else "",
                    r["burst_id"],
                    r["split"],
                ]
            )

    print(f"\nWrote {len(all_records)} rows to {OUTPUT}")

    print("\nPer label x split counts:")
    grid = {}
    for r in all_records:
        grid.setdefault(r["label"], {"train": 0, "test": 0})[r["split"]] += 1
    header = f"{'label':10s} {'train':>6s} {'test':>6s} {'total':>6s} {'test %':>7s}"
    print(header)
    for label in LABELS:
        c = grid[label]
        total = sum(c.values())
        pct = 100 * c["test"] / total if total else 0
        print(f"{label:10s} {c['train']:6d} {c['test']:6d} {total:6d} {pct:6.1f}%")

    print(
        "\nNote: session sizes are uneven (one big shoot per class often holds most "
        "of the photos), so the 80/20 target is approximate, not exact, once whole "
        "sessions are kept together — see the per-class 'test %' above."
    )


if __name__ == "__main__":
    main()
