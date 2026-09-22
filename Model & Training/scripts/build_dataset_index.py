"""
Build train/test splits for the dataset, at two label granularities:

  - "lite" (3-class): valid / invalid / empty — pictures/simple version/
  - "extended" (5-class): valid_day / valid_night / invalid_day /
    invalid_night / empty — pictures/extended version/

Images are the source of truth for filenames; the rename manifest
(Evaluation & Docs/context/rename_manifest.csv) recovers each photo's real
original capture time, which lets each split group photos into "bursts"
(near-duplicate frames shot seconds apart) and keep a whole burst in a
single split — otherwise near-duplicate frames could leak between train
and test.

The two granularities need INDEPENDENT burst groupings: a "valid" burst in
the lite scheme can mix day- and night-labeled photos taken seconds apart
(day/night here turns out not to mean literal time-of-day — see
Project Workflow.md — photos labeled "day" and "night" are interleaved
within the same few seconds in the source timestamps), so reusing the lite
split for the 5-class scheme would be wrong. Bursts and splits are
recomputed from scratch per scheme, each only ever grouping photos that
share that scheme's own label.

Each class only has a handful of genuinely independent shooting sessions,
so there isn't enough independent data for a 3-way train/val/test split
without starving val or test. This does an 80/20 train/test split by
session/burst instead; carve a validation slice out of train at training
time if you need one for comparing runs.

Output: Model & Training/dataset_split.csv with columns:
  filename, original_filename, captured_at,
  label, burst_id, split,                        (lite scheme)
  extended_label, burst_id_extended, split_extended   (extended scheme)

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

LITE_LABELS = ["valid", "invalid", "empty"]
EXTENDED_LABELS = ["valid_day", "valid_night", "invalid_day", "invalid_night", "empty"]

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


def extended_label_of(label, daynight):
    return label if label == "empty" else f"{label}_{daynight}"


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


def assign_bursts(records, burst_field):
    """records: list of dicts with 'captured_at'. Tags each with burst_field,
    grouping any two photos less than BURST_GAP apart into the same burst."""
    with_time = [r for r in records if r["captured_at"] is not None]
    without_time = [r for r in records if r["captured_at"] is None]
    with_time.sort(key=lambda r: r["captured_at"])

    burst_id = 0
    prev_time = None
    for r in with_time:
        if prev_time is None or (r["captured_at"] - prev_time) > BURST_GAP:
            burst_id += 1
        r[burst_field] = f"b{burst_id}"
        prev_time = r["captured_at"]

    for r in without_time:
        burst_id += 1
        r[burst_field] = f"b{burst_id}"


def split_bursts(records, group_name, burst_field, split_field):
    """Greedily assign whole bursts to train/test to approximate
    SPLIT_RATIOS by image count for this group."""
    bursts = {}
    for r in records:
        bursts.setdefault(r[burst_field], []).append(r)

    burst_list = list(bursts.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(burst_list)
    # Largest bursts first so a single oversized burst doesn't blow the ratio.
    burst_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(records)
    targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
    counts = {split: 0 for split in SPLIT_RATIOS}

    for _burst_id, burst_records in burst_list:
        deficit = {s: targets[s] - counts[s] for s in SPLIT_RATIOS}
        chosen = max(deficit, key=deficit.get)
        counts[chosen] += len(burst_records)
        for r in burst_records:
            r[split_field] = chosen

    if total > 0 and counts["test"] == 0:
        print(
            f"  WARNING: '{group_name}' has too few bursts ({len(burst_list)}) to fill "
            f"every split — test={counts['test']}. "
            f"Results on this class will be noisy; treat with caution."
        )

    return counts


def build_scheme(records_by_group, group_names, burst_field, split_field):
    """Runs independent burst-assignment + split for each group, writing
    burst_field/split_field onto each record. records_by_group: group name
    -> list of that group's records (shared record dicts, mutated in place)."""
    print(f"\n--- {burst_field} / {split_field} ---")
    grid = {}
    for group in group_names:
        records = records_by_group.get(group, [])
        assign_bursts(records, burst_field)
        counts = split_bursts(records, group, burst_field, split_field)
        grid[group] = counts
        print(f"{group}: {len(records)} images -> {counts}")

    header = f"{'group':14s} {'train':>6s} {'test':>6s} {'total':>6s} {'test %':>7s}"
    print(header)
    for group in group_names:
        c = grid.get(group, {"train": 0, "test": 0})
        total = sum(c.values())
        pct = 100 * c["test"] / total if total else 0
        print(f"{group:14s} {c['train']:6d} {c['test']:6d} {total:6d} {pct:6.1f}%")


def main():
    manifest = load_manifest()
    all_records = []

    for label in LITE_LABELS:
        label_dir = os.path.join(SIMPLE_DIR, label)
        if not os.path.isdir(label_dir):
            raise RuntimeError(f"Missing expected folder: {label_dir}")

        for fname in sorted(os.listdir(label_dir)):
            if not os.path.isfile(os.path.join(label_dir, fname)):
                continue
            original_filename, class_folder = manifest.get(fname, (None, None))
            captured_at = parse_captured_at(original_filename) if original_filename else None
            daynight = daynight_from_class_folder(class_folder) if class_folder else "n/a"
            all_records.append(
                {
                    "filename": fname,
                    "label": label,
                    "extended_label": extended_label_of(label, daynight),
                    "daynight": daynight,
                    "original_filename": original_filename or "",
                    "captured_at": captured_at,
                }
            )

    # Lite (3-class) scheme: bursts/splits computed within each of valid/invalid/empty.
    by_lite = {label: [r for r in all_records if r["label"] == label] for label in LITE_LABELS}
    build_scheme(by_lite, LITE_LABELS, burst_field="burst_id", split_field="split")

    # Extended (5-class) scheme: recomputed independently within each of the
    # 5 groups, so a burst can never mix two extended labels.
    by_extended = {
        group: [r for r in all_records if r["extended_label"] == group]
        for group in EXTENDED_LABELS
    }
    build_scheme(
        by_extended, EXTENDED_LABELS, burst_field="burst_id_extended", split_field="split_extended"
    )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "original_filename", "captured_at", "daynight",
            "label", "burst_id", "split",
            "extended_label", "burst_id_extended", "split_extended",
        ])
        for r in all_records:
            writer.writerow([
                r["filename"],
                r["original_filename"],
                r["captured_at"].isoformat() if r["captured_at"] else "",
                r["daynight"],
                r["label"], r["burst_id"], r["split"],
                r["extended_label"], r["burst_id_extended"], r["split_extended"],
            ])

    print(f"\nWrote {len(all_records)} rows to {OUTPUT}")
    print(
        "\nNote: session sizes are uneven (one big shoot per class often holds most "
        "of the photos), so the 80/20 target is approximate, not exact, once whole "
        "sessions are kept together — see each scheme's 'test %' above."
    )


if __name__ == "__main__":
    main()
