"""
Build train/test splits for the dataset, at two label granularities and
three split ratios each:

  - "lite" (3-class): valid / invalid / empty — pictures/simple version/
  - "extended" (5-class): valid_day / valid_night / invalid_day /
    invalid_night / empty — pictures/extended version/
  - ratios: 80/20 (columns: split, split_extended — the originals, kept for
    backward compatibility), 75/25 (split_lite_75, split_extended_75),
    70/30 (split_lite_70, split_extended_70)

Images are the source of truth for filenames; the rename manifest
(Evaluation & Docs/context/rename_manifest.csv) recovers each photo's real
original capture time, which lets each split group photos into "bursts"
(near-duplicate frames shot seconds apart) and keep a whole burst in a
single split — otherwise near-duplicate frames could leak between train
and test. Burst grouping is computed ONCE per scheme (it doesn't depend on
the ratio); only which split each burst lands in changes per ratio.

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
without starving val or test. This does a train/test split by session/burst
instead; carve a validation slice out of train at training time if you
need one for comparing runs.

MIN_TRAIN_FLOOR guarantees a minimum train-side count for scarce groups
regardless of ratio (currently: 'empty' always keeps at least 3 of its 8
images in train) — if the ratio's normal greedy assignment doesn't already
reach that on its own, bursts are moved back from test to train until it
does, smallest first.

Output: Model & Training/dataset_split.csv with columns:
  filename, original_filename, captured_at, daynight,
  label, burst_id, split, split_lite_75, split_lite_70,
  extended_label, burst_id_extended, split_extended,
    split_extended_75, split_extended_70

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
NOEMPTY_DIR = os.path.join(TRAINING_DIR, "pictures", "no-empty version")
MANIFEST = os.path.abspath(
    os.path.join(TRAINING_DIR, "..", "Evaluation & Docs", "context", "rename_manifest.csv")
)
OUTPUT = os.path.join(TRAINING_DIR, "dataset_split.csv")

LITE_LABELS = ["valid", "invalid", "empty"]
EXTENDED_LABELS = ["valid_day", "valid_night", "invalid_day", "invalid_night", "empty"]
# No separate "empty" here: those 8 photos were manually redistributed into
# invalid_day/invalid_night in pictures/no-empty version/ (see Project
# Workflow.md) — "empty filter" redefined as a kind of invalid rather than
# its own class.
NOEMPTY_COLLAPSED_LABELS = ["valid", "invalid"]
NOEMPTY_SPLIT_LABELS = ["valid_day", "valid_night", "invalid_day", "invalid_night"]

# Gap between two photos (by their real capture time) beyond which they
# count as separate "bursts" rather than the same near-duplicate group.
BURST_GAP = timedelta(seconds=120)

# (split-column suffix, train ratio). "" keeps the original column names
# (split / split_extended) for backward compatibility with existing scripts.
RATIOS = [("", 0.80), ("_75", 0.75), ("_70", 0.70)]

# Minimum number of a group's images that must land in train, regardless
# of ratio — overrides the ratio's normal greedy assignment if needed.
MIN_TRAIN_FLOOR = {"empty": 3}

# Also build a K-fold assignment per scheme, for cross-validation (a more
# robust accuracy estimate than any single train/test split, given how few
# independent sessions each class has). Different K per scheme: the
# extended scheme's night classes only have 2 independent sessions each,
# so 3-fold CV would leave one fold with zero night examples — 2-fold is
# the most this scheme's data actually supports.
K_FOLDS_LITE = 3
K_FOLDS_EXTENDED = 2
# Desired K for the no-empty schemes — actual K is capped automatically
# (determine_k) to whatever the scarcest group's session count supports.
K_FOLDS_NOEMPTY_DESIRED = 4

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


def split_bursts(records, group_name, burst_field, split_field, train_ratio, min_train=0):
    """Greedily assign whole bursts to train/test to approximate train_ratio
    by image count for this group, then top up train (moving the smallest
    test bursts back) if it's still short of min_train."""
    bursts = {}
    for r in records:
        bursts.setdefault(r[burst_field], []).append(r)

    burst_list = list(bursts.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(burst_list)
    # Largest bursts first so a single oversized burst doesn't blow the ratio.
    burst_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(records)
    targets = {"train": total * train_ratio, "test": total * (1 - train_ratio)}
    counts = {"train": 0, "test": 0}
    assignment = {}  # burst_id -> "train"/"test"

    for burst_id, burst_records in burst_list:
        deficit = {s: targets[s] - counts[s] for s in ("train", "test")}
        chosen = max(deficit, key=deficit.get)
        counts[chosen] += len(burst_records)
        assignment[burst_id] = chosen

    if counts["train"] < min_train:
        # Move test bursts back to train, smallest first, until the floor is met.
        test_bursts = sorted(
            (bid for bid, s in assignment.items() if s == "test"),
            key=lambda bid: len(bursts[bid]),
        )
        for bid in test_bursts:
            if counts["train"] >= min_train:
                break
            assignment[bid] = "train"
            counts["train"] += len(bursts[bid])
            counts["test"] -= len(bursts[bid])
        print(
            f"  Note: '{group_name}' topped up to the {min_train}-image train floor "
            f"(overriding the plain {train_ratio:.0%} ratio) -> {counts}"
        )

    for burst_id, split in assignment.items():
        for r in bursts[burst_id]:
            r[split_field] = split

    if total > 0 and counts["test"] == 0:
        print(
            f"  WARNING: '{group_name}' has too few bursts ({len(burst_list)}) to fill "
            f"every split — test={counts['test']}. "
            f"Results on this class will be noisy; treat with caution."
        )

    return counts


def assign_folds(records, group_name, burst_field, fold_field, k):
    """Greedily distributes whole bursts across k folds as evenly as
    possible by image count, for leave-one-fold-out cross-validation."""
    bursts = {}
    for r in records:
        bursts.setdefault(r[burst_field], []).append(r)

    burst_list = list(bursts.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(burst_list)
    burst_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(records)
    target = total / k
    counts = [0] * k

    for _burst_id, burst_records in burst_list:
        chosen = min(range(k), key=lambda i: counts[i] - target)
        counts[chosen] += len(burst_records)
        for r in burst_records:
            r[fold_field] = str(chosen)

    if any(c == 0 for c in counts):
        print(f"  WARNING: '{group_name}' has an empty fold in {k}-fold CV -> {counts}")

    return counts


def determine_k(records_by_group, group_names, burst_field, desired_k):
    """Caps desired_k to the scarcest group's independent-session count
    (assign_bursts must already have run) so no fold is structurally
    guaranteed to be empty for some class."""
    burst_counts = {
        group: len({r[burst_field] for r in records_by_group.get(group, [])})
        for group in group_names
    }
    max_k = min(burst_counts.values()) if burst_counts else 1
    k = max(1, min(desired_k, max_k))
    if k < desired_k:
        print(
            f"  Note: wanted {desired_k}-fold CV but the scarcest group only has "
            f"{max_k} independent session(s) {burst_counts} -> using {k}-fold instead."
        )
    else:
        print(f"  {desired_k}-fold CV is fully supported {burst_counts}.")
    return k


def build_scheme(records_by_group, group_names, burst_field, ratio_configs):
    """Runs burst-assignment (once) and a split per ratio for each group.
    ratio_configs: list of (split_field, train_ratio)."""
    for group in group_names:
        assign_bursts(records_by_group.get(group, []), burst_field)

    for split_field, train_ratio in ratio_configs:
        print(f"\n--- {burst_field} / {split_field or '(unsuffixed)'} — train ratio {train_ratio:.0%} ---")
        grid = {}
        for group in group_names:
            records = records_by_group.get(group, [])
            counts = split_bursts(
                records, group, burst_field, split_field, train_ratio,
                min_train=MIN_TRAIN_FLOOR.get(group, 0),
            )
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

    lite_ratio_configs = [(f"split{suffix}", ratio) for suffix, ratio in RATIOS]
    extended_ratio_configs = [(f"split_extended{suffix}", ratio) for suffix, ratio in RATIOS]

    # Lite (3-class) scheme: bursts/splits computed within each of valid/invalid/empty.
    by_lite = {label: [r for r in all_records if r["label"] == label] for label in LITE_LABELS}
    build_scheme(by_lite, LITE_LABELS, burst_field="burst_id", ratio_configs=lite_ratio_configs)

    # Extended (5-class) scheme: recomputed independently within each of the
    # 5 groups, so a burst can never mix two extended labels.
    by_extended = {
        group: [r for r in all_records if r["extended_label"] == group]
        for group in EXTENDED_LABELS
    }
    build_scheme(
        by_extended, EXTENDED_LABELS, burst_field="burst_id_extended", ratio_configs=extended_ratio_configs
    )

    print(f"\n--- {K_FOLDS_LITE}-fold CV assignment (lite) ---")
    for label in LITE_LABELS:
        assign_folds(by_lite[label], label, "burst_id", "fold_lite", K_FOLDS_LITE)

    print(f"\n--- {K_FOLDS_EXTENDED}-fold CV assignment (extended) ---")
    for group in EXTENDED_LABELS:
        assign_folds(by_extended[group], group, "burst_id_extended", "fold_extended", K_FOLDS_EXTENDED)

    # --- No-empty scheme: pictures/no-empty version/ — the same 176 photos,
    # but "empty" was manually redistributed by hand into invalid_day /
    # invalid_night, so it isn't its own class here. Two granularities of
    # this, same as lite/extended: collapsed (2-class) and split (4-class).
    if not os.path.isdir(NOEMPTY_DIR):
        raise RuntimeError(
            f"Missing {NOEMPTY_DIR} — this scheme needs pictures/no-empty version/ "
            f"(a copy of extended version/ with the empty photos manually moved "
            f"into invalid_day/invalid_night)."
        )

    noempty_folder_to_split_label = {
        "תקין יום": "valid_day",
        "תקין לילה": "valid_night",
        "לא תקין יום": "invalid_day",
        "לא תקין לילה": "invalid_night",
    }
    noempty_records = []
    for folder_name, split_label in noempty_folder_to_split_label.items():
        folder_path = os.path.join(NOEMPTY_DIR, folder_name)
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"Missing expected folder: {folder_path}")
        collapsed_label, daynight = split_label.rsplit("_", 1)
        for fname in sorted(os.listdir(folder_path)):
            if not os.path.isfile(os.path.join(folder_path, fname)):
                continue
            original_filename, _ = manifest.get(fname, (None, None))
            captured_at = parse_captured_at(original_filename) if original_filename else None
            noempty_records.append({
                "filename": fname,
                "noempty_label": collapsed_label,
                "noempty_split_label": split_label,
                "original_filename": original_filename or "",
                "captured_at": captured_at,
            })

    noempty_by_filename = {r["filename"]: r for r in noempty_records}
    if set(noempty_by_filename) != {r["filename"] for r in all_records}:
        raise RuntimeError(
            "pictures/no-empty version/ doesn't have the exact same 176 filenames "
            "as the other dataset versions — check for a partial move or a stray file."
        )

    noempty_collapsed_ratio_configs = [(f"split_noempty{suffix}", ratio) for suffix, ratio in RATIOS]
    noempty_split_ratio_configs = [(f"split_noempty4{suffix}", ratio) for suffix, ratio in RATIOS]

    by_noempty_collapsed = {
        label: [r for r in noempty_records if r["noempty_label"] == label]
        for label in NOEMPTY_COLLAPSED_LABELS
    }
    build_scheme(
        by_noempty_collapsed, NOEMPTY_COLLAPSED_LABELS,
        burst_field="burst_id_noempty", ratio_configs=noempty_collapsed_ratio_configs,
    )

    by_noempty_split = {
        label: [r for r in noempty_records if r["noempty_split_label"] == label]
        for label in NOEMPTY_SPLIT_LABELS
    }
    build_scheme(
        by_noempty_split, NOEMPTY_SPLIT_LABELS,
        burst_field="burst_id_noempty4", ratio_configs=noempty_split_ratio_configs,
    )

    print(f"\n--- CV assignment (no-empty, collapsed 2-class) — want {K_FOLDS_NOEMPTY_DESIRED}-fold ---")
    k_noempty_collapsed = determine_k(
        by_noempty_collapsed, NOEMPTY_COLLAPSED_LABELS, "burst_id_noempty", K_FOLDS_NOEMPTY_DESIRED
    )
    for label in NOEMPTY_COLLAPSED_LABELS:
        assign_folds(by_noempty_collapsed[label], label, "burst_id_noempty", "fold_noempty", k_noempty_collapsed)

    print(f"\n--- CV assignment (no-empty, split 4-class) — want {K_FOLDS_NOEMPTY_DESIRED}-fold ---")
    k_noempty_split = determine_k(
        by_noempty_split, NOEMPTY_SPLIT_LABELS, "burst_id_noempty4", K_FOLDS_NOEMPTY_DESIRED
    )
    for label in NOEMPTY_SPLIT_LABELS:
        assign_folds(by_noempty_split[label], label, "burst_id_noempty4", "fold_noempty4", k_noempty_split)

    lite_split_cols = [field for field, _ in lite_ratio_configs]
    extended_split_cols = [field for field, _ in extended_ratio_configs]
    noempty_collapsed_split_cols = [field for field, _ in noempty_collapsed_ratio_configs]
    noempty_split_split_cols = [field for field, _ in noempty_split_ratio_configs]

    noempty_merge_cols = (
        ["noempty_label", "burst_id_noempty"] + noempty_collapsed_split_cols + ["fold_noempty"]
        + ["noempty_split_label", "burst_id_noempty4"] + noempty_split_split_cols + ["fold_noempty4"]
    )
    for r in all_records:
        ne = noempty_by_filename[r["filename"]]
        for col in noempty_merge_cols:
            r[col] = ne[col]

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "original_filename", "captured_at", "daynight", "label", "burst_id"]
            + lite_split_cols
            + ["fold_lite"]
            + ["extended_label", "burst_id_extended"]
            + extended_split_cols
            + ["fold_extended"]
            + noempty_merge_cols
        )
        for r in all_records:
            writer.writerow(
                [
                    r["filename"], r["original_filename"],
                    r["captured_at"].isoformat() if r["captured_at"] else "",
                    r["daynight"], r["label"], r["burst_id"],
                ]
                + [r[col] for col in lite_split_cols]
                + [r["fold_lite"]]
                + [r["extended_label"], r["burst_id_extended"]]
                + [r[col] for col in extended_split_cols]
                + [r["fold_extended"]]
                + [r[col] for col in noempty_merge_cols]
            )

    print(f"\nWrote {len(all_records)} rows to {OUTPUT}")
    print(f"Lite split columns: {lite_split_cols}")
    print(f"Extended split columns: {extended_split_cols}")
    print(f"No-empty (2-class) split columns: {noempty_collapsed_split_cols}")
    print(f"No-empty (4-class) split columns: {noempty_split_split_cols}")
    print(
        "\nNote: session sizes are uneven (one big shoot per class often holds most "
        "of the photos), so a ratio target is approximate, not exact, once whole "
        "sessions are kept together — see each scheme's 'test %' above. The 'empty' "
        "class (in lite/extended) is additionally floored at 3 train images "
        "regardless of ratio."
    )


if __name__ == "__main__":
    main()
