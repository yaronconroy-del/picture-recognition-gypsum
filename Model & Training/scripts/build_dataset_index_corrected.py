"""
Builds dataset_split_corrected.csv — the same shape as dataset_split.csv,
but with extended_label / noempty_split_label (and everything derived from
them: bursts, splits, folds) re-derived from the CORRECTED folder
structure in "pictures/extended version (day-night corrected)/" and
"pictures/no-empty version (day-night corrected)/", instead of from the
original rename manifest's class_folder.

Why a separate script rather than re-running build_dataset_index.py: that
script reads each photo's day/night from the manifest's recorded
class_folder (its ORIGINAL sort, at rename time), not from which folder it
currently sits in — so it can't see the one-photo day/night correction
made after a manual visual review (Evaluation & Docs/Project Workflow.md
SS17). This script reads day/night live from the corrected folders instead.

Only the day/night-dependent columns actually change for one photo
(img_dfa74e21.jpeg: valid_night -> valid_day), which changes group
membership for the extended and no-empty-split schemes -- and therefore
their burst/split/fold assignment needs recomputing. The lite scheme and
the no-empty COLLAPSED scheme don't depend on day/night at all (day/night
photos share the same lite/collapsed label either way), so those columns
are carried over unchanged from dataset_split.csv.

Usage: python build_dataset_index_corrected.py
"""

import csv
import os
import random
from datetime import timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ORIGINAL_CSV = os.path.join(TRAINING_DIR, "dataset_split.csv")
EXTENDED_CORRECTED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version (day-night corrected)")
NOEMPTY_CORRECTED_DIR = os.path.join(TRAINING_DIR, "pictures", "no-empty version (day-night corrected)")
OUTPUT = os.path.join(TRAINING_DIR, "dataset_split_corrected.csv")

EXTENDED_LABELS = ["valid_day", "valid_night", "invalid_day", "invalid_night", "empty"]
NOEMPTY_SPLIT_LABELS = ["valid_day", "valid_night", "invalid_day", "invalid_night"]

EXTENDED_FOLDER_TO_LABEL = {
    "תקין יום": "valid_day", "תקין לילה": "valid_night",
    "לא תקין יום": "invalid_day", "לא תקין לילה": "invalid_night",
    "מסנן ריק": "empty",
}
NOEMPTY_FOLDER_TO_LABEL = {
    "תקין יום": "valid_day", "תקין לילה": "valid_night",
    "לא תקין יום": "invalid_day", "לא תקין לילה": "invalid_night",
}

BURST_GAP = timedelta(seconds=120)
RATIOS = [("", 0.80), ("_75", 0.75), ("_70", 0.70)]
MIN_TRAIN_FLOOR = {"empty": 3}
K_FOLDS_EXTENDED = 2
K_FOLDS_NOEMPTY_DESIRED = 4
RANDOM_SEED = 42


def daynight_of(label):
    if label == "empty":
        return "n/a"
    return label.rsplit("_", 1)[1]


def scan_folder(base_dir, folder_to_label, label_field):
    """filename -> {label_field: ..., daynight: ...} from live folder membership."""
    out = {}
    for folder_name, label in folder_to_label.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            raise RuntimeError(f"Missing expected folder: {folder_path}")
        for fname in sorted(os.listdir(folder_path)):
            if not os.path.isfile(os.path.join(folder_path, fname)):
                continue
            out[fname] = {label_field: label, "daynight": daynight_of(label)}
    return out


def assign_bursts(records, burst_field):
    with_time = [r for r in records if r["captured_at"]]
    without_time = [r for r in records if not r["captured_at"]]
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
    bursts = {}
    for r in records:
        bursts.setdefault(r[burst_field], []).append(r)

    burst_list = list(bursts.items())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(burst_list)
    burst_list.sort(key=lambda kv: len(kv[1]), reverse=True)

    total = len(records)
    targets = {"train": total * train_ratio, "test": total * (1 - train_ratio)}
    counts = {"train": 0, "test": 0}
    assignment = {}

    for burst_id, burst_records in burst_list:
        deficit = {s: targets[s] - counts[s] for s in ("train", "test")}
        chosen = max(deficit, key=deficit.get)
        counts[chosen] += len(burst_records)
        assignment[burst_id] = chosen

    if counts["train"] < min_train:
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
        print(f"  Note: '{group_name}' topped up to the {min_train}-image train floor -> {counts}")

    for burst_id, split in assignment.items():
        for r in bursts[burst_id]:
            r[split_field] = split

    print(f"  {group_name}: {len(records)} images -> {counts}")
    return counts


def assign_folds(records, group_name, burst_field, fold_field, k):
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
    burst_counts = {
        group: len({r[burst_field] for r in records_by_group.get(group, [])})
        for group in group_names
    }
    max_k = min(burst_counts.values()) if burst_counts else 1
    k = max(1, min(desired_k, max_k))
    if k < desired_k:
        print(f"  Note: wanted {desired_k}-fold but scarcest group has {max_k} session(s) {burst_counts} -> using {k}-fold.")
    else:
        print(f"  {desired_k}-fold CV is fully supported {burst_counts}.")
    return k


def build_scheme(records_by_group, group_names, burst_field, ratio_configs):
    for group in group_names:
        assign_bursts(records_by_group.get(group, []), burst_field)
    for split_field, train_ratio in ratio_configs:
        print(f"\n--- {burst_field} / {split_field or '(unsuffixed)'} — train ratio {train_ratio:.0%} ---")
        for group in group_names:
            records = records_by_group.get(group, [])
            split_bursts(records, group, burst_field, split_field, train_ratio, min_train=MIN_TRAIN_FLOOR.get(group, 0))


def main():
    # Load the original CSV as the base -- carries over every column that
    # doesn't depend on day/night unchanged.
    with open(ORIGINAL_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows_by_filename = {row["filename"]: row for row in reader}
        fieldnames = reader.fieldnames

    print(f"Loaded {len(rows_by_filename)} rows from {ORIGINAL_CSV}")

    # --- Re-derive extended_label live from the corrected folder ---
    extended_info = scan_folder(EXTENDED_CORRECTED_DIR, EXTENDED_FOLDER_TO_LABEL, "extended_label")
    if set(extended_info) != set(rows_by_filename):
        raise RuntimeError("Corrected extended-version folder doesn't match dataset_split.csv's filenames.")

    changed = [
        fn for fn in rows_by_filename
        if rows_by_filename[fn]["extended_label"] != extended_info[fn]["extended_label"]
    ]
    print(f"\nExtended-label changes vs. original: {len(changed)}")
    for fn in changed:
        print(f"  {fn}: {rows_by_filename[fn]['extended_label']} -> {extended_info[fn]['extended_label']}")

    all_records = []
    for fn, row in rows_by_filename.items():
        all_records.append({
            "filename": fn,
            "original_filename": row["original_filename"],
            "captured_at": row["captured_at"] or None,
            "label": row["label"],
            "burst_id": row["burst_id"],
            **{f"split{suf}": row[f"split{suf}"] for suf, _ in RATIOS},
            "fold_lite": row["fold_lite"],
            "extended_label": extended_info[fn]["extended_label"],
            "daynight": extended_info[fn]["daynight"],
            "noempty_label": row["noempty_label"],
            "burst_id_noempty": row["burst_id_noempty"],
            **{f"split_noempty{suf}": row[f"split_noempty{suf}"] for suf, _ in RATIOS},
            "fold_noempty": row["fold_noempty"],
        })

    # captured_at needs to be a real datetime object for burst assignment.
    from datetime import datetime
    for r in all_records:
        r["captured_at"] = datetime.fromisoformat(r["captured_at"]) if r["captured_at"] else None

    # --- Extended scheme: recompute bursts/splits/folds (group membership changed) ---
    by_extended = {g: [r for r in all_records if r["extended_label"] == g] for g in EXTENDED_LABELS}
    extended_ratio_configs = [(f"split_extended{suf}", ratio) for suf, ratio in RATIOS]
    print("\n=== Extended scheme (recomputed) ===")
    build_scheme(by_extended, EXTENDED_LABELS, burst_field="burst_id_extended", ratio_configs=extended_ratio_configs)
    print(f"\n--- {K_FOLDS_EXTENDED}-fold CV (extended) ---")
    for g in EXTENDED_LABELS:
        assign_folds(by_extended[g], g, "burst_id_extended", "fold_extended", K_FOLDS_EXTENDED)

    # --- No-empty split scheme: live scan of the corrected no-empty folder ---
    noempty_info = scan_folder(NOEMPTY_CORRECTED_DIR, NOEMPTY_FOLDER_TO_LABEL, "noempty_split_label")
    if set(noempty_info) != set(rows_by_filename):
        raise RuntimeError("Corrected no-empty-version folder doesn't match dataset_split.csv's filenames.")

    for r in all_records:
        r["noempty_split_label"] = noempty_info[r["filename"]]["noempty_split_label"]

    by_noempty_split = {g: [r for r in all_records if r["noempty_split_label"] == g] for g in NOEMPTY_SPLIT_LABELS}
    noempty_split_ratio_configs = [(f"split_noempty4{suf}", ratio) for suf, ratio in RATIOS]
    print("\n=== No-empty split scheme (recomputed) ===")
    build_scheme(by_noempty_split, NOEMPTY_SPLIT_LABELS, burst_field="burst_id_noempty4", ratio_configs=noempty_split_ratio_configs)
    print(f"\n--- CV (no-empty split) — want {K_FOLDS_NOEMPTY_DESIRED}-fold ---")
    k_noempty_split = determine_k(by_noempty_split, NOEMPTY_SPLIT_LABELS, "burst_id_noempty4", K_FOLDS_NOEMPTY_DESIRED)
    for g in NOEMPTY_SPLIT_LABELS:
        assign_folds(by_noempty_split[g], g, "burst_id_noempty4", "fold_noempty4", k_noempty_split)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in all_records:
            writer.writerow([
                r["filename"] if col == "filename" else
                r["original_filename"] if col == "original_filename" else
                (r["captured_at"].isoformat() if r["captured_at"] else "") if col == "captured_at" else
                r.get(col, "")
                for col in fieldnames
            ])

    print(f"\nWrote {len(all_records)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
