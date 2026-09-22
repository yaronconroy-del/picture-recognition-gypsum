import csv
import os
import secrets

ROOT = r"E:\AI Projects\picture recognition - gypsum\Model & Training\pictures"
MANIFEST = r"E:\AI Projects\picture recognition - gypsum\Evaluation & Docs\context\rename_manifest.csv"

folders = sorted(
    d for d in os.listdir(ROOT)
    if os.path.isdir(os.path.join(ROOT, d))
)

all_files = []  # (folder, original_filename)
for folder in folders:
    folder_path = os.path.join(ROOT, folder)
    for fname in sorted(os.listdir(folder_path)):
        if os.path.isfile(os.path.join(folder_path, fname)):
            all_files.append((folder, fname))

print(f"Found {len(all_files)} files across {len(folders)} folders")

# Generate globally unique random tokens, independent of folder/order.
used = set()
new_names = []
for folder, fname in all_files:
    ext = os.path.splitext(fname)[1].lower()
    while True:
        token = secrets.token_hex(4)  # 8 hex chars
        if token not in used:
            used.add(token)
            break
    new_names.append(f"img_{token}{ext}")

# Rename in two passes (temp name, then final name) to avoid any
# accidental collision with an original filename mid-rename.
temp_suffix = ".renaming_tmp"
for (folder, fname), new_name in zip(all_files, new_names):
    folder_path = os.path.join(ROOT, folder)
    src = os.path.join(folder_path, fname)
    tmp = os.path.join(folder_path, fname + temp_suffix)
    os.rename(src, tmp)

for (folder, fname), new_name in zip(all_files, new_names):
    folder_path = os.path.join(ROOT, folder)
    tmp = os.path.join(folder_path, fname + temp_suffix)
    dst = os.path.join(folder_path, new_name)
    os.rename(tmp, dst)

os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["class_folder", "original_filename", "new_filename"])
    for (folder, fname), new_name in zip(all_files, new_names):
        writer.writerow([folder, fname, new_name])

print(f"Renamed {len(all_files)} files. Manifest written to {MANIFEST}")

# Sanity check: per-folder counts unchanged, all new names unique.
counts = {}
for folder in folders:
    folder_path = os.path.join(ROOT, folder)
    counts[folder] = len([f for f in os.listdir(folder_path) if f.startswith("img_")])
print("Per-folder counts after rename:", counts)
assert len(set(new_names)) == len(new_names), "Duplicate new names generated!"
