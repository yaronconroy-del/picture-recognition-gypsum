"""
Rename newly added, not-yet-renamed photos in each class folder under
Model & Training/pictures/extended version/ using the project's naming
convention (img_<8 random hex chars>.<ext>, globally unique, no
sequential ranges), append the original -> new filename mapping to the
rename manifest, and mirror the new file into the matching
Model & Training/pictures/simple version/<valid|invalid|empty> folder.

Usage: python add_photos.py
(Run after manually sorting new photos into the correct 5-class folder
under "extended version".)
"""


import csv
import os
import re
import secrets
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PICTURES = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "pictures"))
EXTENDED = os.path.join(PICTURES, "extended version")
SIMPLE = os.path.join(PICTURES, "simple version")
MANIFEST = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "Evaluation & Docs", "context", "rename_manifest.csv")
)

NAME_RE = re.compile(r"^img_[0-9a-f]{8}\.[A-Za-z0-9]+$")

# 5-class (extended) -> 3-class (simple) mapping
CLASS_MAP = {
    "תקין יום": "valid",
    "תקין לילה": "valid",
    "לא תקין יום": "invalid",
    "לא תקין לילה": "invalid",
    "מסנן ריק": "empty",
}


def class_folders():
    return sorted(
        d for d in os.listdir(EXTENDED) if os.path.isdir(os.path.join(EXTENDED, d))
    )


def load_used_tokens():
    used = set()
    for folder in class_folders():
        folder_path = os.path.join(EXTENDED, folder)
        for fname in os.listdir(folder_path):
            if NAME_RE.match(fname):
                used.add(fname.split("_", 1)[1].split(".")[0])
    return used


def main():
    used = load_used_tokens()
    new_rows = []

    for folder in class_folders():
        folder_path = os.path.join(EXTENDED, folder)
        simple_class = CLASS_MAP.get(folder)
        if simple_class is None:
            print(f"Warning: '{folder}' has no simple-version mapping, skipping")
            continue

        for fname in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, fname)
            if not os.path.isfile(path) or NAME_RE.match(fname):
                continue  # already renamed, or not a file

            ext = os.path.splitext(fname)[1].lower()
            while True:
                token = secrets.token_hex(4)  # 8 hex chars
                if token not in used:
                    used.add(token)
                    break
            new_name = f"img_{token}{ext}"

            os.rename(path, os.path.join(folder_path, new_name))
            new_rows.append([folder, fname, new_name])
            print(f"{folder}: {fname} -> {new_name}")

            simple_dir = os.path.join(SIMPLE, simple_class)
            os.makedirs(simple_dir, exist_ok=True)
            shutil.copy2(os.path.join(folder_path, new_name), os.path.join(simple_dir, new_name))

    if not new_rows:
        print("No new (un-renamed) files found - nothing to do.")
        return

    manifest_exists = os.path.exists(MANIFEST)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not manifest_exists:
            writer.writerow(["class_folder", "original_filename", "new_filename"])
        writer.writerows(new_rows)

    print(f"Logged {len(new_rows)} new file(s) to {MANIFEST}")
    print("Mirrored new file(s) into the matching 'simple version' class folder.")


if __name__ == "__main__":
    main()
