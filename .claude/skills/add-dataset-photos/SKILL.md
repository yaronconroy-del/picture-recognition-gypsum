---
name: add-dataset-photos
description: Sort newly captured gypsum-filter photos into the dataset — rename them with the project's random-ID convention and log the mapping, without disturbing already-renamed files.
---

# Add dataset photos

Use this when the user says they've added new filter-camera photos and wants them folded into the dataset (e.g. "I dropped new photos in", "sort these into the dataset", "add these to the training set").

## Steps

1. **Confirm the photos are already sorted by class.** New raw photos must be manually placed into the matching class folder under `Model & Training/pictures/`:
   - `תקין יום` — valid, day
   - `תקין לילה` — valid, night
   - `לא תקין יום` — invalid, day
   - `לא תקין לילה` — invalid, night
   - `מסנן ריק` — empty filter
   If the user hasn't sorted them yet, or you're unsure which class a photo belongs to, ask — don't guess the label from the image yourself. Misclassified training data is worse than a clarifying question.

2. **Run the rename script:**
   ```
   python "Model & Training/scripts/add_photos.py"
   ```
   It only touches files that don't already match the `img_<8-hex-chars>.<ext>` convention, so it's safe to re-run — already-renamed files are left alone.

3. **It automatically:**
   - Renames each new file to `img_<8 random hex chars>.<ext>`, checking uniqueness against every existing filename in the dataset (not just its own folder), so the same random-ID convention from the existing 176 photos is preserved — no sequential runs, no per-folder number blocks.
   - Appends `class_folder, original_filename, new_filename` rows to `Evaluation & Docs/context/rename_manifest.csv` for the new files only (existing manifest rows are untouched).

4. **Report back** the per-class count of newly added files (the script prints each rename as it happens).
