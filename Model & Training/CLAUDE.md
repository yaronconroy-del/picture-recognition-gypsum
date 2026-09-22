# Model & Training

Role: this is where the gypsum-filter classifier actually gets built — the labeled dataset and everything involved in turning it into a trained model.

What's here:
- `pictures/extended version/` — the original 5-class dataset (valid day/night, invalid day/night, empty filter), preserved as-is.
- `pictures/simple version/` — the same 176 photos regrouped into the 3 classes actually used for training (`valid`, `invalid`, `empty` — day/night collapsed). Every file here is a copy of one in `extended version/`, same filename.
- `scripts/rename_images.py` — the one-off script that gave the original 176 photos their random `img_<id>.jpeg` names and wrote the manifest. Kept for reproducibility; don't re-run it (it would re-randomize already-renamed files).
- `scripts/add_photos.py` — the ongoing version for new photos: renames new files in `extended version/`, logs them to the manifest, and mirrors each one into the matching `simple version/` folder (see the `add-dataset-photos` skill).
- `scripts/build_dataset_index.py` — builds `dataset_split.csv`, the leakage-safe train/test split (by real capture session, not per-image).
- `dataset_split.csv` — generated output of the script above; what the notebook reads.
- `notebooks/train_and_evaluate.ipynb` — the actual model: data loading, augmentation, a pretrained-backbone classifier, a small hyperparameter comparison, evaluation on the held-out test set, and export. Runs in Google Colab (see the notebook's own setup cell for how to clone this private repo there).
- `models/` — not populated yet; where the trained weights land after running the notebook (see `models/README.md`).
- `context/` — empty for now. Drop in notes on your training environment, model choices, hyperparameters, etc. as you go.
