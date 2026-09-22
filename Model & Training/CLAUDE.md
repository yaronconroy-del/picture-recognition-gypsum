# Model & Training

Role: this is where the gypsum-filter classifier actually gets built — the labeled dataset and everything involved in turning it into a trained model.

What's here:
- `pictures/` — the labeled photo dataset (5 classes: valid day/night, invalid day/night, empty filter). Moved here from the project root.
- `scripts/rename_images.py` — the one-off script that gave the original 176 photos their random `img_<id>.jpeg` names and wrote the manifest. Kept for reproducibility; don't re-run it (it would re-randomize already-renamed files).
- `scripts/add_photos.py` — the ongoing version for new photos (see the `add-dataset-photos` skill).
- `context/` — empty for now. Drop in notes on your training environment, model choices, hyperparameters, etc. as you go.

There's no training code here yet — add a `src/` (or similar) folder once you start writing the Python training pipeline.
