# Model & Training

Role: this is where the gypsum-filter classifier actually gets built — the labeled dataset and everything involved in turning it into a trained model.

What's here:
- `pictures/extended version/` — the original 5-class dataset (valid day/night, invalid day/night, empty filter), preserved as-is.
- `pictures/simple version/` — the same 176 photos regrouped into the 3 classes actually used for training (`valid`, `invalid`, `empty` — day/night collapsed). Every file here is a copy of one in `extended version/`, same filename.
- `scripts/rename_images.py` — the one-off script that gave the original 176 photos their random `img_<id>.jpeg` names and wrote the manifest. Kept for reproducibility; don't re-run it (it would re-randomize already-renamed files).
- `scripts/add_photos.py` — the ongoing version for new photos: renames new files in `extended version/`, logs them to the manifest, and mirrors each one into the matching `simple version/` folder (see the `add-dataset-photos` skill).
- `scripts/build_dataset_index.py` — builds `dataset_split.csv`, with **two independent leakage-safe train/test splits** (by real capture session, not per-image): the lite 3-class scheme (`label`/`split`) and the extended 5-class scheme (`extended_label`/`split_extended`). They're independent because a "session" in one scheme doesn't line up with sessions in the other — see the note on `daynight` below.
- `dataset_split.csv` — generated output of the script above; what the notebooks/scripts read.
- `notebooks/train_and_evaluate.ipynb` — the main model: data loading, augmentation, a pretrained-backbone classifier, a small hyperparameter comparison, evaluation on the held-out test set, and export. Uses the lite (3-class) scheme. Runs in Google Colab (see the notebook's own setup cell for how to clone this private repo there).
- `scripts/train_and_evaluate.py` — the same pipeline as the notebook above, as a plain script (`--quick` for a fast local CPU smoke test; drop `--quick` for a real run, though Colab's GPU is much faster).
- `notebooks/compare_label_schemes.ipynb` / `scripts/compare_label_schemes.py` — a separate A/B test: trains a lite-scheme model and an extended-scheme model with the same fixed config, to check whether collapsing day/night into one label (what the main model does) loses anything vs. keeping them separate. Saves to `models/label_scheme_comparison/` (only `comparison.json` is meant to be committed — the two `.pt` files there are an experiment artifact, not the project's model).
- **Keep every `.py`/`.ipynb` pair in sync** — when the modeling logic changes in one, mirror the change into the other.
- `models/` — not populated yet; where the trained weights land after running a notebook or script (see `models/README.md`).
- `context/` — empty for now. Drop in notes on your training environment, model choices, hyperparameters, etc. as you go.

**Note on `daynight`**: despite the name, the source timestamps show `day`- and `night`-labeled photos interleaved within the same few seconds of each other (see `Evaluation & Docs/Project Workflow.md` §3.4) — so whatever these folders actually encode, it isn't literal time-of-day. Worth clarifying with whoever collected the photos before leaning on it in the report.
