# models/

**The label-scheme decision is made** (see `Evaluation & Docs/Project
Workflow.md` §12): no-empty split (4-class) is the project's chosen
scheme.

- `gypsum_classifier_split_70_30.pt` / `.json` — **the promoted main
  model**, no-empty split (4-class), 70/30 split. Produced by
  `Model & Training/notebooks/full_pipeline.ipynb` — not yet run for real
  as of this decision, so these files don't exist locally yet. Run the
  notebook in Colab and bring the weights back to populate this.
- `gypsum_classifier_extended_70_30.pt` / `.json` — the extended (5-class)
  runner-up, trained for real by `scripts/train_extended_70_30.py` (kept,
  not deleted — a close second candidate). Also here:
  `extended_70_30_training_curves.png` and `extended_70_30_confusion_matrix.png`.
- `gypsum_classifier.pt` / `.json` — not trained for real (only
  smoke-tested). Would be the lite (3-class) model from
  `train_and_evaluate.py` / `.ipynb` — lite is a ruled-out scheme as of
  §12, so probably not worth finishing.

Every `.pt` file that should be tracked needs its own exception in the
root `.gitignore` (the general rule ignores all `*.pt`) —
`gypsum_classifier_split_70_30.pt` already has one, ready for whenever
the notebook produces it.
