# models/

Two model candidates, pending the open label-scheme decision (see
`Evaluation & Docs/Project Workflow.md` §7):

- `gypsum_classifier_extended_70_30.pt` / `.json` — the extended (5-class),
  70/30 split model, trained for real by
  `Model & Training/scripts/train_extended_70_30.py` (or the matching
  notebook, in Colab). This was the best-performing config in the
  optimization sweep (§6). Also here: `extended_70_30_training_curves.png`
  and `extended_70_30_confusion_matrix.png`.
- `gypsum_classifier.pt` / `.json` — not trained for real yet (only
  smoke-tested). Would be the lite (3-class), 80/20 model from
  `train_and_evaluate.py` / `train_and_evaluate.ipynb`.

Both `.pt` files are named as exceptions in the root `.gitignore` so they
get committed once they exist, unlike other `*.pt` files.
