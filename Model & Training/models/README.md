# models/

Not populated yet — this is where the trained model lands after running
`Model & Training/notebooks/train_and_evaluate.ipynb` in Colab and
downloading its output:

- `gypsum_classifier.pt` — the trained weights (PyTorch state dict)
- `gypsum_classifier.json` — metadata (backbone name, class order, test
  metrics) that `scripts/predict.py` needs to rebuild the same architecture

Both are named as an exception in the root `.gitignore` so they get
committed once they exist, unlike other `*.pt` files.
