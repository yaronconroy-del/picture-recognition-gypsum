# models/

**Final models** (see `Evaluation & Docs/Project Workflow.md` §12 for the
label-scheme decision and §18 for the final run):

- `gypsum_classifier_split_70_30.pt` / `.json` — **the promoted main
  model**, no-empty split (4-class), 70/30 split, trained on the
  day/night-corrected data by the real `full_pipeline.ipynb` run (§18).
  `scripts/predict.py` and `scripts/watch_camera_feed.py` use this one
  automatically.
- `gypsum_classifier_extended_70_30.pt` / `.json` — the extended (5-class)
  runner-up, from the same run. (This replaced an earlier version trained on
  the uncorrected data by `scripts/train_extended_70_30.py` — still in git
  history. `extended_70_30_training_curves.png` and
  `extended_70_30_confusion_matrix.png` here are from that earlier run.)
- `gypsum_classifier.pt` / `.json` — not trained for real (only
  smoke-tested). Would be the lite (3-class) model from
  `train_and_evaluate.py` / `.ipynb` — lite is a ruled-out scheme (§12).

Subfolders hold each analysis's results: `full_pipeline/` (the final run —
every config's training curves + confusion matrix, all threshold/ROC/PR/
calibration and ensemble plots, `full_pipeline_results.json`),
`corrected_daynight/` (§17's first retrain after the label fix),
`threshold_analysis/`, `boost_analysis/`, `optimization_sweep*/`,
`label_scheme_comparison/` (the earlier, uncorrected analyses).

Every `.pt` file that should be tracked needs its own exception in the
root `.gitignore` (the general rule ignores all `*.pt`).
