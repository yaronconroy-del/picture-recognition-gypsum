"""
Three ways to boost the invalid-detection results beyond a single model's
argmax, built on top of threshold_analysis.py's two leading, closely-matched
schemes (extended CV and no-empty-split CV):

  1. Ensemble: average extended's and no-empty-split's P(invalid), per
     image, using only each scheme's own properly held-out (CV) fold model
     for that image — they're different burst/session partitions, so this
     is still leakage-safe even though the two schemes' folds don't line up.
  2. Test-time augmentation (TTA): at inference, average the softmax over
     several mildly-augmented views of each image instead of one
     deterministic center crop.
  3. Calibration: fit Platt scaling (a 1D logistic regression) on the
     ensemble+TTA scores, so the probabilities themselves are better
     calibrated, not just accurate.

Ground truth throughout is the no-empty definition of "invalid" (empty
folded in) — both schemes' predictions are compared against it, since
that's the framing the project's own findings favor.

Reuses threshold_analysis.py's helpers directly (import, not duplicated —
same exception as that script's notebook, for the same reason: this is a
close companion to it, not a standalone piece).

Only 4 trainings needed (extended fold0/fold1, split fold0/fold1) since
the ensemble/TTA/calibration steps are all built on top of those, not new
training runs.

Usage:
  python boost_analysis.py            # full run
  python boost_analysis.py --quick    # fast smoke test (1 epoch, TTA views=2)
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import threshold_analysis as ta

OUT_DIR = os.path.join(ta.TRAINING_DIR, "models", "boost_analysis")


def predict_probs_tta(model, df_subset, path_fn, tta_transform, eval_transform, device, n_views):
    """1 deterministic center-crop view + (n_views - 1) stochastic
    augmented views, softmax-averaged, per image."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for _, row in df_subset.iterrows():
            img = ta.Image.open(path_fn(row)).convert("RGB")
            img = ta.mask_timestamp(img)
            views = [eval_transform(img)] + [tta_transform(img) for _ in range(max(n_views - 1, 0))]
            batch = torch.stack(views).to(device)
            probs = torch.softmax(model(batch), dim=1).mean(dim=0).cpu().numpy()
            all_probs.append(probs)
    return np.array(all_probs)


def train_fold_models(df, label_col, path_fn, invalid_classes, fold_col, epochs, n_tta_views, device):
    """Trains one model per fold (same as threshold_analysis's CV configs),
    and for each fold's held-out images returns both a single-view and a
    TTA-averaged P(invalid). Returns {filename: (p_single, p_tta)}."""
    k = df[fold_col].nunique()
    classes = sorted(df[label_col].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    invalid_idx = [class_to_idx[c] for c in invalid_classes]

    out = {}
    for fold_i in range(k):
        print(f"    fold {fold_i + 1}/{k}")
        test_df = df[df[fold_col] == str(fold_i)].reset_index(drop=True)
        train_df_full = df[df[fold_col] != str(fold_i)].reset_index(drop=True)
        train_df, val_df = train_test_split(
            train_df_full, test_size=0.15, stratify=train_df_full[label_col], random_state=ta.SEED,
        )
        train_transform, eval_transform = ta.make_transforms()
        train_ds = ta.GypsumDataset(train_df, path_fn, label_col, class_to_idx, train_transform)
        val_ds = ta.GypsumDataset(val_df, path_fn, label_col, class_to_idx, eval_transform)
        train_loader = DataLoader(train_ds, batch_size=ta.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=ta.BATCH_SIZE, shuffle=False, num_workers=0)

        class_weights = ta.compute_class_weights(train_df, label_col, classes, class_to_idx)
        model, _ = ta.run_training(epochs, train_loader, val_loader, class_weights, device)

        test_ds = ta.GypsumDataset(test_df, path_fn, label_col, class_to_idx, eval_transform)
        test_loader = DataLoader(test_ds, batch_size=ta.BATCH_SIZE, shuffle=False, num_workers=0)
        probs_single, _ = ta.predict_probs(model, test_loader, device)
        p_single = probs_single[:, invalid_idx].sum(axis=1)

        probs_tta = predict_probs_tta(model, test_df, path_fn, train_transform, eval_transform, device, n_tta_views)
        p_tta = probs_tta[:, invalid_idx].sum(axis=1)

        for i, (_, row) in enumerate(test_df.iterrows()):
            out[row["filename"]] = (float(p_single[i]), float(p_tta[i]))

    return out


def analyze_scores(name, scores, is_invalid):
    fpr, tpr, roc_thresh = roc_curve(is_invalid, scores)
    roc_auc = roc_auc_score(is_invalid, scores)
    precision, recall, pr_thresh = precision_recall_curve(is_invalid, scores)
    ap = average_precision_score(is_invalid, scores)

    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    youden_threshold = float(roc_thresh[best_j_idx]) if len(roc_thresh) else 0.5

    f1 = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12), 0,
    )
    best_f1_idx = int(np.argmax(f1)) if len(f1) else 0
    f1_threshold = float(pr_thresh[best_f1_idx]) if len(pr_thresh) else 0.5

    frac_pos, mean_pred = calibration_curve(is_invalid, scores, n_bins=5, strategy="uniform")

    print(f"  {name:28s} ROC AUC={roc_auc:.3f}  PR AP={ap:.3f}  "
          f"Youden thr={youden_threshold:.3f}  F1 thr={f1_threshold:.3f}")

    return {
        "roc_auc": float(roc_auc), "pr_ap": float(ap),
        "youden_threshold": youden_threshold, "f1_threshold": f1_threshold,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "calibration_curve": {"mean_predicted": mean_pred.tolist(), "frac_positive": frac_pos.tolist()},
    }


def plot_overlays(results, out_dir):
    names = list(results.keys())
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], color=cmap(i),
                label=f"{name} (AUC={r['roc_auc']:.3f})", linewidth=1.6)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — ensemble / TTA / calibration, vs. each scheme alone")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "roc_boost.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["pr_curve"]["recall"], r["pr_curve"]["precision"], color=cmap(i),
                label=f"{name} (AP={r['pr_ap']:.3f})", linewidth=1.6)
    ax.set_xlabel("Recall (invalid)"); ax.set_ylabel("Precision (invalid)")
    ax.set_title("Precision-Recall — ensemble / TTA / calibration")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "pr_boost.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfectly calibrated")
    for i, name in enumerate(names):
        r = results[name]
        ax.plot(r["calibration_curve"]["mean_predicted"], r["calibration_curve"]["frac_positive"],
                marker="o", markersize=4, color=cmap(i), label=name, linewidth=1.3)
    ax.set_xlabel("Mean predicted P(invalid)"); ax.set_ylabel("Observed fraction invalid")
    ax.set_title("Calibration — ensemble / TTA / calibration")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "calibration_boost.png"), dpi=130); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 epoch, 2 TTA views, for a pipeline smoke test only.")
    args = parser.parse_args()

    random.seed(ta.SEED)
    np.random.seed(ta.SEED)
    torch.manual_seed(ta.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    epochs = 1 if args.quick else ta.FIXED_CONFIG["epochs"]
    n_tta_views = 2 if args.quick else 6

    df = pd.read_csv(ta.SPLIT_CSV, dtype={
        "fold_lite": str, "fold_extended": str, "fold_noempty": str, "fold_noempty4": str,
    })
    # Shared ground truth for everything below: the no-empty definition of
    # "invalid" (empty folded in) -- the framing this project's findings favor.
    gt = df.set_index("filename")["noempty_split_label"].str.startswith("invalid").astype(int)

    print("\n=== Training extended CV fold models (single-view + TTA) ===")
    ext_preds = train_fold_models(
        df, "extended_label", ta.extended_path_fn, ["invalid_day", "invalid_night"],
        "fold_extended", epochs, n_tta_views, device,
    )
    print("\n=== Training no-empty split CV fold models (single-view + TTA) ===")
    split_preds = train_fold_models(
        df, "noempty_split_label", ta.noempty_path_fn, ["invalid_day", "invalid_night"],
        "fold_noempty4", epochs, n_tta_views, device,
    )

    # Every filename is present in both dicts (both schemes cover all 176).
    filenames = sorted(set(ext_preds) & set(split_preds))
    assert len(filenames) == len(df), (
        f"expected predictions for all {len(df)} images, got {len(filenames)}"
    )

    p_ext = np.array([ext_preds[f][0] for f in filenames])
    p_ext_tta = np.array([ext_preds[f][1] for f in filenames])
    p_split = np.array([split_preds[f][0] for f in filenames])
    p_split_tta = np.array([split_preds[f][1] for f in filenames])
    is_invalid = gt.loc[filenames].values

    p_ensemble = (p_ext + p_split) / 2
    p_ensemble_tta = (p_ext_tta + p_split_tta) / 2

    # Platt scaling on the ensemble+TTA score. NOTE: fit and evaluated on
    # the same pooled set (176 images is too little for a further clean
    # holdout) -- an optimistic, in-sample calibration check, not a fully
    # independent one; said plainly in the writeup, not hidden.
    platt = LogisticRegression()
    platt.fit(p_ensemble_tta.reshape(-1, 1), is_invalid)
    p_ensemble_tta_calibrated = platt.predict_proba(p_ensemble_tta.reshape(-1, 1))[:, 1]

    print("\n=== Results: extended alone -> split alone -> ensemble -> +TTA -> +calibration ===")
    results = {}
    results["extended_alone"] = analyze_scores("extended_alone", p_ext, is_invalid)
    results["split_alone"] = analyze_scores("split_alone", p_split, is_invalid)
    results["ensemble"] = analyze_scores("ensemble", p_ensemble, is_invalid)
    results["ensemble_tta"] = analyze_scores("ensemble_tta", p_ensemble_tta, is_invalid)
    results["ensemble_tta_calibrated"] = analyze_scores(
        "ensemble_tta_calibrated", p_ensemble_tta_calibrated, is_invalid,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.quick:
        plot_overlays(results, OUT_DIR)
        print(f"Saved roc_boost.png, pr_boost.png, calibration_boost.png to {OUT_DIR}")

    with open(os.path.join(OUT_DIR, "boost_results.json"), "w") as f:
        json.dump({
            "quick": args.quick,
            "n_tta_views": n_tta_views,
            "n_images": len(filenames),
            "platt_coef": platt.coef_.tolist(),
            "platt_intercept": platt.intercept_.tolist(),
            "results": results,
        }, f, indent=2)
    print(f"Saved {os.path.join(OUT_DIR, 'boost_results.json')}")


if __name__ == "__main__":
    main()
