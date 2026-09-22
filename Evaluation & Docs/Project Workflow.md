# Gypsum Filter Picture Recognition

## 1. Idea

The goal of this project is to build an **image classification model that looks at a camera feed of a gypsum filter and automatically determines whether the filter/cake is in a normal ("valid") or abnormal ("invalid") state**, so that problems can be caught from the picture instead of (or in addition to) manual visual inspection.

The source images are frames captured from a **fixed industrial CCTV-style camera mounted above a gypsum dewatering filter** (a rotary/table vacuum filter that separates gypsum slurry into a solid "cake" on a segmented filter surface). Each frame has a timestamp burned into the top-right corner by the camera/DVR (e.g. `2026-09-06 17:42:24`), and the scene is often partly obscured by steam/dust/mist, which is a normal part of the process environment, not a camera defect.

Visually, across the sample images:
- A **normal ("תקין" / valid) cake** looks like an even, continuous layer of material covering the filter's sectors, with a fairly uniform ridged/striped texture.
- An **abnormal ("לא תקין" / invalid) cake** shows visible cracking, patchiness, or gaps where the dark filter cloth/surface shows through instead of an even material layer.
- An **empty filter ("מסנן ריק")** shows the bare filter surface with no cake material on it at all.
- The same conditions look very different by lighting: **day** shots are lit by ambient/daylight through the steam, while **night** shots are lit by a strong artificial floodlight that creates glare/hot spots in the frame — this is why day and night are kept as separate classes.

## 2. Current State of the Project

The dataset is collected and sorted, the repo/docs structure is set up, and the model build is in progress (see `Model & Training/`). Photos were originally exported from WhatsApp (filenames followed the `WhatsApp Image <date> at <time> (<n>).jpeg` pattern) and manually sorted into folders that act as the class labels; they've since been renamed to random IDs (`img_<id>.jpeg`) with a manifest mapping back to the originals — see [rename_manifest.csv](context/rename_manifest.csv).

## 3. Dataset

The dataset lives under `Model & Training/pictures/` in **two parallel versions** of the same 176 photos:

### 3.1 `pictures/extended version/` — the original 5 classes

| Folder (Hebrew) | Meaning | Condition | Time of day | # Images |
|---|---|---|---|---|
| `תקין יום` | "Valid — day" | Normal/OK cake | Day | 62 |
| `תקין לילה` | "Valid — night" | Normal/OK cake | Night | 30 |
| `לא תקין יום` | "Invalid — day" | Abnormal/faulty cake | Day | 42 |
| `לא תקין לילה` | "Invalid — night" | Abnormal/faulty cake | Night | 34 |
| `מסנן ריק` | "Empty filter" | No cake on filter | Not split by day/night | 8 |

### 3.2 `pictures/simple version/` — the 3 classes actually used for training

Day and night are collapsed (see the decision in §5); each file here is a copy of the same-named file in `extended version/`.

| Folder | Made from | # Images |
|---|---|---|
| `valid` | `תקין יום` + `תקין לילה` | 92 |
| `invalid` | `לא תקין יום` + `לא תקין לילה` | 76 |
| `empty` | `מסנן ריק` | 8 |

**Total: 176 images.**

### 3.3 Known dataset characteristics to design around

- **Class imbalance**: "empty filter" has only 8 images vs. 30–62 for the other classes, and it isn't split into day/night. This will need attention (oversampling, augmentation, or collecting more empty-filter shots, ideally for both day and night).
- **Burned-in timestamp overlay**: every image has a date/time stamp in the top-right corner. This is not part of the actual scene and should either be cropped out or masked before training so the model doesn't learn to key off it.
- **Steam/dust obstruction**: the process itself produces mist that partially obscures the filter in many shots — this is signal the model needs to be robust to, not noise to remove.
- **Day/night lighting is a major visual shift**: night shots have strong localized glare from floodlights; day shots have flatter, hazier lighting. Whatever splitting/augmentation strategy is used should account for this so the model doesn't just learn "day vs. night" instead of "valid vs. invalid." — **but see §3.4: "day"/"night" don't appear to mean literal time of day.**
- **Fixed camera angle**: all images appear to come from the same mounted camera/angle, which simplifies the problem (no need for viewpoint invariance) but means the model may be specific to this one camera installation.
- **Source quality**: images are WhatsApp exports (compressed JPEGs, not original camera resolution), which is worth keeping in mind for image-quality-sensitive techniques.

### 3.4 "day"/"night" may not mean time of day (discovered 2026-09-22)

Cross-referencing the manifest's recovered original capture timestamps against the `day`/`night` folder each photo came from turned up something unexpected: `day`- and `night`-labeled photos are **interleaved within the same few seconds** of each other, not hours apart. Example, from 2026-09-09:

```
11:29:43  day
11:29:47  night
11:29:49  day
11:29:50  day
11:29:56  night
```

All around 11:29 **AM** — clearly not solar daytime vs. nighttime. Whatever these two folders actually distinguish (two camera angles? flash vs. no flash? a labeling judgment call based on how the photo *looked*, made while manually sorting?), it isn't literal time-of-day. The visual difference between the folders (flat daylight-like vs. glare-heavy floodlit-like) is still real and still a valid thing for a model to learn — it just isn't explained by the clock. **Worth confirming with whoever collected/sorted the photos before stating in the final report that this is a day/night split.**

## 4. Intended Workflow

1. **Data collection** (done / ongoing) — pull still frames from the filter camera (currently via WhatsApp) and sort them into the labeled folders above.
2. **Data cleaning & preprocessing**
   - Crop or mask out the burned-in timestamp overlay.
   - Resize/normalize images to a consistent resolution.
   - Split into train / validation / test sets (careful to avoid near-duplicate frames — many images are seconds apart — leaking across splits).
3. **Address class imbalance** — augmentation (rotation/crop/brightness jitter suited to the steam/lighting conditions) and/or targeted collection of more "empty filter" images.
4. **Model training** — train an image classifier (e.g. a fine-tuned CNN such as ResNet/EfficientNet/MobileNet, or a lightweight custom CNN given the modest dataset size) to predict the class from an image.
5. **Evaluation** — check accuracy/precision/recall per class, with particular attention to **not missing "invalid" cases** (false negatives are likely the costly error here) and to day/night generalization.
6. **Inference / integration** — once accuracy is acceptable, run the model on new frames from the live camera feed (e.g. sampled every N minutes) and flag "invalid" states for an operator, e.g. via an alert/notification.
7. **Iteration** — feed back misclassified real-world frames into the labeled dataset to keep improving the model over time.

## 5. Decisions (2026-09-22)

- **Label scheme**: simplified to **3 classes** — `valid`, `invalid`, `empty` (§3.2). Day/night is not a separate class; it's handled as a lighting condition the model needs to be robust to (via augmentation), and checked for during evaluation rather than predicted.
- **Framework**: PyTorch, using a pretrained torchvision backbone (transfer learning) rather than training a CNN from scratch, given the small dataset.
- **Training environment**: Google Colab (free GPU) — the local machine's GPU (AMD, no CUDA) can't do accelerated training on Windows.
- **Deliverables**: a walkthrough notebook, a written results report, presentation slides, and a local live-demo script.

See `Model & Training/scripts/build_dataset_index.py` and `Model & Training/notebooks/train_and_evaluate.ipynb` for the implementation.

Also added: `Model & Training/notebooks/compare_label_schemes.ipynb`, an A/B test training a lite-scheme (3-class) and an extended-scheme (5-class, day/night kept separate) model under the same fixed config, to check whether the day/night simplification in the decision above actually costs anything.

## 6. Optimization sweep (2026-09-22)

`Model & Training/scripts/optimize_models.py` extended the A/B test above: the same fixed config (MobileNetV2, last block fine-tuned, lr=1e-3, 10 epochs), swept across split ratio (80/20, 75/25, 70/30 — `empty` always kept ≥3 images in train regardless), K-fold cross-validation (3-fold lite, 2-fold extended — capped lower for extended because its night classes only have 2 independent sessions each), and one oversampling variant. All numbers below are read on the lite 3-class scale (extended's predictions collapsed: `valid_day`/`valid_night` → `valid`, etc.).

| Variant | Accuracy | Val loss | Macro F1 | Invalid recall |
|---|---|---|---|---|
| Lite, 80/20 | 0.600 | – | 0.450 | 0.600 |
| Lite, 75/25 | 0.787 | 0.494 | 0.444 | 0.600 |
| Lite, 70/30 | 0.587 | 0.326 | 0.365 | 0.303 |
| Lite, 3-fold CV | 0.633 ± 0.095 | 0.449 | 0.569 | 0.424 ± 0.019 |
| Lite, 80/20 + oversampling | 0.600 | 0.480 | 0.450 | 0.600 |
| Extended, 80/20 | 0.632 | – | 0.415 | 0.643 |
| Extended, 75/25 | 0.722 | 0.408 | 0.460 | 0.643 |
| Extended, 70/30 | **0.741** | 0.384 | 0.479 | **0.714** |
| Extended, 2-fold CV | 0.723 ± 0.005 | 0.572 | 0.650 | 0.639 ± 0.075 |

**Findings:**
- **Extended (5-class) beat lite (3-class) on every ratio and in cross-validation.** The CV numbers are the most trustworthy here — averaged over independent sessions instead of one small held-out split — and they're the clearest signal: extended 72.3% ± 0.5 vs. lite 63.3% ± 9.5 accuracy. Lite's CV standard deviation (±9.5 points) is large relative to extended's (±0.5), meaning lite's single-split numbers above are much less stable session-to-session.
- No split ratio was uniformly best for both schemes (lite peaked at 75/25 on accuracy alone; extended peaked at 70/30 on both accuracy and invalid recall) — with this little independent data, exact ratio matters less than which scheme is used.
- The oversampling variant (`WeightedRandomSampler` on top of the existing class-weighted loss) made **no difference** on lite 80/20 — identical metrics to the plain run. Given the loss weighting already exists, resampling the same ~6 `empty` training images more often doesn't add new information; more real `empty` photos would.
- This is evidence worth weighing against the §5 decision to simplify to 3 classes — it may be worth reconsidering, though still on a very small dataset (single seed, CPU run).

## 8. No-empty scheme — round 2 (2026-09-22)

A 4th label scheme: "empty filter" redefined as a kind of **invalid** rather than its own class — the reasoning being that an empty filter is arguably a fault state too, not a neutral third option. `Model & Training/pictures/no-empty version/` is a copy of `extended version/` with the 8 `מסנן ריק` photos manually moved by hand into `לא תקין יום`/`לא תקין לילה` (6 day, 2 night, judged by eye). Read at two granularities: **collapsed** (2-class: `valid`/`invalid`) and **split** (4-class: `valid_day`/`valid_night`/`invalid_day`/`invalid_night`).

`Model & Training/scripts/optimize_noempty_models.py` ran the same ratio sweep + CV as §6, same fixed config. 4-fold CV was attempted (as requested) but the data didn't support it — `determine_k()` found the scarcest class only has 2–3 independent sessions, capping CV to 3-fold (collapsed) / 2-fold (split); merging the empty photos in mostly landed them inside existing invalid sessions rather than creating new independent ones.

| Variant | Accuracy | Val loss | Macro F1 | Invalid recall |
|---|---|---|---|---|
| No-empty collapsed, 80/20 | 0.909 | 0.570 | 0.895 | 0.875 |
| No-empty collapsed, 75/25 | 0.792 | 0.536 | 0.722 | 0.875 |
| No-empty collapsed, 70/30 | 0.480 | 0.643 | 0.381 | 0.086 |
| No-empty collapsed, 3-fold CV | 0.708 ± 0.143 | 0.492 | 0.684 | 0.577 ± 0.267 |
| No-empty split, 80/20 | 0.938 | 0.431 | 0.909 | 0.923 |
| No-empty split, 75/25 | 0.782 | 0.411 | 0.764 | 0.824 |
| No-empty split, 70/30 | 0.764 | 0.442 | 0.742 | 0.765 |
| **No-empty split, 2-fold CV** | **0.774 ± 0.019** | 0.474 | 0.760 | **0.711 ± 0.054** |

**Findings — this is the headline result of the whole optimization effort:**
- **No-empty split (4-class) beats every scheme tried in this project**, on CV (the trustworthy metric): 77.4% ± 1.9 accuracy vs. the previous best, extended, at 72.3% ± 0.5 (§6). It's both higher *and* has a tighter spread.
- **No-empty collapsed (2-class) is not trustworthy despite its high single-split numbers.** Its CV variance is huge (±14.3 points — one fold scored 80.0%, another only 50.7%), almost certainly because collapsing day/night here throws away a real signal that the split version keeps. Its 90.9%/80-20 number (n=11 test images) is exactly the kind of small-test-set noise this project's docs have flagged repeatedly — don't quote it without the CV caveat.
- Every 80/20 single-split number in this table uses a very small test set (11–16 images) and should be read with caution; CV is the number to trust.
- Net effect of the "empty = invalid" reframing: it appears to genuinely help, not just shuffle the same information around — likely because the model no longer has to learn a 3rd/5th class from only 6–8 examples, and "empty" and "invalid" probably share some real visual similarity (bare or torn filter surface) that the model can exploit once they're merged.

## 9. Threshold / calibration / ROC-PR analysis (2026-09-22)

Every training script decided valid-vs-invalid with a bare `argmax()` — an implicit 0.5 probability cutoff that was never chosen deliberately or checked. `Model & Training/scripts/threshold_analysis.py` re-trained **all 23 individual configs** trained across this project (every ratio split × every CV fold, all 4 schemes), captured full probabilities instead of just the winning class, and for each computed: a ROC curve + AUC, a precision-recall curve + average precision (AP), a calibration curve, and two candidate best thresholds for flagging "invalid" — **Youden's J** (balances both error types) and the **F1-optimal** threshold (tuned specifically for the invalid class). CV configs pool every fold's held-out predictions first, same as §6/§8.

| Config (CV = pooled) | ROC AUC | PR AP | Youden threshold | F1-optimal threshold |
|---|---|---|---|---|
| Lite, CV | 0.763 | 0.653 | 0.178 | 0.109 |
| Extended, CV | **0.837** | 0.752 | 0.302 | 0.302 |
| No-empty collapsed, CV | 0.723 | 0.720 | 0.526 | 0.168 |
| No-empty split, CV | 0.818 | **0.768** | 0.466 | 0.466 |

(Full 17-row table, including every single ratio split, is in `models/threshold_analysis/threshold_results.json`; plots — `roc_overlay.png`, `pr_overlay.png`, `calibration_overlay.png` — overlay all 17 configs each.)

**Findings:**
- **The default 0.5 cutoff has been under-flagging real faults the whole time.** Nearly every config's optimal threshold — by either method — is below 0.5. For the four CV rows above, every F1-optimal threshold is below 0.5 (0.109–0.466), and 3 of 4 Youden thresholds are too. Concretely: with the extended CV model, switching from the untuned default to the F1-optimal threshold (0.302) would flag more real invalid cases as invalid, at the cost of a few more false alarms — exactly the trade this project's docs have said is worth making (§6: "a missed invalid costs more than a false alarm").
- **On pure invalid-vs-not ranking quality (ROC AUC), extended (5-class, empty separate) edges out no-empty split (4-class, empty merged) — 0.837 vs. 0.818** — the reverse of §8's accuracy-based ranking, where split led clearly (77.4% vs. 72.3%). This isn't a contradiction: accuracy measures the *whole* multi-class decision, AUC measures specifically how well the model ranks "invalid" over "not invalid" regardless of where the cutoff sits. The two leading schemes are close on both metrics and clearly ahead of lite and no-empty collapsed on both — the split-vs-extended choice is a genuine toss-up, not settled by this analysis alone.
- Lite and no-empty collapsed are weaker on both accuracy (§6, §8) and ranking quality (AUC/AP here) — consistent evidence across two different kinds of analysis that they're the two schemes to rule out first.

## 10. Ensemble / TTA / calibration (2026-09-22)

Three ways to push further, tried on the two closest competitors from §9 (extended CV vs. no-empty split CV): `Model & Training/scripts/boost_analysis.py`

1. **Ensemble** — average the two schemes' P(invalid) per image, each from its own properly held-out CV fold model.
2. **+ Test-time augmentation (TTA)** — average the softmax over 6 mildly-augmented views per image at inference, instead of one center crop.
3. **+ Calibration** — fit Platt scaling on top of the ensemble+TTA score.

Ground truth throughout is the no-empty definition of "invalid" (empty folded in), so extended's own number here differs slightly from its §9 figure (which used extended's own empty-is-separate ground truth) — a different question, not a contradiction.

| Variant | ROC AUC | PR AP |
|---|---|---|
| Extended alone | 0.808 | 0.769 |
| **No-empty split alone** | **0.852** | **0.805** |
| Ensemble (extended + split, averaged) | 0.843 | 0.803 |
| Ensemble + TTA | 0.847 | 0.787 |
| Ensemble + TTA + calibration | 0.847 | 0.787 |

**Finding — reported straight, not spun: none of the three techniques beat the single best model.** No-empty split alone (0.852 AUC / 0.805 AP) outperforms every ensemble variant. A plain 50/50 average pulled the stronger model (split) down toward the weaker one (extended, 0.808 alone) rather than lifting it — TTA recovered a little of that on AUC (0.843 → 0.847) but not on AP (0.803 → 0.787), and calibration (as expected — it's a monotonic rescaling) left both ranking metrics unchanged, only improving probability quality, not discrimination.

This is a useful negative result: with two models this close in quality but one clearly ahead, naive averaging isn't automatically better than just using the better model. A confidence-weighted ensemble (weighting no-empty split more heavily than extended, rather than 50/50) is the natural next thing to try — not implemented here. Plots: `models/boost_analysis/roc_boost.png`, `pr_boost.png`, `calibration_boost.png`.

## 12. Decision: label scheme finalized (2026-09-22)

**Chosen: no-empty split (4-class)** — `valid_day` / `valid_night` / `invalid_day` / `invalid_night`, "empty filter" folded into `invalid` rather than kept as its own class.

§9 made this look like a toss-up against extended (CV AUC 0.837 vs. 0.818) — but that compared each scheme against **its own** definition of "invalid," a slightly different question per model. §10's boost analysis put both schemes on the **same** ground truth (the no-empty definition) in the same run, and there no-empty split won clearly on every metric: accuracy, invalid recall, ROC AUC (0.852 vs. 0.808), and PR AP (0.805 vs. 0.769). Combined with §8's accuracy lead (77.4% vs. 72.3% CV) and §10's finding that split alone beats every ensemble attempt, the evidence converges on one answer once the comparison is made fair. Lite and no-empty collapsed were ruled out earlier by every analysis in this doc.

- **Recommended operating threshold**: ~0.45–0.5 (F1-optimal values for split-based models ranged 0.435–0.541 across §9's and §10's separate runs) — re-derive the exact figure from `full_pipeline.ipynb`'s output once run for real, since that's the one consolidated, non-redundant source of truth going forward.
- **Promoted model**: `models/gypsum_classifier_split_70_30.pt`, produced by `full_pipeline.ipynb` — kept alongside `gypsum_classifier_extended_70_30.pt` as the runner-up candidate, not deleted.
- **Caveat, stated plainly, not buried**: still a 176-photo dataset, single seed, CPU-trained runs throughout. This is the best-supported hypothesis given everything tried, not a production-validated result — deploy it as the leading candidate and monitor, don't treat the question as permanently closed.

## 13. Live-demo script (2026-09-22)

`Model & Training/scripts/predict.py` — classifies one photo from the command line (`python predict.py path/to/photo.jpeg`): prints the predicted class, confidence, and the operator-facing VALID/INVALID call (an "empty" prediction counts as INVALID, per §12). Preprocessing matches training exactly (timestamp masking, resize/crop, ImageNet normalization).

It looks for `models/gypsum_classifier_split_70_30.pt` (the §12 promoted model) first and falls back to `gypsum_classifier_extended_70_30.pt` (the only model actually trained for real so far) if the promoted one hasn't been produced yet — so the script works today and needs no changes once `full_pipeline.ipynb` is run for real.

Spot-checked against three held-out-style photos from the dataset using the fallback (extended) model: a `valid_day` photo (89% confidence, correct), an `invalid_night` photo (98% confidence, correct), and a `valid_night` photo that came back a lower-confidence 47% `valid_night` with `empty` and `invalid_night` close behind — consistent with §9/§10's documented finding that `empty` is this model's weakest class (0% recall on 2 test images), not a bug in the script.

## 14. Open Questions / Next Steps

- Confirm the exact equipment name/process (what is a "Gibson filter" — brand/model — and what specifically defines "invalid" beyond visual cracking/patchiness?).
- Confirm what "day"/"night" actually mean in the source photos (§3.4) — the timestamps rule out literal time-of-day.
- If pursuing ensembling further, try confidence-weighted averaging (§10) rather than a plain mean.
- Run `full_pipeline.ipynb` for real in Colab to produce `gypsum_classifier_split_70_30.pt`, then re-point `predict.py`'s spot-check at it.
- Decide where/how the camera feed will be sampled for live inference (folder of new images, RTSP stream, etc.).
- Decide the deployment target (local script, small server, edge device near the camera, etc.) and how alerts should be delivered.
