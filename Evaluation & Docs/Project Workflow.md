# Gypsum Filter Picture Recognition

## 1. Idea

The goal of this project is to build an **image classification model that looks at a camera feed of a gypsum filter and automatically determines whether the filter/cake is in a normal ("valid") or abnormal ("invalid") state**, so that problems can be caught from the picture instead of (or in addition to) manual visual inspection.

The source images are frames captured from a **fixed industrial CCTV-style camera mounted above a gypsum belt filter** — a continuous belt vacuum filter that washes and dewaters the gypsum by-product from phosphoric acid production, separating it into a solid "cake" on the belt surface (see §1.2 for the full process context). Each frame has a timestamp burned into the top-right corner by the camera/DVR (e.g. `2026-09-06 17:42:24`), and the scene is often partly obscured by steam/dust/mist, which is a normal part of the process environment, not a camera defect.

Visually, across the sample images:
- A **normal ("תקין" / valid) cake** looks like an even, continuous layer of material covering the filter's sectors, with a fairly uniform ridged/striped texture.
- An **abnormal ("לא תקין" / invalid) cake** shows visible cracking, patchiness, or gaps where the dark filter cloth/surface shows through instead of an even material layer.
- An **empty filter ("מסנן ריק")** shows the bare filter surface with no cake material on it at all.
- The same conditions look very different by lighting: **day** shots are lit by ambient/daylight through the steam, while **night** shots are lit by a strong artificial floodlight that creates glare/hot spots in the frame — this is why day and night are kept as separate classes.

### 1.1 Related work / similar cases (2026-09-22)

A short survey of existing approaches to similar problems, done to check this project's plan against established practice rather than reinventing it from scratch:

- **Transfer learning from ImageNet-pretrained CNNs is the standard approach for small industrial visual-inspection datasets.** A 2024 systematic review of CNN-based surface-defect detection found transfer learning used in 83% of studies surveyed, specifically because training a CNN from scratch needs a dataset far larger than most factories can realistically label — pretrained backbones (VGG16, ResNet, DenseNet, MobileNet-family) cut both the data and compute needed while resisting overfitting ([A Systematic Review on Deep Learning with CNNs Applied to Surface Defect Detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC10607335/)). This directly matches this project's choice (§5): MobileNetV2, transfer-learned rather than trained from scratch, on a 176-image dataset that would never support training a CNN from zero.
- **Class imbalance between "normal" and "defect" is treated as the norm, not the exception, in industrial defect detection** — because in a working process, defects genuinely are rarer than normal output. One cited industrial dataset example splits 84% nominal / 12% / 4% across defect types ([Tackling class imbalance in computer vision: a contemporary review](https://link.springer.com/article/10.1007/s10462-023-10557-6)). This project's "empty filter" class (8 of 176 images, 4.5%) sits in the same range, and the standard mitigations the literature points to — cost-sensitive/class-weighted loss and resampling — are exactly what this project tried (§4 step 3, and the `WeightedRandomSampler` variant in §6), including the honest negative finding that resampling added nothing once weighted loss was already in place, which matches the literature's framing of these as alternative, not strictly additive, techniques.
- **Reported accuracy in comparable published work is high (often 95%+) but on datasets one to three orders of magnitude larger than this one** — e.g. a transfer-learning assembly-defect inspection system reaching 98.67% accuracy ([A Deep Transfer Learning-Based Visual Inspection System for Assembly Defects](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11945499/)) was trained on thousands of labeled images per station, not 176 total across 4–5 classes. This project's own numbers (§8: ~77% CV accuracy on the leading scheme) should be read against that gap, not against the field's best-case numbers — the caveat already stated plainly in §12 ("still a 176-photo dataset... not a production-validated result") is consistent with what the literature would predict for a dataset this size, not a sign something went wrong.

No existing published work was found on this exact equipment (gypsum belt filter cake inspection specifically) — the closest matches are the general "surface/assembly defect via transfer-learned CNN" literature above, which is why this project's approach follows that pattern rather than a filter-specific one.

### 1.2 Background and motivation (2026-09-22)

This project comes directly from a real process at the author's plant, a **wet-process phosphoric acid** producer: phosphate rock reacts with sulfuric acid to produce phosphoric acid, with **gypsum as a by-product**. The gypsum still carries residual phosphoric acid, so it's washed with water and then **dewatered on the belt filter** this project's camera watches, to recover as much of that phosphoric acid as economically possible before the gypsum is discarded.

**"Invalid" means wet gypsum** — the cake didn't dewater properly. This has two separate costs:
1. **Yield loss** — phosphoric acid trapped in gypsum that's still wet doesn't get recovered, and is lost with the discarded gypsum. **Every 1% of phosphoric acid remaining in the gypsum costs roughly $2,000/hour.** On an annual basis this typically runs to a loss equivalent to about 3% of throughput, with bad-filtration episodes accounting for roughly 6% of downtime.
2. **Equipment damage / downtime** — wet gypsum can damage downstream equipment, adding a further ~3% of annual downtime on top of the yield-loss figure above.

**The business case for this project**: filtration problems are currently caught by manual visual inspection, which is inherently intermittent (an operator can't watch the feed continuously) and inconsistent (judgment varies between operators and shifts). **Automating "valid/invalid" recognition from the existing camera feed, instead of relying on manual checks, is estimated to recover 10–15% of these losses** — by catching bad filtration sooner and more consistently than a person checking periodically can. That gap between "loss happens" and "loss is noticed" is exactly what this project's image classifier is meant to close.

**Monetized, with assumptions stated explicitly** (so they can be corrected with real plant figures):

*Yield loss:*
- $2,000/hour per 1% P2O5, at a typical ~3% equivalent → **~$6,000/hour** effective yield-loss rate during a bad-filtration episode.
- At ~6% of operating hours affected and an assumed **~8,000 operating hours/year** (~91% uptime — replace with the plant's real annual operating hours for an exact figure): ~480 affected hours/year → **~$2,880,000/year in yield-loss exposure**.

*Downtime (equipment damage):*
- Lost-production rate during downtime: **$49,500/hour** (45 × 1,100, per the plant engineer).
- ~3% additional annual downtime (≈240 hours/year, on the same 8,000 hr/year assumption) → **~$11,880,000/year in downtime exposure**.

*Combined:*
- **Total annual exposure: ~$14,760,000/year** (yield loss + downtime).
- **Automated detection recovering 10–15% of that → roughly $1,476,000–$2,214,000/year recovered.**

This also resolves two of this doc's earlier open questions (§15): the equipment is a **gypsum belt filter** (not a "Gibson filter" — an earlier mishearing/typo), and "invalid" specifically means **wet gypsum**, not just visual cracking/patchiness for its own sake — the visual cracking/patchiness *is* how wet, poorly-dewatered gypsum looks on camera, which is why the visual classification task is a meaningful proxy for the real problem (yield loss + downtime), not just a cosmetic check.

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

### 3.5 Exploratory data analysis (2026-09-22)

`Model & Training/scripts/eda.py` — a standalone look at the raw dataset itself (not model outputs), run locally (no GPU needed). Saves to `Model & Training/eda/`:

- `class_distribution.png` — bar chart confirming the counts in §3.1 (62/30/42/34/8).
- `sample_grid.png` — 4 random examples per class, so the classes can actually be seen rather than just described. Confirms the burned-in timestamp (§3.3), the night floodlight glare (§1) — and shows that glare isn't exclusive to "night": at least one `valid_day` sample in the grid has a bright glare spot too, visually reinforcing §3.4's finding that day/night isn't a clean lighting split.
- `brightness_by_class.png` / `brightness_day_vs_night.png` — mean per-image pixel brightness (grayscale), compared across classes and pooled day-vs-night. **Finding: day and night are barely different on this measure** — 123.6 (day, n=104) vs. 121.0 (night, n=64) mean brightness, heavily overlapping distributions. This is a second, independent line of evidence (alongside §3.4's timestamp check) that whatever "day"/"night" actually encodes, it isn't a simple global brightness difference — the floodlight's effect is a *local* hot spot, not a shift in the whole frame's average tone, which is consistent with the sample grid but means brightness alone can't be used to auto-verify the labels.
- Image size check: **no two images in the dataset share identical pixel dimensions** (176 distinct sizes, all within a narrow band around ~1150–1185 × 825–875). Confirms the "WhatsApp export" theory in §3.3 — each photo was re-encoded slightly differently on its way through WhatsApp — and confirms the `Resize((240, 240))` step in every training pipeline is doing necessary work, not just following convention.

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

**Beyond the numbers, there's a stronger operational reason to fold "empty" into "invalid" rather than keep it as a 3rd/5th class**, per the plant engineer's own reasoning (2026-09-23):
1. The "empty" dataset is tiny (8 of 176 images) — the statistical reason already covered above.
2. **A live model doesn't need to detect "empty" at all** — the operator empties the filter themselves, so they already know it's empty; the camera has no information advantage there. Asking the model to spend capacity distinguishing "empty" from "invalid" is solving a problem the operator doesn't have.
3. The natural production design follows from this: **the model gets toggled off when the operator empties the filter, and toggled back on once filtration resumes** — "empty" is an operator-controlled state to route around, not a class the vision model needs to recognize on its own.

- **Recommended operating threshold**: ~0.45–0.5 (F1-optimal values for split-based models ranged 0.435–0.541 across §9's and §10's separate runs) — re-derive the exact figure from `full_pipeline.ipynb`'s output once run for real, since that's the one consolidated, non-redundant source of truth going forward.
- **Promoted model**: `models/gypsum_classifier_split_70_30.pt`, produced by `full_pipeline.ipynb` — kept alongside `gypsum_classifier_extended_70_30.pt` as the runner-up candidate, not deleted.
- **Caveat, stated plainly, not buried**: still a 176-photo dataset, single seed, CPU-trained runs throughout. This is the best-supported hypothesis given everything tried, not a production-validated result — deploy it as the leading candidate and monitor, don't treat the question as permanently closed.

## 13. Live-demo script (2026-09-22)

`Model & Training/scripts/predict.py` — classifies one photo from the command line (`python predict.py path/to/photo.jpeg`): prints the predicted class, confidence, and the operator-facing VALID/INVALID call (an "empty" prediction counts as INVALID, per §12). Preprocessing matches training exactly (timestamp masking, resize/crop, ImageNet normalization).

It looks for `models/gypsum_classifier_split_70_30.pt` (the §12 promoted model) first and falls back to `gypsum_classifier_extended_70_30.pt` (the only model actually trained for real so far) if the promoted one hasn't been produced yet — so the script works today and needs no changes once `full_pipeline.ipynb` is run for real.

Spot-checked against three held-out-style photos from the dataset using the fallback (extended) model: a `valid_day` photo (89% confidence, correct), an `invalid_night` photo (98% confidence, correct), and a `valid_night` photo that came back a lower-confidence 47% `valid_night` with `empty` and `invalid_night` close behind — consistent with §9/§10's documented finding that `empty` is this model's weakest class (0% recall on 2 test images), not a bug in the script.

## 14. Simulated live feed (2026-09-22)

`Model & Training/scripts/watch_camera_feed.py` — watches `pictures/incoming/` (or a folder passed as an argument), classifies every new photo dropped into it within a few seconds, and prints an `<<< ALERT: INVALID` line whenever the call comes back invalid. Reuses `predict.py`'s model loading and preprocessing directly (imported, not duplicated), so it always classifies exactly the way `predict.py` would.

This is a **simulation**, not a real camera integration — a real feed would need an actual source (RTSP stream, the camera's own image export, etc.) and a real alerting channel (SMS, dashboard, log file), neither of which is decided yet (see §15). Tested locally: started the watcher, dropped in a `valid_day` and an `invalid_night` sample photo, both were picked up and classified correctly (the second correctly triggered the alert line) within the poll interval.

## 15. Open Questions / Next Steps

- ~~Confirm the exact equipment name/process and what "invalid" means beyond visual cracking/patchiness~~ — answered, see §1.2: it's a gypsum belt filter, and "invalid" means wet gypsum (yield loss + equipment damage risk).
- Confirm what "day"/"night" actually mean in the source photos (§3.4) — the timestamps rule out literal time-of-day, and §3.5's EDA rules out a simple whole-image-brightness explanation too. Still unresolved — §16's full-scale-pilot plan proposes engineering the ambiguity away with fixed lighting, rather than continuing to try to explain it post-hoc.
- If pursuing ensembling further, try confidence-weighted averaging (§10) rather than a plain mean.
- Run `full_pipeline.ipynb` for real in Colab to produce `gypsum_classifier_split_70_30.pt`, then re-point `predict.py`'s spot-check at it.
- Decide where the real camera feed comes from (RTSP stream, a folder the camera itself writes to, etc.) — §14's watcher is a stand-in for whatever that turns out to be.
- Decide the real deployment target (local script, small server, edge device near the camera, etc.) and the real alerting channel (§14 only prints to the console).

## 16. Conclusions, Limitations & Further Directions (2026-09-23)

### 16.1 Conclusions

- **No-empty split (4-class)** is the strongest model found: ~77% CV accuracy, ROC AUC 0.852 — clearly ahead of every other scheme once compared on the same ground truth (§12).
- Folding "empty filter" into "invalid" measurably helps, and it's the operationally correct design, not just a statistical convenience — §12 now covers both: the empty class is tiny (8 images), *and* a live deployment doesn't need to detect "empty" at all, since the operator empties the filter themselves and already knows it. The model is meant to be toggled off for that operator-controlled window and back on once filtration resumes.
- The naive 0.5 probability cutoff was **under-flagging real faults** — tuned thresholds (~0.3–0.5, F1-optimal) catch more actual invalid cases at a small cost in false alarms (§9).
- Ensembling/TTA/calibration did **not** beat the single best model — reported as a straight negative result rather than spun (§10).
- Ties back to the business case (§1.2): automated detection is estimated to recover **~$1.48M–$2.21M/year**, at 10–15% recovery of the ~$14.76M/year combined exposure (yield loss + downtime from equipment damage).

### 16.2 Limitations

- **Small dataset, by IT constraint, not by choice** — access to plant photos for this project was limited by IT/data-access restrictions, leaving only 176 images (8 of them "empty"). The results here are real and better than chance would predict, but every accuracy number carries meaningful uncertainty (CV std devs of several points) — a full-scale pilot with proper data access would retrain on a much larger database and should do materially better.
- Single seed, mostly CPU-trained runs — results could shift with a different seed or backbone.
- The **promoted model hasn't been trained for a real full run yet** — `full_pipeline.ipynb` still needs a real (non-`QUICK_MODE`) pass in Colab to produce final weights; "promoted" is provisional.
- **Day/night remains unexplained** — ruled out literal time-of-day and whole-image brightness, but the real distinguishing factor is still unknown (§16.3 proposes removing the ambiguity at the source instead).
- The live-feed demo is a **simulation** (folder-watcher + console alert), not a real camera/RTSP integration or real alerting channel.
- Single fixed camera install, uncontrolled lighting — unclear how well this generalizes to a different angle or to genuinely fixed lighting without retraining.
- Source images are WhatsApp-recompressed, not native camera resolution.
- The model currently only outputs a binary valid/invalid call — no sense of *how* valid or invalid, or of trend over time.

### 16.3 Further directions

**Near-term (fixing what this phase left open):**
- Run `full_pipeline.ipynb` for real to get trustworthy final numbers and the actual promoted model weights.
- Try confidence-weighted ensembling (weight the stronger model more) instead of a plain average.
- Build the real camera feed + alerting channel, replacing the folder-watcher simulation.

**Full-scale pilot (plant engineer's roadmap, 2026-09-23):**
- **Retrain on a much larger database** once IT/data-access constraints are lifted for a real pilot — the current 176-image result is a proof of concept, not the ceiling.
- **Add projectors around the filter** to normalize lighting between "day" and "night" conditions and reduce shadow effects — engineers the day/night ambiguity (§3.4, §16.2) away at the source instead of continuing to try to explain it after the fact.
- **Move beyond binary valid/invalid to a continuous filtration-quality score**, so the process's actual trend is visible, not just a pass/fail flag.
- **Make it a time-based model** that tracks change over a sequence of frames and flags when the process is *starting* to turn bad — early warning instead of single-frame classification after the fact.
- **Far future**: a downstream model that calculates the right reactor process parameters based on the predicted filtration state — closing the loop from vision straight through to process control, not just alerting an operator.
