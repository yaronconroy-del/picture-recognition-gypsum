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

## 9. Open Questions / Next Steps

- Confirm the exact equipment name/process (what is a "Gibson filter" — brand/model — and what specifically defines "invalid" beyond visual cracking/patchiness?).
- Confirm what "day"/"night" actually mean in the source photos (§3.4) — the timestamps rule out literal time-of-day.
- **Decide the label scheme** given §6 and §8 together: no-empty split (4-class) is the current leader, ahead of extended (5-class), ahead of lite (3-class) — still on a very small dataset, single seed, CPU-only runs.
- Decide where/how the camera feed will be sampled for live inference (folder of new images, RTSP stream, etc.).
- Decide the deployment target (local script, small server, edge device near the camera, etc.) and how alerts should be delivered.
