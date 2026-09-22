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
- **Day/night lighting is a major visual shift**: night shots have strong localized glare from floodlights; day shots have flatter, hazier lighting. Whatever splitting/augmentation strategy is used should account for this so the model doesn't just learn "day vs. night" instead of "valid vs. invalid."
- **Fixed camera angle**: all images appear to come from the same mounted camera/angle, which simplifies the problem (no need for viewpoint invariance) but means the model may be specific to this one camera installation.
- **Source quality**: images are WhatsApp exports (compressed JPEGs, not original camera resolution), which is worth keeping in mind for image-quality-sensitive techniques.

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

## 6. Open Questions / Next Steps

- Confirm the exact equipment name/process (what is a "Gibson filter" — brand/model — and what specifically defines "invalid" beyond visual cracking/patchiness?).
- Decide where/how the camera feed will be sampled for live inference (folder of new images, RTSP stream, etc.).
- Decide the deployment target (local script, small server, edge device near the camera, etc.) and how alerts should be delivered.
