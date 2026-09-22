"""
Exploratory data analysis on the raw 176-photo dataset: class
distribution, a representative sample-image grid, image-size
consistency, and a brightness comparison across classes and across
day/night — run once as its own analysis, separate from any model
training.

Follows up on Evaluation & Docs/Project Workflow.md SS3.4 (the finding
that "day"/"night" don't correspond to literal time-of-day): if the
folders still differ in actual pixel brightness, that's evidence
they're tracking something real and visual (e.g. flash vs. no flash),
not an arbitrary or meaningless split.

Usage:
  python eda.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EXTENDED_DIR = os.path.join(TRAINING_DIR, "pictures", "extended version")
OUT_DIR = os.path.join(TRAINING_DIR, "eda")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)

CLASS_FOLDERS = ["תקין יום", "תקין לילה", "לא תקין יום", "לא תקין לילה", "מסנן ריק"]
CLASS_LABELS_EN = {
    "תקין יום": "valid_day", "תקין לילה": "valid_night",
    "לא תקין יום": "invalid_day", "לא תקין לילה": "invalid_night",
    "מסנן ריק": "empty",
}
CLASS_COLORS = {
    "valid_day": "#4c8bf5", "valid_night": "#8ab4f8",
    "invalid_day": "#e05252", "invalid_night": "#e88a8a",
    "empty": "#888888",
}


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))


def mean_brightness(path):
    with Image.open(path) as img:
        return float(np.array(img.convert("L")).mean())


# --------------------------------------------------------------- 1. Class distribution
counts = {}
for folder in CLASS_FOLDERS:
    counts[CLASS_LABELS_EN[folder]] = len(list_images(os.path.join(EXTENDED_DIR, folder)))
print("Class counts:", counts)

fig, ax = plt.subplots(figsize=(7, 4))
names = list(counts.keys())
values = [counts[n] for n in names]
ax.bar(names, values, color=[CLASS_COLORS[n] for n in names])
for i, v in enumerate(values):
    ax.text(i, v + 1, str(v), ha="center", fontsize=9)
ax.set_ylabel("# images")
ax.set_title(f"Class distribution — extended (5-class) scheme, n={sum(values)}")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=130)
plt.show()

# --------------------------------------------------------------- 2. Sample image grid
fig, axes = plt.subplots(len(CLASS_FOLDERS), 4, figsize=(10, 12.5))
for row, folder in enumerate(CLASS_FOLDERS):
    path = os.path.join(EXTENDED_DIR, folder)
    files = list_images(path)
    sample = random.sample(files, min(4, len(files)))
    for col in range(4):
        ax = axes[row][col]
        ax.set_xticks([]); ax.set_yticks([])
        if col < len(sample):
            with Image.open(os.path.join(path, sample[col])) as img:
                ax.imshow(img.convert("RGB"))
        else:
            ax.axis("off")
        if col == 0:
            ax.set_ylabel(CLASS_LABELS_EN[folder], fontsize=10)
fig.suptitle("Sample images per class (random, seed=42)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sample_grid.png"), dpi=110)
plt.show()

# --------------------------------------------------------------- 3. Brightness by class
brightness_by_class = {}
for folder in CLASS_FOLDERS:
    path = os.path.join(EXTENDED_DIR, folder)
    brightness_by_class[CLASS_LABELS_EN[folder]] = [
        mean_brightness(os.path.join(path, f)) for f in list_images(path)
    ]

fig, ax = plt.subplots(figsize=(8, 4.5))
labels = list(brightness_by_class.keys())
ax.boxplot(brightness_by_class.values(), tick_labels=labels)
ax.set_ylabel("mean pixel brightness (0-255)")
ax.set_title("Brightness distribution by class")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "brightness_by_class.png"), dpi=130)
plt.show()

# --------------------------------------------------------------- 4. Brightness: day vs night
day_vals = brightness_by_class["valid_day"] + brightness_by_class["invalid_day"]
night_vals = brightness_by_class["valid_night"] + brightness_by_class["invalid_night"]
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.boxplot([day_vals, night_vals], tick_labels=["day", "night"])
ax.set_ylabel("mean pixel brightness (0-255)")
ax.set_title("Brightness: day vs. night (valid + invalid pooled)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "brightness_day_vs_night.png"), dpi=130)
plt.show()

print(f"day    mean brightness: {np.mean(day_vals):.1f}  (n={len(day_vals)})")
print(f"night  mean brightness: {np.mean(night_vals):.1f}  (n={len(night_vals)})")

# --------------------------------------------------------------- 5. Image size consistency
sizes = set()
for folder in CLASS_FOLDERS:
    path = os.path.join(EXTENDED_DIR, folder)
    for f in list_images(path):
        with Image.open(os.path.join(path, f)) as img:
            sizes.add(img.size)
print("Distinct image sizes found:", sizes)

print(f"\nEDA complete. Plots saved to {OUT_DIR}")
