# CLAUDE.md — Design Decisions & Implementation Reference
## Surface Damage Detector

This document explains every design decision made in this project.
It is the source of truth for why things are done the way they are.

---

## 1. Problem Definition

We classify surface images into 3 damage states across 3 surface types:

| | clean | smudged | cracked |
|---|---|---|---|
| **tiles** | uniform tile + grout | rust, mold, mud, footprints | hairline to shattered |
| **wood** | grain pattern, varies in color | burn marks, water stains, paint, chalk | along or across grain |
| **walls** | flat, near-uniform white/grey | tea stains, mold, pen marks | hairline lines |

Each surface gets its **own trained model** because:
- Visual features differ entirely between surfaces (grout vs grain vs flat)
- A crack on wood looks nothing like a crack on tile
- Smaller focused models outperform one large model on limited data
- Failure modes are surface-specific and easier to diagnose separately

---

## 2. Dataset Reality

After observing actual collected images:

**Tiles:**
- Grout lines create a regular grid of edges throughout the image
- Photos taken at angles cause perspective distortion of grout grid
- Smudges are visually diverse: rust (orange), mold (dark green/black),
  mud footprints (brown), soap (white/transparent), water marks
- Some cracks are hairline and nearly parallel to grout lines (hard case)

**Wood:**
- Natural grain creates dark streaks that visually resemble cracks
- Clean class has massive color variation: dark brown to bright orange
- Cracks can run parallel to grain (very hard) or across grain (easier)
- White/chalk smudges are lighter than the surface (dark smudge assumption fails)
- Burn marks are much darker than surface

**Walls:**
- Easiest surface — flat, low texture, high contrast for most defects
- Some smudges are extremely subtle (image 15: barely visible at bottom)
- Pen marks can resemble hairline cracks
- Mold appears as dark speckled pattern vs smudge blob

---

## 3. Why Not Use Raw RGB

Raw RGB gives the CNN three redundant color channels that encode surface
color more than defect type. On small datasets (100 images/class), the CNN
will overfit to surface color rather than learning defect features.

Instead we construct **3 semantically meaningful channels**:

### Channel 0: Saturation (HSV S-channel)
**What it encodes:** Color deviation from neutral surface background.

Clean surfaces (white tile, natural wood, white wall) have **low saturation** —
they are near-neutral in color. Every smudge type raises saturation:
- Rust: high orange saturation
- Mold: greenish/dark saturation spike
- Mud: brownish saturation spike
- Paint/chalk: desaturation spike (white on dark wood — saturation drops)
- Burn marks: dark, low saturation but contrast against wood

This channel works for **both** light smudges (which grayscale brightness misses)
and dark smudges (which grayscale also catches). It is the most universal
smudge detector across all surface types.

**Why not just use grayscale darkness for smudges:**
A user correctly asked this. It fails because:
- Water stains on dark wood are **lighter** than the surface
- White paint on brown wood is **lighter** than the surface
- Soap smudges on tiles have **similar brightness** but different color
- Saturation catches all of these; grayscale darkness only catches some

### Channel 1: CLAHE Grayscale
**What it encodes:** Normalized local texture and brightness.

Raw grayscale has a critical flaw for this dataset: the clean wood class
spans from dark brown to bright orange. If we use raw grayscale, the CNN
could learn "bright = clean wood" or "dark = cracked wood", which is wrong.

CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes contrast
**locally** within small tiles of the image. This means:
- Dark wood and light wood both become medium contrast after CLAHE
- The CNN sees texture patterns, not overall brightness
- Subtle wall smudges (image 15) get amplified — CLAHE reveals them
- Grout lines on tiles become more uniform relative to tile surface

CLAHE clip limits by surface:
- tiles: 2.0 — moderate amplification, grout is already high contrast
- wood:  3.0 — stronger amplification needed due to color variation
- walls: 4.0 — maximum amplification needed for near-invisible defects

### Channel 2: Canny Edge Map + Morphological Operations
**What it encodes:** Structural discontinuities in the surface.

Cracks are fundamentally **structural breaks** — they create sharp edges
that don't belong to the normal surface texture. This channel specifically
extracts those edges.

**Pipeline:**
1. Gaussian blur before Canny: smooths fine regular patterns (grout lines,
   wood grain) so they don't pollute the edge map with false positives
2. Canny edge detection: finds gradients above threshold — actual cracks
   survive because they are high-contrast structural breaks
3. Morphological CLOSE: connects broken crack line fragments (Canny often
   leaves gaps in a continuous crack line)
4. Morphological OPEN: removes isolated noise speckles that aren't cracks

**Why Gaussian blur kernel size differs by surface:**
- tiles 7×7: large blur needed to suppress dense grout grid
- wood 5×5: medium blur — suppresses some grain but preserves cross-grain crack signal
- walls 3×3: minimal blur — wall surface is flat with no regular pattern to suppress

**Why Canny thresholds differ by surface:**
- tiles (80/160): high thresholds — only strong edges survive, grout edges are filtered
- wood (40/100): lower — wood cracks have lower contrast against grain background
- walls (20/60): very low — wall cracks are hairline, extremely low contrast

**The grout line problem:**
Canny on raw tile images fires on every grout line, creating a grid of
edges that drowns out crack edges. The large Gaussian blur (7×7) blurs
grout lines (fine, regular) into background while cracks (irregular,
often wider gradient) survive. The CNN then learns that an edge NOT on
the regular grout grid = crack.

**The wood grain problem:**
Wood grain creates parallel dark streaks — CLAHE makes these uniform, and
the 5×5 blur further softens them. A crack across the grain is then visible
as a perpendicular edge. A crack along the grain is the hard case (image 1
in the dataset) where even human eyes struggle — the CNN must learn from
the slight irregularity vs the smooth grain lines.

---

## 4. No Cropping Policy

Images cannot be cropped because:
- Defects are distributed across the image, not centered
- Cropping risks removing the actual crack/smudge
- Fixed crop size would distort aspect ratios

**Solution: pad_to_square()**
- Scale image so longest side = target (256px)
- Zero-pad shorter side to make it square
- All spatial content preserved
- Zero padding has no features so CNN ignores it naturally

256×256 chosen over 128×128 because:
- Hairline cracks (image 5, 8 in tiles) need resolution to be detectable
- At 128×128 a 1-pixel crack becomes sub-pixel and disappears
- At 256×256 hairline cracks remain visible after preprocessing

---

## 5. CNN Architecture Decisions

### Why not use a single flat model
One model for all surfaces would need to learn surface identification
AND defect detection simultaneously. With only 300 images/surface, that
is too much to ask. Three separate models, each with 300 training images
focused on one surface, is the correct approach. TA confirmed this.

### Why double-conv blocks (VGG-style)
Single conv per block builds shallow feature maps at each spatial scale.
Double conv allows the model to build richer representations before
downsampling, which matters when your input contains subtle hairline cracks.

### Why Global Average Pooling not Flatten
After 4 MaxPool layers, feature maps are 16×16. Flattening gives 256×16×16
= 65,536 values before the FC layer. With only ~210 training images, this
creates massive overfitting risk.

Global Average Pooling collapses each 16×16 feature map to 1 value
(its average activation), giving 256 values into FC. This is a 256×
parameter reduction. The model is forced to learn globally-activating
features (crack present anywhere) rather than position-dependent ones.

### Why BatchNorm after every Conv
Small datasets = noisy gradient estimates = training instability.
BatchNorm normalizes activations per batch, which:
- Stabilizes gradients across training
- Acts as mild regularization
- Allows higher learning rates without divergence
- Critical here given batch_size=16 and ~210 training samples

### Why Dropout2d on conv blocks + Dropout on FC
- Dropout2d drops entire feature map channels (spatial dropout) —
  forces the model not to rely on any single learned filter
- Standard Dropout(0.5) on FC is the heaviest regularizer —
  any single path through the FC layer must not dominate
- Combined effect: significant overfitting resistance on 100 images/class

### Why Adam not SGD
SGD with momentum requires careful LR tuning and usually needs 150+ epochs.
Adam's adaptive per-parameter learning rates converge faster on small datasets
and are more forgiving of LR choice. With ReduceLROnPlateau as backup, this
gives stable training without manual scheduling.

---

## 6. Training Strategy Per Surface

### Tiles (lr=0.001, epochs=60)
Standard settings. Tiles is medium difficulty. The saturation channel
handles smudge variety, edge map handles cracks, grout is suppressed by
heavy blur. 60 epochs with early stopping at 15 is sufficient.

### Wood (lr=0.0005, epochs=80)
Lower LR and more epochs because:
- Natural grain variation creates many false edges — model needs more
  iterations to learn to ignore them
- Color variation in clean class is large — normalization via CLAHE helps
  but the model still needs to see many examples to learn "grain = normal"
- Lower LR prevents the model from making large weight updates on noisy
  gradient signals from ambiguous wood images (crack along grain)

### Walls (lr=0.001, epochs=60)
Standard settings. Walls is easiest but some smudges are very subtle
(image 15). CLAHE at clip=4.0 amplifies these. The model should converge
early — if val accuracy exceeds 90% at epoch 30, early stopping will fire.

---

## 7. Class Weighting

Computed from actual training set counts:
```python
weight[i] = total_train_samples / (num_classes × count_of_class_i)
```

If clean has 70 samples and smudged has 65 samples and cracked has 75:
- weight[clean]   = 210 / (3 × 70) = 1.0
- weight[smudged] = 210 / (3 × 65) = 1.077
- weight[cracked] = 210 / (3 × 75) = 0.933

Loss for a misclassified smudged sample is multiplied by 1.077 —
the model is penalized more for getting the minority class wrong.
This is critical if image collection resulted in uneven class sizes.

---

## 8. Augmentation Justification

Applied to training split only. Never val or test.

| Transform | Justification |
|---|---|
| RandomHorizontalFlip(0.5) | Cracks and smudges have no preferred horizontal direction |
| RandomVerticalFlip(0.5) | Same — tiles especially can be photographed from any orientation |
| RandomRotation(20°) | Handles angled shots; doesn't go full 90° to avoid confusion with grain direction on wood |
| ColorJitter(brightness=0.3, contrast=0.3) | Handles different lighting conditions across photos |

ColorJitter is applied to the preprocessed 3-channel tensor. This shifts
channel intensities rather than raw RGB — acceptable because it simulates
lighting variation that would affect all three feature channels similarly.

**Why no more aggressive augmentation:**
More aggressive augmentation (90° rotations, heavy flips, elastic distortions)
risks creating unrealistic images that hurt rather than help. The model
needs to generalize to real-world variation, not artifact variation.

---

## 9. Accuracy Expectations

| Surface | Expected Accuracy | Most Confused Classes |
|---|---|---|
| walls | 88–93% | clean vs subtle smudge |
| tiles | 83–88% | cracked vs smudged (rust stains can have edges) |
| wood  | 78–85% | clean vs cracked (grain parallel crack), clean vs smudged (dark wood) |

If wood falls below 78%:
1. First check confusion matrix — which class is being misclassified into which
2. If clean→cracked: lower Canny threshold to reduce grain edge false positives
3. If smudged→clean: increase CLAHE clip limit to amplify subtle smudges
4. If any class underperforms: add class weight for that class
5. Last resort: add one more conv block (increase capacity)

---

## 10. What This Is NOT Doing

- No sliding window / patch-based inference
- No attention mechanisms
- No data synthesis / GAN augmentation
- No test-time augmentation (TTA)
- No ensemble of multiple models

All of the above could improve accuracy further but are beyond the scope
of a university programming project and would require justification of
additional complexity that is not necessary to hit the 80–85% target.
