# AGENT.md — Surface Damage Detector

## Project Overview
Build 3 separate CNN models (from scratch in PyTorch, no pretrained weights) to classify
surface damage on 3 different surfaces. Each model is a 3-class classifier.

```
Surfaces:  tiles | wood | walls
Classes:   clean (0) | smudged (1) | cracked (2)
Dataset:   ~100 images per class per surface (~300 images per model)
Target:    ≥85% accuracy on tiles and walls, ≥80% accuracy on wood
```

---

## Repository Structure to Build

```
surface_detector/
├── AGENT.md                  ← this file
├── CLAUDE.md                 ← preprocessing + architecture decisions
├── data/
│   ├── tiles/
│   │   ├── clean/            ← ~100 images
│   │   ├── smudged/          ← ~100 images
│   │   └── cracked/          ← ~100 images
│   ├── wood/
│   │   ├── clean/
│   │   ├── smudged/
│   │   └── cracked/
│   └── walls/
│       ├── clean/
│       ├── smudged/
│       └── cracked/
├── dataset.py                ← Dataset class + all preprocessing pipelines
├── model.py                  ← CNN architecture (shared class, trained separately)
├── train.py                  ← training loop with validation + early stopping
├── evaluate.py               ← confusion matrix, per-class accuracy, F1
├── predict.py                ← single image inference
└── outputs/
    ├── tiles_best.pth
    ├── wood_best.pth
    └── walls_best.pth
```

---

## Task List (execute in this order)

### Step 1 — dataset.py
Build a custom `torch.utils.data.Dataset` class called `SurfaceDataset`.

**Constructor args:**
- `root_dir` (str): path to surface folder e.g. `data/tiles/`
- `surface` (str): one of `'tiles'`, `'wood'`, `'walls'`
- `split` (str): one of `'train'`, `'val'`, `'test'`
- `split_ratios` (tuple): default `(0.70, 0.15, 0.15)`
- `target_size` (int): default `256` — pad-to-square target
- `augment` (bool): apply augmentation only when split is `'train'`
- `seed` (int): default `42` for reproducible splits

**What it must do:**
1. Scan `root_dir` for subfolders `clean/`, `smudged/`, `cracked/`
2. Collect all image paths and integer labels
3. Stratified split into train/val/test (same seed every run)
4. On `__getitem__`:
   a. Load image with `cv2.imread` (BGR)
   b. Run surface-specific preprocessing → 3-channel numpy array shape `(3, H, W)` float32 in `[0, 1]`
   c. Apply augmentation if `self.augment` is True
   d. Return `(tensor, label)`

**Preprocessing pipelines — implement as standalone functions:**

```python
def pad_to_square(img, target=256):
    """
    Scale image so longest side = target, then zero-pad shorter side.
    Preserves aspect ratio. No cropping. No distortion.
    Works for both single-channel and 3-channel arrays.
    """

def get_saturation(img_bgr):
    """
    Convert BGR → HSV, return S channel (uint8).
    Smudges of all types (rust, mold, mud, grease, paint) deviate
    in saturation from neutral surface background.
    Works for both light and dark smudges unlike grayscale darkness.
    """

def get_clahe_gray(img_bgr, clip_limit, tile_grid):
    """
    Convert BGR → grayscale, apply CLAHE (Contrast Limited Adaptive
    Histogram Equalization). Normalizes local contrast so CNN does not
    learn overall brightness as a class signal. Critical for wood
    (massive clean-class color variation) and walls (very low contrast).
    Returns uint8.
    """

def get_edge_map(gray_clahe, blur_ksize, canny_t1, canny_t2):
    """
    1. Gaussian blur: suppresses fine regular patterns (grout lines,
       wood grain) that would pollute Canny output.
    2. Canny: detects strong structural discontinuities (cracks).
    3. Morphological CLOSE (dilate→erode): connects broken crack line
       fragments that Canny leaves as gaps.
    4. Morphological OPEN (erode→dilate): removes isolated noise speckles.
    Returns uint8 binary edge map.
    """

def preprocess(img_bgr, surface, target_size=256):
    """
    Master dispatcher. Calls surface-specific params.
    Returns float32 numpy array shape (3, target_size, target_size)
    with values in [0.0, 1.0].

    Channel layout (same for all surfaces, params differ):
      ch0 = Saturation map         → smudge color signal
      ch1 = CLAHE grayscale        → normalized texture/brightness
      ch2 = Canny + Morph edges    → crack structural signal

    Surface-specific parameters:
      tiles: clip=2.0, tile=(8,8), blur=7, canny=(80,160)
      wood:  clip=3.0, tile=(8,8), blur=5, canny=(40,100)
      walls: clip=4.0, tile=(8,8), blur=3, canny=(20,60)

    Rationale:
      tiles — large blur suppresses grout lines; moderate Canny thresholds
      wood  — medium blur preserves cross-grain crack signal; lower thresholds
              because cracks on wood are lower contrast than on tile
      walls — minimal blur (surface is flat, no grain/grout to suppress);
              very low Canny thresholds because wall cracks are hairline
              and CLAHE clip=4.0 amplifies subtle low-contrast wall features
    """
```

**Augmentation — implement using torchvision.transforms (train split only):**
```python
# Apply AFTER preprocessing tensor is built, on the float32 tensor
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    # Note: ColorJitter on a 3-channel preprocessed image adjusts
    # per-channel intensity — acceptable as a brightness/contrast shift
])
```

---

### Step 2 — model.py
Build a CNN class called `SurfaceCNN` from scratch. No pretrained weights. No torchvision model classes.

**Architecture:**
```
Input: (batch, 3, 256, 256)

Block 1: Conv(3→32, k=3, pad=1) → BN → ReLU → Conv(32→32, k=3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.1)
  Output: (batch, 32, 128, 128)

Block 2: Conv(32→64, k=3, pad=1) → BN → ReLU → Conv(64→64, k=3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.1)
  Output: (batch, 64, 64, 64)

Block 3: Conv(64→128, k=3, pad=1) → BN → ReLU → Conv(128→128, k=3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.2)
  Output: (batch, 128, 32, 32)

Block 4: Conv(128→256, k=3, pad=1) → BN → ReLU → Conv(256→256, k=3, pad=1) → BN → ReLU → MaxPool(2,2) → Dropout2d(0.2)
  Output: (batch, 256, 16, 16)

Global Average Pooling → (batch, 256)
  [replaces Flatten+large FC — reduces parameters, fights overfitting on small dataset]

FC: 256 → 128 → ReLU → Dropout(0.5) → 128 → 3
  Output: (batch, 3) raw logits
```

**Why this architecture:**
- Double conv per block (VGG-style) builds richer feature maps per spatial scale
- BatchNorm after every conv stabilizes training on small datasets
- Dropout2d on conv blocks + Dropout on FC prevents memorization of 100 images
- Global Average Pooling instead of Flatten dramatically reduces parameter count
  (256 params into FC vs 256×16×16=65536), which is critical with only ~210 training images
- 4 blocks gives receptive field large enough to see crack length patterns

**Constructor args:**
- `num_classes=3` (always 3 for this project)
- `dropout_rate=0.5` (FC dropout, tunable)

**Include a `count_parameters()` method that prints total trainable params.**

---

### Step 3 — train.py
Training script. Must be runnable as:
```bash
python train.py --surface tiles --epochs 60 --lr 0.001 --batch_size 16
python train.py --surface wood  --epochs 80 --lr 0.0005 --batch_size 16
python train.py --surface walls --epochs 60 --lr 0.001  --batch_size 16
```

**Must implement:**

1. **DataLoaders** for train/val/test splits from `SurfaceDataset`

2. **Loss:** `nn.CrossEntropyLoss` with class weights computed from training set:
   ```python
   # Inverse frequency weighting handles any class imbalance
   counts = [count_per_class_0, count_per_class_1, count_per_class_2]
   weights = [total / (3 * c) for c in counts]
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
   ```

3. **Optimizer:** `Adam` with `weight_decay=1e-4` (L2 regularization)

4. **Scheduler:** `ReduceLROnPlateau(patience=7, factor=0.5)` on val loss
   — halves LR when val loss stops improving, avoids manual tuning

5. **Early stopping:** stop training if val loss does not improve for 15 epochs
   — prevents overfitting on small dataset

6. **Checkpoint:** save best model (lowest val loss) to `outputs/{surface}_best.pth`

7. **Per-epoch logging** (print to console):
   ```
   Epoch 12/60 | Train Loss: 0.4231 | Train Acc: 84.2% | Val Loss: 0.5102 | Val Acc: 81.3% | LR: 0.001000
   ```

8. **After training ends**, run final evaluation on test set and print:
   ```
   === TEST RESULTS: tiles ===
   Test Accuracy: 87.5%
   Per-class accuracy:
     clean:   91.3%
     smudged: 84.2%
     cracked: 87.0%
   ```

**Surface-specific default hyperparameters (encode as argparse defaults per surface or a config dict):**
```python
SURFACE_CONFIGS = {
    'tiles': {'epochs': 60, 'lr': 0.001,  'batch_size': 16},
    'wood':  {'epochs': 80, 'lr': 0.0005, 'batch_size': 16},
    'walls': {'epochs': 60, 'lr': 0.001,  'batch_size': 16},
}
```
Wood gets more epochs and lower LR because it is the hardest surface —
the model needs longer, more careful training to distinguish natural grain
variation from actual defects.

---

### Step 4 — evaluate.py
Evaluation script. Runnable as:
```bash
python evaluate.py --surface tiles
python evaluate.py --surface wood
python evaluate.py --surface walls
```

**Must produce:**
1. Overall accuracy on test set
2. Per-class precision, recall, F1 (use sklearn.metrics.classification_report)
3. Confusion matrix printed to console as a formatted table
4. Save confusion matrix as `outputs/{surface}_confusion_matrix.png` using matplotlib

**Confusion matrix format:**
```
              Predicted
              clean  smudged  cracked
Actual clean  [  20      2       1  ]
     smudged  [   1     18       3  ]
     cracked  [   0      2      20  ]
```

---

### Step 5 — predict.py
Single image inference. Runnable as:
```bash
python predict.py --surface tiles --image path/to/image.jpg
```

**Output:**
```
Surface:    tiles
Image:      path/to/image.jpg
Prediction: cracked (confidence: 94.3%)
All scores: clean=2.1%  smudged=3.6%  cracked=94.3%
```

Must load the saved `.pth` file, run preprocessing pipeline for the specified surface,
and output softmax probabilities for all 3 classes.

---

## Dependencies
```
torch>=2.0
torchvision>=0.15
opencv-python>=4.8
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
Pillow>=9.0
```

---

## Critical Constraints
- NO pretrained models. No `torchvision.models`. Build CNN from scratch.
- NO cropping of images. Use pad_to_square only.
- Preprocessing must be deterministic (same image always gives same tensor).
- Augmentation only on train split, never val or test.
- All 3 surface models use the same CNN class, trained independently.
- Seed everything: `torch.manual_seed(42)`, `np.random.seed(42)` at top of train.py.
- Save best checkpoint by val loss, not val accuracy (more stable with small dataset).
