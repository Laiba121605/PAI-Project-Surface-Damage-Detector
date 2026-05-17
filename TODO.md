# TODO.md — Surface Damage Detector
## Complete Implementation Checklist

---

## Phase 0 — Project Setup

- [ ] Create top-level project folder: `surface_detector/`
- [ ] Create subdirectory structure:
  - [ ] `data/tiles/clean/`
  - [ ] `data/tiles/smudged/`
  - [ ] `data/tiles/cracked/`
  - [ ] `data/wood/clean/`
  - [ ] `data/wood/smudged/`
  - [ ] `data/wood/cracked/`
  - [ ] `data/walls/clean/`
  - [ ] `data/walls/smudged/`
  - [ ] `data/walls/cracked/`
  - [ ] `outputs/`
- [ ] Copy `AGENT.md` and `CLAUDE.md` into project root
- [ ] Install all dependencies:
  - [ ] `torch>=2.0`
  - [ ] `torchvision>=0.15`
  - [ ] `opencv-python>=4.8`
  - [ ] `numpy>=1.24`
  - [ ] `scikit-learn>=1.3`
  - [ ] `matplotlib>=3.7`
  - [ ] `Pillow>=9.0`
- [ ] Populate each class folder with ~100 images per class per surface (~900 images total)
- [ ] Verify image files are readable by OpenCV (JPG/PNG, not corrupt)

---

## Phase 1 — `dataset.py`

### Preprocessing Helper Functions

- [ ] Implement `pad_to_square(img, target=256)`
  - [ ] Compute scale factor so longest side equals `target`
  - [ ] Resize image with computed scale (no cropping)
  - [ ] Calculate padding needed on shorter side
  - [ ] Apply symmetric zero-padding (split evenly on both sides)
  - [ ] Handle both single-channel (H, W) and 3-channel (H, W, 3) arrays
  - [ ] Return padded image as `uint8`
  - [ ] Verify output shape is exactly `(target, target)` or `(target, target, 3)`

- [ ] Implement `get_saturation(img_bgr)`
  - [ ] Convert BGR → HSV using `cv2.cvtColor`
  - [ ] Extract S channel (index 1)
  - [ ] Return S channel as `uint8` array shape `(H, W)`

- [ ] Implement `get_clahe_gray(img_bgr, clip_limit, tile_grid)`
  - [ ] Convert BGR → grayscale using `cv2.cvtColor`
  - [ ] Create CLAHE object: `cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)`
  - [ ] Apply CLAHE to grayscale image
  - [ ] Return result as `uint8` array shape `(H, W)`

- [ ] Implement `get_edge_map(gray_clahe, blur_ksize, canny_t1, canny_t2)`
  - [ ] Apply Gaussian blur: `cv2.GaussianBlur(gray_clahe, (blur_ksize, blur_ksize), 0)`
  - [ ] Apply Canny: `cv2.Canny(blurred, canny_t1, canny_t2)`
  - [ ] Define morphological kernel `(3, 3)` using `cv2.getStructuringElement`
  - [ ] Apply morphological CLOSE: `cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)`
  - [ ] Apply morphological OPEN: `cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)`
  - [ ] Return final binary edge map as `uint8` array shape `(H, W)`

- [ ] Implement `preprocess(img_bgr, surface, target_size=256)`
  - [ ] Define surface-specific parameter lookup:
    - [ ] `tiles`: `clip=2.0, tile=(8,8), blur=7, canny=(80,160)`
    - [ ] `wood`:  `clip=3.0, tile=(8,8), blur=5, canny=(40,100)`
    - [ ] `walls`: `clip=4.0, tile=(8,8), blur=3, canny=(20,60)`
  - [ ] Raise `ValueError` for unknown surface name
  - [ ] Call `get_saturation(img_bgr)` → ch0 raw
  - [ ] Call `get_clahe_gray(img_bgr, clip, tile)` → ch1 raw
  - [ ] Call `get_edge_map(ch1_raw, blur, canny_t1, canny_t2)` → ch2 raw
  - [ ] Call `pad_to_square(ch0_raw, target_size)` → ch0 padded
  - [ ] Call `pad_to_square(ch1_raw, target_size)` → ch1 padded
  - [ ] Call `pad_to_square(ch2_raw, target_size)` → ch2 padded
  - [ ] Stack channels: `np.stack([ch0, ch1, ch2], axis=0)` → shape `(3, 256, 256)`
  - [ ] Normalize to `[0.0, 1.0]`: divide by `255.0`
  - [ ] Cast to `float32`
  - [ ] Return array of shape `(3, target_size, target_size)` dtype `float32`

### Augmentation Pipeline

- [ ] Define `augmentation` using `torchvision.transforms.Compose`:
  - [ ] `RandomHorizontalFlip(p=0.5)`
  - [ ] `RandomVerticalFlip(p=0.5)`
  - [ ] `RandomRotation(degrees=20)`
  - [ ] `ColorJitter(brightness=0.3, contrast=0.3)`
- [ ] Confirm augmentation operates on float32 tensor (not PIL image)

### `SurfaceDataset` Class

- [ ] Define class `SurfaceDataset(torch.utils.data.Dataset)`
- [ ] Implement `__init__(self, root_dir, surface, split, split_ratios=(0.70, 0.15, 0.15), target_size=256, augment=False, seed=42)`:
  - [ ] Validate `surface` is one of `'tiles'`, `'wood'`, `'walls'`
  - [ ] Validate `split` is one of `'train'`, `'val'`, `'test'`
  - [ ] Validate `split_ratios` sums to 1.0
  - [ ] Scan `root_dir` for subfolders `clean/`, `smudged/`, `cracked/`
  - [ ] Collect all image file paths and assign integer labels: `clean=0`, `smudged=1`, `cracked=2`
  - [ ] Filter out non-image files (accept `.jpg`, `.jpeg`, `.png`)
  - [ ] Implement stratified split (per-class) with `seed`:
    - [ ] For each class, sort paths (for determinism), then shuffle with seeded RNG
    - [ ] Compute cut indices: `n_train = int(n * 0.70)`, `n_val = int(n * 0.15)`, rest = test
    - [ ] Select the correct subset for this `split`
  - [ ] Store final list of `(path, label)` tuples as `self.samples`
  - [ ] Store `self.augment = augment and split == 'train'`
  - [ ] Store `self.surface`, `self.target_size`

- [ ] Implement `__len__(self)`:
  - [ ] Return `len(self.samples)`

- [ ] Implement `__getitem__(self, idx)`:
  - [ ] Load image: `cv2.imread(path)` (returns BGR)
  - [ ] Handle `None` return from `cv2.imread` (raise `IOError` with path)
  - [ ] Call `preprocess(img_bgr, self.surface, self.target_size)` → numpy `(3, H, W)` float32
  - [ ] Convert to `torch.FloatTensor`: `torch.from_numpy(arr)`
  - [ ] If `self.augment` is True, apply `augmentation(tensor)`
  - [ ] Return `(tensor, label)` where label is an `int`

### Dataset Verification

- [ ] Write a quick smoke test (can be a `if __name__ == '__main__':` block):
  - [ ] Instantiate `SurfaceDataset` for each surface and each split
  - [ ] Print length of each split
  - [ ] Load one sample, verify tensor shape is `(3, 256, 256)` and dtype is `float32`
  - [ ] Verify value range is `[0.0, 1.0]`
  - [ ] Confirm train+val+test sizes sum to total image count

---

## Phase 2 — `model.py`

- [ ] Define class `SurfaceCNN(nn.Module)`
- [ ] Implement `__init__(self, num_classes=3, dropout_rate=0.5)`:
  - [ ] **Block 1** — `self.block1`:
    - [ ] `Conv2d(3, 32, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(32)`
    - [ ] `ReLU()`
    - [ ] `Conv2d(32, 32, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(32)`
    - [ ] `ReLU()`
    - [ ] `MaxPool2d(2, 2)` → output `(batch, 32, 128, 128)`
    - [ ] `Dropout2d(0.1)`
  - [ ] **Block 2** — `self.block2`:
    - [ ] `Conv2d(32, 64, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(64)`
    - [ ] `ReLU()`
    - [ ] `Conv2d(64, 64, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(64)`
    - [ ] `ReLU()`
    - [ ] `MaxPool2d(2, 2)` → output `(batch, 64, 64, 64)`
    - [ ] `Dropout2d(0.1)`
  - [ ] **Block 3** — `self.block3`:
    - [ ] `Conv2d(64, 128, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(128)`
    - [ ] `ReLU()`
    - [ ] `Conv2d(128, 128, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(128)`
    - [ ] `ReLU()`
    - [ ] `MaxPool2d(2, 2)` → output `(batch, 128, 32, 32)`
    - [ ] `Dropout2d(0.2)`
  - [ ] **Block 4** — `self.block4`:
    - [ ] `Conv2d(128, 256, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(256)`
    - [ ] `ReLU()`
    - [ ] `Conv2d(256, 256, kernel_size=3, padding=1)`
    - [ ] `BatchNorm2d(256)`
    - [ ] `ReLU()`
    - [ ] `MaxPool2d(2, 2)` → output `(batch, 256, 16, 16)`
    - [ ] `Dropout2d(0.2)`
  - [ ] **Global Average Pooling** — `self.gap = nn.AdaptiveAvgPool2d(1)` → output `(batch, 256)`
  - [ ] **FC Head** — `self.classifier`:
    - [ ] `Linear(256, 128)`
    - [ ] `ReLU()`
    - [ ] `Dropout(dropout_rate)`
    - [ ] `Linear(128, num_classes)`
  - [ ] Do NOT use any `torchvision.models` class

- [ ] Implement `forward(self, x)`:
  - [ ] Pass through `block1` → `block2` → `block3` → `block4`
  - [ ] Apply `self.gap(x)` → shape `(batch, 256, 1, 1)`
  - [ ] Flatten: `x = x.view(x.size(0), -1)` → shape `(batch, 256)`
  - [ ] Pass through `self.classifier`
  - [ ] Return raw logits `(batch, 3)` — no softmax here

- [ ] Implement `count_parameters(self)`:
  - [ ] Sum all `p.numel()` where `p.requires_grad` is True
  - [ ] Print: `Total trainable parameters: X,XXX,XXX`
  - [ ] Return the count as an integer

### Model Verification

- [ ] Add `if __name__ == '__main__':` smoke test:
  - [ ] Instantiate `SurfaceCNN()`
  - [ ] Call `model.count_parameters()`
  - [ ] Pass dummy tensor `torch.zeros(2, 3, 256, 256)` through model
  - [ ] Assert output shape is `(2, 3)`
  - [ ] Confirm no pretrained weight loading anywhere in file

---

## Phase 3 — `train.py`

### Configuration & Setup

- [ ] Define `SURFACE_CONFIGS` dict at top of file:
  ```python
  SURFACE_CONFIGS = {
      'tiles': {'epochs': 60, 'lr': 0.001,  'batch_size': 16},
      'wood':  {'epochs': 80, 'lr': 0.0005, 'batch_size': 16},
      'walls': {'epochs': 60, 'lr': 0.001,  'batch_size': 16},
  }
  ```
- [ ] Set global seeds at top of script (before any torch/numpy calls):
  - [ ] `torch.manual_seed(42)`
  - [ ] `np.random.seed(42)`
  - [ ] `torch.backends.cudnn.deterministic = True` (optional, for full reproducibility)

### Argument Parsing

- [ ] Set up `argparse.ArgumentParser`
- [ ] Add `--surface` argument (required, choices: `tiles`, `wood`, `walls`)
- [ ] Add `--epochs` argument (type int, default pulled from `SURFACE_CONFIGS[surface]`)
- [ ] Add `--lr` argument (type float, default pulled from `SURFACE_CONFIGS[surface]`)
- [ ] Add `--batch_size` argument (type int, default pulled from `SURFACE_CONFIGS[surface]`)
- [ ] Add `--data_dir` argument (type str, default `'data/'`)
- [ ] Add `--output_dir` argument (type str, default `'outputs/'`)
- [ ] Parse args and load surface-specific defaults where CLI args not provided

### DataLoaders

- [ ] Instantiate `SurfaceDataset` for `split='train'`, `augment=True`
- [ ] Instantiate `SurfaceDataset` for `split='val'`, `augment=False`
- [ ] Instantiate `SurfaceDataset` for `split='test'`, `augment=False`
- [ ] Create `DataLoader` for train split: `shuffle=True`, `num_workers=2`, `pin_memory=True`
- [ ] Create `DataLoader` for val split: `shuffle=False`, `num_workers=2`
- [ ] Create `DataLoader` for test split: `shuffle=False`, `num_workers=2`
- [ ] Print dataset sizes: train / val / test counts

### Class Weights

- [ ] Count samples per class in the training set
- [ ] Compute inverse-frequency weights:
  ```python
  total = sum(counts)
  weights = [total / (3 * c) for c in counts]
  ```
- [ ] Create `nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))`
- [ ] Move weight tensor to same device as model

### Model, Optimizer, Scheduler

- [ ] Instantiate `SurfaceCNN(num_classes=3)`
- [ ] Move model to device (`cuda` if available, else `cpu`)
- [ ] Call `model.count_parameters()` and print
- [ ] Instantiate `Adam` optimizer with `lr=args.lr, weight_decay=1e-4`
- [ ] Instantiate `ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5, verbose=True)`

### Training Loop

- [ ] Initialize `best_val_loss = float('inf')`, `epochs_no_improve = 0`
- [ ] Loop over epochs `1` to `args.epochs`:
  - [ ] **Train phase:**
    - [ ] Set `model.train()`
    - [ ] Iterate over train DataLoader
    - [ ] Move `(inputs, labels)` to device
    - [ ] Zero gradients: `optimizer.zero_grad()`
    - [ ] Forward pass: `outputs = model(inputs)`
    - [ ] Compute loss: `loss = criterion(outputs, labels)`
    - [ ] Backward pass: `loss.backward()`
    - [ ] Update weights: `optimizer.step()`
    - [ ] Accumulate train loss and correct predictions
    - [ ] Compute `train_loss` and `train_acc` for the epoch
  - [ ] **Validation phase:**
    - [ ] Set `model.eval()`
    - [ ] Use `torch.no_grad()` context
    - [ ] Iterate over val DataLoader
    - [ ] Compute val loss and correct predictions
    - [ ] Compute `val_loss` and `val_acc` for the epoch
  - [ ] **LR Scheduler step:** `scheduler.step(val_loss)`
  - [ ] **Log epoch results** (exactly this format):
    ```
    Epoch XX/YY | Train Loss: X.XXXX | Train Acc: XX.X% | Val Loss: X.XXXX | Val Acc: XX.X% | LR: X.XXXXXX
    ```
  - [ ] **Checkpoint:** if `val_loss < best_val_loss`:
    - [ ] Update `best_val_loss = val_loss`
    - [ ] Save model state: `torch.save(model.state_dict(), 'outputs/{surface}_best.pth')`
    - [ ] Reset `epochs_no_improve = 0`
  - [ ] **Else:** increment `epochs_no_improve`
  - [ ] **Early stopping check:** if `epochs_no_improve >= 15`: print message and `break`

### Post-Training Test Evaluation

- [ ] Load best checkpoint: `model.load_state_dict(torch.load('outputs/{surface}_best.pth'))`
- [ ] Set `model.eval()` and run inference on test DataLoader
- [ ] Compute overall test accuracy
- [ ] Compute per-class accuracy for `clean`, `smudged`, `cracked`
- [ ] Print results in this format:
  ```
  === TEST RESULTS: {surface} ===
  Test Accuracy: XX.X%
  Per-class accuracy:
    clean:   XX.X%
    smudged: XX.X%
    cracked: XX.X%
  ```

### Script Verification

- [ ] Confirm script runs end-to-end with: `python train.py --surface tiles`
- [ ] Confirm checkpoint file appears at `outputs/tiles_best.pth` after training
- [ ] Confirm early stopping fires correctly if val loss stagnates
- [ ] Confirm `python train.py --surface wood --epochs 80 --lr 0.0005` runs correctly

---

## Phase 4 — `evaluate.py`

### Argument Parsing

- [ ] Add `--surface` argument (required, choices: `tiles`, `wood`, `walls`)
- [ ] Add `--data_dir` argument (type str, default `'data/'`)
- [ ] Add `--output_dir` argument (type str, default `'outputs/'`)

### Evaluation Logic

- [ ] Load best checkpoint from `outputs/{surface}_best.pth`
- [ ] Instantiate `SurfaceCNN`, load state dict, move to device, set `model.eval()`
- [ ] Instantiate test `SurfaceDataset` and `DataLoader`
- [ ] Run inference loop with `torch.no_grad()`
- [ ] Collect all true labels and predicted labels into lists

### Metrics Output

- [ ] Compute and print **overall accuracy**
- [ ] Compute and print `sklearn.metrics.classification_report` with:
  - [ ] `target_names=['clean', 'smudged', 'cracked']`
  - [ ] Shows per-class precision, recall, F1, support
- [ ] Compute `sklearn.metrics.confusion_matrix`
- [ ] Print confusion matrix as formatted console table:
  ```
                Predicted
                clean  smudged  cracked
  Actual clean  [  20      2       1  ]
       smudged  [   1     18       3  ]
       cracked  [   0      2      20  ]
  ```

### Confusion Matrix Plot

- [ ] Create matplotlib figure for confusion matrix heatmap:
  - [ ] Use `matplotlib.pyplot.imshow` or `seaborn.heatmap` (seaborn optional)
  - [ ] Label axes with class names
  - [ ] Add value annotations in each cell
  - [ ] Add title: `{surface} — Confusion Matrix`
  - [ ] Add colorbar
- [ ] Save figure to `outputs/{surface}_confusion_matrix.png`
- [ ] Print: `Saved confusion matrix to outputs/{surface}_confusion_matrix.png`

### Script Verification

- [ ] Confirm runs with: `python evaluate.py --surface tiles`
- [ ] Confirm `.png` file is saved to `outputs/`
- [ ] Confirm accuracy matches post-training test accuracy from `train.py`

---

## Phase 5 — `predict.py`

### Argument Parsing

- [ ] Add `--surface` argument (required, choices: `tiles`, `wood`, `walls`)
- [ ] Add `--image` argument (required, path to image file)
- [ ] Add `--output_dir` argument (type str, default `'outputs/'`)

### Inference Logic

- [ ] Validate image file exists; raise clear error if not
- [ ] Load image: `cv2.imread(args.image)` (BGR)
- [ ] Handle `None` return (unreadable file)
- [ ] Call `preprocess(img_bgr, args.surface, target_size=256)` → numpy `(3, 256, 256)` float32
- [ ] Convert to tensor and add batch dim: `tensor.unsqueeze(0)` → shape `(1, 3, 256, 256)`
- [ ] Load `SurfaceCNN`, load `outputs/{surface}_best.pth`, set `model.eval()`
- [ ] Run forward pass with `torch.no_grad()`
- [ ] Apply softmax to logits: `torch.nn.functional.softmax(logits, dim=1)`
- [ ] Extract predicted class index: `torch.argmax(probs, dim=1).item()`
- [ ] Map index to class name: `{0: 'clean', 1: 'smudged', 2: 'cracked'}`
- [ ] Extract confidence: `probs[0][pred_idx].item() * 100`

### Output Format

- [ ] Print results in exactly this format:
  ```
  Surface:    {surface}
  Image:      {image_path}
  Prediction: {class_name} (confidence: XX.X%)
  All scores: clean=XX.X%  smudged=XX.X%  cracked=XX.X%
  ```

### Script Verification

- [ ] Confirm runs with: `python predict.py --surface tiles --image data/tiles/clean/img001.jpg`
- [ ] Confirm all 3 softmax probabilities sum to ~100%
- [ ] Confirm confidence matches `All scores` entry for predicted class

---

## Phase 6 — End-to-End Testing

### Per-Surface Full Run

- [ ] **Tiles:**
  - [ ] `python train.py --surface tiles`
  - [ ] Confirm `outputs/tiles_best.pth` saved
  - [ ] `python evaluate.py --surface tiles`
  - [ ] Confirm test accuracy ≥ 85%
  - [ ] Confirm `outputs/tiles_confusion_matrix.png` saved
  - [ ] `python predict.py --surface tiles --image <any_tile_image>`

- [ ] **Wood:**
  - [ ] `python train.py --surface wood`
  - [ ] Confirm `outputs/wood_best.pth` saved
  - [ ] `python evaluate.py --surface wood`
  - [ ] Confirm test accuracy ≥ 80%
  - [ ] Confirm `outputs/wood_confusion_matrix.png` saved
  - [ ] `python predict.py --surface wood --image <any_wood_image>`

- [ ] **Walls:**
  - [ ] `python train.py --surface walls`
  - [ ] Confirm `outputs/walls_best.pth` saved
  - [ ] `python evaluate.py --surface walls`
  - [ ] Confirm test accuracy ≥ 85%
  - [ ] Confirm `outputs/walls_confusion_matrix.png` saved
  - [ ] `python predict.py --surface walls --image <any_wall_image>`

### Constraint Checklist

- [ ] No `torchvision.models` import anywhere in codebase
- [ ] No pretrained weights loaded anywhere
- [ ] No `cv2.resize` with crop — only `pad_to_square` used
- [ ] Augmentation never applied to val or test splits
- [ ] All three models use the same `SurfaceCNN` class
- [ ] `torch.manual_seed(42)` and `np.random.seed(42)` present in `train.py`
- [ ] Checkpoints saved by lowest val loss (not val accuracy)
- [ ] Preprocessing is stateless and deterministic (same image → same tensor always)

---

## Accuracy Troubleshooting (If Targets Not Met)

### Wood below 78%
- [ ] Run `evaluate.py` and inspect confusion matrix
- [ ] If clean→cracked confusion: lower Canny threshold (e.g. 40→30) in `preprocess()`
- [ ] If smudged→clean confusion: increase CLAHE `clip_limit` from 3.0 → 3.5
- [ ] If any class has low recall: manually increase its class weight in `train.py`
- [ ] Last resort: add a 5th conv block to `SurfaceCNN` for more capacity

### Tiles below 85%
- [ ] Inspect if cracked→smudged confusion is high (rust edges triggering crack signal)
- [ ] Try increasing Gaussian blur from 7 to 9 for tile edge map to suppress more grout
- [ ] Try Canny thresholds 100/180 for tiles to reduce false crack edges from rust

### Walls below 85%
- [ ] Inspect confusion matrix for clean→smudged confusion on subtle smudges
- [ ] Try CLAHE clip from 4.0 → 4.5 for walls
- [ ] Verify image 15 (subtle smudge) is in training set, not accidentally in val/test

---

## Final Deliverables Checklist

- [ ] `dataset.py` — complete, tested, deterministic
- [ ] `model.py` — complete, verified output shape `(batch, 3)`
- [ ] `train.py` — runs for all 3 surfaces, saves `.pth` checkpoints
- [ ] `evaluate.py` — produces classification report + confusion matrix PNG
- [ ] `predict.py` — single image inference with softmax output
- [ ] `outputs/tiles_best.pth` — trained tile model
- [ ] `outputs/wood_best.pth` — trained wood model
- [ ] `outputs/walls_best.pth` — trained walls model
- [ ] `outputs/tiles_confusion_matrix.png`
- [ ] `outputs/wood_confusion_matrix.png`
- [ ] `outputs/walls_confusion_matrix.png`
- [ ] Tiles test accuracy ≥ 85%
- [ ] Wood test accuracy ≥ 80%
- [ ] Walls test accuracy ≥ 85%
