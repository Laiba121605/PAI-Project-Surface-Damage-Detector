"""Dataset and preprocessing for the Surface Damage Detector.

Per-surface 3-channel pipeline (see CLAUDE.md for rationale):
    ch0 = Saturation map      -> smudge color signal
    ch1 = CLAHE grayscale     -> normalized texture / brightness
    ch2 = Canny + Morph edges -> crack structural signal
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ["clean", "smudged", "cracked"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
VALID_EXTS = (".jpg", ".jpeg", ".png")

SURFACE_PARAMS = {
    # "mode": "engineered" -> 3-channel: Saturation, CLAHE-gray, Canny edges.
    # "mode": "rgb"        -> 3-channel: raw RGB (R, G, B), normalised /255.
    # "mode": "hybrid"     -> 6-channel: Sat, CLAHE, Edges, R, G, B. Gives the
    #                         CNN both the engineered crack/smudge signals
    #                         AND raw color/texture so it can pick what helps.
    #
    # tiles: engineered alone plateaued at ~62% train acc (no aug). RGB alone
    # killed crack detection (cracked recall = 0%). Hybrid keeps Canny for
    # cracks and adds RGB for the model to learn its own smudge/clean cues.
    "tiles": {"mode": "hybrid", "clip": 2.5, "tile": (8, 8), "blur": 5, "canny": (40, 100)},
    "wood":  {"mode": "engineered", "clip": 3.0, "tile": (8, 8), "blur": 5, "canny": (40, 100)},
    # walls: engineered preprocessing got 84.2%/91.2%/78.9% across 3 seeds
    # but smudged was the persistent weak class (61% on seeds 42 and 2024).
    # Switching to 6-channel hybrid so the CNN gets raw RGB alongside the
    # engineered channels - smudges on walls (rust, mold, water stains) are
    # color-distinctive and RGB should help the model see them better.
    "walls": {"mode": "hybrid", "clip": 3.0, "tile": (8, 8), "blur": 5, "canny": (40, 100)},
}


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def pad_to_square(img, target=256):
    """Scale longest side to `target`, then zero-pad shorter side to square.

    Preserves aspect ratio. No cropping. Works for (H, W) and (H, W, 3).
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"pad_to_square expects 2D or 3D array, got shape {img.shape}")

    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target - new_w
    pad_h = target - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    )
    return padded.astype(np.uint8)


def get_saturation(img_bgr):
    """Return HSV S-channel as uint8 (H, W)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def get_clahe_gray(img_bgr, clip_limit, tile_grid):
    """Grayscale + CLAHE. Normalizes local contrast (uint8 (H, W))."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def get_edge_map(gray_clahe, blur_ksize, canny_t1, canny_t2):
    """Blur -> Canny -> morphological CLOSE then OPEN. Returns uint8 binary."""
    blurred = cv2.GaussianBlur(gray_clahe, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, canny_t1, canny_t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def preprocess(img_bgr, surface, target_size=256):
    """Build the (3, target, target) float32 tensor in [0, 1].

    Dispatches by surface:
      mode == "engineered" -> Saturation + CLAHE-gray + Canny-edges.
      mode == "rgb"        -> raw RGB channels (BGR converted to RGB).
    """
    if surface not in SURFACE_PARAMS:
        raise ValueError(f"Unknown surface '{surface}'. Expected one of {list(SURFACE_PARAMS)}.")

    p = SURFACE_PARAMS[surface]

    mode = p.get("mode", "engineered")

    if mode == "rgb":
        # OpenCV loads BGR; convert to RGB so channel 0 = Red as humans expect.
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        padded = pad_to_square(rgb, target_size)  # (target, target, 3) uint8
        # (H, W, C) -> (C, H, W) and normalise to [0, 1].
        return padded.transpose(2, 0, 1).astype(np.float32) / 255.0

    sat = get_saturation(img_bgr)
    clahe_gray = get_clahe_gray(img_bgr, p["clip"], p["tile"])
    edges = get_edge_map(clahe_gray, p["blur"], p["canny"][0], p["canny"][1])

    if mode == "hybrid":
        # 6-channel: engineered (sat, clahe, edges) + raw RGB.
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ch0 = pad_to_square(sat,        target_size)
        ch1 = pad_to_square(clahe_gray, target_size)
        ch2 = pad_to_square(edges,      target_size)
        ch_rgb = pad_to_square(rgb,     target_size)  # (T, T, 3)
        # Stack channels-first: (6, T, T).
        engineered = np.stack([ch0, ch1, ch2], axis=0)
        rgb_chw    = ch_rgb.transpose(2, 0, 1)
        return np.concatenate([engineered, rgb_chw], axis=0).astype(np.float32) / 255.0

    ch0 = pad_to_square(sat, target_size)
    ch1 = pad_to_square(clahe_gray, target_size)
    ch2 = pad_to_square(edges, target_size)

    stacked = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32) / 255.0
    return stacked


# ---------------------------------------------------------------------------
# Augmentation (train split only)
# ---------------------------------------------------------------------------
# Applied to the raw BGR image BEFORE preprocessing, so that Canny/CLAHE/
# saturation channels are recomputed from the augmented image. Rotating an
# already-computed binary edge map (as torchvision.transforms would) creates
# antialiased non-binary values that have no real-world correspondence and
# pollute the crack signal.

def augment_bgr(img_bgr, rng):
    """Random horiz/vert flip + small rotation + brightness/contrast jitter.

    Operates on a BGR uint8 numpy array. Uses BORDER_REFLECT for rotation so
    we do not introduce black borders that Canny would later see as edges.
    """
    h, w = img_bgr.shape[:2]

    if rng.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
    if rng.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 0)

    angle = rng.uniform(-15.0, 15.0)
    if abs(angle) > 0.1:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_bgr = cv2.warpAffine(
            img_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

    # Brightness (additive offset in [-30, 30]) + contrast (multiplicative
    # factor in [0.8, 1.2]) on raw pixel values. Simulates lighting variation.
    contrast = rng.uniform(0.8, 1.2)
    brightness = rng.uniform(-30.0, 30.0)
    img_bgr = np.clip(img_bgr.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    return img_bgr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurfaceDataset(Dataset):
    """Custom dataset that performs surface-specific preprocessing on the fly."""

    def __init__(
        self,
        root_dir,
        surface,
        split,
        split_ratios=(0.70, 0.15, 0.15),
        target_size=256,
        augment=False,
        seed=42,
    ):
        if surface not in SURFACE_PARAMS:
            raise ValueError(f"surface must be one of {list(SURFACE_PARAMS)}, got '{surface}'")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of train/val/test, got '{split}'")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

        self.root_dir = root_dir
        self.surface = surface
        self.split = split
        self.target_size = target_size
        self.augment = augment and split == "train"

        self.samples = self._build_split(root_dir, split, split_ratios, seed)

    @staticmethod
    def _gather_class_paths(root_dir, class_name):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            return []
        files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(VALID_EXTS)
        ]
        files.sort()
        return files

    def _build_split(self, root_dir, split, split_ratios, seed):
        train_r, val_r, _ = split_ratios
        rng = random.Random(seed)
        samples = []

        for class_name in CLASS_NAMES:
            paths = self._gather_class_paths(root_dir, class_name)
            if not paths:
                continue
            # Deterministic shuffle per class given fixed seed.
            shuffled = list(paths)
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(n * train_r)
            n_val = int(n * val_r)

            if split == "train":
                chosen = shuffled[:n_train]
            elif split == "val":
                chosen = shuffled[n_train:n_train + n_val]
            else:  # test gets the remainder so all images are used
                chosen = shuffled[n_train + n_val:]

            label = CLASS_TO_IDX[class_name]
            samples.extend((p, label) for p in chosen)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Failed to read image: {path}")

        if self.augment:
            img_bgr = augment_bgr(img_bgr, random)

        arr = preprocess(img_bgr, self.surface, self.target_size)
        tensor = torch.from_numpy(arr)

        return tensor, label

    def class_counts(self):
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        return counts


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "data")
    for surface in ("tiles", "wood", "walls"):
        surface_root = os.path.join(data_root, surface)
        print(f"\n--- {surface} ---")
        try:
            total = 0
            for split in ("train", "val", "test"):
                ds = SurfaceDataset(
                    root_dir=surface_root,
                    surface=surface,
                    split=split,
                    augment=(split == "train"),
                )
                print(f"  {split:5s}: {len(ds):4d} samples  per-class={ds.class_counts()}")
                total += len(ds)
                if len(ds) > 0 and split == "train":
                    tensor, label = ds[0]
                    print(
                        f"    sample tensor: shape={tuple(tensor.shape)} "
                        f"dtype={tensor.dtype} min={tensor.min():.3f} max={tensor.max():.3f} "
                        f"label={label}"
                    )
            print(f"  total: {total}")
        except Exception as exc:  # noqa: BLE001
            print(f"  skipped ({exc})")
