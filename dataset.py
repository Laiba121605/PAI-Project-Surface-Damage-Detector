import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES  = ["clean", "smudged", "cracked"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
VALID_EXTS   = (".jpg", ".jpeg", ".png")

SURFACE_PARAMS = {
    "tiles": {"mode": "hybrid", "clip": 2.5, "tile": (8, 8), "blur": 5, "canny": (40, 100)},
    "wood":  {"mode": "hybrid", "clip": 2.0, "tile": (6, 6), "blur": 3, "canny": (25, 80)},
    "walls": {"mode": "hybrid", "clip": 3.0, "tile": (8, 8), "blur": 5, "canny": (40, 100)},
}

TARGET_SIZE = 128


def pad_to_square(img, target=TARGET_SIZE):
    if img.ndim not in (2, 3):
        raise ValueError(f"pad_to_square expects 2D or 3D array, got shape {img.shape}")
    h, w  = img.shape[:2]
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w  = target - new_w
    pad_h  = target - new_h
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    ).astype(np.uint8)


def get_saturation(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]


def get_clahe_gray(img_bgr, clip_limit, tile_grid):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)


def get_edge_map(gray_clahe, blur_ksize, canny_t1, canny_t2):
    blurred = cv2.GaussianBlur(gray_clahe, (blur_ksize, blur_ksize), 0)
    edges   = cv2.Canny(blurred, canny_t1, canny_t2)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel),
        cv2.MORPH_OPEN, kernel
    )


def preprocess(img_bgr, surface, target_size=TARGET_SIZE):
    if surface not in SURFACE_PARAMS:
        raise ValueError(f"Unknown surface '{surface}'.")
    p    = SURFACE_PARAMS[surface]
    mode = p.get("mode", "engineered")

    if mode == "rgb":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return pad_to_square(rgb, target_size).transpose(2, 0, 1).astype(np.float32) / 255.0

    sat        = get_saturation(img_bgr)
    clahe_gray = get_clahe_gray(img_bgr, p["clip"], p["tile"])
    edges      = get_edge_map(clahe_gray, p["blur"], p["canny"][0], p["canny"][1])

    if mode == "hybrid":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        engineered = np.stack([
            pad_to_square(sat,        target_size),
            pad_to_square(clahe_gray, target_size),
            pad_to_square(edges,      target_size),
        ], axis=0)
        rgb_chw = pad_to_square(rgb, target_size).transpose(2, 0, 1)
        return np.concatenate([engineered, rgb_chw], axis=0).astype(np.float32) / 255.0

    return np.stack([
        pad_to_square(sat,        target_size),
        pad_to_square(clahe_gray, target_size),
        pad_to_square(edges,      target_size),
    ], axis=0).astype(np.float32) / 255.0


def augment_bgr(img_bgr, rng, surface="wood", label=None):
    h, w = img_bgr.shape[:2]

    if rng.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
    if rng.random() < 0.4:
        img_bgr = cv2.flip(img_bgr, 0)

    angle = rng.uniform(-25.0, 25.0)
    M     = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    if rng.random() < 0.5:
        scale  = rng.uniform(0.8, 1.0)
        crop_h = int(h * scale)
        crop_w = int(w * scale)
        y0     = rng.randint(0, max(1, h - crop_h))
        x0     = rng.randint(0, max(1, w - crop_w))
        img_bgr = cv2.resize(
            img_bgr[y0:y0 + crop_h, x0:x0 + crop_w],
            (w, h), interpolation=cv2.INTER_LINEAR
        )

    contrast   = rng.uniform(0.7, 1.35)
    brightness = rng.uniform(-35.0, 35.0)
    img_bgr = np.clip(
        img_bgr.astype(np.float32) * contrast + brightness, 0, 255
    ).astype(np.uint8)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    is_wood_smudge = (label == CLASS_TO_IDX["smudged"] and surface == "wood")

    if is_wood_smudge:
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + rng.uniform(-12, 12), 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.35, 1.9), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.60, 1.45), 0, 255)
    elif rng.random() < 0.6:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.6, 1.5), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.75, 1.25), 0, 255)

    img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.4:
        ksize   = rng.choice([3, 5])
        img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    if rng.random() < 0.35:
        noise   = np.random.normal(0, rng.uniform(3, 14), img_bgr.shape).astype(np.float32)
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img_bgr


def augment_bgr_tta(img_bgr):
    h, w  = img_bgr.shape[:2]
    views = [img_bgr, cv2.flip(img_bgr, 1)]
    for angle in (12, -12):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        views.append(cv2.warpAffine(img_bgr, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT))
    views.append(np.clip(img_bgr.astype(np.float32) * 1.15 + 15, 0, 255).astype(np.uint8))
    views.append(np.clip(img_bgr.astype(np.float32) * 0.85 - 10, 0, 255).astype(np.uint8))
    return views


class SurfaceDataset(Dataset):

    def __init__(self, root_dir, surface, split,
                 split_ratios=(0.70, 0.15, 0.15),
                 target_size=TARGET_SIZE,
                 augment=False, seed=42, tta=False):
        if surface not in SURFACE_PARAMS:
            raise ValueError(f"surface must be one of {list(SURFACE_PARAMS)}, got '{surface}'")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")

        self.surface     = surface
        self.split       = split
        self.target_size = target_size
        self.augment     = augment and split == "train"
        self.tta         = tta and split == "test"
        self.samples     = self._build_split(root_dir, split, split_ratios, seed)

    @staticmethod
    def _gather_class_paths(root_dir, class_name):
        d = os.path.join(root_dir, class_name)
        if not os.path.isdir(d):
            return []
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.lower().endswith(VALID_EXTS)]
        files.sort()
        return files

    def _build_split(self, root_dir, split, split_ratios, seed):
        train_r, val_r, _ = split_ratios
        rng     = random.Random(seed)
        samples = []
        for class_name in CLASS_NAMES:
            paths = self._gather_class_paths(root_dir, class_name)
            if not paths:
                continue
            shuffled = list(paths)
            rng.shuffle(shuffled)
            n       = len(shuffled)
            n_train = int(n * train_r)
            n_val   = int(n * val_r)
            if split == "train":
                chosen = shuffled[:n_train]
            elif split == "val":
                chosen = shuffled[n_train:n_train + n_val]
            else:
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

        if self.tta:
            views   = augment_bgr_tta(img_bgr)
            tensors = [torch.from_numpy(preprocess(v, self.surface, self.target_size))
                       for v in views]
            return torch.stack(tensors, dim=0), label

        if self.augment:
            img_bgr = augment_bgr(img_bgr, random, surface=self.surface, label=label)

        return torch.from_numpy(preprocess(img_bgr, self.surface, self.target_size)), label

    def class_counts(self):
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        return counts


if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "data")
    for surface in ("tiles", "wood", "walls"):
        surface_root = os.path.join(data_root, surface)
        print(f"\n--- {surface} ---")
        try:
            total = 0
            for split in ("train", "val", "test"):
                ds = SurfaceDataset(root_dir=surface_root, surface=surface,
                                    split=split, augment=(split == "train"))
                print(f"  {split:5s}: {len(ds):4d} samples  per-class={ds.class_counts()}")
                total += len(ds)
                if len(ds) > 0 and split == "train":
                    t, lbl = ds[0]
                    print(f"    tensor shape={tuple(t.shape)} dtype={t.dtype} label={lbl}")
            print(f"  total: {total}")
        except Exception as exc:
            print(f"  skipped ({exc})")