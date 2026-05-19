import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from surface_model import SURFACE_NAMES, SURFACE_TO_IDX

VALID_EXTS = (".jpg", ".jpeg", ".png")
TARGET_SIZE = 128


def pad_to_square(img, target=TARGET_SIZE):
    if img.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got shape {img.shape}")
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w = target - new_w
    pad_h = target - new_h
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=0,
    ).astype(np.uint8)


def preprocess(img_bgr, target_size=TARGET_SIZE):
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    padded = pad_to_square(rgb, target_size)
    return padded.transpose(2, 0, 1).astype(np.float32) / 255.0


def augment_bgr(img_bgr, rng):
    h, w = img_bgr.shape[:2]

    if rng.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
    if rng.random() < 0.3:
        img_bgr = cv2.flip(img_bgr, 0)

    angle = rng.uniform(-20.0, 20.0)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    contrast   = rng.uniform(0.75, 1.25)
    brightness = rng.uniform(-30.0, 30.0)
    img_bgr = np.clip(
        img_bgr.astype(np.float32) * contrast + brightness, 0, 255
    ).astype(np.uint8)

    if rng.random() < 0.4:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.6, 1.4), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.75, 1.25), 0, 255)
        img_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.3:
        ksize = rng.choice([3, 5])
        img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    return img_bgr


def augment_bgr_tta(img_bgr):
    h, w   = img_bgr.shape[:2]
    views  = [img_bgr, cv2.flip(img_bgr, 1)]
    for angle in (10, -10):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        views.append(cv2.warpAffine(img_bgr, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT))
    views.append(np.clip(img_bgr.astype(np.float32) * 1.15 + 15, 0, 255).astype(np.uint8))
    views.append(np.clip(img_bgr.astype(np.float32) * 0.85 - 10, 0, 255).astype(np.uint8))
    return views


class SurfaceTypeDataset(Dataset):

    def __init__(self, root_dir, split,
                 split_ratios=(0.70, 0.15, 0.15),
                 target_size=TARGET_SIZE,
                 augment=False, seed=42, tta=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")
        if abs(sum(split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")

        self.root_dir    = root_dir
        self.split       = split
        self.target_size = target_size
        self.augment     = augment and split == "train"
        self.tta         = tta and split == "test"
        self.samples     = self._build_split(root_dir, split, split_ratios, seed)

    @staticmethod
    def _gather_paths(root_dir, surface_name):
        d = os.path.join(root_dir, surface_name)
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

        for surface_name in SURFACE_NAMES:
            paths = self._gather_paths(root_dir, surface_name)
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

            label = SURFACE_TO_IDX[surface_name]
            samples.extend((p, label) for p in chosen)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)

        if self.tta:
            views   = augment_bgr_tta(img_bgr)
            tensors = [torch.from_numpy(preprocess(v, self.target_size)) for v in views]
            return torch.stack(tensors, dim=0), label

        if self.augment:
            img_bgr = augment_bgr(img_bgr, random)

        return torch.from_numpy(preprocess(img_bgr, self.target_size)), label

    def class_counts(self):
        counts = [0] * len(SURFACE_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        return counts


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/surface_type"
    print(f"Checking dataset at: {root}")
    for split in ("train", "val", "test"):
        ds = SurfaceTypeDataset(root_dir=root, split=split, augment=(split == "train"))
        counts = ds.class_counts()
        print(f"  {split:5s}: {len(ds):4d} samples  "
              f"{dict(zip(SURFACE_NAMES, counts))}")
        if len(ds) > 0 and split == "train":
            t, lbl = ds[0]
            print(f"    tensor shape={tuple(t.shape)} dtype={t.dtype} label={lbl}")