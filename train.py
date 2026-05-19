"""Training script for the Surface Damage Detector.

Run per surface:
    python train.py --surface tiles
    python train.py --surface wood
    python train.py --surface walls
"""

import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CLASS_NAMES, SurfaceDataset
from layers import CrossEntropyLoss
from model import SurfaceCNN


SURFACE_CONFIGS = {
    # Batch size lowered from spec's 16 to 8 to fit the 4GB RTX 2050:
    # our from-scratch Conv2d uses im2col which materialises a large patch
    # tensor at batch=16 and exhausts VRAM. batch=8 trains comfortably.
    "tiles": {"epochs": 60,  "lr": 0.001,   "batch_size": 8, "patience": 15},
    # wood: epochs 100 + patience 25. Paired with the softer LR scheduler
    # (patience 15 / factor 0.7) and extra_block from the notebook flag, the
    # model has both the capacity and the runway to climb past the train-acc
    # 76% plateau seen on v2.
    "wood":  {"epochs": 100, "lr": 0.0005,  "batch_size": 8, "patience": 25},
    "walls": {"epochs": 60,  "lr": 0.001,   "batch_size": 8, "patience": 15},
}

SEED = 42
EARLY_STOP_PATIENCE = 15


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SurfaceCNN on one surface.")
    p.add_argument("--surface", required=True, choices=list(SURFACE_CONFIGS.keys()))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. 0 is safe on Windows.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Seed for both split shuffling and torch/numpy RNG.")
    p.add_argument("--no_augment", action="store_true",
                   help="Disable training augmentation (diagnostic).")
    p.add_argument("--smudge_boost", type=float, default=1.0,
                   help="Multiplier on the smudged class weight (default 1.0).")
    p.add_argument("--cracked_boost", type=float, default=1.0,
                   help="Multiplier on the cracked class weight (default 1.0).")
    p.add_argument("--dropout_fc", type=float, default=0.5,
                   help="Dropout rate on the FC head (default 0.5).")
    p.add_argument("--dropout2d_scale", type=float, default=1.0,
                   help="Scale factor on Dropout2d rates in conv blocks (default 1.0).")
    p.add_argument("--extra_block", action="store_true",
                   help="Add a 5th conv block (256->512). More capacity.")
    p.add_argument("--fc_hidden", type=int, default=128,
                   help="Hidden size of the FC head (default 128).")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing epsilon for CrossEntropyLoss (default 0.0). "
                        "0.05-0.10 is a typical range; helps when classes look similar.")
    p.add_argument("--balanced_sampler", action="store_true",
                   help="Use a WeightedRandomSampler so each class appears with equal "
                        "frequency in training batches. Helpful when one class is a "
                        "persistent bottleneck (e.g. smudged on wood).")
    args = p.parse_args()

    cfg = SURFACE_CONFIGS[args.surface]
    if args.epochs is None:
        args.epochs = cfg["epochs"]
    if args.lr is None:
        args.lr = cfg["lr"]
    if args.batch_size is None:
        args.batch_size = cfg["batch_size"]
    args.patience = cfg.get("patience", EARLY_STOP_PATIENCE)
    return args


def build_dataloaders(args):
    root = os.path.join(args.data_dir, args.surface)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    train_ds = SurfaceDataset(root, args.surface, "train",
                              augment=not args.no_augment, seed=args.seed)
    val_ds   = SurfaceDataset(root, args.surface, "val",   augment=False, seed=args.seed)
    test_ds  = SurfaceDataset(root, args.surface, "test",  augment=False, seed=args.seed)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found under {root}. "
                           "Populate clean/, smudged/, cracked/ folders first.")

    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    if args.balanced_sampler:
        # Each training sample gets weight inversely proportional to its class
        # count, then WeightedRandomSampler draws len(train_ds) samples per
        # epoch with replacement. Net effect: every class appears with roughly
        # equal frequency in the batches, regardless of dataset imbalance.
        counts = train_ds.class_counts()
        per_class_w = [1.0 / c if c > 0 else 0.0 for c in counts]
        sample_weights = [per_class_w[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, **common,
        )
        print(f"Using balanced sampler. Per-class draw weight: "
              f"{dict(zip(CLASS_NAMES, [round(w, 4) for w in per_class_w]))}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, **common,
        )

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def class_weights_from_train(train_ds, device, smudge_boost: float = 1.0,
                             cracked_boost: float = 1.0):
    counts = train_ds.class_counts()
    total = sum(counts)
    n_classes = len(counts)
    # Inverse frequency: rare classes get higher loss weight.
    weights = [total / (n_classes * c) if c > 0 else 0.0 for c in counts]
    # Optionally boost smudged (index 1) or cracked (index 2) class weight
    # so the model is penalised harder for missing that class. Useful when
    # one specific class is the bottleneck on test accuracy.
    if smudge_boost != 1.0:
        weights[1] = weights[1] * smudge_boost
    if cracked_boost != 1.0:
        weights[2] = weights[2] * cracked_boost
    print(f"Train class counts: {dict(zip(CLASS_NAMES, counts))}")
    print(f"Class weights:      {dict(zip(CLASS_NAMES, [round(w, 3) for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate_test(model, loader, device, num_classes: int = 3):
    model.eval()
    total_correct = 0
    total_samples = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(inputs).argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            for c in range(num_classes):
                mask = labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += ((preds == labels) & mask).sum().item()

    overall = total_correct / total_samples if total_samples else 0.0
    per_class = [
        (per_class_correct[c] / per_class_total[c]) if per_class_total[c] else 0.0
        for c in range(num_classes)
    ]
    return overall, per_class


class _Tee:
    """Mirror writes to multiple streams (stdout + a log file)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Mirror all training output to TWO files so nothing is ever lost:
    #   1. outputs/{surface}_seed{N}_train.log         <- latest run (overwritten)
    #   2. outputs/training_history/{surface}_seed{N}_{timestamp}.log
    #      <- permanent record (never overwritten, one per run)
    history_dir = os.path.join(args.output_dir, "training_history")
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_tag = "" if args.seed == SEED else f"_seed{args.seed}"
    latest_path = os.path.join(args.output_dir, f"{args.surface}{log_tag}_train.log")
    history_path = os.path.join(
        history_dir, f"{args.surface}{log_tag}_{timestamp}.log"
    )
    latest_file = open(latest_path, "w", encoding="utf-8")
    history_file = open(history_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, latest_file, history_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Surface: {args.surface} | epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} seed={args.seed}")
    print(f"Latest log:  {latest_path}")
    print(f"History log: {history_path}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(args)
    print(f"Dataset sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    weights = class_weights_from_train(
        train_ds, device,
        smudge_boost=args.smudge_boost,
        cracked_boost=args.cracked_boost,
    )
    criterion = CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing).to(device)
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")

    # Detect input channel count from the dataset (3 for engineered/rgb, 6 for hybrid).
    sample_tensor, _ = train_ds[0]
    in_channels = sample_tensor.shape[0]
    print(f"Input channels: {in_channels}")

    model = SurfaceCNN(
        num_classes=3,
        in_channels=in_channels,
        dropout_rate=args.dropout_fc,
        dropout2d_scale=args.dropout2d_scale,
        extra_block=args.extra_block,
        fc_hidden=args.fc_hidden,
    ).to(device)
    model.count_parameters()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Softer scheduler than the original (patience 7, factor 0.5) which
    # collapsed the wood LR 8x in 80 epochs and killed late-stage learning.
    # patience 15 + factor 0.7 + min_lr 1e-5 keeps the LR alive long enough
    # for the model to escape the train-acc=76% plateau seen on wood.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.7, min_lr=1e-5)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    # Non-default seed gets a seed-tagged checkpoint so verification runs
    # don't clobber the production checkpoint.
    if args.seed == SEED:
        ckpt_name = f"{args.surface}_best.pth"
    else:
        ckpt_name = f"{args.surface}_seed{args.seed}_best.pth"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}% | "
            f"LR: {lr_now:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs).")
                break

    # Final evaluation with best checkpoint.
    print(f"\nLoading best checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc, per_class_acc = evaluate_test(model, test_loader, device)

    print(f"\n=== TEST RESULTS: {args.surface} ===")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print("Per-class accuracy:")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"  {name:8s}: {acc*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
