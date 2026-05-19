import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import CLASS_NAMES, SurfaceDataset
from layers import CrossEntropyLoss
from model import SurfaceCNN


SURFACE_CONFIGS = {
    "tiles": {"epochs": 150, "lr": 0.001,  "batch_size": 16},
    "wood":  {"epochs": 150, "lr": 0.0005, "batch_size": 8},
    "walls": {"epochs": 150, "lr": 0.001,  "batch_size": 16},
}

SURFACE_DEFAULT_BOOSTS = {
    "tiles": {"smudge_boost": 1.0, "cracked_boost": 1.0},
    "wood":  {"smudge_boost": 4.0, "cracked_boost": 1.2},
    "walls": {"smudge_boost": 1.0, "cracked_boost": 1.0},
}

SURFACE_MIXUP = {
    "tiles": 0.2,
    "wood":  0.0,
    "walls": 0.2,
}

SEED = 42
EARLY_STOP_PATIENCE = 35


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--surface",         required=True, choices=list(SURFACE_CONFIGS.keys()))
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--data_dir",        type=str,   default="data")
    p.add_argument("--output_dir",      type=str,   default="outputs")
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--seed",            type=int,   default=SEED)
    p.add_argument("--no_augment",      action="store_true")
    p.add_argument("--smudge_boost",    type=float, default=None)
    p.add_argument("--cracked_boost",   type=float, default=None)
    p.add_argument("--dropout_fc",      type=float, default=0.25)
    p.add_argument("--dropout2d_scale", type=float, default=0.5)
    p.add_argument("--extra_block",     action="store_true")
    p.add_argument("--fc_hidden",       type=int,   default=256)
    p.add_argument("--mixup_alpha",     type=float, default=None)
    p.add_argument("--no_tta",          action="store_true")
    args = p.parse_args()

    cfg = SURFACE_CONFIGS[args.surface]
    if args.epochs     is None: args.epochs     = cfg["epochs"]
    if args.lr         is None: args.lr         = cfg["lr"]
    if args.batch_size is None: args.batch_size = cfg["batch_size"]

    defaults = SURFACE_DEFAULT_BOOSTS[args.surface]
    if args.smudge_boost  is None: args.smudge_boost  = defaults["smudge_boost"]
    if args.cracked_boost is None: args.cracked_boost = defaults["cracked_boost"]
    if args.mixup_alpha   is None: args.mixup_alpha   = SURFACE_MIXUP[args.surface]
    return args


def build_dataloaders(args, tta=False):
    root = os.path.join(args.data_dir, args.surface)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    train_ds = SurfaceDataset(root, args.surface, "train",
                              augment=not args.no_augment, seed=args.seed)
    val_ds   = SurfaceDataset(root, args.surface, "val",
                              augment=False, seed=args.seed)
    test_ds  = SurfaceDataset(root, args.surface, "test",
                              augment=False, seed=args.seed, tta=tta)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training images found under {root}.")

    smudged_idx = CLASS_NAMES.index("smudged")
    per_sample_weights = [
        3.0 if lbl == smudged_idx else 1.0
        for _, lbl in train_ds.samples
    ]
    sampler = WeightedRandomSampler(
        weights=per_sample_weights,
        num_samples=len(per_sample_weights),
        replacement=True
    )

    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, **common)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, **common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def class_weights_from_train(train_ds, device, smudge_boost=1.0, cracked_boost=1.0):
    counts = train_ds.class_counts()
    total  = sum(counts)
    n      = len(counts)
    w = [total / (n * c) if c > 0 else 0.0 for c in counts]
    smudged_idx = CLASS_NAMES.index("smudged")
    cracked_idx = CLASS_NAMES.index("cracked")
    if smudge_boost  != 1.0: w[smudged_idx] *= smudge_boost
    if cracked_boost != 1.0: w[cracked_idx] *= cracked_boost
    print(f"Train class counts: {dict(zip(CLASS_NAMES, counts))}")
    print(f"Class weights:      {dict(zip(CLASS_NAMES, [round(x, 3) for x in w]))}")
    return torch.tensor(w, dtype=torch.float32, device=device)


def mixup_batch(inputs, labels, alpha, device):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(inputs.size(0), device=device)
    mixed = lam * inputs + (1.0 - lam) * inputs[idx]
    return mixed, labels, labels[idx], lam


def run_epoch(model, loader, criterion, optimizer, device, train, mixup_alpha=0.0):
    model.train() if train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train and mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = mixup_batch(inputs, labels, mixup_alpha, device)
                if optimizer: optimizer.zero_grad()
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                preds = outputs.argmax(dim=1)
                total_correct += (lam * (preds == labels_a).float() +
                                  (1.0 - lam) * (preds == labels_b).float()).sum().item()
            else:
                if train and optimizer: optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()

            total_loss    += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate_test(model, loader, device, num_classes=3, use_tta=False):
    model.eval()
    total_correct     = 0
    total_samples     = 0
    per_class_correct = [0] * num_classes
    per_class_total   = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in loader:
            labels = labels.to(device, non_blocking=True)

            if use_tta:
                views = inputs.squeeze(0).to(device, non_blocking=True)
                logits_sum = sum(model(views[v].unsqueeze(0)) for v in range(views.shape[0]))
                preds = logits_sum.argmax(dim=1)
            else:
                preds = model(inputs.to(device, non_blocking=True)).argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            for c in range(num_classes):
                mask = labels == c
                per_class_total[c]   += mask.sum().item()
                per_class_correct[c] += ((preds == labels) & mask).sum().item()

    overall = total_correct / total_samples if total_samples else 0.0
    per_class = [
        per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0.0
        for c in range(num_classes)
    ]
    return overall, per_class


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    history_dir = os.path.join(args.output_dir, "training_history")
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_tag      = "" if args.seed == SEED else f"_seed{args.seed}"
    latest_path  = os.path.join(args.output_dir, f"{args.surface}{log_tag}_train.log")
    history_path = os.path.join(history_dir, f"{args.surface}{log_tag}_{timestamp}.log")
    latest_file  = open(latest_path,  "w", encoding="utf-8")
    history_file = open(history_path, "w", encoding="utf-8")
    sys.stdout   = _Tee(sys.__stdout__, latest_file, history_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Surface: {args.surface} | epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} seed={args.seed}")
    print(f"Smudge boost: {args.smudge_boost} | Cracked boost: {args.cracked_boost}")
    print(f"Mixup alpha: {args.mixup_alpha} | TTA: {not args.no_tta}")

    use_tta = not args.no_tta
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        args, tta=use_tta
    )
    print(f"Dataset sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    weights   = class_weights_from_train(train_ds, device,
                                         smudge_boost=args.smudge_boost,
                                         cracked_boost=args.cracked_boost)
    criterion = CrossEntropyLoss(weight=weights).to(device)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc      = 0.0
    epochs_no_improve = 0

    ckpt_name = (f"{args.surface}_best.pth" if args.seed == SEED
                 else f"{args.surface}_seed{args.seed}_best.pth")
    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, mixup_alpha=args.mixup_alpha
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False
        )
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}% | "
            f"LR: {lr_now:.6f}"
        )

        if val_acc > best_val_acc + 0.005:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_acc improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    print(f"\nLoading best checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    test_acc, per_class_acc = evaluate_test(model, test_loader, device, use_tta=use_tta)

    print(f"\n=== TEST RESULTS: {args.surface} ===")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print("Per-class accuracy:")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"  {name:8s}: {acc*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())