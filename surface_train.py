import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from layers import CrossEntropyLoss
from surface_dataset import SURFACE_NAMES, SurfaceTypeDataset
from surface_model import SurfaceTypeCNN

SEED            = 42
EPOCHS          = 80
LR              = 0.001
BATCH_SIZE      = 16
PATIENCE        = 15
CHECKPOINT_PATH = "outputs/surface_type_best.pth"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=str,   default="data/surface_type")
    p.add_argument("--output_dir",  type=str,   default="outputs")
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--seed",        type=int,   default=SEED)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--dropout",     type=float, default=0.4)
    p.add_argument("--no_augment",  action="store_true")
    p.add_argument("--no_tta",      action="store_true")
    return p.parse_args()


def build_dataloaders(args, tta=False):
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"Surface type dataset not found at '{args.data_dir}'.\n"
            f"Create folders: {args.data_dir}/wood/, {args.data_dir}/tiles/, {args.data_dir}/walls/\n"
            f"and copy all images of each surface type into the matching folder."
        )

    train_ds = SurfaceTypeDataset(args.data_dir, "train",
                                  augment=not args.no_augment, seed=args.seed)
    val_ds   = SurfaceTypeDataset(args.data_dir, "val",
                                  augment=False, seed=args.seed)
    test_ds  = SurfaceTypeDataset(args.data_dir, "test",
                                  augment=False, seed=args.seed, tta=tta)

    if len(train_ds) == 0:
        raise RuntimeError(f"No images found under {args.data_dir}.")

    common = dict(num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=1,               shuffle=False, **common)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = total_correct = total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss    += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def evaluate_test(model, loader, device, use_tta=False):
    model.eval()
    total_correct     = 0
    total_samples     = 0
    n                 = len(SURFACE_NAMES)
    per_class_correct = [0] * n
    per_class_total   = [0] * n

    with torch.no_grad():
        for inputs, labels in loader:
            labels = labels.to(device, non_blocking=True)

            if use_tta:
                views      = inputs.squeeze(0).to(device, non_blocking=True)
                logits_sum = sum(model(views[v].unsqueeze(0)) for v in range(views.shape[0]))
                preds      = logits_sum.argmax(dim=1)
            else:
                preds = model(inputs.to(device, non_blocking=True)).argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            for c in range(n):
                mask = labels == c
                per_class_total[c]   += mask.sum().item()
                per_class_correct[c] += ((preds == labels) & mask).sum().item()

    overall   = total_correct / total_samples if total_samples else 0.0
    per_class = [
        per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0.0
        for c in range(n)
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
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path  = os.path.join(args.output_dir, "surface_type_train.log")
    history_path = os.path.join(history_dir, f"surface_type_{timestamp}.log")
    latest_file  = open(latest_path,  "w", encoding="utf-8")
    history_file = open(history_path, "w", encoding="utf-8")
    sys.stdout   = _Tee(sys.__stdout__, latest_file, history_file)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = not args.no_tta

    print(f"Device: {device}")
    print(f"Surface type classifier | epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch_size} seed={args.seed} TTA={use_tta}")

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders(
        args, tta=use_tta
    )
    print(f"Dataset sizes -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"Class counts (train): {dict(zip(SURFACE_NAMES, train_ds.class_counts()))}")

    criterion = CrossEntropyLoss().to(device)

    sample_tensor, _ = train_ds[0]
    in_channels = sample_tensor.shape[0]
    print(f"Input channels: {in_channels}")

    model = SurfaceTypeCNN(
        num_classes=3,
        in_channels=in_channels,
        dropout_rate=args.dropout,
    ).to(device)
    model.count_parameters()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.5, min_lr=1e-6)

    ckpt_path         = os.path.join(args.output_dir, "surface_type_best.pth")
    best_val_loss     = float("inf")
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion,
                                          None, device, train=False)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.1f}% | "
            f"LR: {lr_now:.6f}"
        )

        improved = False
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            improved = True
        if val_acc > best_val_acc + 0.005:
            best_val_acc = val_acc
            improved = True

        if improved:
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    print(f"\nLoading best checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    test_acc, per_class_acc = evaluate_test(model, test_loader, device, use_tta=use_tta)
    print(f"\n=== TEST RESULTS: Surface Type Classifier ===")
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    print("Per-class accuracy:")
    for name, acc in zip(SURFACE_NAMES, per_class_acc):
        print(f"  {name:6s}: {acc*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())