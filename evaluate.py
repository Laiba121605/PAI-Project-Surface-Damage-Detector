"""Evaluation script for a trained Surface Damage model.

Run per surface:
    python evaluate.py --surface tiles
    python evaluate.py --surface wood
    python evaluate.py --surface walls
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, SurfaceDataset
from model import SurfaceCNN, infer_config_from_state_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained SurfaceCNN.")
    p.add_argument("--surface", required=True, choices=("tiles", "wood", "walls"))
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--tta", action="store_true",
                   help="Test-time augmentation: predict on 4 oriented views, "
                        "average softmax probabilities. Free 1-3pp accuracy gain.")
    return p.parse_args()


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def predict_with_tta(model, inputs: torch.Tensor) -> torch.Tensor:
    """Average softmax probabilities over 4 oriented views of each image.

    Views: original, horizontal flip, vertical flip, 180-degree rotation.
    All 4 are valid orientations for surface photos (no preferred top/bottom).
    """
    views = [
        inputs,
        torch.flip(inputs, dims=[3]),       # horizontal flip
        torch.flip(inputs, dims=[2]),       # vertical flip
        torch.flip(inputs, dims=[2, 3]),    # 180 rotation
    ]
    probs_list = [softmax(model(v), dim=1) for v in views]
    return torch.stack(probs_list).mean(dim=0)


def print_confusion(cm: np.ndarray) -> None:
    print("              Predicted")
    print("              " + "  ".join(f"{n:>7s}" for n in CLASS_NAMES))
    actual_labels = [f"Actual {CLASS_NAMES[0]}", f"     {CLASS_NAMES[1]}", f"     {CLASS_NAMES[2]}"]
    for label, row in zip(actual_labels, cm):
        cells = "  ".join(f"{v:7d}" for v in row)
        print(f"{label:>14s} [{cells} ]")


def save_confusion_plot(cm: np.ndarray, surface: str, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{surface} - Confusion Matrix")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{surface}_confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = os.path.join(args.output_dir, f"{args.surface}_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    root = os.path.join(args.data_dir, args.surface)
    test_ds = SurfaceDataset(root, args.surface, "test", augment=False)
    if len(test_ds) == 0:
        raise RuntimeError(f"No test samples found under {root}.")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test set size: {len(test_ds)}")

    state_dict = torch.load(ckpt_path, map_location=device)
    cfg = infer_config_from_state_dict(state_dict)
    print(f"Detected model config: {cfg}")
    model = SurfaceCNN(num_classes=3, **cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.tta:
        print("Using TTA (4-view averaging)")
    all_true, all_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            if args.tta:
                probs = predict_with_tta(model, inputs)
                preds = probs.argmax(dim=1).cpu().numpy()
            else:
                preds = model(inputs).argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy().tolist())
            all_pred.extend(preds.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall = (all_true == all_pred).mean()
    print(f"\nOverall Accuracy: {overall*100:.2f}%")

    print("\nClassification report:")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CLASS_NAMES))))
    print("Confusion matrix:")
    print_confusion(cm)

    out_path = save_confusion_plot(cm, args.surface, args.output_dir)
    print(f"\nSaved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
