"""Single-image inference for a trained Surface Damage model.

Run:
    python predict.py --surface tiles --image data/tiles/clean/img001.jpg
"""

import argparse
import os

import cv2
import torch

from dataset import CLASS_NAMES, preprocess
from model import SurfaceCNN


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically-stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict surface damage class for one image.")
    p.add_argument("--surface", required=True, choices=("tiles", "wood", "walls"))
    p.add_argument("--image", required=True, type=str)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    ckpt_path = os.path.join(args.output_dir, f"{args.surface}_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Could not read image (corrupt or unsupported format): {args.image}")

    arr = preprocess(img_bgr, args.surface, target_size=256)
    tensor = torch.from_numpy(arr).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = tensor.shape[1]
    model = SurfaceCNN(num_classes=3, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    pred_name = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100.0

    print(f"Surface:    {args.surface}")
    print(f"Image:      {args.image}")
    print(f"Prediction: {pred_name} (confidence: {confidence:.1f}%)")
    scores = "  ".join(f"{name}={p*100:.1f}%" for name, p in zip(CLASS_NAMES, probs))
    print(f"All scores: {scores}")


if __name__ == "__main__":
    main()
