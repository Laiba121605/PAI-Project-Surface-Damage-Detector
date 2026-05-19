import argparse
import os
import sys

import cv2
import numpy as np
import torch

from dataset import SURFACE_PARAMS, preprocess as damage_preprocess
from model import SurfaceCNN, infer_config_from_state_dict
from surface_dataset import TARGET_SIZE, augment_bgr_tta
from surface_dataset import preprocess as surface_preprocess
from surface_model import SURFACE_NAMES, SurfaceTypeCNN
from surface_model import infer_config_from_state_dict as surface_infer_config

DAMAGE_NAMES  = ["clean", "smudged", "cracked"]
DAMAGE_MODELS = {
    "wood":  "outputs/wood_best.pth",
    "tiles": "outputs/tiles_best.pth",
    "walls": "outputs/walls_best.pth",
}
SURFACE_TYPE_MODEL = "outputs/surface_type_best.pth"


def load_surface_type_model(ckpt_path, device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Surface type model not found at '{ckpt_path}'.\n"
            f"Run: python surface_train.py"
        )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg   = surface_infer_config(state_dict)
    model = SurfaceTypeCNN(**cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_damage_model(surface, device):
    ckpt_path = DAMAGE_MODELS[surface]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Damage model for '{surface}' not found at '{ckpt_path}'.\n"
            f"Run: python train.py --surface {surface}"
        )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg   = infer_config_from_state_dict(state_dict)
    model = SurfaceCNN(num_classes=3, **cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_surface_type(model, img_bgr, device, use_tta=True):
    if use_tta:
        views      = augment_bgr_tta(img_bgr)
        tensors    = [torch.from_numpy(surface_preprocess(v, TARGET_SIZE)) for v in views]
        batch      = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits_sum = sum(model(batch[v].unsqueeze(0)) for v in range(batch.shape[0]))
        probs = torch.softmax(logits_sum, dim=1).squeeze(0)
    else:
        tensor = torch.from_numpy(surface_preprocess(img_bgr, TARGET_SIZE)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    pred_idx  = probs.argmax().item()
    pred_name = SURFACE_NAMES[pred_idx]
    confidence = probs[pred_idx].item()
    return pred_name, confidence, probs.cpu().tolist()


def predict_damage(model, img_bgr, surface, device, use_tta=True):
    if use_tta:
        views   = augment_bgr_tta(img_bgr)
        tensors = [torch.from_numpy(damage_preprocess(v, surface, TARGET_SIZE)) for v in views]
        batch   = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits_sum = sum(model(batch[v].unsqueeze(0)) for v in range(batch.shape[0]))
        probs = torch.softmax(logits_sum, dim=1).squeeze(0)
    else:
        tensor = torch.from_numpy(
            damage_preprocess(img_bgr, surface, TARGET_SIZE)
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    pred_idx   = probs.argmax().item()
    pred_name  = DAMAGE_NAMES[pred_idx]
    confidence = probs[pred_idx].item()
    return pred_name, confidence, probs.cpu().tolist()


def predict_image(image_path, device, use_tta=True,
                  surface_type_ckpt=SURFACE_TYPE_MODEL,
                  damage_ckpts=None):
    if damage_ckpts is None:
        damage_ckpts = DAMAGE_MODELS

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Cannot read image: {image_path}")

    surface_model = load_surface_type_model(surface_type_ckpt, device)
    surface_type, surface_conf, surface_probs = predict_surface_type(
        surface_model, img_bgr, device, use_tta=use_tta
    )

    damage_model = load_damage_model(surface_type, device)
    damage_type, damage_conf, damage_probs = predict_damage(
        damage_model, img_bgr, surface_type, device, use_tta=use_tta
    )

    return {
        "image":          image_path,
        "surface_type":   surface_type,
        "surface_conf":   surface_conf,
        "surface_probs":  dict(zip(SURFACE_NAMES, surface_probs)),
        "damage_type":    damage_type,
        "damage_conf":    damage_conf,
        "damage_probs":   dict(zip(DAMAGE_NAMES, damage_probs)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Full pipeline: surface type -> damage prediction.")
    p.add_argument("image", type=str, help="Path to input image.")
    p.add_argument("--no_tta",   action="store_true", help="Disable TTA.")
    p.add_argument("--device",   type=str, default=None,
                   help="Force device: 'cpu' or 'cuda'. Auto-detected if omitted.")
    p.add_argument("--surface_model", type=str, default=SURFACE_TYPE_MODEL)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    result = predict_image(
        args.image, device,
        use_tta=not args.no_tta,
        surface_type_ckpt=args.surface_model,
    )

    print(f"\nImage     : {result['image']}")
    print(f"Surface   : {result['surface_type']}  (confidence: {result['surface_conf']*100:.1f}%)")
    print(f"  Probs   : { {k: f'{v*100:.1f}%' for k, v in result['surface_probs'].items()} }")
    print(f"Damage    : {result['damage_type']}  (confidence: {result['damage_conf']*100:.1f}%)")
    print(f"  Probs   : { {k: f'{v*100:.1f}%' for k, v in result['damage_probs'].items()} }")


if __name__ == "__main__":
    sys.exit(main())