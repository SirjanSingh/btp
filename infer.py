"""
infer.py — Single-image inference with rooftop mask + area estimation.

Usage:
    python infer.py \
        --image   /scratch/airs_crops/test/images/sample.png \
        --ckpt    /scratch/checkpoints/unet_resnet34_best.pth \
        --arch    unet --encoder resnet34 \
        --gsd     0.075 \
        --out_dir /scratch/infer_results
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import build_model, val_aug

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
SOLAR_W_PER_M2 = 150   # typical residential PV panel efficiency


def denorm(tensor):
    t = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(t * np.array(STD) + np.array(MEAN), 0, 1)


def infer_image(model, image_path: str, device, threshold: float = 0.5):
    """Return (img_rgb, pred_mask) for a single image."""
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    transform = val_aug()
    aug = transform(image=img, mask=np.zeros(img.shape[:2], dtype=np.float32))
    inp = aug["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(inp)
    pred = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy().astype(np.uint8)
    return img, pred


def estimate_area(mask_binary, gsd_m: float):
    """Convert binary mask to real-world area and estimated solar capacity."""
    n_pixels    = int(mask_binary.sum())
    area_m2     = n_pixels * (gsd_m ** 2)
    capacity_kw = area_m2 * SOLAR_W_PER_M2 / 1000
    return n_pixels, area_m2, capacity_kw


def save_result(img_rgb, pred_mask, save_path, area_m2, cap_kw, gsd_m):
    overlay = img_rgb.copy().astype(np.float32)
    overlay[pred_mask == 1] = (
        overlay[pred_mask == 1] * 0.55 + np.array([255, 100, 0]) * 0.45
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Boundary
    kernel   = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(pred_mask, kernel) - cv2.erode(pred_mask, kernel)
    boundary_vis = img_rgb.copy()
    boundary_vis[boundary == 1] = [255, 0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb);            axes[0].set_title("Input Image")
    axes[1].imshow(pred_mask, cmap="gray"); axes[1].set_title("Predicted Mask")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\nArea: {area_m2:.1f} m²  |  Est. capacity: {cap_kw:.2f} kW\n"
        f"(GSD={gsd_m} m/px, {SOLAR_W_PER_M2} W/m²)"
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = build_model(args.arch, args.encoder)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"Loaded: {args.ckpt}  (epoch {ckpt.get('epoch','?')}, "
          f"val IoU {ckpt.get('val_iou', 0):.4f})")

    # Find images
    img_path = Path(args.image)
    if img_path.is_dir():
        images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.tif"))
        if args.n:
            images = images[: args.n]
    else:
        images = [img_path]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in images:
        img_rgb, pred_mask = infer_image(model, str(p), device, args.threshold)
        n_px, area_m2, cap_kw = estimate_area(pred_mask, args.gsd)

        print(f"\n{p.name}")
        print(f"  Rooftop pixels : {n_px:,}")
        print(f"  Rooftop area   : {area_m2:.2f} m²  (GSD={args.gsd} m/px)")
        print(f"  Est. capacity  : {cap_kw:.2f} kW")

        save_result(
            img_rgb, pred_mask,
            out_dir / (p.stem + "_result.png"),
            area_m2, cap_kw, args.gsd,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",     required=True,  help="Image file or directory of images")
    p.add_argument("--ckpt",      required=True)
    p.add_argument("--out_dir",   default="./infer_results")
    p.add_argument("--arch",      default="unet")
    p.add_argument("--encoder",   default="resnet34")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gsd",       type=float, default=0.075,
                   help="Ground sample distance in m/pixel (AIRS=0.075)")
    p.add_argument("--n",         type=int,   default=None,
                   help="Max images to process when --image is a directory")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
