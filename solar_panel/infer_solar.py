"""
infer_solar.py — Solar panel inference with sliding-window tiling.

Handles ANY image size:
  - Slices input into overlapping 512×512 tiles (matching training resolution)
  - Runs model on each tile
  - Stitches predictions using Hann-window weighted averaging (no seam artefacts)
  - Outputs full-resolution binary mask + area estimate

Usage (single image):
    python solar_panel/infer_solar.py \
        --image /path/to/aerial.png \
        --ckpt  solar_panel/checkpoints/unet_resnet34_best.pth

Usage (whole folder):
    python solar_panel/infer_solar.py \
        --image /path/to/images/ \
        --ckpt  solar_panel/checkpoints/unet_resnet34_best.pth \
        --out_dir solar_panel/infer_results/

GSD guide:
  Google Maps zoom 19 @ India (~20°N) : ~0.25 m/px
  Google Maps zoom 20 @ India (~20°N) : ~0.12 m/px
  Google Maps zoom 18 @ India (~20°N) : ~0.50 m/px
  BDAPPV dataset (French IGN)         :  ~0.20 m/px
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train_solar import build_model

MEAN      = (0.485, 0.456, 0.406)
STD       = (0.229, 0.224, 0.225)
TILE_SIZE = 512   # must match training crop_size
PANEL_W_PER_M2 = 180   # typical modern residential PV panel (~180 W/m²)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def get_transform():
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_sliding_window(model, img_rgb: np.ndarray, device,
                         tile_size: int = TILE_SIZE,
                         overlap: int = 64,
                         threshold: float = 0.5) -> np.ndarray:
    """
    Run inference on an arbitrarily-sized image using overlapping tiles.

    Steps:
      1. Pad image so it's divisible into tiles
      2. Extract overlapping tile_size × tile_size crops
      3. Run model on each crop
      4. Accumulate sigmoid scores into a full-res score map, weighted by
         a Hann window (smoothly down-weights tile edges → no seam artefacts)
      5. Threshold the averaged score map
      6. Crop padding back off

    Returns binary mask (0/1) same H×W as input.
    """
    H, W   = img_rgb.shape[:2]
    stride = tile_size - overlap

    pad_h = max(0, tile_size - H) if H < tile_size else \
            (stride - (H - tile_size) % stride) % stride
    pad_w = max(0, tile_size - W) if W < tile_size else \
            (stride - (W - tile_size) % stride) % stride

    img_pad = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_REFLECT_101)
    pH, pW = img_pad.shape[:2]

    score_map  = np.zeros((pH, pW), dtype=np.float32)
    weight_map = np.zeros((pH, pW), dtype=np.float32)
    tile_weight = _make_tile_weight(tile_size)

    transform = get_transform()
    model.eval()

    ys = list(range(0, pH - tile_size + 1, stride))
    xs = list(range(0, pW - tile_size + 1, stride))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = img_pad[y:y + tile_size, x:x + tile_size]
                aug  = transform(image=tile,
                                 mask=np.zeros((tile_size, tile_size), dtype=np.float32))
                inp  = aug["image"].unsqueeze(0).to(device)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(inp)
                score = torch.sigmoid(logits).squeeze().cpu().numpy()

                score_map [y:y + tile_size, x:x + tile_size] += score * tile_weight
                weight_map[y:y + tile_size, x:x + tile_size] += tile_weight

    avg_score = score_map / (weight_map + 1e-6)
    mask_pad  = (avg_score > threshold).astype(np.uint8)
    return mask_pad[:H, :W]


def _make_tile_weight(tile_size: int) -> np.ndarray:
    """2-D Hann window — tapers to 0 at edges to avoid stitching seams."""
    w1d = np.hanning(tile_size).astype(np.float32)
    return np.outer(w1d, w1d) + 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Area / power estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_solar(mask_binary: np.ndarray, gsd_m: float):
    """Estimate solar panel area and theoretical peak power."""
    n_pixels    = int(mask_binary.sum())
    area_m2     = n_pixels * (gsd_m ** 2)
    capacity_kw = area_m2 * PANEL_W_PER_M2 / 1000
    return n_pixels, area_m2, capacity_kw


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_result(img_rgb, pred_mask, save_path, area_m2, cap_kw, gsd_m):
    """Save 3-panel PNG: original | binary mask | colour overlay."""
    overlay = img_rgb.copy().astype(np.float32)
    overlay[pred_mask == 1] = (
        overlay[pred_mask == 1] * 0.5 + np.array([255, 255, 0]) * 0.5   # yellow tint
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw panel boundaries in red
    kernel   = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(pred_mask, kernel) - cv2.erode(pred_mask, kernel)
    boundary_vis = img_rgb.copy()
    boundary_vis[boundary == 1] = [255, 0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_rgb);                axes[0].set_title("Input Image")
    axes[1].imshow(pred_mask, cmap="gray"); axes[1].set_title("Predicted Solar Panel Mask")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay  (yellow = detected solar panels)\n"
        f"Panel area: {area_m2:.1f} m²  |  Est. peak power: {cap_kw:.2f} kW\n"
        f"GSD={gsd_m} m/px  |  {PANEL_W_PER_M2} W/m²"
    )
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved : {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")

    # Load model
    model = build_model(args.arch, args.encoder)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"Loaded  : {args.ckpt}  "
          f"(epoch {ckpt.get('epoch','?')}, val IoU {ckpt.get('val_iou',0):.4f})")
    print(f"GSD     : {args.gsd} m/px")

    # Collect images
    img_path = Path(args.image)
    if img_path.is_dir():
        exts   = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.tif"]
        images = []
        for ext in exts:
            images += sorted(img_path.glob(ext))
        if args.n:
            images = images[:args.n]
    else:
        images = [img_path]

    print(f"Images  : {len(images)}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in images:
        print(f"Processing: {p.name}  ", end="", flush=True)
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        print(f"({W}×{H}px)")

        pred_mask = infer_sliding_window(
            model, img, device,
            tile_size=TILE_SIZE,
            overlap=args.overlap,
            threshold=args.threshold,
        )

        n_px, area_m2, cap_kw = estimate_solar(pred_mask, args.gsd)
        print(f"  Panel pixels : {n_px:,}")
        print(f"  Panel area   : {area_m2:.2f} m²")
        print(f"  Est. power   : {cap_kw:.2f} kW peak")

        save_result(
            img, pred_mask,
            out_dir / (p.stem + "_solar_result.png"),
            area_m2, cap_kw, args.gsd,
        )

    print(f"\nAll results saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Solar panel segmentation inference — handles any image size",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image",     required=True,
                   help="Image file or directory (png/jpg/tif)")
    p.add_argument("--ckpt",      required=True,
                   help="Path to checkpoint .pth")
    p.add_argument("--out_dir",   default="solar_panel/infer_results",
                   help="Directory to save result PNGs")
    p.add_argument("--arch",      default="unet",
                   choices=["unet","unetplusplus","fpn","pspnet","deeplabv3plus"])
    p.add_argument("--encoder",   default="resnet34")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binarisation threshold (run evaluate_solar.py --sweep_threshold to find best)")
    p.add_argument("--overlap",   type=int,   default=64,
                   help="Tile overlap in pixels (larger = smoother stitching, slower)")
    p.add_argument("--gsd",       type=float, default=0.25,
                   help="Ground sample distance m/px  "
                        "(Google Maps zoom19@India≈0.25, BDAPPV IGN≈0.20)")
    p.add_argument("--n",         type=int,   default=None,
                   help="Max images to process when --image is a directory")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
