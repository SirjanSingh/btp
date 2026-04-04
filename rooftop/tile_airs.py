"""
tile_airs.py — Tile AIRS 10,000×10,000 images into 512×512 crops.

Usage (AIRS default layout — image/ and label/ subdirs):
    python tile_airs.py --src_dir /workspace/dataset/train \
                        --out_dir /airs/btp/dataset_crops/train \
                        --img_subdir image --mask_subdir label

Usage (standard layout — images/ and masks/ subdirs):
    python tile_airs.py --src_dir /scratch/airs/train \
                        --out_dir /scratch/airs_crops/train

Output always writes to images/ and masks/ under --out_dir.
"""

import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def tile_pair(img_path: Path, mask_path: Path, crop_size: int, stride: int,
              min_mask_frac: float = 0.005):
    """Yield (crop_name, img_crop, mask_crop) for every valid tile."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise IOError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Cannot read mask: {mask_path}")
    # Handle both 0/1 masks (AIRS default) and 0/255 masks
    if mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 127).astype(np.uint8) * 255

    H, W = img.shape[:2]
    idx = 0
    for y in range(0, H - crop_size + 1, stride):
        for x in range(0, W - crop_size + 1, stride):
            mc = mask[y:y + crop_size, x:x + crop_size]
            if mc.mean() / 255.0 < min_mask_frac:
                continue                          # skip near-empty mask crops
            ic = img[y:y + crop_size, x:x + crop_size]
            yield f"{img_path.stem}_{idx:06d}", ic, mc
            idx += 1


def run(args):
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    stride = int(args.crop_size * (1.0 - args.overlap))

    img_src = src / args.img_subdir
    msk_src = src / args.mask_subdir
    img_out = out / "images"
    msk_out = out / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    # check if already done
    existing = list(img_out.glob("*.png"))
    if existing and not args.force:
        print(f"[skip] {len(existing)} crops already in {img_out}. Use --force to redo.")
        return len(existing)

    extensions = ["*.tif", "*.tiff", "*.png", "*.jpg"]
    img_files = []
    for ext in extensions:
        img_files.extend(sorted(img_src.glob(ext)))

    if not img_files:
        raise FileNotFoundError(f"No image files found in {img_src}")

    total = 0
    skipped = 0
    for img_path in tqdm(img_files, desc=f"Tiling {src.name}"):
        # find matching mask
        mask_path = msk_src / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = msk_src / (img_path.stem + ".tif")
        if not mask_path.exists():
            print(f"  [warn] no mask for {img_path.name}, skipping")
            skipped += 1
            continue

        try:
            for name, img_c, msk_c in tile_pair(
                img_path, mask_path, args.crop_size, stride, args.min_mask_frac
            ):
                cv2.imwrite(str(img_out / f"{name}.png"),
                            cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(msk_out / f"{name}.png"), msk_c)
                total += 1
                if args.max_crops and total >= args.max_crops:
                    print(f"  Reached --max_crops={args.max_crops}, stopping.")
                    print(f"  Total crops saved: {total}")
                    return total
        except Exception as e:
            print(f"  [error] {img_path.name}: {e}")
            skipped += 1

    print(f"\nDone. Crops: {total} from {len(img_files) - skipped} images "
          f"({skipped} skipped).")
    return total


def parse_args():
    p = argparse.ArgumentParser(description="Tile AIRS images into overlapping crops")
    p.add_argument("--src_dir",       required=True, help="Parent dir containing image + mask subdirs")
    p.add_argument("--out_dir",       required=True, help="Output dir for tiled crops")
    p.add_argument("--img_subdir",    default="images", help="Image subdir name under src_dir (AIRS uses 'image')")
    p.add_argument("--mask_subdir",   default="masks",  help="Mask subdir name under src_dir  (AIRS uses 'label')")
    p.add_argument("--crop_size",     type=int,   default=512)
    p.add_argument("--overlap",       type=float, default=0.1,   help="Fractional overlap [0,1)")
    p.add_argument("--min_mask_frac", type=float, default=0.005, help="Skip crops where mask coverage < this")
    p.add_argument("--max_crops",     type=int,   default=None,  help="Cap total crops (useful for quick tests)")
    p.add_argument("--force",         action="store_true",       help="Re-tile even if output exists")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
