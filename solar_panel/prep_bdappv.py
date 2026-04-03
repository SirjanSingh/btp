"""
prep_bdappv.py — Prepare BDAPPV solar panel dataset for training.

BDAPPV structure (after unzip):
    bdappv/
      google/
        img/   ← aerial images (~400x400px, Google Maps source)
        mask/  ← binary masks (255=solar panel, 0=background)
      ign/
        img/   ← aerial images (IGN source, higher res)
        mask/
      metadata.csv

This script:
  1. Reads images + masks from google/ and optionally ign/
  2. Checks mask format (0/1 vs 0/255) and stats
  3. Resizes to 400x400 or 512x512 (configurable)
  4. Splits into train/val/test (default 80/10/10)
  5. Writes to:
       bdappv_crops/
         train/images/  train/masks/
         val/images/    val/masks/
         test/images/   test/masks/

Usage:
    python prep_bdappv.py \
        --src_dir /workspace/bdappv/bdappv \
        --out_dir /workspace/bdappv_crops \
        --sources google ign \
        --size 400 \
        --val_frac 0.1 --test_frac 0.1
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def check_dataset(src_dir: Path, sources: list):
    """Print dataset stats before processing."""
    print("\n── Dataset scan ──────────────────────────────────")
    total = 0
    for src in sources:
        img_dir  = src_dir / src / "img"
        mask_dir = src_dir / src / "mask"
        imgs  = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        masks = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))
        print(f"  {src}: {len(imgs)} images, {len(masks)} masks")
        if imgs:
            sample = cv2.imread(str(imgs[0]))
            print(f"    sample size : {sample.shape[1]}×{sample.shape[0]}px")
        if masks:
            m = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
            print(f"    mask values : min={m.min()} max={m.max()} "
                  f"unique={np.unique(m).tolist()}")
            pos_frac = (m > 0).mean()
            print(f"    panel frac  : {pos_frac:.3%} of pixels")
        total += len(imgs)
    print(f"  Total: {total} images")
    print("──────────────────────────────────────────────────\n")


def collect_pairs(src_dir: Path, sources: list):
    """Return list of (img_path, mask_path) pairs."""
    pairs = []
    for src in sources:
        img_dir  = src_dir / src / "img"
        mask_dir = src_dir / src / "mask"

        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img_path in sorted(img_dir.glob(ext)):
                # mask may have same name or different extension
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    # try .png fallback
                    mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    pairs.append((img_path, mask_path, src))
                else:
                    print(f"  [warn] no mask for {img_path.name}, skipping")
    return pairs


def process_and_save(pairs, out_dir: Path, split: str, size: int,
                     min_panel_frac: float = 0.001):
    """
    Resize and save images + masks for one split.
    Skips images with no panel pixels (background-only crops).
    Returns count of saved and skipped.
    """
    img_out  = out_dir / split / "images"
    mask_out = out_dir / split / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0

    for img_path, mask_path, src in tqdm(pairs, desc=f"  {split:5s}"):
        img  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None or img is None:
            skipped += 1
            continue

        # Normalise mask to 0/255
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = (mask > 127).astype(np.uint8) * 255

        # Skip background-only crops
        if (mask > 0).mean() < min_panel_frac:
            skipped += 1
            continue

        # Resize if needed
        h, w = img.shape[:2]
        if h != size or w != size:
            img  = cv2.resize(img,  (size, size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        stem = f"{src}_{img_path.stem}"
        cv2.imwrite(str(img_out  / f"{stem}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_out / f"{stem}.png"), mask)
        saved += 1

    return saved, skipped


def main(args):
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    check_dataset(src_dir, args.sources)

    print("Collecting image-mask pairs ...")
    pairs = collect_pairs(src_dir, args.sources)
    print(f"Found {len(pairs)} valid pairs\n")

    if not pairs:
        raise SystemExit("No pairs found — check --src_dir and --sources")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(pairs)

    n       = len(pairs)
    n_test  = int(n * args.test_frac)
    n_val   = int(n * args.val_frac)
    n_train = n - n_val - n_test

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    print(f"Split: train={len(train_pairs)} | val={len(val_pairs)} | test={len(test_pairs)}")
    print(f"Resize: {args.size}×{args.size}px")
    print(f"Min panel fraction: {args.min_panel_frac:.3%}\n")

    print("Processing ...")
    tr_saved, tr_skip = process_and_save(train_pairs, out_dir, "train",
                                          args.size, args.min_panel_frac)
    va_saved, va_skip = process_and_save(val_pairs,   out_dir, "val",
                                          args.size, args.min_panel_frac)
    te_saved, te_skip = process_and_save(test_pairs,  out_dir, "test",
                                          args.size, args.min_panel_frac)

    print(f"\n── Done ─────────────────────────────────────────")
    print(f"  train : {tr_saved} saved, {tr_skip} skipped")
    print(f"  val   : {va_saved} saved, {va_skip} skipped")
    print(f"  test  : {te_saved} saved, {te_skip} skipped")
    print(f"  total : {tr_saved+va_saved+te_saved} crops")
    print(f"  output: {out_dir}")
    print(f"─────────────────────────────────────────────────\n")
    print("Next step:")
    print(f"  python train.py \\")
    print(f"    --train_dir {out_dir}/train \\")
    print(f"    --val_dir   {out_dir}/val \\")
    print(f"    --ckpt_dir  /workspace/checkpoints_pv \\")
    print(f"    --arch unet --encoder resnet34 \\")
    print(f"    --epochs 100 --batch_size 16 --workers 2 \\")
    print(f"    --simulate_low_res")


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare BDAPPV solar panel dataset for UNet training")
    p.add_argument("--src_dir",   default="./bdappv/bdappv",
                   help="Path to bdappv root (contains google/ and ign/)")
    p.add_argument("--out_dir",   default="./bdappv_crops",
                   help="Output directory for train/val/test splits")
    p.add_argument("--sources",   nargs="+", default=["google", "ign"],
                   choices=["google", "ign"],
                   help="Which sources to include (default: both)")
    p.add_argument("--size",      type=int, default=400,
                   help="Output image size in pixels (400 keeps original, 512 matches AIRS)")
    p.add_argument("--val_frac",  type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--min_panel_frac", type=float, default=0.001,
                   help="Skip crops where panel pixels < this fraction (removes background noise)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
