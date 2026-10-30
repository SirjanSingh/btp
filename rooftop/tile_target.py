"""
tile_target.py — cut large target-domain GeoTIFFs (e.g. Jaipur map67_*.tif) into
512x512 PNG tiles for self_train.py / labelling. Reads one window at a time, so
an 11168 x 13856 file never has to fit in memory.

Usage:
    python tile_target.py --tif daraset/map67_1-1.tif daraset/map67_1-2.tif \
        --out /scratch/jaipur_tiles --tile 512 --stride 512 --max_per_file 800
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    total = 0
    for tif in a.tif:
        stem = Path(tif).stem
        with rasterio.open(tif) as src:
            W, H = src.width, src.height
            print(f"{stem}: {W}x{H}, bands={src.count}, crs={src.crs}")
            coords = [(x, y) for y in range(0, H - a.tile + 1, a.stride)
                      for x in range(0, W - a.tile + 1, a.stride)]
            rng.shuffle(coords)
            kept = 0
            for x, y in coords:
                if a.max_per_file and kept >= a.max_per_file:
                    break
                arr = src.read([1, 2, 3], window=Window(x, y, a.tile, a.tile))  # C,H,W
                img = np.transpose(arr, (1, 2, 0))
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                blank = (img.sum(axis=2) == 0).mean()          # nodata
                white = (img.min(axis=2) >= 250).mean()
                if blank > a.max_blank or white > a.max_blank:
                    continue
                cv2.imwrite(str(out / f"{stem}_x{x}_y{y}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                kept += 1
            print(f"  wrote {kept} tiles")
            total += kept
    print(f"Total tiles: {total} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tif", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tile", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max_per_file", type=int, default=None,
                   help="Random subsample per file (tiles are shuffled first)")
    p.add_argument("--max_blank", type=float, default=0.3,
                   help="Skip tile if >this fraction is nodata/white")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
