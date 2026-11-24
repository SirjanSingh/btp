#!/usr/bin/env python
"""
make_instance_labels.py - one ID per Open Buildings polygon, per 512 px crop.

WHY: the binary weak masks merge touching buildings into one blob, so any per-building
step (per-building alignment, SAM box prompts) needs the polygons kept apart. This burns
every polygon (all confidences, UNSHIFTED) with its own ID on the same tile grid as
scripts/make_weak_labels.py, then writes each crop as a 16-bit PNG with IDs renumbered
1..n (0 = background). Crops that make_weak_labels skips (all-black imagery) are skipped too.

Usage (DGX): ./run_docker.sh 5 "python scripts/make_instance_labels.py --out_dir data/iou/inst"
"""
import argparse
import os
import sys

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

sys.path.insert(0, os.path.dirname(__file__))
from make_weak_labels import load_polygons  # noqa: E402


def main(a):
    geoms, bboxes = load_polygons(a.csv, 0.0)
    os.makedirs(a.out_dir, exist_ok=True)
    tifs = sorted(f for f in os.listdir(a.tiles) if f.endswith(".tif"))
    n_crops = 0
    for name in tifs:
        with rasterio.open(os.path.join(a.tiles, name)) as src:
            b = src.bounds
            sel = np.nonzero((bboxes[:, 0] <= b.right) & (bboxes[:, 2] >= b.left)
                             & (bboxes[:, 1] <= b.top) & (bboxes[:, 3] >= b.bottom))[0]
            shapes = [(geoms[i], int(k + 1)) for k, i in enumerate(sel)]
            full = rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform,
                             fill=0, dtype=np.int32) if shapes else \
                np.zeros((src.height, src.width), np.int32)
            for top in range(0, src.height - a.crop + 1, a.crop):
                for left in range(0, src.width - a.crop + 1, a.crop):
                    img = src.read(1, window=Window(left, top, a.crop, a.crop))
                    if img.max() == 0 and src.read((2, 3), window=Window(left, top, a.crop, a.crop)).max() == 0:
                        continue
                    m = full[top:top + a.crop, left:left + a.crop]
                    ids, inv = np.unique(m, return_inverse=True)
                    out = inv.reshape(m.shape).astype(np.uint16)
                    if ids[0] != 0:          # no background pixel in this crop
                        out += 1
                    stem = f"{os.path.splitext(name)[0]}_{top:06d}_{left:06d}"
                    cv2.imwrite(os.path.join(a.out_dir, stem + ".png"), out)
                    n_crops += 1
        print(f"   {name:18s} {len(shapes):6d} polygons", flush=True)
    print(f"[inst] wrote {n_crops} crops to {a.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--out_dir", default="data/iou/inst")
    p.add_argument("--crop", type=int, default=512)
    main(p.parse_args())
