#!/usr/bin/env python
"""
Build a weakly-supervised training set for Jaipur from Open Buildings.

WHY: `plan/` calls this the single highest-leverage action available — Open
Buildings gives ~523k free Jaipur footprints, so the target domain can be trained
on directly instead of adapted to. It is also the cheapest way to settle the
source-domain question: if weak supervision alone reaches the IoU the plan
projects (0.74-0.82), the missing AIRS dataset stops being on the critical path.

WHAT IT IS NOT: these are GROUND FOOTPRINTS, not roof outlines (MASTER_CONTEXT
Gap 4). Off-nadir on a 4-storey building the two differ by ~8 px at this GSD.
Every number produced from these labels is measured against noisy targets and
must be reported that way. This script does not fix that; it only makes the
noise explicit and reproducible.

SPLIT: whole tiles, never random crops. Adjacent crops from one tile overlap in
content, so a random split leaks train into val and inflates IoU. Val tiles are
chosen to span the density range measured by D1 rather than to be contiguous.

Usage:
    python scripts/make_weak_labels.py --out_dir data/jaipur_weak --stride 512
"""
import argparse
import json
import os

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window
from shapely import wkt
from shapely.geometry import shape

# Chosen from D1's per-tile priors to span the range: 35.5% (dense), 27.5% (mid),
# 12.1% (sparse). A val set drawn only from dense tiles would flatter the model.
VAL_TILES = {"map67_1-1.tif", "map67_2-2.tif", "map67_4-3.tif"}


def load_polygons(csv_path, min_conf, erode_m=0.0):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if min_conf > 0:
        df = df[df.confidence >= min_conf]
    print(f"[weak] {len(df):,} polygons at confidence >= {min_conf}")
    # One batched transform, not one call per polygon — see d1_target_prior.py.
    shp = [wkt.loads(g) for g in df.geometry]
    if erode_m:
        # Shrink every footprint before rasterising so that touching buildings
        # get a visible gap. R8 measured a 50% merge rate with a 0% split rate:
        # the model fuses neighbours and never over-segments, so pushing the
        # labels apart is the cheapest way to teach separation. Buffer works in
        # the CRS's units and this CSV is EPSG:4326, so convert metres to
        # degrees via the latitude scale at Jaipur (~110.9 km/deg).
        d = erode_m / 110_900.0
        shp = [g.buffer(-d) for g in shp]
        before = len(shp)
        shp = [g for g in shp if not g.is_empty and g.is_valid]
        print(f"[weak] eroded {erode_m} m; {before - len(shp):,} polygons "
              f"vanished entirely (too small to survive)")
    geoms = transform_geom("EPSG:4326", "EPSG:3857",
                           [s.__geo_interface__ for s in shp])
    # shapely bounds rather than walking `coordinates`: the AOI holds a couple of
    # MULTIPOLYGONs whose rings nest one level deeper.
    bboxes = np.asarray([shape(g).bounds for g in geoms])
    return geoms, bboxes


def main(a):
    geoms, bboxes = load_polygons(a.csv, a.min_conf, a.erode_m)

    tifs = sorted(f for f in os.listdir(a.tiles) if f.endswith(".tif"))
    stats = {"train": 0, "val": 0}
    skipped_blank = 0
    fg_fractions = []

    for name in tifs:
        split = "val" if name in VAL_TILES else "train"
        # Under --masks_only the images dir is a symlink into another label
        # set; makedirs raises FileExistsError on it, so skip it entirely.
        for sub in (("masks",) if a.masks_only else ("images", "masks")):
            os.makedirs(os.path.join(a.out_dir, split, sub), exist_ok=True)

        with rasterio.open(os.path.join(a.tiles, name)) as src:
            b = src.bounds
            sel = ((bboxes[:, 0] <= b.right) & (bboxes[:, 2] >= b.left)
                   & (bboxes[:, 1] <= b.top) & (bboxes[:, 3] >= b.bottom))
            shapes = [(geoms[i], 1) for i in np.nonzero(sel)[0]]

            # Burn the whole tile once at full resolution (~155 MB as uint8),
            # then cut crops from it. Rasterising per crop would re-run the
            # polygon selection thousands of times for no benefit.
            mask_full = rasterize(shapes, out_shape=(src.height, src.width),
                                  transform=src.transform, fill=0,
                                  dtype=np.uint8) if shapes else \
                np.zeros((src.height, src.width), np.uint8)

            n_here = 0
            for top in range(0, src.height - a.crop + 1, a.stride):
                for left in range(0, src.width - a.crop + 1, a.stride):
                    win = Window(left, top, a.crop, a.crop)
                    img = src.read((1, 2, 3), window=win)          # 3,H,W
                    img = np.transpose(img, (1, 2, 0))             # H,W,3

                    # Mosaic edges and gaps are pure black. Training on them
                    # teaches the model that "dark = no building", which is the
                    # opposite of useful on dark RCC roofs.
                    if img.max() == 0:
                        skipped_blank += 1
                        continue

                    m = mask_full[top:top + a.crop, left:left + a.crop]
                    stem = f"{os.path.splitext(name)[0]}_{top:06d}_{left:06d}"
                    # --masks_only: a second label set at a different confidence
                    # cutoff shares the SAME imagery, and the images are ~3.5 GB
                    # of the 4.2 GB set. Write masks alone and symlink images in,
                    # or a confidence sweep exhausts the 40 GB quota in two runs.
                    if not a.masks_only:
                        Image.fromarray(img).save(
                            os.path.join(a.out_dir, split, "images", stem + ".png"))
                    Image.fromarray(m * 255).save(
                        os.path.join(a.out_dir, split, "masks", stem + ".png"))
                    fg_fractions.append(float(m.mean()))
                    n_here += 1

            stats[split] += n_here
            print(f"   {name:18s} {split:5s} {len(shapes):6d} bldg "
                  f"{n_here:5d} crops", flush=True)

    fg = np.array(fg_fractions)
    summary = {
        "crops": stats,
        "skipped_blank": skipped_blank,
        "crop_px": a.crop,
        "stride": a.stride,
        "min_confidence": a.min_conf,
        "val_tiles": sorted(VAL_TILES),
        "label_semantics": "Open Buildings GROUND FOOTPRINTS, not roof outlines "
                           "(MASTER_CONTEXT Gap 4) — targets are noisy",
        "weak_label_fg_fraction": {
            "mean": round(float(fg.mean()), 5),
            "median": round(float(np.median(fg)), 5),
            "frac_crops_empty": round(float((fg == 0).mean()), 5),
        },
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "weak_labels_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[weak] train {stats['train']:,}  val {stats['val']:,}  "
          f"blank skipped {skipped_blank:,}")
    print(f"[weak] mean fg {fg.mean()*100:.2f}%  (D1 tile prior was 28.19%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--out_dir", default="data/jaipur_weak")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--min_conf", type=float, default=0.75)
    p.add_argument("--erode_m", type=float, default=0.0,
                   help="shrink each footprint by N metres before rasterising, "
                        "to open a gap between touching buildings (targets the "
                        "50%% merge rate measured in R8)")
    p.add_argument("--masks_only", action="store_true",
                   help="write masks but not images (symlink images from an "
                        "existing set; they are identical across cutoffs)")
    p.add_argument("--limit_tiles", type=int, default=0,
                   help="only process the first N tiles (smoke test)")
    a = p.parse_args()
    if a.limit_tiles:
        # Cheap way to sanity-check output before committing an hour of CPU.
        import itertools
        _ls = os.listdir
        os.listdir = lambda d, _f=_ls, _n=a.limit_tiles: sorted(_f(d))[:_n] \
            if d == a.tiles else _f(d)
    main(a)
