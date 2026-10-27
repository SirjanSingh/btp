#!/usr/bin/env python
"""
D1 — measured building-pixel prior for the Jaipur target domain.

Rasterises Google Open Buildings polygons over the Jaipur mosaic footprint and
measures what fraction of pixels are actually buildings. This replaces the
[ASSUMED] "~50%" in the planning docs with a number, which matters because:

  1. It sets the CBST class ratio for self-training. Guessing it wrong biases
     every pseudo-label round.
  2. Paired with D6 (what the AIRS-trained model PREDICTS), the gap between
     measured prior and predicted foreground is the project's motivating figure.
  3. Reporting an estimated prior in the thesis is explicitly forbidden by the
     master context's own "unverified" section.

Open Buildings ships a per-building `confidence`. Low-confidence detections are
noisy, so the prior is reported at several confidence cut-offs rather than one:
the choice of cut-off is a real modelling decision and should be visible.

CAVEAT worth carrying into the writeup: Open Buildings labels GROUND FOOTPRINTS,
while this project predicts ROOF OUTLINES (master context Gap 4). For a 4-storey
Jaipur building viewed off-nadir those differ by ~8 px, so this prior is a good
estimate of building density but is NOT a pixel-accurate roof mask.

Usage:
    python scripts/d1_target_prior.py \
        --csv data/open_buildings/jaipur_open_buildings.csv \
        --tiles data/jaipur --out_dir diagnostics/d1
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely import wkt
from shapely.geometry import shape


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"[D1] reading {a.csv}")
    df = pd.read_csv(a.csv)
    print(f"[D1] {len(df):,} buildings in AOI")
    print(f"[D1] confidence: min {df.confidence.min():.3f} "
          f"median {df.confidence.median():.3f} max {df.confidence.max():.3f}")
    print(f"[D1] area_m2:    median {df.area_in_meters.median():.1f} "
          f"mean {df.area_in_meters.mean():.1f}")

    tifs = sorted(
        os.path.join(a.tiles, f) for f in os.listdir(a.tiles) if f.endswith(".tif")
    )
    results = {}

    # Reproject ONCE for the whole AOI, not once per (tile, polygon). Doing it
    # per tile costs len(df) x len(tifs) transforms -- ~8.4M here -- and dominates
    # runtime. Transform up front, cache each polygon's bbox, then per tile do a
    # cheap numeric bbox test to pick the geometries that actually overlap.
    # transform_geom takes a whole sequence and transforms it in one call, which
    # is far cheaper than 523k Python-level round trips into GDAL.
    print("\n[D1] reprojecting polygons 4326 -> 3857 (once)")
    shp = [wkt.loads(g) for g in df.geometry]
    geoms_m = transform_geom("EPSG:4326", "EPSG:3857",
                             [s.__geo_interface__ for s in shp])
    # Bounds come from shapely rather than by walking `coordinates` by hand: the
    # AOI holds 2 MULTIPOLYGONs among 523,281 POLYGONs, and their coordinates are
    # nested one level deeper, so a hand-rolled `for ring in coords for c in ring`
    # yields coordinate pairs instead of scalars and the bbox array goes ragged.
    bboxes = np.asarray([shape(g).bounds for g in geoms_m])  # minx, miny, maxx, maxy
    conf = df.confidence.to_numpy()
    print(f"[D1] reprojected {len(geoms_m):,} polygons")

    for cutoff in a.cutoffs:
        keep = conf >= cutoff
        print(f"\n[D1] confidence >= {cutoff}: {int(keep.sum()):,} buildings")

        per_tile, tot_bld, tot_px = {}, 0, 0
        for path in tifs:
            with rasterio.open(path) as src:
                b = src.bounds
                sel = keep & (
                    (bboxes[:, 0] <= b.right) & (bboxes[:, 2] >= b.left)
                    & (bboxes[:, 1] <= b.top) & (bboxes[:, 3] >= b.bottom)
                )
                shapes = [(geoms_m[i], 1) for i in np.nonzero(sel)[0]]

                # Rasterise at reduced resolution: the prior is a ratio, and a
                # full 11168x13856 burn per tile per cutoff is needlessly slow.
                scale = a.downscale
                h, w = src.height // scale, src.width // scale
                tr = src.transform * src.transform.scale(scale, scale)
                mask = (
                    rasterize(shapes, out_shape=(h, w), transform=tr,
                              fill=0, dtype=np.uint8)
                    if shapes else np.zeros((h, w), np.uint8)
                )
                frac = float(mask.mean())
                per_tile[os.path.basename(path)] = round(frac, 5)
                tot_bld += mask.sum()
                tot_px += mask.size
                print(f"   {os.path.basename(path):22s} {len(shapes):6d} bldg "
                      f"{frac*100:6.2f}% fg", flush=True)

        overall = float(tot_bld / tot_px)
        vals = np.array(list(per_tile.values()))
        results[str(cutoff)] = {
            "n_buildings": int(keep.sum()),
            "overall_fg_fraction": round(overall, 5),
            "per_tile": per_tile,
            "tile_min": round(float(vals.min()), 5),
            "tile_max": round(float(vals.max()), 5),
            "tile_median": round(float(np.median(vals)), 5),
        }
        print(f"[D1] cutoff {cutoff}: OVERALL PRIOR = {overall*100:.2f}% "
              f"(tiles {vals.min()*100:.1f}%-{vals.max()*100:.1f}%)")

    out = {
        "source": "Google Open Buildings v3 (CC BY 4.0), S2 cell 397",
        "caveat": "ground footprints, not roof outlines (Gap 4)",
        "downscale": a.downscale,
        "by_confidence_cutoff": results,
    }
    with open(os.path.join(a.out_dir, "d1_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[D1] written {a.out_dir}/d1_summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--out_dir", default="diagnostics/d1")
    p.add_argument("--cutoffs", type=float, nargs="+", default=[0.0, 0.75])
    p.add_argument("--downscale", type=int, default=8)
    main(p.parse_args())
