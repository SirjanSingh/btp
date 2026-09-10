#!/usr/bin/env python
"""
viz_label_levels.py — render what the model is actually trained on, and how the
labels change at each erosion level.

WHY. Every rooftop result in this repo depends on labels nobody has looked at.
They come from Google Open Buildings, they are *ground footprints rather than
roof outlines*, and the single most consequential knob -- how far each footprint
is shrunk before rasterising -- was chosen from an aggregate metric
(`pred/label` -> 1.0). An aggregate can be right for the wrong reason, and the
only way to check is to look at the polygons on top of the imagery.

FAITHFULNESS. Erosion is applied exactly as `make_weak_labels.py` does it:
shapely `buffer(-d)` in EPSG:4326 with `d = erode_m / 110_900`, degenerate
polygons dropped, then reprojected to the tile CRS and rasterised through the
window transform. It is NOT a morphological erosion of the finished raster --
that would shrink the mask uniformly in pixel space and would not reproduce the
polygons vanishing entirely, which is the effect that matters at 0.8 m.

The expensive global reprojection in the real pipeline is skipped here by
filtering candidate polygons with their EPSG:4326 bounds against the tile
footprint first, so only a few thousand polygons per tile are ever transformed.

Usage:
    python scripts/viz_label_levels.py --manifest labelling/r12_batch1/manifest.csv \
        --out docs/figures/label_levels
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from shapely import wkt
from shapely.geometry import shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = [0.0, 0.2, 0.4, 0.8]
# Matches make_weak_labels.py -- metres to degrees at Jaipur's latitude.
DEG_PER_M = 1.0 / 110_900.0


def load_polygons(csv_path, min_conf):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if min_conf > 0:
        df = df[df.confidence >= min_conf]
    shp = [wkt.loads(g) for g in df.geometry]
    # Bounds in the CSV's own CRS (4326). Cheap, and enough to reject the ~99 %
    # of polygons that lie outside any given tile before paying for reprojection.
    bounds = np.asarray([s.bounds for s in shp])
    print(f"[viz] {len(shp):,} polygons at confidence >= {min_conf}")
    return shp, bounds


def burn(shp_subset, erode_m, src_crs, transform, out_shape):
    """Erode in 4326, reproject, rasterise into one crop window."""
    geoms = shp_subset
    n_vanished = 0
    if erode_m:
        d = erode_m * DEG_PER_M
        buffered = [g.buffer(-d) for g in geoms]
        kept = [g for g in buffered if not g.is_empty and g.is_valid]
        n_vanished = len(buffered) - len(kept)
        geoms = kept
    if not geoms:
        return np.zeros(out_shape, np.uint8), 0, n_vanished
    proj = transform_geom("EPSG:4326", src_crs.to_string(),
                          [g.__geo_interface__ for g in geoms])
    m = rasterize([(g, 1) for g in proj], out_shape=out_shape,
                  transform=transform, fill=0, dtype=np.uint8)
    n_comp, _ = cv2.connectedComponents(m)
    return m, int(n_comp - 1), n_vanished


def overlay(img, mask, colour=(0, 200, 255)):
    """Translucent fill plus a hard outline.

    The outline is the point: a fill alone makes two touching buildings look
    like one blob, which is exactly the failure the erosion sweep is about.
    """
    out = img.copy()
    tint = np.zeros_like(img)
    tint[mask > 0] = colour
    out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, colour, 1)
    return out


def main(a):
    rows = list(csv.DictReader(open(a.manifest)))
    # One or two crops per density bin -- enough to see the pattern without
    # producing a page nobody scrolls to the end of.
    per_bin = {}
    picked = []
    for r in sorted(rows, key=lambda r: float(r["fg_fraction"])):
        b = r["bin"]
        if per_bin.get(b, 0) >= a.per_bin:
            continue
        if b == "empty":          # nothing to show; the mask is blank at every level
            continue
        per_bin[b] = per_bin.get(b, 0) + 1
        picked.append(r)
    print(f"[viz] {len(picked)} crops: " + ", ".join(r["bin"] for r in picked))

    shp, bounds = load_polygons(a.csv, a.min_conf)
    os.makedirs(a.out, exist_ok=True)
    meta = []

    for r in picked:
        name = r["name"]
        tile = r["tile"] + ".tif"
        # "map67_3-3_001536_006144.png" -> top=1536, left=6144
        stem = name[:-4].split("_")
        top, left = int(stem[-2]), int(stem[-1])
        tif = os.path.join(a.tiles, tile)
        if not os.path.isfile(tif):
            print(f"[skip] {tif} missing")
            continue

        with rasterio.open(tif) as src:
            win = Window(left, top, a.crop, a.crop)
            img = np.transpose(src.read((1, 2, 3), window=win), (1, 2, 0))
            img = np.ascontiguousarray(img[:, :, ::-1])   # RGB -> BGR for cv2
            tr = win_transform(win, src.transform)
            # Window footprint in 4326, to pre-filter polygons.
            wb = rasterio.windows.bounds(win, src.transform)
            l, b_, rt, t = transform_bounds(src.crs, "EPSG:4326", *wb)
            sel = ((bounds[:, 0] <= rt) & (bounds[:, 2] >= l)
                   & (bounds[:, 1] <= t) & (bounds[:, 3] >= b_))
            subset = [shp[i] for i in np.nonzero(sel)[0]]

            entry = {"name": name, "tile": r["tile"], "bin": r["bin"],
                     "fg_fraction": round(float(r["fg_fraction"]), 4),
                     "levels": []}
            cv2.imwrite(os.path.join(a.out, f"{name[:-4]}__raw.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, a.quality])

            for lv in LEVELS:
                m, n_comp, n_van = burn(subset, lv, src.crs, tr,
                                        (a.crop, a.crop))
                vis = overlay(img, m)
                fn = f"{name[:-4]}__e{str(lv).replace('.', 'p')}.jpg"
                cv2.imwrite(os.path.join(a.out, fn), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                entry["levels"].append({
                    "erode_m": lv, "file": fn,
                    "fg_fraction": round(float((m > 0).mean()), 4),
                    "n_components": n_comp,
                    "n_polygons_vanished": n_van,
                    "n_polygons_in_window": len(subset),
                })
            meta.append(entry)
            print(f"[viz] {name}  {r['bin']:11s} "
                  + "  ".join(f"{l['erode_m']}m:{l['n_components']}c"
                              for l in entry["levels"]))

    json.dump({"levels_m": LEVELS, "min_confidence": a.min_conf,
               "crop_px": a.crop, "crops": meta},
              open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="labelling/r12_batch1/manifest.csv")
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--out", default="docs/figures/label_levels")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--min_conf", type=float, default=0.75)
    p.add_argument("--per_bin", type=int, default=2)
    p.add_argument("--quality", type=int, default=82)
    main(p.parse_args())
