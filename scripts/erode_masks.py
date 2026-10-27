#!/usr/bin/env python
"""
erode_masks.py — shrink binary masks in RASTER space.

WHY THIS IS NOT make_weak_labels.py's EROSION, and why that is acceptable here.
`make_weak_labels.py` erodes **polygons** (`shapely buffer(-d)` in EPSG:4326)
before rasterising, which is the correct operation when vector geometry exists:
it shrinks each footprint about its own boundary and can delete a polygon
outright. Pseudo-labels have no vector form -- they are the model's raster
output -- so morphological erosion is the only available operation.

The two differ in ways worth stating rather than glossing:
  - Morphological erosion shrinks uniformly in PIXEL space, so at a fixed GSD it
    is equivalent, but it cannot make a shape "vanish as a polygon"; it just
    erodes to nothing, which for a small blob is nearly the same outcome.
  - It operates on connected components as drawn, so two touching predictions
    that were already fused are eroded as ONE shape and will not separate. That
    is a real limitation for the merging problem and is why this is a repair
    attempt, not a guaranteed fix.

GSD: Jaipur mosaics are ~26.6 cm/px, so 0.4 m is ~1.5 px and 0.2 m ~0.75 px.
Kernel radius is computed from the requested metres rather than hard-coded, and
printed, so the sub-pixel rounding is visible instead of silent.

Usage:
    python scripts/erode_masks.py --src data/jaipur_pseudo_t050/train/masks \
        --dst data/jaipur_pseudo_t050_e04/train/masks --erode_m 0.4
"""
import argparse
import json
import os

import cv2
import numpy as np

GSD_M = 0.266  # measured, not assumed -- see D5 / MASTER_CONTEXT


def main(a):
    radius_px = a.erode_m / a.gsd
    # An elliptical kernel of size k removes about (k-1)/2 px from each side.
    k = max(3, int(round(radius_px * 2 + 1)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    print(f"[erode] {a.erode_m} m at {a.gsd} m/px = {radius_px:.2f} px "
          f"-> {k}x{k} elliptical kernel (removes ~{(k-1)/2:.1f} px per side)")

    os.makedirs(a.dst, exist_ok=True)
    names = sorted(os.listdir(a.src))
    fg_before = fg_after = total = 0
    vanished = 0
    for i, n in enumerate(names):
        m = cv2.imread(os.path.join(a.src, n), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = (m > 127).astype(np.uint8)
        e = cv2.erode(b, kernel, iterations=1)
        fg_before += int(b.sum())
        fg_after += int(e.sum())
        total += b.size
        if b.sum() > 0 and e.sum() == 0:
            vanished += 1
        cv2.imwrite(os.path.join(a.dst, n), e * 255)
        if i % 2000 == 0:
            print(f"  {i}/{len(names)}")

    summary = {
        "src": a.src, "dst": a.dst, "erode_m": a.erode_m, "gsd_m_per_px": a.gsd,
        "kernel_px": k, "n_masks": len(names),
        "fg_fraction_before": round(fg_before / max(total, 1), 5),
        "fg_fraction_after": round(fg_after / max(total, 1), 5),
        "fg_retained": round(fg_after / max(fg_before, 1), 5),
        "masks_emptied": vanished,
        "note": "RASTER morphological erosion, not the polygon buffer used in "
                "make_weak_labels.py -- cannot separate already-fused components",
    }
    json.dump(summary, open(os.path.join(os.path.dirname(a.dst.rstrip('/')),
                                         "erode_summary.json"), "w"), indent=2)
    print(f"[erode] fg {summary['fg_fraction_before']:.4f} -> "
          f"{summary['fg_fraction_after']:.4f} "
          f"(retained {summary['fg_retained']:.3f}), "
          f"{vanished} masks emptied")
    print(f"[ok] {a.dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--erode_m", type=float, default=0.4)
    p.add_argument("--gsd", type=float, default=GSD_M)
    main(p.parse_args())
