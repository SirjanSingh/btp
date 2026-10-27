#!/usr/bin/env python
"""
D2 + D3 — measure the SOURCE domain, so both halves of the prior shift are real.

D2 (source foreground fraction) replaces the `[ASSUMED] AIRS ~15%` in
MASTER_CONTEXT §3.1. D1 already measured the Jaipur side at 28.19%; until this
runs, the project's central "trained on X%, deployed on Y%" claim is half guess,
and MASTER_CONTEXT §11 forbids putting estimated priors in the report.

D3 (building size distribution) compares component areas in m² between AIRS and
Open Buildings Jaipur. It decides whether a 512x512 crop and the model's
receptive field are the right size for the target: same crop covers 38 m of
ground at 7.5 cm but 136 m at 26.6 cm, so "512 px" means very different things
in the two domains.

Both read the same masks, so they share one pass.

Usage:
    python scripts/d2_d3_source_stats.py --out_dir diagnostics/d2_d3
"""
import argparse
import json
import os

import cv2
import numpy as np

AIRS_GSD = 0.075          # m/px, AIRS Christchurch
JAIPUR_GSD = 0.26618      # m/px, measured from the mosaic in D6


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(a.label_dir)
                   if f.endswith(".tif") and not f.endswith("_vis.tif"))
    if a.limit:
        names = names[:a.limit]
    print(f"[d2] {len(names)} AIRS label tiles")

    fg_fracs, areas_m2, per_tile = [], [], {}
    px_area = AIRS_GSD ** 2

    for i, n in enumerate(names, 1):
        m = cv2.imread(os.path.join(a.label_dir, n), cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"   ! unreadable, skipping: {n}")
            continue
        b = (m > 127).astype(np.uint8) if m.max() > 1 else (m > 0).astype(np.uint8)

        frac = float(b.mean())
        fg_fracs.append(frac)
        per_tile[n] = round(frac, 5)

        # D3: connected components. Background is label 0, so skip it. stats
        # column 4 (CC_STAT_AREA) is pixel count.
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        if n_cc > 1:
            areas_m2.extend((stats[1:, 4] * px_area).tolist())

        if i % 50 == 0 or i == len(names):
            print(f"   {i}/{len(names)}  running mean fg "
                  f"{np.mean(fg_fracs)*100:.2f}%", flush=True)

    fg = np.asarray(fg_fracs)
    ar = np.asarray(areas_m2)
    # Drop 1-2 px specks: at 7.5 cm a "building" under ~2 m2 is a labelling
    # artifact, and they dominate the count while contributing no area.
    ar_real = ar[ar >= a.min_area_m2]

    # Open Buildings Jaipur, for the D3 comparison. area_in_meters is supplied,
    # so no rasterisation needed.
    ob = {}
    if os.path.exists(a.ob_csv):
        import pandas as pd
        df = pd.read_csv(a.ob_csv)
        v = df.area_in_meters.to_numpy()
        ob = {
            "n": int(len(v)),
            "median_m2": round(float(np.median(v)), 2),
            "mean_m2": round(float(v.mean()), 2),
            "p10_m2": round(float(np.percentile(v, 10)), 2),
            "p90_m2": round(float(np.percentile(v, 90)), 2),
        }

    out = {
        "d2_source_foreground_fraction": {
            "source": "AIRS train labels (Christchurch, 7.5 cm)",
            "n_tiles": int(len(fg)),
            "mean": round(float(fg.mean()), 5),
            "median": round(float(np.median(fg)), 5),
            "p10": round(float(np.percentile(fg, 10)), 5),
            "p90": round(float(np.percentile(fg, 90)), 5),
            "min": round(float(fg.min()), 5),
            "max": round(float(fg.max()), 5),
            "per_tile": per_tile,
        },
        "d3_building_size_m2": {
            "airs": {
                "n_components": int(len(ar)),
                "n_after_min_area": int(len(ar_real)),
                "min_area_filter_m2": a.min_area_m2,
                "median_m2": round(float(np.median(ar_real)), 2),
                "mean_m2": round(float(ar_real.mean()), 2),
                "p10_m2": round(float(np.percentile(ar_real, 10)), 2),
                "p90_m2": round(float(np.percentile(ar_real, 90)), 2),
            },
            "open_buildings_jaipur": ob,
            "gsd_note": {
                "airs_m_per_px": AIRS_GSD,
                "jaipur_m_per_px": JAIPUR_GSD,
                "crop_512px_covers_m_airs": round(512 * AIRS_GSD, 1),
                "crop_512px_covers_m_jaipur": round(512 * JAIPUR_GSD, 1),
            },
        },
    }
    with open(os.path.join(a.out_dir, "d2_d3_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    d2, d3 = out["d2_source_foreground_fraction"], out["d3_building_size_m2"]
    print(f"\n[D2] AIRS foreground = {d2['mean']*100:.2f}% mean, "
          f"{d2['median']*100:.2f}% median  (tiles {d2['min']*100:.1f}-{d2['max']*100:.1f}%)")
    print(f"[D2] Jaipur (D1) = 28.19%  ->  prior shift {28.19/(d2['mean']*100):.2f}x")
    print(f"[D3] AIRS building median {d3['airs']['median_m2']:.1f} m2 "
          f"| Jaipur OB median {ob.get('median_m2','?')} m2")
    print(f"[D3] a 512 px crop covers {d3['gsd_note']['crop_512px_covers_m_airs']} m "
          f"in AIRS vs {d3['gsd_note']['crop_512px_covers_m_jaipur']} m in Jaipur")
    print(f"\n[ok] {a.out_dir}/d2_d3_summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label_dir",
                   default="data/airs_full/DAtaset/dataset/train/label")
    p.add_argument("--ob_csv",
                   default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--out_dir", default="diagnostics/d2_d3")
    p.add_argument("--min_area_m2", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
