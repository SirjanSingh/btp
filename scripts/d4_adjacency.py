#!/usr/bin/env python
"""
D4 — adjacency rate. How often do Jaipur buildings touch a neighbour?

Decides whether the three-class / instance-split work in the plan is needed at
all. `MASTER_CONTEXT` §3.2 lists instance merging as a distinct failure mode:
party-wall buildings with no visible gap fuse into one blob, which pixel IoU
cannot see and boundary IoU largely cannot either. All of that machinery is only
worth building if buildings actually touch often.

If the adjacency rate is low, a whole planned workstream can be deleted, which is
the cheapest kind of result there is.

Method: buffer every footprint by `--tol` metres and count how many others its
bounding box overlaps, then confirm with a real intersection test. An STRtree
makes this O(n log n) rather than the 523k^2 pairs a naive scan would need.

Usage:
    python scripts/d4_adjacency.py --out_dir diagnostics/d4
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.strtree import STRtree


def main(a):
    os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)
    if a.min_conf > 0:
        df = df[df.confidence >= a.min_conf]
    print(f"[D4] {len(df):,} polygons at confidence >= {a.min_conf}")

    geoms = [wkt.loads(g) for g in df.geometry]
    # Degrees, not metres: the CSV is EPSG:4326. At Jaipur's latitude 1 deg lat
    # is ~110.9 km, so convert the tolerance rather than buffering in degrees by
    # eye. Longitude is compressed by cos(lat); using the latitude scale for both
    # is slightly conservative in x, which biases the rate DOWN, not up.
    tol_deg = a.tol_m / 110_900.0
    print(f"[D4] tolerance {a.tol_m} m = {tol_deg:.8f} deg")

    tree = STRtree(geoms)
    n_touch = 0
    counts = []
    for i, g in enumerate(geoms):
        gb = g.buffer(tol_deg)
        idx = tree.query(gb)
        # query returns candidates by bbox; confirm with a real intersects test
        # and drop self-matches.
        k = sum(1 for j in idx if j != i and geoms[j].intersects(gb))
        counts.append(k)
        if k:
            n_touch += 1
        if (i + 1) % 50000 == 0:
            print(f"   {i+1:,}/{len(geoms):,}  running rate "
                  f"{n_touch/(i+1)*100:.1f}%", flush=True)

    c = np.asarray(counts)
    out = {
        "source": "Google Open Buildings v3, Jaipur AOI",
        "n_polygons": int(len(geoms)),
        "min_confidence": a.min_conf,
        "tolerance_m": a.tol_m,
        "adjacency_rate": round(float((c > 0).mean()), 5),
        "mean_neighbours": round(float(c.mean()), 3),
        "median_neighbours": int(np.median(c)),
        "p90_neighbours": int(np.percentile(c, 90)),
        "max_neighbours": int(c.max()),
        "frac_with_3plus": round(float((c >= 3).mean()), 5),
    }
    with open(os.path.join(a.out_dir, "d4_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[D4] adjacency rate {out['adjacency_rate']*100:.1f}% "
          f"at {a.tol_m} m tolerance")
    print(f"[D4] mean {out['mean_neighbours']} neighbours, "
          f"p90 {out['p90_neighbours']}, max {out['max_neighbours']}")
    print(f"[D4] {out['frac_with_3plus']*100:.1f}% have 3+ neighbours")
    print(f"[ok] {a.out_dir}/d4_summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--out_dir", default="diagnostics/d4")
    p.add_argument("--min_conf", type=float, default=0.75)
    p.add_argument("--tol_m", type=float, default=0.5,
                   help="two buildings count as adjacent within this distance; "
                        "0.5 m is under two pixels at Jaipur's 26.6 cm GSD")
    main(p.parse_args())
