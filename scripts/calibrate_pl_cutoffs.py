#!/usr/bin/env python
"""
calibrate_pl_cutoffs.py - choose pseudo-label cut-offs for a self-training round from the CALIB half
of the hand crops only (same use of hand labels as round 1; the test half is never read).

Given teacher soft maps (8-bit PNG, value/255 = P(roof)) for the hand crops, report for each cut-off
  roof precision  P(hand roof | p >= hi)      background precision  P(hand non-roof | p <= lo)
and pick the loosest hi / tightest lo that reach the target precisions (round 1 used 82 % / 98.6 %).
Teacher "probabilities" are softmax over Mask2Former scores already in [0,1], so they live in about
[0.27, 0.73]: never assume cut-offs, read them off this table.

  python scripts/calibrate_pl_cutoffs.py --index data/hand_r12b2/index.csv \
      --maps data/hand_teacher_r1 --roof_prec 0.82 --bg_prec 0.986
"""
import argparse
import csv
import os

import cv2
import numpy as np


def main(a):
    rows = [r for r in csv.DictReader(open(a.index)) if r["half"] == "calib"]
    P, H = [], []
    for r in rows:
        m = os.path.join(a.maps, r["name"])
        if not os.path.isfile(m):
            continue
        P.append(cv2.imread(m, cv2.IMREAD_GRAYSCALE).ravel())
        H.append((cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 127).ravel())
    assert P, "no calib maps found"
    P, H = np.concatenate(P), np.concatenate(H)
    print(f"{len(rows)} calib crops, {len(P)/1e6:.1f} Mpx, hand roof share {H.mean():.3f}")
    grid = np.arange(0.20, 0.81, 0.01)
    hi_ok, lo_ok = None, None
    print("cut   roof_prec  roof_share>=cut | bg_prec  bg_share<=cut")
    for t in grid:
        v = int(round(t * 255))
        up, dn = P >= v, P <= v
        rp = H[up].mean() if up.any() else float("nan")
        bp = (~H[dn]).mean() if dn.any() else float("nan")
        print(f"{t:.2f}  {rp:9.3f}  {up.mean():14.3f} | {bp:7.3f}  {dn.mean():12.3f}")
        if hi_ok is None and rp >= a.roof_prec:
            hi_ok = t
        if bp >= a.bg_prec:
            lo_ok = t                      # keep the largest lo that still meets the target
    print(f"\nchosen hi={hi_ok} (roof precision >= {a.roof_prec}), lo={lo_ok} (bg precision >= {a.bg_prec})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data/hand_r12b2/index.csv")
    p.add_argument("--maps", required=True)
    p.add_argument("--roof_prec", type=float, default=0.82)
    p.add_argument("--bg_prec", type=float, default=0.986)
    main(p.parse_args())
