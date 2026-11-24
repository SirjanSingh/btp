#!/usr/bin/env python
"""
estimate_offset.py - the Open Buildings -> hand-roof offset, measured on ONE half only.

The 7 px south / 3 px west shift in make_weak_labels.py was measured on all hand crops,
including the test half that reports IoU. This re-measures it on the calib half alone, so
the shift used for training never saw a test crop. Per crop: the (dy, dx) in [-20,20]x[-12,12]
that maximises IoU between the shifted OB mask and the hand mask (24 px border ignored);
crops with under 2000 roof px on either side are skipped. Median over crops.

numpy + PIL only (runs with host python3 on the DGX).
  python3 scripts/estimate_offset.py --index data/hand_r12b2/index.csv --half calib
"""
import argparse
import csv
import json

import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def best_shift(h, o, b=24):
    hh = h[b:-b, b:-b]
    best = (-1, 0, 0)
    for dy in range(-20, 21):
        for dx in range(-12, 13):
            s = np.roll(np.roll(o, dy, 0), dx, 1)[b:-b, b:-b]
            v = (hh & s).sum() / max((hh | s).sum(), 1)
            if v > best[0]:
                best = (v, dy, dx)
    return best


def main(a):
    rows = [r for r in csv.DictReader(open(a.index)) if a.half == "all" or r["half"] == a.half]
    shifts = []
    for r in rows:
        h, o = load(r["mask"]), load(r["ob"])
        if h.sum() < 2000 or o.sum() < 2000:
            continue
        shifts.append(best_shift(h, o)[1:])
    s = np.array(shifts)
    out = dict(half=a.half, n_crops_used=len(s), n_crops=len(rows),
               median_dy=float(np.median(s[:, 0])), median_dx=float(np.median(s[:, 1])),
               iqr_dy=np.percentile(s[:, 0], [25, 75]).tolist(), iqr_dx=np.percentile(s[:, 1], [25, 75]).tolist())
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--half", default="calib", choices=["calib", "test", "all"])
    p.add_argument("--out", default="")
    main(p.parse_args())
