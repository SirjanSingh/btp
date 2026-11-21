#!/usr/bin/env python
"""
per_building_offset_check.py - how much would per-building label alignment (OMAF, CVPR 2026 /
Align-and-Segment 2026) help over our single global shift?

On the 152 hand crops, Open Buildings (OB) masks are aligned four ways and scored against the
hand roofs (pooled IoU per half):
  raw         OB as downloaded
  global      OB rolled by the global shift (7 px S, 3 px W), what training uses now
  perbldg_pred   each OB building moved separately to best overlap the r34_c0_shift+TTA
                 prediction (label-free, the AnS idea, usable on all training tiles)
  perbldg_oracle each OB building moved to best overlap the HAND mask (upper bound on
                 what any per-building alignment can give; uses test labels, never train on it)
Per building: search (dy, dx) in [-S, S]^2 around 0 in a window = bbox padded by S+2;
ties broken toward the global shift. Buildings under --min_px px keep the global shift.

Usage (DGX): ./run_docker.sh 0 "python scripts/per_building_offset_check.py"
"""
import argparse
import csv
import json

import cv2
import numpy as np

G = (7, -3)


def load(p):
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127


def shift(m, dy, dx):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = m[ys, xs]
    return out


def per_building(ob, target, S, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ob.astype(np.uint8), connectivity=8)
    out = np.zeros_like(ob)
    H, W = ob.shape
    shifts = []
    for c in range(1, n):
        comp = lab == c
        x, y, w, h, area = stats[c]
        if area < min_px:
            out |= shift(comp, *G)
            continue
        p = S + 2
        y0, y1, x0, x1 = max(y - p, 0), min(y + h + p, H), max(x - p, 0), min(x + w + p, W)
        cw, tw = comp[y0:y1, x0:x1], target[y0:y1, x0:x1]
        best = (-1, 1e9, 0, 0)
        for dy in range(-S, S + 1):
            for dx in range(-S, S + 1):
                s = shift(cw, dy, dx)
                iou = (s & tw).sum() / max((s | tw).sum(), 1)
                d = abs(dy - G[0]) + abs(dx - G[1])
                if iou > best[0] or (iou == best[0] and d < best[1]):
                    best = (iou, d, dy, dx)
        out |= shift(comp, best[2], best[3])
        shifts.append((best[2], best[3], int(area), float(best[0])))
    return out, shifts


def main(a):
    rows = list(csv.DictReader(open(a.index)))
    probs = np.load(a.probs)[a.pred_key]
    acc = {k: {h: np.zeros(3, np.int64) for h in ("calib", "test")}
           for k in ("raw", "global", "perbldg_pred", "perbldg_oracle")}
    sh = {"pred": [], "oracle": []}
    for i, r in enumerate(rows):
        hand, ob = load(r["mask"]), load(r["ob"])
        pred = probs[i].astype(np.float32) > a.thr
        b = a.border
        variants = {"raw": ob, "global": shift(ob, *G)}
        variants["perbldg_pred"], s1 = per_building(ob, pred, a.S, a.min_px)
        variants["perbldg_oracle"], s2 = per_building(ob, hand, a.S, a.min_px)
        sh["pred"] += s1; sh["oracle"] += s2
        hh = hand[b:-b, b:-b]
        for k, m in variants.items():
            mm = m[b:-b, b:-b]
            acc[k][r["half"]] += [(mm & hh).sum(), (mm & ~hh).sum(), (~mm & hh).sum()]
    res = {k: {h: round(float(v[0] / v.sum()), 4) for h, v in d.items()} for k, d in acc.items()}
    for k, s in sh.items():
        s = np.array(s)
        if len(s):
            res[f"shift_stats_{k}"] = dict(
                n_buildings=len(s), median_dy=float(np.median(s[:, 0])), median_dx=float(np.median(s[:, 1])),
                std_dy=round(float(s[:, 0].std()), 2), std_dx=round(float(s[:, 1].std()), 2),
                share_within_2px_of_global=round(float(np.mean((abs(s[:, 0] - G[0]) <= 2) & (abs(s[:, 1] - G[1]) <= 2))), 3))
    if sh["oracle"]:
        s = np.array(sh["oracle"])
        big = s[:, 2] >= np.median(s[:, 2])
        res["oracle_dy_small_vs_large_bldgs"] = [float(np.median(s[~big, 0])), float(np.median(s[big, 0]))]
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="data/hand_r12b2/index.csv")
    p.add_argument("--probs", default="diagnostics/hand_labels/ensemble_budget_probs.npz")
    p.add_argument("--pred_key", default="r34_c0_shift+tta")
    p.add_argument("--thr", type=float, default=0.4)
    p.add_argument("--S", type=int, default=14)
    p.add_argument("--min_px", type=int, default=150)
    p.add_argument("--border", type=int, default=24)
    p.add_argument("--out", default="experiments/2026-11-20-toward-higher-iou/per_building_offset.json")
    main(p.parse_args())
