#!/usr/bin/env python
"""
r12_select_label_tiles.py — choose the Jaipur crops to hand-label, and export a
labelling package.

WHY THIS EXISTS. Every Jaipur number in this repo is *agreement with Google Open
Buildings*, not accuracy. OB gives ground footprints, not roof outlines, and it
was itself produced by a model. Three separate results currently dead-end on
that: R3 (each label-confidence model wins on its own label distribution, so the
comparison is circular), the merge-rate lower bound, and every IoU quoted
anywhere. A few dozen honestly hand-drawn tiles convert all of them from
"agreement" into "accuracy".

THE TRAP THIS SCRIPT IS BUILT TO AVOID. The obvious move is to pre-fill each
canvas with the OB mask so the labeller only fixes mistakes. That would be much
faster and it would silently destroy the experiment: a labeller shown a mask
accepts most of it, so the resulting "ground truth" agrees with OB *by
construction*, and the one question we are trying to answer -- how wrong is OB?
-- becomes unanswerable. So the package ships the image and the OB mask in
SEPARATE directories, and `labels/` starts empty. Draw first, compare after.
The comparison is the result; it cannot also be the input.

SAMPLING. Stratified by OB foreground fraction, because building density in
Jaipur runs 10.7% to 41.2% across the 16 tiles and error modes are not uniform
across that range -- dense old-city blocks are where merging happens, sparse
outskirts are where false positives live. A uniform random draw of 30 crops
would over-represent the middle and could easily miss both tails. Empty crops
are sampled deliberately too (8.6% of the set): a tile with no buildings is the
only direct test of false-positive rate, and it costs seconds to label.

HELD-OUT SPLIT. `sealed/` is written but should not be looked at until the
method is frozen. Reporting on tiles you have already iterated against is how a
number drifts upward without the method improving.

Usage:
    python scripts/r12_select_label_tiles.py --n 30 --out labelling/r12_batch1
"""
import argparse
import csv
import json
import os
import random
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Density bins over the OB foreground fraction. Edges chosen from D1's measured
# per-tile range (0.107-0.412) rather than round numbers, plus an explicit empty
# bin -- see module docstring on why empties earn their slots.
BINS = [
    ("empty", 0.0, 0.001),
    ("sparse", 0.001, 0.15),
    ("moderate", 0.15, 0.27),
    ("dense", 0.27, 0.36),
    ("very_dense", 0.36, 1.01),
]


def crop_stats(mask_path):
    """Foreground fraction and connected-component count for one OB mask."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    b = (m > 127).astype(np.uint8)
    n_comp, _ = cv2.connectedComponents(b)
    return {
        "fg_fraction": float(b.mean()),
        "n_buildings_ob": int(n_comp - 1),  # label 0 is background
    }


def scan(split_dir):
    """Measure every crop in a split. Re-measured each run, never cached --
    the mask set is regenerated whenever erosion changes."""
    img_dir = os.path.join(split_dir, "images")
    msk_dir = os.path.join(split_dir, "masks")
    rows = []
    for name in sorted(os.listdir(img_dir)):
        mp = os.path.join(msk_dir, name)
        if not os.path.isfile(mp):
            continue
        st = crop_stats(mp)
        if st is None:
            continue
        # "map67_1-2_000000_000512.png" -> tile "map67_1-2"
        tile = "_".join(name.split("_")[:2])
        rows.append(dict(name=name, tile=tile, split=os.path.basename(split_dir),
                         img=os.path.join(img_dir, name), mask=mp, **st))
    return rows


def bin_of(frac):
    for label, lo, hi in BINS:
        if lo <= frac < hi:
            return label
    return BINS[-1][0]


def main(a):
    random.seed(a.seed)
    rows = []
    for split in ("train", "val"):
        d = os.path.join(a.crops, split)
        if os.path.isdir(d):
            rows += scan(d)
    if not rows:
        raise SystemExit(f"no crops found under {a.crops}")

    for r in rows:
        r["bin"] = bin_of(r["fg_fraction"])

    by_bin = {}
    for r in rows:
        by_bin.setdefault(r["bin"], []).append(r)

    # Equal allocation per bin, not proportional: the point is to characterise
    # error in each regime, and the rare regimes are the informative ones.
    per_bin = max(1, a.n // len(BINS))
    picked = []
    for label, _, _ in BINS:
        pool = by_bin.get(label, [])
        if not pool:
            print(f"[warn] bin '{label}' is empty -- no crops in that density range")
            continue
        # Spread across parent tiles so a bin is not filled from one corner of
        # the city; adjacent crops share content and would inflate agreement.
        random.shuffle(pool)
        seen_tiles, spread, rest = set(), [], []
        for r in pool:
            (spread if r["tile"] not in seen_tiles else rest).append(r)
            seen_tiles.add(r["tile"])
        picked += (spread + rest)[:per_bin]

    random.shuffle(picked)
    n_sealed = max(1, int(len(picked) * a.sealed_frac))
    sealed, working = picked[:n_sealed], picked[n_sealed:]

    for group, items in (("working", working), ("sealed", sealed)):
        for sub in ("images", "labels", "openbuildings_reference"):
            os.makedirs(os.path.join(a.out, group, sub), exist_ok=True)
        for r in items:
            shutil.copy2(r["img"], os.path.join(a.out, group, "images", r["name"]))
            # Reference only. Never copied into labels/ -- see docstring.
            shutil.copy2(r["mask"],
                         os.path.join(a.out, group, "openbuildings_reference", r["name"]))

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "tile", "split", "bin",
                                           "fg_fraction", "n_buildings_ob", "group"])
        w.writeheader()
        for group, items in (("working", working), ("sealed", sealed)):
            for r in items:
                w.writerow({k: r[k] for k in
                            ("name", "tile", "split", "bin", "fg_fraction",
                             "n_buildings_ob")} | {"group": group})

    summary = {
        "generated_from": a.crops,
        "seed": a.seed,
        "n_total_crops_scanned": len(rows),
        "n_selected": len(picked),
        "n_working": len(working),
        "n_sealed": len(sealed),
        "per_bin_target": per_bin,
        "bin_population": {k: len(v) for k, v in sorted(by_bin.items())},
        "bin_selected": {k: sum(1 for r in picked if r["bin"] == k)
                         for k, _, _ in BINS},
        "tiles_covered": sorted({r["tile"] for r in picked}),
        "ob_fg_fraction_selected_mean": round(
            float(np.mean([r["fg_fraction"] for r in picked])), 5),
        "protocol": "labels/ starts EMPTY and must be drawn without consulting "
                    "openbuildings_reference/ -- pre-filling makes the OB "
                    "accuracy question unanswerable by construction",
    }
    json.dump(summary, open(os.path.join(a.out, "selection_summary.json"), "w"),
              indent=2)

    print(f"[r12] scanned {len(rows)} crops, selected {len(picked)}")
    for k, _, _ in BINS:
        print(f"       {k:11s} pool={len(by_bin.get(k, [])):5d}  "
              f"selected={summary['bin_selected'][k]}")
    print(f"[r12] {len(working)} working / {len(sealed)} sealed  "
          f"across {len(summary['tiles_covered'])} parent tiles")
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crops", default="/tmp/btp_data/jaipur_weak")
    p.add_argument("--out", default="labelling/r12_batch1")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--sealed_frac", type=float, default=0.35,
                   help="fraction held out until the method is frozen")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
