#!/usr/bin/env python
"""
export_label_batches.py — cut a fresh, stratified set of Jaipur crops for
hand-labelling and split it evenly between labellers.

WHY A SECOND SCRIPT. r12_select_label_tiles.py picks from a pre-built crop set
(data/jaipur_weak), whose images lived on host /tmp and were wiped by the
2026-10-08 reboot. Rebuilding all 9,072 crops (4.2 GB) to keep 200 of them is a
waste of a tight quota, so this reads the 200 windows straight out of the
GeoTIFFs instead. The crop grid is taken from the surviving eroded mask set,
which shares names and positions with the original, so crop names stay
comparable with batch 1.

SAME PROTOCOL AS BATCH 1. Open Buildings masks (confidence >= 0.75, un-eroded,
identical to batch 1's reference) are written to openbuildings_reference/, and
labels/ starts empty. Draw first, compare after; see r12_select_label_tiles.py
for why pre-filling would make the result meaningless.

SPLIT. Each labeller gets the same number of crops from every density bin, so
any later difference between their labels is not a difference in what they
were given.

Usage (inside run_docker.sh):
    python scripts/export_label_batches.py --n 200 \
        --people sirjan,priyal --out labelling/r12_batch2
"""
import argparse
import csv
import json
import os
import random

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.windows import Window

from make_weak_labels import load_polygons
from r12_select_label_tiles import BINS, bin_of

CROP = 512


def candidates(grid_dir, exclude):
    """(tile, top, left) for every non-blank crop on the 512-stride grid."""
    out = []
    for split in ("train", "val"):
        d = os.path.join(grid_dir, split, "masks")
        for name in sorted(os.listdir(d)):
            if name in exclude:
                continue
            # "map67_1-2_000000_000512.png" -> tile, top, left
            parts = os.path.splitext(name)[0].split("_")
            out.append(dict(name=name, tile="_".join(parts[:2]), split=split,
                            top=int(parts[2]), left=int(parts[3])))
    return out


def main(a):
    random.seed(a.seed)
    people = a.people.split(",")
    exclude = set()
    with open(a.exclude_manifest) as fh:
        exclude = {r["name"] for r in csv.DictReader(fh)}

    cands = candidates(a.grid, exclude)
    print(f"[batch] {len(cands):,} candidate crops ({len(exclude)} excluded from batch 1)")
    geoms, bboxes = load_polygons(a.csv, a.min_conf)

    # Measure OB density for every candidate, one tile at a time.
    by_tile = {}
    for c in cands:
        by_tile.setdefault(c["tile"], []).append(c)
    masks = {}
    for tile, cs in sorted(by_tile.items()):
        with rasterio.open(os.path.join(a.tiles, tile + ".tif")) as src:
            b = src.bounds
            sel = ((bboxes[:, 0] <= b.right) & (bboxes[:, 2] >= b.left)
                   & (bboxes[:, 1] <= b.top) & (bboxes[:, 3] >= b.bottom))
            shapes = [(geoms[i], 1) for i in np.nonzero(sel)[0]]
            full = rasterize(shapes, out_shape=(src.height, src.width),
                             transform=src.transform, fill=0, dtype=np.uint8) \
                if shapes else np.zeros((src.height, src.width), np.uint8)
        for c in cs:
            m = full[c["top"]:c["top"] + CROP, c["left"]:c["left"] + CROP]
            c["fg_fraction"] = float(m.mean())
            c["bin"] = bin_of(c["fg_fraction"])
            masks[c["name"]] = m
        # Keep only what could be picked; a full tile mask is ~155 MB.
        del full
        print(f"   {tile:12s} {len(cs):5d} crops measured", flush=True)

    by_bin = {}
    for c in cands:
        by_bin.setdefault(c["bin"], []).append(c)

    per_bin = a.n // len(BINS)
    picked = []
    for label, _, _ in BINS:
        pool = by_bin.get(label, [])
        random.shuffle(pool)
        # Round-robin over parent tiles so no bin comes from one corner of the city.
        tiles = {}
        for c in pool:
            tiles.setdefault(c["tile"], []).append(c)
        order, queues = [], list(tiles.values())
        while queues and len(order) < per_bin:
            for q in list(queues):
                if q:
                    order.append(q.pop())
                if not q:
                    queues.remove(q)
                if len(order) >= per_bin:
                    break
        if len(order) < per_bin:
            print(f"[warn] bin '{label}' has only {len(order)} crops")
        # Deal each bin out alternately so every labeller gets the same mix.
        for i, c in enumerate(order):
            c["person"] = people[i % len(people)]
        picked += order

    for p in people:
        for sub in ("images", "labels", "openbuildings_reference"):
            os.makedirs(os.path.join(a.out, p, sub), exist_ok=True)
    for tile in sorted({c["tile"] for c in picked}):
        with rasterio.open(os.path.join(a.tiles, tile + ".tif")) as src:
            for c in (c for c in picked if c["tile"] == tile):
                img = src.read((1, 2, 3), window=Window(c["left"], c["top"], CROP, CROP))
                img = np.transpose(img, (1, 2, 0))
                d = os.path.join(a.out, c["person"])
                Image.fromarray(img).save(os.path.join(d, "images", c["name"]))
                Image.fromarray(masks[c["name"]] * 255).save(
                    os.path.join(d, "openbuildings_reference", c["name"]))
                c["n_buildings_ob"] = int(cv2.connectedComponents(masks[c["name"]])[0] - 1)

    random.shuffle(picked)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "tile", "split", "bin", "fg_fraction",
                                           "n_buildings_ob", "person"])
        w.writeheader()
        for c in sorted(picked, key=lambda c: (c["person"], c["name"])):
            w.writerow({k: c[k] for k in w.fieldnames})

    summary = {
        "seed": a.seed,
        "n_candidates": len(cands),
        "excluded_from": a.exclude_manifest,
        "n_selected": len(picked),
        "min_confidence": a.min_conf,
        "bin_population": {k: len(v) for k, v in sorted(by_bin.items())},
        "per_person": {p: {lbl: sum(1 for c in picked if c["person"] == p and c["bin"] == lbl)
                           for lbl, _, _ in BINS} for p in people},
        "tiles_covered": sorted({c["tile"] for c in picked}),
        "protocol": "labels/ starts EMPTY and must be drawn without consulting "
                    "openbuildings_reference/",
    }
    json.dump(summary, open(os.path.join(a.out, "selection_summary.json"), "w"), indent=2)
    print(json.dumps(summary["per_person"], indent=1))
    print(f"[ok] {a.out}  ({len(picked)} crops, {len(summary['tiles_covered'])} tiles)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="data/jaipur_weak_erode")
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--csv", default="data/open_buildings/jaipur_open_buildings.csv")
    p.add_argument("--exclude_manifest", default="labelling/r12_batch1/manifest.csv")
    p.add_argument("--out", default="labelling/r12_batch2")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--people", default="sirjan,priyal")
    p.add_argument("--min_conf", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
