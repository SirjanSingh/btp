#!/usr/bin/env python
"""
d8_missed_by_size.py — which buildings does the model miss, by size?

WHY. The default model's missed rate is **0.3257**: a third of labelled buildings
get no prediction covering them. That is the largest single error left in Stage 1,
and every knob measured so far (erosion, inference threshold) only *trades* it
against merging rather than reducing it. Nothing has asked the obvious question —
**are the missed buildings small, or is the miss independent of size?**

The answer decides where effort goes next:
  - misses concentrated in small buildings -> a resolution problem; the median
    Jaipur building is ~30x30 px at 26.6 cm/px, and the smallest are near the
    limit of what a /32-downsampling encoder can represent.
  - misses spread evenly across sizes -> not resolution; look at contrast,
    shadow, or label error instead.

Those two lead to completely different (and differently expensive) experiments,
so measuring first is much cheaper than guessing.

METHOD. Reuses the association rule from `merge_split_rate.py`: a label counts as
found when some predicted component covers >= `min_overlap` of it. Labels are
binned by their own pixel area, and miss rate reported per bin alongside the
share of total misses each bin contributes -- both matter, since a bin can have a
high miss *rate* while holding few buildings.

Usage:
    python scripts/d8_missed_by_size.py --ckpt <best.pth> --encoder mit_b2
"""
import argparse
import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Bin edges in pixels. At 26.6 cm/px: 400 px ~ 28 m^2, 900 px ~ 64 m^2,
# 2000 px ~ 142 m^2. Chosen to straddle the ~913 px median measured in D2/D3.
EDGES = [50, 200, 400, 900, 2000, 10**9]
NAMES = ["50-200", "200-400", "400-900", "900-2000", "2000+"]


@torch.no_grad()
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    img_dir = os.path.join(a.val_dir, "images")
    msk_dir = os.path.join(a.val_dir, "masks")
    names = sorted(os.listdir(img_dir))
    if a.limit:
        # Stride, never a prefix -- crop names sort by parent tile, so a prefix
        # samples one corner of the city (PITFALLS 3.21).
        step = max(1, len(names) // a.limit)
        names = names[::step][:a.limit]

    n_bins = len(NAMES)
    total = np.zeros(n_bins, np.int64)
    missed = np.zeros(n_bins, np.int64)
    areas_missed, areas_found = [], []

    for i in range(0, len(names), a.batch):
        chunk = names[i:i + a.batch]
        xs = []
        for n in chunk:
            im = cv2.cvtColor(cv2.imread(os.path.join(img_dir, n)), cv2.COLOR_BGR2RGB)
            xs.append(((im.astype(np.float32) / 255. - MEAN) / STD).transpose(2, 0, 1))
        p = torch.sigmoid(m(torch.from_numpy(np.stack(xs)).to(dev)))[:, 0].cpu().numpy()

        for n, pm in zip(chunk, p):
            gt = cv2.imread(os.path.join(msk_dir, n), cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            pred = (pm > a.threshold).astype(np.uint8)
            g = (gt > 127).astype(np.uint8)
            n_g, g_lbl = cv2.connectedComponents(g, connectivity=8)
            if n_g <= 1:
                continue
            g_area = np.bincount(g_lbl.ravel(), minlength=n_g)

            # Overlap of each GT component with ANY predicted foreground.
            hit = np.bincount(g_lbl[pred > 0].ravel(), minlength=n_g)
            for gi in range(1, n_g):
                area = int(g_area[gi])
                if area < a.min_area_px:
                    continue
                # searchsorted returns the INSERTION index, which is one past
                # the bin the value belongs to -- without the -1 every label
                # lands a bin too high and the top bin silently absorbs two
                # ranges. The tell was an impossible empty 50-200 bin.
                b = int(np.searchsorted(EDGES, area, side="right")) - 1
                b = min(max(b, 0), n_bins - 1)
                total[b] += 1
                if hit[gi] < a.min_overlap * area:
                    missed[b] += 1
                    areas_missed.append(area)
                else:
                    areas_found.append(area)
        if i % (a.batch * 20) == 0:
            print(f"  {i}/{len(names)}")

    rate = np.where(total > 0, missed / np.maximum(total, 1), 0.0)
    share = missed / max(missed.sum(), 1)
    print(f"\n{'size (px)':<12}{'labels':>9}{'missed':>9}{'miss rate':>11}{'% of all misses':>17}")
    for i, nm in enumerate(NAMES):
        print(f"{nm:<12}{total[i]:>9}{missed[i]:>9}{rate[i]:>11.4f}{100*share[i]:>16.1f}%")
    print(f"{'ALL':<12}{total.sum():>9}{missed.sum():>9}"
          f"{missed.sum()/max(total.sum(),1):>11.4f}")
    if areas_missed and areas_found:
        print(f"\nmedian area missed : {int(np.median(areas_missed)):>6} px")
        print(f"median area found  : {int(np.median(areas_found)):>6} px")

    out = {
        "checkpoint": a.ckpt, "encoder": a.encoder, "val_dir": a.val_dir,
        "threshold": a.threshold, "min_overlap": a.min_overlap,
        "min_area_px": a.min_area_px, "bin_edges_px": EDGES[:-1],
        "bins": [{"size_px": NAMES[i], "n_labels": int(total[i]),
                  "n_missed": int(missed[i]), "miss_rate": round(float(rate[i]), 4),
                  "share_of_misses": round(float(share[i]), 4)} for i in range(n_bins)],
        "overall_miss_rate": round(float(missed.sum() / max(total.sum(), 1)), 4),
        "median_area_missed_px": int(np.median(areas_missed)) if areas_missed else None,
        "median_area_found_px": int(np.median(areas_found)) if areas_found else None,
        "note": "labels are Open Buildings footprints; a 'miss' may be an OB "
                "false positive rather than a model failure (R12)",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_overlap", type=float, default=0.5)
    p.add_argument("--min_area_px", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="diagnostics/d8_missed_by_size.json")
    main(p.parse_args())
