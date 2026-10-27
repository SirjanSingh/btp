#!/usr/bin/env python
"""
d10_missed_gallery.py — show the LARGE buildings the model misses.

WHY LARGE ONLY. D8 found miss rate is 10.5x worse for small buildings, but D9
showed Open Buildings' confidence is essentially a size proxy -- at conf >= 0.85
only 20 small polygons survive city-wide -- so a missed small polygon is
ambiguous between "model failed" and "OB invented it". That ambiguity is blocked
on R12 hand labels.

Large polygons carry no such doubt: 93% of them clear the conf >= 0.75 filter,
and a 2000+ px footprint (>142 m^2) is not something OB hallucinates. So the
**884 missed large buildings are unambiguous model failures**, and unlike the
small ones they can be interpreted today. A handful of them rendered side by
side is the cheapest available route to a *named* failure mode -- dark roofs,
shadow, occlusion, construction -- rather than another aggregate.

WHAT IS DRAWN. Per building: the raw crop, the label outline, and the model's
prediction outline, on the same tile. Showing the prediction matters -- "missed"
under the >=50%-overlap rule includes cases where the model fired on part of the
roof but not enough of it, which looks completely different from predicting
nothing at all, and the fix differs too.

Usage:
    python scripts/d10_missed_gallery.py --ckpt <best.pth> --encoder mit_b2 \
        --min_area_px 2000 --n 12 --out .tmp/missed_gallery
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


@torch.no_grad()
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    img_dir = os.path.join(a.val_dir, "images")
    msk_dir = os.path.join(a.val_dir, "masks")
    names = sorted(os.listdir(img_dir))
    os.makedirs(a.out, exist_ok=True)

    found = []   # (area, crop_name, bbox, coverage_fraction)
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
            n_g, g_lbl, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
            if n_g <= 1:
                continue
            hit = np.bincount(g_lbl[pred > 0].ravel(), minlength=n_g)
            area_all = np.bincount(g_lbl.ravel(), minlength=n_g)
            for gi in range(1, n_g):
                area = int(area_all[gi])
                if area < a.min_area_px:
                    continue
                cov = hit[gi] / max(area, 1)
                if cov < a.min_overlap:
                    x, y, w, h = stats[gi, :4]
                    found.append((area, n, (int(x), int(y), int(w), int(h)),
                                  float(cov)))
        if i % (a.batch * 40) == 0:
            print(f"  scanned {i}/{len(names)}, {len(found)} missed so far")

    # Largest-first shows the most extreme cases, but it is a BIASED draw: the
    # biggest "missed buildings" are exactly the ones most likely to be OB
    # compound polygons rather than single roofs. --sample random is the check
    # that whatever pattern appears is not an artefact of the tail.
    if a.sample == "random":
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(found), size=min(a.n, len(found)), replace=False)
        picked = [found[i] for i in sorted(idx)]
    else:
        found.sort(key=lambda r: -r[0])
        picked = found[:a.n]
    print(f"[gallery] {len(found)} missed >= {a.min_area_px} px; rendering {len(picked)}")

    meta = []
    for k, (area, name, (x, y, w, h), cov) in enumerate(picked):
        im = cv2.imread(os.path.join(img_dir, name))
        gt = cv2.imread(os.path.join(msk_dir, name), cv2.IMREAD_GRAYSCALE)
        xi = ((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
               - MEAN) / STD).transpose(2, 0, 1)
        pmm = torch.sigmoid(m(torch.from_numpy(xi[None]).to(dev)))[0, 0].cpu().numpy()
        pred = (pmm > a.threshold).astype(np.uint8)

        pad = a.pad
        H, W = im.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        raw = im[y0:y1, x0:x1].copy()
        over = raw.copy()
        # Label in cyan, prediction in magenta -- distinguishable in both themes
        # and colour-blind-safe against each other.
        for msk, col in ((gt[y0:y1, x0:x1] > 127, (255, 200, 0)),
                         (pred[y0:y1, x0:x1] > 0, (255, 0, 200))):
            cnts, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, col, 2)

        stem = f"{k:02d}_{name[:-4]}"
        cv2.imwrite(os.path.join(a.out, stem + "__raw.jpg"), raw,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(os.path.join(a.out, stem + "__over.jpg"), over,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        meta.append({"rank": k, "crop": name, "area_px": area,
                     "area_m2": round(area * 0.266 ** 2, 1),
                     "coverage": round(cov, 4),
                     "bbox": [x, y, w, h],
                     "raw": stem + "__raw.jpg", "over": stem + "__over.jpg"})
        print(f"  {k:02d} {name} area {area} px ({area*0.266**2:.0f} m2) "
              f"coverage {cov:.3f}")

    json.dump({"checkpoint": a.ckpt, "threshold": a.threshold,
               "min_area_px": a.min_area_px, "min_overlap": a.min_overlap,
               "n_missed_total": len(found), "items": meta},
              open(os.path.join(a.out, "index.json"), "w"), indent=2)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_overlap", type=float, default=0.5)
    p.add_argument("--min_area_px", type=int, default=2000)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--pad", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--sample", choices=["largest","random"], default="largest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=".tmp/missed_gallery")
    main(p.parse_args())
