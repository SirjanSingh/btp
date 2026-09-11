#!/usr/bin/env python
"""
d13_extra_predictions.py — what are the model's small predictions, if not buildings OB knows about?

WHY. D12 found the model emits 2,946 small components against 1,630 small labels
-- 1.8x more -- while D11 found 62% of small labels get essentially no
prediction. The model is finding small things, and they are largely *different*
small things. Two readings were left open:

  (a) they are genuine buildings Open Buildings missed, or
  (b) they are fragments shed from the edges of larger roofs.

This is decidable without ground truth, because it is a question about the
*spatial relationship* between predictions and labels, not about which is right.
Each small predicted component is classified as:

  isolated  -- overlaps no label at all. Candidate unlabelled building, or noise.
  fragment  -- overlaps a label that some OTHER, larger prediction already
               covers. The label was already found; this is a shed piece.
  sole      -- overlaps a label that no other prediction covers. The model's
               only detection of that building.

`fragments_per_label` (0.9203) hinted against (b) but could not separate these,
because it counts components touching labels without asking whether the label was
already covered.

Renders a sample of the ISOLATED ones, since those are the ones whose status
cannot be settled by counting and has to be looked at -- the same move that made
D10 productive.

Usage:
    python scripts/d13_extra_predictions.py --ckpt <best.pth> --encoder mit_b2 \
        --max_area_px 400 --n 10 --out /tmp/extra_preds
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

    counts = {"isolated": 0, "fragment": 0, "sole": 0}
    isolated = []

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
            n_p, p_lbl, p_stats, _ = cv2.connectedComponentsWithStats(pred, connectivity=8)
            n_g, g_lbl = cv2.connectedComponents(g, connectivity=8)
            if n_p <= 1:
                continue
            p_area = np.bincount(p_lbl.ravel(), minlength=n_p)
            g_area = np.bincount(g_lbl.ravel(), minlength=n_g) if n_g > 1 else np.zeros(1)

            # Joint histogram of (pred component, gt component) over shared pixels.
            both = (p_lbl > 0) & (g_lbl > 0)
            pairs = (p_lbl[both].astype(np.int64) * n_g + g_lbl[both].astype(np.int64))
            cnt = np.bincount(pairs) if pairs.size else np.zeros(1, np.int64)
            nz = np.nonzero(cnt)[0]

            p_to_g = {}          # pred -> {gt: overlap}
            g_cov_by = {}        # gt -> list of (pred, overlap)
            for code in nz:
                pi, gi = divmod(int(code), n_g)
                if pi == 0 or gi == 0:
                    continue
                p_to_g.setdefault(pi, {})[gi] = int(cnt[code])
                g_cov_by.setdefault(gi, []).append((pi, int(cnt[code])))

            for pi in range(1, n_p):
                area = int(p_area[pi])
                if area < a.min_area_px or area > a.max_area_px:
                    continue
                hits = p_to_g.get(pi)
                if not hits:
                    counts["isolated"] += 1
                    x, y, w, h = p_stats[pi, :4]
                    isolated.append((area, n, (int(x), int(y), int(w), int(h))))
                    continue
                # Is some OTHER prediction already covering this label well?
                covered_elsewhere = False
                for gi in hits:
                    for (pj, ov) in g_cov_by.get(gi, []):
                        if pj != pi and ov >= a.min_overlap * max(g_area[gi], 1):
                            covered_elsewhere = True
                            break
                    if covered_elsewhere:
                        break
                counts["fragment" if covered_elsewhere else "sole"] += 1
        if i % (a.batch * 40) == 0:
            print(f"  {i}/{len(names)}  {counts}")

    tot = max(sum(counts.values()), 1)
    print(f"\nsmall predictions ({a.min_area_px}-{a.max_area_px} px): {tot}")
    for k in ("isolated", "fragment", "sole"):
        print(f"  {k:<10}{counts[k]:>7}{100*counts[k]/tot:>8.1f}%")

    rng = np.random.default_rng(a.seed)
    picked = []
    if isolated:
        idx = rng.choice(len(isolated), size=min(a.n, len(isolated)), replace=False)
        picked = [isolated[i] for i in sorted(idx)]

    meta = []
    for k, (area, name, (x, y, w, h)) in enumerate(picked):
        im = cv2.imread(os.path.join(img_dir, name))
        gt = cv2.imread(os.path.join(msk_dir, name), cv2.IMREAD_GRAYSCALE)
        xi = ((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
               - MEAN) / STD).transpose(2, 0, 1)
        pmm = torch.sigmoid(m(torch.from_numpy(xi[None]).to(dev)))[0, 0].cpu().numpy()
        pred = (pmm > a.threshold).astype(np.uint8)
        pad, H, W = a.pad, im.shape[0], im.shape[1]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        over = im[y0:y1, x0:x1].copy()
        for msk, col in ((gt[y0:y1, x0:x1] > 127, (255, 200, 0)),
                         (pred[y0:y1, x0:x1] > 0, (255, 0, 200))):
            cnts, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, col, 1)
        stem = f"{k:02d}_{name[:-4]}"
        cv2.imwrite(os.path.join(a.out, stem + "__over.jpg"), over,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        meta.append({"rank": k, "crop": name, "area_px": area,
                     "area_m2": round(area * 0.266 ** 2, 1),
                     "file": stem + "__over.jpg"})
        print(f"  {k:02d} {name} area {area} px ({area*0.266**2:.0f} m2)")

    json.dump({"checkpoint": a.ckpt, "threshold": a.threshold,
               "size_range_px": [a.min_area_px, a.max_area_px],
               "counts": counts, "n_total": tot, "rendered": meta},
              open(os.path.join(a.out, "index.json"), "w"), indent=2)
    print(f"[ok] {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_overlap", type=float, default=0.5)
    p.add_argument("--min_area_px", type=int, default=50)
    p.add_argument("--max_area_px", type=int, default=400)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--pad", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="/tmp/extra_preds")
    main(p.parse_args())
