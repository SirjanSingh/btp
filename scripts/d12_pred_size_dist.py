#!/usr/bin/env python
"""
d12_pred_size_dist.py — does the model emit small components at all?

WHY THIS ONE IS NOT BLOCKED ON R12. D11 isolated the single real Stage-1 error:
62% of small labelled buildings receive essentially no prediction. Whether those
labels are genuine is blocked on hand labelling (D9 showed OB's own confidence
cannot arbitrate, being a size proxy). But a different question can be answered
today, because it depends only on the model's *own output*:

    across the whole val set, what sizes of connected component does the model
    actually produce?

If the model almost never emits a component below ~400 px anywhere, that is a
**structural** limit -- architecture, receptive field, or the /32 downsampling --
and it holds regardless of whether any particular small label is real. If it
emits plenty of small components but in the wrong places, the problem is
localisation, not capacity. Those imply different and differently expensive
fixes, and the distinction is free to measure.

Reported against the label size distribution on the same crops, so "the model
under-produces small components" is a comparison rather than an assertion.

Usage:
    python scripts/d12_pred_size_dist.py --ckpt <best.pth> --encoder mit_b2
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

EDGES = [50, 200, 400, 900, 2000, 10**9]
NAMES = ["50-200", "200-400", "400-900", "900-2000", "2000+"]


def binof(area):
    b = int(np.searchsorted(EDGES, area, side="right")) - 1   # PITFALLS 3.24
    return min(max(b, 0), len(NAMES) - 1)


@torch.no_grad()
def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = smp.Unet(a.encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model_state"])
    m = m.to(dev).eval()

    img_dir = os.path.join(a.val_dir, "images")
    msk_dir = os.path.join(a.val_dir, "masks")
    names = sorted(os.listdir(img_dir))

    n_pred = np.zeros(len(NAMES), np.int64)
    n_gt = np.zeros(len(NAMES), np.int64)
    pred_areas, gt_areas = [], []

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
            for mask, counter, store in (
                    ((pm > a.threshold).astype(np.uint8), n_pred, pred_areas),
                    ((gt > 127).astype(np.uint8), n_gt, gt_areas)):
                nc, lbl = cv2.connectedComponents(mask, connectivity=8)
                if nc <= 1:
                    continue
                areas = np.bincount(lbl.ravel(), minlength=nc)[1:]
                for ar in areas:
                    ar = int(ar)
                    if ar < a.min_area_px:
                        continue
                    counter[binof(ar)] += 1
                    store.append(ar)
        if i % (a.batch * 40) == 0:
            print(f"  {i}/{len(names)}")

    tp, tg = n_pred.sum(), n_gt.sum()
    print(f"\n{'size (px)':<12}{'labels':>9}{'lab %':>8}{'preds':>9}{'pred %':>8}{'pred/lab':>10}")
    for i, nm in enumerate(NAMES):
        lp = 100 * n_gt[i] / max(tg, 1)
        pp = 100 * n_pred[i] / max(tp, 1)
        ratio = n_pred[i] / max(n_gt[i], 1)
        print(f"{nm:<12}{n_gt[i]:>9}{lp:>7.1f}%{n_pred[i]:>9}{pp:>7.1f}%{ratio:>10.3f}")
    print(f"{'TOTAL':<12}{tg:>9}{'':>8}{tp:>9}")
    if pred_areas and gt_areas:
        print(f"\nmedian component  labels {int(np.median(gt_areas)):>6} px   "
              f"preds {int(np.median(pred_areas)):>6} px")

    out = {
        "checkpoint": a.ckpt, "encoder": a.encoder, "threshold": a.threshold,
        "min_area_px": a.min_area_px, "bin_edges_px": EDGES[:-1],
        "bins": [{"size_px": NAMES[i], "n_labels": int(n_gt[i]),
                  "n_preds": int(n_pred[i]),
                  "label_share": round(float(n_gt[i] / max(tg, 1)), 4),
                  "pred_share": round(float(n_pred[i] / max(tp, 1)), 4),
                  "pred_per_label": round(float(n_pred[i] / max(n_gt[i], 1)), 4)}
                 for i in range(len(NAMES))],
        "n_labels_total": int(tg), "n_preds_total": int(tp),
        "median_label_px": int(np.median(gt_areas)) if gt_areas else None,
        "median_pred_px": int(np.median(pred_areas)) if pred_areas else None,
        "note": "counts components anywhere on the tile; NOT a matched "
                "comparison -- a small predicted component need not correspond "
                "to a small label",
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
    p.add_argument("--min_area_px", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="diagnostics/d12_pred_size_dist.json")
    main(p.parse_args())
