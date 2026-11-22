#!/usr/bin/env python
"""
sam_refine_teacher.py - sharpen a teacher's soft roof map with SAM, with NO Open Buildings and no
Jaipur label: the prompts come from the teacher's own building blobs.

For one tile: binarise the (mean) teacher probability at --thr, split into connected components,
prompt SAM ViT-H with each component's padded box, and keep SAM's outline when it agrees with the
blob (IoU >= tau, area ratio in [amin, amax]); otherwise keep the blob. Around every accepted
instance the refined map is  (1-w) * p + w * SAM_mask ; elsewhere it stays p.

Modes
  gate   hand crops only. Prints calib/test IoU (threshold chosen on calib) for the plain teacher
         and for a grid of (thr, tau, w). Pick settings on CALIB; test column is for the record.
  build  writes 8-bit refined soft maps for all tiles that have teacher maps in every --teacher dir.

Run inside run_docker.sh with PYTHONPATH=data/iou/pylib (segment_anything lives there).
"""
import argparse
import csv
import itertools
import json
import os
import sys

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_crossearth_hand import FINE_SWEEP, agg, counts, load_bin  # noqa: E402


class Sam:
    def __init__(self, ckpt, dev):
        from segment_anything import SamPredictor, sam_model_registry
        self.p = SamPredictor(sam_model_registry["vit_h"](checkpoint=ckpt).to(dev))
        self.dev = dev

    def set(self, img_rgb):
        self.p.set_image(img_rgb)

    @torch.no_grad()
    def masks(self, comps, pad):
        if not comps:
            return []
        H, W = comps[0].shape
        bx = []
        for m in comps:
            ys, xs = np.nonzero(m)
            bx.append([max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                       min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1)])
        bt = self.p.transform.apply_boxes_torch(torch.tensor(bx, dtype=torch.float32, device=self.dev), (H, W))
        out = []
        for i in range(0, len(bt), 64):
            m, _, _ = self.p.predict_torch(None, None, boxes=bt[i:i + 64], multimask_output=False)
            out.extend(m[:, 0].cpu().numpy())
        return out


def components(p, thr, min_px):
    lab, n = ndi.label(p > thr)
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= min_px:
            comps.append(m)
    return comps


def refine(p, comps, sam_masks, tau, amin, amax, w):
    cov = np.zeros(p.shape, bool)
    sm = np.zeros(p.shape, np.float32)
    k = np.ones((7, 7), np.uint8)
    for c, s in zip(comps, sam_masks):
        a, r = s.sum(), c.sum()
        if r == 0 or a < amin * r or a > amax * r:
            continue
        if (s & c).sum() / max((s | c).sum(), 1) < tau:
            continue
        cov |= cv2.dilate((c | s).astype(np.uint8), k).astype(bool)
        sm = np.maximum(sm, s.astype(np.float32))
    return np.where(cov, (1 - w) * p + w * sm, p)


def score(prob, hand, cal, tst):
    cs = {t: counts(prob, hand, t) for t in FINE_SWEEP}
    thr = max(FINE_SWEEP, key=lambda t: agg(cs[t][cal])[0])
    return thr, agg(cs[thr][cal])[0], agg(cs[thr][tst])[0], agg(cs[thr][tst])[1]


def mean_teacher_png(dirs, name):
    return np.mean([cv2.imread(os.path.join(d, name), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                    for d in dirs], 0)


def main(a):
    dev = "cuda"
    sam = Sam(a.sam_ckpt, dev)
    if a.mode == "gate":
        rows = list(csv.DictReader(open(a.index)))
        hand = [load_bin(r["mask"]) for r in rows]
        cal = [i for i, r in enumerate(rows) if r["half"] == "calib"]
        tst = [i for i, r in enumerate(rows) if r["half"] == "test"]
        z = [np.load(f) for f in a.hand_probs]
        P = [np.mean([zz[zz.files[-1]].astype(np.float32)[i] for zz in z], 0) for i in range(len(rows))]
        print("teacher (plain)      thr %.2f calib %.3f TEST %.3f area %.2f" % score(P, hand, cal, tst), flush=True)
        thrs, taus, ws = (0.3, 0.4, 0.5), (0.4, 0.6), (0.5, 1.0)
        cache = {t: [] for t in thrs}
        for i, r in enumerate(rows):
            img = cv2.cvtColor(cv2.imread(r["img"]), cv2.COLOR_BGR2RGB)
            sam.set(img)
            for t in thrs:
                cs = components(P[i], t, a.min_px)
                cache[t].append((cs, sam.masks(cs, a.pad)))
            if i % 30 == 0:
                print(f"{i}/{len(rows)}", flush=True)
        res = {}
        for t, tau, w in itertools.product(thrs, taus, ws):
            R = [refine(P[i], *cache[t][i], tau, a.amin, a.amax, w) for i in range(len(rows))]
            s = score(R, hand, cal, tst)
            res[f"thr{t}_tau{tau}_w{w}"] = s
            print(f"thr {t} tau {tau} w {w}  ->  thr %.2f calib %.3f TEST %.3f area %.2f" % s, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    else:
        names = sorted(n for n in os.listdir(a.teacher[0]) if all(os.path.isfile(os.path.join(d, n)) for d in a.teacher))
        names = names[a.shard::a.nshards]
        os.makedirs(a.out, exist_ok=True)
        imgdirs = a.img_dirs
        for k, n in enumerate(names):
            op = os.path.join(a.out, n)
            if os.path.isfile(op):
                continue
            ip = next(os.path.join(d, n) for d in imgdirs if os.path.isfile(os.path.join(d, n)))
            p = mean_teacher_png(a.teacher, n)
            sam.set(cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB))
            cs = components(p, a.thr, a.min_px)
            r = refine(p, cs, sam.masks(cs, a.pad), a.tau, a.amin, a.amax, a.w)
            cv2.imwrite(op, np.clip(np.round(r * 255), 0, 255).astype(np.uint8))
            if k % 100 == 0:
                print(f"shard {a.shard}: {k}/{len(names)}", flush=True)
        print(f"shard {a.shard} done", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("gate", "build"), required=True)
    p.add_argument("--sam_ckpt", default="data/iou/sam_vit_h_4b8939.pth")
    p.add_argument("--index", default="data/hand_r12b2/index.csv")
    p.add_argument("--hand_probs", nargs="+", help="npz files with hand-crop teacher probs (+tta last key)")
    p.add_argument("--teacher", nargs="+", help="build: teacher soft-map dirs (averaged)")
    p.add_argument("--img_dirs", nargs="+", default=["data/jaipur_weak/train/images", "data/jaipur_weak/val/images"])
    p.add_argument("--thr", type=float, default=0.4)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--w", type=float, default=0.5)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--amin", type=float, default=0.5)
    p.add_argument("--amax", type=float, default=2.0)
    p.add_argument("--min_px", type=int, default=80)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--out", required=True)
    main(p.parse_args())
