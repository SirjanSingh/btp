#!/usr/bin/env python
"""
refine_labels.py - turn Open Buildings footprints into better roof labels, per building.

Three label variants, all built from the ORIGINAL (unshifted) OB polygons every time
(Girard et al. 2019: refining the original labels beats refining the previous output):

  align      each building moved on its own to best overlap the current model's prediction
             (r34_c0_shift + 8-way TTA, prob > 0.4). Search (dy, dx) in [-S, S]^2, ties go to
             the global shift (7, -3); buildings under --min_px keep the global shift.
             Idea from OMAF (CVPR 2026) / Align-and-Segment (2026), done without their code.
  sam        each building, at the global shift, becomes a box prompt (padded by --pad px) for
             SAM ViT-H; SAM's roof outline replaces the footprint when it agrees with it
             (IoU >= --tau and area ratio in [--amin, --amax]), otherwise the footprint stays.
  align_sam  SAM prompted with the per-building aligned box (both ideas together).

Modes
  gate   run on the 152 hand crops only and print pooled IoU vs the hand roofs for each
         variant and a small grid of SAM acceptance settings, per half. Pick settings on
         CALIB; the test column is for the record only.
  build  write <out_root>/lab_<variant>/train/masks/*.png for the training tiles, with
         train/images symlinked to the shared crops, ready for rooftop/train.py.

Usage (DGX, inside run_docker.sh):
  PYTHONPATH=data/iou/pylib python scripts/refine_labels.py --mode gate
  PYTHONPATH=data/iou/pylib python scripts/refine_labels.py --mode build --tau 0.5 --pad 4
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import find_objects

sys.path.insert(0, os.path.dirname(__file__))
from score_models_hand import MEAN, STD, load_model  # noqa: E402

G = (7, -3)


def shift(m, dy, dx):
    out = np.zeros_like(m)
    H, W = m.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = m[ys, xs]
    return out


@torch.no_grad()
def predict(model, imgs, dev, bs=16):
    """8-way dihedral TTA probabilities, same as the 0.706 headline model."""
    out = []
    for i in range(0, len(imgs), bs):
        x = np.stack([(im.astype(np.float32) / 255 - MEAN) / STD for im in imgs[i:i + bs]])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(dev)
        acc = 0
        for k in range(4):
            for f in (False, True):
                t = torch.rot90(x, k, (2, 3))
                t = torch.flip(t, (3,)) if f else t
                p = torch.sigmoid(model(t))
                p = torch.flip(p, (3,)) if f else p
                acc = acc + torch.rot90(p, -k, (2, 3))
        out.append((acc / 8)[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def instances(inst):
    """yield (id, bbox slices, area) for every building in a 16-bit instance crop."""
    for i, sl in enumerate(find_objects(inst), 1):
        if sl is not None:
            yield i, sl


def best_shift(comp, target, sl, S):
    """per-building shift maximising overlap with target; ties -> closest to G."""
    H, W = target.shape
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    pad = np.zeros((y1 - y0 + 2 * S, x1 - x0 + 2 * S), np.float32)
    ty0, tx0 = y0 - S, x0 - S
    a0, a1 = max(ty0, 0), min(y1 + S, H)
    b0, b1 = max(tx0, 0), min(x1 + S, W)
    pad[a0 - ty0:a1 - ty0, b0 - tx0:b1 - tx0] = target[a0:a1, b0:b1]
    r = cv2.matchTemplate(pad, comp.astype(np.float32), cv2.TM_CCORR)   # (2S+1, 2S+1)
    cand = np.argwhere(r >= r.max() - 1e-3) - S                          # rows = dy, cols = dx
    d = np.abs(cand[:, 0] - G[0]) + np.abs(cand[:, 1] - G[1])
    dy, dx = cand[d.argmin()]
    return int(dy), int(dx)


def building_masks(inst, target, S, min_px, do_align):
    """list of (global-shifted mask, aligned mask) per building, full-crop bool arrays."""
    out = []
    for i, sl in instances(inst):
        comp_full = inst == i
        area = int(comp_full.sum())
        g = shift(comp_full, *G)
        if do_align and area >= min_px:
            dy, dx = best_shift(comp_full[sl], target, sl, S)
            a = shift(comp_full, dy, dx)
        else:
            a = g
        out.append((g, a, area))
    return out


class Sam:
    def __init__(self, ckpt, dev):
        from segment_anything import SamPredictor, sam_model_registry
        self.p = SamPredictor(sam_model_registry["vit_h"](checkpoint=ckpt).to(dev))
        self.dev = dev

    def set(self, img):
        self.p.set_image(img)

    @torch.no_grad()
    def boxes(self, masks, pad):
        """one SAM mask per building mask, prompted with its padded bounding box."""
        if not masks:
            return []
        H, W = masks[0].shape
        bx = []
        for m in masks:
            ys, xs = np.nonzero(m)
            if len(ys) == 0:
                bx.append([0, 0, 1, 1])
                continue
            bx.append([max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                       min(xs.max() + pad, W - 1), min(ys.max() + pad, H - 1)])
        bt = torch.tensor(bx, dtype=torch.float32, device=self.dev)
        bt = self.p.transform.apply_boxes_torch(bt, (H, W))
        res = []
        for i in range(0, len(bt), 64):
            m, _, _ = self.p.predict_torch(None, None, boxes=bt[i:i + 64], multimask_output=False)
            res.extend(m[:, 0].cpu().numpy())
        return res


def accept(sam_m, ref, tau, amin, amax):
    a, r = sam_m.sum(), ref.sum()
    if r == 0 or a < amin * r or a > amax * r:
        return False
    return (sam_m & ref).sum() / max((sam_m | ref).sum(), 1) >= tau


def compose(blds, sams_g, sams_a, tau, amin, amax, min_sam_px):
    """the three label variants (+ the two baselines) for one crop."""
    H, W = (blds[0][0].shape if blds else (512, 512))
    out = {k: np.zeros((H, W), bool) for k in ("global", "align", "sam", "align_sam")}
    n_acc = {"sam": 0, "align_sam": 0, "n": 0}
    for j, (g, a, area) in enumerate(blds):
        out["global"] |= g
        out["align"] |= a
        big = area >= min_sam_px
        n_acc["n"] += big
        for key, ref, sams in (("sam", g, sams_g), ("align_sam", a, sams_a)):
            if big and sams is not None and accept(sams[j], ref, tau, amin, amax):
                out[key] |= sams[j]
                n_acc[key] += 1
            else:
                out[key] |= ref
    return out, n_acc


def rgb(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def inst_of(a, name):
    return cv2.imread(os.path.join(a.inst_dir, name), cv2.IMREAD_UNCHANGED)


def gate(a, model, sam, dev):
    rows = list(csv.DictReader(open(a.index)))
    imgs = [rgb(r["img"]) for r in rows]
    probs = predict(model, imgs, dev)
    grid = list(itertools.product(a.tau_grid, [a.amax]))
    acc = {}
    b = a.border
    for pad in a.pad_grid:
        for r, img, pr in zip(rows, imgs, probs):
            hand = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 127
            inst = inst_of(a, r["name"])
            blds = building_masks(inst, pr > a.thr, a.S, a.min_px, True)
            sam.set(img)
            big = [k for k, x in enumerate(blds) if x[2] >= a.min_sam_px]
            sg, sa = [None] * len(blds), [None] * len(blds)
            for k, m in zip(big, sam.boxes([blds[k][0] for k in big], pad)):
                sg[k] = m
            for k, m in zip(big, sam.boxes([blds[k][1] for k in big], pad)):
                sa[k] = m
            hh = hand[b:-b, b:-b]
            for tau, amax in grid:
                labs, _ = compose(blds, sg, sa, tau, a.amin, amax, a.min_sam_px)
                labs["pred"] = pr > a.thr
                # teacher-footprint fusion: w * teacher prob + (1 - w) * refined label, cut at 0.5
                # (the footprint lowers the teacher's threshold inside it and raises it outside)
                for w in a.fuse_grid:
                    labs[f"fuse{w}"] = (w * pr + (1 - w) * labs["align_sam"]) > 0.5
                for k, m in labs.items():
                    plain = k in ("global", "align", "pred")
                    key = k if plain else f"{k}_pad{pad}_tau{tau}"
                    mm = m[b:-b, b:-b]
                    c = acc.setdefault(key, {"calib": np.zeros(3), "test": np.zeros(3)})
                    if plain and (pad != a.pad_grid[0] or tau != grid[0][0]):
                        continue
                    c[r["half"]] += [(mm & hh).sum(), (mm & ~hh).sum(), (~mm & hh).sum()]
    res = {k: {h: round(float(v[0] / v.sum()), 4) for h, v in d.items()} for k, d in acc.items()}
    res = dict(sorted(res.items(), key=lambda kv: -kv[1]["calib"]))
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def build(a, model, sam, dev):
    src = "data/jaipur_weak/train/images"
    names = sorted(os.listdir(src))
    if a.limit:
        names = names[:a.limit]
    names = names[a.shard::a.nshard]
    dirs = {}
    for v in a.variants:
        d = os.path.join(a.out_root, f"lab_{v}", "train")
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)
        try:
            os.symlink("/workspace/" + src, os.path.join(d, "images"))
        except FileExistsError:
            pass
        dirs[v] = os.path.join(d, "masks")
    need_sam = any("sam" in v for v in a.variants)
    stats = {v: 0.0 for v in a.variants}
    tot = {"sam": 0, "align_sam": 0, "n": 0}
    t0 = time.time()
    for i0 in range(0, len(names), 64):
        chunk = names[i0:i0 + 64]
        imgs = [rgb(os.path.join(src, n)) for n in chunk]
        probs = predict(model, imgs, dev)
        for n, img, pr in zip(chunk, imgs, probs):
            inst = inst_of(a, n)
            blds = building_masks(inst, pr > a.thr, a.S, a.min_px, True) if inst is not None else []
            sg = sa = None
            if need_sam and blds:
                sam.set(img)
                big = [k for k, x in enumerate(blds) if x[2] >= a.min_sam_px]
                sg, sa = [None] * len(blds), [None] * len(blds)
                if "sam" in a.variants:
                    for k, m in zip(big, sam.boxes([blds[k][0] for k in big], a.pad)):
                        sg[k] = m
                if "align_sam" in a.variants:
                    for k, m in zip(big, sam.boxes([blds[k][1] for k in big], a.pad)):
                        sa[k] = m
            labs, na = compose(blds, sg, sa, a.tau, a.amin, a.amax, a.min_sam_px)
            for k in tot:
                tot[k] += na[k]
            for v in a.variants:
                cv2.imwrite(os.path.join(dirs[v], n), labs[v].astype(np.uint8) * 255)
                stats[v] += labs[v].mean()
        done = i0 + len(chunk)
        print(f"  {done}/{len(names)} crops  {time.time() - t0:.0f}s  "
              f"accepted sam {tot['sam']}/{tot['n']} align_sam {tot['align_sam']}/{tot['n']}", flush=True)
    summ = dict(variants=a.variants, n_crops=len(names), fg_fraction={v: round(s / len(names), 4) for v, s in stats.items()},
                sam_accepted=tot, params=dict(S=a.S, min_px=a.min_px, thr=a.thr, pad=a.pad, tau=a.tau,
                                              amin=a.amin, amax=a.amax, min_sam_px=a.min_sam_px, ckpt=a.ckpt))
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(os.path.join(a.out_root, f"build_summary_{a.shard}of{a.nshard}.json"), "w"), indent=1)


def pseudo(a, model, dev):
    """Self-training labels (no SAM): teacher TTA probabilities, alone and fused with an
    already-built refined label set. Written as 0..255 PNGs for train.py --soft_masks.
      soft_pred        teacher probability
      soft_fuse<w>     w * teacher + (1 - w) * lab_<fuse_src> (0/1)"""
    src = "data/jaipur_weak/train/images"
    names = sorted(os.listdir(src))[:a.limit or None]
    names = names[a.shard::a.nshard]
    outs = ["soft_pred"] + [f"soft_fuse{w}" for w in a.fuse_grid]
    dirs = {}
    for v in outs:
        d = os.path.join(a.out_root, f"lab_{v}", "train")
        os.makedirs(os.path.join(d, "masks"), exist_ok=True)
        try:
            os.symlink("/workspace/" + src, os.path.join(d, "images"))
        except FileExistsError:
            pass
        dirs[v] = os.path.join(d, "masks")
    lab_dir = os.path.join(a.out_root, f"lab_{a.fuse_src}", "train", "masks")
    t0 = time.time()
    for i0 in range(0, len(names), 64):
        chunk = names[i0:i0 + 64]
        probs = predict(model, [rgb(os.path.join(src, n)) for n in chunk], dev)
        for n, pr in zip(chunk, probs):
            lab = cv2.imread(os.path.join(lab_dir, n), cv2.IMREAD_GRAYSCALE) > 127
            cv2.imwrite(os.path.join(dirs["soft_pred"], n), np.round(pr * 255).astype(np.uint8))
            for w in a.fuse_grid:
                f = w * pr + (1 - w) * lab
                cv2.imwrite(os.path.join(dirs[f"soft_fuse{w}"], n), np.round(f * 255).astype(np.uint8))
        print(f"  {i0 + len(chunk)}/{len(names)} crops  {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(outs=outs, n=len(names), ckpt=a.ckpt, fuse_src=a.fuse_src),
              open(os.path.join(a.out_root, f"pseudo_summary_{a.shard}of{a.nshard}.json"), "w"))


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)
    model, _ = load_model(a.ckpt, dev)
    if a.mode == "pseudo":
        return pseudo(a, model, dev)
    sam = Sam(a.sam_ckpt, dev) if (a.mode == "gate" or any("sam" in v for v in a.variants)) else None
    (gate if a.mode == "gate" else build)(a, model, sam, dev)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gate", "build", "pseudo"], default="gate")
    p.add_argument("--fuse_grid", type=float, nargs="+", default=[0.6, 0.7, 0.8])
    p.add_argument("--fuse_src", default="align_sam")
    p.add_argument("--ckpt", default="experiments/2026-11-19-hand-label-model-scoring/ckpt/r34_c0_shift/"
                                     "unet_resnet34_40ep_20261119_172916/best.pth")
    p.add_argument("--sam_ckpt", default="data/iou/sam_vit_h_4b8939.pth")
    p.add_argument("--inst_dir", default="data/iou/inst")
    p.add_argument("--index", default="data/hand_r12b2/index.csv")
    p.add_argument("--out", default="experiments/2026-11-20-toward-higher-iou/refine_gate.json")
    p.add_argument("--out_root", default="data/iou")
    p.add_argument("--variants", nargs="+", default=["align", "sam", "align_sam"])
    p.add_argument("--thr", type=float, default=0.4)
    p.add_argument("--S", type=int, default=14)
    p.add_argument("--min_px", type=int, default=150)
    p.add_argument("--min_sam_px", type=int, default=60)
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--amin", type=float, default=0.5)
    p.add_argument("--amax", type=float, default=2.0)
    p.add_argument("--pad_grid", type=int, nargs="+", default=[2, 6])
    p.add_argument("--tau_grid", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    p.add_argument("--border", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    main(p.parse_args())
