#!/usr/bin/env python
"""
crossearth_teacher_maps.py - run a CrossEarth checkpoint on UNLABELLED Jaipur tiles and save the
roof probability as an 8-bit PNG (value/255 = P(roof)). These soft maps are the teacher for
soft-label self-training (the UDA step): no Jaipur label is read anywhere here.

Tiles that appear in the hand-label index are skipped (so the student never trains on the
152 evaluation crops, even unlabelled).

Run one process per GPU with --shard i --nshards n.
  python scripts/crossearth_teacher_maps.py --dirs data/jaipur_weak/train/images data/jaipur_weak/val/images \
     --config CrossEarth/configs/jaipur_building_512.py --checkpoint CrossEarth/checkpoints/WHU-Building-A2S.pth \
     --out data/jaipur_teacher_a2s --gpu 1 --shard 0 --nshards 5 --tta 4
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch


def dihedral_k(fn, img, k_list):
    """Mean of fn over a subset of the 8 dihedral transforms; k_list = list of (rot, flip)."""
    acc = 0
    for k, flip in k_list:
        t = np.rot90(img, k)
        if flip:
            t = t[:, ::-1]
        p = fn(np.ascontiguousarray(t))
        if flip:
            p = p[:, ::-1]
        acc = acc + np.rot90(p, -k)
    return acc / len(k_list)


TTA = {1: [(0, False)],
       2: [(0, False), (0, True)],
       4: [(0, False), (1, False), (2, False), (3, False)],
       8: [(k, f) for k in range(4) for f in (False, True)]}


def main(a):
    sys.path.insert(0, os.path.abspath(a.repo))
    import CrossEarth  # noqa: F401
    import rs_dataset  # noqa: F401
    from mmseg.apis import init_model, inference_model

    skip = set()
    if a.exclude_index:
        skip = {r["name"] for r in csv.DictReader(open(a.exclude_index))}
    items = []
    for d in a.dirs:
        for n in sorted(os.listdir(d)):
            if n.endswith(".png") and n not in skip:
                items.append((d, n))
    items = items[a.shard::a.nshards]
    if a.limit:
        items = items[:a.limit]
    os.makedirs(a.out, exist_ok=True)

    model = init_model(a.config, a.checkpoint, device=f"cuda:{a.gpu}")
    ids = a.roof_class_ids

    def predict(img):
        r = inference_model(model, img)
        p = torch.softmax(r.seg_logits.data.float(), 0)
        return p[ids].sum(0).cpu().numpy()

    t0 = time.time()
    done = 0
    for d, n in items:
        out = os.path.join(a.out, n)
        if os.path.isfile(out):
            continue
        img = cv2.imread(os.path.join(d, n), cv2.IMREAD_COLOR)
        p = dihedral_k(predict, img, TTA[a.tta])
        cv2.imwrite(out, np.clip(np.round(p * 255), 0, 255).astype(np.uint8))
        done += 1
        if done % 50 == 0:
            print(f"shard {a.shard}: {done}/{len(items)}  {(time.time()-t0)/done:.2f}s/tile", flush=True)
    print(f"shard {a.shard} done {done} tiles in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--repo", default=os.path.expanduser("~/crossearth_work/CrossEarth"))
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--roof_class_ids", type=int, nargs="+", default=[0])
    p.add_argument("--exclude_index", default="data/hand_r12b2/index.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--tta", type=int, default=4, choices=(1, 2, 4, 8))
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
