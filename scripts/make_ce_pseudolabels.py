#!/usr/bin/env python
"""
make_ce_pseudolabels.py - turn averaged CrossEarth teacher soft maps into mmseg pseudo-label masks
for self-training CrossEarth itself (UDA, no Jaipur label of any kind, no Open Buildings).

  p = mean of the teacher dirs' soft maps (8-bit PNG, value/255 = P(roof))
  label 0   (Building)   where p >= --hi
  label 1   (Background) where p <= --lo
  label 255 (ignore)     in between  -> uncertain boundary pixels are not trained on

Writes <out>/img (symlinks to the tiles) and <out>/lbl (PNG labels). Only tiles that have a map in
every teacher dir are used, so the 152 hand crops (excluded when the teachers were generated) stay out.
Class order matches the WHU checkpoints: ['Building', 'Background'].
"""
import argparse
import os

import cv2
import numpy as np


def main(a):
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "lbl"), exist_ok=True)
    names = sorted(n for n in os.listdir(a.teacher[0])
                   if n.endswith(".png") and all(os.path.isfile(os.path.join(d, n)) for d in a.teacher))
    fb = fi = 0.0
    for k, n in enumerate(names):
        p = np.mean([cv2.imread(os.path.join(d, n), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                     for d in a.teacher], 0)
        lab = np.full(p.shape, 255, np.uint8)
        lab[p >= a.hi] = 0
        lab[p <= a.lo] = 1
        cv2.imwrite(os.path.join(a.out, "lbl", n), lab)
        src = next(os.path.abspath(os.path.join(d, n)) for d in a.img_dirs if os.path.isfile(os.path.join(d, n)))
        dst = os.path.join(a.out, "img", n)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
        fb += (lab == 0).mean()
        fi += (lab == 255).mean()
    print(f"{len(names)} tiles | building {fb/len(names):.3f} | ignored {fi/len(names):.3f} | "
          f"hi {a.hi} lo {a.lo}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", nargs="+", required=True)
    p.add_argument("--img_dirs", nargs="+", default=["data/jaipur_weak/train/images", "data/jaipur_weak/val/images"])
    p.add_argument("--hi", type=float, default=0.55)
    p.add_argument("--lo", type=float, default=0.25)
    p.add_argument("--out", required=True)
    main(p.parse_args())
