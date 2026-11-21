#!/usr/bin/env python3
"""
build_solar_neg.py - solar training set WITH no-panel images (run on the DGX host, not in docker).

Why: every solar checkpoint we had was trained on BDAPPV crops that all contain a panel
(prep_bdappv.py skips images with no mask). On Jaipur they flag canals, metal-sheet roofs and
tarps as panels in 135-151 of 152 crops. The model never saw a "no panel" example.

What it writes (real files, no symlinks, so it works inside the container), under
/tmp/btp_data/solar_neg, which run_docker.sh mounts at data/solar_neg:
  train/images, train/masks   BDAPPV Google positives (same split as data/bdappv_split)
                              + BDAPPV Google images that have no mask (negatives, no mask file:
                                train_solar.py uses an all-black mask for those)
                              + Jaipur patches with no Open Buildings polygon at any confidence
                                (fields, canals, roads), 154x154 px cut and resized to 400x400,
                                i.e. upsampled 2.6x to BDAPPV Google's pixel size
  val/...                     the same three sources, held out (BDAPPV by hash, Jaipur by tile)

Usage: python3 build_solar_neg.py /tmp/bdappv_23ucs715/bdappv.zip
"""
import hashlib
import os
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

ZIP = sys.argv[1]
OUT = "/tmp/btp_data/solar_neg"
BTP = os.path.expanduser("~/btp")
JAIPUR_TRAIN_CAP, JAIPUR_VAL_CAP = 4000, 600
PATCH = 154          # 154 px * 0.266 m = 41 m, about the 400 px * 0.1 m BDAPPV Google footprint


def is_val(name):
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0


def mk(*p):
    d = os.path.join(OUT, *p)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    for s in ("train", "val"):
        mk(s, "images"); mk(s, "masks")
    n = dict(pos_train=0, pos_val=0, neg_train=0, neg_val=0, jai_train=0, jai_val=0)

    # 1. Positives: copy the existing Google split (resolving its symlinks).
    for src, dst in (("google_train", "train"), ("google_val", "val")):
        for sub in ("images", "masks"):
            sd = os.path.join(BTP, "data/bdappv_split", src, sub)
            for f in os.listdir(sd):
                shutil.copyfile(os.path.realpath(os.path.join(sd, f)), os.path.join(OUT, dst, sub, f))
        n[f"pos_{dst}"] = len(os.listdir(os.path.join(BTP, "data/bdappv_split", src, "images")))
    print("positives", n, flush=True)

    # 2. BDAPPV Google negatives: images in google/img with no file in google/mask.
    z = zipfile.ZipFile(ZIP)
    names = z.namelist()
    img = {os.path.splitext(os.path.basename(p))[0]: p for p in names
           if "/google/img/" in p and p.lower().endswith((".png", ".jpg"))}
    msk = {os.path.splitext(os.path.basename(p))[0] for p in names
           if "/google/mask/" in p and p.lower().endswith((".png", ".jpg"))}
    neg = sorted(set(img) - msk)
    print(f"zip: {len(img)} google images, {len(msk)} masks, {len(neg)} negatives", flush=True)
    if not img:
        raise SystemExit("no google/img entries found; zip layout differs, check namelist()")
    for k in neg:
        split = "val" if is_val(k) else "train"
        with z.open(img[k]) as fh:
            im = Image.open(fh).convert("RGB")
            if im.size != (400, 400):
                im = im.resize((400, 400), Image.BILINEAR)
            im.save(os.path.join(OUT, split, "images", f"googleneg_{k}.png"))
        n[f"neg_{split}"] += 1
    print("bdappv negatives", n, flush=True)

    # 3. Jaipur hard negatives: crops with no Open Buildings polygon at any confidence.
    hand = set()
    for ann in ("khushi", "priyal", "sirjan"):
        d = os.path.join(BTP, "labelling/r12_batch2", ann, "images")
        if os.path.isdir(d):
            hand |= {os.path.splitext(f)[0] for f in os.listdir(d)}
    rng = np.random.default_rng(0)
    for split, cap in (("train", JAIPUR_TRAIN_CAP), ("val", JAIPUR_VAL_CAP)):
        md = os.path.join(BTP, "data/jaipur_weak_c0", split, "masks")
        idir = os.path.join(BTP, "data/jaipur_weak", split, "images")
        files = sorted(os.listdir(md))
        rng.shuffle(files)
        for f in files:
            if n[f"jai_{split}"] >= cap:
                break
            if os.path.splitext(f)[0] in hand:
                continue                      # never train on an eval crop
            if np.asarray(Image.open(os.path.join(md, f))).max() > 0:
                continue
            im = np.asarray(Image.open(os.path.join(idir, f)).convert("RGB"))
            for i in range(3):
                for j in range(3):
                    p = im[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]
                    if (p.max(axis=2) < 10).mean() > 0.2:
                        continue              # mosaic edge
                    Image.fromarray(p).resize((400, 400), Image.BILINEAR).save(
                        os.path.join(OUT, split, "images", f"jaipurneg_{os.path.splitext(f)[0]}_{i}{j}.png"))
                    n[f"jai_{split}"] += 1
    print("final", n, flush=True)
    import json
    json.dump(n, open(os.path.join(OUT, "counts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
