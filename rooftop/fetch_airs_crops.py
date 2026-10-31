#!/usr/bin/env python
"""
fetch_airs_crops.py — restore AIRS as 512x512 training crops without ever holding
the full ~10-15 GB of raw imagery (the home quota cannot fit it).

Per batch: download N raw images (and val labels) from the master Drive folder,
cut a capped, random set of crops per image, delete the raw files, repeat.
Resumable: images already recorded in <out>/done.txt are skipped.

Source of truth for file IDs: docs/drive_folder_listing.json (complete listing made
by scripts/drive_list.py; the old gdown listing truncates at ~50, PITFALLS 3.1).
Train labels are read from data/airs_full/DAtaset/dataset/train/label (already on
disk); val labels are downloaded.

Crop rule: 512 px grid (stride 512). Per image, (per_image - n_random) random crops with
roof coverage >= 0.005 (the rooftop/tile_airs.py rule) plus n_random crops drawn from all
positions regardless of roof, so empty ground is kept (no zero-negatives bug).
Names follow the repo convention <stem>_<row:06d>_<col:06d>.png.

Usage:
    python rooftop/fetch_airs_crops.py --out data/airs_crops_v2 --per_image 5 --batch 50
"""
import argparse
import json
import os
import random
import shutil
import time
import urllib.request

import cv2
import numpy as np

URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ROOT = "DAtaset/dataset"


def dir_mb(path):
    """Size of everything this script has written (the quota is not visible from
    inside the container, so we enforce our own byte budget instead)."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**20


def download(fid, dst, tries=5):
    for t in range(tries):
        try:
            urllib.request.urlretrieve(URL.format(fid), dst)
            if os.path.getsize(dst) > 1000:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"   retry {t + 1}: {e}", flush=True)
        time.sleep(2 ** t)
    return False


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return (m > (127 if m.max() > 1 else 0)).astype(np.uint8)


def tile(img, mask, stem, out_split, per_image, n_random, crop, min_frac, rng):
    """(per_image - n_random) crops with roof >= min_frac, plus n_random crops drawn from
    ALL grid positions regardless of roof, so empty ground is represented (avoids the
    zero-negatives bug). Returns (n_written, n_empty)."""
    h, w = mask.shape
    grid = [(r, c) for r in range(0, h - crop + 1, crop) for c in range(0, w - crop + 1, crop)]
    roofy = [p for p in grid if mask[p[0]:p[0] + crop, p[1]:p[1] + crop].mean() >= min_frac]
    rng.shuffle(roofy)
    picked = roofy[:max(per_image - n_random, 0)]
    rest = [p for p in grid if p not in set(picked)]
    rng.shuffle(rest)
    picked += rest[:n_random]
    n_empty = 0
    for r, c in picked:
        m = mask[r:r + crop, c:c + crop]
        n_empty += int(m.max() == 0)
        name = f"{stem}_{r:06d}_{c:06d}.png"
        cv2.imwrite(os.path.join(out_split, "images", name), img[r:r + crop, c:c + crop])
        cv2.imwrite(os.path.join(out_split, "masks", name), m * 255)
    return len(picked), n_empty


def main(a):
    listing = json.load(open(a.listing))
    by_path = {e["path"]: e["id"] for e in listing}
    rng = random.Random(a.seed)
    jobs = []
    for split, sub in (("train", "train"), ("val", "val")):
        imgs = sorted(p for p in by_path if p.startswith(f"{ROOT}/{sub}/image/") and p.endswith(".tif"))
        if a.limit:
            imgs = imgs[:a.limit]
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            lab_drive = f"{ROOT}/{sub}/label/{stem}.tif"
            jobs.append((split, stem, by_path[p], by_path.get(lab_drive)))
    os.makedirs(a.out, exist_ok=True)
    done_path = os.path.join(a.out, "done.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    jobs = [j for j in jobs if f"{j[0]}/{j[1]}" not in done]
    print(f"{len(jobs)} images to process ({len(done)} already done)", flush=True)
    for split in ("train", "val"):
        for s in ("images", "masks"):
            os.makedirs(os.path.join(a.out, split, s), exist_ok=True)
    raw = os.path.join(a.raw_dir)
    total = {"train": 0, "val": 0}
    empty = {"train": 0, "val": 0}
    for b in range(0, len(jobs), a.batch):
        used = dir_mb(a.out)
        if used + a.batch * 25 > a.budget_mb:  # 25 MB = worst-case raw image seen on Drive
            print(f"STOP: crops {used:.0f} MB + next raw batch would exceed budget {a.budget_mb} MB",
                  flush=True)
            break
        batch = jobs[b:b + a.batch]
        print(f"batch {b // a.batch + 1}: {len(batch)} images, crops so far {used:.0f} MB", flush=True)
        os.makedirs(raw, exist_ok=True)
        for split, stem, img_id, lab_id in batch:
            ip = os.path.join(raw, stem + ".tif")
            if not download(img_id, ip):
                print(f"   FAIL download image {stem}", flush=True)
                continue
            if split == "train":
                lp = os.path.join(a.train_labels, stem + ".tif")
            else:
                lp = os.path.join(raw, stem + "_label.tif")
                if not lab_id or not download(lab_id, lp):
                    print(f"   FAIL download label {stem}", flush=True)
                    os.remove(ip)
                    continue
            img = cv2.imread(ip, cv2.IMREAD_COLOR)
            mask = read_mask(lp)
            if img is None or mask is None or img.shape[:2] != mask.shape:
                print(f"   SKIP {stem}: unreadable or shape mismatch "
                      f"{None if img is None else img.shape} vs {None if mask is None else mask.shape}",
                      flush=True)
            else:
                n, ne = tile(img, mask, stem, os.path.join(a.out, split), a.per_image, a.n_random,
                             a.crop, a.min_frac, rng)
                total[split] += n
                empty[split] += ne
                with open(done_path, "a") as fh:
                    fh.write(f"{split}/{stem}\n")
            for p in (ip, os.path.join(raw, stem + "_label.tif")):
                if os.path.exists(p):
                    os.remove(p)
        print(f"   crops so far this run: {total}, of which empty (no roof): {empty}", flush=True)
    shutil.rmtree(raw, ignore_errors=True)
    n_tr = len(os.listdir(os.path.join(a.out, "train", "images")))
    n_va = len(os.listdir(os.path.join(a.out, "val", "images")))
    print(f"DONE. {a.out}: train {n_tr} crops, val {n_va} crops; this run empty share "
          f"train {empty['train'] / max(total['train'], 1):.3f}, val {empty['val'] / max(total['val'], 1):.3f}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--listing", default="docs/drive_folder_listing.json")
    p.add_argument("--train_labels", default="data/airs_full/DAtaset/dataset/train/label")
    p.add_argument("--out", default="data/airs_crops_v2")
    p.add_argument("--raw_dir", default=".tmp/airs_raw")
    p.add_argument("--per_image", type=int, default=5)
    p.add_argument("--n_random", type=int, default=1,
                   help="of per_image, how many are drawn from ALL positions regardless of roof")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--min_frac", type=float, default=0.005)
    p.add_argument("--budget_mb", type=int, default=2500,
                   help="hard cap on crops + one raw batch, in MB (set from host quota before launch)")
    p.add_argument("--limit", type=int, default=0, help="images per split (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
