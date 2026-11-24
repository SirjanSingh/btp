#!/usr/bin/env python
"""
soft_selftrain_student.py - the UDA step. Train a U-Net/ResNet34 student on UNLABELLED Jaipur tiles
using a teacher's SOFT roof probabilities (8-bit PNGs from crossearth_teacher_maps.py) as targets.
No Jaipur label of any kind is used for training. Optionally start from the AIRS checkpoint so the
AIRS training is the student's initialisation (source knowledge -> adapted to target).

Loss: BCE-with-soft-target + soft-Dice, i.e. the target is the probability itself, not 0/1.
  --sharpen T   optional temperature on the target (p^(1/T) renormalised); 1.0 = untouched.

Scoring (same protocol as score_crossearth_hand.py): threshold on calib half, IoU on test half,
crop bootstrap CI. The hand crops are only read for scoring after each epoch; the checkpoint used
for the reported number is the LAST epoch (fixed in advance), never picked by test IoU.

Run inside the btp docker image (smp + albumentations live there):
  ./run_docker.sh 1 "python scripts/soft_selftrain_student.py --teacher data/../teacher_a2s ..."
"""
import argparse
import csv
import json
import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_crossearth_hand import FINE_SWEEP, agg, counts, dihedral, load_bin  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def norm(img_rgb):
    return ((img_rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


class SoftSet(Dataset):
    def __init__(self, img_dirs, teacher_dir, sharpen):
        self.items = []
        for d in img_dirs:
            for n in sorted(os.listdir(d)):
                tps = [os.path.join(t, n) for t in teacher_dir]
                if all(os.path.isfile(t) for t in tps):
                    self.items.append((os.path.join(d, n), tps))
        self.sharpen = sharpen
        self.aug = A.Compose([
            A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(8, 15, 8, p=0.3),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ip, tp = self.items[i]
        img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        # several teacher dirs -> mean of their soft maps (teacher ensemble)
        t = np.mean([cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 for p in tp], 0)
        o = self.aug(image=img, mask=t)
        t = o["mask"]
        if self.sharpen != 1.0:
            a = t ** (1 / self.sharpen)
            b = (1 - t) ** (1 / self.sharpen)
            t = a / np.maximum(a + b, 1e-6)
        return torch.from_numpy(norm(o["image"])), torch.from_numpy(t)[None]


def soft_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return 0.5 * bce + 0.5 * dice.mean()


@torch.no_grad()
def hand_probs(model, imgs, dev, tta):
    model.eval()

    def fn(img_bgr):
        x = torch.from_numpy(norm(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
    return [(dihedral(fn, im) if tta else fn(im)).astype(np.float16) for im in imgs]


def score(prob, hand, cal, tst):
    cs = {t: counts(prob, hand, t) for t in FINE_SWEEP}
    thr = max(FINE_SWEEP, key=lambda t: agg(cs[t][cal])[0])
    ci, _ = agg(cs[thr][cal])
    ti, ta = agg(cs[thr][tst])
    c = cs[thr][tst]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(c), size=(2000, len(c)))
    b = c[:, 0][idx].sum(1) / np.maximum((c[:, 0] + c[:, 1] + c[:, 2])[idx].sum(1), 1)
    return dict(thr=thr, calib_iou=ci, test_iou=ti, test_area_ratio=ta,
                test_iou_boot95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    ds = SoftSet(a.img_dirs, a.teacher, a.sharpen)
    print(f"{len(ds)} unlabelled tiles with teacher maps", flush=True)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)

    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")["model_state"]
        model.load_state_dict(sd)
        print("init from", a.init, flush=True)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    rows = list(csv.DictReader(open(a.index)))
    hand = [load_bin(r["mask"]) for r in rows]
    himgs = [cv2.imread(r["img"], cv2.IMREAD_COLOR) for r in rows]
    cal = [i for i, r in enumerate(rows) if r["half"] == "calib"]
    tst = [i for i, r in enumerate(rows) if r["half"] == "test"]

    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, t in dl:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = soft_loss(model(x).float(), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        r = score(hand_probs(model, himgs, dev, tta=False), hand, cal, tst)
        r.update(epoch=ep, train_loss=tot / len(dl))
        hist.append(r)
        print(f"ep {ep} loss {r['train_loss']:.4f} calib {r['calib_iou']:.3f} test {r['test_iou']:.3f} "
              f"area {r['test_area_ratio']:.2f} thr {r['thr']:.2f}", flush=True)
        torch.save({"model_state": model.state_dict(), "epoch": ep, "args": vars(a)}, a.ckpt)

    final = {}
    for name, tta in (("last_plain", False), ("last_tta8", True)):
        p = hand_probs(model, himgs, dev, tta)
        final[name] = score(p, hand, cal, tst)
        print(name, json.dumps(final[name]), flush=True)
        np.savez_compressed(os.path.splitext(a.out)[0] + f"_{name}_probs.npz", p=np.stack(p))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(args=vars(a), history=hist, final=final), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dirs", nargs="+", default=["data/jaipur_weak/train/images", "data/jaipur_weak/val/images"])
    p.add_argument("--teacher", nargs="+", required=True,
                   help="dir(s) of 8-bit soft-probability PNGs (same names as tiles); several = averaged")
    p.add_argument("--init", default="rooftop/checkpoints/unet_resnet34_best.pth")
    p.add_argument("--index", default="data/hand_r12b2/index.csv")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--sharpen", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
