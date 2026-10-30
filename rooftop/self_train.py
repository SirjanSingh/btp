"""
self_train.py — plain self-training (mean teacher) for unsupervised domain
adaptation of the rooftop UNet to an unlabelled target domain (e.g. Jaipur).

No target labels are used for training. Training uses:
  • labelled SOURCE crops (AIRS)            → normal BCE+Dice loss
  • unlabelled TARGET tiles (e.g. Jaipur)   → pseudo-labels from an EMA teacher

How it works
  1. Student = copy of the source-trained checkpoint. Teacher = EMA copy of student.
  2. Teacher sees a clean target tile and predicts a probability map.
  3. Confident pixels become pseudo-labels: p >= --pl_hi -> roof, p <= --pl_lo ->
     background, everything in between is ignored.
  4. Student sees the SAME tile with strong colour/noise/blur and is trained on
     those pseudo-labels, plus the normal source loss.
  5. Teacher = slow moving average of student (--ema).

Optional --val_target_dir (images/ + masks/, hand-labelled, NEVER trained on) is
used only to MEASURE. The first row of the log (iteration 0) is the source-only
baseline, so the gain from adaptation is in the same table.

Usage (run from rooftop/):
    python self_train.py \
        --ckpt          checkpoints/<run>/best.pth \
        --src_dir       dataset_crops/train \
        --target_dir    /scratch/jaipur_tiles \
        --val_target_dir /scratch/jaipur_labelled/val \
        --simulate_low_res --iters 4000
"""

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train import (MEAN, STD, FolderDataset, build_model, combined_loss,
                   train_aug, val_aug)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

class TargetDataset(Dataset):
    """Unlabelled target tiles. Returns (teacher_view, student_view) of the same
    crop: identical geometry, but the student view gets strong photometric aug."""

    def __init__(self, img_dir, crop=512, exclude_names=(), max_samples=None, seed=0):
        paths = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
        excl = set(exclude_names)
        paths = [p for p in paths if p.name not in excl]
        if max_samples and len(paths) > max_samples:
            paths = random.Random(seed).sample(paths, max_samples)
        if not paths:
            raise FileNotFoundError(f"No target tiles found in {img_dir}")
        self.paths = paths
        self.geo = A.Compose([
            A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
            A.RandomCrop(crop, crop),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])
        self.strong = A.Compose([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.3),
        ])
        self.norm = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        img = self.geo(image=img)["image"]
        weak = self.norm(image=img)["image"]
        strong = self.norm(image=self.strong(image=img)["image"])["image"]
        return weak, strong


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (several thresholds at once — never report a cross-domain number
# at a single fixed threshold only)
# ─────────────────────────────────────────────────────────────────────────────

class SweepMetrics:
    def __init__(self, thresholds):
        self.th = list(thresholds)
        self.reset()

    def reset(self):
        self.c = {t: [0.0, 0.0, 0.0] for t in self.th}  # tp, fp, fn

    @torch.no_grad()
    def update(self, logits, targets):
        p = torch.sigmoid(logits)
        for t in self.th:
            pr = (p > t).float()
            self.c[t][0] += (pr * targets).sum().item()
            self.c[t][1] += (pr * (1 - targets)).sum().item()
            self.c[t][2] += ((1 - pr) * targets).sum().item()

    def compute(self):
        out = {}
        for t, (tp, fp, fn) in self.c.items():
            iou = tp / (tp + fp + fn + 1e-7)
            prec = tp / (tp + fp + 1e-7)
            rec = tp / (tp + fn + 1e-7)
            out[t] = {"iou": iou, "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec + 1e-7)}
        return out


@torch.no_grad()
def evaluate(model, loader, device, thresholds):
    model.eval()
    m = SweepMetrics(thresholds)
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        m.update(model(imgs), masks)
    res = m.compute()
    best_t = max(res, key=lambda t: res[t]["iou"])
    return {"by_threshold": {str(t): {k: round(v, 4) for k, v in r.items()}
                             for t, r in res.items()},
            "best_threshold": best_t, "best_iou": round(res[best_t]["iou"], 4)}


@torch.no_grad()
def unlabelled_stats(model, loader, device, max_batches=20):
    """Without labels we can only watch behaviour: fraction predicted as roof."""
    model.eval()
    fg, n = 0.0, 0
    for i, (weak, _) in enumerate(loader):
        if i >= max_batches:
            break
        p = torch.sigmoid(model(weak.to(device)))
        fg += (p > 0.5).float().mean().item()
        n += 1
    return {"pred_roof_fraction": round(fg / max(n, 1), 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Self-training pieces
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ema_update(teacher, student, alpha):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(alpha).add_(ps.detach(), alpha=1 - alpha)
    for bt, bs in zip(teacher.buffers(), student.buffers()):
        bt.copy_(bs)


def pseudo_label_loss(student_logits, teacher_logits, hi, lo):
    """BCE on confident teacher pixels only. Returns (loss, kept_frac, pos_frac)."""
    p = torch.sigmoid(teacher_logits.detach())
    pos, neg = p >= hi, p <= lo
    valid = (pos | neg).float()
    target = pos.float()
    bce = F.binary_cross_entropy_with_logits(student_logits, target, reduction="none")
    n = valid.sum().clamp(min=1.0)
    loss = (bce * valid).sum() / n
    kept = valid.mean().item()
    pos_frac = (pos.float().sum() / n).item()
    return loss, kept, pos_frac


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"Device: {device}")

    # ── data ──
    src_ds = FolderDataset(a.src_dir, train_aug(a.crop, a.simulate_low_res), a.max_src)
    val_names = ()
    val_loader = None
    if a.val_target_dir:
        val_ds = FolderDataset(a.val_target_dir, val_aug())
        val_names = [p.name for p in val_ds.paths]
        val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.workers)
    tgt_ds = TargetDataset(a.target_dir, a.crop, exclude_names=val_names,
                           max_samples=a.max_target, seed=a.seed)
    src_loader = DataLoader(src_ds, batch_size=a.batch_size, shuffle=True,
                            num_workers=a.workers, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=a.batch_size, shuffle=True,
                            num_workers=a.workers, drop_last=True)
    src_val = None
    if a.src_val_dir:
        src_val = DataLoader(FolderDataset(a.src_val_dir, val_aug(), a.max_src_val),
                             batch_size=a.batch_size, shuffle=False, num_workers=a.workers)
    print(f"Source crops: {len(src_ds)} | Target tiles (unlabelled): {len(tgt_ds)} | "
          f"Target val (labelled, measure-only): {len(val_names)}")

    # ── models ──
    student = build_model(a.arch, a.encoder, encoder_weights=None).to(device)
    ck = torch.load(a.ckpt, map_location=device)
    student.load_state_dict(ck["model_state"])
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([
        {"params": student.encoder.parameters(), "lr": a.lr * 0.1},
        {"params": student.decoder.parameters(), "lr": a.lr},
        {"params": student.segmentation_head.parameters(), "lr": a.lr},
    ], weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── logging ──
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(a.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir) / f"self_train_{stamp}"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    record = {"run": f"self_train_{stamp}", "config": vars(a), "history": []}

    def snapshot(it, extra=None):
        row = {"iter": it}
        for name, net in (("student", student), ("teacher", teacher)):
            if val_loader is not None:
                row[f"{name}_target"] = evaluate(net, val_loader, device, thresholds)
            if src_val is not None:
                row[f"{name}_source"] = evaluate(net, src_val, device, thresholds)
        row["teacher_unlabelled"] = unlabelled_stats(teacher, tgt_loader, device)
        if extra:
            row.update(extra)
        record["history"].append(row)
        (log_dir / f"{record['run']}.json").write_text(json.dumps(record, indent=2))
        msg = [f"[iter {it}]"]
        for k in ("student_target", "teacher_target", "student_source", "teacher_source"):
            if k in row:
                msg.append(f"{k}: IoU@best={row[k]['best_iou']} (thr {row[k]['best_threshold']}), "
                           f"IoU@0.5={row[k]['by_threshold']['0.5']['iou']}")
        msg.append(f"roof_frac={row['teacher_unlabelled']['pred_roof_fraction']}")
        print("\n".join(msg))
        return row

    print("\n-- Iteration 0 = source-only baseline (no adaptation) --")
    snapshot(0)

    src_it, tgt_it = infinite(src_loader), infinite(tgt_loader)
    best, run_stats = -1.0, {"kept": 0.0, "pos": 0.0, "lt": 0.0, "ls": 0.0, "n": 0}
    t0 = time.time()
    pbar = tqdm(range(1, a.iters + 1), dynamic_ncols=True)
    for it in pbar:
        student.train()
        sx, sy = next(src_it)
        tw, ts = next(tgt_it)
        sx, sy, tw, ts = sx.to(device), sy.to(device), tw.to(device), ts.to(device)
        lam = a.lam * min(1.0, it / max(a.warmup, 1))  # ramp target loss in

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            with torch.no_grad():
                t_logits = teacher(tw)
            l_src = combined_loss(student(sx), sy)
            l_tgt, kept, pos = pseudo_label_loss(student(ts), t_logits, a.pl_hi, a.pl_lo)
            loss = l_src + lam * l_tgt

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        ema_update(teacher, student, a.ema)

        for k, v in (("kept", kept), ("pos", pos), ("lt", l_tgt.item()), ("ls", l_src.item())):
            run_stats[k] += v
        run_stats["n"] += 1
        pbar.set_postfix(src=f"{l_src.item():.3f}", tgt=f"{l_tgt.item():.3f}", kept=f"{kept:.2f}")

        if it % a.eval_every == 0 or it == a.iters:
            n = run_stats["n"]
            extra = {"avg_src_loss": round(run_stats["ls"] / n, 4),
                     "avg_tgt_loss": round(run_stats["lt"] / n, 4),
                     "pseudo_label_kept_frac": round(run_stats["kept"] / n, 4),
                     "pseudo_label_roof_frac": round(run_stats["pos"] / n, 4),
                     "elapsed_min": round((time.time() - t0) / 60, 1)}
            run_stats = {"kept": 0.0, "pos": 0.0, "lt": 0.0, "ls": 0.0, "n": 0}
            row = snapshot(it, extra)
            score = row["teacher_target"]["best_iou"] if "teacher_target" in row else it
            torch.save({"model_state": teacher.state_dict(), "iter": it}, ckpt_dir / "last_teacher.pth")
            if score > best:
                best = score
                torch.save({"model_state": teacher.state_dict(), "iter": it, "score": score},
                           ckpt_dir / "best_teacher.pth")
    print(f"\nDone. Log: {log_dir / (record['run'] + '.json')} | ckpts: {ckpt_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Plain self-training UDA for rooftop segmentation")
    g = p.add_argument_group("Data")
    g.add_argument("--ckpt", required=True, help="Source-trained checkpoint (train.py output)")
    g.add_argument("--src_dir", required=True, help="Labelled source crops (images/ masks/)")
    g.add_argument("--src_val_dir", default=None, help="Source val crops, to check forgetting")
    g.add_argument("--target_dir", required=True, help="Folder of UNLABELLED target tiles")
    g.add_argument("--val_target_dir", default=None,
                   help="Hand-labelled target crops (images/ masks/). MEASURE ONLY, excluded from training.")
    g.add_argument("--max_src", type=int, default=None)
    g.add_argument("--max_src_val", type=int, default=None)
    g.add_argument("--max_target", type=int, default=None)
    g.add_argument("--crop", type=int, default=512)
    g.add_argument("--simulate_low_res", action="store_true",
                   help="Downscale augmentation on source (resolution gap)")
    g = p.add_argument_group("Model")
    g.add_argument("--arch", default="unet")
    g.add_argument("--encoder", default="resnet34")
    g = p.add_argument_group("Self-training")
    g.add_argument("--iters", type=int, default=4000)
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--ema", type=float, default=0.999)
    g.add_argument("--pl_hi", type=float, default=0.8, help="p>=hi -> pseudo roof")
    g.add_argument("--pl_lo", type=float, default=0.2, help="p<=lo -> pseudo background")
    g.add_argument("--lam", type=float, default=1.0, help="Weight of target pseudo-label loss")
    g.add_argument("--warmup", type=int, default=500, help="Iterations to ramp lam from 0")
    g.add_argument("--eval_every", type=int, default=500)
    g = p.add_argument_group("Misc")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--ckpt_dir", default="./checkpoints")
    g.add_argument("--log_dir", default="./logs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
