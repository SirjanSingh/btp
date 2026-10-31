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
import segmentation_models_pytorch as smp
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


@torch.no_grad()
def adabn(model, loader, device, n_batches):
    """AdaBN: recompute BatchNorm running statistics on unlabelled target tiles.
    No gradients, no labels. Cumulative average over n_batches (momentum=None)."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    for i, (weak, _) in enumerate(loader):
        if i >= n_batches:
            break
        model(weak.to(device))
    for m in bns:
        m.momentum = 0.1
    model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# BLDA — Balanced Learning for Domain Adaptation (Li et al., ICML 2025),
# online logits adjustment (Sec. 3.3.4 / Algorithm 1) + cumulative density
# estimation loss (Sec. 3.3.5), written for this repo's 1-logit binary model.
#
# Binary reduction: a sigmoid logit z is the 2-class softmax with logits
# (bg, roof) = (-z/2, +z/2). Let v = z/2. Then
#   P_11 = dist(v | y=1)  (roof positive)     P_01 = dist(v | y=0)  (roof negative)
#   P_00 = mirror of P_01 (bg positive)        P_10 = mirror of P_11 (bg negative)
#   anchor P_p = pooled positives {v|y=1} U {-v|y=0}, anchor P_n = mirror of P_p.
# Working Algorithm 1 through, the bg offset is minus the roof offset, so the
# adjusted softmax is exactly a binary logit  z_adj = z - 2*tau*delta(v, y)  with
#   delta(v, 1) = Fp^-1(F_11(v)) - v
#   delta(v, 0) = Fn^-1(F_01(v)) - v = -Fp^-1(1 - F_01(v)) - v
# and the CDE regression target d = F_11(v) for y=1, 1 - F_01(v) for y=0.
# Paper settings used as defaults: K=5, momentum 0.99, T=3 EM loops, tau 0.1,
# lambda 0.2. N_min (minimum samples to update a GMM) is not given numerically
# in the paper; 1000 is our choice.
# ─────────────────────────────────────────────────────────────────────────────

class GMM1D:
    """1-D Gaussian mixture, momentum-based EM (Algorithm 1, lines 11-16)."""

    def __init__(self, k=5, momentum=0.99, em_loops=3, n_min=1000, max_samples=20000):
        self.k, self.m, self.t, self.n_min, self.max_s = k, momentum, em_loops, n_min, max_samples
        self.mu = self.sig = self.pi = None
        self.stale = 0  # iterations since last update (n in the paper)

    @property
    def ready(self):
        return self.mu is not None

    @torch.no_grad()
    def update(self, x):
        self.stale += 1
        x = x.detach().float().flatten()
        if x.numel() < self.n_min:
            return
        if x.numel() > self.max_s:
            x = x[torch.randint(0, x.numel(), (self.max_s,), device=x.device)]
        if self.mu is None:
            q = (torch.arange(self.k, device=x.device, dtype=torch.float32) + 0.5) / self.k
            mu = torch.quantile(x, q)
            sig = x.std().clamp(min=1e-2).repeat(self.k)
            pi = torch.full((self.k,), 1.0 / self.k, device=x.device)
        else:
            mu, sig, pi = self.mu.clone(), self.sig.clone(), self.pi.clone()
        for _ in range(self.t):
            logp = -0.5 * ((x[:, None] - mu) / sig) ** 2 - torch.log(sig) + torch.log(pi)
            r = torch.softmax(logp, dim=1)
            nk = r.sum(0) + 1e-6
            pi = nk / nk.sum()
            mu = (r * x[:, None]).sum(0) / nk
            sig = torch.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk).clamp(min=1e-3)
        if self.mu is None:
            self.mu, self.sig, self.pi = mu, sig, pi
        else:
            w = self.m ** self.stale
            self.mu = (1 - w) * mu + w * self.mu
            self.sig = (1 - w) * sig + w * self.sig
            self.pi = (1 - w) * pi + w * self.pi
        self.stale = 0

    def cdf(self, z):
        return (self.pi * torch.special.ndtr((z[..., None] - self.mu) / self.sig)).sum(-1)

    def icdf(self, u, n_grid=2048):
        lo = (self.mu - 6 * self.sig).min()
        hi = (self.mu + 6 * self.sig).max()
        g = torch.linspace(lo.item(), hi.item(), n_grid, device=u.device)
        c = self.cdf(g)
        c = torch.cummax(c, 0).values  # guard monotonicity
        u = u.clamp(c[0], c[-1])
        idx = torch.searchsorted(c, u.contiguous()).clamp(1, n_grid - 1)
        c0, c1, g0, g1 = c[idx - 1], c[idx], g[idx - 1], g[idx]
        return g0 + (u - c0) / (c1 - c0).clamp(min=1e-12) * (g1 - g0)


class BLDABinary:
    def __init__(self, a, device):
        mk = lambda: GMM1D(a.blda_k, a.blda_momentum, a.blda_em, a.blda_nmin)
        self.src = {1: mk(), 0: mk()}
        self.tgt = {1: mk(), 0: mk()}
        self.anchor = mk()  # P_p; P_n is its mirror image
        self.tau = a.blda_tau

    @torch.no_grad()
    def update(self, v_src, y_src, v_tgt, y_tgt, valid_tgt):
        pos, neg = y_src > 0.5, y_src <= 0.5
        self.src[1].update(v_src[pos]); self.src[0].update(v_src[neg])
        self.anchor.update(torch.cat([v_src[pos], -v_src[neg]]))
        tp, tn = valid_tgt & (y_tgt > 0.5), valid_tgt & (y_tgt <= 0.5)
        self.tgt[1].update(v_tgt[tp]); self.tgt[0].update(v_tgt[tn])

    def ready(self, dom):
        g = self.src if dom == "src" else self.tgt
        return g[1].ready and g[0].ready and self.anchor.ready

    @torch.no_grad()
    def offset_and_cde(self, v, y, dom):
        g = self.src if dom == "src" else self.tgt
        pos = y > 0.5
        f1 = g[1].cdf(v).clamp(1e-6, 1 - 1e-6)
        f0 = g[0].cdf(v).clamp(1e-6, 1 - 1e-6)
        d_pos = self.anchor.icdf(f1) - v
        d_neg = -self.anchor.icdf(1 - f0) - v
        delta = torch.where(pos, d_pos, d_neg)
        cde = torch.where(pos, f1, 1 - f0)
        return delta, cde


def forward_with_features(model, x):
    """smp Unet forward that also returns the decoder output (for the CDE head)."""
    feats = model.encoder(x)
    dec = model.decoder(*feats)
    return model.segmentation_head(dec), dec


_DICE = smp.losses.DiceLoss(mode="binary", from_logits=True)


@torch.no_grad()
def mix_batch(sx, sy, ts, t_logits, hi, lo, mode="dacs"):
    """DACS-style cross-domain mixing (Tranheden et al., WACV 2021), binary form.

    Per pair, a class-mask M is taken from the SOURCE label: in "dacs" mode one of the
    two classes (roof or background) is chosen at random, as DACS selects half of the
    classes present; in "roof" mode M is always the source roof pixels. Source pixels
    under M are pasted onto the target image together with their TRUE labels; the rest
    keep the teacher's pseudo-labels (confident pixels only, ignore band in between).
    Returns mixed image, mixed label, valid mask, and the pasted fraction."""
    B = sx.shape[0]
    if mode == "roof":
        m = (sy > 0.5).float()
    else:
        pick_roof = (torch.rand(B, 1, 1, 1, device=sx.device) < 0.5).float()
        roof = (sy > 0.5).float()
        m = pick_roof * roof + (1 - pick_roof) * (1 - roof)
    p_t = torch.sigmoid(t_logits.float())
    y_t = (p_t >= hi).float()
    valid_t = ((p_t >= hi) | (p_t <= lo)).float()
    xm = m * sx + (1 - m) * ts
    ym = m * sy + (1 - m) * y_t
    valid = torch.clamp(m + (1 - m) * valid_t, 0, 1)
    return xm, ym, valid, m.mean().item(), ((1 - m) * valid_t).sum().item(), ((1 - m) * y_t * valid_t).sum().item()


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
        val_ds = FolderDataset(a.val_target_dir, val_aug(), a.max_val)
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

    groups = [
        {"params": student.encoder.parameters(), "lr": a.lr * 0.1},
        {"params": student.decoder.parameters(), "lr": a.lr},
        {"params": student.segmentation_head.parameters(), "lr": a.lr},
    ]
    if a.mix and a.blda:
        raise SystemExit("--mix with --blda is not implemented yet; run them separately")
    blda, cde_head = None, None
    if a.blda:
        blda = BLDABinary(a, device)
        dec_ch = student.segmentation_head[0].in_channels
        cde_head = nn.Conv2d(dec_ch, 1, kernel_size=1).to(device)  # CDE regression head (Sec. 3.3.5)
        groups.append({"params": cde_head.parameters(), "lr": a.lr})
        print(f"BLDA on: K={a.blda_k} momentum={a.blda_momentum} T={a.blda_em} "
              f"tau={a.blda_tau} lambda={a.blda_lambda} start={a.blda_start}")
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
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
    snapshot(0, {"tag": "source_only"})
    if a.adabn_batches > 0:
        print(f"\n-- AdaBN on {a.adabn_batches} unlabelled target batches (no labels) --")
        adabn(student, tgt_loader, device, a.adabn_batches)
        teacher.load_state_dict(student.state_dict())
        snapshot(0, {"tag": "adabn"})

    src_it, tgt_it = infinite(src_loader), infinite(tgt_loader)
    fresh_stats = lambda: {"kept": 0.0, "pos": 0.0, "lt": 0.0, "ls": 0.0, "n": 0,
                           "delta_roof": 0.0, "cde": 0.0, "blda_on": 0.0, "mix_paste": 0.0}
    best, run_stats = -1.0, fresh_stats()
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
            if blda is None and a.mix:
                l_src = combined_loss(student(sx), sy)
                xm, ym, vm, pasted, n_tv, n_tpos = mix_batch(sx, sy, ts, t_logits,
                                                              a.pl_hi, a.pl_lo, a.mix_mode)
                bce_m = F.binary_cross_entropy_with_logits(student(xm).float(), ym, reduction="none")
                l_tgt = (bce_m * vm).sum() / vm.sum().clamp(min=1.0)
                loss = l_src + lam * l_tgt
                # stats over the TARGET-origin pixels only (comparable with unmixed runs)
                n_t = (1.0 - pasted) * ym.numel()
                kept = n_tv / max(n_t, 1.0)
                pos = n_tpos / max(n_tv, 1.0)
                run_stats["mix_paste"] += pasted
            elif blda is None:
                l_src = combined_loss(student(sx), sy)
                l_tgt, kept, pos = pseudo_label_loss(student(ts), t_logits, a.pl_hi, a.pl_lo)
                loss = l_src + lam * l_tgt
            else:
                s_logits, s_dec = forward_with_features(student, sx)
                u_logits, u_dec = forward_with_features(student, ts)

        if blda is not None:
            # BLDA losses in fp32 (GMM/CDF maths is not fp16-safe)
            s_logits, u_logits = s_logits.float(), u_logits.float()
            p_t = torch.sigmoid(t_logits.float())
            y_t = (p_t >= a.pl_hi).float()
            valid = (p_t >= a.pl_hi) | (p_t <= a.pl_lo)
            v_s, v_u = s_logits / 2, u_logits / 2
            blda.update(v_s, sy, v_u, y_t, valid)
            vf = valid.float(); nv = vf.sum().clamp(min=1.0)
            kept = vf.mean().item()
            pos = ((y_t * vf).sum() / nv).item()
            on = it >= a.blda_start and blda.ready("src") and blda.ready("tgt")
            if on:
                d_s, c_s = blda.offset_and_cde(v_s.detach(), sy, "src")
                d_u, c_u = blda.offset_and_cde(v_u.detach(), y_t, "tgt")
                z_s = s_logits - 2 * blda.tau * d_s
                z_u = u_logits - 2 * blda.tau * d_u
            else:
                z_s, z_u = s_logits, u_logits
            l_src = 0.5 * F.binary_cross_entropy_with_logits(z_s, sy) \
                + 0.5 * _DICE(s_logits, sy)
            bce_u = F.binary_cross_entropy_with_logits(z_u, y_t, reduction="none")
            l_tgt = (bce_u * vf).sum() / nv
            loss = l_src + lam * l_tgt
            if on:
                r_s = torch.sigmoid(cde_head(s_dec.float()))
                r_u = torch.sigmoid(cde_head(u_dec.float()))
                l_cde = ((r_s - c_s) ** 2).mean() + (((r_u - c_u) ** 2) * vf).sum() / nv
                loss = loss + a.blda_lambda * l_cde
                run_stats["delta_roof"] += (d_u * y_t * vf).sum().item() / max((y_t * vf).sum().item(), 1.0)
                run_stats["cde"] += l_cde.item()
            run_stats["blda_on"] += float(on)

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
            if a.mix:
                extra["mix_pasted_frac"] = round(run_stats["mix_paste"] / n, 4)
            if blda is not None:
                on = max(run_stats["blda_on"], 1.0)
                extra.update({"blda_active_frac": round(run_stats["blda_on"] / n, 3),
                              "blda_mean_roof_offset": round(run_stats["delta_roof"] / on, 4),
                              "blda_avg_cde_loss": round(run_stats["cde"] / on, 4)})
            run_stats = fresh_stats()
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
    g.add_argument("--max_val", type=int, default=None, help="Cap target val crops (smoke tests only)")
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
    g = p.add_argument_group("BLDA (Li et al., ICML 2025) — defaults are the paper's")
    g.add_argument("--mix", action="store_true",
                   help="DACS-style mixing: paste source class regions (true labels) into target images")
    g.add_argument("--mix_mode", choices=["dacs", "roof"], default="dacs",
                   help="dacs = paste roof OR background (random per pair); roof = always paste roofs")
    g.add_argument("--blda", action="store_true", help="Enable BLDA logit adjustment + CDE loss")
    g.add_argument("--blda_k", type=int, default=5, help="Gaussian components per GMM")
    g.add_argument("--blda_momentum", type=float, default=0.99)
    g.add_argument("--blda_em", type=int, default=3, help="EM loops per iteration")
    g.add_argument("--blda_tau", type=float, default=0.1, help="Scaling/temperature on the offset")
    g.add_argument("--blda_lambda", type=float, default=0.2, help="CDE regression loss weight")
    g.add_argument("--blda_nmin", type=int, default=1000, help="Min samples to update a GMM (our choice)")
    g.add_argument("--blda_start", type=int, default=100, help="Iteration to start applying offsets")
    g = p.add_argument_group("Self-training (cont.)")
    g.add_argument("--adabn_batches", type=int, default=0,
                   help="If >0, recompute BN stats on this many target batches before self-training")
    g.add_argument("--eval_every", type=int, default=500)
    g = p.add_argument_group("Misc")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--ckpt_dir", default="./checkpoints")
    g.add_argument("--log_dir", default="./logs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
