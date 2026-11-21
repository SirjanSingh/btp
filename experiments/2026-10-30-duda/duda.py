"""
duda.py — DUDA (Distilled Unsupervised Domain Adaptation; Kang et al., WACV 2026,
arXiv 2504.09814) for binary rooftop segmentation, AIRS -> Jaipur, no target labels.

WHY. Plain mean-teacher self-training on the small UNet collapsed in arms A-C
(experiments/2026-10-30-annotation-free-selftrain): its own pseudo-labels are
too poor to bootstrap from. DUDA's claim is that a LARGE teacher makes better
pseudo-labels than a small one can make for itself, and that distilling them
into the small network beats adapting the small network alone. The small
network is the one we deploy, so it is also the one we score.

Three networks, as in the paper:
  LT  large teacher   UNet/<--large_encoder>  EMA of LS, makes the pseudo-labels
  LS  large student   same architecture       trained by gradients
  SS  small student   UNet/resnet34           the AIRS seed (+ AdaBN), deployed

Stage 1, pre-adaptation (--pre_iters):
  LS: L_src + L_tgt(hard pseudo-labels from LT)
  SS: L_src + L_tgt(same pseudo-labels) + KL(LT soft || SS)
  LT <- EMA(LS) every step
Stage 2, inconsistency: per class c, I_c = mean over images of 1 - IoU(LT, SS),
  counting an image only when the class covers >= 0.1% of its pixels; then
  I'_c = C * I_c / sum(I_c).
Stage 3, fine-tuning (--ft_iters): LT and LS frozen, pseudo-labels from LS,
  SS: L_src + sum_c (2 - I'_c) * L_tgt^(c) + sum_c I'_c * KL^(c)
  so the class the two networks disagree on most leans on soft distillation.

Binary adaptation, stated because the paper is multi-class:
  * C = 2 (background, roof). Pseudo-label = argmax, i.e. p > 0.5. The paper
    uses no confidence threshold, so neither does this.
  * L^(c) is read as the loss over pixels whose pseudo-label is c, normalised
    by all pixels, so the weights redistribute rather than rescale the loss.
  * KL is the two-class KL between sigmoid outputs, no temperature (none given).
Deviations forced by our data, also stated:
  * Only 125 labelled AIRS crops exist on the server, and LS starts from
    ImageNet, so LS first trains on source alone for --large_src_warmup
    iterations. Without it LT's first pseudo-labels are noise.
  * Iterations are scaled down from the paper's 40k + 80k to fit a shared node.
  * Source loss is the repo's combined_loss (BCE + Dice), matching arms A-D.

Reporting rule (fixed before the run, same as arms A-D): the primary number is
the LAST small-student checkpoint. Best-over-evals is logged only as an oracle.

Usage (from repo root, inside run_docker.sh):
    env PYTHONPATH=rooftop:experiments/2026-10-30-duda python \
        experiments/2026-10-30-duda/duda.py --ss_ckpt rooftop/checkpoints/unet_resnet34_best.pth \
        --src_dir data/st_run/airs_crops/train --target_dir data/st_run/target_tiles \
        --val_target_dir data/st_run/eval/val --adabn_batches 40
"""

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import FolderDataset, build_model, combined_loss, train_aug, val_aug
# Snapshot of the arm A-D helpers (copied at launch), so a later edit to that
# file cannot change what this run did.
from st_common import (TargetDataset, adabn, ema_update, evaluate, infinite,
                       unlabelled_stats)

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
EPS = 1e-6


def binary_kl(teacher_logits, student_logits):
    """Per-pixel KL(teacher || student) between two-class sigmoid outputs."""
    pt = torch.sigmoid(teacher_logits.detach()).clamp(EPS, 1 - EPS)
    ps = torch.sigmoid(student_logits).clamp(EPS, 1 - EPS)
    return pt * (pt.log() - ps.log()) + (1 - pt) * ((1 - pt).log() - (1 - ps).log())


def target_losses(student_logits, teacher_logits, weights=None):
    """Hard pseudo-label CE and soft KL on target, optionally class-weighted.

    weights: None (stage 1, plain sums) or a pair of (ce_w, kl_w) tensors of
    shape [2], indexed by each pixel's pseudo-label class (0 bg, 1 roof)."""
    y = (torch.sigmoid(teacher_logits.detach()) > 0.5).float()
    ce = F.binary_cross_entropy_with_logits(student_logits, y, reduction="none")
    kl = binary_kl(teacher_logits, student_logits)
    if weights is None:
        return ce.mean(), kl.mean(), y.mean().item()
    ce_w, kl_w = weights
    cls = y.long()
    return (ce * ce_w[cls]).mean(), (kl * kl_w[cls]).mean(), y.mean().item()


@torch.no_grad()
def inconsistency(lt, ss, loader, device, n_batches, min_frac=0.001):
    """I_c (paper Eqs. 6-9) over n_batches of target tiles, c in {bg, roof}."""
    lt.eval(); ss.eval()
    num, den = np.zeros(2), np.zeros(2)
    for i, (weak, _) in enumerate(loader):
        if i >= n_batches:
            break
        weak = weak.to(device)
        a = torch.sigmoid(lt(weak)) > 0.5
        b = torch.sigmoid(ss(weak)) > 0.5
        for c, (ma, mb) in enumerate(((~a, ~b), (a, b))):
            inter = (ma & mb).flatten(1).sum(1).float()
            union = (ma | mb).flatten(1).sum(1).float()
            present = union >= min_frac * ma[0].numel()
            num[c] += (1 - inter / union.clamp(min=1))[present].sum().item()
            den[c] += present.sum().item()
    I = num / np.maximum(den, 1)
    I_norm = 2 * I / max(I.sum(), EPS)
    return {"I": I.round(4).tolist(), "I_norm": I_norm.round(4).tolist(),
            "images_counted": den.tolist()}


def param_groups(model, lr, enc_mult):
    return [{"params": model.encoder.parameters(), "lr": lr * enc_mult},
            {"params": model.decoder.parameters(), "lr": lr},
            {"params": model.segmentation_head.parameters(), "lr": lr}]


def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    src_ds = FolderDataset(a.src_dir, train_aug(a.crop, a.simulate_low_res))
    val_ds = FolderDataset(a.val_target_dir, val_aug())
    tgt_ds = TargetDataset(a.target_dir, a.crop,
                           exclude_names=[p.name for p in val_ds.paths], seed=a.seed)
    kw = dict(batch_size=a.batch_size, num_workers=a.workers)
    src_loader = DataLoader(src_ds, shuffle=True, drop_last=True, **kw)
    tgt_loader = DataLoader(tgt_ds, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    print(f"Source crops: {len(src_ds)} | Target tiles: {len(tgt_ds)} | "
          f"Target eval (measure only): {len(val_ds)}")

    # ── networks ──
    ss = build_model("unet", "resnet34", encoder_weights=None).to(device)
    ss.load_state_dict(torch.load(a.ss_ckpt, map_location=device)["model_state"])
    if a.adabn_batches:
        adabn(ss, tgt_loader, device, a.adabn_batches)
    ls = build_model("unet", a.large_encoder, encoder_weights="imagenet").to(device)
    lt = copy.deepcopy(ls).eval()
    for p in lt.parameters():
        p.requires_grad_(False)

    opt_ls = torch.optim.AdamW(param_groups(ls, a.lr_large, 0.1), weight_decay=1e-2)
    opt_ss = torch.optim.AdamW(param_groups(ss, a.lr_small, 0.1), weight_decay=1e-4)
    amp = device.type == "cuda"
    sc_ls = torch.cuda.amp.GradScaler(enabled=amp)
    sc_ss = torch.cuda.amp.GradScaler(enabled=amp)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = f"duda_{stamp}"
    log_dir = Path(a.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(a.ckpt_dir) / run; ckpt_dir.mkdir(parents=True, exist_ok=True)
    record = {"run": run, "config": vars(a), "history": []}

    def snapshot(it, stage, extra=None, large_too=False):
        row = {"iter": it, "stage": stage,
               "small_target": evaluate(ss, val_loader, device, THRESHOLDS),
               "small_unlabelled": unlabelled_stats(ss, tgt_loader, device)}
        if large_too:
            row["large_teacher_target"] = evaluate(lt, val_loader, device, THRESHOLDS)
            row["large_teacher_unlabelled"] = unlabelled_stats(lt, tgt_loader, device)
        if extra:
            row.update(extra)
        record["history"].append(row)
        (log_dir / f"{run}.json").write_text(json.dumps(record, indent=2))
        s = row["small_target"]
        msg = (f"[{stage} iter {it}] small IoU@0.5={s['by_threshold']['0.5']['iou']} "
               f"best={s['best_iou']} (thr {s['best_threshold']}) "
               f"roof_frac={row['small_unlabelled']['pred_roof_fraction']}")
        if large_too:
            t = row["large_teacher_target"]
            msg += (f" | LT IoU@0.5={t['by_threshold']['0.5']['iou']} "
                    f"roof_frac={row['large_teacher_unlabelled']['pred_roof_fraction']}")
        print(msg, flush=True)
        return row

    print("\n-- iter 0: small student = AIRS seed (+AdaBN), before any DUDA step --")
    snapshot(0, "start")

    src_it, tgt_it = infinite(src_loader), infinite(tgt_loader)
    t0 = time.time()

    def step_ss(sx, sy, ts, teacher_logits, weights):
        ss.train()
        with torch.cuda.amp.autocast(enabled=amp):
            l_src = combined_loss(ss(sx), sy)
            l_ce, l_kl, pos = target_losses(ss(ts), teacher_logits, weights)
            loss = l_src + l_ce + l_kl
        opt_ss.zero_grad()
        sc_ss.scale(loss).backward(); sc_ss.step(opt_ss); sc_ss.update()
        return l_src.item(), l_ce.item(), l_kl.item(), pos

    # ── stage 1: pre-adaptation ──
    stats = []
    for it in tqdm(range(1, a.pre_iters + 1), dynamic_ncols=True, mininterval=30):
        sx, sy = next(src_it); tw, ts = next(tgt_it)
        sx, sy, tw, ts = sx.to(device), sy.to(device), tw.to(device), ts.to(device)
        warm = it <= a.large_src_warmup

        ls.train()
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():
                lt_logits = lt(tw)
            l_ls = combined_loss(ls(sx), sy)
            if not warm:
                l_ls = l_ls + target_losses(ls(ts), lt_logits)[0]
        opt_ls.zero_grad()
        sc_ls.scale(l_ls).backward(); sc_ls.step(opt_ls); sc_ls.update()
        # A fast EMA during source warm-up lets LT catch up with an LS that
        # started from ImageNet; the paper-standard rate applies afterwards.
        ema_update(lt, ls, a.ema_warm if warm else a.ema)

        if not warm:  # SS only distils once LT has seen source labels
            stats.append(step_ss(sx, sy, ts, lt_logits, None))

        if it % a.eval_every == 0 or it == a.pre_iters:
            extra = {"elapsed_min": round((time.time() - t0) / 60, 1),
                     "large_src_warmup": warm}
            if stats:
                m = np.mean(stats, axis=0)
                extra.update(avg_src=round(m[0], 4), avg_tgt_ce=round(m[1], 4),
                             avg_tgt_kl=round(m[2], 4), pseudo_roof_frac=round(m[3], 4))
            stats = []
            snapshot(it, "pre", extra, large_too=True)

    # ── stage 2: inconsistency ──
    inc = inconsistency(lt, ss, tgt_loader, device, a.inc_batches)
    record["inconsistency"] = inc
    print(f"\n[inconsistency] I={inc['I']}  I_norm={inc['I_norm']}  "
          f"(order: background, roof)", flush=True)
    i_norm = torch.tensor(inc["I_norm"], device=device, dtype=torch.float32)
    weights = (2 - i_norm, i_norm)

    # ── stage 3: fine-tune SS against frozen LS ──
    lt.eval(); ls.eval()
    for p in ls.parameters():
        p.requires_grad_(False)
    stats = []
    for it in tqdm(range(1, a.ft_iters + 1), dynamic_ncols=True, mininterval=30):
        sx, sy = next(src_it); tw, ts = next(tgt_it)
        sx, sy, tw, ts = sx.to(device), sy.to(device), tw.to(device), ts.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
            ls_logits = ls(tw)
        stats.append(step_ss(sx, sy, ts, ls_logits, weights))
        if it % a.eval_every == 0 or it == a.ft_iters:
            m = np.mean(stats, axis=0); stats = []
            snapshot(a.pre_iters + it, "ft",
                     {"elapsed_min": round((time.time() - t0) / 60, 1),
                      "avg_src": round(m[0], 4), "avg_tgt_ce": round(m[1], 4),
                      "avg_tgt_kl": round(m[2], 4), "pseudo_roof_frac": round(m[3], 4)})

    # Only the deployable network is kept by default: the quota is nearly full.
    torch.save({"model_state": ss.state_dict(), "iter": a.pre_iters + a.ft_iters},
               ckpt_dir / "last_small_student.pth")
    if a.save_large:
        torch.save({"model_state": lt.state_dict()}, ckpt_dir / "large_teacher.pth")
    hist = record["history"]
    record["primary_last_small"] = hist[-1]["small_target"]
    record["oracle_best_small_iou@0.5"] = max(
        h["small_target"]["by_threshold"]["0.5"]["iou"] for h in hist)
    (log_dir / f"{run}.json").write_text(json.dumps(record, indent=2))
    print(f"\nDone. primary (last SS) IoU@0.5="
          f"{hist[-1]['small_target']['by_threshold']['0.5']['iou']} | "
          f"log {log_dir / (run + '.json')}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="DUDA for binary rooftop UDA")
    g = p.add_argument_group("Data")
    g.add_argument("--ss_ckpt", required=True, help="AIRS-trained resnet34 (small student)")
    g.add_argument("--src_dir", required=True)
    g.add_argument("--target_dir", required=True)
    g.add_argument("--val_target_dir", required=True, help="measure only, never trained on")
    g.add_argument("--crop", type=int, default=512)
    g.add_argument("--simulate_low_res", action="store_true")
    g = p.add_argument_group("DUDA")
    g.add_argument("--large_encoder", default="mit_b5")
    g.add_argument("--adabn_batches", type=int, default=40)
    g.add_argument("--pre_iters", type=int, default=3000)
    g.add_argument("--large_src_warmup", type=int, default=1000)
    g.add_argument("--ft_iters", type=int, default=1500)
    g.add_argument("--inc_batches", type=int, default=40)
    g.add_argument("--ema", type=float, default=0.999)
    g.add_argument("--ema_warm", type=float, default=0.99)
    g.add_argument("--lr_large", type=float, default=6e-4,
                   help="decoder/head lr for LS; encoder gets 0.1x (DAFormer: 6e-5)")
    g.add_argument("--lr_small", type=float, default=5e-5)
    g.add_argument("--batch_size", type=int, default=4)
    g.add_argument("--eval_every", type=int, default=500)
    g = p.add_argument_group("Misc")
    g.add_argument("--workers", type=int, default=3)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--save_large", action="store_true")
    g.add_argument("--ckpt_dir", default="experiments/2026-10-30-duda/ckpt")
    g.add_argument("--log_dir", default="experiments/2026-10-30-duda/logs")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
