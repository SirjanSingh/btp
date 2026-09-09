#!/usr/bin/env python
"""
compare_methods.py — run several domain-adaptation methods on ONE eval set.

The point is a controlled comparison: same 1,701 held-out Jaipur crops, same
model family, same metric, so the *ranking* is meaningful even though the
absolute numbers are not clean (see CAVEATS).

Methods, cheapest first (plan/03 tiers):
  seed        Tier 0  the AIRS checkpoint applied to Jaipur unchanged
  seed_tta    Tier 1  + horizontal/vertical flip test-time augmentation
  adabn       Tier 1  recompute BatchNorm running stats on target images.
                      No gradients, no labels -- the cheapest real adaptation
                      there is. Classic result: sometimes several IoU points.
  histmatch   Tier 1  match each target crop's histogram to an AIRS reference
                      before inference (photometric alignment)
  fda         Tier 1  Fourier Domain Adaptation -- swap the low-frequency
                      amplitude spectrum of the target for the source's,
                      keeping target phase. Style transfer without a GAN.
  weak        Tier 4  the checkpoint fine-tuned on Open Buildings weak labels

CAVEATS, and they matter:
  1. Eval labels are Open Buildings GROUND FOOTPRINTS, not roof outlines. Every
     number here is agreement-with-footprints, not roof accuracy.
  2. `weak` trained on exactly this label distribution, so it has home-field
     advantage over the zero-training methods. Comparing the Tier-1 methods
     against EACH OTHER is clean; comparing them to `weak` is not.
  3. The oracle column tunes the threshold on the eval set. It is an upper
     bound, not a fair score, and is labelled as such.

Usage:
    python scripts/compare_methods.py --out diagnostics/method_comparison.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

# Sweep starts low on purpose. D6 measured the seed predicting 5.7% foreground
# where truth is 28.2%, so its operating point sits far below the usual 0.5 --
# a sweep that bottoms out at 0.30 has every method pinned to the boundary and
# reports an optimum it never found.
THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40,
              0.50, 0.60, 0.70]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class Crops(Dataset):
    """Val crops, optionally passed through a photometric transform first."""

    def __init__(self, root, transform=None, limit=0):
        self.img_dir = os.path.join(root, "images")
        self.msk_dir = os.path.join(root, "masks")
        self.names = sorted(os.listdir(self.img_dir))
        if limit:
            self.names = self.names[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, n)),
                           cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.msk_dir, n), cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img = self.transform(img)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy((msk > 127).astype(np.float32)))


# ── photometric transforms ──────────────────────────────────────────────────
def make_histmatch(ref):
    """Per-channel CDF matching of a target crop onto an AIRS reference."""
    ref_cdfs = []
    for c in range(3):
        h = np.bincount(ref[:, :, c].ravel(), minlength=256).astype(np.float64)
        ref_cdfs.append(np.cumsum(h) / h.sum())

    def f(img):
        out = np.empty_like(img)
        for c in range(3):
            h = np.bincount(img[:, :, c].ravel(), minlength=256).astype(np.float64)
            cdf = np.cumsum(h) / h.sum()
            lut = np.interp(cdf, ref_cdfs[c], np.arange(256)).astype(np.uint8)
            out[:, :, c] = lut[img[:, :, c]]
        return out

    return f


def make_fda(ref, beta=0.01):
    """FDA: give the target the source's low-frequency amplitude, keep phase.

    beta sets the radius of the swapped square as a fraction of the image; the
    paper's point is that it must stay SMALL -- a large window drags semantic
    content across, not just style.
    """
    ref_amp = np.abs(np.fft.fftshift(np.fft.fft2(ref.astype(np.float32),
                                                 axes=(0, 1)), axes=(0, 1)))

    def f(img):
        h, w = img.shape[:2]
        b = max(1, int(min(h, w) * beta))
        cy, cx = h // 2, w // 2
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32), axes=(0, 1)),
                            axes=(0, 1))
        amp, pha = np.abs(F), np.angle(F)
        amp[cy - b:cy + b, cx - b:cx + b] = \
            ref_amp[cy - b:cy + b, cx - b:cx + b]
        out = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha),
                                            axes=(0, 1)), axes=(0, 1))
        return np.clip(np.real(out), 0, 255).astype(np.uint8)

    return f


# ── model helpers ───────────────────────────────────────────────────────────
def build(ckpt, device):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd["model_state"])
    return m.to(device).eval()


def adabn(model, root, device, n_batches=25, batch=16):
    """Reset BN running stats and re-estimate them on TARGET images.

    No labels, no gradients: just forward passes in train() mode so the
    BatchNorm layers recompute mean/var against Jaipur's statistics instead of
    Christchurch's.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None            # cumulative average over all batches
    model.train()
    dl = DataLoader(Crops(root, limit=n_batches * batch), batch_size=batch,
                    num_workers=4)
    with torch.no_grad():
        for x, _ in dl:
            model(x.to(device))
    return model.eval()


@torch.no_grad()
def evaluate(model, ds, device, tta=False, batch=16):
    """Global (dataset-level) IoU per threshold, not the mean of per-crop IoUs.

    Per-crop averaging lets a crop with three foreground pixels count as much as
    a dense one, which flatters a model that under-predicts -- exactly the
    failure being measured here.
    """
    inter = {t: 0 for t in THRESHOLDS}
    union = {t: 0 for t in THRESHOLDS}
    tp = {t: 0 for t in THRESHOLDS}
    fp = {t: 0 for t in THRESHOLDS}
    fn = {t: 0 for t in THRESHOLDS}
    dl = DataLoader(ds, batch_size=batch, num_workers=4)
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x))[:, 0]
        if tta:
            for dims in ((3,), (2,), (2, 3)):
                p = p + torch.sigmoid(model(torch.flip(x, dims)))[:, 0].flip(
                    [d - 1 for d in dims])
            p = p / 4.0
        for t in THRESHOLDS:
            b = (p > t).float()
            inter[t] += (b * y).sum().item()
            union[t] += ((b + y) > 0).float().sum().item()
            tp[t] += (b * y).sum().item()
            fp[t] += (b * (1 - y)).sum().item()
            fn[t] += ((1 - b) * y).sum().item()
    out = {}
    for t in THRESHOLDS:
        prec = tp[t] / max(tp[t] + fp[t], 1)
        rec = tp[t] / max(tp[t] + fn[t], 1)
        out[f"{t:.2f}"] = {
            "iou": round(inter[t] / max(union[t], 1), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        }
    return out


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cmp] device {device}")

    # A source reference crop for the photometric methods. Any AIRS tile does;
    # this uses the first one present locally.
    ref = None
    if os.path.isdir(a.airs_dir):
        for f in sorted(os.listdir(a.airs_dir)):
            if f.endswith(".tif"):
                im = cv2.imread(os.path.join(a.airs_dir, f))
                if im is not None:
                    ref = cv2.cvtColor(im[:512, :512], cv2.COLOR_BGR2RGB)
                    print(f"[cmp] photometric reference: {f}")
                    break
    if ref is None:
        print("[cmp] WARNING: no AIRS reference found; skipping histmatch/fda")

    results = {}

    def run(name, model, transform=None, tta=False):
        print(f"\n[cmp] {name} ...", flush=True)
        ds = Crops(a.val_dir, transform=transform, limit=a.limit)
        results[name] = evaluate(model, ds, device, tta=tta)
        best = max(results[name].items(), key=lambda kv: kv[1]["iou"])
        print(f"[cmp] {name}: best IoU {best[1]['iou']:.4f} @ thr {best[0]} "
              f"| IoU@0.50 {results[name]['0.50']['iou']:.4f}")

    seed = build(a.seed_ckpt, device)
    run("seed", seed)
    run("seed_tta", seed, tta=True)
    if ref is not None:
        run("histmatch", seed, transform=make_histmatch(ref))
        run("fda", seed, transform=make_fda(ref, a.fda_beta))
    run("adabn", adabn(build(a.seed_ckpt, device), a.train_dir, device))
    if a.weak_ckpt and os.path.exists(a.weak_ckpt):
        run("weak", build(a.weak_ckpt, device))
        run("weak_tta", build(a.weak_ckpt, device), tta=True)

    payload = {
        "eval_set": a.val_dir,
        "n_crops": len(Crops(a.val_dir, limit=a.limit)),
        "caveats": [
            "eval labels are Open Buildings ground footprints, not roof outlines",
            "`weak` trained on this label distribution: home-field advantage",
            "best-threshold column is tuned on the eval set = upper bound",
        ],
        "results": results,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 78)
    print(f"{'method':12s} {'IoU@0.50':>9s} {'best IoU':>9s} {'@thr':>6s} "
          f"{'prec':>7s} {'rec':>7s}")
    print("-" * 78)
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["0.50"]["iou"]):
        b = max(v.items(), key=lambda kv: kv[1]["iou"])
        print(f"{k:12s} {v['0.50']['iou']:9.4f} {b[1]['iou']:9.4f} {b[0]:>6s} "
              f"{b[1]['precision']:7.4f} {b[1]['recall']:7.4f}")
    print("=" * 78)
    print(f"[cmp] wrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val_dir", default="data/jaipur_weak/val")
    p.add_argument("--train_dir", default="data/jaipur_weak/train")
    p.add_argument("--airs_dir", default="data/airs/image")
    p.add_argument("--seed_ckpt", default="rooftop/checkpoints/unet_resnet34_best.pth")
    p.add_argument("--weak_ckpt", default="")
    p.add_argument("--out", default="diagnostics/method_comparison.json")
    p.add_argument("--fda_beta", type=float, default=0.01)
    p.add_argument("--limit", type=int, default=0, help="cap crops (smoke test)")
    main(p.parse_args())
