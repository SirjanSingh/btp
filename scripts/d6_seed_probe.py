#!/usr/bin/env python
"""
D6 — seed-model probe on the unlabelled Jaipur target.

Runs the existing AIRS-trained checkpoint over random crops sampled from the
Jaipur mosaic and records what it PREDICTS. There are no target labels, so this
cannot measure accuracy — that is the point. What it measures is the model's
predicted foreground fraction and its confidence distribution.

Paired with D1 (the true building-pixel prior measured from Open Buildings),
this answers the question the whole project rests on: does an AIRS-trained model
systematically under-predict buildings on Indian imagery, and by how much?

If predicted foreground sits far below the D1 prior, that gap IS the motivating
figure, and it also tells you whether this checkpoint is usable as a self-training
teacher at all. A teacher that predicts 8% where truth is 45% will collapse under
self-training: sparse predictions -> sparse pseudo-labels -> sparser teacher.

Outputs (to --out_dir):
    d6_summary.json    aggregate + per-tile stats, threshold sweep
    d6_confidence.csv  histogram of sigmoid probabilities
    crops/*.png        qualitative overlays for eyeballing

Usage:
    python scripts/d6_seed_probe.py \
        --tiles data/jaipur --ckpt rooftop/checkpoints/unet_resnet34_best.pth \
        --n_crops 480 --out_dir diagnostics/d6
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import rasterio
import torch
from rasterio.windows import Window


def build_model(arch, encoder, ckpt_path, device):
    import segmentation_models_pytorch as smp

    model = getattr(smp, {"unet": "Unet"}[arch])(
        encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1
    )
    state = torch.load(ckpt_path, map_location="cpu")
    # This repo's train.py saves {"epoch","model_state","optimizer_state",
    # "val_iou","val_f1","args"}; other conventions are accepted too. DataParallel
    # training prefixes every key with "module.".
    for key in ("model_state", "model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def sample_crops(tif_paths, n_crops, crop, seed):
    """Sample crop windows spread evenly across all mosaic tiles."""
    rng = random.Random(seed)
    per_tile = max(1, n_crops // len(tif_paths))
    jobs = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            w, h = src.width, src.height
        for _ in range(per_tile):
            jobs.append((p, rng.randint(0, w - crop - 1), rng.randint(0, h - crop - 1)))
    rng.shuffle(jobs)
    return jobs[:n_crops]


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    tifs = sorted(
        os.path.join(a.tiles, f) for f in os.listdir(a.tiles) if f.endswith(".tif")
    )
    if not tifs:
        raise SystemExit(f"no .tif under {a.tiles}")

    # Read the true ground GSD from the file rather than assuming it. Web-Mercator
    # pixel scale is not ground distance: it must be scaled by cos(latitude).
    with rasterio.open(tifs[0]) as src:
        merc_scale = abs(src.transform.a)
        b = src.bounds
        lat = math.degrees(
            2 * math.atan(math.exp((b.top + b.bottom) / 2 / 6378137.0)) - math.pi / 2
        )
    gsd = merc_scale * math.cos(math.radians(lat))

    print(f"[D6] {len(tifs)} tiles | mercator {merc_scale:.7f} | lat {lat:.4f}")
    print(f"[D6] TRUE ground GSD = {gsd:.5f} m/px  (AIRS = 0.075 -> gap {gsd/0.075:.2f}x)")

    model = build_model(a.arch, a.encoder, a.ckpt, device)
    jobs = sample_crops(tifs, a.n_crops, a.crop, a.seed)
    print(f"[D6] {len(jobs)} crops of {a.crop}px on {device}")

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    hist = np.zeros(50, dtype=np.int64)          # sigmoid prob histogram, 0..1
    fg_at = {t: [] for t in thresholds}          # per-crop fg fraction
    per_tile = defaultdict(list)
    blank = 0

    with torch.no_grad():
        for i in range(0, len(jobs), a.batch):
            chunk = jobs[i : i + a.batch]
            imgs, keys = [], []
            for path, x, y in chunk:
                with rasterio.open(path) as src:
                    arr = src.read(
                        (1, 2, 3), window=Window(x, y, a.crop, a.crop)
                    ).astype(np.float32)
                if arr.max() == 0:               # padding / off-mosaic
                    blank += 1
                    continue
                imgs.append(arr / 255.0)
                keys.append(os.path.basename(path))
            if not imgs:
                continue

            batch = torch.from_numpy(np.stack(imgs)).to(device)
            # ImageNet normalisation — the encoder was pretrained with it.
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            prob = torch.sigmoid(model((batch - mean) / std)).squeeze(1).cpu().numpy()

            hist += np.histogram(prob, bins=50, range=(0, 1))[0]
            for j, key in enumerate(keys):
                for t in thresholds:
                    fg_at[t].append(float((prob[j] > t).mean()))
                per_tile[key].append(float((prob[j] > a.ref_threshold).mean()))

            if i % (a.batch * 10) == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    summary = {
        "gsd_true_m_per_px": round(gsd, 5),
        "gsd_gap_vs_airs": round(gsd / 0.075, 3),
        "n_crops_scored": len(fg_at[a.ref_threshold]),
        "n_blank_skipped": blank,
        "crop_px": a.crop,
        "checkpoint": a.ckpt,
        "ref_threshold": a.ref_threshold,
        "predicted_fg_fraction": {
            str(t): {
                "mean": round(float(np.mean(v)), 5),
                "median": round(float(np.median(v)), 5),
                "p10": round(float(np.percentile(v, 10)), 5),
                "p90": round(float(np.percentile(v, 90)), 5),
                "frac_crops_empty": round(float(np.mean(np.array(v) < 1e-4)), 4),
            }
            for t, v in fg_at.items()
        },
        "per_tile_mean_fg": {
            k: round(float(np.mean(v)), 5) for k, v in sorted(per_tile.items())
        },
    }
    with open(os.path.join(a.out_dir, "d6_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    edges = np.linspace(0, 1, 51)
    with open(os.path.join(a.out_dir, "d6_confidence.csv"), "w") as fh:
        fh.write("bin_low,bin_high,count,fraction\n")
        tot = hist.sum()
        for k in range(50):
            fh.write(f"{edges[k]:.2f},{edges[k+1]:.2f},{hist[k]},{hist[k]/tot:.6f}\n")

    ref = summary["predicted_fg_fraction"][str(a.ref_threshold)]
    print("\n" + "=" * 62)
    print(f"D6 RESULT  (threshold {a.ref_threshold})")
    print("=" * 62)
    print(f"  predicted foreground  mean   {ref['mean']*100:.2f}%")
    print(f"                        median {ref['median']*100:.2f}%")
    print(f"                        p10-p90 {ref['p10']*100:.2f}% - {ref['p90']*100:.2f}%")
    print(f"  crops predicted empty        {ref['frac_crops_empty']*100:.1f}%")
    print(f"  confidence mass below 0.1    {hist[:5].sum()/hist.sum()*100:.1f}%")
    print(f"  confidence mass above 0.9    {hist[45:].sum()/hist.sum()*100:.1f}%")
    print("=" * 62)
    print(f"  -> compare against D1 measured prior. Large shortfall = the")
    print(f"     under-prediction this project exists to fix.")
    print(f"  written: {a.out_dir}/d6_summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles", default="data/jaipur")
    p.add_argument("--ckpt", default="rooftop/checkpoints/unet_resnet34_best.pth")
    p.add_argument("--out_dir", default="diagnostics/d6")
    p.add_argument("--arch", default="unet")
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--n_crops", type=int, default=480)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ref_threshold", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
