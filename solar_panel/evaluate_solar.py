"""
evaluate_solar.py — Run evaluation on the solar panel test split.

Outputs
-------
  eval_results/
    per_sample_iou.csv          — IoU for every test crop
    overlays/<name>_overlay.png — side-by-side: image | GT | prediction | blend
    eval_<run>_<timestamp>.json — structured log (config + metrics + threshold sweep)
    eval_<run>_<timestamp>.txt  — human-readable summary

Key difference from rooftop evaluate.py:
  Uses SolarFolderDataset (with black-mask fallback for images without panels).

Usage:
    python solar_panel/evaluate_solar.py \
        --test_dir  /tmp/bdappv_crops/test \
        --ckpt      solar_panel/checkpoints/unet_resnet34_best.pth \
        --out_dir   solar_panel/eval_results \
        --arch unet --encoder resnet34 \
        --sweep_threshold
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train_solar import SolarFolderDataset, build_model, GlobalMetrics, val_aug

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def denorm(tensor):
    t = tensor.permute(1, 2, 0).cpu().numpy()
    t = t * np.array(STD) + np.array(MEAN)
    return np.clip(t, 0, 1)


def save_overlay(img_t, pred_mask, gt_mask, save_path, iou):
    """
    Save a 4-panel PNG:
      Panel 1 — original aerial image
      Panel 2 — ground truth mask (blue = solar panel)
      Panel 3 — predicted mask (white = predicted panel)
      Panel 4 — prediction blended on image (yellow tint = predicted panel)
    """
    img = (denorm(img_t) * 255).astype(np.uint8)

    overlay = img.copy().astype(np.float32)
    overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([255, 255, 0]) * 0.5

    gt_vis = np.zeros_like(img)
    gt_vis[gt_mask == 1] = [0, 100, 255]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img);                      axes[0].set_title("Image")
    axes[1].imshow(gt_vis);                   axes[1].set_title("Ground Truth (blue=panel)")
    axes[2].imshow(pred_mask, cmap="gray");   axes[2].set_title(f"Prediction  IoU={iou:.3f}")
    axes[3].imshow(overlay.astype(np.uint8)); axes[3].set_title("Overlay (yellow=predicted)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Eval Logger
# ─────────────────────────────────────────────────────────────────────────────

class EvalLogger:
    def __init__(self, args, ckpt_meta: dict, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"eval_{args.arch}_{args.encoder}_{ts}"
        if args.max_samples:
            stem = f"eval_{args.arch}_{args.encoder}_{args.max_samples}samples_{ts}"

        self.json_path = self.log_dir / f"{stem}.json"
        self.txt_path  = self.log_dir / f"{stem}.txt"

        hw = {"device": "cpu"}
        if torch.cuda.is_available():
            hw = {"device": "cuda", "n_gpus": torch.cuda.device_count(),
                  "gpu_name": torch.cuda.get_device_name(0),
                  "cuda_version": torch.version.cuda}

        self.record = {
            "run_name":        stem,
            "timestamp":       datetime.now().isoformat(),
            "hardware":        hw,
            "config":          vars(args),
            "checkpoint":      ckpt_meta,
            "threshold_sweep": [],
            "results":         {},
        }

        self._txt_write(
            f"Eval Run  : {stem}\n"
            f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Checkpoint: epoch={ckpt_meta.get('epoch','?')}  "
            f"val_IoU={ckpt_meta.get('val_iou', 0):.4f}\n"
            f"Hardware  : {hw}\n"
            f"Config    : {json.dumps(vars(args), indent=2)}\n\n"
        )

    def log_sweep(self, threshold: float, metrics: dict):
        entry = {"threshold": threshold, **{k: round(v, 6) for k, v in metrics.items()}}
        self.record["threshold_sweep"].append(entry)
        self._flush_json()
        self._txt_write(
            f"  threshold={threshold:.2f}  IoU={metrics['iou']:.4f}"
            f"  F1={metrics['f1']:.4f}  Prec={metrics['precision']:.4f}"
            f"  Rec={metrics['recall']:.4f}\n"
        )

    def finish(self, results: dict, threshold: float, n_samples: int,
               csv_path: str, out_dir: str):
        self.record["results"] = {
            "threshold":      threshold,
            "n_test_samples": n_samples,
            "iou":            round(results["iou"],       6),
            "f1":             round(results["f1"],        6),
            "precision":      round(results["precision"], 6),
            "recall":         round(results["recall"],    6),
            "csv_path":       str(csv_path),
            "out_dir":        str(out_dir),
        }
        self._flush_json()
        self._txt_write(
            "\n" + "═" * 50 + "\n"
            f" Test Results  (threshold={threshold}  n={n_samples})\n"
            "─" * 50 + "\n"
            f"  Global IoU  : {results['iou']:.4f}\n"
            f"  F1 Score    : {results['f1']:.4f}\n"
            f"  Precision   : {results['precision']:.4f}\n"
            f"  Recall      : {results['recall']:.4f}\n"
            "═" * 50 + "\n"
            f"Log (JSON) : {self.json_path}\n"
            f"Log (TXT)  : {self.txt_path}\n"
        )
        print(f"[log] {self.json_path}")
        print(f"[log] {self.txt_path}")

    def _flush_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.record, f, indent=2)

    def _txt_write(self, text: str):
        with open(self.txt_path, "a") as f:
            f.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_ds = SolarFolderDataset(args.test_dir, val_aug(args.crop_size),
                                 max_samples=args.max_samples)
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"Test crops : {len(test_ds)}"
          + ("  (capped)" if args.max_samples else ""))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.arch, args.encoder)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    ckpt_meta = {
        "path":    args.ckpt,
        "epoch":   ckpt.get("epoch", "?"),
        "val_iou": float(ckpt.get("val_iou", 0)),
        "val_f1":  float(ckpt.get("val_f1",  0)),
    }
    print(f"Loaded     : {args.ckpt}  "
          f"(epoch {ckpt_meta['epoch']}, val IoU {ckpt_meta['val_iou']:.4f})")

    # ── Logger ────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    logger = EvalLogger(args, ckpt_meta, log_dir=args.log_dir)

    # ── Threshold sweep ───────────────────────────────────────────────────────
    if args.sweep_threshold:
        print("\nSweeping thresholds ...")
        logger._txt_write("Threshold sweep:\n")
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
        best_t, best_iou = 0.5, 0.0
        for t in thresholds:
            m = GlobalMetrics(threshold=t)
            with torch.no_grad():
                for imgs, masks in loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        preds = model(imgs)
                    m.update(preds, masks)
            res = m.compute()
            logger.log_sweep(t, res)
            print(f"  t={t:.2f}  IoU={res['iou']:.4f}  F1={res['f1']:.4f}")
            if res["iou"] > best_iou:
                best_iou, best_t = res["iou"], t
        args.threshold = best_t
        print(f"Best threshold: {best_t}  →  IoU {best_iou:.4f}\n")

    # ── Full evaluation ───────────────────────────────────────────────────────
    gm = GlobalMetrics(threshold=args.threshold)
    per_sample = []
    img_paths  = sorted((Path(args.test_dir) / "images").glob("*.png"))
    if args.max_samples:
        img_paths = img_paths[:args.max_samples]
    idx = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
            gm.update(logits, masks)

            preds_bin = (torch.sigmoid(logits) > args.threshold).float()
            for b in range(imgs.size(0)):
                p = preds_bin[b, 0].cpu().numpy().astype(np.uint8)
                g = masks[b, 0].cpu().numpy().astype(np.uint8)
                tp = (p * g).sum()
                fp = (p * (1 - g)).sum()
                fn = ((1 - p) * g).sum()
                sample_iou = tp / (tp + fp + fn + 1e-7)
                name = img_paths[idx].stem if idx < len(img_paths) else f"sample_{idx}"
                per_sample.append({"name": name, "iou": float(sample_iou)})

                if idx < args.n_overlays:
                    save_overlay(
                        imgs[b].cpu(), p, g,
                        out_dir / "overlays" / f"{name}_overlay.png",
                        sample_iou,
                    )
                idx += 1

    results = gm.compute()

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*50}")
    print(f" Test Results  (threshold={args.threshold}  n={len(per_sample)})")
    print(f"{'─'*50}")
    print(f"  Global IoU  : {results['iou']:.4f}")
    print(f"  F1 Score    : {results['f1']:.4f}")
    print(f"  Precision   : {results['precision']:.4f}")
    print(f"  Recall      : {results['recall']:.4f}")
    print(f"{'═'*50}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / "per_sample_iou.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "iou"])
        w.writeheader()
        w.writerows(per_sample)
    print(f"Per-sample CSV : {csv_path}")
    print(f"Overlays       : {out_dir / 'overlays'}  ({min(args.n_overlays, len(per_sample))} saved)")

    logger.finish(results, args.threshold, len(per_sample), csv_path, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate solar panel segmentation on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--test_dir",        required=True,
                   help="Dir with images/ (and optionally masks/) — test split")
    p.add_argument("--ckpt",            required=True,
                   help="Path to best checkpoint .pth")
    p.add_argument("--out_dir",         default="solar_panel/eval_results",
                   help="Output dir for overlays + CSV")
    p.add_argument("--log_dir",         default="solar_panel/logs",
                   help="Dir for JSON + TXT eval logs")
    p.add_argument("--arch",            default="unet",
                   choices=["unet","unetplusplus","fpn","pspnet","deeplabv3plus"])
    p.add_argument("--encoder",         default="resnet34")
    p.add_argument("--crop_size",       type=int,   default=512,
                   help="Must match training crop_size")
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--workers",         type=int,   default=2)
    p.add_argument("--threshold",       type=float, default=0.5)
    p.add_argument("--max_samples",     type=int,   default=None,
                   help="Cap test samples for quick eval")
    p.add_argument("--sweep_threshold", action="store_true",
                   help="Sweep thresholds 0.30–0.55 and pick best automatically")
    p.add_argument("--n_overlays",      type=int,   default=20,
                   help="Number of overlay PNGs to save (0 to skip)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
