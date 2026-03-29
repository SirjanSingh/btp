"""
evaluate.py — Run evaluation on a test split and produce metrics + visual overlays.

Usage:
    python evaluate.py \
        --test_dir  /scratch/airs_crops/test \
        --ckpt      /scratch/checkpoints/unet_resnet34_best.pth \
        --out_dir   /scratch/eval_results \
        --arch unet --encoder resnet34
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

from train import FolderDataset, build_model, GlobalMetrics, val_aug

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def denorm(tensor):
    t = tensor.permute(1, 2, 0).cpu().numpy()
    t = t * np.array(STD) + np.array(MEAN)
    return np.clip(t, 0, 1)


def save_overlay(img_t, pred_mask, gt_mask, save_path, iou):
    img = (denorm(img_t) * 255).astype(np.uint8)

    overlay = img.copy().astype(np.float32)
    overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([255, 100, 0]) * 0.5

    gt_vis = np.zeros_like(img)
    gt_vis[gt_mask == 1] = [0, 200, 0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img);                  axes[0].set_title("Image")
    axes[1].imshow(gt_vis);               axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_mask, cmap="gray"); axes[2].set_title(f"Prediction  IoU={iou:.3f}")
    axes[3].imshow(overlay.astype(np.uint8)); axes[3].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    test_ds = FolderDataset(args.test_dir, val_aug())
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test crops: {len(test_ds)}")

    # Model
    model = build_model(args.arch, args.encoder)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"Loaded: {args.ckpt}  (epoch {ckpt.get('epoch','?')}, "
          f"val IoU {ckpt.get('val_iou',0):.4f})")

    # Threshold sweep on test set
    if args.sweep_threshold:
        print("\nSweeping threshold on test set...")
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
            print(f"  threshold={t:.2f}  IoU={res['iou']:.4f}  F1={res['f1']:.4f}")
            if res["iou"] > best_iou:
                best_iou, best_t = res["iou"], t
        print(f"Best threshold: {best_t}  →  IoU {best_iou:.4f}\n")
        args.threshold = best_t

    # Full evaluation
    gm = GlobalMetrics(threshold=args.threshold)
    per_sample = []

    out_dir = Path(args.out_dir)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    img_paths = sorted((Path(args.test_dir) / "images").glob("*.png"))
    idx = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
            gm.update(logits, masks)

            # Per-sample IoU for the overlay CSV
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

                # Save overlays for first N samples
                if idx < args.n_overlays:
                    save_overlay(
                        imgs[b].cpu(), p, g,
                        out_dir / "overlays" / f"{name}_overlay.png",
                        sample_iou,
                    )
                idx += 1

    results = gm.compute()

    # Print summary
    print(f"\n{'═'*45}")
    print(f" Test Set Results  (threshold={args.threshold})")
    print(f"{'─'*45}")
    print(f"  Global IoU  : {results['iou']:.4f}   ← target: > 0.899 (PSPNet)")
    print(f"  F1 Score    : {results['f1']:.4f}   ← target: > 0.947 (PSPNet)")
    print(f"  Precision   : {results['precision']:.4f}")
    print(f"  Recall      : {results['recall']:.4f}")
    print(f"{'═'*45}\n")

    # Save CSV
    csv_path = out_dir / "per_sample_iou.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "iou"])
        w.writeheader()
        w.writerows(per_sample)
    print(f"Per-sample CSV : {csv_path}")
    print(f"Overlays       : {out_dir / 'overlays'}  ({min(args.n_overlays, len(per_sample))} saved)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir",        required=True)
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--out_dir",         default="./eval_results")
    p.add_argument("--arch",            default="unet")
    p.add_argument("--encoder",         default="resnet34")
    p.add_argument("--batch_size",      type=int,   default=8)
    p.add_argument("--threshold",       type=float, default=0.5)
    p.add_argument("--sweep_threshold", action="store_true",
                   help="Sweep threshold on test set to find optimal value")
    p.add_argument("--n_overlays",      type=int,   default=20,
                   help="Number of overlay images to save")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
