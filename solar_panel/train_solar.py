"""
train_solar.py — Training script for solar panel segmentation.

Key difference from rooftop training:
  • Many images have NO solar panels → no paired mask file exists.
  • Instead of skipping those images, we fall back to an all-black (all-zero)
    mask, which is the correct ground truth: "no solar panels present."
  • This provides essential negative examples so the model learns to predict
    nothing when there are no panels, not just learn what panels look like.

Supports:
  • Folder-based dataset: images/ + masks/ subdirs (pre-tiled 512×512 crops)
    - If a mask file is missing for an image, a black mask is used (see above).
  • Architectures: UNet, UNet++, FPN, PSPNet, DeepLabV3+  (via SMP)
  • Encoders: resnet34/50, efficientnet-b4, etc.  (ImageNet pretrained)
  • Loss: BCE + Dice combined
  • Metrics: global IoU / F1 / Precision / Recall
  • Mixed precision (AMP)
  • Multi-GPU via DataParallel
  • TensorBoard logging  →  solar_panel/logs/
  • Checkpoints          →  solar_panel/checkpoints/

Usage:
    python solar_panel/train_solar.py \
        --train_dir /data/solar/train \
        --val_dir   /data/solar/val   \
        --arch unet --encoder resnet34 \
        --epochs 50 --batch_size 8

See --help for all options.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Run Logger
# ─────────────────────────────────────────────────────────────────────────────

class RunLogger:
    def __init__(self, args, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag  = f"{args.arch}_{args.encoder}"
        if getattr(args, "max_samples", None):
            tag += f"_{args.max_samples}samples"
        tag += f"_{args.epochs}ep"
        stem = f"{tag}_{ts}"
        self.stem      = stem
        self.json_path = self.log_dir / f"{stem}.json"
        self.txt_path  = self.log_dir / f"{stem}.txt"
        hw = {"device": "cpu"}
        if torch.cuda.is_available():
            hw = {"device": "cuda", "n_gpus": torch.cuda.device_count(),
                  "gpu_name": torch.cuda.get_device_name(0),
                  "cuda_version": torch.version.cuda}
        self.record = {"run_name": stem, "timestamp": datetime.now().isoformat(),
                       "hardware": hw, "config": vars(args), "epochs": [], "summary": {}}
        self._txt_write(
            f"Run: {stem}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Hardware: {hw}\nConfig: {json.dumps(vars(args), indent=2)}\n\n"
            f"{'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  "
            f"{'IoU':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'LR':>9}  Note\n"
            + "─" * 80 + "\n")

    def log_epoch(self, epoch, tr_loss, va_loss, va_m, lr, elapsed, note=""):
        entry = {"epoch": epoch, "tr_loss": round(tr_loss, 6), "va_loss": round(va_loss, 6),
                 **{k: round(v, 6) for k, v in va_m.items()}, "lr": lr,
                 "elapsed_s": round(elapsed, 1), "note": note}
        self.record["epochs"].append(entry)
        self._flush_json()
        self._txt_write(
            f"{epoch:6d}  {tr_loss:8.4f}  {va_loss:8.4f}  "
            f"{va_m['iou']:7.4f}  {va_m['f1']:7.4f}  "
            f"{va_m['precision']:7.4f}  {va_m['recall']:7.4f}  "
            f"{lr:9.2e}  {elapsed:8.1f}  {note}\n")

    def finish(self, best_iou, best_epoch, ckpt_path):
        self.record["summary"] = {"best_val_iou": round(best_iou, 6),
                                   "best_epoch": best_epoch, "checkpoint": str(ckpt_path)}
        self._flush_json()
        self._txt_write("─" * 80 + f"\nBest val IoU : {best_iou:.4f}  (epoch {best_epoch})\n"
                        f"Checkpoint   : {ckpt_path}\n")
        print(f"[log] {self.json_path}")

    def _flush_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.record, f, indent=2)

    def _txt_write(self, text):
        with open(self.txt_path, "a") as f:
            f.write(text)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SolarFolderDataset(Dataset):
    """
    Load pre-tiled crops from images/ subdirectory.

    Mask handling:
      - If a matching file exists in masks/, load it normally.
      - If NO mask file exists (image has no solar panels), fall back to an
        all-black mask (all zeros). This is the correct ground truth for
        negative examples and keeps them in training.

    Reports how many images use the black-mask fallback at construction time.
    """

    def __init__(self, data_dir: str, transform=None, max_samples: int = None):
        self.img_dir   = Path(data_dir) / "images"
        self.msk_dir   = Path(data_dir) / "masks"
        self.transform = transform

        all_paths = sorted(self.img_dir.glob("*.png"))
        self.paths = all_paths[:max_samples] if max_samples else all_paths
        if not self.paths:
            raise FileNotFoundError(f"No .png files found in {self.img_dir}")

        # Count how many images lack a paired mask
        missing = sum(1 for p in self.paths if not (self.msk_dir / p.name).exists())
        print(f"[SolarFolderDataset] {data_dir}: "
              f"{len(self.paths)} images, "
              f"{len(self.paths) - missing} with masks, "
              f"{missing} without masks (→ black mask fallback)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ip = self.paths[idx]
        mp = self.msk_dir / ip.name

        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)

        if mp.exists():
            # Normal case: solar panels present, load ground-truth mask
            mask_raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            mask = (mask_raw > 127).astype(np.float32)
        else:
            # Fallback: no solar panels in this image → all-black mask
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].unsqueeze(0)

        return img, mask


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)


def train_aug(crop_size: int = 512):
    return A.Compose([
        A.Resize(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def val_aug(crop_size: int = 512):
    return A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

ARCH_MAP = {
    "unet":          smp.Unet,
    "unetplusplus":  smp.UnetPlusPlus,
    "fpn":           smp.FPN,
    "pspnet":        smp.PSPNet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(arch: str, encoder: str, encoder_weights: str = "imagenet") -> nn.Module:
    cls = ARCH_MAP.get(arch.lower().replace("-", "").replace("+", "plus"))
    if cls is None:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_MAP)}")
    return cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

_bce  = smp.losses.SoftBCEWithLogitsLoss()
_dice = smp.losses.DiceLoss(mode="binary", from_logits=True)


def combined_loss(pred, target, bce_w=0.5, dice_w=0.5):
    return bce_w * _bce(pred, target) + dice_w * _dice(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  — global accumulation
# ─────────────────────────────────────────────────────────────────────────────

class GlobalMetrics:
    """Accumulate TP/FP/FN across all batches, compute metrics at epoch end."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()

    def compute(self) -> dict:
        tp, fp, fn = self.tp, self.fp, self.fn
        iou       = tp / (tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1        = 2 * precision * recall / (precision + recall + 1e-7)
        return {"iou": iou, "f1": f1, "precision": precision, "recall": recall}


# ─────────────────────────────────────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scaler, device, metrics, is_train,
              epoch=0, n_epochs=0):
    model.train() if is_train else model.eval()
    metrics.reset()
    total_loss = 0.0

    phase = "Train" if is_train else "Val  "
    pbar  = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs} {phase}",
                 leave=False, dynamic_ncols=True)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(imgs)
                loss  = combined_loss(preds, masks)

            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            metrics.update(preds, masks)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        n_gpus = torch.cuda.device_count()
        print(f"GPUs    : {n_gpus} × {torch.cuda.get_device_name(0)}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    # SolarFolderDataset automatically handles images without paired masks
    # by using an all-black (all-zero) fallback mask.
    val_cap = None
    if args.max_samples:
        import glob
        n_train = len(glob.glob(str(Path(args.train_dir) / "images" / "*.png")))
        ratio   = args.max_samples / max(n_train, 1)
        n_val   = len(glob.glob(str(Path(args.val_dir) / "images" / "*.png")))
        val_cap = max(1, int(n_val * ratio))

    train_ds = SolarFolderDataset(args.train_dir, train_aug(args.crop_size), max_samples=args.max_samples)
    val_ds   = SolarFolderDataset(args.val_dir,   val_aug(args.crop_size),   max_samples=val_cap)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(f"Train   : {len(train_ds)} crops  ({len(train_loader)} batches)")
    print(f"Val     : {len(val_ds)} crops  ({len(val_loader)} batches)")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.arch, args.encoder, args.encoder_weights)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Wrapping in DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model   : {args.arch} + {args.encoder}  ({n_params:.1f}M params)")

    # ── Optimiser & Scheduler ────────────────────────────────────────────────
    # Encoder (pretrained) gets 10× lower LR than fresh decoder
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = [
        {"params": base_model.encoder.parameters(),          "lr": args.lr * 0.1},
        {"params": base_model.decoder.parameters(),          "lr": args.lr},
        {"params": base_model.segmentation_head.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Checkpoint dir & TensorBoard ─────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # RunLogger generates the timestamped stem — reuse it for the ckpt folder
    # so every run gets its own subdirectory and never overwrites another run.
    run_logger = RunLogger(args, log_dir=str(log_dir))
    ckpt_dir   = Path(args.ckpt_dir) / run_logger.stem
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "best.pth"
    writer    = SummaryWriter(log_dir=str(log_dir / run_logger.stem))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_val_iou  = 0.0
    patience_left = args.patience

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        base_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("val_iou", 0.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, IoU {best_val_iou:.4f})")

    # ── Training loop ────────────────────────────────────────────────────────
    train_metrics = GlobalMetrics(args.threshold)
    val_metrics   = GlobalMetrics(args.threshold)

    print(f"\n{'─'*72}")
    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'IoU':>7}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'LR':>9}  Note")
    print(f"{'─'*72}")

    best_epoch = start_epoch

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_m = run_epoch(
            model, train_loader, optimizer, scaler, device, train_metrics,
            is_train=True, epoch=epoch, n_epochs=args.epochs)
        va_loss, va_m = run_epoch(
            model, val_loader,   optimizer, scaler, device, val_metrics,
            is_train=False, epoch=epoch, n_epochs=args.epochs)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[1]["lr"]

        # TensorBoard
        writer.add_scalars("loss",  {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("iou",   {"train": tr_m["iou"], "val": va_m["iou"]}, epoch)
        writer.add_scalars("f1",    {"train": tr_m["f1"],  "val": va_m["f1"]},  epoch)
        writer.add_scalar ("lr",    lr_now, epoch)

        elapsed = time.time() - t0
        note = ""
        if va_m["iou"] > best_val_iou:
            best_val_iou  = va_m["iou"]
            best_epoch    = epoch
            patience_left = args.patience
            note = "← best"
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch":           epoch,
                "model_state":     base_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou":         va_m["iou"],
                "val_f1":          va_m["f1"],
                "args":            vars(args),
            }, best_ckpt)
        else:
            patience_left -= 1

        # Periodic checkpoint for crash recovery
        if epoch % args.save_every == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch":           epoch,
                "model_state":     base_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou":         va_m["iou"],
            }, ckpt_dir / f"epoch{epoch:03d}.pth")

        print(f"{epoch:6d}  {tr_loss:8.4f}  {va_loss:8.4f}  "
              f"{va_m['iou']:7.4f}  {va_m['f1']:7.4f}  "
              f"{va_m['precision']:7.4f}  {va_m['recall']:7.4f}  "
              f"{lr_now:9.2e}  {note}")

        run_logger.log_epoch(epoch, tr_loss, va_loss, va_m, lr_now, elapsed, note)

        if patience_left <= 0:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
            break

    run_logger.finish(best_val_iou, best_epoch, best_ckpt)
    print(f"\nBest val IoU : {best_val_iou:.4f}")
    print(f"Checkpoint   : {best_ckpt}")
    writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train solar panel segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data — folder mode (pre-tiled crops)")
    g.add_argument("--max_samples", type=int, default=None,
                   help="Cap training samples (val capped proportionally)")
    g.add_argument("--train_dir", required=True,
                   help="Directory with images/ (and optionally masks/) for training. "
                        "Images without a paired mask file use an all-black mask.")
    g.add_argument("--val_dir", required=True,
                   help="Directory with images/ (and optionally masks/) for validation.")

    # ── Model ────────────────────────────────────────────────────────────────
    g2 = p.add_argument_group("Model")
    g2.add_argument("--arch", default="unet",
                    choices=["unet", "unetplusplus", "fpn", "pspnet", "deeplabv3plus"])
    g2.add_argument("--encoder",         default="resnet34")
    g2.add_argument("--encoder_weights", default="imagenet")

    # ── Training ─────────────────────────────────────────────────────────────
    g3 = p.add_argument_group("Training")
    g3.add_argument("--epochs",     type=int,   default=50)
    g3.add_argument("--batch_size", type=int,   default=8)
    g3.add_argument("--crop_size",  type=int,   default=512,
                    help="Resize input images to this size (must be divisible by 32)")
    g3.add_argument("--lr",         type=float, default=1e-4)
    g3.add_argument("--threshold",  type=float, default=0.5,
                    help="Binarisation threshold for metrics")
    g3.add_argument("--patience",   type=int,   default=15,
                    help="Early-stop patience (epochs)")
    g3.add_argument("--workers",    type=int,   default=4)
    g3.add_argument("--seed",       type=int,   default=42)

    # ── Paths ─────────────────────────────────────────────────────────────────
    g4 = p.add_argument_group("Output paths")
    g4.add_argument("--ckpt_dir",   default="solar_panel/checkpoints",
                    help="Directory to save model checkpoints")
    g4.add_argument("--log_dir",    default="solar_panel/logs",
                    help="Directory for TensorBoard logs")
    g4.add_argument("--resume",     default=None,
                    help="Path to a checkpoint to resume training from")
    g4.add_argument("--save_every", type=int, default=10,
                    help="Save a crash-recovery checkpoint every N epochs")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
