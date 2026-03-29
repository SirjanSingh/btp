"""
train.py — DGX training script for rooftop / solar-panel segmentation.

Supports:
  • Folder-based dataset: images/ + masks/ subdirs (pre-tiled 512×512 crops)
  • Metadata-CSV dataset: CSV with image_name / mask_name columns
  • Architectures: UNet, UNet++, FPN, PSPNet, DeepLabV3+  (via SMP)
  • Encoders: resnet34/50, efficientnet-b4, etc.  (ImageNet pretrained)
  • Loss: BCE + Dice combined
  • Metrics: global IoU / F1 / Precision / Recall  (correct, matches AIRS paper methodology)
  • Mixed precision (AMP)
  • Multi-GPU via DataParallel
  • TensorBoard logging
  • Checkpoint to configurable path

Usage:
    python train.py \
        --train_dir /scratch/airs_crops/train \
        --val_dir   /scratch/airs_crops/val   \
        --ckpt_dir  /scratch/checkpoints      \
        --arch unet --encoder resnet34        \
        --epochs 50 --batch_size 8

See --help for all options.
"""

import argparse
import os
import random
import time
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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FolderDataset(Dataset):
    """Load pre-tiled crops from images/ + masks/ subdirectories."""

    def __init__(self, data_dir: str, transform=None):
        self.img_dir  = Path(data_dir) / "images"
        self.msk_dir  = Path(data_dir) / "masks"
        self.paths    = sorted(self.img_dir.glob("*.png"))
        self.transform = transform
        if not self.paths:
            raise FileNotFoundError(f"No .png files found in {self.img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ip = self.paths[idx]
        mp = self.msk_dir / ip.name

        img  = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]
            mask = out["mask"].unsqueeze(0)
        return img, mask


class CSVDataset(Dataset):
    """Load samples listed in a metadata CSV (image_name, mask_name columns)."""

    def __init__(self, csv_path: str, image_dir: str, mask_dir: str, transform=None):
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "image_name" in df.columns and "mask_name" in df.columns
        self.df        = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = cv2.cvtColor(
            cv2.imread(str(self.image_dir / row["image_name"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / row["mask_name"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def val_aug():
    return A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

ARCH_MAP = {
    "unet":       smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn":        smp.FPN,
    "pspnet":     smp.PSPNet,
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
# Metrics  — global accumulation (matches AIRS paper methodology)
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

def run_epoch(model, loader, optimizer, scaler, device, metrics, is_train):
    model.train() if is_train else model.eval()
    metrics.reset()
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
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
    if args.train_csv:
        train_ds = CSVDataset(args.train_csv, args.image_dir, args.mask_dir, train_aug())
        val_ds   = CSVDataset(args.val_csv,   args.image_dir, args.mask_dir, val_aug())
    else:
        train_ds = FolderDataset(args.train_dir, train_aug())
        val_ds   = FolderDataset(args.val_dir,   val_aug())

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
    # Encoder (pretrained) gets lower LR than fresh decoder
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    param_groups = [
        {"params": base_model.encoder.parameters(),         "lr": args.lr * 0.1},
        {"params": base_model.decoder.parameters(),         "lr": args.lr},
        {"params": base_model.segmentation_head.parameters(),"lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Checkpoint dir & TensorBoard ─────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.arch}_{args.encoder}"
    best_ckpt = ckpt_dir / f"{run_name}_best.pth"
    writer    = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs" / run_name))

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

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_m = run_epoch(
            model, train_loader, optimizer, scaler, device, train_metrics, is_train=True)
        va_loss, va_m = run_epoch(
            model, val_loader,   optimizer, scaler, device, val_metrics,   is_train=False)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[1]["lr"]

        # TensorBoard
        writer.add_scalars("loss",  {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("iou",   {"train": tr_m["iou"], "val": va_m["iou"]}, epoch)
        writer.add_scalars("f1",    {"train": tr_m["f1"],  "val": va_m["f1"]},  epoch)
        writer.add_scalar ("lr",    lr_now, epoch)

        note = ""
        if va_m["iou"] > best_val_iou:
            best_val_iou  = va_m["iou"]
            patience_left = args.patience
            note = "← best"
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch":          epoch,
                "model_state":    base_model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "val_iou":        va_m["iou"],
                "val_f1":         va_m["f1"],
                "args":           vars(args),
            }, best_ckpt)
        else:
            patience_left -= 1

        # Also save latest checkpoint every N epochs (for crash recovery)
        if epoch % args.save_every == 0:
            base_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "epoch":          epoch,
                "model_state":    base_model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "val_iou":        va_m["iou"],
            }, ckpt_dir / f"{run_name}_epoch{epoch:03d}.pth")

        print(f"{epoch:6d}  {tr_loss:8.4f}  {va_loss:8.4f}  "
              f"{va_m['iou']:7.4f}  {va_m['f1']:7.4f}  "
              f"{va_m['precision']:7.4f}  {va_m['recall']:7.4f}  "
              f"{lr_now:9.2e}  {note}")

        if patience_left <= 0:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
            break

    print(f"\nBest val IoU : {best_val_iou:.4f}  (PSPNet baseline: 0.899)")
    print(f"Checkpoint   : {best_ckpt}")
    writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train rooftop/PV segmentation model on DGX")

    # ── Data (folder mode) ───────────────────────────────────────────────────
    g = p.add_argument_group("Data — folder mode (pre-tiled crops)")
    g.add_argument("--train_dir", default=None, help="Dir with images/ masks/ for training")
    g.add_argument("--val_dir",   default=None, help="Dir with images/ masks/ for validation")

    # ── Data (CSV mode) ──────────────────────────────────────────────────────
    g2 = p.add_argument_group("Data — CSV mode")
    g2.add_argument("--train_csv",  default=None, help="CSV with image_name/mask_name for train")
    g2.add_argument("--val_csv",    default=None, help="CSV with image_name/mask_name for val")
    g2.add_argument("--image_dir",  default=None)
    g2.add_argument("--mask_dir",   default=None)

    # ── Model ────────────────────────────────────────────────────────────────
    g3 = p.add_argument_group("Model")
    g3.add_argument("--arch",            default="unet",
                    choices=["unet","unetplusplus","fpn","pspnet","deeplabv3plus"])
    g3.add_argument("--encoder",         default="resnet34")
    g3.add_argument("--encoder_weights", default="imagenet")

    # ── Training ─────────────────────────────────────────────────────────────
    g4 = p.add_argument_group("Training")
    g4.add_argument("--epochs",      type=int,   default=50)
    g4.add_argument("--batch_size",  type=int,   default=8)
    g4.add_argument("--lr",          type=float, default=1e-4)
    g4.add_argument("--threshold",   type=float, default=0.5,  help="Binarisation threshold")
    g4.add_argument("--patience",    type=int,   default=15,   help="Early-stop patience (epochs)")
    g4.add_argument("--workers",     type=int,   default=4)
    g4.add_argument("--seed",        type=int,   default=42)

    # ── Checkpoints ──────────────────────────────────────────────────────────
    g5 = p.add_argument_group("Checkpoints")
    g5.add_argument("--ckpt_dir",   default="./checkpoints")
    g5.add_argument("--resume",     default=None,  help="Path to checkpoint to resume from")
    g5.add_argument("--save_every", type=int, default=10,
                    help="Save a crash-recovery checkpoint every N epochs")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate: must provide either folder mode OR csv mode
    folder_mode = args.train_dir and args.val_dir
    csv_mode    = args.train_csv and args.val_csv and args.image_dir and args.mask_dir
    if not folder_mode and not csv_mode:
        raise SystemExit(
            "Error: provide either --train_dir/--val_dir  OR  "
            "--train_csv/--val_csv/--image_dir/--mask_dir"
        )

    main(args)
