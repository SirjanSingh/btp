# 2026-09-10-segformer-backbone — does a transformer encoder beat ResNet-34 here?

| | |
|---|---|
| **Status** | running |
| **Date** | 2026-09-10 |

## Question

Swap the U-Net encoder from **ResNet-34 (24.4M)** to **MiT-B2 (27.5M)**, the SegFormer
backbone, and retrain on the same weak labels.

**Why it matters, and why it is next:** DAFormer's central empirical claim is that the
*architecture* mattered more than the adaptation algorithm — self-attention features are less
domain-specific than early convolutional ones. `plan/03` §2.3 asks for this table regardless
of which wins.

It is also what [R2](../2026-09-09-does-the-airs-seed-help/) leaves as the live lever. R2
showed source pretraining contributes nothing here, so gains have to come from the target
side: architecture, label quality, or scale.

**Prediction (before running):** **+0.03 to +0.08 IoU** (so ~0.68–0.73). A transformer's
global receptive field should help most where buildings are dense and share walls — exactly
Jaipur, and exactly the instance-merging failure the project worries about. Risk: 7,371 crops
is small for a transformer, which may underperform at this data scale.

Secondary: if MiT wins mainly on **precision**, that suggests better boundary separation
between adjacent buildings, which is the interesting claim. A pure recall gain would be less
meaningful.

## Setup

Identical to the weak-supervision baseline except `--encoder mit_b2`. ImageNet init (R2
showed the AIRS seed is worthless, so it is dropped — this also keeps the comparison clean,
since no AIRS checkpoint exists for a MiT encoder).

```bash
./run_docker.sh 7 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak/train --val_dir data/jaipur_weak/val \
    --arch unet --encoder mit_b2 \
    --epochs 40 --batch_size 12 --workers 3 --save_every 999 \
    --ckpt_dir experiments/2026-09-10-segformer-backbone/checkpoints \
    --log_dir  experiments/2026-09-10-segformer-backbone/outputs"
```

Batch 12 rather than 16: MiT-B2 attention is heavier per sample, and the box is shared.
`DeepLabV3+` was ruled out — smp raises `MixVisionTransformer encoder does not support
dilated mode`. FPN also works if U-Net disappoints.

## Results

*pending*

## Threats to validity

- **Batch size differs from the baseline (12 vs 16)**, so this is not a pure encoder
  ablation — batch size affects BatchNorm statistics and effective learning rate. A confound
  worth stating rather than hiding.
- Same LR and schedule as the ResNet run; transformers often want lower LR and warmup, so an
  underperformance here may reflect tuning rather than architecture.
- Single seed. Footprint labels, not roof ground truth.
