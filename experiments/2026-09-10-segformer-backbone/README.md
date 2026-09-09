# 2026-09-10-segformer-backbone — does a transformer encoder beat ResNet-34 here?

| | |
|---|---|
| **Status** | ✅ done — modest win |
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

| | ResNet-34 (ImageNet) | **MiT-B2** | delta |
|---|---|---|---|
| best val IoU | 0.6483 @ep33 | **0.6569** @ep37 | **+0.0086** |
| precision | 0.7363 | **0.7536** | **+0.0173** |
| recall | 0.8443 | 0.8366 | −0.0077 |
| F1 | 0.7866 | 0.7929 | +0.0063 |
| **first epoch ≥ 0.64** | 16 | **8** | **2× faster** |

For reference, the AIRS-seeded ResNet run scored 0.6475 — see
[R2](../2026-09-09-does-the-airs-seed-help/), which showed that seed is worthless.

**Predicted +0.03 to +0.08. Measured +0.0086** — right direction, 3–9× too optimistic.

## Interpretation

**The transformer wins, but modestly, and the interesting part is *how*.**

The entire gain comes from **precision** (+0.017) at a small cost in recall (−0.008). That
was the secondary prediction, and it held: a global receptive field should help most in
telling adjacent buildings apart, and Jaipur's party-wall density is exactly where a
convolutional model over-merges. This is weak evidence for the instance-merging concern being
real and architecturally addressable — though pixel IoU cannot confirm that directly
(`MASTER_CONTEXT` §6.2: **boundary IoU does not catch instance merging**; merge/split rate
would).

**The convergence speed is the more practical result.** MiT-B2 reached 0.64 at **epoch 8**
versus ResNet's **16** — half the epochs for the same score. On a shared, heavily contended
box that halves the wall-clock cost of every subsequent experiment, which may matter more
than +0.0086 of IoU.

**Against DAFormer's claim** that architecture beats adaptation algorithm: here architecture
is worth +0.009 while *weak supervision* was worth **+0.51** (0.6483 vs the unadapted seed's
0.1419). At this stage of the project the data dominates the architecture by roughly 60×.
DAFormer's claim is about choices *within* an adaptation pipeline; it is not a licence to
prefer model swaps over label work.

## Decision

- [x] **Adopt MiT-B2 as the default encoder** — it is better and converges twice as fast.
- [ ] Re-run the method comparison with MiT-B2 to check the ranking is architecture-invariant.
- [ ] If instance merging is to be claimed as improved, **implement merge/split rate first**.
      Pixel IoU cannot support that claim.
- [x] Do not treat architecture as the main lever. Label quality is worth far more here.

## Threats to validity

- **Batch size differs (12 vs 16)**, so this is not a pure encoder ablation. Batch size
  affects normalisation statistics and effective learning rate. Stated up front rather than
  buried — a clean re-run would match batch size.
- Same LR and schedule as the ResNet run; transformers usually prefer lower LR with warmup,
  so MiT-B2 is likely *under*-tuned here and the gap may be a floor.
- Single seed. +0.0086 is small enough that seed variance could account for a fair share of
  it; the 2× convergence difference is the more robust observation.
- Footprint labels, not roof ground truth.

## Threats to validity

- **Batch size differs from the baseline (12 vs 16)**, so this is not a pure encoder
  ablation — batch size affects BatchNorm statistics and effective learning rate. A confound
  worth stating rather than hiding.
- Same LR and schedule as the ResNet run; transformers often want lower LR and warmup, so an
  underperformance here may reflect tuning rather than architecture.
- Single seed. Footprint labels, not roof ground truth.
