# 2026-09-09-weak-supervision-jaipur — can Open Buildings labels alone train a Jaipur rooftop model?

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-09 |
| **Commit** | `2fcfef0` on `feat/init-project-setup` |
| **Supersedes** | — |

## Question

Fine-tune the AIRS-trained seed on **weak labels rasterised from Google Open Buildings** over
the Jaipur mosaic, and measure val IoU on held-out tiles.

**Why it matters:** `plan/README.md` calls Open Buildings weak supervision *"★ biggest win"*
and projects **IoU 0.74–0.82**. If that holds, the target domain can be trained on *directly*
and the whole unsupervised-domain-adaptation apparatus — DAFormer/HRDA/MIC, CBST thresholds,
self-training rounds — becomes optional rather than central. That is the largest single fork
in the project, and it is decidable in one GPU-evening with data already on disk.

It also settles the source-domain argument empirically instead of on paper.

**Prediction (written before the run):** val IoU **0.60–0.72** against the weak labels —
below `plan/`'s 0.74–0.82, because that projection assumes SAM2 refinement and shift
correction which this run does not do. Predicted foreground should move from D6's **5.69 %**
up toward the label prior of **23.06 %**; if it does not, the fine-tune is not taking.

Secondary expectation: **precision > recall.** Footprint labels are systematically offset
from roof outlines, so the model should learn a slightly shrunken roof.

## Setup

| | |
|---|---|
| Data | `data/jaipur_weak/` — 7,371 train + 1,701 val crops, 512², stride 512, 4.3 GB |
| Labels | Open Buildings v3, confidence ≥ 0.75 (318,207 polygons), rasterised at full res |
| Split | **whole tiles**: val = `map67_1-1`, `2-2`, `4-3` (D1 priors 35.5 / 27.5 / 12.1 %) |
| Init | `rooftop/checkpoints/unet_resnet34_best.pth` (AIRS, val IoU 0.8784) via `--init_weights` |
| Model | U-Net + ResNet-34, 24.4M params |
| Hardware | 1× V100-SXM2-32GB (**GPU 7**, the only idle device) |
| Runtime | ~2 h expected |

```bash
./run_docker.sh 7 "python -u rooftop/train.py \
    --train_dir data/jaipur_weak/train --val_dir data/jaipur_weak/val \
    --init_weights rooftop/checkpoints/unet_resnet34_best.pth \
    --epochs 40 --batch_size 16 --workers 4 \
    --ckpt_dir experiments/2026-09-09-weak-supervision-jaipur/checkpoints \
    --log_dir  experiments/2026-09-09-weak-supervision-jaipur/outputs"
```

Weak set built by `scripts/make_weak_labels.py` (`data/jaipur_weak/weak_labels_summary.json`);
its mean foreground is **23.06 %**, matching D1's confidence-≥0.75 prior of 23.06 % exactly —
an independent check that the rasterisation is geometrically aligned with the imagery.

## Results

Raw: `outputs/unet_resnet34_40ep_20260909_152425.json`

**Best val IoU 0.6475** (40 epochs, plateaued from epoch ~24; LR decayed to 2e-5).
Head-to-head against every alternative on the same eval set:
[`2026-09-09-method-comparison`](../2026-09-09-method-comparison/) — weak 0.6476,
weak+TTA **0.6534**, unadapted seed 0.1419.

| | prediction | measured |
|---|---|---|
| Val IoU | 0.60 – 0.72 | **0.6475** ✅ in band |
| Predicted fg moves 5.7 % → 23 % | yes | yes — recall 0.84 |
| Precision > recall | **yes** | **NO — recall exceeded precision by ~0.11** ❌ |

## Interpretation

Weak supervision works, and it is not close: **4.6× the unadapted seed** (0.6476 vs
0.1419 at threshold 0.5). One evening, zero manual labels, data already on disk.

**The failed prediction is the informative one.** I expected precision > recall, reasoning
that footprint labels are shrunken versions of roofs. The opposite held all through
training — recall ran ~0.11 above precision. The model predicts *more* area than the
footprints mark, which is the Gap-4 label-semantics mismatch appearing directly in the
metrics: it was initialised on AIRS roof *outlines*, and off-nadir at 26.6 cm those sit
larger than and offset from the ground footprints it is now scored against.

Consequence: **this IoU understates true roof accuracy**, and the precision-recall gap is
a rough proxy for the footprint-vs-roof offset. That argues for the boundary-relaxed loss
(ignore a 4 px band) in `plan/03` §Tier 4, which exists precisely for this.

## Decision

- [ ] If IoU ≥ ~0.70: weak supervision is viable; AIRS and the UDA stack drop off the
      critical path and the project pivots to label refinement (SAM2, shift correction).
- [ ] If IoU ≈ 0.5–0.7: viable but not sufficient alone — combine with adaptation.
- [ ] If IoU < 0.5: weak labels are too noisy at this GSD; UDA stays primary.

## Threats to validity

- **The targets are noisy, and the metric inherits that.** Open Buildings marks **ground
  footprints**, not roof outlines (Gap 4); off-nadir on a 4-storey building they differ by
  ~8 px. This IoU measures agreement with footprints, **not roof-segmentation accuracy**, and
  must never be reported as the latter or compared against AIRS's 0.8784.
- **No clean test set exists.** There are still zero hand-drawn Jaipur labels, so nothing here
  is validated against ground truth. The 400-tile hand-labelling task remains the only route
  to an honest number.
- Val is 3 tiles of 16. Chosen to span D1's density range, but 3 is a small sample and
  per-tile IoU should be reported alongside the aggregate.
- Confidence ≥ 0.75 discards 205k of 523k polygons. Buildings Open Buildings is unsure about
  are plausibly the *hard* ones, so this may flatter the result.
- Single seed, single run. No error bars.

## Reproduce

Needs `data/jaipur/` and `data/open_buildings/`, then `scripts/make_weak_labels.py`
(~40 min CPU) before the command above. The seed checkpoint is gitignored.
