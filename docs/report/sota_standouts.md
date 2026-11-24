# Where we stand against the state of the art (data sheet for the presentation)

Collected 2026-11-24. Raw numbers for building the slides later. Every number names the repo file it
came from. External numbers were spot-checked by web search on 2026-11-24 (sources in section 6).

**Test protocol for all "our" numbers:** 152 hand-labelled Jaipur crops (512 x 512, 26.6 cm/px), split by
tile into calib/test (76/76). The threshold is chosen on calib and IoU is reported on test, with a crop
bootstrap 95% CI. Zero hand labels were used for training in any headline number.

---

## 1. Headline numbers (ours)

| What | Test IoU (95% CI) | Labels used for training | Source |
|---|---|---|---|
| **Self-trained CrossEarth + 8-way TTA** (label-free) | **0.717** (0.689-0.745) | none at all (no OB, no hand) | `experiments/2026-11-22-crossearth-hand-scoring/README.md` |
| Self-trained CrossEarth, single pass | 0.711 (0.685-0.738) | none | same |
| Ensemble (shift+SAM model + self-trained student) + TTA | **0.712** (+0.015 over plain, CI +0.009 to +0.022); reproduces with a 2nd seed | Open Buildings (shifted) | `experiments/2026-11-20-toward-higher-iou/README.md` |
| ResNet34 student on CrossEarth soft maps + TTA, 3 seeds | 0.706 +- 0.005 | none | crossearth README |
| ResNet34 U-Net, OB labels shifted 7 px S / 3 px W, + TTA | 0.706 | Open Buildings (shifted) | toward-higher-iou README |
| Same, single pass, seeds 42/43/44 | 0.697 / 0.693 / 0.693 | Open Buildings (shifted) | toward-higher-iou README |
| IoU ignoring a 1 / 2 / 3 px band round roof edges | 0.736 / 0.759 / 0.776 | | toward-higher-iou README |

Starting point for the "how far we came" slide: September default model 0.564, Open Buildings itself 0.504.

---

## 2. Where we beat published methods on the same test set (strongest claims)

These are like-for-like: same 152 crops, same split, same threshold rule.

| Published method | Its score on our test | Ours | Gap | Note |
|---|---|---|---|---|
| **CrossEarth zero-shot** (Gong et al., RS domain-generalisation foundation model, arXiv 2410.22629; repo README lists TPAMI 2026, venue not confirmed by search) WHU-A2S checkpoint + TTA | 0.677 (0.650-0.704) | **0.717** (our self-trained CrossEarth) | **+0.040** TTA / **+0.064** single pass (0.647 to 0.711) | Our label-free self-training improves the SOTA foundation model itself |
| CrossEarth WHU-S2A + TTA | 0.664 | 0.717 | +0.053 | |
| **DUDA** (Kang et al., WACV 2026, doi 10.1109/WACV61042.2026.00784), latest UDA for segmentation | 0.275 | 0.706-0.717 | **~2.6x** | crop set of 136 for DUDA (`2026-11-19-hand-label-model-scoring`) |
| Best of our 10 annotation-free UDA arms (arm R) | 0.317 (0.262 on test half) | 0.717 | 2.3x | arms A-D, G, M: 0.04-0.13 |
| AIRS source model, no adaptation | 0.154 | 0.717 | 4.7x | |
| **Google Open Buildings** (Sirko et al. 2021), the product itself, scored as a mask | 0.504 (0.482-0.526) | 0.706 | **+0.20** | We start from its labels and beat it by 0.20 |

Talking point: "On the same hand-labelled Jaipur roofs, our label-free pipeline beats the 2026 UDA SOTA
(DUDA) by 2.6x and improves the remote-sensing foundation model CrossEarth by +0.04 IoU, with no Jaipur
labels of any kind."

---

## 3. Where we are close to clean-label SOTA (different datasets, so label it "for scale")

Not like-for-like (different city, resolution and labels). Show as context, never as "we beat X".

| Benchmark SOTA | Score | Training labels | Ours (0 labels) | Ours as % of SOTA |
|---|---|---|---|---|
| Inria aerial (0.3 m, mixed cities; closest to our setting): UANet, IEEE 2024 | 0.831 IoU | fully hand-labelled | 0.717 | **86%** |
| Inria: UAGLNet (arXiv 2512.12941) | 0.837 (per `docs/research_2023_2026.md`) | fully labelled | 0.717 | 86% |
| WHU building: DeepSwinLite 2025 | ~0.92 | fully labelled | 0.717 | 78% (WHU is an easier dataset) |
| **Zero-shot foundation-model pipeline: Grounded-SAM on WHU** (Akbulut et al., ISPRS Archives 2025) | 0.71 IoU (F1 0.83) | none | **0.717** | **matches / slightly above**, on a harder off-nadir Indian city at finer resolution |
| SAM+CLIP zero-shot on WHU (same paper) | 0.49 | none | 0.717 | 1.46x |
| With the 2 px edge band ignored | | | 0.759 | 91% of Inria SOTA |

Talking point: "With zero labels we reach 86% of the best fully supervised Inria result, and we match
the best zero-shot foundation-model pipeline (Grounded-SAM, 0.71 on WHU)."

---

## 4. Against the two 2026 papers that do our exact method (misaligned footprints on Google Earth)

| Paper | Their setup and number | Ours | How to say it |
|---|---|---|---|
| **OMAF**, Xu et al., CVPR 2026 | Google Earth Pro 0.5 m + Microsoft footprints (Turkey). Self-alignment adds 9.22 mIoU in ablation; up to +40.6% mIoU over baselines (poster page). Full paper not read (PDF blocked) | Global shift alone: +0.038 IoU (CI +0.031 to +0.045), reproduced over 3 seeds; shift gives MiT-B2 +0.077 | "Same idea as a CVPR 2026 paper, independently arrived at, and validated against hand-drawn roofs, not footprints" |
| **Align and Segment**, Guthula et al., arXiv 2607.10841 (Jul 2026) | Real Google Earth (ReBO): 0.54 to 0.62 IoU. Synthetic 50 px shift, Las Vegas: 0.46 to 0.78. Both from a machine summary of the paper, **check against the PDF before showing** | 0.706 on real Google imagery of Jaipur (different data) | Only if verified; different dataset, so "for scale" |

Our own measured finding to pair with these: per-building alignment (their mechanism) adds only
0.606 to 0.610 label IoU after the global shift; even perfect per-building alignment caps at ~+0.05
label IoU. "After a global shift, position is no longer the bottleneck, shape is."
(Source: `docs/research_2023_2026.md` section 6, `experiments/2026-11-20-toward-higher-iou/per_building_offset.json`.)

---

## 5. Efficiency and design findings that stand out

| Finding | Numbers | Source |
|---|---|---|
| Small encoder is enough | ResNet34 0.697 vs MiT-B2 0.700 vs DINOv3 ConvNeXt-T -0.011 vs baseline; MiT-B5 0.571 vs B2 0.564 (unshifted) | toward-higher-iou, hand-label-model-scoring |
| Self-trained student = TTA teacher in one pass | st_soft_fuse08 0.706 single pass vs teacher + TTA 0.706: **8x cheaper inference** for the city run | toward-higher-iou |
| Self-training beats its teacher | ResNet34 student 0.706 +- 0.005 vs CrossEarth teacher 0.677 (~5 sd of seed noise) | crossearth README |
| Seed stability | shift gain CI excludes zero for every seed (+0.038 / +0.035 / +0.034); seed spread 0.004 | toward-higher-iou |
| Area bias fixed | default model predicted 0.83x true roof area; best models 0.95-1.07x | hand-label-model-scoring, crossearth README |
| Hand fine-tuning does not help | 76 hand crops, 2-fold: +0.026 / -0.021 (not reliable), so zero labels is the right call | toward-higher-iou |
| Labels trained on > human-edited labels | the model's own TTA prediction scores 0.723 as a label vs best refined footprint 0.650 | toward-higher-iou |

## 6. Energy result (application, not a SOTA beat)

- Jaipur: **4.59 GWp, 7.62 TWh/yr, +-15%**, 175 km2, from open labels only, with a propagated uncertainty
  budget: segmentation is only 2% of the error; usable-roof fraction k_usable is 68%.
- Likely 15-20% low because the September default under-predicted area (0.83x); re-run with the 0.70+ model pending.
- If k_usable is 0.30 (the one verified Indian source, Bhopal) instead of the assumed 0.60: 2.9-3.3 GWp.
- Differentiator vs Google Satellite Sunroof (arXiv 2408.14400) and Singla & Kalase (arXiv 2411.04610,
  Indian cities with DSM): open-label pipeline, held-out hand-drawn roof test set, and an explicit
  error budget. No direct numeric comparison with them has been run.
- Source: `docs/report/index.html` energy section, memory of the Sep 2026 run.

---

## 7. Caveats to say out loud (examiners will ask)

1. Section 3 compares different datasets. Only section 2 is like-for-like.
2. Open Buildings is output of Google's detector trained on large labelled data, so "OB-trained beats UDA"
   is not a fair UDA comparison. The CrossEarth 0.717 avoids this: it uses no OB labels. CrossEarth's
   source model was trained on labelled WHU, so it is domain generalisation from a labelled source.
3. Single seed for the 0.717 run; its +0.011 over the ResNet student is inside the CI.
4. Student trained on unlabelled images of the same city (transductive); hand crops excluded by name only.
5. Imagery is Google Maps tiles (ToS forbids ML training); resolve before any submission.

## 8. Sources checked 2026-11-24

- UANet Inria 83.08 IoU (as reported in UAGLNet's comparison table): https://arxiv.org/pdf/2512.12941 · UANet: https://arxiv.org/abs/2307.12309
- Grounded-SAM 0.71 / SAM+CLIP 0.49 IoU on WHU: https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/23/2025/
- CrossEarth (arXiv, venue not confirmed): https://arxiv.org/abs/2410.22629
- Align and Segment: https://arxiv.org/abs/2607.10841 (IoU numbers not confirmed by search)
- OMAF poster: https://cvpr.thecvf.com/virtual/2026/poster/38531
- DUDA (WACV 2026, Crossref-verified in `docs/report/lit_review.md`): https://doi.org/10.1109/WACV61042.2026.00784
