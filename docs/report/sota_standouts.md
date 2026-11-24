# Where we stand against the state of the art (data sheet for the presentation)

Collected 2026-11-24. Raw material for the slides. Every number names the repo file it came from.
External numbers were checked by web search on 2026-11-24 (sources in section 12). Anything not
confirmed is marked **unverified**.

**Wording rule for the slides:** say "no Jaipur labels used for training", not "zero labels". The
decision threshold (and the CrossEarth pseudo-label cut-offs) were set on the 76 hand-labelled calib
crops, and CrossEarth's source model was trained on labelled WHU data.

**Test protocol for our headline numbers:** 152 hand-labelled Jaipur crops (512 x 512, 26.6 cm/px), split
by tile into calib/test (76/76). Threshold chosen on calib, IoU reported on test, crop bootstrap 95% CI.

---

## 1. Headline numbers (ours)

| What | Test IoU (95% CI) | Training labels | Source |
|---|---|---|---|
| **Self-trained CrossEarth + 8-way TTA** | **0.717** (0.689-0.745) | none from Jaipur, no Open Buildings | `experiments/2026-11-22-crossearth-hand-scoring/README.md` |
| Self-trained CrossEarth, single pass | 0.711 (0.685-0.738) | same | same |
| Ensemble (shift+SAM model + self-trained student) + TTA | **0.712** (+0.015 over plain, CI +0.009 to +0.022), reproduces with a 2nd seed | Open Buildings, shifted | `experiments/2026-11-20-toward-higher-iou/README.md` |
| ResNet34 student on CrossEarth soft maps + TTA, 3 seeds | 0.706 +- 0.005 | none from Jaipur | crossearth README |
| ResNet34 U-Net, OB labels shifted 7 px S / 3 px W, + TTA | 0.706 | Open Buildings, shifted | toward-higher-iou README |
| Same, single pass, seeds 42/43/44 | 0.697 / 0.693 / 0.693 | Open Buildings, shifted | toward-higher-iou README |
| IoU ignoring a 1 / 2 / 3 px band round hand roof edges | 0.736 / 0.759 / 0.776 | | toward-higher-iou README |

Starting point for the "how far we came" slide: September default model 0.564, Open Buildings itself 0.504.

---

## 2. Head-to-head on our own hand-labelled test (strongest claims)

### 2a. Same protocol (test half, calib-chosen threshold)

| Method | Its score | Ours | Gap |
|---|---|---|---|
| **CrossEarth zero-shot**, released WHU-A2S weights + TTA (Gong et al., arXiv 2410.22629; the repo README says TPAMI 2026, **venue unverified**) | 0.677 (0.650-0.704) | **0.717** (self-trained CrossEarth) | **+0.040** with TTA, **+0.064** single pass (0.647 to 0.711) |
| CrossEarth WHU-S2A + TTA | 0.664 | 0.717 | +0.053 |
| Open Buildings masks (Google's product), test half | 0.499 | 0.706 | **+0.21** |

### 2b. Our runs of domain-adaptation baselines (all 152 crops, different protocol)

These come from `diagnostics/hand_labels/models_r12b2_152.md`. That file scores all 152 crops, not the
test half. To be fair to the baselines, the table quotes their **best threshold chosen with hindsight**,
which flatters them.

| Baseline (our implementation, adapting from the AIRS source model) | Fixed 0.5 | Best threshold, hindsight | Ours |
|---|---|---|---|
| DUDA (Kang et al., WACV 2026) | 0.274 | 0.289 | 0.706-0.717 |
| Best annotation-free arm (R) | 0.317 | 0.379 | |
| Arms A, B, C, C2, D, D2, G, M | 0.04-0.13 | 0.05-0.16 | |
| AIRS source model, no adaptation | 0.154 | 0.174 | |

Say "our run of DUDA", not "we beat the 2026 UDA SOTA". These are our implementations from a small
AIRS source, not the authors' numbers.

Talking point: "On the same hand-labelled Jaipur roofs, self-training lifts the released CrossEarth
foundation model from 0.677 to 0.717, and beats Google's own Open Buildings masks by 0.21 IoU, without
training on any Jaipur label."

---

## 3. Close to clean-label results (different datasets: scale only, never "we beat")

| Reference | Score | Its training labels | Ours (0.717) as % |
|---|---|---|---|
| Inria aerial (0.3 m, mixed cities, closest to our setting): UANet, IEEE 2024, as reported in UAGLNet's table | 0.831 IoU | fully hand-labelled | **86%** |
| Inria: UAGLNet (arXiv 2512.12941), per `docs/research_2023_2026.md` | 0.837 | fully labelled | 86% |
| WHU building: DeepSwinLite 2025 | ~0.92 | fully labelled | 78% (WHU is the easiest common benchmark) |
| **Zero-shot foundation-model pipeline: Grounded-SAM on WHU** (Akbulut et al., ISPRS Archives 2025) | 0.71 IoU (F1 0.83) | none | about equal, on harder off-nadir imagery of an Indian city |
| SAM+CLIP zero-shot on WHU (same paper) | 0.49 | none | |
| **Rooftop segmentation, rural India** (Rangdal et al., ISPRS Annals 2025, YOLOv11 + super-resolution) | 0.6115 **mIoU** | supervised, hand labels (QGIS) | ours is higher, but theirs is mIoU (likely averaged with background), rural villages, different imagery; **check the PDF** |

Do not put Google Satellite Sunroof (~56% IoU, arXiv 2408.14400) in this table: it measures roof
*segments* (individual roof planes) against segment annotations, a different and harder task.

---

## 4. Why Indian data is a different problem, and how little work there is

What the searches on 2026-11-24 found. This is **search-bounded**: "we found no X", not "X does not exist".

- **No public hand-labelled urban Indian roof benchmark found.** The standard building benchmarks are
  Inria (US and Austrian cities), WHU (Christchurch, NZ), Massachusetts, SpaceNet (Las Vegas, Paris,
  Shanghai, Khartoum). None is Indian.
- **Indian studies found, and their limits:**
  - Rangdal et al. 2025 (C-DAC Pune): rural India, supervised, YOLOv11. The authors say urban-trained models
    do not suit rural roofs and that annotated satellite data is lacking.
  - Singla & Kalase 2024 (arXiv 2411.04610): solar potential of Indian cities, using a DSM.
  - A Chandigarh study (arXiv 2505.05321) had to combine several *foreign* open datasets to get training data.
  - A Mumbai slum study built its own ~15,000-building dataset.
  - A Cartosat-1 stereo study of Indian cities (UN World Geospatial Congress) reports no city-level IoU.
  - Kumar 2018 (arXiv 1812.11606): early rooftop solar for India; notes poor image quality.
- **Open Buildings has no published India accuracy.** Google gives confidence thresholds at estimated
  precision levels, but we found no India-specific precision or recall. An ISPRS 2025 study looked at
  seven Indian heritage sites only. **Our 152-crop test (OB = 0.50 IoU against hand roofs) appears to
  be the first city-scale hand-labelled check of Open Buildings roof accuracy in an Indian city**
  (search-bounded; worth one more search before claiming it).
- **No paper found that trains a roof segmenter on Open Buildings weak labels for an Indian city.** The
  closest are ETHOS.PASSION (Germany, open footprints in a PV workflow) and an India solar-farm dataset
  built from weak labels (Sentinel-2, utility-scale, not rooftops).

---

## 5. Why our absolute IoU is lower than Western benchmarks

Lead with what **we measured**; use published context only as support.

**Measured on our data:**
1. **The error is at the boundaries.** 93% of wrong pixels are on edges or partly covered roofs. Ignoring a
   3 px (80 cm) band round each roof lifts IoU from 0.706 to 0.776.
2. **The imagery is not orthorectified.** Google tiles are off-nadir, so roofs are displaced from their
   footprints in proportion to building height. Open Buildings sits 6-8 px north and 2-4 px east of hand
   roofs (`experiments/2026-11-19-hand-label-model-scoring/README.md`).
3. **The labels describe footprints, not roofs.** Even perfectly aligned, OB footprints reach only 0.58 IoU
   against hand-drawn roofs (`docs/research_2023_2026.md` section 6), so a model trained on them inherits that shape error.
4. **The test set itself has an annotator style.** Hand/OB area ratios differ by annotator (1.02 vs
   1.24-1.37), so part of the remaining gap is label disagreement, not model error.
5. **Benchmarks are cleaner by construction.** Inria and WHU are orthorectified aerial photos with labels
   drawn on the same image. Our test labels and training labels come from different sources.

**Published context (supporting, not proof about India):**
- SpaceNet Round 2, same winning method in four cities: F1 **0.885 Las Vegas, 0.745 Paris, 0.597
  Shanghai, 0.544 Khartoum** (DeepGlobe 2018 paper, arXiv 1805.06561). The SpaceNet paper puts
  Khartoum's difficulty down to varied building sizes and low contrast. That means a **0.34 drop** from a
  US city to a Global South city with the same model.
- A 2020 paper on developing-world cities (arXiv 2011.12847) argues current methods work better on
  Western cities.
- We found **no study measuring India-specific difficulty**, so do not claim "Indian imagery is harder"
  as a published fact. Say "Global South cities score lower in the benchmarks that have them (SpaceNet),
  and our own error analysis shows why for Jaipur".

---

## 6. Why classical UDA failed, and what we did instead

**What failed (our runs, `models_r12b2_152.md`, `experiments/2026-10-30-duda`, `2026-09-09-method-comparison`):**
- Ten annotation-free arms and DUDA, all adapting from the AIRS model: 0.04-0.33 hand IoU.
- Photometric fixes (FDA, histogram matching) did worse than nothing. Much of the source model's gap is
  calibration: it predicts 5.69% roof on Jaipur against a ~28% prior.
- AIRS buildings are ~23x larger in pixels and the city is very different (`experiments_digest.md`).

**What worked, and the honest framing:**
- **Weak supervision:** train on in-domain but noisy labels (Open Buildings, globally shifted). 0.697-0.712.
- **Foundation-model self-training:** start from a remote-sensing foundation model (CrossEarth), then
  self-train it on unlabelled Jaipur tiles. 0.717.
- **Be careful with "UDA didn't work".** The 0.717 run is itself transductive self-training from a
  labelled source (WHU), i.e. a form of UDA. The supportable story is narrower: *classical UDA from a
  small Western source model failed (0.04-0.33); a strong foundation-model source plus self-training,
  or in-domain weak labels, both reach about 0.71.*
- This changes framing A in `docs/report/index.html` ("open footprints beat domain adaptation"). The
  label-free CrossEarth route now ties the Open Buildings route, so the claim should become "the
  choice of source matters more than the adaptation method".

---

## 7. Is our IoU "comparable to UDA on foreign data"?

**What we could not find:** a clean published table of UDA building IoU between Western datasets. The
leads (DATA, arXiv 2004.11819; Self-Mutating Network, ICCV 2021; WCMNet, Remote Sensing 2025) did not
show readable per-pair IoU in the search results. **Do not quote UDA numbers from memory; read those
tables first.** One partial anchor: a remote-sensing UDA paper (CPCA, arXiv 2403.03704) reports a
source-only mIoU of 41.22 and a +9.5 gain from adaptation, for land cover, not buildings alone.

**Arguments that do hold up:**
1. **Head-to-head with released weights.** CrossEarth's own checkpoint on our test (0.677) vs ours (0.717).
   This is the cleanest SOTA comparison we have.
2. **Source-only to adapted gain.** Standard UDA papers report how much of the gap adaptation closes.
   Ours: AIRS source-only 0.154 (0.174 best threshold) to 0.717; CrossEarth zero-shot 0.677 to 0.717.
3. **A labelled-target reference ("oracle").** UDA papers treat a model trained on target labels as
   the upper bound. Ours: fine-tuning our best model on 76 hand crops (2-fold) gave no reliable gain:
   fold A 0.697 to 0.675, fold B 0.656 to 0.682, average about +0.003. So adding a small set of Jaipur
   hand labels does not beat what we get without them. This is weak evidence (76 crops); a proper
   oracle would need a few hundred labelled crops.
4. **The test set's own ceiling.** Edge-band analysis (0.776 at 3 px) and annotator style show how much
   of the gap no model can close on these labels.
5. **Scale comparisons (section 3).** Only as context, with the dataset differences stated.

---

## 8. Novelty: what is defensible and what to consider

Reuse the skeptical ratings from `docs/report/index.html` ("What is publishable") and
`docs/report/lit_review.md`. Updated view:

| Claim | Rating | What it still needs |
|---|---|---|
| Hand-labelled evaluation of open footprints and models on an Indian city at 26.6 cm/px | **plausibly new** (no Indian urban roof benchmark found) | One more targeted search; release the crops if the imagery licence allows |
| Label-free self-training of a remote-sensing foundation model (CrossEarth) on an Indian city: 0.677 to 0.717 | **plausibly new as an application**, not as a method (self-training is standard) | Precedent search for CrossEarth self-training; a 2nd seed (single run now); a second city |
| Open footprints vs domain adaptation for Global South cities | publishable with fixes | The missing control: a UDA method given Open Buildings as its source labels. Reframe as "source choice matters more than adaptation method" (section 6) |
| Label-quality gate (rank label sets against held-out roofs before training) | plausibly new, weak evidence | 10+ label variants (n = 6 now, Spearman 0.77 not significant) |
| Annotator style changes model ranking | plausibly new | Same 30-50 crops drawn by 2-3 annotators |
| Global 7/3 px offset, per-building alignment | incremental / not new | Already done better by OMAF (CVPR 2026) and Align and Segment (2026); keep only as a negative result |
| City-scale GWp / TWh with an error budget | application | Measured k_usable; comparison with Google Solar API figures |

**What reviewers will check (consider before claiming novelty):**
1. Is there prior work on the *same city or country with the same data*? (Searched: none found for
   Open Buildings weak labels on Indian urban roofs.)
2. Is the method new, or only the application? Most of ours is a new **setting and evaluation**, not a new
   algorithm. Pitch it as an empirical / dataset / application paper.
3. Are the baselines fair? DUDA from AIRS is weak by construction; add the OB-source UDA control.
4. Is the test set big and independent enough? 152 crops, one city, one labelling team. Add a second
   city or at least a second tile set; double-annotate a subset.
5. Are the gains beyond noise? Report seed variance (we have it for most rows; the 0.717 run is 1 seed).
6. **Blocker:** the imagery is Google Maps tiles, whose terms forbid ML training, and the repo is public.
   Resolve this (or rerun on licensed imagery such as ISRO/NRSC or Maxar) before any submission.

**Likely venues** (from the report): CVPR EarthVision workshop, IGARSS, NeurIPS Climate Change AI workshop,
ISPRS Geospatial Week; extended version to IEEE JSTARS or MDPI Remote Sensing.

---

## 9. Against the two 2026 papers that do our exact method

| Paper | Their setup and number | Ours | How to say it |
|---|---|---|---|
| **OMAF**, Xu et al., CVPR 2026 | Google Earth Pro 0.5 m + Microsoft footprints (Turkey). Self-alignment adds 9.22 mIoU in ablation; up to +40.6% mIoU over baselines (poster page). Full paper not read (PDF blocked) | Global shift: +0.038 IoU (CI +0.031 to +0.045), 3 seeds; +0.077 for MiT-B2 | "Same idea as a CVPR 2026 paper, measured against hand-drawn roofs rather than footprints" |
| **Align and Segment**, Guthula et al., arXiv 2607.10841 | Real Google Earth (ReBO): 0.54 to 0.62; synthetic 50 px shift: 0.46 to 0.78. **Unverified** (machine summary) | 0.706 on real Google imagery of Jaipur | Only after reading the PDF; different data |

Our measured finding to pair with these: per-building alignment adds only 0.606 to 0.610 label IoU after
the global shift; even perfect alignment caps at about +0.05. After a global shift, the bottleneck is
shape, not position.

---

## 10. Efficiency and design findings

| Finding | Numbers | Source |
|---|---|---|
| Small encoder is enough | ResNet34 0.697 vs MiT-B2 0.700; DINOv3 ConvNeXt-T 0.011 worse; MiT-B5 0.571 vs B2 0.564 (unshifted) | toward-higher-iou, hand-label-model-scoring |
| Self-trained student = TTA teacher in one pass | st_soft_fuse08 0.706 single pass = teacher + TTA 0.706: **8x cheaper** city run | toward-higher-iou |
| Self-training beats its teacher | student 0.706 +- 0.005 vs CrossEarth teacher 0.677 | crossearth README |
| Seed stability | shift gain CI excludes zero for every seed (+0.038 / +0.035 / +0.034) | toward-higher-iou |
| Area bias fixed | Sept default predicted 0.83x true roof area; best models 0.95-1.07x | hand-label-model-scoring, crossearth README |
| Model output beats edited labels | model's TTA prediction as a label: 0.723 vs best refined footprint 0.650 | toward-higher-iou |

## 11. Energy result (application, not a SOTA claim)

- Jaipur: **4.59 GWp, 7.62 TWh/yr, +-15%**, 175 km2, with an uncertainty budget: segmentation is 2% of the
  error, usable-roof fraction k_usable is 68%.
- Likely 15-20% low (the Sept model under-predicted area at 0.83x); re-run with the 0.70+ model pending.
- With k_usable 0.30 (the one verified Indian source, Bhopal) instead of 0.60: 2.9-3.3 GWp.
- vs Satellite Sunroof and Singla & Kalase: ours is open-label with a hand-labelled test and an error
  budget. No numeric comparison has been run.

## 12. Caveats to say out loud

1. Only section 2a is like-for-like. 2b uses a different protocol; section 3 uses different datasets.
2. CrossEarth's source was trained on labelled WHU, so 0.717 is domain generalisation plus self-training
   from a labelled source, not "no labels anywhere".
3. Single seed for the 0.717 run; its +0.011 over the ResNet student is inside the CI.
4. The student trained on unlabelled tiles of the same city (transductive); hand crops excluded by name.
5. Imagery licence (section 8, item 6).

## 13. Sources checked 2026-11-24

- UANet Inria 83.08 (UAGLNet comparison table): https://arxiv.org/pdf/2512.12941 · UANet: https://arxiv.org/abs/2307.12309
- Grounded-SAM 0.71 / SAM+CLIP 0.49 on WHU: https://isprs-archives.copernicus.org/articles/XLVIII-M-6-2025/23/2025/
- CrossEarth (venue unverified): https://arxiv.org/abs/2410.22629
- Align and Segment (IoU unverified): https://arxiv.org/abs/2607.10841 · OMAF: https://cvpr.thecvf.com/virtual/2026/poster/38531
- DUDA, WACV 2026 (Crossref-verified in `lit_review.md`): https://doi.org/10.1109/WACV61042.2026.00784
- SpaceNet per-city F1 (DeepGlobe 2018): https://arxiv.org/pdf/1805.06561 · SpaceNet: https://arxiv.org/pdf/1807.01232
- Developing-world cities: https://arxiv.org/pdf/2011.12847
- Rangdal et al., rural India rooftops: https://isprs-annals.copernicus.org/articles/X-5-W2-2025/537/2025/
- Chandigarh multiscale building segmentation: https://arxiv.org/pdf/2505.05321
- Open Buildings: https://sites.research.google/open-buildings/ · Indian heritage-site footprint check: https://isprs-annals.copernicus.org/articles/X-M-2-2025/73/2025/
- ETHOS.PASSION: https://www.sciencedirect.com/science/article/pii/S0038092X23007284 · India solar-farm dataset: https://www.nature.com/articles/s41597-022-01499-9
- Satellite Sunroof: https://arxiv.org/abs/2408.14400 · Kumar 2018: https://arxiv.org/abs/1812.11606
- UDA leads not yet read: https://arxiv.org/abs/2004.11819 · https://openaccess.thecvf.com/content/ICCV2021/html/Lee_Self-Mutating_Network_for_Domain_Adaptive_Segmentation_in_Aerial_Images_ICCV_2021_paper.html · https://doi.org/10.3390/rs18132265 · CPCA: https://arxiv.org/pdf/2403.03704
