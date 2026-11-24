# Literature notes: what each paper gives this project

Read on 2026-11-20 from the arXiv HTML versions. Where only part of a page could be read word for word,
the claim is marked *(summary, unverified)*. dblp record links for every paper are in
`experiments/2026-11-20-toward-higher-iou/dblp_check.md`.

Our setting, for comparison: ResNet34 U-Net, 26.6 cm/px Google imagery of Jaipur, weak labels from
Open Buildings shifted 7 px S / 3 px W, test IoU 0.706 against hand labels, 93% of the error on roof
boundaries.

---

## Kaiser et al. 2017, Learning Aerial Image Segmentation From Online Maps (IEEE TGRS 55(11))
- **What they did:** trained an FCN (a VGG-16 FCN-4s with one extra early skip connection) on Google
  aerial images, with labels rasterised automatically from OpenStreetMap. Classes: building, road,
  background. Imagery was 9 to 11 cm/px, so 2.4 to 2.9 times finer than ours.
- **Noisy versus clean labels on Potsdam** *(summary, unverified)*: average F1 was 0.779 with OSM labels
  only and 0.874 with 21 clean images. Pre-training on OSM and then fine-tuning on 3 clean images gave
  0.837, recovering most of the gap. Building F1 was 0.796 with OSM only and 0.913 clean.
- **Misalignment:** OSM and Google "align relatively well"; most misalignment comes from the facades of
  tall buildings, generally under 10 px.
- **For us:** this is the strongest support for pre-training on weak labels and then fine-tuning on a
  few hand labels. Our step 6 tried this with 76 crops and got no reliable gain (+0.026 in one fold,
  -0.021 in the other). The difference may be that their few clean labels covered the whole scene at
  5 to 9 cm, while our hand labels come from three annotators who draw roofs differently.

## Girard, Charpiat, Tarabalka 2019, Noisy Supervision for Correcting Misaligned Cadaster Maps (IGARSS)
- **What they did:** a U-Net takes the image plus the misaligned polygon raster and outputs a per-pixel
  displacement field, at 4 scales, handling up to 32 px of misalignment. It is trained on synthetic
  random deformations.
- **Multiple rounds:** train, correct the labels, retrain. Each round corrects the *original* labels,
  not the previous round's output; correcting the previous output did worse. Round 2 cut the error by
  more than 3x versus round 1. Round 3 added nothing, except on noisier data.
- **Metric:** fraction of polygon vertices within τ px of a hand-aligned reference (771 buildings).
  There is no segmentation IoU, so the paper cannot tell us how many IoU points alignment buys.
- **Limitations they state:** most footprints are coarse polygons, so aligning them to complex roofs
  has many equally good answers. Smooth displacement fields fail where adjoining buildings need
  different shifts.
- **For us:** our single global 7/-3 shift is the simplest case of this method. Check whether the
  leftover misalignment varies across tiles (per-tile IQR in `offset_*.json`). If it does, a per-tile or
  per-building shift is the next step. In step 4, always refine from the original OB labels, never from
  the previous round's output. Their coarse-polygon limitation matches our boundary-dominated error.

## Sirko et al. 2021, Continental-Scale Building Detection (the Open Buildings paper, arXiv 2107.12283)
- **Model:** U-Net with a ResNet-50-v2 encoder and residual decoder blocks, on 50 cm imagery in Africa.
  ResNet-101 and 152 gave no gain, which matches our finding that encoder size doesn't matter.
- **Loss:** cross-entropy + 0.5 × Focal Tversky (β = 0.99, γ = 0.25). They kept γ below 1 on purpose,
  because a higher γ over-weights mislabelled "hard" pixels.
- **Edge weighting:** mark instance boundaries, blur them with a Gaussian of σ = 3 px, multiply by 200,
  and use the result as a per-pixel loss weight. This was the single biggest contributor in their
  ablation *(summary, unverified)*.
- **Mixup:** λ = 0.05 on images, with the CE loss mixed across both labels. The abstract reports
  mAP +0.12.
- **Self-training:** soft teacher labels with a KL loss (γ = 0.25), mAP +0.06. Hard pseudo-labels did
  not help. A random black-mask augmentation fixed missed buildings near no-data tile edges.
- **Labels:** each instance is eroded by 1 px so touching buildings separate.
- **Confidence score** *(summary)*: the mean pixel confidence per building. Pick thresholds per region
  to hit a target precision; the paper gives no global threshold.
- **Positional accuracy:** no number is given. Orthorectification can shift footprints by "a few
  metres" *(summary)*, which is consistent with our 1.9 m offset.
- **India or South Asia:** not covered by this paper; Open Buildings was later extended beyond Africa.
- **For us, the cheapest things to try next:**
  1. Edge-weighted loss (σ about 5.6 px at our resolution, to keep the same ground distance).
  2. Focal Tversky with γ < 1 in place of Dice.
  3. Soft-label KL self-training in place of the hard-label self-training arms that failed.

  Note that training on OB is partly distilling Google's own detector.

## Kasmi et al. 2024, Space-scale Exploration of ... Rooftop Photovoltaic Systems (arXiv 2408.07828)
- **dblp lists this as CoRR 2024.** Earlier notes said 2025.
- **What they did:** ResNet-50 PV *classifiers* trained on BDAPPV (France). A model trained on Google
  imagery scored F1 0.98 on Google and 0.46 on IGN *(summary)*. The drop came mostly from acquisition
  conditions, not resolution or geography.
- **Analysis:** wavelet-based attribution (WCAM) shows the CNN relies on features at 2 to 8 px scales,
  which break under acquisition shift.
- **Fix:** Gaussian blur (σ = 2) plus random wavelet-coefficient dropout (20% of coefficients at
  targeted scales), with no target data needed. On IGN, F1 went from 0.44 (ERM) to 0.74 with blur and
  0.58 with blur + wavelet perturbation *(summary)*. The paper's text and table disagree on recall.
- **For us:** relevant to the solar-panel model, which also transfers across imagery, and to robustness
  of the roof model if the imagery source changes. Blur augmentation is cheap to try. All their numbers
  are for classification in France, not segmentation in Jaipur.

---

## What this changes in our plan
1. **Step 4 (label refinement):** refine from the original OB labels each round (Girard's ablation).
   Expect at most 2 useful rounds.
2. **New cheap step (from Sirko):** retrain r34 with the edge-weighted loss and Focal Tversky γ = 0.25.
   This targets boundary error directly, which is 93% of our error. It costs one training run and needs
   no new data.
3. **Step 6:** keep reporting the fine-tune result as negative. Kaiser's larger gain came from cleaner,
   finer labels.
4. **Mentor framing:** Kaiser and Sirko both train on map-derived or model-derived labels at scale.
   Our method sits in that line of work, not in unsupervised domain adaptation.
