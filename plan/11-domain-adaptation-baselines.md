# 11. Domain adaptation papers: baseline and what to build on (researched 19 Oct 2026)

**Our task:** models trained on foreign data (AIRS rooftops, New Zealand, 7.5 cm/px; BDAPPV
solar panels, France, 10–20 cm/px) must work on **Jaipur** (26.6 cm/px). Rooftops first,
then solar panels, then energy.

This page lists papers that solve the same problem, which one to use as the **baseline**,
and which one to **build our method on**. Every paper below was looked up on 19 Oct 2026.
Code links were checked where marked.

---

## The short answer

| Role | Paper | Why |
|---|---|---|
| **Codebase to build on (rooftops)** | **HRDA** (Hoyer et al., ECCV 2022), which includes **DAFormer** (CVPR 2022) | The standard domain adaptation method, with public code. Built to handle high-resolution detail and wide context together, which suits our 3.55× resolution gap. |
| **Strongest standard DA baseline** | **MIC** (Hoyer et al., CVPR 2023) | Current best of the same family. Same author, same kind of code. |
| **Scale-gap baseline** | **ResiDualGAN** (Zhao et al., *Remote Sensing* 2023) | Built for exactly our problem: two remote-sensing datasets at **different resolutions**. Raised building IoU from 65.7 to 80.6 on Potsdam (5 cm) → Vaihingen (9 cm). |
| **Precedent for our method** | **Iqbal & Ali, ISPRS J. 2020**, weakly supervised DA for built-up areas | Shows that rough target labels beat purely unsupervised DA. That is our Open Buildings idea. |
| **Solar panel DA** | **Kasmi et al. 2025**, *Environmental Data Science* | Same data (BDAPPV), same imagery-shift problem, code and weights public |
| **Zero-shot comparison** | **SAM 3** (`plan/09`) | The "no adaptation at all, just a big model" row |

**Our method, in one line:** HRDA-style domain adaptation **plus weak Jaipur labels from
Open Buildings** in the training loss, **plus explicit resolution matching**. Our September
runs already show that weak labels are the part that works (IoU 0.65 vs 0.14 without
adaptation).

**The gap we fill:** this search found **no paper** that adapts a building or rooftop model
to an **Indian city** using **Open Buildings or Microsoft footprints as weak labels**, and
then carries on to **solar energy**. That doesn't prove no such paper exists, but no search
here turned one up. It is a reasonable novelty claim for a BTP, so word it carefully.

---

## 1. Rooftop / building domain adaptation

### HRDA and DAFormer: the codebase to build on
- **Papers:** DAFormer (CVPR 2022), HRDA (ECCV 2022), both by Lukas Hoyer et al.
- **What they do:** train on labelled source images and unlabelled target images. A
  "teacher" model makes pseudo-labels on the target and a "student" learns from them.
  HRDA adds a **small high-resolution crop for detail plus a large low-resolution crop for
  context**, fused by learned attention.
- **Why it fits us:** our main gap is resolution (7.5 → 26.6 cm). HRDA is designed around
  mixing resolutions.
- **Code (checked):** [github.com/lhoyer/HRDA](https://github.com/lhoyer/HRDA), which
  includes the DAFormer self-training. **Apache 2.0.**
- **Catch:** built on **mmsegmentation 0.16.0 and mmcv-full 1.3.7**, which are old. Our
  `Dockerfile` (PyTorch 2.1.2) will not run it, so it needs its **own Docker image** on the
  DGX. GPU memory isn't stated, so test on one V100 first. Our binary rooftop task has to
  be added as a custom dataset.
- **How we build on it:** (1) run it as-is, AIRS → Jaipur, as the standard DA row; (2) add
  Open Buildings masks as an extra **weak supervised loss on the Jaipur images**; (3) add
  resolution matching (downsample AIRS to 26.6 cm). That modified version is "ours".

### MIC: strongest standard baseline
- **Paper:** Hoyer et al., *Masked Image Consistency for Context-Enhanced Domain
  Adaptation*, CVPR 2023.
- **What it adds:** hides patches of the target image and asks the model to still predict
  them correctly, so it learns from context.
- **Code (checked):** [github.com/lhoyer/MIC](https://github.com/lhoyer/MIC), with a `seg/`
  folder for segmentation.
- **Use:** one row in the results table ("best published UDA"). Run it only if DGX time
  allows.

### ResiDualGAN: the scale-gap baseline
- **Paper:** Zhao et al., *ResiDualGAN: Resize-Residual DualGAN for Cross-Domain Remote
  Sensing Images Semantic Segmentation*, *Remote Sensing* 15(5):1428, 2023
  ([arXiv 2201.11523](https://arxiv.org/abs/2201.11523)).
- **What it does:** turns source images into target-looking images with a GAN, and adds a
  **resizer** that uses each dataset's ground resolution to fix the scale difference.
- **Its results, building class IoU (Potsdam IRRG 5 cm → Vaihingen 9 cm):**

  | Method | Building IoU |
  |---|---|
  | Source only (no adaptation) | 65.7 |
  | CycleGAN | 65.7 |
  | AdaptSegNet | 70.6 |
  | ResiDualGAN | 75.5 |
  | ResiDualGAN + output-space adaptation | **80.6** |

- **Code (link from the paper):**
  [github.com/miemieyanga/ResiDualGAN-DRDG](https://github.com/miemieyanga/ResiDualGAN-DRDG)
- **Why it matters for us:** their gap is 1.8× (5 → 9 cm). **Ours is 3.55×** (7.5 → 26.6
  cm), so it is a direct test of whether scale-aware translation still helps at a bigger
  gap. Note: our September FDA and histogram-matching runs (also image translation)
  made things worse, so this may not win either. That would be a result too.

### Other building DA papers (related work, not baselines)
- **SPENet**, *Select, Purify, and Exchange* (IEEE TNNLS 2023): uses **several** source
  datasets. 59.1% IoU on Austin + Kitsap → Potsdam. Code:
  [github.com/QZangXDU/SPENet](https://github.com/QZangXDU/SPENet). Relevant if we add
  Inria or WHU as a second source.
- **LoveDA** (NeurIPS 2021 datasets track): urban ↔ rural DA benchmark with 8 UDA methods
  already run on it. Code and data:
  [github.com/Junjue-Wang/LoveDA](https://github.com/Junjue-Wang/LoveDA). Useful to check
  our DA code works on a known benchmark before trusting it on Jaipur.
- **Survey:** *Deep Learning Based Domain Adaptation Methods in Remote Sensing: A
  Comprehensive Survey* ([arXiv 2510.15615](https://arxiv.org/abs/2510.15615), 2025). Good
  for the related-work chapter.

---

## 2. Learning from rough labels (our Open Buildings idea)

- **Iqbal & Ali**, *Weakly Supervised Domain Adaptation for Built-up Region Segmentation in
  Aerial and Satellite Imagery*, ISPRS J. Photogrammetry & Remote Sensing
  ([arXiv 2007.02277](https://arxiv.org/abs/2007.02277)). Uses only **image-level** labels
  on the target and reports IoU gains of 11.6–52% over unsupervised methods. **This is the
  closest precedent for "weak target labels beat pure UDA"**, which is what we measured. No
  code link was found.
- **Kaiser et al.**, *Learning Aerial Image Segmentation from Online Maps*, IEEE TGRS 2017
  ([arXiv 1707.06879](https://arxiv.org/abs/1707.06879)). Training on noisy
  OpenStreetMap labels greatly reduces the need for hand labels. The classic citation for
  "map data as training labels".
- **Robinson et al.**, *Fast building segmentation from satellite imagery and few local
  labels*, CVPR EarthVision Workshop 2022 ([arXiv 2206.05377](https://arxiv.org/abs/2206.05377)).
  Amman, Jordan, 0.5 m imagery: a small U-Net trained on **527 sparse local polygons** beat
  big pretrained building models from elsewhere. Supports labelling a little local data
  (our campus and 30 crops) over trusting foreign models.
- **Sirko et al. 2021**, Open Buildings ([arXiv 2107.12283](https://arxiv.org/abs/2107.12283)).
  The source of our weak labels. Also a baseline: our model must beat the footprints it
  learned from.

---

## 3. Solar panel domain adaptation

The search found **no paper doing unsupervised DA for rooftop PV segmentation across
regions**. Closest work:

- **Kasmi et al. 2025**, *Space-scale exploration of the poor reliability of deep learning
  models: the case of the remote sensing of rooftop photovoltaic systems*, Environmental
  Data Science ([arXiv 2408.07828](https://arxiv.org/abs/2408.07828)). BDAPPV, trained on
  Google, F1 **0.98 → 0.46** on IGN imagery. Blurring augmentation gets 0.74.
  **Code:** [github.com/gabrielkasmi/robust_pv_mapping](https://github.com/gabrielkasmi/robust_pv_mapping),
  weights on [Zenodo](https://zenodo.org/records/12179554). **Use as the solar baseline.**
- **Kasmi et al. 2023**, BDAPPV dataset paper, *Scientific Data*
  ([arXiv 2209.03726](https://arxiv.org/abs/2209.03726)). Our solar training data.
- **García et al. 2024**, *Generalized deep learning model for photovoltaic module
  segmentation from satellite and aerial imagery*, Solar Energy
  ([OSTI](https://www.osti.gov/pages/biblio/2345251)). Trains on mixed resolutions to
  generalise. Method details not checked.

**So for solar, our contribution can be:** apply the rooftop DA recipe (weak labels +
resolution matching) to PV, and test on Indian roofs **with water heaters as hard
negatives**. No paper found here does that.

---

## 4. The results table this gives us

Each row needs the **hand-labelled Jaipur test set** (campus + 30 crops) to be scored.

| # | Method | Type | Status |
|---|---|---|---|
| 1 | AIRS model, no adaptation | lower bound | done, IoU 0.14 |
| 2 | FDA / histogram matching | simple image translation | done, worse than 1 |
| 3 | ResiDualGAN | scale-aware translation | to run |
| 4 | DAFormer / HRDA | standard UDA (self-training) | to run with official code |
| 5 | MIC | best published UDA | optional |
| 6 | SAM 3 | zero-shot foundation model | Priyal, `plan/09` |
| 7 | Open Buildings footprints as the prediction | the label source itself | free |
| 8 | Weak labels only (Open Buildings) | weakly supervised | done, IoU 0.65 |
| 9 | **Ours: HRDA + weak labels + resolution matching** | weakly supervised DA | to build |

Same structure for solar: no adaptation → Kasmi augmentation → ours.

---

## 5. Order of work

1. **Hand labels first** (campus + 30 crops). Without them, no row above has a score.
2. **Set up HRDA in its own Docker image** on the DGX and check it reproduces a published
   number (e.g. on LoveDA) before pointing it at Jaipur.
3. Run rows 3, 4 and 6 on Jaipur.
4. Build row 9 (add the Open Buildings loss to HRDA). This is the main contribution.
5. Fix the BDAPPV no-negatives bug, run Kasmi's code, then apply the same recipe to solar.
6. Energy estimate from the best rooftop and solar masks.
