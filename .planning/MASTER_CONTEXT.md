# BTP MASTER CONTEXT — reconciled against disk

**Owner:** Sirjan Singh (23ucs715, LNMIIT) · **Repo:** `github.com/SirjanSingh/btp`
**Source doc compiled:** 08 Sep 2026 · **Reconciled against `lnmdgx1` disk:** 08 Sep 2026

The authoritative master context was compiled off-machine. This file preserves its
substance and **corrects the places where it disagrees with what is actually on this
server**. Corrections are marked **[CORRECTED]**. Where the source doc is right and the
repo is wrong, that becomes a task, marked **[REPO DEBT]**.

Confidence tiers from the source doc are kept: **[MEASURED]** trust it ·
**[PLANNED]** change only deliberately · **[ASSUMED]** every one is a task.

> **Prime directive, unchanged:** the goal is solar **energy** estimation (kWh/yr), not
> rooftop segmentation. Segmentation is instrumental. When a design choice trades
> segmentation IoU against energy-estimate accuracy, **energy wins**.

---

## 0. Reconciliation log — read this first

| # | Source doc claims | Disk reality | Severity |
|---|---|---|---|
| **C1** | `train_solar.py` black-mask fallback "keeps ~12,232 negative crops (correct, non-obvious, **keep it**)" | The fallback **exists and is correct** (`train_solar.py:162-164`) but **never fires**. `prep_bdappv.py:85` prints `no mask for X, skipping` and drops negatives *upstream*. Sampled 400 train masks: **0 all-zero**. Images 16,763 = masks 16,763, 1:1, **zero negatives in the crop set.** | **CRITICAL** |
| **C2** | `plan/` with `00-README … 08-sources.md` (8 docs), cited ~20× as `plan/02`, `plan/04`, `plan/07`, `plan/08-sources.md` | **No `plan/` directory.** Disk has `.planning/` with `PROJECT.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, `config.json`, `research/{ARCHITECTURE,FEATURES,PITFALLS,STACK,SUMMARY}.md`. **None of the cited section numbers resolve.** | **HIGH** |
| **C3** | `daraset/*.tif` — 16 Jaipur tiles, ~8.2 GB, 2.48 gigapixels | ~~Not on this machine.~~ **RESOLVED 2026-09-08** — all 16 `map67_*.tif` restored to `data/jaipur/` (8.0 GB), EPSG:3857, GSD measured 0.26618 m/px by D6. | ~~HIGH~~ resolved |
| **C4** | Phases A–F | `.planning/ROADMAP.md` has **Phases 1–5**, different decomposition. `STATE.md` says "Phase 1 of 5, Progress 0%" while checkpoints through epoch 050 exist. | MEDIUM |
| **C5** | AIRS crops available for training | **STILL OPEN, partially restored.** `data/airs/image/` has **50 of 857** source `.tif`s (871 MB); `data/airs/label/` has 7 files of which 4 are `_vis` previews, so **3 real masks**. Only **3 image/label pairs actually match** (`christchurch_15`, `_48`, `_77`). `data/airs/train/`, `data/airs_crops/` and `rooftop/dataset_crops/` are still **empty**. The Drive fetch was interrupted mid-run on 2026-09-08. | **HIGH** |
| **C6** | *(new 2026-09-09)* Open Buildings weak labels unavailable | **Present** — `data/open_buildings/jaipur_open_buildings.csv`, **523,283 buildings** over the AOI, with `confidence` and `area_in_meters`. Enables D1 and the weak-label set (§12.5). | — |
| ✅ | Hann-window blended sliding inference | **Confirmed** — `rooftop/infer.py:134`, `solar_panel/infer_solar.py:124` | — |
| ✅ | `--simulate_low_res` exists | **Confirmed** — `rooftop/train.py:238`, flag at `:596` | — |
| ✅ | BDAPPV `google_`/`ign_` provenance in filenames | **Confirmed** — sandbox needs only a filename filter | — |

### C1 in full — the one that changes a result

The source doc lists the negative-crop handling under "**built and working**… correct,
non-obvious, keep it." It is half right in a way that inverts its meaning:

- `train_solar.py` **does** implement the black-mask fallback, correctly, with a count
  report at construction (`:142-146`).
- `prep_bdappv.py` **never gives it the chance** — it skips mask-less images at prep time.
- Result on disk: **every single training crop contains a panel.**

Why this matters more than it looks. §3.6 of the source doc argues that in India the PV
base rate is very low, that solar water heaters are near-identical from above and *more
common than PV*, and therefore **"precision matters far more than recall."** R12 in the
run table is entirely about hard-negative mining.

**A training set with zero negatives is the precise opposite of that setup.** The model
is never shown a roof without a panel, so it cannot learn "no panel here" — it can only
learn "where is the panel in this crop, which certainly has one." Then it meets Jaipur,
where the overwhelming majority of roofs have no PV and many have a water tank or SWH.

This also explains the shape of the Kasmi failure mode the doc wants to reproduce
(Recall 0.08 / Precision 1.0): degenerate operating points are exactly what you get when
base rate is mishandled across domains.

**Fix before any Stage 2 number is quoted:** re-run `prep_bdappv.py` retaining mask-less
images so the fallback fires, then verify the negative fraction is non-zero and report it.
Every solar metric produced before this is measuring the wrong task.

---

## 1. Project facts

### 1.1 Source domains [MEASURED]

| | Rooftop (Stage 1) | Solar PV (Stage 2) |
|---|---|---|
| Dataset | AIRS (WHU Building), Christchurch NZ | BDAPPV, France |
| GSD | **7.5 cm/px** | `google/` 10 cm · `ign/` 20 cm |
| Labels | Roof **outlines** (not ground footprints) | PV masks; negatives have no mask file |
| Arch | SMP U-Net, ResNet-34, ImageNet-pretrained | same |
| Loss | `0.5 × SoftBCEWithLogits + 0.5 × Dice` | same |
| Crop | 512², 10% overlap | ~400² |
| Result | IoU **0.8664** full test (0.9016 on a 210-image subset) | — |
| Paper baseline | PSPNet 0.899 (Chen et al. 2019, ISPRS) | — |

**On-disk crop inventory [MEASURED 08 Sep]:**

| Split | `google_` (10 cm) | `ign_` (20 cm) | total | all-zero masks |
|---|---|---|---|---|
| train | 10,665 | 6,098 | 16,763 | **0** ⚠ C1 |
| val | 1,323 | 771 | 2,094 | — |
| test | 1,315 | 780 | 2,095 | — |

AIRS crops: **absent** (C5).

### 1.2 Target domain — Jaipur [MEASURED off-machine, data ABSENT here — C3]

```
ModelPixelScale (z19 Web Mercator) = 0.2985821 m/px
true ground GSD = 0.2985821 × cos(26.9576°) = 0.26613 m/px ≈ 26.6 cm/px
resolution gap to AIRS = 3.55×
ground area per pixel = 0.0708 m²  (vs 0.005625 m² AIRS)  = 12.6×
```

| Quantity | Value |
|---|---|
| Files | `map67_<col>-<row>.tif`, col,row ∈ 1–4 · 16 tiles · ~8.2 GB |
| Mosaic | 44,672 × 55,424 px = 2.48 gigapixels |
| Extent | 11.9 × 14.8 km ≈ **175 km²** (essentially all of Jaipur) |
| CRS | EPSG:3857, zoom 19 |
| 512² tiles @10% overlap | ≈ **11,140** |
| Buildings | 300,000–500,000 (Open Buildings) |
| Labels | **ZERO. Not one polygon.** |

Grid: first index = column (eastward), second = row (southward); tiles abut exactly.

> **Never hard-code GSD — read it from the GeoTIFF.** The original brief said "~10 cm";
> it is 26.6 cm. Hard-coding would have pushed a **2.8× error into every capacity
> number.** This single line is why the pipeline must stay resolution-parametric.

### 1.3 Compute [MEASURED]

| Resource | Capacity | Use for |
|---|---|---|
| LNMIIT DGX (`lnmdgx1`) | multi-GPU V100/A100-class | all long training |
| Colab | 600 compute units | data prep, SAM2 labelling, eval, short ablations |
| Local RTX 4050 | 6 GB | correctness, 10-sample smoke tests |

DGX rules: `screen` on the **host**, not inside Docker · `docker run --gpus '"device=0,1"'
--shm-size=16g` · copy `/scratch → /tmp` (local NVMe ~10× faster; NFS ~4 s/it) ·
**`/scratch` is not persistent — rsync every checkpoint off immediately.**

---

## 2. Repo state

### 2.1 Built and working [verified]
```
rooftop/         train.py · evaluate.py · infer.py · tile_airs.py
solar_panel/     train_solar.py · evaluate_solar.py · infer_solar.py · prep_bdappv.py
.planning/       PROJECT · REQUIREMENTS · ROADMAP · STATE · config.json · research/
test_indian/     9 PNG screenshots — unlabelled, unregistered, undocumented
Dockerfile       PyTorch 2.1.2 / CUDA 11.8
```
Confirmed features: DGX multi-GPU + AMP · **Hann-window blended** sliding-window inference ·
threshold sweep 0.30–0.55 · per-run JSON/TXT/TensorBoard logging · `--resume` ·
`--simulate_low_res`.

### 2.2 Not built
- No `domain_adapt/`
- No georeferencing-aware Jaipur tiler (`tile_airs.py` has **no** `--no_mask`, no CRS/transform sidecar — verified absent)
- No Jaipur labels, weak or clean
- No SegFormer/MiT run — prerequisite for the entire DAFormer family
- No boundary IoU, no merge/split rate, no per-density-bin IoU
- No irradiance stage — pipeline stops at **kW** and never reaches **kWh**

### 2.3 Errors to fix first

| # | Problem | Fix |
|---|---|---|
| **E0** | **[CORRECTED, NEW]** Solar crop set has **zero negatives**; `prep_bdappv.py:85` skips mask-less images, so the correct fallback in `train_solar.py` never fires | Re-prep retaining negatives; report the negative fraction. **Blocks every Stage 2 number.** |
| E1 | README compares 0.9016 (210-sample subset) to full-test PSPNet 0.899 with a ✅ | Report **0.8664** as the comparable number |
| E2 | `evaluate.py` sweeps threshold on `--test_dir`; README calls 0.35 a result | Sweep on **val**, freeze, report test once. DA gains are *smaller than the threshold effect* |
| E3 | `prep_bdappv.py --sources google ign` pulls the Google-derived split | Default to **`ign` only** (CC BY 4.0, clean) |
| E4 | `.gitignore` hygiene; untracked stray artifacts | Fix before the next `git add -A` |
| E5 | `test_indian/` undocumented | Document; run the current ckpt on it today for free qualitative signal |
| **E6** | **[CORRECTED]** `.planning/STATE.md` says "Phase 1 of 5, Progress 0%" while checkpoints through epoch 050 exist | Docs are stale vs code — reconcile, and pick **one** phase vocabulary (C2/C4) |

---

## 3. The domain gap

### 3.1 Gap 1 — Resolution (**dominant**) [MEASURED]

| | AIRS | Jaipur |
|---|---|---|
| GSD | 7.5 cm | 26.6 cm |
| 3 m feature | 40 px | **11 px** |
| 0.5 m parapet | 6.7 px | **1.9 px** |

Fully simulable. **Simulate the degradation chain, not just the resize:** anti-alias-correct
downsample → mild JPEG (q 65–90) → slight unsharp mask. Naive `cv2.resize` yields
unrealistically clean low-res that does not transfer. Simulate a **range (0.15–0.40 m)**,
not a single value.

### 3.2 Gap 2 — Sensor / photometric
Airborne near-nadir single campaign vs multi-date web mosaic with **visible seams inside a
single 512² crop**, JPEG blocking, low-latitude high sun (short hard shadows), dust haze,
Google tone-mapping. Seams are not fixable by global colour normalisation — robustness
must come from augmentation.

### 3.3 Gap 3 — Morphology and density (**cannot be simulated from AIRS**)

| Property | Christchurch | Jaipur |
|---|---|---|
| Roof form | Pitched, tiled, distinct ridges | **Flat RCC slabs** — no ridge, no shading cue |
| Separation | Detached, 3–10 m gaps | **Shared party walls** — zero gap |
| Density | ~10–20% cover | ~40–60%, up to **80%** in the walled city |
| Clutter | Rare | **Ubiquitous** — Sintex tanks, parapets, stairwell headrooms, laundry, dishes, SWH |
| Street width | Wide, tree-lined | 2–3 m alleys = **8–11 px** at 26.6 cm |
| Height variance | Low (1–2 storey) | High (1–8), heavy inter-building shadow |

**Five distinct failure modes, different fixes:**

1. **Class-prior shift. [MEASURED 2026-09-09 — D1 × D6]** Jaipur's true building-pixel prior
   is **28.2%** over the 16 tiles (23.1% keeping only confidence ≥ 0.75), ranging 10.7%–41.2%
   across tiles. The AIRS-trained seed checkpoint predicts only **5.7%** foreground on those
   same tiles at its reference threshold 0.35 (D6, 640 crops).

   > **The seed under-predicts target foreground by ~5× (28.2% → 5.7%). This is the
   > motivating figure of the project** — the gap the adaptation has to close, measured
   > rather than argued. Even against the conservative 23.1% prior it is ~4×.

   Caveat carried from D1: Open Buildings marks **ground footprints**, not roof outlines
   (Gap 4, §3.4), so 28.2% is a sound density estimate, not a pixel-exact roof prior. The
   AIRS-side number (~15% fg) is **still [ASSUMED]** — D2 needs the AIRS crops (C5).

   A model calibrated on 15% under-predicts. Self-training *amplifies* it: sparse predictions
   → sparse pseudo-labels → sparser teacher → **foreground collapse**. With the prior now
   measured, R10's kill-switch (§ run table) has a real number to compare against.
2. **Instance merging.** No visible gap between party-wall units → whole blocks fuse into
   one component. **Invisible to pixel IoU and largely invisible to boundary IoU** (§6.2).
3. **Vegetation-cue loss [ASSUMED].** A suburban-NZ model may have learned "green texture
   adjacent to bright quadrilateral ⇒ roof". Dense Jaipur wards have no canopy. D5 decides
   in an afternoon; **if ΔIoU is small, drop the idea entirely.**
4. **Flat vs pitched geometry.** Kumar & Indu: US slanted vs Indian planar roofs; corner and
   spectral-graph methods built for US imagery "failed miserably" on planar rooftops. Same
   paper: Google Maps India caps at zoom 20, no 3D, vs zoom 21 US.
5. **Roof clutter — the biggest threat to Stage 2.** A dark rectangular water tank on grey
   RCC resembles a PV module at 26.6 cm. Literature FPs: skylights, glass roofs, dark
   roofing, puddles, shadows; one study finds even the **10 → 20 cm step degrades
   skylight-vs-PV discrimination — and you are at 26.6 cm.**

### 3.4 Gap 4 — Label semantics (easy to miss, costly)
AIRS labels **roof outlines**; Open Buildings / Microsoft label **ground footprints**. A
4-storey Jaipur building (~12 m) at 10° off-nadir diverges by ~2 m ≈ **8 px**. With the
2–8 m georeferencing offset in the open datasets, raw rasterised footprints can be
displaced **10–35 px**.

> **Decision, recorded:** this project predicts **roof outlines** — correct for solar area
> estimation, consistent with AIRS.

Mitigations in order: SAM2 point-prompt snapping → phase-correlation shift estimation per
1024² block → boundary-relaxed loss (ignore 3–5 px band) → drop Open Buildings < 0.75
confidence.

### 3.5 Gap 5 — Spectral separability
Christchurch: dark tiles vs green lawn, strongly chromatic. Jaipur: bare concrete roof,
unpaved lane and construction plot are all the same desaturated grey-beige. **The colour cue
carrying much of the AIRS model's decision is absent.** Forces texture/geometry reliance →
argues for larger effective receptive field (SegFormer, HRDA context crop) and strong colour
augmentation to break the colour shortcut.

### 3.6 Gap 6 — Stage 2's own, worse gap

| | France (BDAPPV) | India |
|---|---|---|
| Mounting | Flush on pitched tiled roofs | **Tilted frames on flat roofs** — cast own shadow |
| Array shape | Large contiguous rectangles | Small, scattered, irregular |
| Confusers | Few | **Solar water heaters** — near-identical from above, *more common than PV* in Indian residential |
| Base rate | Moderate | **Very low** |

> With a very low base rate, **precision matters far more than recall, and IoU is a poor
> headline metric.** Report precision/recall separately plus per-roof detection rate.
> **See C1 — the current crop set makes this unlearnable.**

**In-domain evidence for the DA need:** the official BDAPPV model card reports the
IGN-trained model on Google imagery at **Recall 0.08 / Precision 1.0** — a degenerate point
where the model almost never predicts positive. A labelled cross-domain failure you can
reproduce and fix (R7).

### 3.7 What each gap needs

| Gap | Simulable from source? | Needs target images? | Needs target labels? |
|---|:---:|:---:|:---:|
| 1 Resolution | ✅ fully | — | — |
| 2 Photometric | 🟡 partly | ✅ | — |
| 3 Morphology | ❌ | ✅ | ✅ |
| 4 Label semantics | ❌ | ✅ | ✅ |
| 5 Spectral | 🟡 partly | ✅ | — |
| 6 Solar | ❌ | ✅ | ✅ |

Gaps 1, 2, 5 are **free**. Gaps 3, 4, 6 need target labels — which is why the labelled
Jaipur set is the critical path and **weak supervision from Open Buildings is the
highest-leverage single action.**

---

## 4. Ranked methods

Venue column = supervisor's list: CVF Open Access portal (CVPR/ICCV/WACV/ACCV), WACV, CVPR,
ECCV, ICCV, ACCV, ICIP, MICCAI, ICML, ICPR.

| # | Method | Venue | Yr | ✅ | Fixes | Published gain | Days | GPU | Verdict |
|---|---|---|---|:--:|---|---|---|---|---|
| 1 | Diagnostics D1–D7 | — | — | — | all | — | 2 | none | **Do first, unconditionally** |
| 2 | Resolution-match + geometric + photometric aug | RS 18(8):1176 | 2026 | ❌ | GSD, style | mIoU **0.572→0.688** (res shift, 20% data); 0.444→0.533 (geographic) | 2–4 | low | **Adopt first** |
| 3 | Three-class relabel (bg/interior/boundary) | classic | — | — | **merging** | — | 1–2 | low | **Adopt — best effort:effect here** |
| 4 | FDA (Fourier amplitude swap) | CVPR | 2020 | ✅ | style | 50.45 mIoU19 GTA→CS, +4.0 over BDL; **no network, no adversarial training** | 2–3 | low | **Adopt** |
| 5 | Weak supervision: Open Buildings + SAM2 | — | — | — | morphology, prior, labels | ~40–70k free Jaipur labels | 5–8 | med | **Adopt ★** |
| 6 | CBST per-class thresholds, prior from D1 | — | — | — | **class-prior shift** | — | 2–3 | low | **Adopt — replaces fixed 0.95→0.7** |
| 7 | Merge rate + split rate metrics | — | — | — | measurement | — | 1 | none | **Adopt — without it #3 is unmeasurable** |
| 8 | Multi-source: + Inria (30 cm) + SpaceNet-Khartoum (30 cm) | IGARSS/SpaceNet | 17/18 | 🟡 | GSD **and** morphology, with real labels | — | 3–5 | med | **Adopt ★** |
| 9 | HRDA (detail + context crops) | ECCV | 2022 | ✅ | GSD, small objects | **73.8 GTA→CS (+5.5 over DAFormer)** | 8–15 | **high** | **Adopt (core)** |
| 10 | DAFormer (RCS + FD + warmup, MiT-B5) | CVPR | 2022 | ✅ | self-training, rare class | **68.3 GTA→CS (+10.8 over ProDA)** | 6–10 | med–high | **Adopt (core)** |
| 11 | SegFormer / MiT backbone | ICCV/NeurIPS | 2021 | ✅ | receptive field; **prereq for #9–10** | large share of DAFormer's gain | 2–3 | med | **Adopt** |
| 12 | Confidence self-training (2–3 rounds) | — | — | — | target consistency | — | 5–8 | med | **Adopt** |
| 13 | Hard-negative mining: tanks, AC, SWH | — | — | — | **PV false positives** | — | 3–4 | low | **Adopt — but fix C1/E0 first** |
| 14 | MIC | CVPR | 2023 | ✅ | target context | **75.9 GTA→CS** with HRDA | 3–5 | high | **Stretch goal** |
| 15 | Contour head + SPLIT separation loss | Springer | 2025 | ❌ | merging | differentiable merge-rate proxy; backbone-independent | 3–5 | med | **If #3 insufficient** |
| 16 | SegDesicNet (geo-coord embeddings) | **WACV** | 2025 | ✅ | domain shift | ~+6% mIoU, −27% params | 5–8 | low–med | **Cheap + approved** |
| 17 | Scheibenreif PEFT geospatial DA | **CVPR** | 2024 | ✅ | PEFT self-sup on unlabelled target | — | 5–8 | med | **Consider** |
| 18 | ST-DASegNet | ISPRS J. | 2024 | ❌ | RS-specific DA | SOTA Potsdam/Vaih/LoveDA | 8–12 | med | **Baseline/compare** |
| 19 | ResiDualGAN (in-net resizer) | RS | 2023 | ❌ | scale | in-net resize **55.83 vs 53.46** pre-resize | 6–10 | med | **Compare only** |
| 20 | Scale-Aware Adaptation | **WACV** | 2021 | ✅ | explicit scale align | "large margin" over SOTA DA | 5–8 | med | **Compare (isolates scale)** |
| 21 | SAM2 as label engine | ICCV | 2023 | ✅ | label semantics | — | 3–5 | med | **Adopt as engine, not model** |
| 22–25 | ProDA · DACS · AdaptSegNet/AdvEnt/CLAN/MCD · IAST/PyCDA | CVPR/WACV/ECCV/ICCV | 18–21 | ✅ | — | dominated by self-training | — | med | **Baseline rows** |
| 28 | AdaBN | — | — | — | BN statistics | "sometimes several IoU points" | 0.5 | trivial | **Phase B; N/A for SegFormer (LayerNorm)** |
| 29 | Multi-scale + flip TTA | — | — | — | robustness | +1–3 IoU under shift, 6× cost | 1 | low | **Adopt** |
| 30–36 | Earth-Adapter · CrossEarth · SAM2+GNN (AAAI'26, India) · RS-Mamba · source-free · DUDA · frame fields | AAAI/arXiv | 24–26 | ❌ | various | — | 6–20 | high | **Defer, cite** |
| 37–38 | nDSM / elevation-aware fusion | MDPI RS | — | ❌ | merging | under-seg 35.7%→5.0%, BF1 79.78% | — | — | **Skip — no Jaipur DSM. Cite, don't implement** |
| 39–42 | Watershed post-proc · CycleGAN/ColorMapGAN · super-resolve to 7.5 cm · train-from-scratch-on-Jaipur | — | — | — | — | FDA gets most of translation free; SR fabricates detail | — | — | **Skip** |

### Dominance, stated plainly
- **Self-training dominates adversarial-only UDA.** AdaptSegNet, AdvEnt, CLAN, CyCADA, MCD are baseline rows, not engineering targets.
- **HRDA's multi-resolution design is the best architectural match to a 3.55× GSD gap** — it handles scale *inside* the adaptation loop, not as preprocessing.
- **Cheap augmentation + FDA likely dominate foundation models on cost-adjusted ROI** for a single-city BTP. Cite CrossEarth/Earth-Adapter; don't build the thesis on them.
- **Weak supervision from Open Buildings likely dominates every clever unsupervised algorithm here.** ~40–70k free noisy target labels beats a model that never saw an Indian roof.
- **ResiDualGAN's resizer and the WACV scale discriminator are probably dominated by "HRDA + resolution augmentation" combined** — keep as comparators that isolate scale, which strengthens the ablation narrative.
- **A labelled dense-morphology source domain beats any unsupervised morphology fix.** Inria (30 cm, dense European cores) and SpaceNet-Khartoum (30 cm, semi-arid dense low-rise) attack adjacency, geometry, prior and vegetation-absence *simultaneously, with real labels.*

---

## 5. Pipeline

**[CORRECTED]** The source doc's Phases A–F do not match `.planning/ROADMAP.md` Phases 1–5
(C2/C4). A–F is the better decomposition for the DA-framed project; adopt it and **retire
the 1–5 vocabulary** rather than maintaining two.

```
Phase A — MEASURE
  Repo hygiene E0–E6            ← E0 (negatives) is new and blocking
  Georeferencing-aware Jaipur tiler → ~11,140 tiles + sidecar JSON
  Open Buildings + MS footprints → clip, confidence-filter, rasterise
  Diagnostics D1–D7
  SAM2 label engine → weak labels
  Hand-label 400 tiles (25/mosaic tile) → 200 train / 60 val / 140 SEALED test
    + mark superstructures & SWH as extra classes on ~100   ← feeds k_usable AND Stage 2
  Zero-shot baseline on the Jaipur eval set
  Run current ckpt on test_indian/ for free qualitative signal

Phase B — FREE WINS (no training)
  GSD resampling both directions · AdaBN · FDA · histogram match
  Threshold recalibration on target VAL · multi-scale + flip TTA

Phase C — ROBUST SOURCE RETRAIN
  AIRS↓(0.15–0.40 m) + Inria(30 cm) + SpaceNet-Khartoum(30 cm)
  Heavy scale + photometric aug · FDA(Jaipur) as augmentation
  Two arms: unet/resnet34 (controlled baseline) AND segformer/mit_b2
  Three-class targets (bg / interior / boundary)

Phase D — WEAK SUPERVISION  ★ biggest single win
  Finetune on ~11,140 noisy Jaipur tiles, boundary-relaxed loss, LR÷5
  Then clean finetune on 200 hand-labelled tiles, LR÷10
  ← CHECKPOINT THE DELIVERABLE HERE. Everything after is upside.

Phase E — UDA
  Confidence self-training, 2–3 rounds, CBST thresholds with ratio from D1
  Log pseudo-label fg fraction vs measured prior every round
  Stretch: MIC(HRDA), mit_b5, mmsegmentation container

Phase F — STAGE 2 + ENERGY
  Fix E0 first. Solar adaptation BDAPPV ign → Jaipur, SWH as third class
  Hard-negative mining: water tanks, AC units, parapets
  Area → usable area → kW → **kWh/yr via GTI@OPTA × PR**
  Validate against PM Surya Ghar + prior Indian studies
```

> **Do not put MIC(HRDA) on the critical path.** Phase D + Phase E self-training gets most
> of the way; MIC is the headline number you add if DGX time materialises.

---

## 6. Experiments, diagnostics, metrics

### 6.1 DGX run sequence

| Run | Config | Metric that matters | Gate |
|---|---|---|---|
| R0 | Existing ckpt, threshold fixed on **val** | IoU full test | Establishes honest **0.8664** |
| R1 | Diagnostics D1–D7 | prior gap, adjacency rate, ΔIoU_veg | If D4 adjacency low → **skip three-class work** |
| R2 | + GSDResize (0.15–0.40 m, degradation chain) | Jaipur IoU, qualitative | — |
| R3 | + geometric/photometric aug + FDA | Jaipur IoU | Reference 0.572→0.688 |
| R4 | + three-class targets | **merge rate**, Boundary F1 | Merge rate must drop materially. **Pixel IoU may barely move — that divergence is the finding** |
| R5 | SegFormer `mit_b2`, else = R4 | IoU, merge rate | Quantifies backbone contribution |
| R6 | + Inria + SpaceNet-Khartoum multi-source | IoU, merge rate on held-out dense tile | Expect the **largest single jump** |
| **R7** | **BDAPPV google→ign labelled sandbox**: tune λ_st, CBST proportion, class-mix rate | **target IoU (real labels)** | **Freeze hyperparameters here, then transfer to Jaipur** |
| R8 | Weak supervision, ~11,140 tiles, boundary-relaxed | IoU on sealed test | ★ expect 0.74–0.82 |
| R9 | Clean finetune, 200 tiles, LR÷10 | IoU | **SHIPPABLE CHECKPOINT** |
| R10 | Self-training, CBST, R7 hyperparameters | pseudo-label fg fraction vs D1 | **Kill run if fg fraction diverges from measured prior** |
| R11 | Solar S1, BDAPPV `ign` only — **after E0** | precision, recall **separately** | Watch for Recall≈0.08 degenerate point |
| R12 | Solar S2 + clutter hard negatives | **precision**, FP count on tanks/AC/SWH | Expect the biggest precision gain |
| R13 | Stretch: MIC(HRDA) mit_b5 | IoU | Only if DGX time is free |

> **R7 is the highest-value experiment in the list, and it is runnable today** — the crops
> are on disk with `google_`/`ign_` filename provenance (verified). It is a *labelled*
> cross-domain pair with a *documented* failure, so you can tune a self-training pipeline
> against ground truth before applying it to Jaipur, where you have none. Kasmi et al.
> built the dataset that way deliberately.

### 6.2 Metrics — log every epoch

Keep globally-accumulated IoU / F1 / precision / recall. **Add:**
- **Boundary IoU** (5 px band around GT edges)
- **Merge rate** / **split rate**:
  ```
  merge_rate = (# pred components containing ≥2 GT buildings) / (# pred components)
  split_rate = (# GT buildings covered by ≥2 pred components) / (# GT buildings)
  ```
- **Per-density-bin IoU** — low / medium / high cover, reported separately
- **Pseudo-label foreground fraction** vs measured prior (self-training only)

> **Boundary IoU does not catch merging.** Boundary IoU measures edge *localisation quality*
> in a band around GT edges; merging is a **topology** error. A prediction can score well on
> the outer perimeter of a block of eight party-wall houses while containing **zero internal
> separations** — there are no predicted edges there to score badly. Calibration: a
> high-density Hong Kong study cut under-segmentation **35.7% → 5.0%** while reporting
> Boundary F1 **79.78%** — the numbers move independently.

### 6.3 Diagnostics D1–D7 — before any adaptation work

| # | Diagnostic | Method | Decides |
|---|---|---|---|
| D1 | **Target class prior** | Rasterise Open Buildings over the AOI; building-pixel fraction per mosaic tile and density bin | CBST ratio; replaces ASSUMED ~50% |
| D2 | Source foreground fraction | Histogram over `airs_crops/train` masks | Baseline for D1; replaces ASSUMED ~15% |
| D3 | Building size distribution | Connected-component area histogram in **m²**, AIRS vs OB-Jaipur | Whether 512² crop and receptive field are right |
| D4 | Adjacency rate | Fraction of OB polygons sharing a boundary with ≥1 neighbour; same for AIRS | **Whether three-class / SPLIT work is needed at all** |
| D5 | Vegetation dependence | Grey out vegetation in AIRS test crops, re-run ckpt, ΔIoU | Whether Gap 3.3 is real |
| D6 | Seed-model probe | Run ckpt on Jaipur tiles; predicted fg fraction + confidence histogram | Whether the seed is usable as a teacher at all |
| D7 | Clutter inventory | Hand-inspect 20 Jaipur tiles; tally obstacle types/counts | Which hard negatives to mine; feeds `k_usable` |

> **D1 + D6 together are the most informative pair.** If D6's predicted foreground fraction
> is far below D1's measured prior, you have quantified the exact failure the project exists
> to fix. **That is your motivating figure.**

**[UPDATED 2026-09-09 — supersedes the earlier "zero of D1–D7 can run" note]** The Jaipur
GeoTIFFs are restored (C3), so the tile-side diagnostics are unblocked. AIRS is only
partially restored (C5), so the source-side ones are not.

| # | Needs | Status |
|---|---|---|
| **D1** | Jaipur tiles + Open Buildings | ✅ **run 2026-09-09** → `diagnostics/d1/d1_summary.json` |
| **D2** | AIRS crops | ⛔ blocked on C5 |
| **D3** | AIRS crops + OB | ⛔ AIRS half blocked on C5; OB half runnable |
| **D4** | OB polygons (+ AIRS for comparison) | 🟡 OB half runnable today |
| **D5** | AIRS crops + ckpt | ⛔ blocked on C5 |
| **D6** | Jaipur tiles + ckpt | ✅ **run 2026-09-08** → `diagnostics/d6/d6_summary.json`, 640 crops |
| **D7** | Jaipur tiles (manual) | 🟡 runnable, hand-inspection not started |

Remaining blocker is therefore **AIRS only**: finish the interrupted
`scripts/fetch_drive_folder.py` pull (it skips completed files, so it is safe to re-run),
then tile with `rooftop/tile_airs.py`. **The Drive folder ID is not recorded anywhere in
the repo — capture it in this file when you next run the fetch.**

### 6.4 Evaluation protocol — non-negotiable

| Rule | Reason |
|---|---|
| Spatial block splits; hold out **2–3 entire mosaic tiles** | Overlapping tiles leak. Geographically disjoint > block-disjoint |
| Test tiles span fabric types: walled-city, planned-colony, peri-urban | A test set from the easy outskirts flatters the model |
| Threshold tuned on target **val**, reported on target **test** | Otherwise you tuned on your test set (E2) |
| Sealed test read **exactly once**, at the end | Every peek is an implicit fit |
| Report AIRS in-domain IoU alongside Jaipur IoU | **The gap is the result** |
| Zero-shot baseline in every table | Without it nobody sees what adaptation bought |
| Report negative results | Things that didn't help are findings |
| Characterise weak-label noise | IoU between raw Open Buildings labels and hand labels |
| State every capacity constant in text | `k_usable`, η, GSD, PR, GTI — not buried in code |
| State imagery provenance and licence | §8 |

### 6.5 Expected outcome [PLANNED — targets, not predictions]

| Phase | Expected Jaipur IoU |
|---|---|
| A — zero-shot | 0.35–0.55 (expect ugly) |
| B — input-space alignment | 0.50–0.65 |
| C — robust multi-source | 0.62–0.72 |
| D — weak supervision | **0.74–0.82** |
| E — self-training / MIC | 0.78–0.85 |

AIRS in-domain is 0.8664. **Do not expect to reach that on Jaipur.** Published cross-domain
building segmentation in dense Indian cities lands in **0.70–0.85**. A well-characterised
0.80 with an honest gap analysis is a strong result; **a claimed 0.90 means labels leaked.**
The 0.35–0.55 floor may be pessimistic given `--simulate_low_res` already exists — don't
anchor the report on beating your own conservative estimate.

---

## 7. The energy chain ★ — completing the stated goal

**The largest single gap in the plan.** The pipeline terminates at:
```
kW ≈ roof_pixels × 0.07083 × k_usable(0.60) × η(0.20) × 1.0 kW/m²
```
That is **installed capacity at STC**. It is not energy. There is no irradiance term
anywhere, and no kWh figure.

### 7.1 The missing arithmetic
```
E_annual (kWh/yr) = P_installed (kWp) × PVOUT (kWh/kWp/yr) × PR
        or        = usable_area_m² × η × GTI_annual (kWh/m²/yr) × PR
```
**PR** bundles temperature derating, soiling, inverter/wiring losses, mismatch. For Rajasthan
both heat and dust are material — Singla et al. find seasonal/environmental/technical factors
can cost **up to 50%** of generation across Indian cities. PR 0.75–0.80 is the standard
starting assumption; **state and justify it rather than burying it.**

### 7.2 Data sources — free and licence-clean
World Bank / Solargis **Global Solar Atlas** publishes India layers as long-term yearly
averages: **PVOUT, GHI, DIF, GTI, OPTA, DNI**. NREL **NSRDB** is the other reference standard.
India GHI ranges ~3.8 (NE hills) to ~6.5 kWh/m²/day (western Rajasthan).

1. **OPTA removes the DSM requirement for tilt.** Elevation-aware UDA was correctly rejected
   for lack of a Jaipur DSM — but for *energy* you don't need one: Indian flat RCC roofs take
   frame-mounted arrays at optimum tilt, not conforming to a roof plane. Read OPTA and GTI
   from the atlas. Western studies spend most of their DSM effort on pitch and aspect; the
   flat-roof target removes that. **Claim this as a methodological simplification enabled by
   the target domain, not as a limitation.**
2. **Use GTI, not GHI.** GHI is horizontal; panels are tilted. GHI understates output. GTI at
   OPTA is the correct pairing for the mounting assumption.

### 7.3 Constants

| Constant | Value | Status |
|---|---|---|
| GSD | read from GeoTIFF | [MEASURED] |
| Area per pixel | 0.07083 m² | [MEASURED] |
| `k_usable` | 0.60 | **[ASSUMED] — measure it, §7.4** |
| η (module efficiency) | 0.20 (mono-PERC) | [PLANNED] |
| **GTI @ OPTA, Jaipur** | from Global Solar Atlas | **[TODO]** |
| **OPTA, Jaipur** | from Global Solar Atlas | **[TODO]** |
| **PR** | 0.75–0.80 | **[TODO — state and justify]** |

Add a **kWh/yr column to every capacity table.** Extend the ethics caveat
("order-of-magnitude estimate, not an engineering assessment") to cover PR and irradiance
uncertainty.

### 7.4 `k_usable` is the weakest number — turn it into a result
0.60 moves the headline GW figure by **±25%** and has no measurement behind it. A 2024
*Applied Energy* study argues existing work generally **does not account for roof
superstructures**, and that ignoring them **overestimates solar potential**.

You already have the ingredients: Gap 3 names the exact Jaipur superstructures, and you are
already hand-labelling 400 tiles.

> **Proposal:** while hand-labelling, mark superstructures as a third class on ~100 tiles.
> Compute `k_usable = 1 − (superstructure + setback area) / roof area`, **per density bin**,
> and report it as a **measured distribution, not a scalar assumption.** Cost: a few hours
> inside labelling you are doing anyway. Same pass gives the SWH class for Stage 2 — do them
> together. **The cheapest genuine contribution available in the project.**

### 7.5 Validate the energy number, not just the mask
Nothing currently validates the thing the project exists to produce. **PM Surya Ghar**
publishes installed rooftop capacity, and Rajasthan + Gujarat are ~40% of national scheme
completions — so Jaipur has real deployment to check against.

1. **Existing-installation recall.** Stage 2 detected count/kW per Jaipur ward vs reported
   PM Surya Ghar installations in the same wards. A genuine external check on a model with
   no target labels.
2. **Potential plausibility.** Total Jaipur technical potential vs published per-city Indian
   estimates and MNRE state figures.

Neither is rigorous — deployment lags potential, reporting granularity is coarse. But it is
the difference between "we produced a figure" and "we produced a figure and checked it."
**None of the prior Indian works does this well.**

---

## 8. Licensing

**The blocker:** Google Maps Platform terms prohibit using Maps content to "train, test,
validate or fine-tune" ML models, and separately prohibit derived content and derivative map
datasets. The Jaipur `.tif`s are Web Mercator z19 mosaics with Google tiling geometry.
BDAPPV's `google/` split carries the same exposure.

| Source | Licence | ML training? | Notes |
|---|---|---|---|
| Google Maps/Earth (Jaipur tiles) | Proprietary ToS | **NO** | **Publication-blocking** |
| BDAPPV `google/` | CC BY-NC 4.0 | NC only | OK for non-commercial BTP w/ attribution |
| **BDAPPV `ign/`** | **CC BY 4.0** | **YES** | Cleanest PV source — **make it the default** |
| Esri World Imagery | Esri MLA | NC, in ArcGIS only | Tiles can't leave ArcGIS |
| Maxar/Vantor Open Data | CC BY-NC 4.0 | NC only | 25–30 cm, disaster AOIs; Jaipur unlikely |
| Planet E&R / NICFI | Custom NC | NC only | 3 m / 4.77 m — too coarse |
| Airbus Pléiades / Neo | Commercial / academic EULA | Licence-dependent | 50/30 cm; ESA TPM or DINAMIS access possible |
| WorldStrat | CC BY-NC 4.0 imagery | NC only | 1.5 m — too coarse |
| **OpenAerialMap** | CC BY 4.0 (per layer) | **YES** | 10–30 cm; check Jaipur coverage |
| OpenEarthMap | CC BY-NC-SA 4.0 | NC only | 0.25–0.5 m, built for DA research |
| **SpaceNet-2 / DeepGlobe** | CC BY-NC 4.0 | **NC — fine for BTP** | 30 cm WV-3; **Khartoum + Shanghai are the closest morphological analogues to Jaipur** |
| **Inria Aerial Image Labeling** | Open (research) | **YES** | **30 cm ≈ Jaipur's GSD**; 810 km², dense cores |
| Open Cities AI Challenge | Open | **YES** | 20 cm, ~790k footprints, African urban — dense, flat-roofed, no canopy |
| RAMP | Open | **YES** | Built for Global South microplanning |
| **Google Open Buildings v3** | CC BY 4.0 **or** ODbL | **YES** | 1.8 B detections, India covered, per-building confidence |
| **Microsoft GlobalMLBuildingFootprints** | ODbL v1.0 | **YES** | 110 M India footprints (2024) |
| VIDA combined Google–MS–OSM | ODbL | YES | Usually the most convenient single download |
| OpenStreetMap Jaipur | ODbL | YES | Walled city well-mapped, outskirts sparse — spot-check only |
| ISRO Bhuvan / Bhoonidhi | NDSAP / OGDL India | Free med-res only | LISS-III 23.5 m, AWiFS 56 m — **too coarse**; sub-metre RGB is priced |
| SVAMITVA drone ortho (~5 cm) | None public | **NO / unclear** | Not publicly downloadable |
| Local UAV capture | yours | YES | Even 1 km² self-captured is a clean, citable, **novel** contribution |

**ODbL is share-alike** — a published derived database must also be ODbL. Choosing CC BY 4.0
for Open Buildings avoids that for that portion; Microsoft is ODbL-only. Attribute both.

**Recommendation:** develop methods on the Google-sourced tiles while sourcing licensed
imagery in parallel. **The pipeline reads GSD from the GeoTIFF, so swapping imagery late
costs a re-run, not a redesign.** For a publishable version pivot the target to
**OpenAerialMap** over Jaipur (if coverage exists) with **Open Buildings** labels, and add
**Inria + SpaceNet-Khartoum** as licensed source domains. Write a provenance-and-licence
paragraph regardless.

---

## 9. Positioning and novelty

Four groups have done versions of the end-to-end task in India:

| Work | Setting | Had that you don't | You have that they don't |
|---|---|---|---|
| **Singla, Kalase et al.** (arXiv:2411.04610; JISRS 2025) — ISRO/SAC | Ahmedabad, Gandhinagar, Delhi | 50 cm + **1 m DEM** + IMD ground-station radiation; U-Net+VGG19; shadow, slope, aspect | No DEM needed (flat roofs + OPTA); no ground stations; **no target labels at all** |
| *Rooftop Solar Potential of an Indian Metropolis* (IEEE 9553088) | Indian metropolis | Medium-res; **OSM footprints as labels**; ~1.89 GW; validated on 25 km² | Higher resolution; DA framing; a PV-detection stage |
| *Rooftop Solar Energy via Satellite Image Segmentation* (IEEE 8971578) | **Bangalore** | Hand-labelled; U-Net vs SegNet vs FCN | **Scale — 175 km² vs a small AOI** |
| **Singh & Banerjee**, IEEE PVSC 2013 | IIT Bombay campus → macro extrapolation | PVsyst micro-simulations | City-scale segmentation vs category extrapolation |

> **Your defensible novelty is the zero-target-label constraint.** Every one of them had
> labels — DEM-assisted, OSM-derived, or hand-drawn. You are doing city-scale Indian rooftop
> solar estimation by **transferring from foreign labelled data with no Indian ground
> truth.** Frame the BTP that way and those four become your **comparison baselines rather
> than your competition.** Add a row comparing your Jaipur estimate to their published GW
> figures for comparable Indian urban areas.

### Venue-filter audit
Only **4 of ~35** sources in the original bibliography pass the supervisor's list: DAFormer
(CVPR'22), HRDA (ECCV'22), MIC (CVPR'23), FDA (CVPR'20). This is **structural, not an
oversight** — remote-sensing DA lives in TGRS, ISPRS J., RSE and MDPI *Remote Sensing*, not
the CVF circuit.

**Two actions:**
1. **Ask whether the venue list is a hard requirement or a quality proxy.** Excluding TGRS
   and ISPRS J. from a rooftop-segmentation BTP would be unusual — those are the field's top
   venues. If it's a proxy for "peer-reviewed and reputable", keep everything and add the
   CVF spine.
2. **If hard**, these also pass and are relevant: SAM (ICCV'23) · DACS (WACV'21) ·
   ProDA (CVPR'21) · IAST (ECCV'20) · CLAN (CVPR'19) · PyCDA (ICCV'19) ·
   OpenEarthMap (WACV'23) · SegDesicNet (WACV'25) · Scheibenreif (CVPR'24) ·
   Scale-Aware Adaptation (WACV'21) · CyCADA (ICML'18) · crossMoDA (MICCAI'21).

### The MICCAI entry earns its place
**crossMoDA** (Dorent et al., MICCAI 2021; *Medical Image Analysis* 2022, arXiv:2201.02831):
105 labelled ceT1 + 105 unpaired unlabelled hrT2, 55 teams to validation, 16 to evaluation,
best median Dice 88.4% (VS) / 85.7% (cochlea) — close to full supervision. **Transferable
finding: all top-performing methods used image-to-image translation to convert source into
pseudo-target, then self-trained.**

Independent, outside-RS, venue-approved evidence for **translation/alignment before
self-training**. It mildly argues against the outright rejection of CycleGAN-style
translation. The rejection is still defensible on timeline grounds; the honest framing is
*"the medical UDA literature finds translation carries real weight; we substitute the
training-free FDA variant for schedule reasons."*

---

## 10. Bibliography

Full 69-entry bibliography is in the source master document (session transcript,
08 Sep 2026) and in `.planning/research/` literature notes. Key anchors:

**Venue-approved ✅:** DAFormer (CVPR'22, `lhoyer/DAFormer`) · HRDA (ECCV'22,
`lhoyer/HRDA`) · MIC (CVPR'23, `lhoyer/MIC`) · FDA (CVPR'20, `YanchaoYang/FDA`) ·
ProDA (CVPR'21) · AdaptSegNet (CVPR'18) · AdvEnt (CVPR'19) · CLAN (CVPR'19) · MCD (CVPR'18) ·
CyCADA (ICML'18) · IAST (ECCV'20) · PyCDA (ICCV'19) · Araslanov & Roth (CVPR'21) ·
DACS (WACV'21) · Scale-Aware Adaptation (WACV'21, `xdeng7/scale-aware_da`) ·
SegDesicNet (WACV'25) · Scheibenreif PEFT (CVPR'24, `HSG-AIML/GDA`) ·
crossMoDA (MICCAI'21) · OpenEarthMap (WACV'23) · SAM (ICCV'23) · SegFormer (NeurIPS'21) ·
Inria benchmark (IGARSS'17).

**RS / domain-specific ❌:** ST-DASegNet (ISPRS J.'24, `cv516Buaa/ST-DASegNet`) ·
ResiDualGAN (RS'23, `miemieyanga/ResiDualGAN-DRDG`) · HighDAN/C2Seg (RSE'23) ·
CrossEarth (arXiv:2410.22629, `Cuzyoung/CrossEarth`) · Earth-Adapter (AAAI'26) ·
Aljabri et al. augmentation study (RS 18(8):1176, 2026) · SPLIT loss (Springer'25) ·
ACM-PSPNet (RS 18(17):2971) · SpaceNet (arXiv:1807.01232).

**Solar / energy:** Singla et al. (arXiv:2411.04610) · Kumar & Indu (arXiv:1812.11606) ·
Kasmi BDAPPV dataset (*Sci Data* 10, 2023) · Kasmi reliability (*Env. Data Sci.* 2024,
model card `gabrielkasmi/bdappv-models`) · RPS (RS 15(21):5232) · S3Former (arXiv:2405.04489) ·
roof-superstructures (*Applied Energy* 2024).

**Data/policy:** Global Solar Atlas (PVOUT/GHI/GTI/OPTA) · NREL NSRDB · PM Surya Ghar ·
Google Open Buildings v3 · Microsoft GlobalMLBuildingFootprints · VIDA · Open Cities AI ·
RAMP · Google Maps Platform Service Specific Terms.

---

## 11. Unverified — do not treat as ground truth

- **`k_usable`=0.60, η=0.20, PR=0.75–0.80** are assumptions until measured or sourced. §7.4 fixes the first.
- **"AIRS ~15% / Jaipur ~50% foreground"** is an estimate. D1/D2 replace it. **Do not put estimated priors in the report.**
- **The vegetation-cue hypothesis (§3.3.3)** has no paper behind it. D5 decides. **If ΔIoU is small, drop the idea entirely.**
- **Published mIoU numbers** (DAFormer 68.3, HRDA 73.8, MIC 75.9, FDA 50.45, ProDA 57.5) are **GTA/Synthia→Cityscapes street scenes, not aerial imagery.** Relative-ordering evidence, not absolute targets for a binary building task.
- **HRDA's gains require full-resolution training** (confirmed in the Instance-Warp ablations: HRDA underperforms DAFormer at DAFormer's training scales). On V100 32 GB this constrains batch size — **the main compute risk.**
- **§6.5 expected IoUs** are planning targets, not predictions.
- **Aljabri et al. 2026, ACM-PSPNet, CrossEarth-Gate, DenseUIS, AAAI'26 dense-settlement** are recent/preprint. Verify final published versions.
- **Google Maps licensing is unresolved and publication-blocking.** Everything here assumes you fix it. **Re-read the current Terms yourself** before relying on §8 in a submitted document.
- **[NEW] The `plan/` directory this doc's cross-references point to does not exist** (C2). Section citations like `plan/04 §6` cannot be checked. Treat them as descriptions, not pointers, until the planning docs are reconciled.

---

## 12. If you only do seven things

0. **[NEW, blocking] Fix the zero-negatives bug (E0/C1).** Re-prep BDAPPV keeping mask-less
   images so the black-mask fallback actually fires. Every Stage 2 precision number produced
   before this measures the wrong task.
1. Fix E1 and E2 — honest baseline, no threshold leakage onto test.
2. **Finish restoring AIRS** — Jaipur GeoTIFFs and Open Buildings are now on disk (C3, C6),
   but AIRS stopped at 50/857 images and 3 usable pairs (C5). It is the only dataset still
   blocking D2/D3/D5 and all of Phases A–E. This is the real critical path today.
3. Run the rest of D1–D7. **D1 and D6 are done** (§6.3); D4-on-OB and D7 are runnable now,
   D2/D3/D5 wait on item 2. Every number replaces an assumption.
4. Add merge rate + split rate to `evaluate.py`, **then** do the three-class relabel. Without
   the metric the fix is unmeasurable.
5. Build the Open Buildings weak-label set. ★ highest-leverage single action.
6. **Tune self-training hyperparameters on the labelled BDAPPV `google`→`ign` sandbox before
   touching Jaipur** — runnable today, the only rung whose data is fully present.
7. Add the irradiance stage — **GTI @ OPTA × PR → kWh/yr.** Without it the project doesn't do
   the thing it's named after.
