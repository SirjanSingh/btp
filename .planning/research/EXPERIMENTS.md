# UDA Experiment Plan — ordered by what we can actually run

Derived from the ranked methodology report (2026-09-08). This file turns that report
into runnable experiments with explicit decision thresholds, ordered by *what data
exists on disk today*, not by what looks most impressive.

## Ground truth about the data (verified 2026-09-08)

| Asset | State | Consequence |
|---|---|---|
| `solar_panel/bdappv_crops/` | **41,904 files, present** | Sandbox is runnable *now* |
| — `google_*` prefix | 10,665 train / 1,323 val / 1,315 test | Source domain (10 cm/px) |
| — `ign_*` prefix | 6,098 train / 771 val / 780 test | Target domain (20 cm/px) |
| `data/airs/`, `data/airs_crops/` | **EMPTY (0 bytes)** | Rooftop source data must be re-fetched/re-tiled |
| `rooftop/dataset_crops/` | **0 files** (only deleted `.gitkeep`s) | Same |
| `rooftop/checkpoints/*.pth` | present (resnet34) | Old in-domain weights survive |
| `test_indian/` | 9 PNG screenshots | Not a corpus; Jaipur target set does not exist yet |

**The single most useful fact:** BDAPPV crops keep provenance in the *filename prefix*
(`google_` / `ign_`). The google→ign UDA sandbox the report asks for needs **no new
download and no re-preprocessing** — only a filename filter.

## Why the sandbox goes first (not rooftop)

The report's Stage 0 says "stand up BDAPPV google→ign as a fully-labelled proxy sandbox."
Three independent reasons this is now the *only* sensible starting point:

1. **It is the only stage whose data exists.** AIRS crops are gone; Jaipur target set was
   never built. Rooftop work is blocked on data, not ideas.
2. **Both domains are labelled**, so target IoU is *measurable*. Jaipur is unlabelled —
   you cannot tune a confidence threshold or λ_st against a number you cannot compute.
   Every hyperparameter tuned here transfers to Jaipur as a frozen config.
3. **The 10 cm → 20 cm gap is a 2× GSD shift** — a scaled-down rehearsal of the
   7.5 cm → 26.6 cm (3.55×) Jaipur gap, same *kind* of shift, one order less severe.

Additional gift: Kasmi et al. published the exact failure mode
(**IGN-trained model on Google imagery: Recall = 0.08, Precision = 1.0** — degenerate,
almost never predicts positive). Reproducing a *published* failure is the cheapest
possible proof that the harness measures the right thing.

## Experiment ladder

Each rung is one row in the final ablation table. Do not skip rungs — the value is the
*deltas*, and a skipped rung is a hole in the paper.

### E0 — Harness + oracle/floor bracket  *(no new modelling)*
Establishes the two numbers every later result is read against.

| Run | Train on | Test on | Purpose |
|---|---|---|---|
| E0.a **source-only** | `google_*` | `ign_*` test | The floor. This is the drop UDA must close. |
| E0.b **target-oracle** | `ign_*` | `ign_*` test | The ceiling. Full supervision on target. |
| E0.c **reverse** | `ign_*` | `google_*` test | Reproduce Kasmi's Recall≈0.08 collapse. |

**Gate:** if E0.a ≈ E0.b there is no domain gap worth adapting and this whole plan is
misaimed — stop and re-examine. Expect a clear gap; report it with a bootstrap CI over
tiles, not a bare point IoU.

**Metrics from day one** (report says plain IoU under-weights small objects):
IoU · **boundary-IoU** · precision · **recall** · **ECE (calibration)**.
Recall is the canary — Kasmi's collapse is invisible in IoU alone.

### E1 — Cheap wins (report's highest-ROI stage)
Layer these one at a time onto E0.a. Aljabri et al. reference: mIoU 0.572 → 0.688.

1. **geometric aug** — report says consistently the strongest single family
2. **resolution-matching** — downsample `google_*` toward 20 cm (`--simulate_low_res`
   already exists at `rooftop/train.py:238`, needs porting to `train_solar.py`)
3. **photometric jitter + histogram matching**
4. **FDA** (Fourier amplitude swap, CVPR 2020) — cheapest style alignment, no adversarial
   training, no second network

**Decision threshold (from the report):** if E1 recovers ≈+10 mIoU of the E0.a→E0.b gap,
**deprioritise the heavy UDA stages** and spend the time on evaluation rigor and
comparators instead. This is a real possible outcome, not a consolation prize.

### E2 — Self-training core  *(only if E1 leaves a gap worth closing)*
Prerequisite: **encoder swap ResNet-34 → SegFormer MiT-B5.** The published DAFormer/HRDA/MIC
numbers assume it; on U-Net+ResNet-34 the family will materially underperform and the
comparison would be dishonest.

1. **DAFormer** (RCS + FD + LR warmup) — RCS directly targets the rare small-PV class
2. **HRDA** (detail crop + context crop) — the mechanism matched to the GSD gap
3. **MIC** — only if V100 budget allows; full-res HRDA+MIC is the heaviest config

**Compute risk (flagged in the report):** HRDA's gains *require full-resolution training*,
independently confirmed. On V100 32 GB this constrains batch size. Budget for it or
report honestly that HRDA ran below its intended scale.

**Ordering evidence (crossMoDA, MICCAI 2021):** every top team did
**translate → then self-train**, and self-training carried most of the gain. So if solar
recall is still < 0.5 after E1, do translation *before* self-training, not after.

### E3 — Rooftop stage  *(blocked: needs data)*
Unblock by re-fetching AIRS **or** pivoting to the legally-clean substitute the report
recommends. Do not re-tile AIRS at scale before deciding, because of E4.

### E4 — Licensing pivot decision  *(do this before generating any Jaipur figures)*
The report is blunt: **Google Maps ToS prohibits using Maps content to train, test,
validate or fine-tune ML models.** The current Jaipur tiles and the BDAPPV `google/`
split are both encumbered (`google/` is CC BY-NC — fine for a non-commercial BTP with
attribution, but not for a product).

Clean path if publishing: **SpaceNet-2 Khartoum/Shanghai (30 cm ≈ Jaipur's 26.6 cm)** as
an extra labelled source + **OpenAerialMap** for Indian target tiles + **Google/Microsoft
Open Buildings** as auxiliary labels. Keep **BDAPPV `ign/` (CC BY 4.0)** as the clean
primary PV source.

Deciding this *late* means re-running everything downstream. Decide it early.

## Standing rules

- **Tune on the sandbox, freeze, then transfer.** λ_st, confidence-threshold schedule and
  class-mix rate get fixed on google→ign where target IoU is measurable, then applied
  unchanged to Jaipur.
- **Never set a threshold on the test set.** Fix it on val; state the protocol.
- **One variable per run.** The deltas are the contribution.
- Published numbers (DAFormer 68.3, HRDA 73.8, MIC 75.9) are **GTA→Cityscapes street
  scenes** — relative-ordering evidence only, never absolute targets for a binary
  aerial task.
