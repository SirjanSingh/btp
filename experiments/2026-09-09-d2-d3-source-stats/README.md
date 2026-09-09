# 2026-09-09-d2-d3-source-stats — the source domain, measured at last

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-09 |
| **Supersedes** | the `[ASSUMED] AIRS ~15%` in `MASTER_CONTEXT` §3.1 |

## Question

**D2** — what fraction of an AIRS image is actually building? **D3** — how big are buildings
in AIRS versus Jaipur, in m² and in pixels?

**Why it matters:** D1 measured the Jaipur prior (28.19%). The AIRS half stayed a guess, so
the project's central "trained on X%, deployed on Y%" claim was half estimate — and
`MASTER_CONTEXT` §11 forbids estimated priors in the report. D3 decides whether a 512² crop
and the receptive field are sized correctly for the target.

**Prediction:** AIRS ~15% (the planning-doc figure); prior shift therefore ~1.9×.

## Setup

All **857** AIRS train labels (271 MB, downloaded 2026-09-09 — masks only, no imagery, so
~1% of the storage a full AIRS pull would need). Connected components at 8-connectivity,
components under 2 m² dropped as labelling specks. Jaipur sizes come from Open Buildings'
supplied `area_in_meters`, so no rasterisation is involved.

## Results

Raw: [`diagnostics/d2_d3/d2_d3_summary.json`](../../diagnostics/d2_d3/d2_d3_summary.json)

### D2 — source foreground

| | AIRS (measured) | assumed | Jaipur (D1) |
|---|---|---|---|
| mean | **7.69 %** | ~15 % | 28.19 % |
| median | **2.06 %** | — | 23.12 % |
| p10 / p90 | 0.00 % / 23.00 % | — | — |
| range | 0.0 – 35.7 % | — | 10.7 – 41.2 % |

**Prior shift = 3.66×**, not the assumed 1.9×.

### D3 — building size

| | AIRS | Jaipur (OB) |
|---|---|---|
| median area | 118.6 m² | 64.7 m² |
| p10 / p90 | 17.5 / 262.8 m² | 16.8 / 195.8 m² |
| n components | 237,799 | 523,283 |
| **median area in PIXELS** | **21,084 px** (≈145×145) | **913 px** (≈30×30) |

A 512 px crop covers **38.4 m** of ground in AIRS and **136.3 m** in Jaipur.

## Interpretation

**1. Both assumed priors were ~2× too high, in the same direction — so the ratio survived
while both absolute numbers were wrong.** Assumed 15% → 50% gives 3.33×; measured 7.69% →
28.19% gives 3.66×. Anyone who had reasoned only about the *ratio* would have been roughly
right by accident. Anyone quoting the absolute numbers in a report would have been wrong
twice.

**2. The distribution shift is larger than the mean shift, and matters more.** AIRS has mean
7.69% but median **2.06%** and p10 **0.00%** — most tiles are nearly empty, a few are dense.
Jaipur's mean and median are both ~23%: uniformly built. So the seed was not trained on "a
sparser version of Jaipur"; it was trained on **mostly background, with buildings as rare
events**, then deployed where every crop is a quarter building. That is a sharper account of
the failure than the mean ratio gives, and it is what makes naive self-training dangerous —
the model's prior is "buildings are rare" and self-training would reinforce it.

**3. ★ The same building is 23× smaller in pixels.** Median 21,084 px in AIRS versus 913 px
in Jaipur — 145×145 versus 30×30. Jaipur buildings are genuinely smaller in metres (64.7 vs
118.6 m²) **and** imaged at 3.55× coarser GSD, and the two effects multiply. The model is
looking for objects roughly 23× larger than the ones present.

That is the strongest empirical argument yet for `plan/03`'s Tier 1.1 GSD resampling, which
it called "the highest value-per-effort action in the plan" — and it reframes the target: a
512 px crop that frames ~3 buildings in AIRS frames ~20 in Jaipur.

## Decision

- [x] Replace `[ASSUMED] AIRS ~15%` with **7.69%** everywhere.
- [x] Report the **distribution**, not just the mean — the median/p10 gap is the real story.
- [ ] Prioritise **R7 (low-resolution simulation)**: with a 23× pixel-area gap, matching the
      training GSD is likely worth more than any adaptation algorithm.
- [ ] Reconsider crop size for the target: 512 px frames a very different scene in each domain.

## Threats to validity

- AIRS labels are **roof outlines**, Open Buildings are **ground footprints** (Gap 4). The
  m² comparison is therefore not exactly like-for-like; footprints run slightly smaller than
  roofs, which inflates part of the 118.6 vs 64.7 difference. The *pixel* gap is dominated by
  GSD and survives this caveat.
- Components < 2 m² dropped. They are numerous and would drag the AIRS median down.
- Train split only; the val split may differ.
- Open Buildings' own recall in Jaipur is unknown — missed small buildings would bias its
  median upward.
