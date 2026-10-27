# 2026-09-09-d1-target-prior — Jaipur's true building-pixel prior is 28.2 %, not the assumed 50 %

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-09 |
| **Commit** | `114208d` on `feat/init-project-setup` |
| **Supersedes** | the `[ASSUMED] ~50 %` in `MASTER_CONTEXT` §3.1 |

## Question

Diagnostic **D1** (`MASTER_CONTEXT` §6.3). Rasterise Google Open Buildings polygons over the
Jaipur mosaic footprint and measure what fraction of pixels are actually buildings.

**Why it matters, three ways:**

1. It sets the **CBST class ratio** for self-training. Guessing it wrong biases every
   pseudo-label round.
2. Paired with D6 it is the **motivating figure** of the project.
3. `MASTER_CONTEXT` §11 explicitly forbids putting estimated priors in the report.

**Prediction:** the planning docs assumed ~50 %, on the intuition that dense Indian wards are
near-fully built. Expectation going in was 40–55 %.

## Setup

| | |
|---|---|
| Data | `data/open_buildings/jaipur_open_buildings.csv` — Open Buildings v3 (CC BY 4.0), **523,283** polygons, WKT EPSG:4326. Confidence min 0.650 / median 0.773 / max 0.977; area median 64.7 m² |
| Tiles | `data/jaipur/` — 16 GeoTIFFs, EPSG:3857 |
| Method | reproject 4326→3857 once, per-tile bbox prefilter, rasterise at `--downscale 8` |
| Hardware | CPU-only |
| Runtime | ~10 min, dominated by the reprojection |

```bash
./run_docker.sh "" "python -u scripts/d1_target_prior.py --cutoffs 0.0 0.6 0.75 --out_dir diagnostics/d1"
```

## Results

Raw output: [`diagnostics/d1/d1_summary.json`](../../diagnostics/d1/d1_summary.json)

| confidence cutoff | buildings kept | **overall prior** | per-tile range |
|---|---|---|---|
| ≥ 0.0 | 523,283 | **28.19 %** | 10.70 – 41.23 % |
| ≥ 0.6 | 523,283 | 28.19 % | 10.70 – 41.23 % |
| ≥ 0.75 | 318,207 | **23.06 %** | 9.06 – 31.39 % |

### Against D6 — the headline

| | |
|---|---|
| True building-pixel prior (D1) | **28.19 %** |
| Seed ckpt predicts (D6, thr 0.35) | **5.69 %** |
| **Under-prediction** | **≈ 5×** (≈ 4× against the conservative 23.06 %) |

## Interpretation

**The assumed 50 % was nearly 2× too high.** Worth noting *why*: footprint density in a dense
city is not the same as the visual impression of one. Any design decision that was tuned
against 50 % should be revisited.

**The ~5× under-prediction is the gap the adaptation exists to close**, and it is now a
measured number rather than an argued one. It also gives run-table row R10 its kill-switch: if
pseudo-label foreground fraction drifts far from ~28 %, stop the run.

**The per-tile spread matters more than the mean.** 10.7 % → 41.2 % across 16 adjacent tiles
is a 4× swing inside one city. This is direct evidence for reporting `k_usable` as a measured
distribution per density bin rather than the assumed scalar 0.60 (§7.4), which swings the
headline GW figure by ±25 %.

## Decision

- [x] Replace the `[ASSUMED] ~50 %` in `MASTER_CONTEXT` §3.1 with the measured figure.
- [x] Use **28.19 %** as the CBST target ratio; keep 23.06 % as the conservative bound.
- [ ] Report `k_usable` per density bin, using the per-tile spread as justification.
- [ ] D2 (AIRS source prior) still needed for the other half of the shift — **blocked on the
      unfinished AIRS restore**, which leaves the "~15 % AIRS" side still assumed.

## Threats to validity

- **Ground footprints, not roof outlines** (`MASTER_CONTEXT` Gap 4). Off-nadir on a 4-storey
  Jaipur building these differ by ~8 px. This is a sound *density* prior and **not** a
  pixel-accurate roof mask — do not use it as ground truth.
- **The confidence sweep is partly degenerate.** The export is pre-filtered at ≥ 0.65, so
  cutoffs 0.0 and 0.6 return all 523,283 polygons and an identical number. Only cutoffs above
  0.65 move it. The three-row table above looks like a sweep but is really two points.
- Rasterised at `--downscale 8`. Fine for a ratio; it would understate thin structures if the
  output were used as a mask.
- Open Buildings' own recall is unknown for this AOI. Missed buildings bias the prior **down**,
  so 28.19 % is more likely a floor than a ceiling — which makes the under-prediction gap
  conservative.

## Reproduce

Needs `data/jaipur/` and `data/open_buildings/` (both restored, both gitignored).

**Known failure mode, fixed in `114208d`:** the AOI contains **2 MULTIPOLYGON rows among
523,281 POLYGONs**. Their `coordinates` nest one level deeper, so a hand-rolled
`for ring in coords for c in ring` bbox walk yields coordinate pairs instead of scalars and
`np.asarray` raises `inhomogeneous shape` — *after* the ~10-minute reprojection has already
run. Bounds now come from `shapely.geometry.shape(...).bounds`, which is type-agnostic.
