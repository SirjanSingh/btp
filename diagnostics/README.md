# diagnostics/ — measured numbers that replace planning assumptions

Each `D<n>/` holds the output of one diagnostic from `.planning/MASTER_CONTEXT.md` §6.3.
Anything in here is **measured on this machine**; nothing in here is an estimate. The point
of the whole set is that the master context forbids putting assumed priors in the report.

| # | Status | Output | Script |
|---|---|---|---|
| D1 | ✅ 2026-09-09 | `d1/d1_summary.json` | `scripts/d1_target_prior.py` |
| D2 | ⛔ needs AIRS crops | — | — |
| D3 | ⛔ AIRS half blocked | — | — |
| D4 | 🟡 OB half runnable | — | — |
| D5 | ⛔ needs AIRS crops | — | — |
| D6 | ✅ 2026-09-08 | `d6/d6_summary.json` (`d6_smoke/` = 96-crop dry run) | `scripts/d6_seed_probe.py` |
| D7 | 🟡 manual, not started | — | — |

## The headline: D1 × D6

|  | measured |
|---|---|
| Jaipur building-pixel prior, all polygons (D1) | **28.19 %** (per-tile 10.70 – 41.23 %) |
| same, confidence ≥ 0.75 (318,207 of 523,283) | **23.06 %** (per-tile 9.06 – 31.39 %) |
| Seed ckpt predicted foreground on Jaipur (D6, thr 0.35) | **5.69 %** (median 4.25 %, 11.6 % of crops empty) |

**The AIRS-trained seed under-predicts target foreground by ~5×.** That gap is the project's
motivating figure, and it is what the domain adaptation has to close. It also sets the CBST
class ratio for self-training, and gives R10 its kill-switch: if pseudo-label foreground
fraction diverges from ~28 %, stop the run.

## Reading D1 carefully

- **Ground footprints, not roof outlines.** Open Buildings labels the building's footprint on
  the ground; this project predicts the roof. Off-nadir on a 4-storey Jaipur building those
  differ by ~8 px (MASTER_CONTEXT Gap 4, §3.4). D1 is a good *density* prior and **not** a
  pixel-accurate roof mask — do not use it as ground truth.
- **The confidence sweep is partly degenerate.** The AOI export is already filtered at
  confidence ≥ 0.65 (min 0.650, median 0.773, max 0.977), so any cutoff at or below 0.65
  returns all 523,283 polygons and gives an identical prior. Only cutoffs above 0.65
  (e.g. 0.75) actually move the number.
- **Rasterised at `--downscale 8`.** The prior is a ratio, so a full 11168×13856 burn per
  tile per cutoff buys nothing. Change `--downscale` if you need per-tile masks rather than
  the ratio.
- **Tile spread matters more than the mean.** 10.7 % to 41.2 % across 16 adjacent tiles is
  a 4× swing within one city, which is why `k_usable` should be reported per density bin
  rather than as the assumed scalar 0.60 (§7.4).

## Reproducing

Both run CPU-only inside the standard container; D6 wants a GPU.

```bash
./run_docker.sh "" "python -u scripts/d1_target_prior.py --cutoffs 0.0 0.75 --out_dir diagnostics/d1"
./run_docker.sh 3  "python -u scripts/d6_seed_probe.py --out_dir diagnostics/d6"
```

D1 takes ~10 min, dominated by reprojecting 523k polygons EPSG:4326 → 3857.
