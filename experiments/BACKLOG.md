# Experiment backlog — the queue the 30-minute loop pulls from

Top unblocked item wins. Move an entry to `## Done` with its result when it lands, and
write the full entry under `experiments/<date>-<slug>/`.

**Launch rules** (see [`docs/PITFALLS.md`](../docs/PITFALLS.md) for why each exists)
- Only use a GPU whose **free VRAM ≥ job need + 2 GB margin**. Our runs take ~6 GB at
  batch 16, so ~8 GB free is the bar. Never evict or disrupt another student's job.
- **Quota guard:** `/home` is a hard 40 GB. If free space < 3 GB, do not launch — delete
  `epoch*.pth` first (never any `best.pth`), then re-check.
- New runs should pass `--save_every 999` so only `best.pth` is written.
- **Never delete a checkpoint before running `python scripts/build_run_ledger.py`.** Weights
  are regenerable; the numbers inside them are what the paper needs. The ledger consolidates
  every run's config, hardware and full per-epoch curve into `experiments/RUN_LEDGER.{md,json}`
  so the stats outlive the `.pth`. Rebuild it after every run too.
- **Two concurrent jobs, not three.** Measured twice on this box: at three jobs load average
  sits above 300 on 80 cores and throughput collapses (~1 epoch per 29-min tick). Two runs
  sustain roughly 5 epochs per tick. GPU VRAM is not the binding constraint; CPU is.
- Every run: prediction written **before** launch, results after, index updated, pushed.

---

## Queue — rooftop (Stage 1)

### R11 · Fix the split-rate metric ★★ it is silently blind
A label counts as "split" only if >= 2 predictions each cover >= 50 % of it, so fragments
smaller than half a building never qualify. At 0.8 m erosion the model shattered buildings
badly enough to reach pred/label 1.4743 while split rate read **0.0**. Every split rate in
this repo is suspect. Count a label as split if >= 2 predictions overlap it at all, and
report fragments-per-label alongside.
*Needs:* small change in `scripts/merge_split_rate.py`, then re-run the four saved
checkpoints. CPU + brief GPU.




### R5 · Self-training / CBST on top of the weak model
Pseudo-label Jaipur with the weak model, keep confident pixels using the **measured** 23 %
class ratio (not a fixed 0.95), retrain. This is the Tier-3 UDA arm the supervisor's
CVF-venue list wants.
*Needs:* new script. ~3 h GPU. **Predict: +0.01–0.04; collapse risk if the ratio is wrong.**

### R6 · Multi-source co-training
Add the 2,326 labelled American-house pairs sitting unused in Drive. Tests `plan/03` §2.2
(a second source domain) with data we already have rather than downloading Inria.
*Needs:* download ~2 GB, quota permitting. ~3 h GPU. **Predict: +0.01–0.03.**

### R7 · Low-resolution simulation
Train on AIRS downsampled to 0.15–0.40 m so the training distribution matches Jaipur's
26.6 cm. `plan/03` calls this the highest value-per-effort action; `--simulate_low_res`
already exists.
*Blocked:* needs AIRS imagery (~14 GB). Use the container-`/tmp` route so the quota never
sees it — but write `make_airs_crops.py` first, so the crops are regenerable. The 2026-03
baseline is unreproducible precisely because that script never existed.

---

## Queue — solar (Stage 2)

### S2 · google → ign with self-training
The same CBST recipe as R5, tuned where the score is visible.
*Needs:* S1 first.

### S3 · Fix the zero-negatives bug (`MASTER_CONTEXT` C1) ⚠ BLOCKED — raw BDAPPV absent
`prep_bdappv.py:85` drops mask-less images, so **every** training crop contains a panel and
the model never learns "no panel here". Re-prep keeping negatives, then re-run S1.
**Blocked 2026-09-10:** `solar_panel/bdappv/` is **empty** — only the derived crops survive,
so there is nothing to re-prep from. Fixing C1 requires **re-downloading BDAPPV** from source
first. Until then every Stage-2 precision number, S1's included, measures the wrong task.

---

## Queue — diagnostics (CPU, cheap)

- **D7 · clutter inventory** — build a contact sheet of 20 tiles for manual tallying.

---

## Done

| ID | Result |
|---|---|
| D1 target prior | 28.19 % (23.06 % @ conf ≥ 0.75) |
| D6 seed probe | 5.69 % predicted foreground — ~5× under |
| Weak supervision | **IoU 0.6475** |
| **Erosion sweep (0.8 m)** | over-erodes: merge 0.1339 but pred/label **1.47**, missed 0.47 — 0.4 m is the optimum |
| **Eroded labels (0.4 m)** | merge 0.4615→**0.3256**, pred/label 0.760→**0.9745**, −0.016 IoU; synergistic with MiT |
| **S1 solar google→ign** | source 0.8723 → **target 0.5611**; best thr 0.5 on both, so not calibration |
| **R8 merge/split rate** | **50 % of buildings merged** at IoU 0.648; MiT-B2 4.3 pts better; split rate 0 |
| **D4 adjacency** | **78 %** of buildings touch a neighbour — merging workstream justified |
| **R3 label confidence** | ⚠️ confounded — 0.6281 vs 0.6483 on shared val, but each model wins on its own labels |
| **R4 MiT-B2 backbone** | ✅ 0.6569 vs 0.6483 — +0.0086, 2x faster convergence, gain all precision |
| **R1 boundary relax (4 px)** | ❌ 0.6366 vs 0.6475 — band removes signal, not just noise |
| **R2 AIRS seed ablation** | **ImageNet init 0.6483 vs AIRS-seeded 0.6475 — seed worth nothing** |
| D2 source prior | **7.69 %** mean / 2.06 % median (assumed 15 %) — shift 3.66× |
| D3 building size | AIRS 21,084 px vs Jaipur 913 px — **23× smaller** |
| Method comparison | weak 0.65 ≫ adabn 0.25 > seed 0.14 > fda 0.09 > histmatch 0.03 |
