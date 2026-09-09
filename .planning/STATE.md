# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-30)

**Core value:** Given an aerial image, produce accurate rooftop masks and solar panel masks — and
translate pixel counts into actionable solar capacity estimates.
**Current focus:** Phase 1 — Data Foundation (restore) + diagnostics D1–D7

> **This file is the GSD phase tracker and lags the code.** The authoritative, disk-reconciled
> picture is `.planning/MASTER_CONTEXT.md` (read its §0 first). The project has since been
> reframed as **unsupervised domain adaptation with zero target labels**; the Phase 1–5 roadmap
> below predates that framing (see MASTER_CONTEXT C4).

## Current Position

Phase: 1 of 5 (Data Foundation) — but Stage 1 and Stage 2 baselines already trained
Plan: 0 of TBD in current phase
Status: Blocked on the AIRS restore; diagnostics partially run
Last activity: 2026-09-09 — D1 run; MASTER_CONTEXT §0 reconciled against disk

Progress: [██░░░░░░░░] 0% *(plan-completion metric only; not a measure of work done)*

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:** No data yet

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: 5-phase coarse structure derived from feature dependency chain (data → loop → architecture → Stage 2 → demo)
- Phase 1: Split source images BEFORE tiling (official 857/94/96 split) to prevent train/val leakage
- Phase 1: Use rasterio for both image and mask to avoid spatial misalignment
- Phase 2: Differential LR — encoder 1e-4, decoder 1e-3 (SMP parameter group API)
- Phase 3: Stop encoder/decoder upgrade experiments once IoU > 0.900 on val set

### Pending Todos

1. **Decide the source domain — this is now a decision, not a download.** AIRS cannot be
   restored from Drive: it holds 75 labelled pairs against `train.txt`'s 857 (`plan/01` §2).
   Options: fetch real AIRS from airs-dataset.com (~28 GB), switch to Inria +
   SpaceNet-Khartoum (MASTER_CONTEXT §333 argues this on morphology *and* licensing), or
   commit to weak supervision on Open Buildings (`plan/` calls it the biggest win) where the
   source set matters much less. Blocks D2, D3, D5 and every training run.
2. **The 0.8784 baseline is not reproducible** — trained on 2,000 crops from a deleted
   `/tmp`. Re-establish it once the source domain is settled, before it is quoted anywhere.
3. Diagnostics D4 (Open Buildings half) and D7 (hand-inspection) are runnable now.
4. Fix the BDAPPV zero-negatives bug (MASTER_CONTEXT C1) before quoting any Stage 2 number.

### Blockers/Concerns

- AIRS data must be on Drive in correct folder structure before Phase 1 can begin — verify manually
- DGX Docker container must be set up and rclone tested before any DGX training run (Phase 2 blocker)
- Stage 2 canonical GSD (target: 0.1 m/pixel) needs empirical validation before committing to Phase 4

## Session Continuity

Last session: 2026-09-09
Stopped at: D1 run to completion (`diagnostics/d1/`); D6 already complete from 2026-09-08;
  MASTER_CONTEXT §0, `data/README.md` and this file reconciled against disk.
Resume file: `.planning/MASTER_CONTEXT.md` §6.3 (diagnostics status table)

### Session log

- **2026-09-08** — Datasets restored to `data/` (Jaipur 16 tiles ✅, Open Buildings ✅,
  AIRS ⚠️ interrupted at 50/857). `run_docker.sh` written (no `docker build`; deps in
  `.pydeps` on /home). `MASTER_CONTEXT.md` written and reconciled. D6 seed probe run over
  640 crops. D1 written; started but did not complete.
- **2026-09-09** — D1's per-tile reprojection replaced with a single batched transform;
  fixed a crash on the 2 MULTIPOLYGON rows that had left the previous run with no output.
  D1 run to completion. Docs above brought in line with disk.
