# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-30)

**Core value:** Given an aerial image, produce accurate rooftop masks and solar panel masks — and
translate pixel counts into actionable solar capacity estimates.
**Current focus:** Phase 1 — Data Foundation

## Current Position

Phase: 1 of 5 (Data Foundation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-30 — Roadmap created; Phase 1 is immediately actionable

Progress: [██░░░░░░░░] 0%

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

None yet.

### Blockers/Concerns

- AIRS data must be on Drive in correct folder structure before Phase 1 can begin — verify manually
- DGX Docker container must be set up and rclone tested before any DGX training run (Phase 2 blocker)
- Stage 2 canonical GSD (target: 0.1 m/pixel) needs empirical validation before committing to Phase 4

## Session Continuity

Last session: 2026-03-30
Stopped at: Roadmap and STATE.md written; no plans created yet
Resume file: None
