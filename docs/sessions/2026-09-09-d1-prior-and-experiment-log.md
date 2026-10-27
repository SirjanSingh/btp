# Session — 2026-09-09 — D1 target prior, and an experiment log

**Branch:** `feat/init-project-setup`
**Scope:** Finish what the 2026-09-08 session left hanging, then set up a
convention so the next handover does not need this kind of archaeology.
Covers the tail of 2026-09-08 as well, which ended mid-run without a note.

---

## What changed

| Area | Change |
|------|--------|
| `scripts/d1_target_prior.py` | Fixed a crash, ran D1 to completion |
| `diagnostics/d1/` | New — `d1_summary.json`, the measured target prior |
| `diagnostics/README.md` | New — status of D1–D7, the D1×D6 headline, how to reproduce |
| `experiments/` | New — run log, `TEMPLATE.md`, backfilled D1 and D6 write-ups |
| `scripts/new_experiment.sh` | New — scaffolds an entry with date/branch/SHA stamped |
| `.planning/MASTER_CONTEXT.md` | §0 C2 **withdrawn**, C3 resolved, C5 rewritten, C6 added; measured prior replaces the assumed one |
| `.planning/STATE.md` | Was "Phase 1, 0%, last activity 2026-03-30"; now current, with a session log |
| `data/README.md` | Documents `jaipur/` and `open_buildings/`; records restore status and provenance |
| `.gitignore` | `data/`, `.pydeps/`, `.venv-docker/`, `.tmp/`; checkpoints at any depth |

---

## The result

**Jaipur's true building-pixel prior is 28.19 %** (23.06 % keeping only Open Buildings
confidence ≥ 0.75), per-tile range 10.70 – 41.23 % across the 16 tiles.

Against D6's **5.69 %** predicted foreground from the AIRS-trained seed, that is a **~5×
under-prediction**. This is the project's motivating figure, and it is now measured rather
than argued. Full write-ups in [`experiments/`](../../experiments/).

Two consequences worth carrying forward:

- **The assumed ~50 % was nearly 2× too high.** Footprint density in a dense city is not the
  same as the visual impression of one. Anything tuned against 50 % should be revisited.
- **The per-tile spread is 4× within one city.** That is direct evidence for reporting
  `k_usable` as a measured distribution per density bin rather than the assumed scalar 0.60,
  which swings the headline GW figure by ±25 %.

---

## Findings that changed the project

**C2 was wrong, and the way it was wrong matters.** `MASTER_CONTEXT` §0 declared that `plan/`
does not exist and that its ~20 cross-references resolve to nothing. `plan/` exists — 9
documents, 3,655 lines, including a 1,941-line implementation plan — committed in `db26b9e`
on 2026-08-04. The check had been run against a local checkout **11 commits behind the
remote**, with no `git fetch` first.

The lesson is now written into §0 as a method note: **"not on this machine" and "does not
exist" are different claims**, and a reconciliation pass that conflates them produces
confident, wrong corrections. A document whose whole purpose is correcting other documents
has to hold itself to a higher standard of checking than this.

**D1 was not merely unrun — it was broken.** The 2026-09-08 session rewrote the reprojection
to batch it, started the run, and ended. The rewrite crashes: the AOI holds **2 MULTIPOLYGON
rows among 523,281 POLYGONs**, whose `coordinates` nest one level deeper, so the hand-rolled
bbox walk yields coordinate pairs instead of scalars and `np.asarray` raises on a ragged
array — *after* the ~10-minute reprojection has already run. Bounds now come from
`shapely.geometry.shape(...).bounds`, which is type-agnostic.

**`git add -A` would have pushed 9 GB.** `data/`, `.pydeps/` and `.venv-docker/` were
untracked but unignored, and `data/README.md` claimed a gitignore rule that did not exist.
This is the second time this repo has hit this (see `a81b4d3`, 2026-08-04).

**`git -C` does not exist here.** This node runs git 1.8.3.1; the flag arrived in 1.8.5. It
fails *silently* in a script, which stamped every scaffolded experiment `UNCOMMITTED` on
branch `unknown`. `new_experiment.sh` uses a subshell `cd` instead.

---

## Decisions and why

**An experiment log, separate from the session notes.** Sessions record *what happened on a
day*; experiments record *what a run measured*. The thesis needs the second, indexed by
question rather than by date. `experiments/README.md` holds the index, one folder per run.

**The template demands the prediction before the run, and a threats-to-validity section.**
An expectation written down afterwards is a rationalisation, and a surprise is only visible
if the expectation was recorded first. The caveats section is what a viva actually probes —
D1's is where the ground-footprint-vs-roof-outline mismatch and the degenerate confidence
sweep live, rather than in a footnote.

**Write-ups link to raw output rather than restating it,** so there is exactly one copy of
every number. Diagnostics keep their home under `diagnostics/` so the master context's
cross-references stay valid.

**Negative and abandoned results stay in the index,** marked as such. Deleting a failed run
is how a project repeats it six weeks later.

---

## Verification

- **Verified:** D1 ran to completion on all 16 tiles and 523,283 polygons; output committed.
  Tile CRS confirmed EPSG:3857 before trusting the hard-coded reprojection. The MULTIPOLYGON
  fix was smoke-tested on a 2,002-row subset containing both such rows before the full run.
  `new_experiment.sh` was run and its output inspected, then removed.
- **Not verified:** no model was trained or evaluated this session. **No Jaipur IoU number
  exists and cannot exist** — there are no target labels. D6's figure is what the model
  *predicts*, not how accurate it is.
- **Superseded numbers:** the assumed "~50 % Jaipur prior" is dead. The "~15 % AIRS" figure
  is **still assumed** — D2 needs the AIRS crops.

---

## Open items for next session

1. **Finish the AIRS restore** — the single critical path. 50 of 857 images, and only 3
   image/label pairs actually match (`christchurch_15`, `_48`, `_77`). Blocks D2, D3, D5 and
   every training run. `fetch_drive_folder.py` is resumable.
2. **Record the AIRS source link in `plan/01` §2**, next to the Jaipur `final_dataset` one.
   It is written down nowhere, which is why item 1 is currently un-runnable by anyone else.
3. **Read `plan/06-implementation-plan.md`.** 1,941 lines of task-by-task plan that the last
   two sessions worked without, having concluded it did not exist.
4. **Reconcile `plan/` against `.planning/` and `MASTER_CONTEXT`** — three planning documents
   with different phase decompositions (Phases A–F, 1–5, and A–E). Pick one.
5. D4 (Open Buildings half) and D7 are runnable now; D2/D3/D5 wait on item 1.
6. Fix the BDAPPV zero-negatives bug (`MASTER_CONTEXT` C1) before quoting any Stage 2 number.
7. Still open from 2026-08-04: decide imagery provenance (`plan/07` §1) — the Google Maps ToS
   issue blocks publication, not development.

### Closed from the 2026-08-04 list

- ~~Download the remaining 14 tiles~~ — all 16 are on the DGX at `data/jaipur/`.
- ~~Check Open Buildings coverage before building the label engine~~ — D1 measures it
  (28.19 %, 523,283 polygons). The *visual* check over 5 tiles is still worth doing before
  trusting the footprints as weak labels.
- ~~`gitignore` for target-domain data~~ — done in `a81b4d3`, extended this session.

## Repo note

The upstream-branch problem flagged on 2026-08-04 is **still not fixed**:
`feat/init-project-setup` still tracks `origin/master`, so a bare `git push` from this branch
targets master. Pushed explicitly again this session. Fix with
`git branch --set-upstream-to=origin/feat/init-project-setup`.

The branch had also diverged — 11 remote commits were never pulled locally, which is the root
cause of the C2 error above. Merged this session.
