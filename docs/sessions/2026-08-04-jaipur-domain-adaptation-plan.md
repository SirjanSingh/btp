# Session — 2026-08-04 — Jaipur domain-adaptation plan

**Branch:** `feat/init-project-setup`
**Scope:** Research and plan the AIRS (Christchurch NZ) → Jaipur domain
adaptation. No model code written this session; deliverable is the `plan/`
folder.

---

## What changed

Added `plan/` — nine documents covering the full domain-adaptation effort:

| File | Contents |
|------|----------|
| `README.md` | TL;DR, reading order, recommended path, expected-outcome table |
| `01-situation-and-assets.md` | Measured inventory of models, imagery, and label sources |
| `02-domain-gap-analysis.md` | Six gaps (resolution, photometric, morphology, label semantics, spectral, solar), each quantified |
| `03-methods-and-literature.md` | Five method tiers with sources; explicit reject list |
| `04-pipeline.md` | End-to-end pipeline with diagrams; evaluation protocol |
| `05-compute-and-schedule.md` | 600 Colab CU budget, DGX plan, 6-week Gantt |
| `06-implementation-plan.md` | 13 TDD tasks with real code and pytest tests |
| `07-risks-licensing-ethics.md` | Risk register, imagery licensing, integrity checklist |
| `08-sources.md` | Every external source cited |

Also: `.gitignore` extended, `README.md` gained a domain-adaptation section
and a `plan/` entry in the repo tree.

---

## Decisions and why

**Predict roof outlines, not ground footprints.** AIRS labels roof outlines and
the project estimates rooftop *solar* potential, so the roof is the thing that
matters. Recorded because the free Indian label sources are footprints, and the
two diverge by ~8 px for a 4-storey building at this GSD.

**Weak supervision (Phase D) is the primary bet, not UDA.** Open Buildings +
Microsoft give hundreds of thousands of free Jaipur labels. For a deadline-bound
project that beats any clever unsupervised algorithm applied to a model that has
never seen an Indian roof. MIC(HRDA) is explicitly a stretch goal, off the
critical path, because it needs a separate pinned `mmsegmentation` environment.

**Add Inria (30 cm) and SpaceNet-Khartoum (30 cm) as co-training sources.**
Inria is near-GSD-matched to Jaipur; Khartoum is the closest public
morphological analogue (semi-arid, dense, low-rise). Nothing else in the
augmentation-only tier addresses the morphology gap.

**Checkpoint the deliverable after Run 3 (clean finetune).** Everything after
is upside. Avoids arriving at the deadline mid-experiment with nothing saved.

**Spatial splits, sealed test set.** With 10 % tile overlap, random tile splits
leak the same pixels into train and test. With 16 mosaic tiles available, hold
out 2–3 whole tiles rather than merely block-splitting.

---

## Findings that changed the project

1. **The Jaipur imagery is 26.6 cm/px, not 10 cm.** Read from the GeoTIFF tags:
   Web Mercator z19, `ModelPixelScale` 0.2985821 *Mercator* metres, corrected by
   `cos(26.958°)` → **0.26613 m/px**. Ground area per pixel is 0.0708 m², vs
   0.0056 m² for AIRS — a 12.6× difference that would have propagated into every
   capacity figure. Resolution gap to AIRS is **3.55×**, not 1.3×.

2. **The dataset is 16 tiles (~175 km², 8.2 GB), not 2 (~22 km²).** Drive folder
   `final_dataset` holds `map67_<col>-<row>.tif` for a 4 × 4 grid. Grid
   orientation derived from the two local tiles' tiepoints: same Mercator *x*,
   *y* differing by exactly one tile height, so index 1 = column (east),
   index 2 = row (south), abutting with no overlap. Full mosaic
   44,672 × 55,424 px → **≈ 11,140** training tiles.

3. **Disk space is a hard blocker.** C: 1.6 GB free, D: 7.8 GB free, G: 43.8 GB
   free. End-to-end footprint is 18–27 GB. The dataset must be staged on G: or
   pulled directly to the DGX — it cannot live on D:.

4. **The Colab SAM2 budget did not survive the scale-up.** Refining all 11,140
   tiles would cost ~1,300 CU against a 600 CU balance. Revised to refine only
   what training consumes (400 clean + ~2,500 stratified weak ≈ 290 CU).

5. **Google Maps terms prohibit using Maps content to train, test, validate or
   fine-tune ML models.** Unresolved — needs a provenance decision before this
   is submitted or published. Three options documented in `plan/07`.

6. **`daraset/projs.tif`** (3924 × 2092, origin 75.918 °E / 26.938 °N) sits
   *outside* the 4 × 4 grid, to its east. Provenance unknown; flagged, not used.

---

## Verification

- GSD, grid orientation, tile dimensions and extents: computed directly from
  the TIFF `ModelPixelScale` / `ModelTiepoint` tags via PIL, not assumed.
- Tile counts: `(11168-512)//461+1 = 24` cols × `(13856-512)//461+1 = 29` rows
  = 696 per file, × 16 = 11,136.
- Disk space: `Get-PSDrive` on all filesystem drives.
- `.gitignore` change verified with `git status --porcelain` — `daraset/`,
  `airs.pdf`, `christchurch_370.tif` and eval PNGs no longer appear.
- Every external claim in `plan/` traced to a source in `plan/08-sources.md`.
- **Not verified:** no code was run against real data; no model was trained; no
  Jaipur IoU number exists yet. All figures in the expected-outcome table are
  planning targets, explicitly labelled as such.

---

## Open items for next session

1. **Decide imagery provenance** (`plan/07` §1) — blocks publication, not
   development.
2. **Stage the dataset on G: or the DGX.** Do not download to D:.
3. Download the remaining 14 tiles from the Drive folder.
4. Start Task 0–2 of `plan/06`: gitignore is done; next is `adapt/geo.py` and
   `adapt/tile_jaipur.py` with their tests.
5. Check Open Buildings coverage over Jaipur *before* building the whole label
   engine — visualise 5 tiles' rasterised footprints over the imagery first.
6. Measure actual Colab CU burn rates; published figures disagree by up to 3×.
7. Begin hand-labelling early — 400 tiles, 13–17 h, on the critical path.
8. Decide the branch cleanup (see *Repo topology findings* below) — retire
   `feat/init-project-setup` and work on `master`, or fix its upstream.

---

## Merge to master (end of session)

`feat/init-project-setup` was merged into `master` with `--no-ff` as `951a990`
and pushed. Clean merge, no conflicts.

**Divergence before the merge was 3 behind / 9 ahead**, and the three "behind"
commits turned out to be nothing but the merge bubbles from PRs #1, #3 and #5 —
each of which had merged *this same branch* into master. No independent work had
ever landed on master. `git diff HEAD..origin/master` showed 4,725 deletions and
a single insertion, and that insertion was a trailing space on the README title
line. After merging, `git diff master feat/init-project-setup` is empty.

19 files, +4,725 lines: `plan/` (9 docs), `docs/sessions/`, the per-module
READMEs, `plot_training.py`, the training-curve assets, and the `.gitignore`
extension.

## Repo topology findings

Three things worth knowing before the next session:

1. **`main` is an orphan branch.** It shares *no* history with `master` —
   separate root commit (`2770da4 "chore: initial empty main branch"`) and a
   completely empty tree. Tooling reports `main` as the repository's default
   branch, so anything that auto-targets the default branch will point at an
   empty branch. **The real trunk is `master`.** Either delete `main` or set the
   GitHub default branch to `master`.

2. **`claude/solar-panel-segmentation-qyLv7` has unmerged commits.**
   `0397a14 "Add solar panel segmentation training pipeline"` is *not* an
   ancestor of master. It may be superseded by the current `solar_panel/` code,
   or it may hold something that was lost track of. Inspect before deleting.

3. **`feat/init-project-setup` tracks `origin/master`.** A bare `git push` from
   that branch targets master. It was pushed explicitly by name this session, so
   nothing went astray. Now that the branch is fully merged and 0 ahead, the
   cleanest fixes are `git branch -u origin/feat/init-project-setup` to keep it,
   or retire it and work directly on `master`. **Not yet decided.**
