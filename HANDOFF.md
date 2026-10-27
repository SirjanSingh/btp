# Handoff: where the project stands (written 27 Oct 2026)

Read this first if you are picking up the project. It summarises what past work found, where
everything lives, and what is still open. Every number below comes from a file in this repo;
the source is named next to it.

---

## 1. Branches (updated: merged on 27 Oct 2026)

`feat/init-project-setup` has now been merged into `master`, so a fresh checkout of `master`
contains both the plan docs and the September experiments. The table below describes where each
set of work was done; before the merge the two branches had **diverged** and neither contained
the other. Keep running `git fetch --all` before you describe the project state, because a
stale checkout once made a session declare that `plan/` did not exist.

| Where the work was done | What it is |
|---|---|
| Started on `master` | `plan/` docs 01 to 11, including the SAM 3 plan (`09`), the Oct 2026 do/don't page (`10`), the domain adaptation baselines (`11`), and the reading list |
| Started on `feat/init-project-setup` | **All the September experiments** (168 commits, 9 to 12 Sep, run on the DGX): `experiments/`, `diagnostics/`, `scripts/`, `labelling/`, `docs/sessions/`, `docs/PITFALLS.md`, `.planning/MASTER_CONTEXT.md` |

Both are now on `master`. The old `feat/init-project-setup` branch is still on GitHub, but
nothing new needs to go there. Do new work on `master` (pull with `--rebase` before you push).
Where this file says "on the feat branch", read it as "on `master`".

To read a file from an older ref without switching branches (in Git Bash on Windows, set
`MSYS_NO_PATHCONV=1` first or the `ref:path` argument gets mangled):

```bash
git show origin/feat/init-project-setup:.planning/MASTER_CONTEXT.md
```

---

## 2. What the project is now

The goal changed during September. It is **no longer** "beat PSPNet IoU 0.899 on AIRS". It is:

> Estimate rooftop solar **energy** (kWh per year) for Jaipur, India, using models trained on
> foreign data (AIRS rooftops from Christchurch NZ at 7.5 cm/px, BDAPPV solar panels from
> France), with **zero hand-labelled target data**.

The main work is domain adaptation: make the AIRS rooftop model and the BDAPPV panel model work
on Jaipur, then turn pixels into area, capacity and energy. Segmentation is a means to that.
Open Buildings footprints (Google, 523,283 polygons over the Jaipur area) are used as weak
labels, so the method is best described as weakly supervised domain adaptation.

Jaipur imagery facts that matter (`plan/01`):
- 16 tiles, `map67_<col>-<row>.tif`, about 175 km², 8.2 GB, on Drive (link in `plan/README.md`).
- **26.6 cm/px on the ground, not 10 cm.** The files are Web Mercator, so multiply the stored
  pixel size by cos(latitude). Never hard-code a resolution; read it from the raster.
- That is 3.55 times coarser than AIRS (7.5 cm/px), which is the main source of domain shift.

---

## 3. What has been measured (as of 12 Sep, source: `experiments/` on the feat branch)

| Finding | Number |
|---|---|
| True building-pixel share of Jaipur (Open Buildings) | 28.2% |
| What the AIRS-trained model predicts on Jaipur | 5.7% (about 5 times too low) |
| Rooftop IoU vs Open Buildings, AIRS-only model | 0.142 |
| Same, trained on Open Buildings weak labels | **0.648** |
| FDA and histogram matching (image translation) | made it worse |
| AIRS pretraining vs plain ImageNet init | no difference |
| Self-training on Jaipur | rejected after 4 attempts |
| Self-training on BDAPPV google to ign, plain confidence threshold | 0.561 to 0.617 |
| Same, CBST ratio-matched thresholds | collapsed to 0.375 |
| First end-to-end energy figure | **4.59 GWp, 7.62 TWh/yr, plus or minus 15.2%** |

Current default model: U-Net with a MiT-B2 encoder, trained on Open Buildings footprints
**eroded by 0.4 m**. Erosion matters because 78% of Jaipur buildings touch a neighbour, and
without it the model merges half of them into blobs (invisible to IoU, fatal for per-building
estimates). The rule is "tune erosion until predicted count over label count is about 1", not
"always use 0.4 m".

**Caveat on every Jaipur number:** the "ground truth" is Open Buildings, which is also the
training label. Relative comparisons hold. Absolute accuracy has not been measured because no
hand labels exist.

**Run-to-run noise:** the first time a config was run on three seeds, IoU varied by about
plus or minus 0.004 but instance metrics (merge rate, predicted/label count) by about 20 times
more. Roughly twenty earlier comparisons used single runs. Run important comparisons on more
than one seed.

### Where the error in the energy number comes from

| Term | Share of the error bar |
|---|---|
| `k_usable` (fraction of roof area that can take panels, assumed 0.60) | 68% |
| Jaipur PVOUT (not yet looked up) | about 14% |
| Module efficiency (no module chosen yet) | about 11% |
| Segmentation | about 2% |

So better segmentation barely moves the answer. Measuring `k_usable` and getting any real
ground truth moves it a lot. Also, the energy chain must **not** multiply PVOUT by a
performance ratio: Global Solar Atlas PVOUT already includes those losses, and applying them
again underestimates by about 22%.

---

## 4. What is still open (priority order)

1. **No hand labels exist.** 20 + 10 crops were prepared in `labelling/` (brief in
   `labelling/TEAM_BRIEF.md` on the feat branch). None have come back. This blocks every
   absolute accuracy claim.
2. **`k_usable` has never been measured.** `labelling/SUPERSTRUCTURE_PASS.md` describes a pass
   over about 100 tiles marking water tanks, stairwell boxes, AC units and parapets. It was
   written but not sent to anyone.
3. **Jaipur PVOUT has not been read** from the Global Solar Atlas (a two-minute job).
4. **Stage 2 (solar panels on Jaipur) has not started.** There is also a known bug: the BDAPPV
   crop set contains **zero negative (no-panel) crops** because `prep_bdappv.py` skips images
   without masks. Every Stage 2 precision number so far measures the wrong task. The raw BDAPPV
   data is not on the DGX, so it has to be re-downloaded to fix this.
5. **The AIRS source set is incomplete.** Only about 75 labelled pairs are on Drive against the
   857 in the official split, and the 0.8784 baseline came from a deleted `/tmp`, so it is not
   reproducible. Either fetch real AIRS (about 28 GB from airs-dataset.com), switch source set,
   or lean fully on weak supervision, where the source set matters much less.
6. **Imagery licence is unresolved.** The Jaipur tiles look Google-sourced, and Google's terms
   forbid using Maps content to train, test or validate ML models. See `plan/07`. Labels are
   fine (Open Buildings is CC BY 4.0 or ODbL). The final report needs an explicit imagery
   provenance statement.

---

## 5. Next steps already planned (from `plan/10` and `plan/11`, 19 Oct)

- **Use the campus image `daraset/projs.tif` as a held-out test site only, never for training.**
  It shows the LNMIIT campus at 0.267 m/px, outside the 16 Jaipur tiles. Hand-label roofs for a
  first real IoU, mark unusable roof area for a measured `k_usable`, and ask the estate office
  for installed kWp and yearly kWh so a prediction can be compared with a meter. Details in
  `plan/10` section 2.
- **Baselines for the domain adaptation work** (`plan/11`): build on HRDA / DAFormer, add MIC as
  the strong standard baseline, ResiDualGAN as the scale-gap baseline, Kasmi 2025 for solar
  panels, and SAM 3 (`plan/09`) as the zero-adaptation row. HRDA needs its own Docker image
  (old mmsegmentation 0.16.0 and mmcv-full 1.3.7).
- **Jev 1.14** (a hosted image model) can act as a checker for "PV or water heater?", but it
  returns one box per question and cannot measure area. See `plan/10` section 3.

---

## 6. Reading order

1. This file.
2. `plan/10-do-and-dont-oct-2026.md` (current status and next actions).
3. `plan/11-domain-adaptation-baselines.md` (what to build on).
4. On the feat branch: `docs/sessions/2026-09-11-energy-chain-and-error-bars.md`, then the
   earlier session notes in `docs/sessions/`.
5. On the feat branch: `.planning/MASTER_CONTEXT.md`, section 0 first. It lists where earlier
   docs disagreed with the disk. `docs/PITFALLS.md` lists 15 traps already hit.
6. `plan/01` to `plan/08` for the original research and the licensing discussion.

---

## 7. Traps that have already cost time

- **Checking only one branch** (section 1). Always `git fetch --all` first.
- **`git add -A` can push gigabytes.** `data/`, `.pydeps/` and `.venv-docker/` are large. They
  are in `.gitignore` on the feat branch, along with `*.pth`; a 280 MB checkpoint once reached
  GitHub through a missing rule.
- **Do not rewrite history on a shared branch.** A `git filter-branch` run in September
  destroyed two DGX checkpoints. Only the metrics in `experiments/RUN_LEDGER.json` survived.
- **The DGX (`lnmdgx1`) runs git 1.8.3.1**, which has no `git -C`. It fails silently inside
  scripts. `/scratch/` is not persistent, so copy anything you need to keep.
- **Instance metrics are noisy.** See the seed-variance note in section 3.
- **Check CBST-style ratio thresholds before trusting them** when the domain gap is large. On
  Jaipur and BDAPPV they forced a near-zero threshold and the pseudo-labels became noise.
