# Timeline — what was run, what wasn't, and in what order

*Generated 2026-09-10 09:38 by `scripts/build_timeline.py`. Do not edit by hand — rerun the script.*

This answers the question the other documents do not: **what was tried, in what order, and what came of it?** Months later, when writing up, the hard question is usually not "what did X score" but "did we ever actually test X, or did we just plan to?" — so §3 records what was **never run**, and why, as deliberately as §2 records what was.

| See also | For |
|---|---|
| [`README.md`](README.md) | results indexed **by question** |
| [`RUN_LEDGER.md`](RUN_LEDGER.md) | every run's **metrics + per-epoch curves** |
| [`BACKLOG.md`](BACKLOG.md) | the **queue** |
| [`../docs/sessions/`](../docs/sessions/) | per-day **narrative** |

---

## 1. Experiments, oldest first

| Date | Experiment | Status | Headline result |
|---|---|---|---|
| 2026-09-08 | [`2026-09-08-d6-seed-probe`](2026-09-08-d6-seed-probe/) | ✅ done | **5.69 %** foreground — ~5× below the D1 prior |
| 2026-09-09 | [`2026-09-09-boundary-relaxed-loss`](2026-09-09-boundary-relaxed-loss/) | ✅ done — **negative result** | **No** — 0.6366 vs 0.6475; +0.004 precision for −0.024 recall |
| 2026-09-09 | [`2026-09-09-d1-target-prior`](2026-09-09-d1-target-prior/) | ✅ done | **28.19 %** (23.06 % at conf ≥ 0.75) |
| 2026-09-09 | [`2026-09-09-d2-d3-source-stats`](2026-09-09-d2-d3-source-stats/) | ✅ done | AIRS **7.69 %** fg (assumed 15 %); buildings **23× larger in pixels** |
| 2026-09-09 | [`2026-09-09-does-the-airs-seed-help`](2026-09-09-does-the-airs-seed-help/) | ✅ done | **No** — ImageNet init 0.6483 vs seeded 0.6475, identical trajectory |
| 2026-09-09 | [`2026-09-09-method-comparison`](2026-09-09-method-comparison/) | ✅ done | **weak sup 0.65** vs seed 0.14; histmatch/FDA *harmful* |
| 2026-09-09 | [`2026-09-09-weak-supervision-jaipur`](2026-09-09-weak-supervision-jaipur/) | ✅ done | **IoU 0.6475** — 4.6× the unadapted seed |
| 2026-09-10 | [`2026-09-10-d4-adjacency`](2026-09-10-d4-adjacency/) | ✅ done | **78 %** do — instance-merging workstream justified |
| 2026-09-10 | [`2026-09-10-eroded-labels`](2026-09-10-eroded-labels/) | ✅ done | **Yes** — merge 0.46→0.33, count 0.76→**0.97** per building, −0.016 IoU |
| 2026-09-10 | [`2026-09-10-erosion-sweep`](2026-09-10-erosion-sweep/) | running | — |
| 2026-09-10 | [`2026-09-10-label-quantity-vs-quality`](2026-09-10-label-quantity-vs-quality/) | ✅ done — **confounded; see cross-eval** | **Unresolved** — each model wins on its own labels; needs ground truth |
| 2026-09-10 | [`2026-09-10-merge-split-rate`](2026-09-10-merge-split-rate/) | ✅ done | **50 % merged**, 21 % under-counted — invisible to IoU |
| 2026-09-10 | [`2026-09-10-segformer-backbone`](2026-09-10-segformer-backbone/) | ✅ done — modest win | **+0.0086** (0.6569) and **2× faster convergence**; gain is all precision |
| 2026-09-10 | [`2026-09-10-solar-google-to-ign`](2026-09-10-solar-google-to-ign/) | ✅ done | **0.8723 → 0.5611** (−31 pts); a capability drop, not miscalibration |

**14 experiments written up.** Status legend: ✅ done · ❌ negative result (kept deliberately) · ⚠️ confounded or unresolved · running.

---

## 2. Full commit history, newest first

81 commits. Each is a unit of work — a run launched, a result recorded, a bug found, a document corrected.

### 2026-09-10  ·  20 commits

- `09:20` **e00f501** docs: the push failure was a 280 MB file, not the network
- `09:16` **f8bf942** fix: gitignore all *.pth -- pattern missed checkpoints_mit/ and friends
- `09:11` **5c95100** feat: erosion sweep at 0.8 m -- find where splitting finally starts
- `09:07` **4cc2d7b** result: inference-time dilation backfires -- it trades misses back for merges
- `09:01` **7d016d3** result: erosion x architecture 2x2 -- the two fixes are synergistic, not redundant
- `08:58` **85caa45** docs: record the HTTP 408 push failure and the false-positive retry check
- `08:37` **129f570** result: eroding labels cuts merging 29% and nearly fixes the building count
- `06:45` **aa54221** feat: eroded-label 2x2 -- can shrinking labels stop the model fusing buildings?
- `06:38` **0fb8474** feat: --erode_m, to attack the 50% merge rate at the label level
- `06:36` **93bbf5a** result: solar domain gap is 31 points, and it is capability not calibration
- `06:09` **40fb5fd** result: half of all buildings are merged, and IoU never said so
- `06:02` **0f07118** result: D4 -- 78% of Jaipur buildings touch a neighbour
- `06:00` **1c20288** feat: D4 adjacency diagnostic; record that S3 is blocked on missing raw BDAPPV
- `05:49` **30936af** result: R3 label-confidence comparison is confounded -- cross-eval shows why
- `04:53` **6ad967b** result: MiT-B2 beats ResNet-34 by +0.0086 and converges twice as fast
- `02:37` **e7ee0a5** feat: R3 label quantity vs quality -- do the discarded 205k buildings matter?
- `02:14` **717e933** fix: relative symlinks for dataset splits -- absolute ones dangle in the container
- `02:09` **afe825c** feat: S1 solar google->ign, the only measurable domain gap
- `02:04` **eb33597** result: boundary relaxation at 4 px hurts -- negative result, kept
- `01:43` **eac6b6e** feat: R2 result -- the AIRS seed is worth nothing; launch R4 SegFormer backbone

### 2026-09-09  ·  16 commits

- `23:41` **ef5f9b9** feat: D2/D3 -- source prior measured; buildings are 23x smaller in pixels
- `23:17` **e735898** docs: operational pitfalls -- every bug and wrong conclusion from this session
- `23:10` **4b755f2** perf: cap OMP threads in run_docker.sh
- `23:02` **c6670c3** feat: --boundary_relax; launch R1 boundary-relaxed loss on GPU 2
- `22:56` **bd27e36** feat: run ledger -- keep every stat even when the weights are deleted
- `22:48` **a6517e0** feat: experiment backlog + 24/7 runner; launch the AIRS-seed ablation
- `22:45` **58aaf1f** feat: method comparison -- weak supervision wins 4.6x, photometric alignment harms
- `21:02` **bd427ce** docs: weak-supervision experiment, prediction recorded before results
- `20:53` **2fcfef0** feat: --init_weights for cross-dataset fine-tuning; scoped Drive fetches
- `20:43` **0ba5953** fix: enumerate Drive folders completely -- AIRS was there all along
- `20:10` **29da20b** docs: enumerate the master Drive folder — AIRS is not in it
- `19:48` **53c9f88** docs: withdraw C2, record provenance, add session note
- `19:46` **7ae6340** Merge remote-tracking branch 'origin/feat/init-project-setup' into feat/init-project-setup
- `19:32` **c192797** docs: add experiments/ run log, template and scaffold script
- `19:26` **114208d** fix: repair D1 reprojection and run it; reconcile docs with disk
- `00:04` **02852a0** feat: docker runner + D1/D6 diagnostics, reconciled master context

### 2026-08-04  ·  3 commits

- `16:19` **0421043** docs: link the domain-adaptation plan from README, add session notes
- `16:18` **db26b9e** docs: add Jaipur domain-adaptation research plan
- `16:18` **a81b4d3** chore: gitignore target-domain data, papers and reference rasters

### 2026-04-07  ·  1 commits

- `12:01` **a1bf351** feat: add training curve plots for rooftop and solar panel across all epochs

### 2026-04-06  ·  4 commits

- `22:37` **4b79fdf** Update README.md
- `11:50` **075e54f** fix: replace broken ResNet image with Figma-generated architecture diagram
- `11:40` **32c7bde** fix: use PNG thumbnail for ResNet SVG that was rendering without text
- `11:37` **3a60938** fix: replace broken image URLs in module READMEs

### 2026-04-05  ·  3 commits

- `10:43` **323a63f** docs: add detailed per-module READMEs for rooftop and solar panel stages
- `10:31` **a84f932** Merge pull request #4 from priyal2505/patch-1
- `10:29` **0d7bf43** Update README.md

### 2026-04-04  ·  3 commits

- `18:05` **68f75ea** added eval results 2
- `17:44` **9961ef0** added eval results
- `17:36` **76e83a8** added new logs

### 2026-04-03  ·  7 commits

- `21:41` **bbb5f43** feat: each training run saves checkpoints in its own timestamped folder
- `11:03` **568af43** feat: add evaluate_solar.py and infer_solar.py for solar panel stage
- `05:14` **b96ca9d** chore: track prep_bdappv.py move to solar_panel/
- `04:57` **497a294** added logs,checkpoints for 30 epochs , and added pre bdappv.py whcih prepares datasaet for training
- `01:19` **d60f7af** docs: update README with full project state and newbie quickstart
- `01:16` **80397fd** fix: resize inputs to 512×512 in solar augmentation pipeline
- `00:06` **debb019** feat: add .gitignore and update train_solar.py with RunLogger, tqdm, max_samples

### 2026-04-02  ·  8 commits

- `18:59` **ac95fcd** chore: force-add .gitkeep for gitignored data dirs, restore logs
- `18:54` **359456f** chore: add .gitkeep for empty data directories
- `18:51` **446ac98** chore: resolve .gitignore merge conflict, keep remote version
- `18:43` **aee5e05** refactor: restructure repo into rooftop/ and solar_panel/ modules
- `18:13` **0f95040** docs: add directory exploration commands for DGX terminal
- `12:15` **2b1799c** docs: add DGX Docker build/run commands and full usage guide to README
- `12:06` **9974069** refactor: reorganise project into rooftop/ and solar_panel/ modules
- `11:23` **5e6c54d** Add solar panel segmentation training pipeline

### 2026-04-01  ·  1 commits

- `18:32` **424e643** feat: add prep_bdappv.py for solar panel dataset preparation

### 2026-03-31  ·  3 commits

- `02:13` **a331212** feat: add --simulate_low_res augmentation for domain adaptation
- `01:49` **ada95f5** feat: rewrite infer.py with sliding-window inference for arbitrary image sizes
- `00:40` **c51faa0** feat: update evaluate.py with max_samples, EvalLogger, and overlay docs

### 2026-03-30  ·  12 commits

- `17:52` **443562a** logs: add DGX training run results (2000-sample, 100ep)
- `15:38` **5cd67ef** feat: add RunLogger to train.py + first DGX result log
- `13:42` **f730a14** feat: add --max_samples flag for quick subset training runs
- `13:30` **95165c8** fix: add per-batch tqdm progress bar to training loop
- `05:30` **22083fa** fix: add --img_subdir/--mask_subdir args to tile_airs.py
- `05:07` **bca4f56** fix: pin timm==0.9.2 to match smp==0.3.3 hard dependency
- `04:41` **c581e10** feat: add DGX training pipeline (train.py, evaluate.py, infer.py, tile_airs.py)
- `03:40` **bc02979** feat: add AIRS paper text and existing rooftop segmentation notebooks
- `02:40` **902c533** docs: add requirements, roadmap, research, and state
- `02:36` **617f286** docs: complete project research for rooftop & solar panel segmentation pipeline
- `01:58` **be736e3** chore: add project config
- `01:58` **53a5599** docs: initialize project

---

## 3. Queued and NOT run

The half of the record that is normally lost. An idea absent from this repo was either never had, or was had and forgotten — and there is no way to tell later.

- queued — R10 · Erosion sweep + inference-time dilation ★ next
- queued — R5 · Self-training / CBST on top of the weak model
- queued — R6 · Multi-source co-training
- queued — R7 · Low-resolution simulation
- queued — S2 · google → ign with self-training
- ⛔ **blocked** — S3 · Fix the zero-negatives bug (`MASTER_CONTEXT` C1) ⚠ BLOCKED — raw BDAPPV absent

---

## 4. How to regenerate

```bash
python scripts/build_run_ledger.py   # metrics first
python scripts/build_timeline.py     # then the timeline
```
