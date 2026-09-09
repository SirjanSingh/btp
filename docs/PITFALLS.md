# Operational pitfalls — bugs, traps and mistakes, with the rule each one taught

Engineering and process traps hit while working on this repo and this server.
Distinct from [`.planning/research/PITFALLS.md`](../.planning/research/PITFALLS.md), which
covers *domain* pitfalls (segmentation, imagery, labels).

Every entry here cost real time at least once. Several produced **confident, wrong
conclusions that were written into planning documents** and had to be withdrawn later —
those are the expensive ones, and they are listed first.

---

## 1. Reasoning failures — the expensive category

### 1.1 "Not on this machine" ≠ "does not exist"
`MASTER_CONTEXT` §0 C2 declared the `plan/` folder non-existent and its ~20 cross-references
dangling. `plan/` had **9 documents and 3,655 lines**, including a 1,941-line implementation
plan. The check had run against a local checkout **11 commits behind the remote**, with no
`git fetch` first. Two working sessions then proceeded without a detailed plan they believed
did not exist.

> **Rule: `git fetch` before reconciling anything against "the disk".** A document whose job
> is correcting other documents must check itself harder than the documents it corrects.

### 1.2 A verification that cannot fail is not a verification ★
`gdown` reported exactly 50 files in every Drive folder. Suspecting a cap, the check was to
raise `gdown.download_folder.MAX_NUMBER_FILES` from 50 to 100,000 and re-run. Counts did not
move, so the 50s were declared real and "the full AIRS dataset is NOT in Drive and never
was" was committed to three planning documents.

`MAX_NUMBER_FILES` **only gates a warning**. It cannot change how many files are fetched, so
that experiment could not have detected the bug under any circumstance. The folder held
**857 of 857** AIRS training images all along. The right suspicion was reached and then
argued away with a null test.

> **Rule: before trusting a negative result, ask what would have happened if the hypothesis
> were true.** If the answer is "the same thing", the test is worthless.

### 1.3 Check the layer the question is about
Asked what two tasks the UI showed as running, the reply checked processes and containers,
found nothing, and said "nothing of ours is running". Correct about the machine, wrong about
the question — the entries were stale rows in the task registry, which is where the user was
looking. It took a screenshot to find that out.

> **Rule: answer at the layer the question was asked at, then widen.**

### 1.4 Read the quota, not the filesystem
`df -h /home` shows **1.2 TB free**. The actual limit is a per-user quota:

```
$ quota -s
/dev/sdb1  39815M  quota 40960M   <- 97% full, ~1.1 GB left
```

Acting on `df`, a 14 GB download was started and training was left writing 280 MB checkpoints
against 1.1 GB of headroom. The run would have died around epoch 30.

> **Rule: on any shared machine, `quota -s` is the number that matters. `df` is a lie about
> what you may use.**

### 1.5 Don't delete before recording
Checkpoints were deleted to free quota, *then* it was verified that their statistics survived
in the JSON logs. They did — but that was luck, not process. `.pth` weights are regenerable;
the numbers inside them are what a paper needs.

> **Rule: `python scripts/build_run_ledger.py` **before** deleting any checkpoint.** Now in
> the backlog launch rules.

### 1.6 Two slow jobs are worse than one fast one
Two training runs were launched concurrently on free GPUs. Load average hit **292 on 80
cores** and each epoch slowed ~8× (3 min → 25 min). Neither job was broken; both were
starved. Throughput was worse than running them serially, and it degraded the box for 13
other users.

> **Rule: "GPU is free" does not mean "capacity is free". Check load average and CPU
> contention, not just VRAM.**

---

## 2. Environment traps — this server specifically

| Trap | Detail |
|---|---|
| **`git` is 1.8.3.1** | No `git -C` (added in 1.8.5) — and it **fails silently inside scripts**, which stamped every scaffolded experiment `UNCOMMITTED` on branch `unknown`. Also missing: `git branch --show-current`, `git status --cached`. Use a subshell `cd`, `rev-parse --abbrev-ref HEAD`, `diff --cached`. |
| **`/home` quota is 40 GB hard** | See 1.4. `df` reports the 7 TB filesystem. |
| **`/` is 96–99% full** | 442 GB disk, ~19 GB free. ~120 GB of it is *other users'* container writable layers. |
| **`OMP_NUM_THREADS` defaults to 80** | Torch spawns one CPU thread per core **per process**. Capped at 4 in `run_docker.sh`; override with `OMP_THREADS=n`. |
| **`/scratch`, `/raid`, `/data` are not writable** | They exist but are plain directories on `/`, not separate mounts. |
| **The container-`/tmp` quota trick no longer works** | `run_docker.sh` bind-mounts `.tmp` (on `/home`) to container `/tmp`, so writes land in the quota. Without that mount it *does* bypass the quota — but see 3.6. |

---

## 3. Tooling and code bugs found

### 3.1 gdown silently truncates folder listings at ~50 files ★
`gdown.download_folder()` **scrapes the folder's rendered HTML page**. Google puts only the
first page of items in that HTML; the rest arrive by XHR on scroll, which gdown never issues.
No error, no warning — just a short list that looks like the real contents.
**Fix:** `scripts/drive_list.py`, which walks `embeddedfolderview` (full listing, plain HTML,
no API key). It retries **empty** results rather than believing them, because a throttled
response and an empty folder look identical.

### 3.2 Two MULTIPOLYGONs in 523,281 rows
`d1_target_prior.py` extracted bounding boxes with `for ring in coords for c in ring`. Two
`MULTIPOLYGON` rows nest one level deeper, so that yields coordinate *pairs* instead of
scalars and `np.asarray` raises `inhomogeneous shape` — **after** the ~10-minute reprojection
had already run. **Fix:** `shapely.geometry.shape(g).bounds`, which is type-agnostic.

### 3.3 `--resume` is not "load these weights"
It restores optimizer state, the epoch counter and `best_val_iou`. Resuming the AIRS
checkpoint into a Jaipur run set `start_epoch=91` (so a 40-epoch run does nothing) and
inherited a 0.8784 bar measured on clean labels, so **no checkpoint would ever be saved**
against noisier targets. **Fix:** `--init_weights` loads weights only.

### 3.4 A threshold sweep that reports its own boundary has not found an optimum
The method comparison bottomed out at 0.30, then at 0.05 — **every method pinned to the
lowest value tested, twice.** The seed under-predicts so badly on Jaipur that its operating
point sits below 0.05. Sweep now starts at 0.01.

> **Rule: if the best value is at the edge of the range, the range is wrong.**

### 3.5 Mask a Dice loss *after* the sigmoid
Scaling logits by zero maps to sigmoid 0.5 — mid-confidence foreground, not background. Zero
out probabilities, not logits.

### 3.6 The March baseline is unreproducible, and why
`logs/..._125206.json` shows `train_dir=/tmp/train_crops`, `max_samples=2000`. Those crops
lived in ephemeral container storage and vanished. **The 0.8784 val IoU cannot be re-derived
from anything on disk or in Drive.** The quota trick was not the mistake — the mistake was
that no script recorded how the crops were built, so they are not regenerable.

> **Rule: `/tmp` is fine for intermediates you can regenerate from a committed script, and
> only then.**

### 3.7 `.gitignore` did not cover the data directories
`data/` (9 GB), `.pydeps/` and `.venv-docker/` were untracked but **unignored** — one
`git add -A` would have staged the entire Jaipur mosaic. `data/README.md` claimed a gitignore
rule that did not exist. Second occurrence in this repo (see `a81b4d3`). Checkpoint patterns
also missed timestamped subdirectories.

### 3.8 Metrics hidden by a schema assumption
The first run ledger showed every evaluation run as a blank row, because eval logs store
scores in `threshold_sweep`, not `epochs`. That hid the project's best numbers — **rooftop
IoU 0.9016**, beating the PSPNet baseline of 0.899 — in a JSON nobody had opened since March.

### 3.9 Deduplicating on path is not deduplicating on identity
Several logs exist as genuine **copies** under both `logs/` and `<stage>/logs/`. `realpath`
dedup does not catch copies, only symlinks; the ledger reported 43 runs where there are 34.
Collapse on run identity instead.

### 3.10 Silent path mismatch skipped a whole comparison arm
The method comparison ran without its most important row because checkpoints live in a
*timestamped subdirectory* (`checkpoints/<run>/best.pth`) and the path given pointed one level
too high. The script treated "file not found" as "method not requested" and printed a clean
table with the key row missing.

> **Rule: an optional input that silently disappears should log a warning, not vanish.**

---

## 4. Pre-existing, still open

- **BDAPPV has zero negative crops** (`MASTER_CONTEXT` C1). `prep_bdappv.py:85` drops
  mask-less images, so every training crop contains a panel and the model cannot learn "no
  panel here" — the opposite of what a precision-critical, low-base-rate task needs.
  **Every Stage-2 number is measuring the wrong task until this is fixed.**
- **Three planning documents with three different phase decompositions** — `plan/` (A–F),
  `.planning/ROADMAP.md` (1–5), `MASTER_CONTEXT` (A–E). Unreconciled.
- **`feat/init-project-setup` tracked `origin/master`**, so a bare `git push` from it targeted
  master. Flagged 2026-08-04, fixed 2026-09-09.
