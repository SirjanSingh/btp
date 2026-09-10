# Operational pitfalls — bugs, traps and mistakes, with the rule each one taught

Engineering and process traps hit while working on this repo and this server.
Distinct from [`.planning/research/PITFALLS.md`](../.planning/research/PITFALLS.md), which
covers *domain* pitfalls (segmentation, imagery, labels).

Every entry here cost real time at least once. Several produced **confident, wrong
conclusions that were written into planning documents** and had to be withdrawn later —
those are the expensive ones, and they are listed first.

---

## 0. Root patterns — read this instead of the whole list

Fifteen entries below collapse into **three mistakes made repeatedly**. Learning the list item
by item is useless; these are the shapes to recognise.

### Pattern A — trusting a check that cannot detect what it is checking for
**Seven instances in one session.** Every one produced a confident wrong statement.

| The check | Why it could never work |
|---|---|
| Raised gdown's `MAX_NUMBER_FILES` and saw no change | That constant only gates a *warning*; it cannot change what is fetched |
| `push \| tail -1 \| grep -qv fatal` | The failing line was not last, so "PUSHED" printed on a failed push |
| `tail` on push output | The decisive `100 MB` rejection sat **four lines above** `fatal:` |
| `split_rate` requiring ≥50 % overlap | Fragments smaller than half a building score 0.0 — reported "surprising 0.0" **three times** |
| Smoke test on 32 crops | Contained no empty-label crop, so the crash path never ran; the full run died immediately |
| `df -h /home` | Reports the 7 TB filesystem, not the 40 GB quota that actually binds |
| `ps` / `docker ps` for "what's running" | The question was about the task panel — right answer, wrong layer |

> **Preventive check: before believing a negative result, state what would have happened if
> the hypothesis were TRUE.** If the answer is "the same output", the test is worthless. Ask
> it *out loud* before reporting, not after being contradicted.

### Pattern B — assuming an exact name where code generates the name
**Three instances**, each failing *silently* and looking like missing data rather than a bug.

- `.gitignore` `**/checkpoints/**/*.pth` — missed `checkpoints_mit/`; 280 MB reached GitHub.
- `build_run_ledger.py` `experiments/*/outputs/*.json` — missed `outputs_mit/`, silently
  omitting **two complete runs** from the ledger.
- Symlinks built with host-absolute paths — dangle inside the container at `/workspace`.

> **Preventive check: glob `prefix*`, and after writing any collector, assert the count it
> found against a count obtained a different way.** A collector that finds nothing looks
> identical to a directory that contains nothing.

### Pattern C — applying a stale rule instead of re-deriving it
- A flat "don't launch under 3 GB quota" floor blocked launches for hours, when the job in
  question writes **280 MB**. The floor was invented once and never re-checked against what
  jobs actually cost.
- Reported "GPUs free" for several ticks while treating the box as busy, because the
  two-job rule was being applied as dogma rather than re-derived from the current load.

> **Preventive check: re-measure resources every tick and compare against the job's actual
> cost. Never carry forward a number or a threshold from a previous reading.**

### What actually saved the session
The **record-before-deleting** rule. When `filter-branch` destroyed two checkpoints, every
metric survived in `RUN_LEDGER.json` because it had been written first. That rule was the one
piece of process that paid for itself — and it did so by accident, since the deletion was not
the deliberate one the rule was written for.

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

### 3.10 Absolute symlinks break under a bind mount ★
Filtered dataset splits were built as symlink trees pointing at host-absolute paths
(`/home/23ucs715/btp/...`). `run_docker.sh` bind-mounts the repo at **`/workspace`**, so every
link dangled inside the container. Two failures, both misleading:

- `cv2.imread` returns **`None`** for a dangling link rather than raising, so training died
  20 frames deep in a dataloader worker with
  `cvtColor: (-215) !_src.empty()` — nothing pointing at the real cause.
- `os.makedirs(..., exist_ok=True)` **raises `FileExistsError`** on a dangling symlink,
  because the path exists but is not a directory. The job crashed instantly, and the wrapper
  still exited 0, so it looked like it had completed successfully and written nothing.

> **Rule: symlinks that cross into a container must be relative** (`os.path.relpath`), and
> verify one read *inside* the container before launching hours of work.

### 3.11 Silent path mismatch skipped a whole comparison arm
The method comparison ran without its most important row because checkpoints live in a
*timestamped subdirectory* (`checkpoints/<run>/best.pth`) and the path given pointed one level
too high. The script treated "file not found" as "method not requested" and printed a clean
table with the key row missing.

> **Rule: an optional input that silently disappears should log a warning, not vanish.**

---

### 3.12b The real cause was a 280 MB file, not the network ★
The 408s below were a **symptom**, and I chased the symptom for four ticks. The actual error
was visible only in the full push output:

```
remote: error: File .../best.pth is 279.90 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected.
```

The pending pack was **546 MB**. GitHub's pre-receive hook rejects it, and pushing half a
gigabyte over a throttled uplink also produces timeouts — so the network symptoms were real
but secondary, and "fixing" them with `http.postBuffer` and `http.lowSpeedLimit` could never
have worked.

**Why the files were tracked at all:** `.gitignore` had `**/checkpoints/**/*.pth`, which
matches only directories named *exactly* `checkpoints`. An experiment writing to
`checkpoints_mit/` and `checkpoints_resnet34/` slipped straight through. Now `*.pth` outright
— weights are regenerable and their statistics live in `RUN_LEDGER.json`.

**Fix for the already-committed files** (unpushed, so rewriting is safe):
`git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch "*.pth"' origin/<branch>..HEAD`
Pack fell 546.74 MB → **0.07 MB** and the push went through instantly.

Two snags worth knowing: `filter-branch` refuses to run with **any** unstaged change, and a
*live training run* keeps rewriting its own tracked log files, so the tree is never clean —
`git checkout -- <logdir>` immediately before the rewrite is what got a clean window.

> **Rule: read the whole push output, not the last line.** The decisive error was four lines
> above `fatal:`, and every tail-based check I ran hid it.

### 3.12 `git push` fails with HTTP 408 on this network
Pushes to GitHub began failing with `error: RPC failed; result=22, HTTP code = 408` followed
by `fatal: The remote end hung up unexpectedly` — while *reads* (`git ls-remote`, `curl
https://github.com`) worked fine. 408 is **Request Timeout**: git sends the pack with chunked
transfer encoding, and this node's throttled uplink is slow enough that the server gives up.

Confusingly, git then prints **`Everything up-to-date`** immediately after the failure, which
reads as success. It is not — verify with SHAs, never with the message.

**Fix:** `git config http.postBuffer 524288000`, which makes git send `Content-Length` instead
of chunked for payloads under that size. The push then succeeds, slowly.

> **Rule: confirm a push by comparing `git rev-parse HEAD` with
> `git rev-parse origin/<branch>` after a fetch.** Output text lies in both directions here.

### 3.13 Checking output text instead of exit status
A retry loop tested `git push ... | tail -1 | grep -qv fatal` and reported "PUSHED on attempt
1" when nothing had been pushed — the last line happened not to contain "fatal" even though
the push failed. Same shape as 1.2: a check that cannot detect the thing it is checking for.

> **Rule: branch on exit status, or on the actual state you care about, never on a substring
> of stdout.**

### 3.14 `git filter-branch` deletes the files it strips, from disk ★
Stripping two 280 MB checkpoints out of unpushed history (3.12b) also **removed them from
the working tree**. `filter-branch` finishes with a `git reset --hard` to sync the tree to
the rewritten HEAD, so anything dropped from the commits is dropped from disk too.

Only *tracked* files are affected — every other `.pth` in the repo was gitignored and
survived untouched. Which means the two lost were exactly the ones that had slipped past
`.gitignore` in the first place.

What saved it: `RUN_LEDGER.json` and the diagnostics JSONs already held every metric, so
only the **weights** were lost, not the results. That is the entire reason the "record stats
before deleting a checkpoint" rule exists, and it paid for itself here — though by accident
rather than by my applying it deliberately.

> **Rule: before `filter-branch`, copy the files being stripped somewhere outside the repo.**
> `git rm --cached` removes from the index only; `filter-branch` removes from the disk.

### 3.15 The same directory-naming assumption, three times
1. `.gitignore` had `**/checkpoints/**/*.pth` — missed `checkpoints_mit/`, which is how
   280 MB reached GitHub (3.12b).
2. `build_run_ledger.py` globbed `experiments/*/outputs/*.json` — missed `outputs_mit/` and
   `outputs_resnet34/`, silently omitting **two complete runs** from the ledger. Only noticed
   because those runs' checkpoints were destroyed and I went looking for their numbers.
3. Same shape as 3.10's absolute-symlink assumption.

> **Rule: glob for `prefix*`, not `prefix`, whenever a directory name is generated by code.**
> A pattern that assumes an exact directory name fails silently and looks like absence of
> data rather than a bug.

### 3.16 Training was dataloader-bound, not GPU-bound — 3.7× left on the table
Sampling `utilization.gpu` every 2 s during two live runs gave a **0 → 97 → 0 sawtooth**
(means ~44 % and ~67 %). That pattern is not a busy GPU; it is a GPU idling between batches.

Measured cause: **PNG decode costs 11.6 ms/image** on this node, so one worker sustains ~86
img/s and three sustain far less at a load average of 200+. CPU is the contended resource on
this box, and every crop was being decoded again on every epoch.

The machine has **397 GB RAM free** and the whole crop set is **7.7 GB**. Decoding once into
RAM at startup removes the bottleneck outright; DataLoader workers are forked, so the arrays
are copy-on-write shared rather than duplicated per worker.

| 1,600 crops, batch 16, 3 workers | epoch time |
|---|---|
| default | 74.4 s |
| `--cache_ram` | **20.2 s** |

**3.68× on identical config under the same contention.** Also added
`persistent_workers=True` and `prefetch_factor=4`.

> **Rule: before tuning a model, sample GPU utilisation. A sawtooth means the bottleneck is
> upstream, and no amount of batch-size or architecture work will fix it.**

### 3.17 Parking regenerable data on host `/tmp` — and why a plain symlink is not enough
The 40 GB `/home` quota binds long before disk does; host `/tmp` sits on `/` (18 GB free) and
is outside the quota. `data/jaipur_weak` (4.2 GB, regenerable from
`scripts/make_weak_labels.py`) now lives at `/tmp/btp_data/jaipur_weak` with a symlink from
`data/`.

**The symlink alone does not work in the container.** `run_docker.sh` bind-mounts `.tmp`
(on `/home`) to container `/tmp`, so a link to `/tmp/btp_data/...` resolves *inside* the
container to `.tmp/btp_data/...` — which does not exist. Same class as 3.10: a path that is
valid on the host and dangling under the bind mount.

**Fix:** `run_docker.sh` now mounts each directory under `/tmp/btp_data` directly over its
expected `/workspace/data/<name>` path, so the container sees real data where the host sees a
symlink. Verified with a read *inside* the container before deleting the original.

**Only regenerable data goes here.** Host `/tmp` is shared, has no retention guarantee
(oldest surviving files were 9 days old, no active cleanup timer), and filling it hurts every
user on the box. Source imagery and anything not rebuildable from a committed script stays on
`/home`.

### 3.18 A list cache defeats copy-on-write — 8.4× slower than its own timer claimed ★
The solar runs reported **2.8 min/epoch** while taking **23.5 min** of wall clock. The
instrumented region covered train + val + tensorboard, so 85 % of the time was vanishing
*outside* any timer — invisible to every per-epoch number in the logs.

**Cause:** the RAM cache stored a **list of 33k numpy arrays**. DataLoader workers are forked,
and copy-on-write only helps while pages are not written — but **Python refcounting writes to
the header of every object a worker touches**, so each worker copies the whole structure. One
contiguous `ndarray` is a single object and stays genuinely shared.

**Why I got it wrong:** I chose the list deliberately, to avoid assuming a fixed crop shape
(pattern B). Avoiding one failure mode created a worse one. **The right move was neither
assuming nor avoiding — it was verifying:** probe the first crop, build the contiguous array,
check each crop matches as it loads, and fall back to the list only if something is ragged.
BDAPPV turned out to be uniformly 400×400.

> **Rule: when wall-clock and instrumented time disagree, the gap is the bug.** Per-epoch
> timers only measure what you wrapped; compare against `timestamp` and the clock.

### 3.19 Judging a label-space knob by label statistics ★

**What happened.** Rendered the Open Buildings footprints at four erosion levels and counted
connected components in the *labels*. 0.2 m separated +98 % more buildings for −7.2 % area;
0.4 m added only 7 more points for another 7 % of area. The obvious reading: 0.2 m is the
efficient choice and 0.4 m overpays. I reported that reading to Sirjan before checking the
model-side sweep, which had already been run and had already answered it.

**Why it was wrong.** The model trained on 0.2 m labels still under-counts buildings by 16 %
(`pred/label` 0.8412); only 0.4 m reaches 0.9914. At 26.6 cm/px a 0.2 m erosion is ~0.75 px —
the gap exists in the label but is too thin to survive the encoder's downsampling. It takes
~1.5 px before the model reliably emits two components.

**Shape.** A variant of pattern A: the label-component count *cannot* detect whether the model
learns the gap, so it could not have contradicted the conclusion I drew from it. It is also a
plain failure to check the experiment index before claiming something was untested — the
answer was already in `experiments/2026-09-10-erosion-02/`.

**Rule.** Never choose a label-space parameter from label statistics. The label is the input;
`pred/label` on a trained model is the output, and only the output is the deliverable. And
before proposing an experiment, grep `experiments/` for it — this sweep was closed hours
earlier.

## 4. Pre-existing, still open

- **BDAPPV has zero negative crops** (`MASTER_CONTEXT` C1). `prep_bdappv.py:85` drops
  mask-less images, so every training crop contains a panel and the model cannot learn "no
  panel here" — the opposite of what a precision-critical, low-base-rate task needs.
  **Every Stage-2 number is measuring the wrong task until this is fixed.**
- **Three planning documents with three different phase decompositions** — `plan/` (A–F),
  `.planning/ROADMAP.md` (1–5), `MASTER_CONTEXT` (A–E). Unreconciled.
- **`feat/init-project-setup` tracked `origin/master`**, so a bare `git push` from it targeted
  master. Flagged 2026-08-04, fixed 2026-09-09.
