# experiments/ — the run log

One folder per experiment, named `YYYY-MM-DD-<short-slug>`, each containing a `README.md`
written from `TEMPLATE.md`. This is the project's memory: when the thesis asks "why did you
choose 0.35?" or "did you ever try X?", the answer lives here, not in a terminal scrollback
that is gone.

**The rule: an experiment that is not written up did not happen.** A result nobody can
reproduce or date is not usable in a report.

## Index

Newest first. Keep this table current — it is the first thing anyone reads.

| ID | Date | Question | Headline result | Status |
|---|---|---|---|---|
| [`2026-09-10-solar-self-training`](2026-09-10-solar-self-training/) | 2026-09-10 | Does CBST self-training close the solar domain gap? | **No — it destroys it.** 0.5611 → **0.3752** on target | ❌ |
| [`2026-09-10-eroded-labels-rerun`](2026-09-10-eroded-labels-rerun/) | 2026-09-10 | Reproduce the destroyed default, measure it honestly | **0.6393** (orig 0.6405); pred/label **0.9914**, split 0.084 | ✅ |
| [`2026-09-10-erosion-sweep`](2026-09-10-erosion-sweep/) | 2026-09-10 | How far can erosion go before it hurts? | **0.8 m over-erodes** — merge 0.13 but pred/label 1.47; 0.4 m is the optimum |
| [`2026-09-10-eroded-labels`](2026-09-10-eroded-labels/) | 2026-09-10 | Can shrinking labels stop the model fusing buildings? | **Yes** — merge 0.46→0.33, count 0.76→**0.97** per building, −0.016 IoU | ✅ |
| [`2026-09-10-solar-google-to-ign`](2026-09-10-solar-google-to-ign/) | 2026-09-10 | How big is the solar domain gap where it can be measured? | **0.8723 → 0.5611** (−31 pts); a capability drop, not miscalibration | ✅ |
| [`2026-09-10-merge-split-rate`](2026-09-10-merge-split-rate/) | 2026-09-10 | How many buildings does the model fuse together? | **50 % merged**, 21 % under-counted — invisible to IoU | ✅ |
| [`2026-09-10-d4-adjacency`](2026-09-10-d4-adjacency/) | 2026-09-10 | Do Jaipur buildings touch each other? | **78 %** do — instance-merging workstream justified | ✅ |
| [`2026-09-10-label-quantity-vs-quality`](2026-09-10-label-quantity-vs-quality/) | 2026-09-10 | Are the 205k low-confidence buildings worth keeping? | **Unresolved** — each model wins on its own labels; needs ground truth | ⚠️ |
| [`2026-09-10-segformer-backbone`](2026-09-10-segformer-backbone/) | 2026-09-10 | Does a transformer encoder beat ResNet-34? | **+0.0086** (0.6569) and **2× faster convergence**; gain is all precision | ✅ |
| [`2026-09-09-boundary-relaxed-loss`](2026-09-09-boundary-relaxed-loss/) | 2026-09-09 | Does ignoring a 4 px label band help? | **No** — 0.6366 vs 0.6475; +0.004 precision for −0.024 recall | ❌ |
| [`2026-09-09-does-the-airs-seed-help`](2026-09-09-does-the-airs-seed-help/) | 2026-09-09 | Is AIRS pretraining worth anything for Jaipur? | **No** — ImageNet init 0.6483 vs seeded 0.6475, identical trajectory | ✅ |
| [`2026-09-09-d2-d3-source-stats`](2026-09-09-d2-d3-source-stats/) | 2026-09-09 | How sparse is AIRS, and how big are its buildings? | AIRS **7.69 %** fg (assumed 15 %); buildings **23× larger in pixels** | ✅ |
| [`2026-09-09-method-comparison`](2026-09-09-method-comparison/) | 2026-09-09 | Which adaptation method actually wins on Jaipur? | **weak sup 0.65** vs seed 0.14; histmatch/FDA *harmful* | ✅ |
| [`2026-09-09-weak-supervision-jaipur`](2026-09-09-weak-supervision-jaipur/) | 2026-09-09 | Can Open Buildings labels alone train a Jaipur model? | **IoU 0.6475** — 4.6× the unadapted seed | ✅ |
| [`2026-09-09-d1-target-prior`](2026-09-09-d1-target-prior/) | 2026-09-09 | What fraction of Jaipur pixels are actually buildings? | **28.19 %** (23.06 % at conf ≥ 0.75) | ✅ |
| [`2026-09-08-d6-seed-probe`](2026-09-08-d6-seed-probe/) | 2026-09-08 | What does the AIRS-trained seed predict on Jaipur? | **5.69 %** foreground — ~5× below the D1 prior | ✅ |

> **The pair above is the project's motivating figure.** D1 measures what is there, D6
> measures what the model sees, and the gap between them is what the adaptation must close.

## Starting a new one

```bash
./scripts/new_experiment.sh <short-slug>      # scaffolds the folder + README from TEMPLATE
```

Then, in order:

1. **Fill in Question and Prediction before you run anything.** Writing the expected number
   first is the only way a surprising result stays visible as a surprise.
2. Run it. Put the exact command in the doc, not a paraphrase of it.
3. Record the commit SHA that produced the result (`git rev-parse --short HEAD`). A metric
   without the code that made it cannot be reproduced.
4. Fill in Results, Interpretation, Decision, Threats to validity.
5. Add a row to the index above.
6. Commit the write-up **together with** the outputs it describes, and push.

## Where outputs live

| kind | location | in git? |
|---|---|---|
| Diagnostics D1–D7 | `diagnostics/d<n>/` — where `MASTER_CONTEXT` §6.3 points | ✅ small JSON/CSV |
| Training / eval runs | `experiments/<id>/outputs/` | ✅ metrics only |
| Checkpoints (`.pth`) | `rooftop/checkpoints/`, `solar_panel/checkpoints/` | ❌ gitignored — record the path and epoch |
| Qualitative overlays | alongside the outputs | ⚠️ a handful only, they are large |

The write-up **links** to its raw output rather than restating it, so there is exactly one
copy of every number. Diagnostics keep their existing home so the master context's
cross-references stay valid.

## The four records, and what each is for

| File | Answers | Indexed by |
|---|---|---|
| [`TIMELINE.md`](TIMELINE.md) | *what was tried, in what order, and what came of it* — **including what was never run** | time |
| `README.md` (this file) | what did we learn about X? | question |
| [`RUN_LEDGER.md`](RUN_LEDGER.md) | what did run Y score? metrics + per-epoch curves | run |
| [`BACKLOG.md`](BACKLOG.md) | what is next? | priority |

All four except this one are **generated** — rerun `scripts/build_timeline.py` and
`scripts/build_run_ledger.py` rather than editing them. A hand-maintained record drifts the
moment someone forgets, and a drifted record is worse than none because it still looks
authoritative.

## Relationship to `docs/sessions/`

They answer different questions and both are worth keeping.

| | records | indexed by |
|---|---|---|
| `docs/sessions/` | what happened on a working day — decisions, findings, handover | date |
| `experiments/` | what a run measured, and what it decided | question |

A session note links to the experiments it produced; an experiment write-up stands alone and
is what the thesis cites.

## What counts as an experiment

Anything whose result could change a decision: diagnostics, training runs, ablations,
hyperparameter sweeps, dataset-construction choices that affect a metric.

Not: refactors, doc edits, dependency fixes. Those are ordinary commits.

**Negative and abandoned results stay.** Mark them `❌ abandoned` with a reason and leave
them in the index. Silently deleting a failed run is how a project ends up repeating it six
weeks later — and the run table in `MASTER_CONTEXT` is built on exactly these kill/keep
decisions.
