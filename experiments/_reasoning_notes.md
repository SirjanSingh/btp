<!-- Hand-written. Embedded into TIMELINE.md by scripts/build_timeline.py.
     Everything else in the timeline is generated; this is the part that cannot be. -->

## 0. What I was actually thinking

The results table says what happened. This says **why each thing was tried, what I expected,
and what changed my mind** — the part that evaporates fastest.

### The thread, in order

**It started with a gap nobody had measured.** The project's premise was "a New Zealand model
fails on Jaipur", but both halves of that were guesses: the planning docs assumed AIRS ~15 %
foreground and Jaipur ~50 %. D1 and D6 turned the premise into a number — the model predicts
**5.7 %** where truth is **28.2 %**, a ~5× under-prediction. Everything after that is an
attempt to close, explain, or re-frame that gap.

**Then weak supervision made most of the plan look overbuilt.** Open Buildings gives 523k free
Jaipur footprints, and simply training on them scored **0.6476** against the unadapted seed's
**0.1419** — 4.6×, in one evening, with data already on disk. My honest reaction: the
elaborate UDA apparatus (DAFormer → HRDA → MIC, CBST thresholds, self-training rounds) was
being planned to solve a problem that free labels mostly dissolve. **Data beat algorithms by
roughly 60× over anything else tried since.** I kept coming back to that ratio when deciding
what to run next.

**The AIRS ablation is the result I'd defend hardest.** I expected the seed to be worth
*something* — generic aerial features at minimum — and predicted ImageNet init would land
0.60–0.64, below it. It landed **0.6483 vs 0.6475**, and more tellingly both runs hit 0.63 at
**exactly epoch 9** and peaked at **exactly epoch 33**. Identical trajectories. That killed
the source-domain question (AIRS vs Inria vs Khartoum) that had blocked two sessions, and
retired the "missing AIRS dataset" blocker. D2/D3 then explained *why*: AIRS buildings are
**21,084 px** and Jaipur's are **913 px** — a 23× scale difference. It isn't a harder version
of the same problem, it's a different one.

**The turn came from asking what IoU wasn't telling us.** D4 measured **78 %** of Jaipur
buildings touching a neighbour. `MASTER_CONTEXT` §6.2 already warned that pixel IoU cannot see
instance merging — so I implemented merge/split rate expecting maybe 20–35 %. It came back at
**50 %**. A model reporting a respectable 0.6483 IoU was fusing **half** the buildings it was
meant to count, and emitting **21 % fewer components than exist**. Since the pipeline's output
is a *per-building* kW estimate, that is the error that actually matters, and no metric in the
repo could see it.

**That reframed what "better" means here.** Erosion — shrinking each footprint 0.4 m so
touching buildings get a gap — costs 0.016 IoU and looks like a regression. It takes
`pred/label` from **0.760 to 0.9745**: under-counting falls from 24 % to 2.6 %. Judged on IoU
it fails; judged on the project's actual goal it is the largest win of the run. **R8 had to
exist before that experiment could be interpreted at all.**

### Instincts that were wrong, and what they cost

- **"Photometric alignment is nearly free"** (`plan/03` Tier 1, and I believed it). Histogram
  matching took the seed from 0.142 to **0.026** — *worse than doing nothing*. AIRS is
  vegetated suburban Christchurch; forcing Jaipur's colours onto that reference destroys the
  roof/ground contrast. Deleted from the plan.
- **"Boundary relaxation will help"** (+0.02–0.05 predicted). It cost **0.011**. The median
  Jaipur building is ~30×30 px, so a ±4 px ignore band is most of the object. Sound for large
  buildings, self-defeating for small ones.
- **"Erosion helps ResNet more than MiT"** — reasoning that MiT already merged less so had
  less to gain. Wrong by 2×: −0.064 vs **−0.136**. The two are *synergistic*; a global
  receptive field can exploit a label gap that convolutions cannot.
- **"Dilating predictions back will recover the misses"** — I wrote that into a code comment
  and the next run refuted it. Merge went 0.3256 → **0.4163**. Two components 3 px apart are
  closed by a 2 px dilation from each side. The geometry was checkable in advance and I did
  not check it.

### Judgement calls, and why

- **Predictions written before every run.** Not ceremony — the boundary-relaxation and
  erosion-interaction misses are only visible *as* surprises because the expectation was
  timestamped first.
- **Negative results kept in the index**, marked ❌. Deleting a failed run is how a project
  repeats it six weeks later.
- **Two concurrent jobs, never three.** At three, load passed 300 on 80 cores and throughput
  collapsed to ~1 epoch/tick; two sustain ~5. VRAM was never the constraint — CPU was.
- **Never selected a model on the target set.** S1 picks its checkpoint on *source* val;
  choosing on `ign_val` would leak the target into model choice and flatter every later
  method's gap-closure.
- **R3 is marked unresolved rather than answered.** Each label-confidence model wins on its
  own label distribution, so the metric measures agreement, not accuracy. Choosing a
  confidence threshold by scoring against a val set built at *some* threshold is circular —
  which turns the 400-tile hand-labelling task into the blocker for a specific decision rather
  than generic good practice.

### What I would do next, and why

1. **Finish the erosion sweep**, since split rate stayed at exactly 0.0 at both 0.4 m and
   0.8 m-so-far — the overshoot bound is genuinely unfound, and that is unusual enough to be
   worth pinning down.
2. **Hand-label the 400 tiles.** Three separate results now dead-end on the absence of ground
   truth (R3, the merge-rate lower bound, and every IoU quoted anywhere).
3. **Self-training on google→ign before Jaipur.** It is the only place a target IoU exists;
   tuning on Jaipur is guessing with extra steps.
4. **Re-read the weak-supervision headline.** "0.6483 IoU" reads far better than "half the
   buildings merged, 21 % under-counted" — and the second sentence is the true one.
