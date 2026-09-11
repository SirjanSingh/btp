# Timeline — what was run, what wasn't, and in what order

*Generated 2026-09-11 14:47 by `scripts/build_timeline.py`. Do not edit by hand — rerun the script.*

This answers the question the other documents do not: **what was tried, in what order, and what came of it?** Months later, when writing up, the hard question is usually not "what did X score" but "did we ever actually test X, or did we just plan to?" — so §3 records what was **never run**, and why, as deliberately as §2 records what was.

| See also | For |
|---|---|
| [`README.md`](README.md) | results indexed **by question** |
| [`RUN_LEDGER.md`](RUN_LEDGER.md) | every run's **metrics + per-epoch curves** |
| [`BACKLOG.md`](BACKLOG.md) | the **queue** |
| [`../docs/sessions/`](../docs/sessions/) | per-day **narrative** |

---


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
- **"Split rate staying at 0.0 is a real surprise"** — I said that three times across three
  runs. It was partly **my own metric being blind**: a label counts as split only if two
  predictions each cover half of it, so a model shattering buildings into thirds scores 0.0.
  At 0.8 m erosion `pred/label` hit **1.4743** while split rate still read 0.0. The lesson is
  sharper than the bug: I trusted a metric I had written that morning, and reported its
  silence as evidence. **A metric with no failing case in your data is not validated.**
- **"Dilating predictions back will recover the misses"** — I wrote that into a code comment
  and the next run refuted it. Merge went 0.3256 → **0.4163**. Two components 3 px apart are
  closed by a 2 px dilation from each side. The geometry was checkable in advance and I did
  not check it.

### The self-training arc — the clearest thing the bench bought

Three runs, one variable, and the plan turned out to be wrong about its own recommendation.

`MASTER_CONTEXT` prescribes **CBST class-ratio thresholding** over a fixed high threshold,
because a fixed threshold causes *foreground collapse*. Before running anything I measured
what the ratio policy implied: to select the source's 1.83 % of target pixels, the threshold
had to fall to **0.0004** — essentially zero. I wrote "at 0.0004 nearly any activation counts
as a panel" into the experiment and predicted it would fail.

It failed harder than predicted: **0.5611 → 0.3752**, precision collapsing 0.741 → 0.391.

Then the same pipeline with a plain confidence threshold (0.4563): **0.5611 → 0.6165**,
*beating* the source-only baseline, with **both** precision and recall up. I predicted
0.50–0.58 and it beat that too.

So the prescribed fix was the entire cause of the failure. CBST has an unstated precondition —
**it is safe only while the model retains calibrated confidence on the target** — and a
31-point domain gap is precisely what destroys that. On Jaipur, where nothing can be measured,
following the plan faithfully would have degraded the model by a third with no way to notice:
S2's *source* score barely moved (0.8723 → 0.8678) and would have looked healthy.

That is the whole argument for the ordering principle, demonstrated rather than asserted.

### The pattern behind the mistakes

Individually the errors above look unrelated. They are not — `docs/PITFALLS.md` §0 groups all
fifteen into **three shapes**, and I hit each repeatedly in one session:

**A · Trusting a check that cannot detect what it checks for** — seven times. Raising gdown's
`MAX_NUMBER_FILES` (a warning gate, not a fetch limit); grepping `tail -1` of a push for
"fatal"; a split-rate threshold that fragments can never trip; a 32-crop smoke test containing
no empty-label crop; `df` instead of `quota`. **Every one produced a confident wrong
statement.** The fix is one question asked *before* reporting: *if the hypothesis were true,
would this output differ?*

**B · Assuming an exact directory name where code generates it** — three times, always
silently. `checkpoints_mit/` past `.gitignore`, `outputs_mit/` past the ledger's glob,
host-absolute symlinks dangling at `/workspace`. A collector that finds nothing looks
identical to a directory containing nothing.

**C · Applying a stale rule instead of re-deriving it** — a flat 3 GB quota floor blocked
launches for hours when the job wrote 280 MB, and the two-job rule got applied as dogma while
four GPUs sat idle at 0 %. Sirjan caught both. **Re-measure every tick; never carry a
threshold forward.**

The one process that paid for itself was **record-before-deleting**: when `filter-branch`
destroyed two checkpoints, every metric survived because the ledger had been written first.

### Avoiding one mistake caused a worse one

Worth recording because it is the subtlest error of the session. I cached the solar crops as a
**list** of arrays specifically to avoid assuming a fixed shape — pattern B, which had already
bitten three times. That choice made training **8.4× slower than its own timer reported**:
forked DataLoader workers copy a list of 33k Python objects because refcounting writes to
every object header, while a single contiguous `ndarray` stays copy-on-write shared.

Defensiveness is not free. The right move was **neither assuming the shape nor avoiding it,
but verifying it** — probe, build the fast path, check as you go, fall back if ragged. The
crops were uniformly 400×400 all along.

And I only found it by comparing wall-clock against the instrumented per-epoch time. **Every
number in the log said 2.8 min/epoch; reality was 23.5.** A timer only measures what you
wrapped around.

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
| 2026-09-10 | [`2026-09-10-eroded-labels-rerun`](2026-09-10-eroded-labels-rerun/) | ✅ done — reproduced | **0.6393** (orig 0.6405); pred/label **0.9914**, split 0.084 |
| 2026-09-10 | [`2026-09-10-erosion-02`](2026-09-10-erosion-02/) | ✅ done — completes the sweep | **No** — monotonic curve; 0.4 m is the only point with pred/label ≈ 1 |
| 2026-09-10 | [`2026-09-10-erosion-sweep`](2026-09-10-erosion-sweep/) | ✅ done — **0.8 m over-erodes** | **0.8 m over-erodes** — merge 0.13 but pred/label 1.47; 0.4 m is the optimum |
| 2026-09-10 | [`2026-09-10-jaipur-selftrain`](2026-09-10-jaipur-selftrain/) | ❌ negative — teacher wins; kill criterion fired | **No.** +0.004 IoU but `pred/label` 0.9914 → 0.9134 — self-training discards the label erosion |
| 2026-09-10 | [`2026-09-10-label-quantity-vs-quality`](2026-09-10-label-quantity-vs-quality/) | ✅ done — **confounded; see cross-eval** | **Unresolved** — each model wins on its own labels; needs ground truth |
| 2026-09-10 | [`2026-09-10-merge-split-rate`](2026-09-10-merge-split-rate/) | ✅ done | **50 % merged**, 21 % under-counted — invisible to IoU |
| 2026-09-10 | [`2026-09-10-segformer-backbone`](2026-09-10-segformer-backbone/) | ✅ done — modest win | **+0.0086** (0.6569) and **2× faster convergence**; gain is all precision |
| 2026-09-10 | [`2026-09-10-selftrain-round2`](2026-09-10-selftrain-round2/) | ❌ negative — one round is the recipe | **No — it saturates.** 0.6135 → 0.6104; teacher selects the same pixels |
| 2026-09-10 | [`2026-09-10-selftrain-threshold-cliff`](2026-09-10-selftrain-threshold-cliff/) | ✅ done — cliff located; stated mechanism refuted | **No cliff — a plateau.** thr 0.01–0.46 within 0.023 IoU; CBST's prescribed ratio is **2× past break-even** (0.0091) |
| 2026-09-10 | [`2026-09-10-solar-google-to-ign`](2026-09-10-solar-google-to-ign/) | ✅ done | **0.8723 → 0.5611** (−31 pts); a capability drop, not miscalibration |
| 2026-09-10 | [`2026-09-10-solar-self-training`](2026-09-10-solar-self-training/) | ✅ done — **strong negative result** | **No — it destroys it.** 0.5611 → **0.3752** on target |
| 2026-09-10 | [`2026-09-10-solar-selftrain-conf`](2026-09-10-solar-selftrain-conf/) | ✅ done — **self-training works; CBST was the problem** | **CBST's policy.** Confidence threshold: **0.5611 → 0.6165**, beats baseline |
| 2026-09-10 | [`2026-09-10-split-metric-audit`](2026-09-10-split-metric-audit/) | ✅ done — metric vindicated, interpretation corrected | **Metric sound; strict def is 0.0 everywhere.** 0.8 m *hallucinates* buildings (30 % of preds touch no label), not fragments |
| 2026-09-11 | [`2026-09-11-inference-threshold-sweep`](2026-09-11-inference-threshold-sweep/) | ✅ done — complements, not substitutes; follow-up launched | **Erosion is necessary.** Eroded model wins recall *and* counting at every matched merge; un-eroded `pred/label` tops out at 0.9119 |
| 2026-09-11 | [`2026-09-11-mit-b5`](2026-09-11-mit-b5/) | ❌ negative — +0.0063 IoU (1.5× noise) for 3.6× compute; B2 stays | **Not worth it.** +0.0063 IoU (1.5× noise) for 3.6× compute; every other metric inside noise |
| 2026-09-11 | [`2026-09-11-replication`](2026-09-11-replication/) | ✅ done — claim 1 promoted to solid, claim 2 demoted to directional | **Erosion: yes, 1.7–2.2× pooled noise at 4 matched points.** Self-training regression: direction only (d/SE 1.78) |
| 2026-09-11 | [`2026-09-11-seed-variance`](2026-09-11-seed-variance/) | ★ done (n=3) — `pred/label` 2 sd = 0.076; flagship number needs an error bar | ★ **n=3: 2sd = 0.076 `pred/label`, 0.050 merge, 0.004 IoU.** Flagship 0.9914 is really 0.99 ± 0.08; three claims unsupported |
| 2026-09-11 | [`2026-09-11-selftrain-eroded-pseudo`](2026-09-11-selftrain-eroded-pseudo/) | ❌ negative — diagnosis confirmed, teacher still wins | **Diagnosis yes, method no.** `pred/label` 0.9134→0.9672, but recall falls and the teacher still wins |

### Why each was run, and what was expected

The rationale in each experiment's own words — extracted, not retyped, so it cannot drift from the write-up.

**[`2026-09-08-d6-seed-probe`](2026-09-08-d6-seed-probe/)** — ✅ done
> **Why:** it decides whether this checkpoint is usable as a self-training teacher at all. A teacher that predicts far below the true prior collapses under self-training — sparse predictions → sparse pseudo-labels → sparser teacher. Paired with D1 it is the project's motivating figure.
>
> **Expected:** *(not recorded in advance — reconstructed)* `MASTER_CONTEXT` §3.1 assumed AIRS ~15 % fg and Jaipur ~50 %, and predicted the seed would under-predict substantially. The direction was expected; the magnitude was not quantified.

**[`2026-09-09-boundary-relaxed-loss`](2026-09-09-boundary-relaxed-loss/)** — ✅ done — **negative result**
> **Why:** the first weak-supervision run showed **recall ~0.11 above precision for all 40 epochs**. The labels are Open Buildings *ground footprints*; the model predicts *roof* outlines; off-nadir at 26.6 cm those disagree by roughly 8 px (Gap 4). So the loss was actively teaching the model to shrink roofs down to footprint size — punishing it for roof area that is genuinely there. Relaxing the boundary rem
>
> **Expected:** **+0.02 to +0.05 IoU** (so ~0.67–0.70), with **precision rising** and the precision/recall gap narrowing. Risk: a 4 px band on a 512 px crop removes a large share of the informative pixels at this building density, so it may instead blur edges and cost IoU.

**[`2026-09-09-d1-target-prior`](2026-09-09-d1-target-prior/)** — ✅ done
> **Why:** 1. It sets the **CBST class ratio** for self-training. Guessing it wrong biases every pseudo-label round. 2. Paired with D6 it is the **motivating figure** of the project. 3. `MASTER_CONTEXT` §11 explicitly forbids putting estimated priors in the report.
>
> **Expected:** the planning docs assumed ~50 %, on the intuition that dense Indian wards are near-fully built. Expectation going in was 40–55 %.

**[`2026-09-09-d2-d3-source-stats`](2026-09-09-d2-d3-source-stats/)** — ✅ done
> **Why:** D1 measured the Jaipur prior (28.19%). The AIRS half stayed a guess, so the project's central "trained on X%, deployed on Y%" claim was half estimate — and `MASTER_CONTEXT` §11 forbids estimated priors in the report. D3 decides whether a 512² crop and the receptive field are sized correctly for the target.
>
> **Expected:** AIRS ~15% (the planning-doc figure); prior shift therefore ~1.9×.

**[`2026-09-09-does-the-airs-seed-help`](2026-09-09-does-the-airs-seed-help/)** — ✅ done
>
> **Expected:** **0.60–0.64**, a little below the seeded 0.6475. The AIRS seed should be worth something — generic aerial-imagery features, roof-shaped priors — but much less than its 0.8784 source score suggests, because 7,371 target crops is plenty to learn from directly. If the gap is under 0.01 I would call the seed worthless here.

**[`2026-09-09-method-comparison`](2026-09-09-method-comparison/)** — ✅ done
>
> **Expected:** Tier-1 photometric methods (histogram matching, FDA) give small positive gains — `plan/03` calls Tier 1 *"nearly free"* wins. AdaBN gives a real gain. Weak supervision wins overall.

**[`2026-09-09-weak-supervision-jaipur`](2026-09-09-weak-supervision-jaipur/)** — ✅ done
> **Why:** `plan/README.md` calls Open Buildings weak supervision *"★ biggest win"* and projects **IoU 0.74–0.82**. If that holds, the target domain can be trained on *directly* and the whole unsupervised-domain-adaptation apparatus — DAFormer/HRDA/MIC, CBST thresholds, self-training rounds — becomes optional rather than central. That is the largest single fork in the project, and it is decidable in one GPU-
>
> **Expected:** val IoU **0.60–0.72** against the weak labels — below `plan/`'s 0.74–0.82, because that projection assumes SAM2 refinement and shift correction which this run does not do. Predicted foreground should move from D6's **5.69 %** up toward the label prior of **23.06 %**; if it does not, the fine-tune is not taking.

**[`2026-09-10-d4-adjacency`](2026-09-10-d4-adjacency/)** — ✅ done
> **Why:** `MASTER_CONTEXT` §3.2 lists **instance merging** as a distinct failure mode — party-wall buildings with no visible gap fusing into one blob — and a whole planned workstream (three-class labelling, split-aware metrics) exists to address it. That work is only worth doing if buildings actually touch. A low rate would let the project *delete* a workstream, which is the cheapest kind of result availabl
>
> **Expected:** 40–60%. Dense Indian urban form suggests high, but Open Buildings footprints are individually delineated and I expected visible gaps between many of them.

**[`2026-09-10-eroded-labels`](2026-09-10-eroded-labels/)** — ✅ done
> **Why:** [R8](../2026-09-10-merge-split-rate/) measured a **50.4 % merge rate with a 0.0 % split rate** — the model fuses neighbours and never over-segments. [D4](../2026-09-10-d4-adjacency/) explains it: 78 % of Jaipur buildings have a neighbour within half a metre, under two pixels, so frequently **there is no gap in the label for the model to learn**. The cheapest possible intervention is to put one the
>
> **Expected:** - **Merge rate 0.30–0.40**, down from 0.5040. This is the number the experiment lives or dies by. - **IoU 0.60–0.64**, i.e. slightly *worse* than 0.6483 — predictions will be systematically smaller than the un-eroded val targets. An IoU drop is an acceptable price and is expected. - **Split rate rises above 0**, possibly to a few percent. If erosion overshoots it will start cutting single building

**[`2026-09-10-eroded-labels-rerun`](2026-09-10-eroded-labels-rerun/)** — ✅ done — reproduced
> **Why:** the original run's *metrics* survived in `RUN_LEDGER` (best IoU **0.6405** @ep19), so nothing scientific was lost. But two things need the weights themselves:
>
> **Expected:** should reproduce **0.6395–0.6415** — the same recipe, same data, differing only by seed noise. If it lands outside that, something is non-deterministic that I have not accounted for, which would itself be worth knowing. Expect merge ≈ 0.33, `pred/label` ≈ 0.97 as before, plus a split rate that is now *visible* (the old run reported 0.0 under the strict definition; the true value is probably a few 

**[`2026-09-10-erosion-02`](2026-09-10-erosion-02/)** — ✅ done — completes the sweep
> **Why:** 0.4 m already achieves `pred/label` **0.9914** — within 0.9 % of one prediction per building — but costs 0.018 IoU and 8.4 % split rate. If 0.2 m gets most of the count benefit for half the cost, it is the better default. If it barely moves, 0.4 m is confirmed as a genuine threshold rather than an arbitrary point on a slope.
>
> **Expected:** everything should land **between** the un-eroded and 0.4 m values, since the sweep has been monotonic in every metric so far.

**[`2026-09-10-erosion-sweep`](2026-09-10-erosion-sweep/)** — ✅ done — **0.8 m over-erodes**
> **Why:** [eroded-labels](../2026-09-10-eroded-labels/) cut merging 29 % and took `pred/label` from 0.760 to 0.9745 — but **split rate stayed at exactly 0.0**, in all four cells of the 2 × 2. The failure mode erosion is supposed to risk has not appeared at all, which means the useful range has not been explored to its end. If 0.8 m keeps split at 0 while cutting merges further, 0.4 m was simply too timid.
>
> **Expected:** - **merge rate below 0.25**, down from 0.3256. - **split rate finally rises above 0** — somewhere around 0.02–0.08. If it stays at exactly 0.0 again, that is a real surprise and means the model simply never over-segments at any erosion this side of destroying the labels. - **missed rate rises further**, ~0.36–0.42; more erosion means more conservatism. - **IoU 0.61–0.63**, below 0.6405. - **`pred/

**[`2026-09-10-jaipur-selftrain`](2026-09-10-jaipur-selftrain/)** — ❌ negative — teacher wins; kill criterion fired
> **Why:** this is the first UDA method in the project applied to the actual target with evidence behind its hyperparameters rather than a guess. `MASTER_CONTEXT`'s ordering principle in full: settle where measurable, freeze, transfer blind.
>
> **Expected:** 1. **The ratio→threshold map will be far flatter than solar's.** Selecting 23 % of pixels should land at a threshold of order **0.2–0.5**, not 0.0004. If it lands below 0.05, the rooftop model is as poorly calibrated on Jaipur as the solar model was on IGN, and the S6 warning transfers intact. 2. **Self-training will help, modestly: `pred/label` stays within 0.95–1.05 and merge rate improves by 0.

**[`2026-09-10-label-quantity-vs-quality`](2026-09-10-label-quantity-vs-quality/)** — ✅ done — **confounded; see cross-eval**
> **Why:** the baseline discards **205,076 buildings** — 39% of the dataset — on a confidence threshold nobody has justified with a measurement. Open Buildings is least confident about buildings that are small, irregular, or densely packed, which in Jaipur are plausibly the **hard and important** ones. If discarding them costs accuracy, the threshold is throwing away exactly the signal the project needs; if 
>
> **Expected:** within **±0.02** of the 0.6475 baseline — more labels but noisier, roughly cancelling. Slight lean to a small *gain* in recall and a small *loss* in precision, since the extra buildings are real but their outlines are less reliable.

**[`2026-09-10-merge-split-rate`](2026-09-10-merge-split-rate/)** — ✅ done
> **Why:** [D4](../2026-09-10-d4-adjacency/) measured **78 % of Jaipur buildings touching a neighbour**. `MASTER_CONTEXT` §6.2 states that pixel IoU cannot detect instance merging and boundary IoU largely cannot either. So every IoU in this repo has been silent about a failure mode affecting most buildings — and the pipeline ends in a **per-building kW estimate**, where merging two houses corrupts the count,
>
> **Expected:** merge rate 20–35 % for ResNet-34, with MiT-B2 a few points lower.

**[`2026-09-10-segformer-backbone`](2026-09-10-segformer-backbone/)** — ✅ done — modest win
> **Why:** DAFormer's central empirical claim is that the *architecture* mattered more than the adaptation algorithm — self-attention features are less domain-specific than early convolutional ones. `plan/03` §2.3 asks for this table regardless of which wins.
>
> **Expected:** **+0.03 to +0.08 IoU** (so ~0.68–0.73). A transformer's global receptive field should help most where buildings are dense and share walls — exactly Jaipur, and exactly the instance-merging failure the project worries about. Risk: 7,371 crops is small for a transformer, which may underperform at this data scale.

**[`2026-09-10-selftrain-round2`](2026-09-10-selftrain-round2/)** — ❌ negative — one round is the recipe
> **Why:** rounds are the cheapest remaining lever — no new data, no new architecture, just another pass. If they compound, the Jaipur transfer should run 2–3 rounds. If they saturate or degrade, one round is the recipe and the extra compute goes elsewhere.
>
> **Expected:** (0.819 → 0.819) and recall moved −0.004. The pre-registered failure mode — round 2 landing below round 1 through error amplification — did not occur either: this is not drift, it is a null result.

**[`2026-09-10-selftrain-threshold-cliff`](2026-09-10-selftrain-threshold-cliff/)** — ✅ done — cliff located; stated mechanism refuted
> **Why:** a Jaipur transfer has no labels, so the threshold must be chosen blind. Knowing *where* the cliff is — and how sharp — determines how much margin to leave.
>
> **Expected:** thr **0.0100** → **0.42–0.52** (already deep in the noise floor, so most of the damage should already be done); thr **0.0015** → **0.37–0.45** (essentially S2). If 0.0100 lands near 0.6 instead, the cliff is sharper and further down than the distribution suggests, and threshold choice is safer than I think.

**[`2026-09-10-solar-google-to-ign`](2026-09-10-solar-google-to-ign/)** — ✅ done
>
> **Expected:** in-domain google val IoU **0.82–0.86** (prior solar runs reached ~0.85). On IGN, a drop to **0.55–0.70**. The GSD ratio here is only 2× versus Stage 1's 3.55×, and panels are far more visually distinctive than roofs, so I expect a smaller relative drop than the rooftop domain gap — but a clear one.

**[`2026-09-10-solar-self-training`](2026-09-10-solar-self-training/)** — ✅ done — **strong negative result**
> **Why:** Jaipur has zero labels, so a self-training run there can never be scored, only argued about. Both BDAPPV domains are labelled, so every knob — threshold policy, class ratio, number of rounds — can be tuned against a real number, then frozen and transferred. That is `MASTER_CONTEXT`'s ordering principle, and this is the first run to actually exercise it.
>
> **Expected:** I expect this to **fail or barely move** — IGN IoU **0.52–0.60**, i.e. plausibly *below* the 0.5611 source-only baseline. Pseudo-labels drawn at a 0.0004 threshold are close to noise-shaped, and training on them should teach over-prediction. If it lands above 0.60 I will have badly misread the threshold evidence.

**[`2026-09-10-solar-selftrain-conf`](2026-09-10-solar-selftrain-conf/)** — ✅ done — **self-training works; CBST was the problem**
> **Why:** S2 cost **18.6 points** of target IoU (0.5611 → 0.3752) with precision falling 0.741 → 0.391. The suspected cause is CBST's ratio policy: forcing the source's 1.83 % foreground on a target whose confidence distribution is crushed drove the threshold to **0.0004**, so pseudo-labels were largely noise.
>
> **Expected:** **0.50–0.58** — recovering most of the loss but landing at or slightly below the 0.5611 source-only baseline. Reasoning: a sane threshold stops the noise-labelling, but self-training can still only reinforce what the model already believes, and the source-only model is wrong about 44 % of the target. Precision should recover to ~0.65–0.75.

**[`2026-09-10-split-metric-audit`](2026-09-10-split-metric-audit/)** — ✅ done — metric vindicated, interpretation corrected
> **Why:** merge and split rates are the metrics this project reports *instead of* IoU, on the grounds that IoU cannot see instance errors. If they are themselves blind, the argument collapses.
>
> **Expected:** the published split rates are wrong and will move once recomputed with the loose (≥ 10 %) definition.

**[`2026-09-11-inference-threshold-sweep`](2026-09-11-inference-threshold-sweep/)** — ✅ done — complements, not substitutes; follow-up launched
> **Why:** the teacher's merge rate is **0.3155** — a third of buildings still fused, and after tonight's work that is the largest remaining weakness in Stage 1. If a threshold change buys a meaningful part of what erosion buys, it is strictly cheaper. If it does not, that is itself informative: it would mean fusion is happening in the model's *confident* interior rather than at soft edges, which erosion can
>
> **Expected:** I expected thresholding to trace a *worse* merge/missed trade than erosion. Matched at equal merge rate, the two curves are nearly the same:

**[`2026-09-11-mit-b5`](2026-09-11-mit-b5/)** — ❌ negative — +0.0063 IoU (1.5× noise) for 3.6× compute; B2 stays
> **Why:** the median Jaipur building is ~30×30 px (D2/D3: 913 px against AIRS's 21,084). Small-object segmentation is usually capacity- and receptive-field-limited, and MiT-B5 has a wider global receptive field — the property that made B2 exploit label gaps ResNet could not.
>
> **Expected:** against the B2 mean (+0.0087 against seed 42 alone) ✅. "`pred/label` will move less than IoU" — ambiguous as I wrote it: in absolute terms it moved *more* (0.0756 vs 0.0063), but relative to each metric's own noise it moved **less** (1.0× vs 1.5× its 2 sd). The intended meaning holds; the wording did not.

**[`2026-09-11-replication`](2026-09-11-replication/)** — ✅ done — claim 1 promoted to solid, claim 2 demoted to directional
> **Why:** every remaining backlog item is blocked (R12 on hand labels, S3 on absent data) or low-value (R7, since the AIRS seed was shown worthless). Turning the project's central claim from "probably" into "measured" is worth more than another arm.
>
> **Expected:** The more interesting result: self-training is 2.7× noisier.** The t050 arm's `pred/label` sd is **0.1013** against the teacher's **0.0382**, from a 0.1432 range across just two seeds. Both arms trained on the *same fixed* pseudo-label set — generated once from the seed-42 teacher — so this is not pseudo-label variability. **Training a student on noisy pseudo-labels amplifies its sensitivity to ini

**[`2026-09-11-seed-variance`](2026-09-11-seed-variance/)** — ★ done (n=3) — `pred/label` 2 sd = 0.076; flagship number needs an error bar
> **Why:** it retroactively determines which of tonight's conclusions are real. Three in particular hinge on it:
>
> **Expected:** Two runs of the identical configuration differ by 0.0495 merge and 0.0597 `pred/label` — larger than several differences already written up as findings. IoU is the *stable* metric here, at 0.0030; the metrics this project elevated over IoU precisely because they capture instance structure are the ones that swing.

**[`2026-09-11-selftrain-eroded-pseudo`](2026-09-11-selftrain-eroded-pseudo/)** — ❌ negative — diagnosis confirmed, teacher still wins
> **Why:** it decides whether "self-training fails on Stage 1" is a fact about the method or an artefact of one fixable implementation detail. R5's conclusion is currently the stronger claim and this is the test that could weaken it.
>
> **Expected:** Eroding the pseudo-labels recovers most of what raw self-training threw away: `pred/label` **0.9134 → 0.9672** (under-counting 8.7 % → 3.3 %) and merge **0.3371 → 0.2666**, now *better* than the teacher's 0.3155. The mechanism proposed in R5 — that self-training discards the label erosion — is confirmed by repairing exactly that and watching both metrics move back.


**27 experiments written up.** Status legend: ✅ done · ❌ negative result (kept deliberately) · ⚠️ confounded or unresolved · running.

---

## 2. Full commit history, newest first

167 commits. Each is a unit of work — a run launched, a result recorded, a bug found, a document corrected.

### 2026-09-11  ·  33 commits

- `14:30` **ae94c01** Regenerate ledger and timeline (un-eroded s43 ep32)
- `14:23` **252be15** Replication, claim 2: direction survives, precision does not, and self-training is 2.7x noisier
- `14:00` **868829f** Regenerate ledger and timeline (replication: un-eroded s43 ep21, st050 s43 ep15)
- `13:58` **563226d** Regenerate ledger and timeline (replication: un-eroded s43 ep20, st050 s43 ep14)
- `13:30` **7cc5cf4** Regenerate ledger and timeline (replication: un-eroded s43 ep8, st050 s43 ep7)
- `13:00` **7e3f5c7** Regenerate ledger and timeline (both replication arms in cache-load phase)
- `12:59` **19d6411** Replicate the two claims sitting closest to the noise floor
- `12:40` **7c1e9e6** MiT-B5: +0.0063 IoU at 1.5x noise for 3.6x compute -- B2 stays the default
- `12:32` **4399f9e** Seed variance at n=3: the flagship pred/label number needs an error bar
- `08:29` **e29b7db** Regenerate ledger and timeline (seed-44 ep13, MiT-B5 ep13)
- `08:00` **08e9f55** Regenerate ledger and timeline (seed-44 ep1, MiT-B5 ep9)
- `07:58` **597c0dd** Regenerate ledger and timeline (seed-44 starting, MiT-B5 ep9)
- `07:54` **4a5a436** Seed replicate: instance metrics have a +/-0.06 floor, and it weakens three conclusions
- `07:29` **5085ae9** Regenerate ledger and timeline (seed-43 ep23 at 0.6412, MiT-B5 ep6)
- `07:00` **49e57c2** Regenerate ledger and timeline (seed-43 ep11, MiT-B5 ep2)
- `06:59` **02c12b9** Quota recovery: delete 13 epoch snapshots, keep every best.pth
- `06:31` **47ce1e3** Launch seed-variance replicate and MiT-B5
- `06:28` **5ac635f** Un-eroded curve settles it: erosion is necessary, not redundant with thresholding
- `06:00` **5927321** Regenerate ledger and timeline (un-eroded ep32 at 0.6571)
- `05:58` **a45ea00** Regenerate ledger and timeline (un-eroded retrain ep31 at 0.6559)
- `05:37` **13251b8** E04's curve is compressed, not coincident -- and my criterion was defective again
- `05:13` **6d6b2e6** R5b complete: erosion strength is a smooth dial, and no setting beats the teacher
- `05:00` **39b28ee** Regenerate ledger and timeline (E02 ep37, un-eroded ep7)
- `04:58` **bad1a07** Regenerate ledger and timeline (E02 ep36, un-eroded retrain ep6 at 0.6408)
- `04:39` **7702f09** Inference threshold traces the same merge curve as erosion -- but erosion wins on counting
- `04:19` **48fe3f5** R5b E04: diagnosis confirmed, method still rejected
- `04:00` **e11d62d** Regenerate ledger and timeline (R5b E04 ep33, E02 ep21)
- `03:58` **942a661** Regenerate ledger and timeline (R5b E04 ep32, E02 ep20)
- `03:29` **149f128** Regenerate ledger and timeline (R5b E04 ep20, E02 ep13)
- `03:00` **f39e677** Regenerate ledger and timeline (R5b E04 ep8, E02 ep5)
- `02:58` **e71d07a** Regenerate ledger and timeline (R5b E04 ep7, E02 ep5)
- `02:36` **a8e52e5** R5 rejected: self-training discards the label erosion. R5b launched as the repair
- `00:00` **cb70fd3** Regenerate ledger and timeline (arm A ep16, arm B ep14)

### 2026-09-10  ·  73 commits

- `23:58` **118319e** Regenerate ledger and timeline (arm A ep15 plateauing 0.5606, arm B ep13 at 0.6395)
- `23:30` **3bae58a** Regenerate ledger and timeline (R5 arm A ep9, arm B ep6 at 0.6284)
- `23:00` **eb5effb** Regenerate ledger and timeline (R5 arm A ep3)
- `22:58` **4cebdb3** Regenerate ledger and timeline (R5 arm A ep2, arm B starting)
- `22:55` **c906b01** Retract the "model over-predicts" finding; --limit sampled one corner of Jaipur
- `22:43` **fee4295** R5 arm A launched (thr 0.80); correct arm-A rationale — 600-crop probe was unrepresentative, thr 0.80 selects 17.2% not 23%
- `22:41` **57fa38f** Crossover located at ratio 0.0091; Jaipur self-training launched
- `22:23` **df8097b** S7 round 2 saturates; R11 audit corrects the 0.8 m diagnosis
- `22:00` **44d03f8** Regenerate ledger and timeline (round2 ep27, r010 ep22)
- `21:58` **5224391** Regenerate ledger and timeline (round2 ep27, r010 ep21)
- `21:30` **efe1381** Regenerate ledger and timeline (round2 ep21, r010 ep17)
- `21:00` **ebea40c** Regenerate ledger and timeline (round2 ep15, r010 ep12)
- `20:58` **218b526** Regenerate ledger and timeline (round2 ep15, r010 ep11)
- `20:30` **ac10fbe** Regenerate ledger and timeline (round2 ep9, r010 ep7)
- `20:01` **c705744** Regenerate ledger and timeline (S7 round2 through ep3)
- `19:58` **7337c57** Regenerate ledger and timeline (S7 round2 ep2, ratio-0.010 arm starting)
- `19:55` **8b78c93** S6 threshold cliff: there is no cliff, there is a plateau
- `19:30` **afc8b68** Regenerate ledger and timeline (S6 arms at ep29/ep27 of 30)
- `19:01` **2144180** Regenerate ledger and timeline (S6 arms mid-flight: r008 ep18, r012 ep17)
- `19:00` **56d831a** Visual label review, and the wrong conclusion it nearly produced
- `18:34` **2df17c6** R12: stratified hand-labelling package for 30 Jaipur crops
- `18:14` **c17d43e** docs: session note for 2026-09-10, the experiment-runner day
- `18:09` **9b559ad** perf: contiguous cache -- the list version was 8.4x slower than its own timer said
- `17:58` **37aee53** chore: regenerate ledger and timeline
- `17:50` **bb16f0d** chore: park the 4.2 GB crop set on host /tmp -- quota 2.0 -> 6.1 GB free
- `17:31` **95efd6d** chore: free quota safely, and write down the deletion order
- `17:16` **809c8a2** feat: solar RAM cache (S5); launch the self-training threshold cliff (S6)
- `17:00` **a11ff32** result: self-training WORKS -- CBST's ratio policy was the entire problem
- `16:32` **aee4f5b** result: erosion sweep complete -- monotonic curve, 0.4 m is the only pred/label ~ 1
- `16:01` **418386d** chore: queue S5 -- port --cache_ram to train_solar.py
- `15:58` **dcd7239** chore: regenerate ledger and timeline
- `15:30` **9808420** chore: regenerate ledger and timeline
- `15:01` **5282e4e** chore: regenerate ledger and timeline
- `14:58` **bab38c8** chore: regenerate ledger and timeline
- `14:33` **6e4ff16** feat: 0.2 m erosion arm, first run using --cache_ram
- `14:09` **84f8092** chore: regenerate ledger and timeline
- `14:08` **f3814ec** result: CBST self-training destroys the solar domain gap -- 0.5611 -> 0.3752
- `13:32` **cb82ec6** result: 0.4 m arm reproduced and measured honestly -- pred/label 0.9914
- `13:00` **550ed79** chore: regenerate ledger and timeline
- `12:58` **5dcc98d** chore: regenerate ledger and timeline
- `12:54` **6574975** perf: --cache_ram makes training 3.7x faster; we were dataloader-bound
- `12:30` **d438820** chore: regenerate ledger and timeline
- `12:01` **30e6688** chore: regenerate ledger and timeline
- `11:58` **26e30ab** chore: regenerate ledger and timeline
- `11:30` **137a95a** chore: regenerate ledger and timeline
- `11:09` **7c262fd** chore: regenerate ledger and timeline
- `11:08` **7f62d77** docs: group the mistakes into three root patterns, and relax the quota rule
- `10:45` **2248231** fix: split metric now sees fragmentation; ledger glob missed two whole runs
- `10:38` **4913d59** fix: split-rate metric was blind to fragmentation; queue hand-labelling as R12
- `10:22` **97b8734** result: 0.8 m over-erodes -- and my split-rate metric is blind to it
- `10:17` **4e35d43** feat: CBST self-training on google->ign, plus a finding that arrived before it ran
- `09:40` **0a28fb9** docs: put the reasoning in the timeline, not just the results
- `09:38` **48c2ad4** feat: generated TIMELINE.md -- what was run, what wasn't, in order
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

- queued — R12 · Hand-label ~30 Jaipur tiles ★★★ nothing else can be validated without it
- queued — R6 · Multi-source co-training
- queued — R7 · Low-resolution simulation
- ⛔ **blocked** — S3 · Fix the zero-negatives bug (`MASTER_CONTEXT` C1) ⚠ BLOCKED — raw BDAPPV absent

---

## 4. How to regenerate

```bash
python scripts/build_run_ledger.py   # metrics first
python scripts/build_timeline.py     # then the timeline
```
