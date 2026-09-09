# 2026-09-09-method-comparison — which adaptation method actually wins on Jaipur?

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-09 |
| **Commit** | `2fcfef0` on `feat/init-project-setup` |
| **Supersedes** | the Tier-1 recommendations in `plan/03-methods-and-literature.md` |

## Question

Run six adaptation methods against **one** held-out eval set and rank them. `plan/03` ranks
methods by literature and reasoning; nothing in the repo had ever compared them on this data.

**Prediction:** Tier-1 photometric methods (histogram matching, FDA) give small positive
gains — `plan/03` calls Tier 1 *"nearly free"* wins. AdaBN gives a real gain. Weak
supervision wins overall.

## Setup

Same 1,701 held-out crops (tiles `1-1`, `2-2`, `4-3`), same U-Net/ResNet-34, same metric,
**global** IoU (dataset-level, not the mean of per-crop IoUs — per-crop averaging lets a
crop with three foreground pixels weigh as much as a dense one, which flatters exactly the
under-prediction being measured).

```bash
./run_docker.sh 7 "python -u scripts/compare_methods.py \
    --weak_ckpt experiments/2026-09-09-weak-supervision-jaipur/checkpoints/*/best.pth \
    --out diagnostics/method_comparison.json"
```

## Results

Raw: [`diagnostics/method_comparison.json`](../../diagnostics/method_comparison.json)

| Method | Tier | IoU @0.50 | IoU @best thr | best thr | Prec | Rec |
|---|---|---|---|---|---|---|
| **weak + TTA** | 4 | **0.6534** | 0.6534 | 0.50 | 0.744 | 0.843 |
| **weak** | 4 | **0.6476** | 0.6476 | 0.50 | 0.736 | 0.844 |
| adabn | 1 | 0.2523 | 0.2941 | 0.01 | 0.423 | 0.491 |
| seed (no adaptation) | 0 | 0.1419 | 0.3329 | 0.01 | 0.528 | 0.474 |
| seed + TTA | 1 | 0.1331 | **0.3863** | 0.01 | 0.529 | 0.589 |
| fda | 1 | 0.0891 | 0.2735 | 0.01 | 0.580 | 0.341 |
| histmatch | 1 | 0.0258 | 0.1117 | 0.01 | 0.428 | 0.131 |

## Interpretation

**1. Weak supervision wins by 4.6×** and it is not a close contest.

**2. Photometric alignment does not merely underperform — it actively harms.** `plan/03`
lists histogram matching and FDA as Tier-1 near-free wins. Measured, both are **worse than
doing nothing**: histmatch takes the seed from 0.142 to 0.026, FDA to 0.089. Plausible
cause: AIRS is vegetated suburban Christchurch, Jaipur is dense arid rooftop. Forcing
Jaipur's colour distribution onto a green-heavy reference destroys the roof/ground contrast
the model relies on. **Recommend deleting both from the plan rather than scheduling them.**

**3. Most of the seed's "domain gap" is calibration, not representation.** The seed scores
**0.142 at threshold 0.5 and 0.333 at its best threshold** — same weights, same pixels,
2.3× the score from one dial. A large part of what looked like catastrophic domain failure
is a model calibrated for 15 % foreground meeting a 28 % domain.

**4. AdaBN's advantage is mostly that same recalibration.** At a fixed 0.5 threshold AdaBN
looks strong (0.252 vs 0.142). Allow both a threshold sweep and plain **seed+TTA (0.386)
beats AdaBN (0.294)**. So AdaBN is not learning target-specific features; it is re-centring
statistics, which a threshold sweep achieves for free. Report it as an ablation; do not
build on it.

**5. Every zero-training method is still pinned at the sweep's floor (0.01).** The seed's
operating point on Indian imagery is off the bottom of the usual range. Any future
comparison must sweep this low or it will report an optimum it never found — the first two
runs of this experiment bottomed out at 0.30 and then 0.05 and were both wrong.

## Decision

- [x] **Weak supervision is the primary track.** Consistent with `plan/`'s own bottom line
      ("build the data, then adapt") — now with numbers.
- [x] **Drop histmatch and FDA.** Measured harmful here.
- [x] **Keep TTA** — free, and +0.006 on the weak model.
- [ ] Next: boundary-relaxed loss (targets the precision/recall gap), self-training on top
      of the weak model, and SegFormer vs ResNet-34.

## Threats to validity

- **Eval labels are Open Buildings ground footprints, not roof outlines.** Every number is
  agreement-with-footprints, not roof accuracy.
- **`weak` trained on this exact label distribution — home-field advantage.** Comparing the
  Tier-1 methods *against each other* is clean; comparing them against `weak` is not. The
  fair reading is "free labels beat zero-label tricks", not "0.65 is the roof accuracy".
- The best-threshold column is tuned **on the eval set**: an upper bound, not a fair score.
  The @0.50 column is the honest one for the zero-training methods.
- Single reference image for histmatch/FDA. A better reference might make them less bad,
  though a 5× deficit is unlikely to be reference choice alone.
- One seed, one run, three val tiles.

## Reproduce

`scripts/compare_methods.py`. Needs `data/jaipur_weak/`, the AIRS seed checkpoint, and at
least one AIRS image in `data/airs/image/` as the photometric reference.
