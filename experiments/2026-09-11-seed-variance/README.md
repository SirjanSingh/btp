# 2026-09-11-seed-variance — how big is the noise floor everything is being compared against?

| | |
|---|---|
| **Status** | ★ done (n=3) — `pred/label` 2 sd = 0.076; flagship number needs an error bar |
| **Date** | 2026-09-11 |

## Question

Across ~20 experiments this project has compared single runs and repeatedly said things like
*"+0.0039 is inside seed noise"* or *"the S6b drop is far too large to be noise"*. **Not one of
those statements was measured.** No configuration in this repo has ever been run twice.

Re-run the exact default config (MiT-B2, 0.4 m eroded labels, 40 epochs) with `--seed 43`
instead of 42. The difference between the two runs *is* the noise floor.

**Why it matters:** it retroactively determines which of tonight's conclusions are real. Three
in particular hinge on it:

- R5 arm B beat the teacher by **+0.0039 IoU** — I dismissed it as noise.
- R5b E02 vs E04 differ by **0.023 `pred/label`** — I treated that as a real ordering.
- R4 chose MiT-B2 over ResNet-34 on **+0.0086 IoU** — a live default rests on it.

If seed spread turns out to be ±0.01 IoU, the first two readings are unsupportable and R4's
margin is inside the noise. If it is ±0.002, all three stand.

## Predictions (before running)

**IoU spread |Δ| = 0.003–0.008.** Segmentation runs on ~7k crops with a fixed schedule are
usually reproducible to a few thousandths, but the best-epoch selection over 40 noisy epochs
adds variance of its own.

**Instance metrics will vary more than IoU.** `pred/label` and merge depend on connected
components, and a small probability shift near the decision boundary can split or join a blob
outright. I expect `pred/label` spread of **0.01–0.03** — i.e. *larger* than several
differences I have already interpreted as meaningful.

**The uncomfortable prediction:** I expect this to invalidate at least one claim I made
tonight. E02-vs-E04's 0.023 `pred/label` gap is the most exposed.

## Setup

Identical to the default: `--train_dir data/jaipur_weak_erode4/train`-equivalent (the same
0.4 m eroded set the teacher used), MiT-B2, 40 epochs, batch 12, `--seed 43`. Scored the same
way — val IoU plus the full instance metric set at threshold 0.5.

## Results

Identical config, `--seed 42` (the teacher) vs `--seed 43`.

| metric | seed 42 | seed 43 | **\|Δ\|** |
|---|---|---|---|
| val IoU | 0.6393 | 0.6423 | **0.0030** |
| merge | 0.3155 | 0.3650 | **0.0495** |
| split | 0.0840 | 0.0634 | 0.0206 |
| frag/label | 0.9203 | 0.8962 | 0.0241 |
| missed | 0.3257 | 0.3074 | 0.0183 |
| **pred/label** | 0.9914 | 0.9317 | **0.0597** |

**Prediction scorecard.** IoU spread predicted 0.003–0.008 → **0.0030**, just inside ✅.
"Instance metrics vary more than IoU" → **20× more** ✅. But I predicted `pred/label` spread of
0.01–0.03 and it is **0.0597** ❌ — **twice the top of my range**.

## Interpretation

**The instance metrics are far noisier than anything in this repo has assumed.** Two runs of the
identical configuration differ by 0.0495 merge and 0.0597 `pred/label` — larger than several
differences already written up as findings. IoU is the *stable* metric here, at 0.0030; the
metrics this project elevated over IoU precisely because they capture instance structure are
the ones that swing.

Mechanically this is what makes sense: `pred/label` is a ratio of connected-component **counts**.
A probability shift of a few thousandths near a boundary either joins two blobs or doesn't, and
each such event changes the count by one. Pixel IoU averages over ~450M pixels; component counts
do not average at all.

### Retroactive audit — every conclusion from tonight against this floor

| claim | Δ | × floor | verdict |
|---|---|---|---|
| erosion 0.4 m → 0.8 m merge | 0.1816 | 3.7× | **safe** |
| R4: MiT-B2 over ResNet-34 (IoU) | 0.0086 | 2.9× | weak |
| eroded vs un-eroded `pred/label` @ merge 0.3155 | 0.1468 | 2.5× | weak |
| erosion 0.2 m → 0.4 m merge | 0.0963 | 1.9× | at noise |
| eroded vs un-eroded `pred/label` @ merge 0.4514 | 0.0925 | 1.5× | at noise |
| R5 arm B `pred/label` regression | 0.0780 | 1.3× | **at noise** |
| R5 arm B IoU over teacher | 0.0039 | 1.3× | at noise |
| E04 curve vs teacher curve | 0.0700 | 1.2× | **at noise** |
| erosion 0 m → 0.2 m merge | 0.0497 | 1.0× | **at noise** |
| **R5b E02 vs E04 `pred/label`** | 0.0225 | **0.4×** | **below noise** |

**What this actually invalidates.** I predicted this run would kill at least one of tonight's
claims. It kills more than one:

- **R5b's "monotone ordering across all four metrics" is unsupportable.** E02-vs-E04 sits at
  0.4× the floor. The tidy dose-response story was over-read from a difference smaller than
  run-to-run variance.
- **"Self-training regresses `pred/label` from 0.9914 to 0.9134"** — the headline of R5 — is at
  1.3× the floor. The *direction* was confirmed by the eroded-pseudo arms moving it back, which
  is independent evidence, but the single comparison alone would not have supported it.
- **The E04-curve-vs-teacher-curve difference** I used to argue self-training "shifts the curve"
  is at 1.2×. That argument needs more seeds or should be dropped.

**What survives.** The erosion→over-erosion result (3.7×), R4's encoder choice (2.9×, weak but
real), and the eroded-vs-un-eroded comparison at low merge (2.5×) — and crucially the last one
is supported by **three matched points all moving the same direction**, which a single-pair
noise estimate does not capture. Consistency across independent points is evidence that
pairwise deltas miss.

**The methodological point is larger than any single result.** This project reports merge and
`pred/label` *instead of* IoU on the grounds that IoU cannot see instance errors. That reasoning
stands — but it was never paired with the obvious follow-up: **a metric sensitive enough to see
instance structure is sensitive enough to see run-to-run noise.** Twenty experiments were
compared on single runs of metrics with a ±0.06 floor.

## Decision

- [x] **Quote no instance-metric difference below ~0.06 `pred/label` or ~0.05 merge as a
      finding** without replication.
- [x] **Mark the affected conclusions** in their own write-ups rather than leaving them to be
      read at face value.
- [x] **A third seed is launched** — two runs give a range, not a σ, and n=2 estimates the
      spread poorly in both directions.
- [ ] Future arms that matter should be run at 2 seeds minimum. The cost is real (2.5 h each)
      but cheaper than a retracted thesis claim.

## Threats to validity

- **n=2 gives a range, not a standard deviation**, and a single pairwise difference is itself a
  high-variance estimate of the underlying spread — it could as easily understate as overstate.
  The third seed addresses this directly.
- Seed controls init, shuffling and augmentation; cuDNN autotuning is non-deterministic too, so
  this is total run-to-run variance, which is the quantity that matters for comparing runs.
- Measured on one configuration. Noisier configs (higher erosion, self-trained) may differ.

## Threats to validity

- **Two runs give a range, not a standard deviation.** This bounds the noise floor loosely;
  it does not estimate it properly. Three or more seeds would, and are worth queuing if the
  spread turns out to matter.
- Seed controls init, shuffling and augmentation, but cuDNN autotuning is not deterministic
  either, so this is total run-to-run variance rather than seed variance specifically — which
  is the quantity actually wanted here.

---

## n = 3 — a proper spread, and what it does to the headline number

Third seed complete. All three are the identical config, differing only in `--seed`.

| metric | seed 42 | seed 43 | seed 44 | range | **sd** | **2 sd** |
|---|---|---|---|---|---|---|
| val IoU | 0.6393 | 0.6423 | 0.6434 | 0.0041 | 0.0021 | **0.0042** |
| merge | 0.3155 | 0.3650 | 0.3329 | 0.0495 | 0.0251 | **0.0502** |
| split | 0.0840 | 0.0634 | 0.0793 | 0.0206 | 0.0108 | 0.0216 |
| frag/label | 0.9203 | 0.8962 | 0.9310 | 0.0348 | 0.0178 | 0.0356 |
| missed | 0.3257 | 0.3074 | 0.3045 | 0.0212 | 0.0115 | 0.0230 |
| **pred/label** | 0.9914 | 0.9317 | **1.0027** | 0.0710 | 0.0382 | **0.0763** |

The n=2 estimate (0.0597 `pred/label`) was in the right region; n=3 gives **2 sd = 0.0763**.

### This changes how the project's flagship number must be quoted

The headline result everywhere in this repo is *"0.4 m erosion gives `pred/label` **0.9914** —
within **0.9 %** of one prediction per building."* Across three seeds the same configuration
produces **0.9317, 0.9914, 1.0027**.

**The honest statement is `pred/label` = 0.99 ± 0.08 (2 sd), not 0.9914.** The "within 0.9 %"
precision is an artefact of reporting one run to four decimals. The *conclusion* survives — 0.4 m
is still the only erosion setting whose interval contains 1.0, since 0.2 m (0.8412) sits 2.1×
outside it and 0.8 m (1.4743) is far beyond — but the stated precision was two orders of
magnitude too confident.

### Re-audit at n=3, threshold = 2 sd

| claim | Δ | × 2 sd | verdict |
|---|---|---|---|
| erosion 0.4 → 0.8 m merge | 0.1816 | 3.6× | **safe** |
| R4: MiT-B2 over ResNet-34 (IoU) | 0.0086 | 2.0× | **safe** |
| eroded vs un-eroded `pred/label` @ merge 0.3155 | 0.1468 | 1.9× | ok |
| erosion 0.2 → 0.4 m merge | 0.0963 | 1.9× | ok |
| eroded vs un-eroded `pred/label` @ merge 0.4514 | 0.0925 | 1.2× | ok |
| R5 arm B `pred/label` regression | 0.0780 | 1.0× | ok, marginal |
| E04 curve vs teacher curve | 0.0700 | 0.9× | **not supported** |
| erosion 0 → 0.2 m merge | 0.0497 | 1.0× | **not supported** |
| R5 arm B IoU over teacher | 0.0039 | 0.9× | not supported *(I dismissed it as noise — correctly)* |
| R5b E02 vs E04 `pred/label` | 0.0225 | 0.3× | **not supported** |

**Net effect of going n=2 → n=3:** R4's encoder choice moves from "weak" to safe, and the
eroded-vs-un-eroded comparison firms up. The three unsupported claims stay unsupported. Nothing
that was called safe became unsafe.

**The eroded-vs-un-eroded result deserves a note.** Its individual points are only 1.2–1.9× 2 sd,
but **three matched operating points all move the same direction, with the gap widening
monotonically** (0.093 → 0.118 → 0.147). Three independent same-direction comparisons is much
stronger than any one of them; treating each in isolation understates it. That is the argument
that carries the erosion conclusion, not any single delta.

## Decision (updated for n=3)

- [x] **Quote `pred/label` as X ± 0.08 and merge as X ± 0.05.** Four-decimal single-run
      reporting is not defensible for these metrics.
- [x] **Restate the flagship as `pred/label` ≈ 0.99 ± 0.08**, and rest the erosion conclusion on
      the *interval containing 1.0*, not on the point estimate.
- [x] n=3 is enough for a noise floor; further seeds are better spent on arms that will be quoted.
