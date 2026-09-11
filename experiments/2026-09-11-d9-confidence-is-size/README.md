# 2026-09-11-d9-confidence-is-size — can Open Buildings' own confidence validate its small polygons?

| | |
|---|---|
| **Status** | ✅ done — no, and the reason matters |
| **Date** | 2026-09-11 |

## Question

D8 left two hypotheses it could not separate: the model genuinely fails on small buildings **(a)**,
or Open Buildings invents small polygons that were never buildings **(b)**. I said hand labels
(R12) were the only way to tell.

Before accepting a 3-hour experiment is blocked, it is worth checking whether OB's **own
per-polygon confidence** can arbitrate. If OB is systematically unsure about its small polygons,
that is evidence for (b) — obtainable in seconds from a CSV already on disk.

**Prediction:** OB confidence will be lower for small polygons, but *weakly* — enough to be
suggestive, not enough to settle it. I expect the smallest bin to average ~0.05–0.10 below the
largest, and I expect to be able to test (b) by rebuilding val labels at a higher threshold and
re-running D8.

## Results

**Confidence rises monotonically with size across all 523,283 Jaipur polygons:**

| size (px) | area (m²) | n | mean confidence | frac < 0.75 | frac ≥ 0.90 |
|---|---|---|---|---|---|
| 50–200 | 4–14 | 36,225 | **0.6998** | **0.8900** | 0.0001 |
| 200–400 | 14–28 | 81,215 | 0.7201 | 0.7364 | 0.0001 |
| 400–900 | 28–64 | 140,977 | 0.7509 | 0.5006 | 0.0004 |
| 900–2000 | 64–142 | 158,250 | 0.7951 | 0.2215 | 0.0112 |
| 2000+ | 142+ | 106,407 | **0.8347** | 0.0677 | 0.0971 |

The gap is 0.135 — larger than predicted. But the decisive number is the next table.

**Polygons surviving a raised threshold, starting from the label set's current conf ≥ 0.75:**

| size (px) | n @ 0.75 | n @ 0.80 | n @ 0.85 | keep @ 0.80 | keep @ 0.85 |
|---|---|---|---|---|---|
| 50–200 | 3,983 | 428 | **20** | 0.107 | **0.005** |
| 200–400 | 21,410 | 4,569 | 253 | 0.213 | 0.012 |
| 400–900 | 70,407 | 29,233 | 4,775 | 0.415 | 0.068 |
| 900–2000 | 123,192 | 79,350 | 28,339 | 0.644 | 0.230 |
| 2000+ | 99,200 | 80,460 | **44,719** | 0.811 | **0.451** |

## Interpretation

**The planned test is impossible, and that is the finding.** Raising the confidence threshold to
0.85 leaves **20 small polygons in the entire city** against 44,719 large ones. There is no
high-confidence small-building population to measure, so "are OB's *confident* small buildings
still missed?" has no answerable form.

**Open Buildings' confidence is largely a proxy for size.** Filtering by confidence *is*
filtering by size. The two cannot be varied independently in this dataset, so OB's metadata
cannot arbitrate between (a) and (b) — not weakly, as I predicted, but not at all.

**Prediction wrong in both directions.** The confidence gap was *larger* than I guessed (0.135
vs 0.05–0.10), and the follow-up test I assumed I could run is unrunnable. Being more separated
than expected is exactly what makes it useless as an instrument.

### The label set's confidence threshold is a size filter nobody registered as one

`min_conf 0.75` — inherited as a reasonable-looking default — keeps:

- **11 %** of the smallest buildings (3,983 of 36,225)
- **93 %** of the largest (99,200 of 106,407)

So the training labels were already heavily size-filtered before any experiment ran, and the
small buildings that remain are OB's *most confident* small ones. **The model still misses 75 %
of them** (D8). That cuts against the lazier version of (b): these are not OB's marginal
guesses, they are its best small-building calls.

### This retroactively explains why R3 was confounded

R3 (label-confidence sweep) is marked *"⚠️ confounded — each model wins on its own label
distribution"*, with no mechanism given. The mechanism is here: **varying the confidence
threshold varies the size distribution of the labels.** A model trained at conf 0.85 is not
seeing the same task with cleaner labels — it is being trained almost entirely on large
buildings. R3 compared models across different size regimes and scored each on its own regime.

That is a sharper statement than "confounded", and it belongs in R3's write-up.

## Decision

- [x] **(a) vs (b) is genuinely blocked on R12.** Not for want of trying a cheaper route — the
      cheaper route is provably unavailable.
- [x] **Record that `min_conf` is a size filter.** Any future change to it silently changes which
      buildings the model is asked to find, and that must be stated whenever the threshold moves.
- [x] **R3's confounding now has a named mechanism** — cross-referenced there.
- [ ] Do not raise `min_conf` to "improve label quality" without accounting for the size shift.
      It would look like a cleaner label set and be a different task.

## Threats to validity

- `area_in_meters` is OB's own area estimate; the px conversion uses the measured 26.6 cm/px
  GSD, so bin edges are approximate at the boundaries. The effect is far too large to be a
  binning artefact.
- City-wide statistics; D8's miss rates are measured on the val tiles only. The confidence/size
  relationship is a property of the OB dataset and is not expected to differ by tile, but this
  was not checked per tile.
- Confidence semantics are OB's, undocumented in detail here beyond being a per-polygon score.
