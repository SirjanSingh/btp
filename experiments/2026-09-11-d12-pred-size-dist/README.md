# 2026-09-11-d12-pred-size-dist — can the model produce small components at all?

| | |
|---|---|
| **Status** | ★ done — capacity hypothesis refuted; resolution experiment cancelled |
| **Date** | 2026-09-11 |

## Question

D11 isolated the one real remaining Stage-1 error: **62 % of small labelled buildings receive
essentially no prediction** (under 5 % coverage at any association threshold). Whether those
labels are genuine is blocked on R12 — D9 proved Open Buildings' own confidence cannot arbitrate,
because it is a size proxy.

But a different question is **not** blocked, because it depends only on the model's own output:

> Across the whole val set, what sizes of connected component does the model actually emit?

- **If the model almost never emits a component below ~400 px anywhere** → a **structural** limit
  (receptive field, /32 downsampling, capacity). That holds whatever R12 says about any
  individual label, and it would justify the resolution experiment on its own terms.
- **If it emits plenty of small components, just in the wrong places** → a **localisation**
  problem, not a capacity one. Resolution would be the wrong fix.

Two different and differently expensive conclusions, separable for free.

## Predictions (before running)

1. **The model under-produces small components.** Labels have 5.8 % of components in 50–200 px
   (1,630 / 27,918); I expect the model's share there to be **under 3 %**.
2. **`pred/label` per bin rises monotonically with size** — well under 1 in the small bins,
   near or above 1 in the large ones.
3. **Median predicted component will be larger than median label** — I expect roughly
   **1.5–3×**.

**What would flip the conclusion:** if the model's small-component share *matches or exceeds* the
labels', it is producing small objects freely and the 62 % non-detection is about *where*, not
*whether* — which would remove the main standalone argument for a resolution experiment.

## Setup

`scripts/d12_pred_size_dist.py` on the default teacher (MiT-B2, 0.4 m eroded labels), 1,701 val
crops, threshold 0.5, components ≥ 50 px, binned identically to D8.

**This is deliberately an unmatched comparison** — it counts components anywhere on the tile
rather than pairing them. A small predicted component need not correspond to a small label, and
that is fine: the question is whether the model's output *distribution* contains small objects at
all.

## Results

| size (px) | labels | label share | preds | pred share | pred/label |
|---|---|---|---|---|---|
| **50–200** | 1,630 | 5.8 % | **2,946** | **12.0 %** | **1.807** |
| 200–400 | 2,847 | 10.2 % | 2,464 | 10.0 % | 0.865 |
| 400–900 | 4,909 | 17.6 % | 3,780 | 15.3 % | 0.770 |
| 900–2000 | 6,156 | 22.1 % | 4,454 | 18.1 % | 0.724 |
| 2000+ | 12,376 | 44.3 % | 10,992 | 44.6 % | 0.888 |
| **total** | **27,918** | | **24,636** | | |

median component — **labels 1,657 px · predictions 1,619 px** (ratio 0.98)

**All three predictions wrong, the first two in the opposite direction.**

1. Predicted the model's small-component share under 3 % against the labels' 5.8 %. It is
   **12.0 %** — the model emits **1.8× more** small components than the labels contain ❌
2. Predicted `pred/label` rising monotonically with size. It is **highest in the smallest bin**
   (1.807) and roughly flat at 0.72–0.89 everywhere else ❌
3. Predicted a median predicted component 1.5–3× larger than the median label. They are
   **essentially identical** (1,619 vs 1,657 px, ratio 0.98) ❌

## Interpretation

**The model is not structurally incapable of producing small objects. It produces them freely —
almost twice as many as the label set contains.** The size distribution of its output tracks the
labels closely across every other bin, and the two medians agree to 2 %.

**This fires the flip condition I registered before running:** *"if the model's small-component
share matches or exceeds the labels', it is producing small objects freely and the 62 %
non-detection is about **where**, not **whether** — which would remove the main standalone
argument for a resolution experiment."*

**So the resolution experiment loses its justification.** The hypothesis was that a /32
downsampling encoder cannot represent a 14×14 px building. That hypothesis predicts a suppressed
small-component tail, and the tail is *over*-populated instead. A code change plus ~3 h/arm would
have been spent on a limit the model does not have.

**What the 62 % actually is.** The model emits ~1,300 *more* small components than there are
small labels, while failing to cover 62 % of those labels. It is finding small things — different
small things. Two readings, and D9/D10 bear on both:

- **The model's small detections are real and OB's are misplaced or phantom.** D10 saw OB drawing
  plot boundaries over bare ground at the large end; nothing rules out equivalent errors at the
  small end, and D9 showed 89 % of small OB polygons sit below the confidence threshold already.
- **The model's small components are fragments** — corners and edges shed from larger roofs.
  `fragments_per_label` of 0.9203 argues against this being dominant, since most predicted
  components do land on labels.

Distinguishing them still needs R12. But the *capacity* question is now closed.

## Decision

- [x] **Do not run the resolution experiment.** Its standalone argument is refuted; the model
      already produces small components at 1.8× the label rate.
- [x] **Reframe the 62 % as a localisation/label-disagreement problem**, not a capacity one.
      Every downstream plan that assumed "the model can't see small buildings" needs this.
- [x] **`MASTER_CONTEXT`'s scale argument is weakened for Stage 1.** The AIRS-vs-Jaipur 23× size
      gap (D2/D3) is real, but it does not manifest as an inability to emit small components.
- [ ] R12 remains the only route to deciding whose small objects are correct.

## Threats to validity

- **Unmatched by design.** This counts components anywhere on the tile; it cannot show that the
  model's small predictions sit on small buildings. That is precisely what makes it answerable
  without ground truth, and precisely what limits it.
- The label distribution here is the **eroded** one the model was trained to reproduce, which is
  the right reference for "does it emit what it was taught to" but not for real building sizes.
- Threshold 0.5 only; component counts are threshold-sensitive (D11), though both distributions
  shift together.

## Threats to validity

- Unmatched by design (above); it cannot say small predictions land on small buildings.
- Erosion shrinks labels, so the label size distribution here is the *eroded* one, which is what
  the model was trained to reproduce — the right reference for this question, but not the
  distribution of real buildings.
- Threshold 0.5 only. Component counts are threshold-sensitive (D11), though the *relative* shape
  of the two distributions should be less so.
