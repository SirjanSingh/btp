# 2026-09-11-d8-missed-by-size — which buildings does the model miss?

| | |
|---|---|
| **Status** | ✅ done — strong size effect; interpretation blocked on R12 |
| **Date** | 2026-09-11 |

## Question

The default model's **missed rate is 0.3257** — a third of labelled buildings have no
prediction covering them. After two days of work it is the largest single error left in Stage 1,
and **every knob measured so far only trades it against merging**: erosion, inference threshold
and self-training all move along one fusion/recall curve rather than reducing misses.

Nothing has asked the obvious question: **are the missed buildings small?**

**Why it matters — the two answers lead to different, differently expensive experiments:**

- **Misses concentrated in small buildings** → a *resolution* problem. The median Jaipur
  building is ~913 px (D2/D3) — roughly 30×30 — and a /32-downsampling encoder sees the
  smallest ones across barely a single feature-map cell. The fix is training at higher effective
  resolution, which needs a code change (512 is hard-coded in the cache path) and ~3 h/arm.
- **Misses spread evenly across sizes** → not resolution. Look at contrast, shadow, dark RCC
  roofs, or label error instead — all much cheaper to probe.

Measuring first costs ~5 minutes of inference. Guessing costs a 3-hour arm and a code change
that might target the wrong thing.

## Predictions (before running)

1. **Miss rate falls monotonically with size**, and is **> 0.60 in the smallest bin (50–200 px)**
   versus **< 0.10 above 2000 px**.
2. **The smallest two bins (< 400 px) contribute > 50 % of all misses**, despite being small
   buildings that contribute far less to total roof area.
3. **Median area of missed buildings < 400 px; median area of found buildings > 900 px** — at
   least a 2× separation.

**If prediction 2 fails** — if misses are spread evenly, or dominated by *large* buildings —
then resolution is the wrong lever and the resolution experiment should not be run at all. That
is the outcome that would save the most time.

**A caveat I want on record before seeing numbers:** a "miss" is measured against Open
Buildings. Small OB polygons are also the **most likely to be OB false positives** — noise,
vegetation, or a shed OB hallucinated. So a high miss rate on tiny buildings is genuinely
ambiguous between "model fails on small buildings" and "OB invents small buildings". R12's hand
labels are the only way to separate those, and this diagnostic cannot.

## Setup

`scripts/d8_missed_by_size.py` on the default teacher (MiT-B2, 0.4 m eroded labels), the same
1,701-crop held-out val tiles, threshold 0.5, same ≥50 % association rule as
`merge_split_rate.py` so the overall number reconciles with the published 0.3257.

Labels binned by their own pixel area: 50–200, 200–400, 400–900, 900–2000, 2000+ px
(26.6 cm/px → 400 px ≈ 28 m², 900 px ≈ 64 m²).

## Results

**Default teacher** (MiT-B2, 0.4 m eroded labels), threshold 0.5:

| size (px) | labels | missed | miss rate | % of all misses |
|---|---|---|---|---|
| 50–200 | 1,630 | 1,219 | **0.7479** | 15.4 % |
| 200–400 | 2,847 | 2,006 | **0.7046** | 25.4 % |
| 400–900 | 4,909 | 2,335 | 0.4757 | 29.5 % |
| 900–2000 | 6,156 | 1,460 | 0.2372 | 18.5 % |
| 2000+ | 12,376 | 884 | **0.0714** | 11.2 % |
| **all** | 27,918 | 7,904 | 0.2831 | |

median area **missed 499 px** · median area **found 2,455 px**

**Prediction scorecard.**
1. Monotonic decline ✅, > 0.60 in the smallest bin (**0.7479**) ✅, < 0.10 above 2000 px
   (**0.0714**) ✅ — all three parts hit.
2. "< 400 px contributes > 50 % of misses" → **40.8 %** ❌.
3. Median missed < 400 px → **499** ❌ (just over); median found > 900 px → **2,455** ✅;
   separation ≥ 2× → **4.9×** ✅.

**The size effect is strong.** Miss rate is **10.5× higher** in the smallest bin than the
largest. Buildings under 900 px are **70.3 %** of all misses. Buildings over 2000 px are 44 % of
all labels but only 11.2 % of misses.

## A binning bug, caught by an impossible result

The first run reported the **50–200 px bin as empty** — impossible, since 1,630 labels sit there.
`np.searchsorted(EDGES, area, "right")` returns the *insertion* index, one past the correct bin,
so every label landed one bin too high and the top bin silently absorbed two ranges.

**This mattered.** The buggy output showed misses rising with size and the 2000+ bin
contributing the most (29.7 %), which reads as *"misses are dominated by large buildings —
resolution is the wrong lever"*. That is the opposite of the corrected conclusion. The only tell
was a bin that could not legitimately be empty. Fixed, self-tested across 15 boundary values,
and re-run.

## Erosion is not the cause — same diagnostic on the un-eroded model

Erosion shrinks small polygons hardest in relative terms, so it was a live third hypothesis that
erosion *creates* the small-building misses. It does not:

| size (px) | eroded miss rate | un-eroded miss rate | ratio |
|---|---|---|---|
| 50–200 | 0.7479 | 0.6669 | 1.12× |
| 200–400 | 0.7046 | 0.6287 | 1.12× |
| 400–900 | 0.4757 | 0.3868 | 1.23× |
| 900–2000 | 0.2372 | 0.1788 | 1.33× |
| 2000+ | 0.0714 | 0.0516 | 1.38× |
| **all** | 0.2831 | 0.2334 | 1.21× |

**The un-eroded model — with no erosion whatsoever — still misses 67 % of the smallest buildings
and 5 % of the largest.** The profile's shape is essentially identical; erosion adds a roughly
uniform penalty on top. Its *relative* cost is actually largest for big buildings (1.38×), not
small. **Erosion is eliminated as the explanation.**

## Interpretation

**Two hypotheses remain, and this diagnostic cannot separate them.**

- **(a) The model genuinely fails on small buildings** — a resolution/receptive-field problem.
  A 200-px building is ~14×14 px, and a /32-downsampling encoder sees it across well under one
  feature-map cell.
> **Corrected 2026-09-11 by [D10](../2026-09-11-d10-missed-gallery/):** the framing below
> treats *large*-polygon misses as trustworthy. They are not. Rendering the largest misses
> shows Open Buildings drawing **compound walls and plot boundaries as buildings** — two of
> the three largest are polygons over bare ground and vegetation with no roof at all. OB
> error contaminates **both** ends of the size range, not just the small end.

- **(b) Open Buildings invents small polygons** that were never buildings — vegetation, shade
  structures, noise. Small OB polygons are precisely the ones most likely to be false positives.

**(b) predicts exactly the same measurement as (a).** 70 % of "misses" being small is equally
consistent with "the model can't see them" and "they aren't there". I flagged this before
running, and it is the binding limitation.

## Decision

- [ ] **Do not run the resolution experiment yet.** It needs a code change (512 is hard-coded in
      the cache path) plus ~3 h/arm, and under hypothesis (b) it would be chasing phantoms. The
      evidence is suggestive but the interpretation is blocked.
- [x] **This raises R12's value concretely.** The hand labels now gate a specific, costed
      decision rather than being generically good practice: *if hand-labelled small buildings
      are real, resolution is the top lever; if OB invented them, ~70 % of the miss rate is
      not a model failure at all.* Worth saying to whoever picks up the labelling.
- [x] **Erosion is ruled out** as the cause of small-building misses.
- [x] **Report miss rate by size band**, not as a single scalar. "0.2831 missed" hides a
      10.5× spread and is nearly meaningless on its own.

## Note on the headline number

Overall miss here is **0.2831**, against the published **0.3257** from
`merge_split_rate.py`. My setup section claimed these would reconcile; they do not, for a real
reason. `merge_split_rate` requires a **single predicted component** to cover ≥ 50 % of a label;
this script counts overlap with **all predicted foreground combined**, so a label covered 30 %
by each of two components counts as found here and missed there. The looser rule is right for
*this* question (was the building detected at all?) and the stricter one is right for instance
counting. **The size profile is unaffected; only the absolute level shifts.**

## Threats to validity

- **Agreement, not accuracy (R12).** The ambiguity noted above is fundamental to this
  diagnostic, not incidental.
- Single checkpoint, single threshold. Miss rate rises sharply with threshold (0.2402 at 0.3 →
  0.5771 at 0.8), so the *absolute* numbers are operating-point specific; the *size profile* is
  the transferable part.
- `min_area_px 50` excludes the very smallest OB polygons, which are the most suspect. Their
  exclusion makes the small-bin miss rate a lower bound.
