# 2026-09-11-d13-extra-predictions — are the model's extra small components fragments or findings?

| | |
|---|---|
| **Status** | ✅ done — fragment hypothesis refuted |
| **Date** | 2026-09-11 |

## Question

D12 found the model emits **1.8× more** small components than the label set contains, while D11
found **62 %** of small labels get essentially no prediction. The model finds small things, and
largely *different* small things. Two readings were left open:

- **(a)** genuine buildings Open Buildings missed, or
- **(b)** fragments shed from the edges of larger roofs.

This is decidable **without ground truth**, because it asks about the *spatial relationship*
between predictions and labels rather than which is correct. Each small predicted component is
classified:

| class | meaning |
|---|---|
| **isolated** | overlaps no label at all — candidate unlabelled building, or noise |
| **fragment** | overlaps a label that some *other, larger* prediction already covers |
| **sole** | overlaps a label that no other prediction covers — the model's only detection of it |

`fragments_per_label` (0.9203) hinted against (b) but could not separate these: it counts
components touching labels without asking whether the label was *already* covered.

**Prediction:** fragments will be the largest class, ~40–60 %, with isolated ~20–30 %. The 0.9203
figure suggested most small components sit on labels, and I read "sits on a label" as mostly
meaning "is a piece of one".

## Results

Small predicted components on the 1,701 val crops, default teacher, threshold 0.5:

| range | total | isolated | fragment | sole |
|---|---|---|---|---|
| 50–400 px | 5,418 | **2,114 (39.0 %)** | **591 (10.9 %)** | **2,713 (50.1 %)** |
| 150–400 px | 3,266 | 1,115 (34.1 %) | 364 (11.1 %) | 1,787 (54.7 %) |

**Prediction wrong, and decisively.** I predicted fragments would dominate at 40–60 %. They are
**10.9 %** — the smallest class by a wide margin, and stable across both size ranges.

## Interpretation

**The fragment hypothesis from D12 is refuted.** Only about one in nine small predictions is a
shed piece of an already-detected building. The model's small-component output is not noise
around large roofs.

**Half of them are doing exactly the job they should.** At 50.1 %, the largest class is *sole* —
small predictions that are the model's only detection of a labelled building. These are correct
small-building detections, and they coexist with D11's finding that 62 % of small labels get
nothing. Both are true: the model detects a substantial number of small buildings and misses more
of them.

**39 % overlap no label at all.** Two inspected closely: one is a pale-roofed roadside structure
with no OB polygon anywhere near it — a plausible genuine building OB missed. The other sits in a
dense residential block where prediction and label outlines cover similar structures but are
**visibly offset**, consistent with the project's standing roof-vs-ground-footprint gap. Two
cases cannot quantify the split between "OB missed it" and "model hallucinated it".

**What this establishes for R12.** The disagreement over small objects is **two-sided and
substantive**: the model both misses small labels *and* produces small detections the labels lack,
and neither side is dominated by noise. The hand labels will therefore **adjudicate a genuine
disagreement between two imperfect sources**, not simply measure model error against a reference.
That is a stronger reason to do the labelling than "we need ground truth", and it should be said
to whoever does it.

## Decision

- [x] **Refute the fragment reading** of D12's excess small components — recorded there.
- [x] **Stop describing small-object behaviour as a model deficiency.** Half the model's small
      components are correct sole detections; the remainder is a two-sided disagreement.
- [x] **R12's framing upgraded** from "establish ground truth" to "adjudicate a measured
      two-sided disagreement", which also tells labellers what to look for.
- [ ] Quantifying the isolated class (OB miss vs hallucination) needs R12. Not estimable from 2
      inspected cases.

## Threats to validity

- **Only 2 isolated cases inspected visually.** Enough to show at least one is a plausible real
  building; nowhere near enough to split the 39 %.
- "Fragment" depends on the same ≥50 % overlap constant D11 showed inflates miss rate 1.77×. A
  looser rule would move some *sole* into *fragment* by making other predictions count as
  covering. The fragment share is small enough that this cannot reverse the conclusion.
- Threshold 0.5 only; component counts are threshold-sensitive.
- Val tiles only — three of the sixteen.
