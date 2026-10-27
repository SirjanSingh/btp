# 2026-09-11-d10-missed-gallery — what does a "missed building" actually look like?

| | |
|---|---|
| **Status** | ✅ done — refutes an assumption I made in D8 |
| **Date** | 2026-09-11 |

## Question

D8 found the model misses 33 % of labelled buildings, and D9 showed small-polygon misses are
uninterpretable until R12 (Open Buildings' confidence is a size proxy, so "is this small polygon
real?" cannot be asked of OB's own metadata).

I then wrote, in D10's own design note: *"Large polygons carry no such doubt — 93 % clear the
conf ≥ 0.75 filter, and a 2000+ px footprint is not something OB hallucinates. So the 884 missed
large buildings are unambiguous model failures."*

This renders them to find the named failure mode — dark roofs, shadow, occlusion, construction.

**Prediction:** the large misses will show a small number of recurring, physical causes, most
likely low-contrast flat roofs and tree/building shadow.

## Results

884 labelled components ≥ 2000 px are missed under the ≥50 %-overlap rule. Two samples were
rendered with the label in **cyan** and the model's prediction in **magenta**.

### Sample A — the 12 largest (1,162–2,551 m²)

| rank | area | coverage |
|---|---|---|
| 00 | 2,551 m² | 0.442 |
| 01 | 2,466 m² | **0.000** |
| 02 | 1,967 m² | 0.306 |
| 04 | 1,675 m² | 0.052 |
| 09 | 1,275 m² | **0.000** |

**Three inspected, three are Open Buildings errors — not model failures:**

- **#01 (0.000 coverage)** — the polygon covers a **bare, cleared plot**: brown earth and rubble,
  no roof anywhere. The model correctly predicts nothing.
- **#09 (0.000)** — the polygon covers a **walled compound containing trees and open ground**.
  Again no roof; again the model is right.
- **#05 (0.350)** — the polygon spans an entire **institutional compound including a sports
  court** and open yard. The model correctly segments only the actual roofed buildings inside it,
  and is penalised for it.

**The assumption I wrote into this experiment's own design is refuted.** Open Buildings does
hallucinate at 2000+ px, and its characteristic error at that scale is drawing **compound walls
and plot boundaries as building footprints**.

### Sample B — 8 drawn at random from the same 884 (158–779 m²)

Sample A is a **biased draw**: the very largest "buildings" are exactly the ones most likely to be
compounds. A random sample checks whether the pattern is a tail artefact.

| rank | area | coverage |
|---|---|---|
| 02 | 779 m² | 0.067 |
| 07 | 707 m² | 0.457 |
| 03 | 373 m² | 0.148 |
| 04 | 262 m² | 0.417 |
| 00 | 177 m² | 0.472 |

**The picture changes.** #02 shows a large **low, flat, tree-shadowed roof** that the model
almost entirely fails to detect — a genuine model failure of exactly the predicted kind. Others
at this scale are ambiguous at the rendered zoom.

**So the 7.1 % large-building miss rate is a mixture**, weighted toward OB error at the extreme
tail and toward real model failure in the 150–800 m² range. It cannot be cleanly attributed
either way, which is the opposite of what D10 was set up to assume.

### The finding that holds across both samples

**"Missed" almost never means "predicted nothing".** Coverage across all 20 inspected cases runs
**0.05–0.50**, with only 2 at exactly zero — and both of those are OB false positives over
roofless ground. Every genuine case is a **partial detection scored as a binary miss** by the
≥50 % rule.

That matters because OB polygons are frequently *larger than the roof they nominally describe*
(compounds, plot boundaries). A model that correctly outlines only the roof will land well under
50 % coverage and be recorded as having missed the building entirely.

## Interpretation

**The miss rate is not a clean model-quality metric at either end of the size range.** Small
misses are contaminated by OB inventing polygons (D9, unresolvable without R12); large misses are
contaminated by OB drawing compounds. The 33 % headline is an upper bound on model failure by an
unknown margin.

**The ≥50 % overlap rule is doing more work than it looks.** It is inherited from
`merge_split_rate.py` and was never examined. Against labels that systematically over-cover, it
converts correct partial detections into misses. A rule based on *the prediction's* overlap
(did the model's component land mostly inside a label?) would behave very differently.

**This does not overturn anything already decided.** Erosion, self-training and encoder
conclusions were all comparisons *between models on the same labels*, where a shared label bias
largely cancels. It affects **absolute** claims about how good the model is, which is precisely
what R12 exists to establish.

## Decision

- [x] **Retract the D8/D10 assumption** that large-polygon misses are unambiguous model failures.
      Corrected in D8's write-up.
- [x] **Stop quoting the 33 % miss rate as a model property.** It is agreement-with-OB and
      contaminated at both size extremes.
- [ ] **Add compound/plot-boundary cases to the R12 labelling brief** as an explicit call-out —
      labellers should trace roofs, not walls, and this is now known to be where OB diverges most.
- [ ] Consider a prediction-centred association rule alongside the label-centred one. Not run;
      it would change every published instance metric and needs its own experiment.

## Threats to validity

- **Small visual sample** — 3 of 12 and 2 of 8 inspected closely. Enough to refute a
  universal claim ("unambiguous model failures"), not enough to quantify the mixture.
- Interpretation is my own eyeballing of satellite imagery, not ground truth. The bare-plot and
  vegetation cases are unambiguous; the mid-size ones genuinely are not.
- Both samples come from the same three val tiles, which is where the val split lives.
