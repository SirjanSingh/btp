# 2026-09-10-d4-adjacency — 78% of Jaipur buildings touch a neighbour

| | |
|---|---|
| **Status** | ✅ done |
| **Date** | 2026-09-10 |

## Question

What fraction of Jaipur building footprints share a boundary with at least one neighbour?

**Why it matters:** `MASTER_CONTEXT` §3.2 lists **instance merging** as a distinct failure
mode — party-wall buildings with no visible gap fusing into one blob — and a whole planned
workstream (three-class labelling, split-aware metrics) exists to address it. That work is
only worth doing if buildings actually touch. A low rate would let the project *delete* a
workstream, which is the cheapest kind of result available.

**Prediction:** 40–60%. Dense Indian urban form suggests high, but Open Buildings footprints
are individually delineated and I expected visible gaps between many of them.

## Setup

All **318,207** footprints at confidence ≥ 0.75. Two buildings count as adjacent if within
**0.5 m** — under two pixels at Jaipur's 26.6 cm GSD, so "adjacent" means "the model cannot
see a gap". STRtree spatial index rather than the 318k² naive pair scan.

The metre tolerance is converted to degrees using the **latitude** scale (the CSV is
EPSG:4326). Longitude is compressed by cos(lat), so using the latitude scale for both makes
the x-tolerance slightly *tight* — biasing the measured rate **down**, not up.

```bash
./run_docker.sh "" "python -u scripts/d4_adjacency.py --out_dir diagnostics/d4"
```

## Results

Raw: [`diagnostics/d4/d4_summary.json`](../../diagnostics/d4/d4_summary.json)

| | |
|---|---|
| **adjacency rate** | **78.0 %** |
| mean neighbours | 1.52 |
| median / p90 | 1 / 3 |
| max | 9 |
| ≥ 3 neighbours | 19.6 % |

**Predicted 40–60%. Measured 78%** — higher than the top of the band.

## Interpretation

**The instance-merging concern is real and the workstream is justified.** More than three in
four buildings have a neighbour the model cannot see a gap to, and a fifth have three or more.
A segmentation model that fuses touching buildings will produce badly wrong building *counts*
while its pixel IoU looks fine.

This matters for the project's actual goal. The pipeline ends in a **kW estimate per
building**; merging two houses into one changes the count, the per-roof area distribution and
therefore `k_usable`, without moving IoU at all.

**And the metric to catch it does not exist yet.** `MASTER_CONTEXT` §6.2 is explicit: pixel
IoU cannot see merging, and **boundary IoU largely cannot either** — a Hong Kong study cut
under-segmentation 35.7% → 5.0% while Boundary F1 sat at 79.78%, i.e. the two move
independently. So every IoU reported in this project so far is silent about a failure that
affects 78% of the buildings.

This also puts [R4](../2026-09-10-segformer-backbone/) in perspective: MiT-B2's gain was
entirely precision, which *might* be better separation of adjacent buildings — but that claim
cannot be made without merge/split rate. D4 turns implementing that metric from a nice-to-have
into a prerequisite for interpreting results already collected.

## Decision

- [x] **Keep the instance-merging workstream.** 78% justifies it.
- [ ] ★ **Implement merge rate and split rate before any further architecture comparison.**
      Otherwise R4-style precision gains cannot be attributed.
- [ ] The three-class relabel (building / boundary / background) is now supported by a
      measurement rather than an assumption.

## Threats to validity

- **Footprints, not roofs.** Roofs overhang, so the true visual adjacency at the *roof* level
  is likely **higher** than 78%. This is a lower bound.
- 0.5 m tolerance is a choice; a sweep (0 / 0.5 / 1 / 2 m) would show sensitivity. At 0 m the
  rate would be strictly lower.
- Open Buildings delineates individually; where it has *already* merged two structures into
  one polygon, that pair is invisible here — again biasing the estimate down.
- **A subsample cannot measure this.** The smoke test on the first 3,000 rows returned
  **0.9%**, 87× too low, because those polygons are scattered across 175 km² and almost none
  of their real neighbours were in the sample. Spatial statistics need spatially complete
  data, not a random slice.
