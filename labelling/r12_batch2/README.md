# R12 batch 2: 200 Jaipur crops for hand-labelling (100 each)

Generated on the DGX on 2026-10-28 by `scripts/export_label_batches.py` (seed 7).
None of these crops are in batch 1.

```
sirjan/   100 crops
priyal/   100 crops
  images/                  512x512 RGB crop (0.266 m/px, about 136 m x 136 m of ground)
  labels/                  draw here: same filename, black/white PNG (255 = roof, 0 = rest)
  openbuildings_reference/ Open Buildings mask: do NOT open until you have finished drawing
manifest.csv            per crop: parent tile, density bin, OB building count, person
selection_summary.json  how the sample was drawn
```

Each person gets the same mix: 20 crops from each of the five density bins
(empty, sparse, moderate, dense, very dense), spread across all 16 tiles.
Twenty of each person's crops are empty according to Open Buildings. If you find a
building in one, draw it, because that is exactly what those crops are for.

Labelling rules are the same as batch 1 (see `../r12_batch1/README.md` and
`../TEAM_BRIEF.md`):

- Trace the roof as seen from above, not the ground footprint.
- Leave a visible gap between buildings that touch.
- Keep water tanks and stairwells inside the roof.
- Count under-construction slabs with walls; don't count bare foundations.
- Write ambiguous crops down in a notes file instead of checking the reference.

The imagery appears to be Google-sourced and its licence is unresolved, so do not
push these images to the public repo.
