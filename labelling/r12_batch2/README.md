# R12 batch 2: 200 Jaipur crops for hand-labelling (100 each)

> **Priyal:** start with [`PRIYAL_GUIDE.md`](PRIYAL_GUIDE.md) (tool setup, rules, how to send results back).

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

## Progress (19 Nov 2026): 152 crops, frozen as the eval set

Labels come back as zips (not through GitHub) and are copied into each person's folder.
A crop only counts as labelled once `labels/<name>` exists.

| Folder | Crops | Labelled | Who drew them | Still open |
|---|---|---|---|---|
| `khushi/` | 50 | 50 | Khushi | none (7 blank crops added as empty: Open Buildings is empty there too) |
| `priyal/` 1-50 | 50 | 50 | Priyal (LabelMe 6.3.1) | none |
| `priyal/` 51-100 | 50 | 45 | Khushi, masks made with her `masks_with_gaps.py` (cuts a 2 px gap between touching buildings) | 9 blank crops added as empty (OB empty too); 5 blank crops dropped because OB has buildings there (`map67_3-1_002048_002560`, `map67_4-1_009216_008704`, `map67_4-2_001536_005120`, `map67_4-2_003584_000000`, `map67_4-2_004096_002560`) |
| `sirjan/` | 50 | 7 | Sirjan | `map67_1-1_007680_004608` has a .json but is left out (may be unfinished) |

Score against Open Buildings: `python scripts/eval_hand_labels.py score --batches labelling/r12_batch2`
(writes `diagnostics/hand_labels/score.json`). On the labelled crops: IoU 0.50,
precision 0.74, recall 0.61. Open Buildings sits about 6 px (1.6 m) north and 2 to 4 px east
of the hand-drawn roofs for every annotator, and hand-drawn roof area is 1.0 to 1.4 times the
Open Buildings area depending on the annotator.
