# Priyal: your 100 crops for batch 2

Hi Priyal, this is everything you need to label your half of batch 2.
It takes about 3 to 5 minutes per crop once you get going.

## 1. Get the images

The images are **not on GitHub**. The imagery licence is unresolved and this repo is
public, so they stay off it. Sirjan will send you `r12_batch2_priyal.zip` directly.

Unzip it into this folder so it looks like this:

```
labelling/r12_batch2/priyal/
  images/                  100 crops, 512x512, about 136 m x 136 m of ground each
  labels/                  empty for now; masks go here at the end
  openbuildings_reference/ DO NOT OPEN until you have finished all 100
```

## 2. The tool: LabelMe

We use **LabelMe** (desktop app, free). Sirjan's labels were made with version 7.9.1.

```bash
pip install labelme
cd labelling/r12_batch2
labelme priyal/images --labels roof
```

Autosave is on by default in 7.x. Each image gets a `.json` file saved next to it.

Shortcuts:

| Key | Does |
|---|---|
| `Ctrl+N` | start a new polygon (click corners, click the first point to close) |
| `D` / `A` | next / previous image |
| `Ctrl+Z` | undo |
| `Ctrl+Scroll` | zoom |

Files you have already labelled show a tick in the file list on the right, so you can
stop and resume any time.

## 3. Labelling rules

Use the same rules as Sirjan so our labels can be compared.

- **Trace the roof as seen from above**, not the ground footprint. If the roof overhangs,
  follow the roof.
- **Buildings that touch get separate polygons with a visible gap.** This is the most
  important rule: 78% of Jaipur buildings touch a neighbour, and fused buildings are the
  model's biggest error. Never trace a whole terrace block as one shape.
- **Water tanks and stairwells stay inside the roof.** Don't cut them out.
- **Under construction:** has walls, label it. Bare foundation slab, skip it.
- **Walled compound:** trace the roofs inside, never the boundary wall. An empty yard gets nothing.
- **Empty crops:** 20 of your crops have no buildings according to Open Buildings. If you
  spot one anyway, label it. That is exactly what those crops are for.
- **Not sure?** Write the filename and the reason in `priyal/notes.txt`. Do not peek at
  `openbuildings_reference/` to decide.
- **No AI pre-segmentation tools.** The whole point is a human answer key.

Longer explanation with examples: [`../TEAM_BRIEF.md`](../TEAM_BRIEF.md).

## 4. When you finish: convert to masks

From the repo root:

```bash
python scripts/labels_from_json.py \
  --json_dir labelling/r12_batch2/priyal/images \
  --out labelling/r12_batch2/priyal/labels \
  --images labelling/r12_batch2/priyal/images
```

It converts the polygons to black and white PNGs (255 = roof) and prints a check table.
Fix anything it flags and re-run.

## 5. Send it back

Zip `priyal/images/*.json`, `priyal/labels/` and `priyal/notes.txt` and send them to
Sirjan. Don't push them to GitHub (the masks are derived from the same imagery).

## Progress so far (15 Nov 2026)

| Person | Crops | Labelled |
|---|---|---|
| Sirjan | 100 | 6 |
| Priyal | 100 | 0 |
