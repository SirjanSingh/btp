# Rooftop labelling — team brief

**What:** hand-trace rooftops on **20 aerial crops of Jaipur**.
**Time:** ~4–5 hours total, ~1 hour each if four people split it.
**Deliverable:** one black-and-white PNG per crop.

Everything you need is in this folder:
👉 **https://github.com/SirjanSingh/btp/tree/feat/init-project-setup/labelling/r12_batch1**

---

## Why we're doing this

The model is trained on rooftop outlines from **Google Open Buildings** — which is itself a
model's output, and traces *ground footprints* rather than *roof outlines*. So every accuracy
number in the project currently means **"agrees with Open Buildings"**, not **"is correct"**.

Three separate results are stuck on that, and a few dozen honestly hand-drawn tiles unstick all
of them. 20 crops is small, but it's the difference between having an answer key and not.

---

## ⚠️ The one rule

> ### Do not open `openbuildings_reference/` until you've finished drawing.

That folder holds the machine's answer for each crop. It ships with the batch so we can
**compare afterwards** — that comparison *is* the result.

If you use it as a starting point or even peek, your labels will end up agreeing with the
machine by construction, and the question we're trying to answer ("how wrong is it?") becomes
unanswerable. Draw from the photo alone.

If a crop is genuinely ambiguous, **write it down** in your notes rather than peeking. An
ambiguity count is a useful finding on its own.

---

## Step 1 — get the files

```bash
git clone -b feat/init-project-setup https://github.com/SirjanSingh/btp.git
cd btp/labelling/r12_batch1/working
```

Or just download the `working/images/` folder from the GitHub link above (20 PNGs, ~14 MB).

**Only touch `working/`.** The `sealed/` folder is a held-out test set — leave it completely
alone.

---

## Step 2 — pick a tool

**Recommended: [makesense.ai](https://www.makesense.ai)** — free, runs in the browser, nothing
to install, and your images never leave your machine.

1. Open it → **Get Started** → drag in the images from `working/images/`
2. Choose **Object Detection → Polygon**
3. Create one label called `roof`
4. Trace, then **Actions → Export Annotations → VOC XML / Single file JSON**

**Alternative: [labelme](https://github.com/wkentaro/labelme)** (desktop, if you prefer):

```bash
pip install labelme
labelme working/images --nodata --autosave
```

Either way you'll get **polygon files**, not masks — step 4 converts them.

*(If you'd rather just paint white-on-black in GIMP/Photoshop and export PNGs directly, that
works too — skip step 4 and go straight to verifying.)*

---

## Step 3 — how to trace

Draw around **each roof as seen from above**. One polygon per building.

| Situation | What to do |
|---|---|
| **Roof overhangs the walls** | Follow the **roof edge**, not the ground outline. This is exactly where the machine is expected to be wrong. |
| **Buildings touching each other** | Draw them as **separate polygons with a visible gap**. ⭐ **This is the most important thing in the whole task** — see below. |
| **Stairwell / water tank on the roof** | Leave it **inside** the roof polygon. Don't cut it out. |
| **Under construction** | Has walls → label it. Bare foundation slab → skip. |
| **Tiny shed / awning** | If you'd call it a building from the air, label it. |
| **Walled compound / plot boundary** | Trace the **roofs inside it**, never the boundary wall. A walled yard with no building in it gets **nothing**. This is the single place the machine is most often wrong, so your call here is especially valuable. |
| **Crop looks empty** | That's expected — 3 of your crops have no buildings. If you *do* spot one, label it; that's a real finding. |

### ⭐ Why touching buildings matter so much

78% of Jaipur buildings touch a neighbour, and the model's biggest weakness is **fusing them
into one blob**. The project's headline output is a *per-building* solar estimate, so two houses
merged into one is a direct error in the final number.

If two roofs share a wall, draw **two polygons with a small gap between them**. Never trace a
whole terrace block as one shape.

---

## Step 4 — convert and check

From the repo root:

```bash
# convert polygons → masks
python scripts/labels_from_json.py \
  --json_dir <folder with your .json files> \
  --out labelling/r12_batch1/working/labels \
  --images labelling/r12_batch1/working/images
```

It converts **and** verifies, printing a table. Fix anything it flags, then re-run.

Already have PNGs (GIMP route)? Just verify:

```bash
python scripts/labels_from_json.py --verify_only \
  --out labelling/r12_batch1/working/labels \
  --images labelling/r12_batch1/working/images
```

**What it checks:** every image has a mask · masks are the right size · masks are pure black and
white · and — the one that actually catches mistakes — whether your region count is far below
Open Buildings' count, which usually means touching buildings got merged.

You want to see:

```
[verify] ALL GOOD — ready to submit
```

*(The check reads `openbuildings_reference/` for you. Running it after you've drawn is fine —
just don't look at those files yourself beforehand.)*

---

## Step 5 — send it back

Put your finished PNGs in `working/labels/` (same filename as the image) and either:

- **open a PR** against `feat/init-project-setup`, or
- **zip `labels/` and send it to Sirjan** — simpler if you'd rather not deal with git.

Include a short note listing any crops you found ambiguous and why. That note is genuinely
useful, not a formality.

---

## Suggested split — 5 crops each

Each person gets a mix of easy and hard so nobody draws 55 buildings all afternoon.
`OB` = how many buildings the machine thinks are there (a rough difficulty hint, **not** a
target to match).

| | Person 1 | Person 2 | Person 3 | Person 4 |
|---|---|---|---|---|
| dense | `map67_1-3_004096_005120` (28) | `map67_3-1_006656_009216` (55) | `map67_3-3_005120_005632` (23) | `map67_2-1_011264_008192` (1) |
| empty | `map67_2-3_013312_010240` (0) | `map67_4-3_007680_007680` (0) | `map67_1-2_008192_001536` (17) | `map67_1-3_000512_005120` (19) |
| moderate | `map67_2-1_005120_008192` (15) | `map67_3-2_012800_009728` (27) | `map67_3-3_001536_006144` (22) | `map67_2-1_004096_005632` (1) |
| sparse | `map67_2-2_002048_002048` (8) | `map67_3-2_006144_008192` (10) | `map67_3-3_009216_008192` (12) | `map67_4-3_002560_000000` (5) |
| very dense | `map67_2-3_008704_008704` (22) | `map67_3-1_008704_002560` (33) | `map67_3-3_002048_004608` (26) | `map67_4-1_010240_003072` (55) |

All filenames end in `.png`. Full details per crop are in
[`manifest.csv`](https://github.com/SirjanSingh/btp/blob/feat/init-project-setup/labelling/r12_batch1/manifest.csv).

---

## Quick answers

**How precise do I need to be?** Roughly building-accurate, not pixel-perfect. A few pixels of
slop on an edge doesn't matter. **Missing a building, or merging two, does.**

**What if I can't tell whether it's a roof?** Make a judgement call and note the filename. Don't
agonise, and don't check the reference.

**The image is blurry / weird colours.** Expected — it's satellite imagery at ~26 cm per pixel.
Label what you can make out.

**Can I use AI tools to pre-segment?** No. That reintroduces exactly the machine bias this batch
exists to measure.

**Does consistency between people matter?** Yes, more than individual perfection. If you're
unsure about a convention, ask in the group rather than deciding alone — the table in step 3
should cover most cases.

---

## Background, if you're curious

- [`r12_batch1/README.md`](https://github.com/SirjanSingh/btp/blob/feat/init-project-setup/labelling/r12_batch1/README.md)
  — how these 20 crops were chosen (stratified across five density bins, all 16 source tiles)
- [`experiments/README.md`](https://github.com/SirjanSingh/btp/blob/feat/init-project-setup/experiments/README.md)
  — every experiment run so far and what it found
