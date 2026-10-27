# Optional second pass — roof superstructures

> **Standalone add-on. Not part of the main brief.**
> Only start this if Sirjan asks — it is deliberately kept separate so it doesn't change the
> scope of work already in progress. Main task:
> [`TEAM_BRIEF.md`](https://github.com/SirjanSingh/btp/blob/feat/init-project-setup/labelling/TEAM_BRIEF.md)

**What:** on ~100 already-labelled crops, mark the things on a roof that a solar panel **cannot**
go on.
**Time:** noticeably faster than the first pass — same images, and you only outline obstacles.

---

## Why this one is worth more than everything else

The project estimates Jaipur's rooftop solar potential at **4.59 GWp / 7.62 TWh per year**, with
an error bar of **±15.2 %**.

**Two-thirds of that error bar — 68 % — comes from a single number nobody has measured:**
`k_usable`, the fraction of a roof that is actually available for panels. It is currently
assumed to be **0.60** with nothing behind it.

Measuring it on ~100 crops would take the whole estimate from **±15.2 % to about ±9 %**.

For comparison, the segmentation model this project spent days optimising contributes **2 %** of
the error bar. Driving its error to *zero* would improve the headline figure by **0.1 percentage
points**. This pass is worth roughly **50× more per hour spent.**

That is the entire argument. It is a few hours of clicking against the largest single source of
uncertainty in the result.

---

## What to mark

On each crop, outline anything **on a roof** that blocks panel installation:

| Mark it | Examples |
|---|---|
| **Water tanks** | the ubiquitous black/blue plastic tanks, usually on a stand |
| **Stairwell / lift head** | the boxy structure over the staircase |
| **Existing solar** | panels **and** solar water heaters — both already occupy the roof |
| **Parapet setback** | a ~1 m strip inside the roof edge, where panels can't sit |
| **Machinery** | AC units, ducts, dish antennas, tall vent pipes |
| **Permanent clutter** | built-up sheds, pergolas, fixed canopies |

**Do not mark:** loose objects (drying laundry, furniture, stored junk), shadows, or roof
markings/discolouration. We want **permanent obstructions**, not whatever happened to be up there
on the day the satellite passed.

**Solar water heaters get their own label if your tool allows it** — they're a separate research
question for the project's second stage, and tagging them here costs nothing extra.

---

## How

Same images as the main pass, in `working/images/`.

1. Open the crop **and** your finished roof mask from the first pass (as a reference layer if
   your tool supports it — this time looking at your own work is fine and expected).
2. Draw polygons around the obstructions **inside** roof areas only. Anything on the ground is
   irrelevant.
3. Export to a **separate** folder — `working/superstructures/`, not `working/labels/`. Do not
   overwrite the roof masks.

Rough edges are fine. A water tank traced as a slightly-too-big blob costs almost nothing; a
water tank **missed entirely** is what biases the number.

---

## What happens to it

`k_usable = 1 − (superstructure area + setback area) / roof area`, computed **per density bin**
rather than as one city-wide scalar — dense old-city roofs are far more cluttered than sparse
outskirts, and collapsing them into one number is part of why the current 0.60 is untrustworthy.

Reported as a **measured distribution**, which is a genuine result in its own right: a 2024
*Applied Energy* study argues that existing rooftop-solar work generally **does not** account for
superstructures and therefore **overestimates** potential. Measuring it here is a small, real
contribution rather than a chore.

---

## The honest caveat

Some obstructions are invisible from directly overhead — a tank tucked against a parapet, a low
unit hidden in shadow. **You will miss some, and that is expected.** It means the measured
`k_usable` is an **upper bound** on usable roof, and the potential estimate stays an
upper-bound-ish figure.

Say so rather than working around it. A number with a known direction of bias is far more useful
than one pretending to be exact.
