# 2026-09-11-d16-capacity — the project's first end-to-end capacity and energy figure

| | |
|---|---|
| **Status** | ★★ done — 4.59 GWp / 7.62 TWh/yr (±15.2 %) |
| **Date** | 2026-09-11 |

## Question

`MASTER_CONTEXT` refers to *"the headline GW figure"* repeatedly, sets an ethics caveat for it,
and benchmarks against a published **~1.89 GW** for an Indian metropolis. **Nothing in the repo
computes one.** The pipeline has always stopped before the arithmetic.

D14 built the uncertainty budget and D15 corrected a 22.5 % double-count in the formula, so the
chain is finally specifiable. This runs it over the **full AOI** — all 9,072 crops, not just val.

## Results

Default teacher (MiT-B2, 0.4 m eroded labels), threshold 0.5, all 9,072 crops.

| | value |
|---|---|
| AOI covered | **168.3 km²** |
| predicted roof area | **38.22 km²** — 22.7 % of AOI |
| usable area (`k_usable` 0.60) | 22.93 km² |
| **installed capacity** | **4.59 GWp** |
| **annual energy** | **7.62 TWh/yr** |
| uncertainty (1 sd) | **±15.2 %** → 3.89–5.29 GWp · 6.46–8.78 TWh/yr |

| term | rel sd | share of variance | provenance |
|---|---|---|---|
| **`k_usable`** | 12.5 % | **68.0 %** | ASSUMED |
| PVOUT | 5.7 % | 14.2 % | SECONDARY — not citable (D15) |
| η | 5.0 % | 10.9 % | PLANNED |
| rooftop derate | 3.2 % | 4.3 % | ASSUMED (D15) |
| segmentation area | 2.2 % | 2.0 % | MEASURED, n = 3 seeds |
| erosion un-do | 1.1 % | 0.6 % | SELF-CHECKED |

## An error caught by the script's own output ★

The first run reported **5.24 GWp / 8.72 TWh/yr** — **14.3 % too high.**

I had built in an "un-erosion" correction: the model trains on 0.4 m-eroded labels, so its
footprints should be smaller than real roofs, and dividing by the measured eroded/un-eroded area
ratio (0.8748) should recover true area. The reasoning is sound and the correction is wrong.

**The teacher is trained on eroded labels but validated and best-epoch-selected on the
*un-eroded* val set.** Its output therefore calibrates to un-eroded extent. The script printed
both sums, and they settle it: predicted area is **0.9847×** the un-eroded label area. There was
nothing to un-erode; the correction inflated the answer by 14.3 %.

**The check was already on screen.** I wrote a script that printed `pred` and `label` side by
side and then applied a correction contradicted by those two numbers. The fix is not just the
value — `d16_capacity_estimate.py` now **computes the ratio and refuses the correction** when the
output is already at un-eroded extent, printing what it ignored and why.

**The guard is tested, not just written.** An untested guard is the same pattern-A trap it
exists to prevent, and the confirmation re-run could not exercise it (the default is now
1.0, so the condition is false by construction). Exercised standalone against the real
pixel sums: it fires for 0.8748 / 0.90 / 0.50 and correctly **does not** fire when the model
genuinely sits at eroded extent (pred/label 0.8748), where the correction *should* apply.

**Confirmation run:** 4.59 GWp · 7.62 TWh/yr · ±15.2 %, pred/label 0.9849 — matching the
hand calculation exactly.

**Generalisable lesson:** a correction derived from *how a model was trained* must be validated
against *what the model actually emits*. Those differ whenever training and validation use
different label sets — which is exactly this project's setup, deliberately, since the erosion
work.

## Interpretation

**4.59 GWp technical potential over 168 km² of Jaipur.** For scale, the published Indian-metropolis
comparison is ~1.89 GW; a direct comparison is not valid without matching AOI definitions, roof
fractions and `k_usable`, and the write-up should not imply one.

**22.7 % of the AOI is roof.** Consistent with D1's measured Open Buildings prior of 23.1 % at
confidence ≥ 0.75 — as it should be, since the model reproduces that label set to within 1.5 %.
This is a coherence check, not independent confirmation.

**7.62 TWh/yr is technical potential, not a forecast.** It assumes every usable roof square metre
is covered. Economic, tenure, structural and grid constraints are all outside it. The plan's
ethics caveat — *"order-of-magnitude estimate, not an engineering assessment"* — applies in full
and should travel with the number wherever it is quoted.

**`k_usable` is 68 % of the error bar.** The figure is ±15.2 %, and two-thirds of that is one
assumed constant with no measurement behind it. The superstructure labelling pass would take the
whole estimate to roughly ±9 %.

## Decision

- [x] **Record 4.59 GWp / 7.62 TWh/yr ±15.2 % as the project's first end-to-end figure**, with
      every term's provenance attached.
- [x] **Never quote it without the error bar and the technical-potential caveat.**
- [x] **Script self-checks the un-erosion correction** rather than trusting the parameter.
- [ ] Replace PVOUT with a Global Solar Atlas map read (D15) — second-largest term, minutes of work.
- [ ] Measure `k_usable` — 68 % of the error bar, and the single highest-leverage action in the
      project.

## Threats to validity

- **PVOUT is secondary-source and not citable** (D15). The central value moves the headline
  linearly.
- **Partly in-sample:** 7,371 of 9,072 crops were training data. Val-only fg is 0.2015 against
  train 0.2330 — a difference driven mainly by which tiles fell in each split rather than by
  memorisation, but the total is not an out-of-sample estimate.
- **Agreement with Open Buildings, not accuracy.** D10 showed OB over-covers by drawing compound
  walls as buildings, so true roof area is plausibly *lower*. This is a systematic bias absent
  from the ±15.2 %.
- Technical potential only (above).
