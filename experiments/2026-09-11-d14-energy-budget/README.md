# 2026-09-11-d14-energy-budget — which term actually controls the headline energy figure?

| | |
|---|---|
| **Status** | ★★ done — segmentation is 1.7 % of the uncertainty |
| **Date** | 2026-09-11 |

## Question

`MASTER_CONTEXT` §7 specifies the energy chain but leaves three constants open: **GTI/PVOUT**
and **PR** are `[TODO]`, **`k_usable`** is `[ASSUMED]` at 0.60 with a noted ±25 % swing.
Meanwhile this session has spent a full day measuring segmentation error to four decimal places.

Nobody has asked the question that decides where effort belongs:

> **If every term is uncertain, which one controls the answer?**

**Why it matters:** the project's prime directive is energy (kWh/yr), not segmentation. If
segmentation contributes a few percent of the output variance and `k_usable` contributes twenty,
then no amount of further segmentation work can move the headline figure — and the last day of
diagnostics, however sound, was spent on the wrong term.

## Method

The chain is a **product**, so relative variances add and each term's share of output variance is
its squared relative sd over the sum.

**Area, not count.** Energy depends on total usable roof *area*. This project's headline instance
metric `pred/label` is a **count** ratio and is the wrong input here — a model can count
buildings perfectly and still mis-estimate area. The segmentation term is therefore measured
directly as predicted foreground area ÷ labelled foreground area, across the **three seeds
already trained**.

**No fabricated irradiance.** Jaipur PVOUT is `[TODO]` in the plan and is not invented. The run
uses the plan's own stated India range as a wide interval, specifically to *size the cost of not
having looked it up*.

## Results

**Segmentation area ratio, measured across three seeds:** 0.9942 · 1.0380 · 1.0146
→ **1.0156 ± 0.0219** (1 sd) — the model's total roof area is within **1.6 %** of the labels.

| term | mean | sd | rel sd | provenance |
|---|---|---|---|---|
| segmentation area ratio | 1.0156 | 0.0219 | **2.2 %** | MEASURED, n = 3 seeds |
| `k_usable` | 0.60 | 0.075 | **12.5 %** | ASSUMED — plan notes ±25 % |
| η (module efficiency) | 0.20 | 0.010 | 5.0 % | PLANNED — mono-PERC |
| PVOUT (kWh/kWp/yr) | 1650 | 150 | **9.1 %** | TODO — India range, *not* Jaipur |
| PR | 0.775 | 0.025 | 3.2 % | TODO — plan states 0.75–0.80 |

### Share of output variance

| term | share |
|---|---|
| **`k_usable`** | **56.0 %** |
| **PVOUT** | **29.6 %** |
| η | 9.0 % |
| PR | 3.7 % |
| **segmentation** | **1.7 %** |

**Total uncertainty on the energy figure: ±16.7 % (1 sd), ±33.4 % (2 sd).**

## Interpretation

**Segmentation contributes 1.7 % of the variance. Driving its error to *zero* would take the
headline from ±16.7 % to ±16.6 %.** A day of erosion sweeps, self-training arms, encoder
comparisons and seed replication — all of it sound — moves the project's actual deliverable by a
tenth of a percentage point.

**`k_usable` alone is 56 %.** Reducing its relative sd from 12.5 % to 4 % — which is what
measuring superstructure coverage on ~100 tiles would achieve — takes the headline from
**±16.7 % to ±11.8 %**. That single labelling pass is worth **~50× more** than perfect
segmentation.

This is exactly what `MASTER_CONTEXT` §7.4 asserted — *"the cheapest genuine contribution
available in the project"* — and it is now **quantified rather than argued**.

**PVOUT at 29.6 % is free to fix.** It is a lookup from the Global Solar Atlas, not an
experiment. It is second only to `k_usable` and costs an afternoon of nobody's GPU time.

### The resulting priority order

| rank | action | variance removed | cost |
|---|---|---|---|
| 1 | Measure `k_usable` (superstructure pass, ~100 tiles) | 56 % | hours of labelling already planned |
| 2 | Look up Jaipur PVOUT/OPTA from Global Solar Atlas | 30 % | an afternoon, no compute |
| 3 | Fix η by choosing and stating a module | 9 % | a decision, not work |
| 4 | Justify PR for Rajasthan heat + dust | 4 % | a paragraph with citations |
| 5 | **Any further segmentation work** | **1.7 %** | **days of GPU** |

## The caveat that matters most

**±2.2 % is the *random* component only.** The area ratio is agreement with Open Buildings, not
accuracy. D10 showed OB systematically **over-covers** — drawing compound walls and plot
boundaries as buildings — so true roof area is plausibly *lower* than OB's, by an amount this
number cannot see. That is a **systematic bias**, and it does not appear anywhere in the budget
above.

So the honest statement is: **segmentation *precision* is a solved problem for this purpose;
label *accuracy* may not be.** And the fix for label accuracy is the same R12 pass that fixes
`k_usable` — which makes the labelling batch the answer to both the largest quantified term and
the largest unquantified one.

## Decision

- [x] **Stop optimising segmentation for the energy deliverable.** It is measured, it is precise
      to 2.2 %, and it is 1.7 % of the budget. Further arms need a different justification —
      e.g. the per-building counting result, which is a separate claim from total area.
- [x] **Reprioritise onto `k_usable`.** The superstructure pass moves the headline ~50× more per
      hour spent than segmentation does.
- [ ] **Look up Jaipur PVOUT and OPTA** from the Global Solar Atlas and replace the India-range
      placeholder. Second-largest term, near-zero cost. Not done here because inventing a number
      would defeat the purpose of the exercise.
- [ ] Re-run this budget once `k_usable` is measured and PVOUT is real — the shares will move and
      the ordering should be rechecked, not assumed.

## Threats to validity

- **Systematic OB bias is absent from the budget** (above). It is the single largest omission.
- sd values for `k_usable`, η, PVOUT and PR are 1-σ stand-ins for the plan's *stated ranges*, not
  measured dispersions. Only the segmentation term is measured. The ordering is robust to
  plausible changes in those stand-ins — `k_usable` would have to be ~5× better specified than
  stated before segmentation mattered — but the exact percentages are not precise.
- Independence is assumed between terms. `k_usable` and segmentation are plausibly correlated
  (a model that over-covers roofs also over-counts usable area), which would change the total but
  not the ordering.
- Area ratio measured at threshold 0.5 on the val tiles only.
