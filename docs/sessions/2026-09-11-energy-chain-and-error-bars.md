# 2026-09-11 — the day the project stopped optimising segmentation

Continuation of the autonomous runner. Day 1 (see `2026-09-10-*.md`) settled the segmentation
pipeline. Day 2 measured how much that was worth, and the answer reframed the project.

---

## The arc in one line

**Segmentation contributes 2 % of the uncertainty on the number the project exists to produce.**
Everything else in this document follows from discovering that, and from the three method errors
found while getting there.

---

## Part 1 — finishing the segmentation line

**Self-training rejected on four arms, not two.** R5 (Jaipur, thresholds 0.80/0.50) and R5b
(eroded pseudo-labels, 0.4 m/0.2 m) all failed to beat the teacher on `pred/label`. The
diagnosis held up — pseudo-labels are the teacher's *raw* output and discard the label erosion
that made the teacher good, and eroding them recovers 0.9134 → 0.9672 — but no arm won.

**MiT-B5 rejected on cost.** +0.0063 IoU (1.5× the noise floor) for 3.6× compute and a 971 MB
checkpoint that nearly breached the quota. Capacity is not what limits this problem.

**Erosion confirmed as necessary**, curve-against-curve after retraining the un-eroded model
that `filter-branch` destroyed. At matched merge rate the eroded model wins recall *and* counting
at every point; the un-eroded model's `pred/label` tops out at 0.9119 at any threshold.

---

## Part 2 — the seed-variance result, which invalidated things ★

**First time any configuration in this project had been run twice.** Three seeds of the default:

| metric | 2 sd |
|---|---|
| val IoU | **0.0042** |
| merge | **0.0502** |
| `pred/label` | **0.0763** |

IoU is the *stable* metric. The instance metrics this project elevated *over* IoU — precisely
because they see instance structure — swing 20× more. `pred/label` is a ratio of connected-
component **counts**: a few-thousandths probability shift near a boundary joins or splits a blob
and moves the count by one. Pixel IoU averages over ~450 M pixels; counts do not average at all.

**Roughly twenty experiments had been compared on single runs of a metric with a ±0.076 floor.**
Three claims sat at or below it and are now marked unsupported in their own write-ups, including
R5b's "monotone dose-response across all four metrics", which was over-read from a difference
smaller than run-to-run variance.

**The flagship number needed an error bar.** "0.4 m erosion gives `pred/label` 0.9914 — within
0.9 %" became **0.99 ± 0.08**. The conclusion survives (0.4 m is still the only erosion setting
whose interval contains 1.0) but the stated precision was two orders of magnitude too confident.

**Replication then promoted one claim and demoted another.** Erosion went from 1.2–1.9× 2 sd
against a single run to four matched points all clearing a pooled floor. Self-training's
counting regression went from "0.9914 → 0.9134" to "lowers counting accuracy and makes it 2.7×
less reproducible", d/SE 1.78 — directional, not precise.

---

## Part 3 — decomposing the 33 % miss rate (D8–D13)

Five cheap diagnostics, each minutes of inference, which between them **cancelled or redirected
hours of planned GPU work**:

- **D8** — miss rate is 10.5× worse for small buildings; 70 % of misses are under 900 px.
- **D9** — Open Buildings' confidence **is a size proxy**. At conf ≥ 0.85 only 20 small polygons
  survive city-wide, so OB's metadata cannot arbitrate whether its small polygons are real.
  Also explains why R3 was confounded: varying the confidence threshold varies the *size
  distribution* of the labels.
- **D10** — *looked at* the missed buildings. The largest are OB drawing **compound walls and
  plot boundaries** over bare ground and gardens. Refuted an assumption written into D10's own
  design.
- **D11** — the inherited **≥50 % overlap rule inflates the miss rate 1.77×**. Large-building
  misses are 81 % artefact; small-building misses are 83 % real.
- **D12** — the model emits **1.8× more** small components than the labels contain. The capacity
  hypothesis is refuted and the resolution experiment was **cancelled before being run**.
- **D13** — only 10.9 % of those small predictions are fragments; 50 % are correct sole
  detections, 39 % isolated. The disagreement with OB is **two-sided**.

**Net:** the "largest remaining Stage-1 error" decomposed into a rule artefact, OB errors at both
size extremes, and one residual localisation disagreement — every remaining question routing
through the hand labels.

---

## Part 4 — the energy chain, and three errors in it

**D14** built the uncertainty budget and found segmentation is **1.7 %** while `k_usable` is
**56 %**. Driving segmentation error to zero moves the headline by 0.1 pp.

**D15** went to look up PVOUT and found something larger: `MASTER_CONTEXT` §7.1 offered
`PVOUT × PR` and `GTI × η × PR` as interchangeable. **They are not.** Solargis PVOUT already
includes 3.5 % soiling + 7.5 % shading/mismatch/inverter losses — it *is* specific yield.
Multiplying by PR applies them twice: **a 22.5 % underestimate**, thirteen times the whole
segmentation term, sitting unexercised because the chain had never been run end to end.
Corrected in place with a `[CORRECTED]` block.

**D16** ran the chain for the first time. **4.59 GWp / 7.62 TWh/yr, ±15.2 %.** The first run said
5.24/8.72 — I had applied an un-erosion correction (model trains on eroded labels, so divide by
0.8748) that the script's own printout contradicted two lines above: the teacher is validated and
best-epoch-selected on the **un-eroded** val set, so its output is already at un-eroded extent
(`pred/label` 0.9849). **14.3 % inflation, caught by a number already on screen.** The script now
computes the ratio and refuses the correction; the guard was then tested in both directions,
since an untested guard is the same trap it exists to prevent.

---

## What this leaves

Priority order, none of it compute:

| rank | action | removes | cost |
|---|---|---|---|
| 1 | Measure `k_usable` (superstructure pass, ~100 tiles) | 68 % | hours of labelling |
| 2 | Read Jaipur PVOUT/OPTA off the Global Solar Atlas | 14 % | two minutes |
| 3 | Choose and state a module (η) | 11 % | a decision |
| 4 | Hand-label the 20 staged tiles | — | unblocks every accuracy claim |

**Deliverables produced:** `labelling/TEAM_BRIEF.md` (20-crop task, tool setup, conventions,
suggested split), `labelling/SUPERSTRUCTURE_PASS.md` (standalone, not yet sent), and two
artifacts — a label-erosion visual review and a results briefing.

**Artifact URLs** (private to Sirjan's account):
- label review — https://claude.ai/code/artifact/0bbbc818-7325-4484-8686-938ddcffc0f8
- results briefing — https://claude.ai/code/artifact/ca01711e-a9e4-4bc0-ae5a-368446ee9ac3

**Cron:** job `730b84ee`, hourly at :13, **session-only — dies when the Claude session exits**,
auto-expires after 7 days. It now checks for arriving hand labels each tick and is explicitly
told not to invent GPU work to fill idle slots.

---

## The lesson I would carry forward

Five of the day's findings were errors in the *method*, not results from it — the double-count,
the overlap rule, the un-erosion correction, the missing error bars, and the assumption that
large OB polygons are trustworthy. Every one was found by **looking at something already
available**: a printed intermediate, a second seed, a rendered image, a swept constant.

None needed new data or new compute. The expensive thing all day was not measurement — it was
the assumptions that went unexamined long enough to become load-bearing.
