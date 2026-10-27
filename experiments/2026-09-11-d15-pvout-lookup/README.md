# 2026-09-11-d15-pvout-lookup — PVOUT for Jaipur, and a double-count in the plan's formula

| | |
|---|---|
| **Status** | ★ done — formula error found; PVOUT provisional |
| **Date** | 2026-09-11 |

## Question

D14 ranked **PVOUT at 29.6 %** of the energy figure's variance — second only to `k_usable`, and
fixable by a lookup rather than an experiment. This is that lookup.

## Result 1 — a double-count in `MASTER_CONTEXT` §7.1 ★

The plan gives two forms:

```
E_annual = P_installed (kWp) × PVOUT (kWh/kWp/yr) × PR          ← form 1
         = usable_area_m² × η × GTI_annual (kWh/m²/yr) × PR      ← form 2
```

**Form 1 double-counts system losses.** The World Bank / Solargis methodology behind PVOUT states
that the figure already incorporates:

- **3.5 %** loss from dirt and soiling
- **7.5 %** cumulative loss from inter-row shading, mismatch, inverters, cables and transformers

plus module temperature response. PVOUT *is* specific yield — energy already net of losses.
Multiplying it by a PR of 0.75–0.80 applies those losses **a second time**.

**Magnitude: a 22.5 % underestimate of annual energy.** That is **13× the entire segmentation
term** measured in D14, and it was sitting unexercised in the plan because the chain has never
been run end to end.

**Form 2 is correct.** GTI is pure irradiance with no losses applied, so a PR of 0.75–0.80
belongs there. The two forms are not interchangeable, and the plan presents them as if they were.

**Corrected guidance:** use **either** `PVOUT × (small rooftop-specific derate)` **or**
`GTI × η × PR`. Never `PVOUT × PR`. A rooftop derate below PVOUT is still justified — Rajasthan
dust exceeds the 3.5 % soiling GSA assumes, and roof mounting ventilates worse than the
free-standing racking it models — but it is a **residual** adjustment of order 0.95, not a full PR.

## Result 2 — Jaipur PVOUT, provisional

| source type | value (kWh/kWp/yr) |
|---|---|
| secondary/commercial estimates for Jaipur | **1,550 – 1,900**, clustering **1,650–1,825** |
| stated basis | tier-1 modules, south-facing at tilt, PR 0.75–0.80 already applied |

**Provenance: [SECONDARY — NOT CITABLE].** The World Bank's *Global Photovoltaic Power Potential
by Country* does not publish India-specific PVOUT in its text, and the authoritative per-location
figure requires reading the Global Solar Atlas interactive map at the AOI centroid. Everything
above is from commercial solar-industry pages, which is adequate for **sizing uncertainty** and
inadequate for **quoting a result**.

**This is deliberately not resolved by inventing a number.** The remaining action is a two-minute
map read at globalsolaratlas.info for Jaipur's centroid, recording **PVOUT, GTI, OPTA** and the
data vintage.

**Useful side note:** GSA models *"free-standing structures with monofacial crystalline silicon
PV modules fixed mounted at an optimum tilt"* — which matches the project's flat-roof,
frame-mounted-at-OPTA assumption (§7.2). The configuration is appropriate; only the rooftop
derate differs.

## Result 3 — the corrected uncertainty budget

| term | rel sd | share (D14, wrong) | share (corrected) |
|---|---|---|---|
| **`k_usable`** | 12.5 % | 56.0 % | **68.4 %** |
| PVOUT | 5.7 % | 29.6 % | 14.3 % |
| η | 5.0 % | 9.0 % | 10.9 % |
| rooftop derate | 3.2 % | 3.7 % | 4.4 % |
| **segmentation** | 2.2 % | 1.7 % | **2.0 %** |
| **total (1 sd)** | | **±16.7 %** | **±15.1 %** |

Narrowing PVOUT to a Jaipur-specific range and removing the spurious PR term **raises
`k_usable`'s share to 68 %**. The conclusion from D14 strengthens: `k_usable` is not merely the
largest term, it is approaching two-thirds of the total.

## Decision

- [x] **Record the double-count as a correction** in `MASTER_CONTEXT` §0. It is a 22.5 % error in
      the project's headline deliverable and the most consequential single thing found today.
- [x] **Segmentation drops to 2.0 %** of the budget under the corrected chain.
- [ ] **Read PVOUT / GTI / OPTA for Jaipur off the Global Solar Atlas map** and replace the
      provisional range. Two minutes, and it is the only remaining blocker on the second-largest
      term. Not doable from here — the value needs the interactive map.
- [ ] Choose and state the module (η) — currently `[PLANNED]` at 0.20 and worth 10.9 %.

## Threats to validity

- **PVOUT range is secondary-source.** Suitable for uncertainty sizing only; the write-up says so
  and the number is not used for any quoted result.
- The rooftop derate of 0.95 ± 0.03 is my own estimate for Rajasthan dust and roof-mounting
  ventilation relative to GSA's free-standing assumption. It is `[ASSUMED]` and flagged as such —
  it should be justified from literature before publication, though at 4.4 % of variance it is
  not urgent.
- The loss percentages quoted are from the global PV-potential methodology; whether the live GSA
  layers use identical figures for every vintage was not verified.
