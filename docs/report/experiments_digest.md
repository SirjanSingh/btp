# Experiments digest (input for the HTML research report)

Compiled 2026-11-21 from all 40 `experiments/*/README.md`, `README.md`, `HANDOFF.md` and a skim of `plan/`.
Numbers are copied from the READMEs. **OB** = scored against Open Buildings footprints (also the training
labels, so it measures agreement, not accuracy). **Hand** = scored against hand-drawn roofs (`labelling/r12_batch2`).
**BDAPPV** = scored on the labelled French solar set. Sept OB scores use the 1,701 val crops of tiles 1-1, 2-2, 4-3.

Gap in the record: the annotation-free self-training arms A-R (Oct 2026) have no README in this checkout
(`experiments/2026-10-30-annotation-free-selftrain/` is referenced by `duda/` and `HANDOFF.md` but absent).
Their numbers below come second-hand from `2026-10-30-duda` and `2026-11-19-hand-label-model-scoring`.

## 1. Chronological table

| Date / folder | Question | Method | Key result (verbatim, metric, vs) | Verdict | Technique / source named |
|---|---|---|---|---|---|
| 09-08 d6-seed-probe | What does the AIRS seed predict on Jaipur? | AIRS U-Net/R34 on 640 random crops, threshold sweep | mean predicted fg **5.69 %** @0.35 (10.19 % @0.05, 4.31 % @0.70); per tile 0.95-11.01 %. No labels | kept (seed unusable as naive teacher) | none |
| 09-09 d1-target-prior | True building-pixel prior | rasterise OB v3 (523,283 polygons) | prior **28.19 %** (all), **23.06 %** (conf>=0.75); per tile 10.70-41.23 %; "≈ 5×" under-prediction vs D6 | kept | sets CBST class ratio |
| 09-09 d2-d3-source-stats | AIRS prior and building size | 857 AIRS train masks, connected components | AIRS fg mean **7.69 %**, median 2.06 %; prior shift **3.66×**; median building 21,084 px (AIRS) vs **913 px** (Jaipur), "23×" | kept | none |
| 09-09 weak-supervision-jaipur | Can OB labels alone train a Jaipur model? | fine-tune AIRS seed on OB conf>=0.75 rasters, 7,371 train / 1,701 val crops | best val IoU **0.6475** (OB); P 0.7361, R 0.8434; recall > precision by ~0.11 | kept (primary track) | OB weak supervision (plan Tier 4) |
| 09-09 method-comparison | Which adaptation method wins? | 6 methods, global IoU, same 1,701 crops | IoU@0.5 (OB): weak+TTA **0.6534**, weak 0.6476, AdaBN 0.2523, seed 0.1419, seed+TTA 0.1331 (0.3863 at oracle thr 0.01), FDA 0.0891, histmatch 0.0258 | weak kept; FDA and histmatch rejected (harmful); AdaBN ablation only; TTA kept | FDA, histogram matching, AdaBN, TTA |
| 09-09 does-the-airs-seed-help | Is AIRS pretraining worth anything? | same run, ImageNet init | **0.6483** vs 0.6475 (OB); both best at epoch 33 | AIRS pretraining rejected | none |
| 09-09 boundary-relaxed-loss | Ignore ±4 px around label edges? | masked BCE + Dice | 0.6366 vs 0.6475 (**−0.0110**, OB); recall−precision gap 0.1073 → 0.0795 | rejected (1-2 px untested) | plan/03 Tier 4 boundary relaxation |
| 09-10 d4-adjacency | Do Jaipur buildings touch? | STRtree, 0.5 m tolerance, 318,207 polygons | **78.0 %** adjacent; 19.6 % have >=3 neighbours | kept (justifies instance metrics) | none |
| 09-10 merge-split-rate | Are buildings fused? | merge / split / missed at >=50 % overlap, 27,918 labels | R34 merge **0.5040**, pred/label 0.789; MiT-B2 merge 0.4615, pred/label 0.760; split 0.0000 (OB) | metric adopted | none |
| 09-10 segformer-backbone | MiT-B2 vs ResNet-34 | encoder swap (batch 12 vs 16) | **0.6569** vs 0.6483 (+0.0086, OB); precision +0.0173; reached 0.64 at epoch 8 vs 16 | adopted in Sept; not better on hand labels (Nov) | SegFormer; DAFormer's "architecture matters" claim |
| 09-10 label-quantity-vs-quality | All OB confidences vs conf>=0.75 | train on conf>=0.0 masks, val on conf>=0.75 | 0.6281 vs 0.6483 (OB conf-0.75 val); cross-eval: conf-0 model **0.7205** on conf-0 val ("do not quote") | Sept: confounded, keep 0.75. Nov hand labels: all-conf is best | none |
| 09-10 eroded-labels | Erode labels 0.4 m to stop merging | 2×2 {eroded, not} × {R34, MiT-B2}; val un-eroded | MiT-B2 eroded: IoU 0.6405, merge **0.3256**, pred/label **0.9745**; R34 eroded: 0.6395 / 0.4397 / 0.903; dilate 2 px: merge 0.4163, pred/label 0.8071 | erosion kept; dilation rejected | none |
| 09-10 erosion-sweep | Erode 0.8 m | MiT-B2 | IoU 0.5911, merge 0.1339, missed 0.4689, pred/label **1.4743**; + dilate 3 px: merge 0.3562, pred/label 0.9288 | 0.8 m rejected; dilation dropped | none |
| 09-10 erosion-02 | Erode 0.2 m | MiT-B2, `--cache_ram` | IoU 0.6540, merge 0.4118, split 0.0496, pred/label **0.8412** | 0.4 m confirmed | none |
| 09-10 eroded-labels-rerun | Rebuild deleted default weights | same recipe | IoU **0.6393** @ep19 (orig 0.6405); merge 0.3155, split 0.0840, pred/label **0.9914** (OB) | kept (reproduced) | none |
| 09-10 split-metric-audit | Is the split metric blind? | loose (>=10 %) vs strict (>=50 %) split | split_strict 0.0 at 0.2/0.4/0.8 m; loose 0.0496 / 0.0840 / 0.1762; at 0.8 m ~30 % of predictions touch no label | loose kept, strict dropped | none |
| 09-10 solar-google-to-ign (S1) | Source-only gap on BDAPPV | R34 trained on Google, tested on IGN | google_val **0.8723**, ign_val **0.5611** (−35.7 %); best threshold 0.5 on both | baseline (zero-negative bug C1) | none |
| 09-10 solar-self-training (S2) | CBST on IGN | ratio-matched 1.83 % → threshold 0.0004 | ign **0.3752** (best thr 0.8), P 0.391, R 0.902 | rejected | CBST |
| 09-10 solar-selftrain-conf (S4) | Fixed threshold instead of ratio | pseudo threshold 0.4563 | ign **0.6165**, P 0.789, R 0.738; "+0.0554", "17.8 % of the gap closed" | kept | confidence self-training |
| 09-10 selftrain-threshold-cliff (S6) | Where does self-training break? | ratios 0.008 / 0.010 / 0.012 | ign @0.5: 0.5904 / **0.5377** / 0.4884; crossover ratio **≈ 0.0091**; source IoU spans 0.0067 vs target 0.256 | threshold policy kept; CBST rejected | CBST |
| 09-10 selftrain-round2 (S7) | Does a 2nd round compound? | pseudo thr 0.45 from S4 | ign **0.6104** vs 0.6135 (−0.0031) | rejected (one round) | CBST multi-round |
| 09-10 jaipur-selftrain (R5) | Transfer S4 recipe to Jaipur | pseudo thr 0.5 (B) / 0.8 (A), MiT-B2 student | teacher 0.6393 / pred/label 0.9914; B: IoU 0.6432, merge 0.3371, pred/label **0.9134**; A: IoU 0.5673, pred/label 1.2282 (OB) | rejected (kill criterion fired) | S4 recipe |
| 09-11 d8-missed-by-size | Which buildings are missed? | miss rate by size bin | 50-200 px **0.7479** to 2000+ px **0.0714**; overall 0.2831; un-eroded model overall 0.2334 | kept; resolution experiment deferred | none |
| 09-11 d9-confidence-is-size | Can OB confidence validate small polygons? | conf by size, 523,283 polygons | mean conf 0.6998 (50-200 px) to 0.8347 (2000+); conf>=0.75 keeps **11 %** of small, **93 %** of large; only 20 small polygons at >=0.85 | kept (conf is a size filter) | none |
| 09-11 d10-missed-gallery | What do large misses look like? | render 20 missed cases | 3/3 largest inspected are OB errors (bare plot, compound); coverage 0.05-0.50 | kept; retracts D8 assumption | none |
| 09-11 d11-overlap-rule | How much miss rate is the rule? | min_overlap 0.5 → 0.05 | overall miss 0.2831 → **0.1598** (1.77×); 2000+ −81 %; 50-200 px −17 % (0.6215 left) | kept | none |
| 09-11 d12-pred-size-dist | Can the model emit small components? | unmatched size histograms | 50-200 px: 2,946 preds (12.0 %) vs 1,630 labels (5.8 %); medians 1,619 vs 1,657 px | resolution experiment cancelled | none |
| 09-11 d13-extra-predictions | Are extra small predictions fragments? | classify 5,418 small preds | isolated **39.0 %**, fragment **10.9 %**, sole **50.1 %** | kept | none |
| 09-11 d14-energy-budget | Which term controls the energy error? | relative-variance budget | seg area ratio 1.0156 ± 0.0219; shares k_usable **56.0 %**, PVOUT 29.6 %, η 9.0 %, PR 3.7 %, segmentation **1.7 %**; total ±16.7 % | superseded by D15 | none |
| 09-11 d15-pvout-lookup | PVOUT for Jaipur | literature / methodology read | PVOUT × PR double-counts losses: **22.5 %** underestimate; secondary PVOUT 1,550-1,900; corrected shares k_usable 68.4 %, seg 2.0 %, total ±15.1 % | formula fix kept; PVOUT provisional | World Bank / Solargis PVOUT methodology |
| 09-11 d16-capacity | First end-to-end figure | default model on all 9,072 crops | roof 38.22 km², **4.59 GWp, 7.62 TWh/yr, ±15.2 %**; first run 5.24 GWp was 14.3 % too high (bad un-erosion) | kept; revised Nov | none |
| 09-11 inference-threshold-sweep | Can threshold replace erosion? | thr 0.3-0.9 on teacher, E04, retrained un-eroded | teacher pred/label crosses 1.0 at ≈0.51; un-eroded retrain IoU 0.6580, best pred/label 0.9119 (thr 0.8, missed 0.4304) | erosion necessary; threshold complementary | none |
| 09-11 mit-b5 | More encoder capacity? | MiT-B5, batch 8 | IoU **0.6480** vs B2 3-seed mean 0.6417 (+0.0063, 1.5× 2 sd); 530 vs 147 s/epoch | rejected | MiT-B5 (DAFormer/HRDA prerequisite) |
| 09-11 seed-variance | Noise floor | default config, seeds 42/43/44 | IoU 0.6393 / 0.6423 / 0.6434 (2 sd **0.0042**); merge 2 sd 0.0502; pred/label 2 sd **0.0763** | kept: quote pred/label 0.99 ± 0.08 | none |
| 09-11 replication | Second seeds for two marginal claims | un-eroded s43, self-train t050 s43 | erosion: matched-merge pred/label gaps +0.1411 to +0.1813 (1.7-2.2× pooled 2 sd 0.0830); self-train pred/label mean 0.8418 vs teacher 0.9753, d/SE 1.78 | claim 1 solid; claim 2 directional | none |
| 09-11 selftrain-eroded-pseudo (R5b) | Does eroding pseudo-labels fix R5? | raster-erode pseudo masks (E04 0.4 m, E02 0.2 m) | E04: IoU 0.6068, merge 0.2666, missed 0.3974, pred/label 0.9672; E02: 0.6299 / 0.3051 / 0.3477 / 0.9447 | rejected (diagnosis confirmed) | none |
| 10-30 duda | Does large-teacher distillation avoid collapse? | MiT-B5 teacher/student, R34 small student, AIRS → Jaipur, no target labels | small student last IoU@0.5 **0.2673** (OB) vs its AdaBN start 0.3068; arms A-D 0.0720 / 0.0548 / 0.0405 / 0.1643 | rejected | DUDA, Kang et al. (README says WACV 2026, arXiv 2504.09814) |
| 11-19 hand-label-model-scoring | How good are models on hand-drawn roofs? | 29 checkpoints on 136 then 152 hand crops; tile split calib/test (76/76); crop bootstrap | Hand IoU@0.5 (136): R34 all-conf **0.632**, R34 conf>=0.75 0.605, MiT-B2 uneroded 0.593, default 0.564, **OB itself 0.504**, arm R 0.317, DUDA 0.275, AIRS seed 0.154. Shifted labels (7 px S / 3 px W): R34 all-conf test IoU **0.697** @0.4 (+0.038), seeds 0.697 / 0.693 / 0.693; MiT-B2 shifted 0.700 | kept; new best = R34 all-conf on shifted OB | none (k_usable leads: Saha & Mandal 2016; arXiv 2009.02524) |
| 11-20 solar-with-negatives | Panel detector usable on Jaipur? | add ~15k BDAPPV no-mask + 3,955 Jaipur ground negatives | val IoU **0.823** (val includes negatives); existing PV **0.444 km², 0.83 % of roof, ~89 MWp** (scale 2.6), 0.282 km² / ~56 MWp (scale 3.4) | kept (lower bound; no Jaipur panel labels) | none |
| 11-20 toward-higher-iou | Steps toward >0.70 hand IoU | offset leak check, TTA/ensembles, error budget, hand fine-tune, round-1 label variants | offset on calib alone 7 / −3 (no leak); R34 shifted + TTA test **0.706** (+0.010); ensembles add nothing; IoU ignoring 3 px edge band 0.776; 93 % of wrong pixels on edges; fine-tune folds −0.021 / +0.026; label gate: shift+SAM 0.629 / 0.650 vs model prediction 0.689 / 0.723 (calib / test) | TTA kept; hand fine-tune rejected; round 1 open (no training results yet) | Sirko et al. 2021, DINOv3, SAM (Kirillov 2023), OMAF / AnS 2026 |

## 2. Themes

**(a) Source model and domain adaptation.** The AIRS seed (in-domain val IoU 0.8784, unreproduced: HANDOFF says it came from a deleted `/tmp`) predicts 5.69 % foreground on Jaipur against a 28.19 % prior (d6, d1); AIRS itself is 7.69 % foreground with buildings 23× larger in pixels (d2-d3). Much of the zero-training gap is calibration (seed 0.1419 @0.5 vs 0.3329 at oracle threshold 0.01); FDA and histmatch are worse than nothing, and ImageNet init equals AIRS init. Jaipur self-training (R5 raw, R5b eroded pseudo-labels) never beats the teacher. Oct annotation-free arms A-D collapse to 0.04-0.16 OB IoU; DUDA ends at 0.2673, below AdaBN. On hand labels the best annotation-free arm (R) scores 0.317 (held-out 0.247), DUDA 0.275, AIRS seed 0.154. BLDA: no README or result in this checkout mentions it.

**(b) Weak supervision with OB.** Fine-tuning on OB conf>=0.75 gives 0.6475 OB IoU (4.6× the seed). Confidence: training on all 523k polygons looked worse on OB (0.6281) but cross-eval and d9 show conf>=0.75 is a size filter (keeps 11 % of small, 93 % of large). Hand labels settle it: all-conf R34 0.632 vs 0.605. Erosion: 0 / 0.2 / 0.4 / 0.8 m gives pred/label 0.7600 / 0.8412 / 0.9914 / 1.4743 (OB); 0.4 m adopted. Boundary relaxation (±4 px) −0.011. Offset shift: OB sits 7 px south / 3 px west of hand roofs (IQR dy 5-10, dx −4 to −1, all tiles dy 5-11); training on shifted labels lifts hand test IoU 0.659 → 0.697 while OB IoU drops 0.636 → 0.591. Every OB-trained model beats OB itself on hand labels (0.53-0.63 vs 0.50).

**(c) Architecture.** MiT-B2 +0.0086 over R34 on OB (2.0× 2 sd at n=3), gain all precision and 2× faster convergence; merge 4.3 points lower. MiT-B5 +0.0063 (1.5× 2 sd) at 3.6× compute: rejected. Both comparisons have batch-size confounds (12 vs 16, 8 vs 12). On hand labels MiT-B2/B5 do not beat R34 (uneroded 0.593 vs 0.632); on shifted labels MiT-B2 0.700 vs R34 0.697, "a tie, inside seed noise". DINOv3 ConvNeXt-T is queued in round 1, no result. "SegFormer" here means a U-Net with a MiT encoder; DeepLabV3+ was not possible with MiT.

**(d) Metric audits.** Merge/split (≥50 % overlap) exposed 50.4 % merging hidden by IoU. Split-metric audit: strict split is 0.0 everywhere, loose split 0.0496-0.1762. d11: the ≥50 % rule inflates miss rate 1.77×. Seed variance: IoU 2 sd 0.0042, merge 0.0502, pred/label 0.0763 (instance metrics ~20× noisier); several claims re-graded "not supported" (erosion 0 → 0.2 m merge, E04 curve vs teacher, R5b E02 vs E04). Replication found every run is early-stopped (`--patience 15`), not "40 epochs". Threshold sweep: 0.5 is near counting-optimal on OB (≈0.51), but on hand labels 0.3 beats 0.5 for every model and calib picks 0.3-0.4. Hand-label protocol: tile split calib/test, threshold chosen on calib, crop bootstrap, seed spread ~0.015 (136 set) or 0.004 (shifted R34 seeds).

**(e) Diagnostics.** D1 prior 28.19 % (23.06 % at conf>=0.75). D2/D3 AIRS 7.69 %, 23× pixel-size gap. D4 78 % adjacency. D5, D7 have no README. D6 seed predicts 5.69 %. D8 miss rate 10.5× higher for 50-200 px than 2000+ px; erosion not the cause. D9 OB confidence is a size proxy. D10 largest "misses" are OB drawing compounds and bare plots. D11 rule artefact 1.77×; small-building non-detection (62 %) is the one real residual. D12 model emits 1.8× more small components than labels; capacity hypothesis refuted. D13 small preds: 50.1 % sole, 39.0 % isolated, 10.9 % fragments. D14-D16 energy (below).

**(f) Solar stage.** Google → IGN source-only 0.8723 → 0.5611. CBST ratio matching (thr 0.0004) collapses to 0.3752; fixed threshold 0.4563 gives 0.6165; ratio crossover ≈0.0091 (CBST's prescribed 0.0183 is 2× past break-even); round 2 adds nothing (0.6104). All Sept solar numbers share bug C1 (zero negative crops). Nov: all 8 old checkpoints flag 3-8 % of Jaipur roof as panel at 2.6×, 31-54 % of it off-roof, in 135-151 / 152 crops. Retrained with negatives: 0.84 % flagged, 5 % off-roof, 54 / 152 crops at scale 2.6; existing PV 56-89 MWp, a lower bound.

**(g) Energy chain.**
- Capacity: `P = A_roof × k_usable × η × 1 kW/m²`, with k_usable 0.60, η 0.20. Check: 38.22 km² × 0.60 × 0.20 = 4.59 GWp.
- Energy: `E = P × PVOUT × derate`, derate 0.95, never `PVOUT × PR` (D15: that double-counts 3.5 % soiling plus 7.5 % system losses, a 22.5 % underestimate). The Sept figure used PVOUT 1,750 (stated in the Nov rerun); 7.62 / 4.59 = 1,660 ≈ 1,750 × 0.95 (my arithmetic).
- D14 budget (relative sds): segmentation 2.2 %, k_usable 12.5 %, η 5.0 %, PVOUT 9.1 % (1650 ± 150), PR 3.2 %. Shares 56.0 / 29.6 / 9.0 / 3.7 / 1.7 %, ±16.7 %. D15/D16 corrected: k_usable 68.0 %, PVOUT 14.2 % (5.7 %), η 10.9 %, derate 4.3 % (3.2 %), segmentation 2.0 %, erosion un-do 0.6 % (1.1 %); total **±15.2 %** → 3.89-5.29 GWp, 6.46-8.78 TWh/yr.
- Nov: hand area bias (pred/hand) 0.91 calib / 0.76 test for the old default; R34 shifted @0.4 predicts 53.3 km² (6.40 GWp / 10.64 TWh), bias 1.21 / 1.07, calibrated **5.29-5.98 GWp**.
- PVOUT now sourced: Global Solar Atlas **1,627-1,641 kWh/kWp/yr** (5 points). With 1,630 × 0.95: 5.29-5.98 GWp, **8.2-9.3 TWh/yr**.
- Minus existing PV: **5.20 GWp / 8.0 TWh** (calib) to **5.89 GWp / 9.1 TWh** (test).
- k_usable sensitivity: 0.50 → 4.41-4.98 GWp; 0.33 → 2.91-3.29 GWp. Verified sources: Bhopal 0.30, Kathmandu ~21 % gross. Mumbai, Nagpur and the 25-50 % study are not verified.

**(h) Hand labels and higher IoU.** The first real accuracy: the default eroded MiT-B2 scores 0.564 hand IoU and predicts 83 % of true roof area, while R34 all-conf scores 0.632. The model ranking under OB flips under hand labels. The 7/−3 shift has no test leak, since calib alone gives 7/−3. TTA adds +0.010 (0.706), and ensembles add nothing, because seeds make the same mistakes. 93 % of the error is on edges or partly covered roofs. Fine-tuning on 76 hand crops gives no reliable gain (folds −0.021 / +0.026). SAM and per-building alignment improve label quality (0.589 → 0.629 calib), but the model's own prediction is a better label still (0.689). Round 1 (edge-weighted BCE, DINOv3, lab_align, lab_sam, lab_align_sam) has no training results yet. The target is 0.75-0.80; the mentor asked for ~0.90. The README cites Inria's best supervised result as about 0.83.

## 3. Numbers that matter

1. Jaipur building prior 28.19 % (23.06 % at conf>=0.75), AIRS 7.69 %: shift 3.66× (d1, d2-d3).
2. AIRS seed predicts 5.69 % foreground on Jaipur (d6).
3. Median building 913 px Jaipur vs 21,084 px AIRS (d2-d3).
4. AIRS seed scores 0.1419 OB IoU @0.5, and 0.154 on hand labels (method-comparison, hand-label-model-scoring).
5. OB weak supervision scores 0.6475 OB IoU; ImageNet init scores 0.6483 (weak-supervision-jaipur, does-the-airs-seed-help).
6. FDA (0.0891) and histmatch (0.0258) score below no adaptation (method-comparison).
7. 78.0 % of buildings touch a neighbour (d4-adjacency).
8. Un-eroded models merge 46-50 % of buildings at ~0.65 IoU (merge-split-rate).
9. 0.4 m erosion gives pred/label 0.99 ± 0.08 (2 sd, n=3) (seed-variance).
10. Instance-metric noise: pred/label 2 sd 0.0763 vs IoU 2 sd 0.0042 (seed-variance).
11. Erosion beats un-eroded at matched merge by +0.14 to +0.18 pred/label, 1.7-2.2× pooled 2 sd (replication).
12. Jaipur self-training lowers counting (0.9753 → 0.8418, d/SE 1.78) (replication, jaipur-selftrain).
13. BDAPPV google → IGN: 0.8723 → 0.5611 (solar-google-to-ign).
14. On IGN, CBST scores 0.3752 vs fixed threshold 0.6165; crossover ratio ≈0.0091 (solar-self-training, solar-selftrain-conf, selftrain-threshold-cliff).
15. OB confidence is a size filter: conf>=0.75 keeps 11 % of small and 93 % of large buildings (d9).
16. The ≥50 % rule inflates miss rate 1.77× (0.2831 → 0.1598) (d11).
17. DUDA scores 0.2673 OB IoU, below AdaBN (0.3068); annotation-free arm R scores 0.317 hand (duda, hand-label-model-scoring).
18. OB itself scores 0.504 hand IoU; every OB-trained model beats it (hand-label-model-scoring).
19. The OB offset is 7 px south / 3 px west; shifted training reaches 0.697 hand test IoU, +0.038 (hand-label-model-scoring).
20. Best model: R34 + TTA, 0.706 hand test IoU; ignoring 3 px edges gives 0.776 (toward-higher-iou).
21. PVOUT × PR double-count would underestimate by 22.5 % (d15).
22. The Sept figure was 4.59 GWp / 7.62 TWh/yr ±15.2 %; k_usable is 68 % of the error bar and segmentation 2 % (d16).
23. GSA PVOUT is 1,627-1,641 kWh/kWp/yr; calibrated potential is 5.29-5.98 GWp, 8.2-9.3 TWh/yr (hand-label-model-scoring).
24. Existing rooftop PV is 0.444 km², ~89 MWp (56-89 MWp range), 1.5-1.7 % of potential (solar-with-negatives).
25. k_usable 0.33 instead of 0.60 cuts capacity to 2.91-3.29 GWp (hand-label-model-scoring).

## 4. Contradictions and numbers that changed

1. **Priors.** Assumed 50 % (Jaipur) and ~15 % (AIRS); measured 28.19 % and 7.69 %. The ratio survived (3.33× vs 3.66×); both absolutes were ~2× too high.
2. **"Weak model over-predicts at 31.2 %"** came from a 600-crop probe drawn from 2 dense tiles. Full set: 23.30 % vs a 23.1 % prior (jaipur-selftrain).
3. **D6 smoke vs full run:** 6.99 % vs 5.69 % at the same threshold.
4. **Flagship pred/label:** 0.9745 (original run), then 0.9914 (rerun, "within 0.9 %"), then 0.99 ± 0.08 across seeds 0.9317 / 0.9914 / 1.0027.
5. **Split rate.** eroded-labels, erosion-sweep and merge-split-rate tables show split 0.0. The rerun gives 0.0341 / 0.0840 / 0.1762 under the corrected definition. split-metric-audit says published numbers were "already" loose. **The READMEs disagree** about whether the early 0.0 values were strict or loose.
6. **Miss rate:** 0.3257 (single component ≥50 %), 0.2831 (all foreground ≥50 %, d8), 0.1598 (≥5 %, d11). "33 % missed" was later withdrawn as a model property.
7. **Erosion vs thresholding.** The first verdict ("deflationary", same curve) was reversed on the full un-eroded sweep (erosion strictly better).
8. **Best model flips between OB and hand labels.** On OB: MiT-B2 uneroded 0.6569 (Sept val) / 0.671 (OB column of the hand table) leads, and all-conf R34 trails (0.6281 / 0.636). On hand: all-conf R34 0.632 leads, MiT-B2 uneroded 0.593, default eroded 0.564. Shifted labels raise hand IoU (0.659 → 0.697) while lowering OB IoU (0.636 → 0.591).
9. **OB-scored IoUs for the same checkpoints differ between documents.** The Sept 1,701-crop val gives 0.6483 / 0.6569 / 0.6393. The hand-table OB column gives 0.655 / 0.671 / 0.647. The README does not say which crops its OB column uses (presumably the 136 hand crops).
10. **R3 "confounded, keep conf>=0.75"** became "all-conf is best" once scored on hand labels.
11. **Erosion, the "single most consequential improvement" (OB counts), costs area.** On hand labels the default predicts 0.83 of true roof area (0.76 on the test half) and is the weakest OB-trained R34/B2 variant in IoU.
12. **Segmentation error in the energy budget:** 2.2 % (seed spread vs OB, d14). Hand labels show area bias of 0.76-1.27 depending on model and threshold, and calib/test halves disagree by ~15 %. The README says the 2.2 % "understated this".
13. **Energy headline:** 5.24 GWp (first d16 run, un-erosion bug), then 4.59 GWp / 7.62 TWh (Sept), then "about 5-6 GWp (8.3-10 TWh/yr)", then 5.29-5.98 GWp / 8.2-9.3 TWh (GSA PVOUT), then 5.20-5.89 GWp / 8.0-9.1 TWh after subtracting existing PV.
14. **PVOUT:** 1,650 ± 150 (D14, India range), then 1,550-1,900 secondary (D15), then 1,750 (actually used in d16, per the Nov rerun), then 1,627-1,641 GSA.
15. **Total uncertainty:** ±16.7 % (D14), ±15.1 % (D15), ±15.2 % (D16). Later figures are given as ranges without a single ±.
16. **Threshold.** OB analysis: 0.5 is near counting-optimal (≈0.51). Hand labels: all models under-confident, 0.3 beats 0.5. The calib split picks 0.3-0.4.
17. **AdaBN:** 0.2523 (Sept) vs 0.306 / 0.3068 (Oct arms; "different BN sample").
18. **Existing PV:** ~79 MWp (epoch-10 snapshot), then ~89 MWp (final, scale 2.6), then 56 MWp (scale 3.4).
19. **Solar S4:** 0.6165 is given as "target @0.5" in solar-selftrain-conf. threshold-cliff and round2 list S4 @0.5 = 0.6135 with best 0.6165. HANDOFF quotes 0.617. Ambiguous which is the fixed-threshold number.
20. **Epochs:** every Sept write-up says "40 epochs", but all runs were early-stopped (patience 15).
21. **Citation years:** DUDA is "WACV 2026" in duda/ but dblp only has arXiv 2025. Kasmi et al. was "2025" in notes but dblp says 2024 (toward-higher-iou).
22. **AIRS baseline:** 0.8784 (seed val IoU in experiments) vs 0.9016 / 0.8664 (README AIRS table). HANDOFF says the 0.8784 run is unreproduced.
23. **Annotation-free self-training:** HANDOFF says self-training on Jaipur was "rejected after 4 attempts" (Sept R5/R5b). The Oct arms A-R plus DUDA add at least 10 more, with no README in this checkout.
