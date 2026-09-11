# Run ledger — every training and evaluation run, consolidated

Generated 2026-09-11 21:00 by `scripts/build_run_ledger.py`. Rebuild it after every run, and **always before deleting a checkpoint**.

Weights get deleted when the 40 GB quota runs low; these numbers do not. Full per-epoch curves for every run are in `RUN_LEDGER.json` — this table is just the summary.

**107 runs · 33 checkpoints currently on disk**

| Run | Stage | Kind | Epochs | Best IoU | @ep | F1 | Prec | Rec |
|---|---|---|---|---|---|---|---|---|
| `unet_resnet34_2000samples_10ep_20260330_manual` | rooftop | train | 10 | **0.8397** | 10 | 0.9128 | 0.9187 | 0.9070 |
| `unet_resnet34_2000samples_100ep_20260330_102932` | rooftop | train | 1 | **0.6782** | 1 | 0.8082 | 0.7212 | 0.9191 |
| `unet_resnet34_2000samples_100ep_20260330_125206` | rooftop | train | 100 | **0.8784** | 90 | 0.9353 | 0.9505 | 0.9205 |
| `eval_unet_resnet34_210samples_20260330_192603` | rooftop | eval | 0 | **0.9016** | thr 0.45 | 0.9483 | 0.9527 | 0.9438 |
| `eval_unet_resnet34_2000samples_20260330_193154` | rooftop | eval | 0 | **0.8723** | thr 0.35 | 0.9318 | 0.9424 | 0.9214 |
| `eval_unet_resnet34_20260330_193934` | rooftop | eval | 0 | **0.8664** | thr 0.3 | 0.9284 | 0.9390 | 0.9181 |
| `unet_resnet34_500samples_3ep_20260402_200251` | solar | train | 3 | **0.0546** | 3 | 0.1035 | 0.0548 | 0.9379 |
| `unet_resnet34_4000samples_100ep_20260402_201304` | solar | train | 30 | **0.8535** | 15 | 0.9209 | 0.9215 | 0.9204 |
| `eval_unet_resnet34_20260403_054802` | solar | eval | 0 | **0.7719** | thr 0.55 | 0.8713 | 0.8670 | 0.8756 |
| `unet_resnet34_100ep_20260403_062049` | solar | train | 33 | **0.8442** | 33 | 0.9155 | 0.9235 | 0.9077 |
| `unet_resnet34_100ep_20260403_163032` | solar | train | 5 | **0.8463** | 38 | 0.9167 | 0.9212 | 0.9123 |
| `unet_resnet34_100ep_20260403_185446` | solar | train | 19 | **0.8506** | 47 | 0.9193 | 0.9190 | 0.9196 |
| `unet_resnet34_100ep_20260404_051627` | solar | train | 2 | **0.8476** | 56 | 0.9175 | 0.9181 | 0.9169 |
| `unet_resnet34_100ep_20260404_061348` | solar | train | 38 | **0.8540** | 78 | 0.9212 | 0.9216 | 0.9209 |
| `eval_unet_resnet34_20260404_171828` | solar | eval | 0 | **0.8484** | thr 0.55 | 0.9180 | 0.9153 | 0.9207 |
| `eval_unet_resnet34_20260404_175648` | solar | eval | 0 | **0.8484** | thr 0.55 | 0.9180 | 0.9153 | 0.9207 |
| `unet_resnet34_40ep_20260909_152425` | rooftop | train | 40 | **0.6475** | 33 | 0.7861 | 0.7361 | 0.8434 |
| `unet_resnet34_40ep_20260909_171814` | rooftop | train | 40 | **0.6483** | 33 | 0.7866 | 0.7363 | 0.8443 |
| `unet_resnet34_40ep_20260909_173038` | rooftop | train | 1 | **0.5823** | 1 | 0.7360 | 0.7091 | 0.7651 |
| `unet_resnet34_40ep_20260909_173651` | rooftop | train | 37 | **0.6366** | 22 | 0.7779 | 0.7402 | 0.8197 |
| `unet_mit_b2_40ep_20260909_201300` | rooftop | train | 40 | **0.6569** | 37 | 0.7929 | 0.7536 | 0.8366 |
| `unet_resnet34_40ep_20260909_204035` | solar | train | 40 | **0.8723** | 40 | 0.9318 | 0.9308 | 0.9328 |
| `unet_resnet34_40ep_20260909_210501` | rooftop | train | 40 | **0.6281** | 36 | 0.7716 | 0.6771 | 0.8967 |
| `unet_resnet34_40ep_20260910_011232` | rooftop | train | 37 | **0.6395** | 22 | 0.7801 | 0.7463 | 0.8171 |
| `unet_mit_b2_40ep_20260910_011238` | rooftop | train | 34 | **0.6405** | 19 | 0.7808 | 0.7799 | 0.7818 |
| `unet_mit_b2_40ep_20260910_033956` | rooftop | train | 25 | **0.5911** | 10 | 0.7430 | 0.7978 | 0.6952 |
| `unet_resnet34_30ep_20260910_044423` | solar | train | 30 | **0.8678** | 26 | 0.9292 | 0.9260 | 0.9324 |
| `unet_mit_b2_40ep_20260910_053557` | rooftop | train | 34 | **0.6393** | 19 | 0.7799 | 0.7823 | 0.7776 |
| `unet_resnet34_30ep_20260910_083614` | solar | train | 30 | **0.8745** | 30 | 0.9331 | 0.9301 | 0.9360 |
| `unet_mit_b2_40ep_20260910_090326` | rooftop | train | 40 | **0.6540** | 26 | 0.7908 | 0.7647 | 0.8188 |
| `unet_resnet34_30ep_20260910_142120` | rooftop | train | 30 | **0.8708** | 26 | 0.9309 | 0.9321 | 0.9298 |
| `unet_mit_b5_40ep_20260911_010428` | rooftop | train | 40 | **0.6480** | 25 | 0.7864 | 0.7773 | 0.7958 |
| `d11_ov0.05` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d11_ov0.1` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d11_ov0.2` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d11_ov0.3` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d11_ov0.5` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d12_pred_size_dist` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d14_energy_budget` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d16_capacity` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d1_summary` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d2_d3_summary` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d4_summary` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d6_summary` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d8_missed_by_size` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `d8_uneroded` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `e04_thr_0.3` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `e04_thr_0.4` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `e04_thr_0.6` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `e04_thr_0.7` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `eval_unet_resnet34_210samples_20260330_192349` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `method_comparison` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_erode02` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_erode08` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_erode08_dil3` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_eroded` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_eroded_dilate2` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b2_eroded_rerun` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `mit_b5` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r11_split_e02` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r11_split_e04` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r11_split_e08` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r3_cross_eval` | diagnostic | eval | 0 | — | — | — | — | — |
| `r5_t050` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r5_t080` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r5b_e02` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `r5b_e04` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `rep_st050_s43` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `rep_uneroded_s43_thr0.5` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `rep_uneroded_s43_thr0.7` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `rep_uneroded_s43_thr0.8` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `resnet34` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `resnet34_eroded` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s1_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s2_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s4_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s6_r008_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s6_r010_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s6_r012_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `s7_round2_crossdomain` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `seed43` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `seed44` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.3` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.4` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.6` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.7` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.8` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `thr_sweep_0.9` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.3` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.4` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.5` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.6` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.7` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `uneroded_thr_0.8` | diagnostic | diagnostic | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260330_111230` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260330_112025` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260330_115938` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260330_124500` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260402_200839` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260403_180709` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260404_050947` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_100ep_20260404_051053` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_2000samples_100ep_20260330_112130` | rooftop | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_500samples_3ep_20260402_192118` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_500samples_3ep_20260402_193129` | solar | train (txt only) | 0 | — | — | — | — | — |
| `unet_resnet34_500samples_3ep_20260402_194953` | solar | train (txt only) | 0 | — | — | — | — | — |

## Runs that completed zero epochs

Kept deliberately — a crashed configuration is evidence too.

- `eval_unet_resnet34_210samples_20260330_192603` (rooftop/logs/eval_unet_resnet34_210samples_20260330_192603.json)
- `eval_unet_resnet34_2000samples_20260330_193154` (rooftop/logs/eval_unet_resnet34_2000samples_20260330_193154.json)
- `eval_unet_resnet34_20260330_193934` (rooftop/logs/eval_unet_resnet34_20260330_193934.json)
- `eval_unet_resnet34_20260403_054802` (solar_panel/logs/eval_unet_resnet34_20260403_054802.json)
- `eval_unet_resnet34_20260404_171828` (solar_panel/logs/eval_unet_resnet34_20260404_171828.json)
- `eval_unet_resnet34_20260404_175648` (solar_panel/logs/eval_unet_resnet34_20260404_175648.json)
- `d11_ov0.05` (diagnostics/d11_ov0.05.json)
- `d11_ov0.1` (diagnostics/d11_ov0.1.json)
- `d11_ov0.2` (diagnostics/d11_ov0.2.json)
- `d11_ov0.3` (diagnostics/d11_ov0.3.json)
- `d11_ov0.5` (diagnostics/d11_ov0.5.json)
- `d12_pred_size_dist` (diagnostics/d12_pred_size_dist.json)
- `d14_energy_budget` (diagnostics/d14_energy_budget.json)
- `d16_capacity` (diagnostics/d16_capacity.json)
- `d1_summary` (diagnostics/d1/d1_summary.json)
- `d2_d3_summary` (diagnostics/d2_d3/d2_d3_summary.json)
- `d4_summary` (diagnostics/d4/d4_summary.json)
- `d6_summary` (diagnostics/d6_smoke/d6_summary.json)
- `d8_missed_by_size` (diagnostics/d8_missed_by_size.json)
- `d8_uneroded` (diagnostics/d8_uneroded.json)
- `e04_thr_0.3` (diagnostics/e04_thr_0.3.json)
- `e04_thr_0.4` (diagnostics/e04_thr_0.4.json)
- `e04_thr_0.6` (diagnostics/e04_thr_0.6.json)
- `e04_thr_0.7` (diagnostics/e04_thr_0.7.json)
- `eval_unet_resnet34_210samples_20260330_192349` (rooftop/logs/eval_unet_resnet34_210samples_20260330_192349.txt)
- `method_comparison` (diagnostics/method_comparison.json)
- `mit_b2` (diagnostics/merge_split/mit_b2.json)
- `mit_b2_erode02` (diagnostics/merge_split/mit_b2_erode02.json)
- `mit_b2_erode08` (diagnostics/merge_split/mit_b2_erode08.json)
- `mit_b2_erode08_dil3` (diagnostics/merge_split/mit_b2_erode08_dil3.json)
- `mit_b2_eroded` (diagnostics/merge_split/mit_b2_eroded.json)
- `mit_b2_eroded_dilate2` (diagnostics/merge_split/mit_b2_eroded_dilate2.json)
- `mit_b2_eroded_rerun` (diagnostics/merge_split/mit_b2_eroded_rerun.json)
- `mit_b5` (diagnostics/mit_b5.json)
- `r11_split_e02` (diagnostics/r11_split_e02.json)
- `r11_split_e04` (diagnostics/r11_split_e04.json)
- `r11_split_e08` (diagnostics/r11_split_e08.json)
- `r3_cross_eval` (diagnostics/r3_cross_eval.json)
- `r5_t050` (diagnostics/r5_t050.json)
- `r5_t080` (diagnostics/r5_t080.json)
- `r5b_e02` (diagnostics/r5b_e02.json)
- `r5b_e04` (diagnostics/r5b_e04.json)
- `rep_st050_s43` (diagnostics/rep_st050_s43.json)
- `rep_uneroded_s43_thr0.5` (diagnostics/rep_uneroded_s43_thr0.5.json)
- `rep_uneroded_s43_thr0.7` (diagnostics/rep_uneroded_s43_thr0.7.json)
- `rep_uneroded_s43_thr0.8` (diagnostics/rep_uneroded_s43_thr0.8.json)
- `resnet34` (diagnostics/merge_split/resnet34.json)
- `resnet34_eroded` (diagnostics/merge_split/resnet34_eroded.json)
- `s1_crossdomain` (diagnostics/s1_crossdomain.json)
- `s2_crossdomain` (diagnostics/s2_crossdomain.json)
- `s4_crossdomain` (diagnostics/s4_crossdomain.json)
- `s6_r008_crossdomain` (diagnostics/s6_r008_crossdomain.json)
- `s6_r010_crossdomain` (diagnostics/s6_r010_crossdomain.json)
- `s6_r012_crossdomain` (diagnostics/s6_r012_crossdomain.json)
- `s7_round2_crossdomain` (diagnostics/s7_round2_crossdomain.json)
- `seed43` (diagnostics/seed43.json)
- `seed44` (diagnostics/seed44.json)
- `thr_sweep_0.3` (diagnostics/thr_sweep_0.3.json)
- `thr_sweep_0.4` (diagnostics/thr_sweep_0.4.json)
- `thr_sweep_0.6` (diagnostics/thr_sweep_0.6.json)
- `thr_sweep_0.7` (diagnostics/thr_sweep_0.7.json)
- `thr_sweep_0.8` (diagnostics/thr_sweep_0.8.json)
- `thr_sweep_0.9` (diagnostics/thr_sweep_0.9.json)
- `uneroded_thr_0.3` (diagnostics/uneroded_thr_0.3.json)
- `uneroded_thr_0.4` (diagnostics/uneroded_thr_0.4.json)
- `uneroded_thr_0.5` (diagnostics/uneroded_thr_0.5.json)
- `uneroded_thr_0.6` (diagnostics/uneroded_thr_0.6.json)
- `uneroded_thr_0.7` (diagnostics/uneroded_thr_0.7.json)
- `uneroded_thr_0.8` (diagnostics/uneroded_thr_0.8.json)
- `unet_resnet34_100ep_20260330_111230` (logs/unet_resnet34_100ep_20260330_111230.txt)
- `unet_resnet34_100ep_20260330_112025` (logs/unet_resnet34_100ep_20260330_112025.txt)
- `unet_resnet34_100ep_20260330_115938` (logs/unet_resnet34_100ep_20260330_115938.txt)
- `unet_resnet34_100ep_20260330_124500` (logs/unet_resnet34_100ep_20260330_124500.txt)
- `unet_resnet34_100ep_20260402_200839` (solar_panel/logs/unet_resnet34_100ep_20260402_200839.txt)
- `unet_resnet34_100ep_20260403_180709` (solar_panel/logs/unet_resnet34_100ep_20260403_180709.txt)
- `unet_resnet34_100ep_20260404_050947` (solar_panel/logs/unet_resnet34_100ep_20260404_050947.txt)
- `unet_resnet34_100ep_20260404_051053` (solar_panel/logs/unet_resnet34_100ep_20260404_051053.txt)
- `unet_resnet34_2000samples_100ep_20260330_112130` (logs/unet_resnet34_2000samples_100ep_20260330_112130.txt)
- `unet_resnet34_500samples_3ep_20260402_192118` (solar_panel/logs/unet_resnet34_500samples_3ep_20260402_192118.txt)
- `unet_resnet34_500samples_3ep_20260402_193129` (solar_panel/logs/unet_resnet34_500samples_3ep_20260402_193129.txt)
- `unet_resnet34_500samples_3ep_20260402_194953` (solar_panel/logs/unet_resnet34_500samples_3ep_20260402_194953.txt)
