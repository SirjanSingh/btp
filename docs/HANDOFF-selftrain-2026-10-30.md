# Handoff — annotation-free self-training on Jaipur (2026-10-30)

Everything runs on the DGX (`23ucs715@172.22.2.151`, repo at `~/btp`). The jobs keep
running if the laptop that started them shuts down (they are `nohup` + Docker on the server).

## Goal
Adapt the AIRS-trained rooftop UNet (ResNet-34) to Jaipur with NO target labels
(unsupervised domain adaptation), so the same pipeline works for any country.

## State at handoff
- Two runs are executing back-to-back on **GPU 1** inside Docker, from
  `experiments/2026-10-30-annotation-free-selftrain/run_arms.sh`:
  - **Arm A**: self-training from the AIRS seed (no AdaBN).
  - **Arm B**: AdaBN (40 target batches) then the same self-training.
  - Both: 3000 iters, eval every 500, pseudo-label thresholds hi=0.8 / lo=0.2, seed 42.
- Progress so far (Arm A, teacher, on the 1,701 eval crops):
  - iter 0 source-only: IoU@0.5 = **0.1419** (matches the earlier method-comparison table exactly, so the eval set was rebuilt correctly)
  - iter 500: IoU@0.5 = 0.1526, best-threshold IoU 0.2010 (thr 0.2)
- Finished marker: `experiments/2026-10-30-annotation-free-selftrain/logs/ARMS_DONE`

## How to read results (on the DGX)
```bash
cd ~/btp/experiments/2026-10-30-annotation-free-selftrain
ls logs                       # one self_train_*.json per arm (written every eval)
python3 - <<'EOF'
import json,glob
for f in sorted(glob.glob("logs/self_train_*.json")):
    r=json.load(open(f)); print("==",f,"adabn_batches=",r["config"]["adabn_batches"])
    for h in r["history"]:
        t=h["teacher_target"]
        print(h["iter"],h.get("tag",""),"IoU@0.5=%.4f best=%.4f(thr %s) roof_frac=%.3f"%(
          t["by_threshold"]["0.5"]["iou"],t["best_iou"],t["best_threshold"],h["teacher_unlabelled"]["pred_roof_fraction"]))
EOF
ps aux | grep self_train | head; docker ps | head     # is it still running?
```
Checkpoints: `ckpt/armA_seed/`, `ckpt/armB_adabn/` (`best_teacher.pth`, `last_teacher.pth`).

## Reference numbers (earlier work, `diagnostics/method_comparison.json`)
Same 1,701 crops (val tiles map67_1-1, 2-2, 4-3), IoU@0.5: weak (Open Buildings fine-tune) 0.6476,
AdaBN 0.2523, seed 0.1419, FDA 0.0891, histmatch 0.0258.
**Eval labels are Open Buildings GROUND FOOTPRINTS, not roof outlines** — every number is
agreement-with-footprints. The weak model trained on that label type, so it is not a fair
comparison to annotation-free methods.

## Things to know
- Smoke test (30 iters, 64 crops) showed AdaBN lifting IoU@0.5 0.135 -> 0.348, and plain
  self-training then pulling it back down with predicted roof fraction collapsing
  (0.18 -> 0.08). That is the expected failure: the seed under-predicts, so hi=0.8
  yields almost no positive pseudo-labels. Not a result — 64 crops, 30 iters. Watch
  `pseudo_label_roof_frac` and `roof_frac` in the logs. If Arm B collapses, try lower
  `--pl_hi` (e.g. 0.5), chosen from the unlabelled pseudo-label stats, NOT by tuning on the eval set.
- Source set is tiny: only 125 AIRS crops (3 labelled AIRS images exist on the server:
  christchurch_15/48/77). Raw AIRS on the server is incomplete (57 files).
- Noise floor from the earlier notes: pixel IoU 2sd ~0.004; count-based metrics swing ~20x
  more. Single seed here — do not over-read small differences.
- Disk quota is tight (~5 GB headroom of 90 GB at last check). This run uses `data/st_run`
  (1.7 GB: eval crops, source crops, 1,560 unlabelled target tiles) — delete it when done.
- Node is shared (13 users, 80 cores). `run_docker.sh` caps threads; pick a free GPU
  (`nvidia-smi`); GPU 3 got filled by someone else during this session.
- Launch quirk: `run_docker.sh` does `exec ${CMD}`, so set env with `env PYTHONPATH=rooftop python ...`,
  not `PYTHONPATH=... python`.
- The original `/tmp/btp_data/jaipur_weak` dataset was wiped (shared /tmp). Rebuilt eval set =
  `scripts/make_weak_labels.py` on only the 3 val tiles; crop names verified identical to the old set.

## Files (all in `experiments/2026-10-30-annotation-free-selftrain/` on the DGX)
`self_train.py` (mean-teacher self-training + optional `--adabn_batches`), `tile_target.py`,
`prep.sh` (builds `data/st_run`), `run_arms.sh` (the two arms), `logs/`, `ckpt/`.
Same scripts on GitHub branch `feat/annotation-free-self-training` (under `rooftop/`).

## Next steps
1. Read both arms' logs (above). Compare teacher IoU@0.5 and best-threshold IoU vs source-only 0.1419 and AdaBN 0.2523.
2. If self-training collapses: lower `--pl_hi`, or add class-balance guard; consider BLDA (logit-adjusted loss) — it targets exactly this under-predicted-class problem.
3. Then DUDA / BLDA from the paper shortlist; hand-labelled Jaipur test set (roof + panel) so results are not only footprint-agreement.
4. Solar-panel model (BDAPPV) is untouched by this work.
5. Google Maps imagery terms may prohibit training/testing ML on it — unresolved, see `plan/07-risks-licensing-ethics.md`.

## Access for the other laptop
Own SSH key needed: the key used here (`~/.ssh/id_ed25519_gpu`) lives only on the first laptop and was
NOT copied anywhere. On the new laptop: `ssh-keygen -t ed25519`, install the public key with your
password (`ssh-copy-id 23ucs715@172.22.2.151`), then `ssh 23ucs715@172.22.2.151`.
Login banner says all activity is logged.
