cd ~/btp
./run_docker.sh 5 "env PYTHONPATH=/workspace/data/iou/pylib python -u scripts/refine_labels.py --mode gate --pad_grid 10 --tau_grid 0.3 --fuse_grid 0.5 0.6 0.7 0.8 0.9 --out experiments/2026-11-20-toward-higher-iou/refine_gate_fuse.json" > experiments/2026-11-20-toward-higher-iou/logs/refine_gate_fuse.txt 2>&1
./run_docker.sh 5 "python -u scripts/refine_labels.py --mode pseudo --fuse_grid 0.6 0.7 0.8" > experiments/2026-11-20-toward-higher-iou/logs/pseudo_build.txt 2>&1
touch experiments/2026-11-20-toward-higher-iou/logs/PSEUDO_DONE
