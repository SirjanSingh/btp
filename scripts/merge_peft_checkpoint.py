#!/usr/bin/env python
"""
merge_peft_checkpoint.py - CrossEarth's mmengine training checkpoints hold ONLY the trainable weights
(Reins tokens + decoder head); the frozen DINOv2 backbone is absent. mmseg's init_model() does not
re-initialise the backbone from init_cfg, so scoring such a checkpoint directly runs a RANDOM backbone
(we got a meaningless test IoU 0.278, area 3.5, before merging). Overlay the trained keys onto the full
released checkpoint the run was started from, then score the merged file.

  python scripts/merge_peft_checkpoint.py --base CrossEarth/checkpoints/WHU-Building-A2S.pth \
      --trained work_dirs/r1/iter_6000.pth --out work_dirs/r1/merged_r1.pth
"""
import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("--base", required=True, help="full checkpoint the fine-tune started from (load_from)")
p.add_argument("--trained", required=True, help="mmengine checkpoint written by tools/train.py")
p.add_argument("--out", required=True)
a = p.parse_args()

base = torch.load(a.base, map_location="cpu")
tr = torch.load(a.trained, map_location="cpu")["state_dict"]
unknown = [k for k in tr if k not in base["state_dict"]]
assert not unknown, f"{len(unknown)} trained keys not in base, e.g. {unknown[:3]}"
changed = sum(int(not torch.equal(base["state_dict"][k].float(), tr[k].float())) for k in tr)
base["state_dict"].update(tr)
torch.save(base, a.out)
print(f"merged {len(tr)} trained tensors ({changed} changed) into {a.base} -> {a.out}")
