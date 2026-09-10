#!/usr/bin/env python
"""Evaluate one solar checkpoint on both BDAPPV domains at several thresholds.

The source-only arm of the google->ign benchmark. Jaipur has no labels, so this
is the only place in the project where a target-domain IoU can be measured at
all; the drop reported here is what every later adaptation method must close.

Thresholds are swept because a domain shift usually moves the operating point as
well as the accuracy -- reporting a single fixed threshold conflates "the model
is worse" with "the model is miscalibrated", and those need different fixes.
"""
import argparse, glob, json, os, cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
THRS=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

class DS(Dataset):
    def __init__(s,root): s.i=os.path.join(root,"images"); s.m=os.path.join(root,"masks"); s.n=sorted(os.listdir(s.i))
    def __len__(s): return len(s.n)
    def __getitem__(s,k):
        n=s.n[k]
        im=cv2.cvtColor(cv2.imread(os.path.join(s.i,n)),cv2.COLOR_BGR2RGB)
        mk=cv2.imread(os.path.join(s.m,n),cv2.IMREAD_GRAYSCALE)
        if im.shape[:2]!=(512,512): im=cv2.resize(im,(512,512))
        if mk.shape[:2]!=(512,512): mk=cv2.resize(mk,(512,512),interpolation=cv2.INTER_NEAREST)
        x=(im.astype(np.float32)/255.-MEAN)/STD
        return torch.from_numpy(x.transpose(2,0,1)), torch.from_numpy((mk>127).astype(np.float32))

@torch.no_grad()
def ev(m,root,dev):
    I={t:0 for t in THRS}; U=dict(I); TP=dict(I); FP=dict(I); FN=dict(I)
    for x,y in DataLoader(DS(root),batch_size=16,num_workers=3):
        x,y=x.to(dev),y.to(dev); p=torch.sigmoid(m(x))[:,0]
        for t in THRS:
            b=(p>t).float()
            I[t]+=(b*y).sum().item(); U[t]+=((b+y)>0).float().sum().item()
            TP[t]+=(b*y).sum().item(); FP[t]+=(b*(1-y)).sum().item(); FN[t]+=((1-b)*y).sum().item()
    out={}
    for t in THRS:
        pr=TP[t]/max(TP[t]+FP[t],1); rc=TP[t]/max(TP[t]+FN[t],1)
        out[f"{t:.1f}"]=dict(iou=round(I[t]/max(U[t],1),4),precision=round(pr,4),recall=round(rc,4))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--ckpt",required=True)
    p.add_argument("--out",default="diagnostics/s1_crossdomain.json"); a=p.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=3,classes=1)
    m.load_state_dict(torch.load(a.ckpt,map_location=dev)["model_state"]); m=m.to(dev).eval()
    res={}
    for name,root in [("google_val (source)","data/bdappv_split/google_val"),
                      ("ign_val (TARGET)","data/bdappv_split/ign_val")]:
        res[name]=ev(m,root,dev)
        b=max(res[name].items(),key=lambda kv:kv[1]["iou"])
        print(f"{name:22s} best IoU {b[1]['iou']:.4f} @thr {b[0]}  P {b[1]['precision']:.3f} R {b[1]['recall']:.3f}  | @0.5 {res[name]['0.5']['iou']:.4f}")
    os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
    json.dump({"checkpoint":a.ckpt,"results":res},open(a.out,"w"),indent=2)
    print("[ok]",a.out)
