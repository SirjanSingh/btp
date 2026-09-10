"""Cross-evaluate both label-confidence models on BOTH val label sets.

The R3 comparison scored a conf-0.0-trained model against conf-0.75 val labels.
A building that exists only in the conf-0.0 set is absent from those labels, so
correctly predicting it counts as a FALSE POSITIVE. That alone could produce the
observed precision drop without any real noise. Scoring both models on both label
sets separates "the extra labels are noise" from "the extra labels are real
buildings the val set refuses to credit".
"""
import glob, json, os, cv2, numpy as np, torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)

class DS(Dataset):
    def __init__(s, img_dir, msk_dir):
        s.i, s.m = img_dir, msk_dir
        s.n = sorted(os.listdir(img_dir))
    def __len__(s): return len(s.n)
    def __getitem__(s, k):
        n=s.n[k]
        im=cv2.cvtColor(cv2.imread(os.path.join(s.i,n)),cv2.COLOR_BGR2RGB)
        mk=cv2.imread(os.path.join(s.m,n),cv2.IMREAD_GRAYSCALE)
        x=(im.astype(np.float32)/255.-MEAN)/STD
        return torch.from_numpy(x.transpose(2,0,1)), torch.from_numpy((mk>127).astype(np.float32))

@torch.no_grad()
def ev(model, ds, dev, thr=0.5):
    inter=union=tp=fp=fn=0
    for x,y in DataLoader(ds,batch_size=16,num_workers=3):
        x,y=x.to(dev),y.to(dev)
        b=(torch.sigmoid(model(x))[:,0]>thr).float()
        inter+=(b*y).sum().item(); union+=((b+y)>0).float().sum().item()
        tp+=(b*y).sum().item(); fp+=(b*(1-y)).sum().item(); fn+=((1-b)*y).sum().item()
    p=tp/max(tp+fp,1); r=tp/max(tp+fn,1)
    return dict(iou=round(inter/max(union,1),4), precision=round(p,4), recall=round(r,4))

dev="cuda"
ck={"conf0.75": sorted(glob.glob("experiments/2026-09-09-does-the-airs-seed-help/checkpoints/*/best.pth"))[-1],
    "conf0.00": sorted(glob.glob("experiments/2026-09-10-label-quantity-vs-quality/checkpoints/*/best.pth"))[-1]}
vals={"val@0.75":"data/jaipur_weak/val/masks", "val@0.00":"data/jaipur_weak_c0/val/masks"}
img="data/jaipur_weak/val/images"
out={}
for mn,cp in ck.items():
    m=smp.Unet("resnet34",encoder_weights=None,in_channels=3,classes=1)
    m.load_state_dict(torch.load(cp,map_location=dev)["model_state"]); m=m.to(dev).eval()
    for vn,vm in vals.items():
        out[f"{mn} on {vn}"]=ev(m, DS(img,vm), dev)
        print(f"{mn} on {vn}: {out[f'{mn} on {vn}']}", flush=True)
json.dump(out, open("diagnostics/r3_cross_eval.json","w"), indent=2)
