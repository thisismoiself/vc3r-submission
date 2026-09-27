import numpy as np, torch, sys, glob
sys.path.insert(0,"/usr/prakt/s0016/vc3r/scripts")
from tsdf_dataset import _pad_center, OBSERVED, FREE, UNKNOWN
from diffcomplete_ddpm import DiffCompleteUNet, Diffusion
from progress_strip import load_ema
BAND=0.24; G=128
rooms=sorted(glob.glob("outputs/tsdf_cache_front3d_coarse/*.npz"))
model,step=load_ema("outputs/consecutive_windows/diffusion_coarse.pt")
diff=Diffusion(T=1000,device="cuda")
out=[]
for p in [rooms[-3],rooms[-17],rooms[-41]]:   # a few held-out-ish rooms
    d=np.load(p); gt=np.abs(d["gt_tsdf"].astype(np.float32)); pt=d["partial_tsdf"].astype(np.float32); mask=d["mask"]; v=float(d["voxel"])
    if max(mask.shape)>G: continue
    gtp=_pad_center(gt,G,BAND); ptp=_pad_center(pt,G,BAND); mp=_pad_center(mask,G,UNKNOWN)
    ptn=torch.from_numpy(ptp)/BAND
    oh=torch.nn.functional.one_hot(torch.from_numpy(mp.astype(np.int64)),3).permute(3,0,1,2).float()
    cond=torch.cat([ptn[None],oh,torch.ones(1,*ptp.shape)],0)[None].cuda()
    mk=torch.from_numpy(mp.astype(np.int64))[None,None].cuda()
    with torch.no_grad():
        pred=diff.ddim_sample(model,cond,mask=mk,x_known=(2*cond[:,:1].abs()-1),steps=100,replace=True)
    udf=((pred.float()+1)*0.5*BAND).clamp(0,BAND)[0,0].cpu().numpy()
    out.append((p.split("/")[-1][:20],udf,gtp,mp,ptp))
    print("done",p.split("/")[-1][:20],flush=True)
np.savez("outputs/coarse_sample.npz", step=step, voxel=0.08, band=BAND,
         **{f"udf{i}":o[1] for i,o in enumerate(out)}, **{f"gt{i}":o[2] for i,o in enumerate(out)},
         **{f"mask{i}":o[3] for i,o in enumerate(out)}, **{f"pt{i}":o[4] for i,o in enumerate(out)})
print("saved coarse_sample.npz step",step,flush=True)
