#!/usr/bin/env python3
"""Does finer ODE sampling reduce the decoder's point scatter?

Decodes a cached window's z_star (oracle tokens) at several ODE settings and
measures the decoded-point distance to the clean input surface (pts_norm),
all in the normalized frame.
"""
import os, sys, glob
from pathlib import Path
import numpy as np, torch
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for p in [str(NOVA/"third_party"), str(NOVA), str(REPO/"da3"/"src")]:
    if p not in sys.path: sys.path.insert(0, p)
from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

DEV = torch.device("cuda")
nova, ncfg = load_nova3r_model(str(NOVA/"checkpoints"/"scene_ae"/"checkpoint-last.pth"), str(DEV))
nova.eval(); [p.requires_grad_(False) for p in nova.parameters()]; OmegaConf.set_struct(ncfg, False)

win = sorted(glob.glob(str(REPO/"scripts/data/run1_span24_100_nf4_10_l13/office0/*/z_star_online_mean.pt")))[0]
d = Path(win).parent
z = torch.load(d/"z_star_online_mean.pt", map_location="cpu", weights_only=True).float()
pts_norm = torch.load(d/"pts_norm.pt", map_location="cpu", weights_only=True).float()
surf = cKDTree(pts_norm[0].numpy())
print(f"window {d.name}")

@torch.no_grad()
def decode(method, step, n=20000, seed=42):
    torch.manual_seed(seed)
    x = torch.rand(1, n, 3, device=DEV)*2-1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    T = torch.linspace(0,1,int(1//step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x, method=method, step_size=step,
                            return_intermediates=False, images=torch.zeros(1,1,3,1,1,device=DEV),
                            token_mask=None, encoder_data={"tokens": z.to(DEV)}, pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol,list) else sol)[0].cpu().numpy()

for method, step in [("euler",0.04),("euler",0.02),("euler",0.01),("euler",0.005),("midpoint",0.04),("midpoint",0.02)]:
    import time; t=time.time()
    pred = decode(method, step)
    e = surf.query(pred, k=1)[0]
    nsteps = int(1//step)
    print(f"{method:9s} step={step:<5} ({nsteps:3d} steps)  scatter->surf: mean {e.mean()*100:5.2f}  median {np.median(e)*100:5.2f}  <2u {100*(e<0.02).mean():4.1f}%  >5u {100*(e>0.05).mean():4.1f}%   [{time.time()-t:.1f}s]")
