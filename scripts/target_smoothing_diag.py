#!/usr/bin/env python3
"""Is the 30-seed consensus MEAN (the adapter's training target) blurrier than a
single encode (the oracle we benchmark against)?

For several cached windows, decode (midpoint) each of:
  - consensus mean  (z_star_online_mean)         <- training target
  - single test encode (nova._encode test=True)  <- the stitch 'oracle'
  - a few individual stochastic seeds (samples)  <- raw encoder outputs
and measure scatter to the input surface (sharpness proxy). If decode(mean) is
consistently worse, averaging smooths detail -> the target caps achievable sharpness.
Decoder is permutation-invariant (tokens are a cross-attn set), so token ordering
across these does not matter for the comparison.
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

@torch.no_grad()
def decode(tokens, pts_norm, n=20000, seed=42, method="midpoint", step=0.04):
    torch.manual_seed(seed)
    x = torch.rand(1, n, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x, method=method, step_size=step,
                            return_intermediates=False, images=torch.zeros(1,1,3,1,1,device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().numpy()

def scatter(pred, surf_tree):
    d = surf_tree.query(pred, k=1)[0]
    return d.mean()*100, np.median(d)*100, 100*(d<0.02).mean(), 100*(d>0.05).mean()

wins = sorted(glob.glob(str(REPO/"scripts/data/run1_span24_100_nf4_10_l13/office0/*/")))[:5]
agg = {"mean": [], "test": [], "seed": []}
for w in wins:
    d = Path(w)
    pts_norm = torch.load(d/"pts_norm.pt", map_location="cpu", weights_only=True).float()
    mean = torch.load(d/"z_star_online_mean.pt", map_location="cpu", weights_only=True).float()
    samples = torch.load(d/"z_star_online_samples.pt", map_location="cpu", weights_only=True).float()
    if samples.dim() == 4: samples = samples[0]          # (S,768,128)
    surf = cKDTree(pts_norm[0].numpy())
    z_test = nova._encode(pointmaps=pts_norm.to(DEV), test=True)["tokens"].float().cpu()

    m  = scatter(decode(mean,    pts_norm), surf)
    t  = scatter(decode(z_test,  pts_norm), surf)
    ss = [scatter(decode(samples[k:k+1], pts_norm), surf) for k in range(min(4, samples.shape[0]))]
    s  = np.mean(ss, axis=0)
    agg["mean"].append(m); agg["test"].append(t); agg["seed"].append(s)
    print(f"{d.name:22s}  mean(target): {m[0]:5.2f}cm <2u{m[2]:4.0f}% >5u{m[3]:4.0f}%   "
          f"test(oracle): {t[0]:5.2f}cm <2u{t[2]:4.0f}% >5u{t[3]:4.0f}%   "
          f"seed: {s[0]:5.2f}cm <2u{s[2]:4.0f}% >5u{s[3]:4.0f}%")

print("\n=== averages over windows (scatter to input surface) ===")
for k in ["mean", "test", "seed"]:
    a = np.mean(agg[k], axis=0)
    label = {"mean":"consensus MEAN (training target)","test":"single TEST encode (oracle)","seed":"single stochastic SEED"}[k]
    print(f"  {label:34s}  mean {a[0]:5.2f}cm  median {a[1]:5.2f}cm  <2cm {a[2]:4.0f}%  >5cm {a[3]:4.0f}%")

# token-space relationship: how far is mean from test-encode / from samples
print("\n=== token-space (per-token L2, set-matched via min over the other set) ===")
