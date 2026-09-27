#!/usr/bin/env python3
"""GT-free (da3pose) adapter operating point for the token-sensitivity figure.

The isotropic-noise curve + sampler floor are adapter-INDEPENDENT (decoder + complete
z*), so they are reused from token_sensitivity_complete.csv. This script recomputes only
the two adapter-specific numbers for the da3pose adapter on held-out office4:
  ADPT_MSE : Hungarian-matched per-element MSE(pred tokens, z*)  (its x-position)
  REAL_CM  : Chamfer(decode(pred), decode(z*)) in cm             (structured token error)
plus a fresh sampler FLOOR (cross-seed decode scatter of z*).
"""
import sys, glob
from pathlib import Path
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
sys.path.insert(0, "/usr/prakt/s0016/vc3r/scripts")
import stitch_office4 as S
from omegaconf import OmegaConf

CACHE = Path("/usr/prakt/s0016/vc3r/scripts/data/fc_nf16_span24_100_l13_complete_da3pose/office4")
CKPT = Path("/usr/prakt/s0016/vc3r/outputs/consecutive_windows/da3pose_nf16_vel03_complete_best.pt")
N, NUMQ, SEED = 8, 16384, 42


def chamfer(a, b):
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def main():
    nova, ncfg = S.load_nova3r_model(str(S.NOVA_CKPT), str(S.DEV))
    nova.eval(); [p.requires_grad_(False) for p in nova.parameters()]
    OmegaConf.set_struct(ncfg, False)
    ncfg["fm_sampling"] = "midpoint"; ncfg["fm_step_size"] = 0.04   # match isotropic curve
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]; acfg = S.adapter_cfg(sd)
    adapter = S.DA3ToNOVA3RAlignment(**acfg, drop=0.0).to(S.DEV); adapter.load_state_dict(sd); adapter.eval()

    wins = sorted(glob.glob(str(CACHE / "*/da3_tokens.pt")))[:N]
    mses, real_cms, floors = [], [], []
    for w in wins:
        d = Path(w).parent
        da3 = torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True).float()
        pts_norm = torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True).float()
        zstar = torch.load(d / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float()
        meta = torch.load(d / "meta.pt", map_location="cpu", weights_only=False)
        to_cm = float(meta["norm_factor"]) / 3.0 * 100.0
        with torch.no_grad():
            zpred = adapter(da3.to(S.DEV), geom_xyz=None).cpu()
        a, b = zpred[0].numpy(), zstar[0].numpy()
        C = np.linalg.norm(a[:, None] - b[None], axis=2)
        ri, ci = linear_sum_assignment(C)
        mses.append(float(((a[ri] - b[ci]) ** 2).mean()))
        dp = S.decode(nova, ncfg, zpred.to(S.DEV), pts_norm.to(S.DEV), NUMQ, SEED)
        dz = S.decode(nova, ncfg, zstar.to(S.DEV), pts_norm.to(S.DEV), NUMQ, SEED)
        dz2 = S.decode(nova, ncfg, zstar.to(S.DEV), pts_norm.to(S.DEV), NUMQ, SEED + 7)
        real_cms.append(chamfer(dp, dz) * to_cm)
        floors.append(chamfer(dz, dz2) * to_cm)
        print(f"  {d.name:<24} MSE={mses[-1]:.4f}  real={real_cms[-1]:.2f}cm  floor={floors[-1]:.2f}cm", flush=True)

    print(f"\nda3pose ADPT_MSE = {np.mean(mses):.4f}")
    print(f"da3pose REAL_CM  = {np.mean(real_cms):.2f} cm")
    print(f"sampler FLOOR    = {np.mean(floors):.2f} cm")


if __name__ == "__main__":
    main()
