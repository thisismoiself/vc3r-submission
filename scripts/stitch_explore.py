#!/usr/bin/env python3
"""Single-window exploration for the office4 full-scene stitch.

Checks, for ONE cached office4 window:
  1. cached DA3 layer matches the adapter:  CD( decode(adapter(da3)) , decode(z_star_gt) )
     small  => cached da3 tokens are the layers the model was trained on.
  2. metric-scale recovery: fit a 1-D scale  a  so that
        world = c2w0 @ (a * pts_norm)
     lands on the office4 GT cloud  (a == norm_factor/3).
  3. report world-frame CD( pred_world , GT )  and  CD( gt_zstar_world , GT ).
"""
import os, sys
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r")
NOVA = REPO / "nova3r_lib"
for p in [str(NOVA / "third_party"), str(NOVA), str(REPO / "da3" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_nova3r import load_model as load_nova3r_model           # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper          # noqa: E402
from nova3r.flow_matching.solver import ODESolver                  # noqa: E402
from omegaconf import OmegaConf                                    # noqa: E402
import trimesh                                                     # noqa: E402
from scipy.spatial import cKDTree                                  # noqa: E402

CKPT = Path("/usr/prakt/s0016/online_var_hungarian_replica_all8_l13_s1p75_s10_loo_office4_b24_drop02_tokdrop02_warm1k_ema999_restart6k_best.pt")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
DATA = REPO / "experiments/overfit_8frames/data/multi_scene_N200_s20/office4"
GT_PLY = REPO / "outputs/replica/gt_pointclouds/office4/office4_gt_2m.ply"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def adapter_cfg(sd):
    tt, hd = sd["target_queries"].shape
    return dict(source_dim=sd["source_proj.weight"].shape[1], hidden_dim=hd,
                target_tokens=tt, target_dim=sd["out_proj.weight"].shape[0],
                depth=sum(1 for k in sd if k.endswith(".query_norm.weight")),
                num_heads=hd // 64)


@torch.no_grad()
def decode(nova, ncfg, tokens, pts_norm, num_q=8192, seed=42):
    torch.manual_seed(seed)
    x_init = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x_init,
                            method=ncfg.get("fm_sampling", "euler"), step_size=step,
                            return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def chamfer(a, b, gt_tree=None, n=8000):
    rng = np.random.default_rng(0)
    if len(a) > n: a = a[rng.choice(len(a), n, replace=False)]
    if gt_tree is None:
        if len(b) > n: b = b[rng.choice(len(b), n, replace=False)]
        ta, tb = torch.from_numpy(a).float(), torch.from_numpy(b).float()
        d = torch.cdist(ta, tb)
        return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)
    # one-directional pred->GT via KDTree (completeness handled elsewhere)
    return float(gt_tree.query(a, k=1)[0].mean())


def fit_scale(pts_norm_cam, c2w0, gt_tree):
    """1-D scale a s.t. world = c2w0 @ (a*pts_norm) lands on GT. Coarse->fine."""
    pts = pts_norm_cam  # [M,3]
    R, t = c2w0[:3, :3], c2w0[:3, 3]
    def cost(a):
        w = (R @ (a * pts).T).T + t
        return gt_tree.query(w, k=1)[0].mean()
    grid = np.linspace(0.05, 3.0, 60)
    costs = [cost(a) for a in grid]
    a0 = grid[int(np.argmin(costs))]
    fine = np.linspace(max(0.01, a0 - 0.1), a0 + 0.1, 40)
    fc = [cost(a) for a in fine]
    return float(fine[int(np.argmin(fc))]), float(min(fc))


def main():
    print("loading NOVA3R ..."); nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV))
    nova.eval(); [p.requires_grad_(False) for p in nova.parameters()]
    OmegaConf.set_struct(ncfg, False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]; acfg = adapter_cfg(sd); print("adapter cfg", acfg)
    adapter = DA3ToNOVA3RAlignment(**acfg, drop=0.0).to(DEV); adapter.load_state_dict(sd); adapter.eval()

    print("loading GT ply ..."); gt = np.asarray(trimesh.load(str(GT_PLY), process=False).vertices, np.float32)
    rng = np.random.default_rng(0); gt_sub = gt[rng.choice(len(gt), 300000, replace=False)]
    gt_tree = cKDTree(gt_sub)

    s = "sample_080"
    d = DATA / s
    da3 = torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True).float()
    pts_norm = torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True).float()
    zgt = torch.load(d / "z_star.pt", map_location="cpu", weights_only=True).float()
    meta = torch.load(d / "meta.pt", map_location="cpu", weights_only=False)
    c2w0 = meta["poses_c2w"][0].numpy().astype(np.float32)
    print(f"\nwindow {s}: da3 {tuple(da3.shape)}  frames {meta['frame_ids'][0]}-{meta['frame_ids'][-1]}")

    with torch.no_grad():
        zpred = adapter(da3.to(DEV)).cpu()
    pred_n = decode(nova, ncfg, zpred, pts_norm)
    gtz_n  = decode(nova, ncfg, zgt,   pts_norm)

    cd_local = chamfer(pred_n, gtz_n)
    print(f"[1] DA3-match  CD(pred, gt_zstar) in normalized frame = {cd_local:.4f}")
    print(f"    (token MSE pred vs gt z_star = {torch.nn.functional.mse_loss(zpred, zgt).item():.4f})")

    a_pn, c_pn = fit_scale(pts_norm[0].numpy(), c2w0, gt_tree)
    print(f"[2] scale fit on INPUT pts_norm : a={a_pn:.3f} (norm_factor={3*a_pn:.3f} m)  resid={c_pn:.4f} m")

    R, t = c2w0[:3, :3], c2w0[:3, 3]
    to_world = lambda pn: (R @ (a_pn * pn).T).T + t
    pred_w, gtz_w = to_world(pred_n), to_world(gtz_n)
    print(f"[3] world CD pred->GT      = {chamfer(pred_w, None, gt_tree):.4f} m")
    print(f"    world CD gt_zstar->GT  = {chamfer(gtz_w, None, gt_tree):.4f} m")
    print(f"    pred_world extent: {pred_w.min(0).round(2)} .. {pred_w.max(0).round(2)}")

    out = REPO / "outputs/replica/stitch_explore"; out.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(pred_w).export(out / f"{s}_pred_world.ply")
    trimesh.PointCloud(gtz_w).export(out / f"{s}_gtzstar_world.ply")
    print(f"\nPLYs -> {out}")


if __name__ == "__main__":
    main()
