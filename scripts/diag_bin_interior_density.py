#!/usr/bin/env python3
"""DISAMBIGUATE the stuck 5mm interior recall from diag_bin_interior_overfit: is it a DECODE-SAMPLING
floor or a genuine z* bandlimit? Take the OVERFIT decoder head (interiors maximally trained) and decode
the same office4 windows at rising query density, measuring GT->pred recall at tight thresholds.

  rec@5mm CLIMBS with density  => 5mm was a sampling floor; z* HAS the finest interior => detail fully
                                  recoverable by (denser/importance) decode + detail-weighted FT. No encoder touch.
  rec@5mm SATURATES low        => z* genuinely bandlimits the finest interior => encoder-side needed.
"""
import sys, math
from pathlib import Path
import numpy as np, torch, trimesh
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for _p in [str(NOVA / "third_party"), str(NOVA), str(NOVA / "demo")]:
    if _p not in sys.path: sys.path.insert(0, _p)
from demo_nova3r import load_model as load_nova3r_model           # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper         # noqa: E402
from nova3r.flow_matching.solver import ODESolver                 # noqa: E402
from omegaconf import OmegaConf                                   # noqa: E402

DEV = torch.device("cuda")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
OVERFIT_HEAD = REPO / "outputs/consecutive_windows/diag_bin_interior_head.pt"
CACHE = REPO / "scripts/data/fc_nf16_span24_100_l13_complete/office4"
WINS = ["1028_1050_12f_28s", "1040_1089_16f_57s", "1067_1139_15f_81s"]
DENSITIES = [200_000, 800_000, 2_000_000]
THRS = [0.005, 0.01, 0.02]
CHUNK, SEED = 50_000, 42


@torch.no_grad()
def decode_dense(nova, ncfg, z, pn, nq):
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    outs = []
    for ci, s in enumerate(range(0, nq, CHUNK)):
        n = min(CHUNK, nq - s); torch.manual_seed(SEED + ci)
        xi = torch.rand(1, n, 3, device=DEV) * 2 - 1
        with torch.amp.autocast("cuda", enabled=False):
            sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
                                return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                                token_mask=None, encoder_data={"tokens": z.to(DEV)}, pointmaps=pn.to(DEV))
        outs.append((sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()); torch.cuda.empty_cache()
    return np.concatenate(outs, 0)


def main():
    nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
    OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
    for p in nova.parameters(): p.requires_grad_(False)
    base = {k: v.detach().clone() for k, v in nova.pts3d_head.state_dict().items()}
    over = torch.load(OVERFIT_HEAD, map_location="cpu", weights_only=True)

    for which, sd in (("BASE", base), ("OVERFIT", over)):
        nova.pts3d_head.load_state_dict(sd, strict=True)
        print(f"\n=== {which} head ===", flush=True)
        agg = {nq: {t: [] for t in THRS} for nq in DENSITIES}
        for w in WINS:
            wd = CACHE / w
            z = torch.load(wd / "z_star_online_mean.pt", weights_only=True).float()
            pn = torch.load(wd / "pts_norm.pt", weights_only=True).float()
            gt = pn[0].numpy()
            for nq in DENSITIES:
                d_gp = cKDTree(decode_dense(nova, ncfg, z, pn, nq)).query(gt, k=1)[0]
                for t in THRS: agg[nq][t].append((d_gp < t).mean())
        for nq in DENSITIES:
            rec = "  ".join(f"rec@{int(t*1000)}mm={np.mean(agg[nq][t])*100:5.1f}%" for t in THRS)
            print(f"  q={nq:>9,}  {rec}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
