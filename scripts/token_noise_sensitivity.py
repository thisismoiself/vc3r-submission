#!/usr/bin/env python3
"""Decoder stability to token perturbation (sensitivity analysis).

Take the AE's true tokens z* (cached z_star_online_mean.pt), add isotropic Gaussian
noise at growing scale s, decode each perturbed token set with the SAME query init
(so only the tokens change), and measure how far the decoded point cloud drifts from
the clean decode(z*). This is the decoder's robustness curve: it says whether a given
token error lands in the flat (decoder absorbs it) or steep (geometry breaks) regime.

Noise is expressed as a fraction of the token element std sigma_tok, and also as the
per-element MSE s^2 -- directly comparable to the adapter's val_plain_mse, so we can
mark where the trained adapter actually operates on the curve.

Two references are measured:
  - sampler floor: decode(z*) at different query seeds -> irreducible scatter (s=0 line).
  - adapter op-point: sqrt(val_plain_mse)/sigma_tok, drawn as a vertical marker.

All distances in cm (normalized /3*nf via the window's cached norm_factor).
Runs on a handful of held-out office4 windows; averages over windows x noise seeds.
"""
import os, sys, glob, argparse, csv
from pathlib import Path
import numpy as np, torch
os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for p in [str(NOVA / "third_party"), str(NOVA), str(REPO / "da3" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from demo_nova3r import load_model as load_nova3r_model
from nova3r.models.model_wrapper import BatchModelWrapper
from nova3r.flow_matching.solver import ODESolver
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

DEV = torch.device("cuda")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-room", type=Path,
                    default=REPO / "scripts/data/fc_nf16_span24_100_l13/office4")
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--num-queries", type=int, default=16384)
    ap.add_argument("--noise-seeds", type=int, default=3, help="Token-noise realizations per level.")
    ap.add_argument("--fracs", type=float, nargs="*",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5],
                    help="Noise std as a fraction of the token element std.")
    ap.add_argument("--fm-sampling", default="midpoint")
    ap.add_argument("--fm-step", type=float, default=0.04)
    ap.add_argument("--decode-seed", type=int, default=42)
    ap.add_argument("--val-mse", type=float, default=0.0508,
                    help="Adapter val_plain_mse to mark its operating point (full-recipe=0.0508).")
    ap.add_argument("--out", type=Path, default=REPO / "experiments/overfit_8frames/token_sensitivity.csv")
    return ap.parse_args()


def main():
    args = parse()
    nova, ncfg = load_nova3r_model(str(REPO / "checkpoints/nova3r/scene_ae/checkpoint-last.pth"), str(DEV))
    nova.eval(); [q.requires_grad_(False) for q in nova.parameters()]; OmegaConf.set_struct(ncfg, False)
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    T = torch.linspace(0, 1, int(1 // args.fm_step)).to(DEV)

    @torch.no_grad()
    def decode(z, pts_norm, seed):
        torch.manual_seed(seed)
        x = torch.rand(1, args.num_queries, 3, device=DEV) * 2 - 1
        with torch.amp.autocast("cuda", enabled=False):
            sol = solver.sample(time_grid=T, x_init=x, method=args.fm_sampling, step_size=args.fm_step,
                                return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                                token_mask=None, encoder_data={"tokens": z.to(DEV)}, pointmaps=pts_norm.to(DEV))
        return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().numpy()

    def chamfer(a, b):
        return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())

    win_dirs = sorted(glob.glob(str(args.cache_room / "*/z_star_online_mean.pt")))[:args.n_windows]
    assert win_dirs, f"no windows under {args.cache_room}"

    # accumulate per-frac deviation (cm) and per-frac surf-scatter (cm), plus s=0 sampler floor
    dev = {f: [] for f in args.fracs}
    floor = []
    sigma_toks = []
    for wp in win_dirs:
        d = Path(wp).parent
        z = torch.load(d / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float()
        pts_norm = torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True).float()
        meta = torch.load(d / "meta.pt", map_location="cpu", weights_only=False)
        nf = float(meta["norm_factor"]); to_cm = nf / 3.0 * 100.0
        s_tok = float(z.std()); sigma_toks.append(s_tok)

        clean = decode(z, pts_norm, args.decode_seed)                 # reference decode
        # sampler floor: same z, different query seeds
        for sd in range(1, args.noise_seeds + 1):
            floor.append(chamfer(clean, decode(z, pts_norm, args.decode_seed + 1000 + sd)) * to_cm)
        for f in args.fracs:
            if f == 0.0:
                dev[f].append(0.0); continue
            s = f * s_tok
            accs = []
            for sd in range(args.noise_seeds):
                g = torch.Generator().manual_seed(10_000 + sd)
                zn = z + s * torch.randn(z.shape, generator=g)
                accs.append(chamfer(decode(zn, pts_norm, args.decode_seed), clean) * to_cm)
            dev[f].append(float(np.mean(accs)))
        fmax = max(args.fracs)
        print(f"  {d.name:<22} sigma_tok={s_tok:.3f} nf={nf:.3f}  "
              f"dev@{fmax:g}={dev[fmax][-1]:.2f}cm", flush=True)

    s_tok = float(np.mean(sigma_toks))
    floor_cm = float(np.mean(floor))
    adapter_frac = (args.val_mse ** 0.5) / s_tok
    print(f"\ntoken element std sigma_tok = {s_tok:.4f}")
    print(f"sampler floor (s=0, cross-seed) = {floor_cm:.2f} cm  (irreducible)")
    print(f"adapter op-point: sqrt(val_mse={args.val_mse}) = {args.val_mse**0.5:.3f} "
          f"= {adapter_frac:.2f} x sigma_tok\n")
    print(f"{'frac*sig':>9} {'noise_std':>9} {'MSE(=s^2)':>9} {'dev_cm':>8}   (dev from clean decode)")
    print("-" * 58)
    rows = []
    for f in args.fracs:
        s = f * s_tok
        m = float(np.mean(dev[f]))
        mark = "  <-- ~adapter" if abs(f - adapter_frac) < 0.05 else ""
        print(f"{f:9.2f} {s:9.4f} {s*s:9.4f} {m:8.2f}{mark}")
        rows.append({"frac_sigma": f, "noise_std": s, "mse": s * s, "dev_cm": m})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frac_sigma", "noise_std", "mse", "dev_cm"])
        w.writeheader(); w.writerows(rows)
    print(f"\nsigma_tok={s_tok:.4f}  floor={floor_cm:.2f}cm  adapter_frac={adapter_frac:.2f}  -> {args.out}")


if __name__ == "__main__":
    main()
