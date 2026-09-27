#!/usr/bin/env python3
"""Averaged permutation-invariance study over many windows.

Claim 1 (decoder order-invariance): per window, decode z*, then
  - shuffle token ORDER  -> Chamfer to orig (should be ~0),
  - add Gaussian noise at sigma in {0.25,0.5,1.0}x token-std -> graded control,
  - FULL random replacement N(mean,std) -> chance-level control.
Claim 2 (encoder arbitrary order): per window, re-encode K seeds from the SAME
  points, report identity / random-perm / Hungarian token distances, and the
  decode(seed_i) vs decode(seed_j) Chamfer.
Aggregates mean +- std ACROSS WINDOWS. Writes perm_invariance_avg.json.
"""
import contextlib, json, sys
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import scene_adapt as SA

DATA = EXP / "data" / "multi_scene_N200_s20"
ROOMS = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
SAMPLES = ["sample_000", "sample_100"]      # 2 per room -> 16 windows
NOISE_SIGMAS = [0.25, 0.5, 1.0]
K_SEEDS = 6
NQ = 4096
ODE = 0.01
DECSEED = 42


@contextlib.contextmanager
def det_rng(seed):
    orig = np.random.default_rng
    np.random.default_rng = lambda s=None: orig(seed if s is None else s)
    try:
        yield
    finally:
        np.random.default_rng = orig


def chamfer(a, b):
    d = torch.cdist(a, b)
    return float((d.min(1).values.mean() + d.min(0).values.mean()) / 2)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(EXP / "config.yaml")
    print("Loading NOVA3R ...", flush=True)
    nova, ncfg = SA.load_nova3r_model(str(cfg.nova3r_ckpt), str(dev))
    nova.eval()
    for p in nova.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(ncfg, False)

    dec = lambda z, pn: SA.decode_tokens(nova, ncfg, z, pn, dev,
                                         num_queries=NQ, seed=DECSEED, ode_step_size=ODE)[0]

    wins = [(r, s) for r in ROOMS for s in SAMPLES
            if (DATA / r / s / "z_star.pt").exists()]
    print(f"{len(wins)} windows", flush=True)

    A = {"shuffle": [], "ptdelta": [], "random": [],
         **{f"noise{sg}": [] for sg in NOISE_SIGMAS}}
    B = {"identity": [], "randperm": [], "hungarian": [], "decode_seedvar": []}
    rng = np.random.default_rng(0)

    for wi, (r, s) in enumerate(wins):
        root = DATA / r / s
        z = torch.load(root / "z_star.pt", map_location="cpu", weights_only=True).float()
        pn = torch.load(root / "pts_norm.pt", map_location="cpu", weights_only=True).float()
        T = z.shape[1]; zstd = z.std(); zmean = z.mean()

        # ---- Claim 1: decoder ----
        pa = dec(z, pn)
        perm = torch.randperm(T, generator=torch.Generator().manual_seed(wi))
        pshuf = dec(z[:, perm, :].clone(), pn)
        A["shuffle"].append(chamfer(pa, pshuf))
        A["ptdelta"].append(float((pa - pshuf).abs().mean()))
        for sg in NOISE_SIGMAS:
            zn = z + torch.randn(z.shape, generator=torch.Generator().manual_seed(100 + wi)) * (sg * zstd)
            A[f"noise{sg}"].append(chamfer(pa, dec(zn, pn)))
        zr = torch.randn(z.shape, generator=torch.Generator().manual_seed(200 + wi)) * zstd + zmean
        A["random"].append(chamfer(pa, dec(zr, pn)))

        # ---- Claim 2: encoder ----
        Zs = []
        for sd in range(K_SEEDS):
            with det_rng(sd):
                Zs.append(nova._encode(pointmaps=pn.to(dev), test=True)["tokens"].float()[0].cpu().numpy())
        Z = np.stack(Zs)
        ident, rand, hung = [], [], []
        for i in range(K_SEEDS):
            for j in range(i + 1, K_SEEDS):
                a, b = Z[i], Z[j]
                ident.append(np.linalg.norm(a - b, axis=1).mean())
                rand.append(np.linalg.norm(a - b[rng.permutation(T)], axis=1).mean())
                C = np.linalg.norm(a[:, None] - b[None], axis=2)
                ri, ci = linear_sum_assignment(C)
                hung.append(C[ri, ci].mean())
        B["identity"].append(float(np.mean(ident)))
        B["randperm"].append(float(np.mean(rand)))
        B["hungarian"].append(float(np.mean(hung)))
        # geometry consistency: decode every seed, mean pairwise Chamfer over all seed pairs
        decs = [dec(torch.tensor(Z[k]).unsqueeze(0), pn) for k in range(K_SEEDS)]
        pair_ch = [chamfer(decs[i], decs[j]) for i in range(K_SEEDS) for j in range(i + 1, K_SEEDS)]
        B["decode_seedvar"].append(float(np.mean(pair_ch)))
        print(f"[{wi+1}/{len(wins)}] {r}/{s} shuf={A['shuffle'][-1]:.1e} "
              f"rand={A['random'][-1]:.3f} ident={B['identity'][-1]:.2f} "
              f"hung={B['hungarian'][-1]:.2f} dec_seedvar={B['decode_seedvar'][-1]:.1e}", flush=True)

    def ms(x):
        x = np.array(x); return {"mean": float(x.mean()), "std": float(x.std()), "n": len(x)}
    out = {"n_windows": len(wins),
           "decoder": {k: ms(v) for k, v in A.items()},
           "encoder": {k: ms(v) for k, v in B.items()},
           "noise_sigmas": NOISE_SIGMAS, "K_seeds": K_SEEDS}
    (EXP / "perm_invariance_avg.json").write_text(json.dumps(out, indent=2))

    print("\n===== AVERAGED (mean +- std over windows) =====")
    print("DECODER (Chamfer to orig decode):")
    print(f"  shuffle ORDER      : {out['decoder']['shuffle']['mean']:.2e} +- {out['decoder']['shuffle']['std']:.1e}"
          f"   (per-pt |d|={out['decoder']['ptdelta']['mean']:.1e})")
    for sg in NOISE_SIGMAS:
        d = out['decoder'][f'noise{sg}']; print(f"  +noise sig={sg:<4}      : {d['mean']:.3f} +- {d['std']:.3f}")
    d = out['decoder']['random']; print(f"  FULL random        : {d['mean']:.3f} +- {d['std']:.3f}")
    print("ENCODER (token distance):")
    for k in ["identity", "randperm", "hungarian"]:
        print(f"  {k:11s}: {out['encoder'][k]['mean']:.2f} +- {out['encoder'][k]['std']:.2f}")
    print(f"  decode seed-pair Chamfer (mean over pairs): {out['encoder']['decode_seedvar']['mean']:.2e} "
          f"+- {out['encoder']['decode_seedvar']['std']:.1e}")
    print("wrote perm_invariance_avg.json")


if __name__ == "__main__":
    main()
