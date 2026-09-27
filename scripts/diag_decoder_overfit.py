#!/usr/bin/env python3
"""DIAGNOSTIC: is the crisp-geometry ceiling the NOVA3R DECODER WEIGHTS or the LATENT REPRESENTATION?

We take DA3 and the adapter completely OUT of the loop and probe the AE's own oracle path:
    complete geometry  --(frozen encoder)-->  z*  --(decoder)-->  geometry     vs  GT.

If we hard-OVERFIT the decoder (pts3d_head) on a single scene's own (z*, GT) pairs and the oracle
sharpens, the ceiling was decoder weights (fixable by fine-tuning). If it does NOT budge even when
trained directly on that scene, z* physically cannot encode crisper geometry -> representation wall
(needs a different backbone / structured latent), and no adapter or decoder FT can help.

Encoder (aggregator) frozen throughout, so z* is fixed -- exactly the target the adapter aims at.
Loss = the decoder's native flow-matching objective (analytic cosine-path velocity), x1 = pts_norm
(the cached complete surface), conditioned on the cached z*. Reports normalized-space Chamfer of the
oracle decode(z*) -> pts_norm BEFORE vs AFTER, plus a 50k/150k query-density check (rules out that the
oracle is merely undersampled). Saves the fine-tuned head to --head-out for a real stitch re-eval.
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r")
NOVA = REPO / "nova3r_lib"
for _p in [str(NOVA / "third_party"), str(NOVA), str(NOVA / "demo")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from demo_nova3r import load_model as load_nova3r_model            # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper          # noqa: E402
from nova3r.flow_matching.solver import ODESolver                  # noqa: E402

DEV = torch.device("cuda")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--room", default=str(REPO / "scripts/data/fc_nf16_span24_100_l13_complete/office4"))
    p.add_argument("--max-windows", type=int, default=80, help="windows to overfit on")
    p.add_argument("--eval-windows", type=int, default=24, help="subset for before/after oracle Chamfer")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--query-points", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-queries", type=int, default=50000)
    p.add_argument("--eval-queries-hi", type=int, default=150000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--head-out", default=str(REPO / "outputs/consecutive_windows/diag_decoder_ft_head.pt"))
    return p.parse_args()


def fm_to_gt_velocity(surf_pts, n_q):
    """Analytic cosine-path FM sample + target velocity (NOVA3R's own path)."""
    B, K = surf_pts.shape[0], surf_pts.shape[1]
    sel = torch.randint(K, (B, n_q), device=surf_pts.device)
    x1 = torch.gather(surf_pts, 1, sel.unsqueeze(-1).expand(B, n_q, 3))
    x0 = torch.rand(B, n_q, 3, device=surf_pts.device) * 2 - 1
    t = torch.rand(B, 1, 1, device=surf_pts.device)
    a, s = torch.sin(math.pi / 2 * t), torch.cos(math.pi / 2 * t)
    da, ds = (math.pi / 2) * torch.cos(math.pi / 2 * t), -(math.pi / 2) * torch.sin(math.pi / 2 * t)
    xt = s * x0 + a * x1
    v_tgt = ds * x0 + da * x1
    return xt, t.view(B, 1).expand(B, n_q), v_tgt


@torch.no_grad()
def oracle_decode(nova, ncfg, z_star, pts_norm, num_q, seed=0):
    torch.manual_seed(seed)
    x_init = torch.rand(1, num_q, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=x_init, method=ncfg.get("fm_sampling", "euler"),
                            step_size=step, return_intermediates=False,
                            images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": z_star.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def chamfer_norm(pred, gt):
    """Symmetric Chamfer + F@thr in NORMALIZED units (both clouds in [-1,1] pts_norm frame)."""
    tp, tg = cKDTree(pred), cKDTree(gt)
    d_pg, _ = tg.query(pred, k=1)   # pred -> gt (accuracy)
    d_gp, _ = tp.query(gt, k=1)     # gt -> pred (completeness)
    thr = 0.02                      # normalized threshold; relative before/after is what matters
    return {"chamfer": float((d_pg.mean() + d_gp.mean()) / 2),
            "acc": float(d_pg.mean()), "comp": float(d_gp.mean()),
            "f_thr": float(((d_pg < thr).mean() + (d_gp < thr).mean()) / 2)}


def eval_oracle(nova, ncfg, windows, num_q, tag):
    accs = []
    for z, pn in windows:
        pred = oracle_decode(nova, ncfg, z, pn, num_q)
        m = chamfer_norm(pred, pn[0].numpy())
        accs.append(m["chamfer"])
    print(f"    [{tag}] q={num_q:>6}  oracle Chamfer(norm) mean={np.mean(accs):.4f} "
          f"median={np.median(accs):.4f}  (n={len(accs)})", flush=True)
    return float(np.mean(accs))


def main():
    args = parse_args()
    room = Path(args.room)
    win_dirs = sorted([d for d in room.iterdir() if d.is_dir()])[:args.max_windows]
    print(f"[data] {room.name}: {len(win_dirs)} windows", flush=True)

    Z, PN = [], []
    for wd in win_dirs:
        Z.append(torch.load(wd / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float())
        PN.append(torch.load(wd / "pts_norm.pt", map_location="cpu", weights_only=True).float())
    eval_set = list(zip(Z[:args.eval_windows], PN[:args.eval_windows]))

    print("[load] NOVA3R scene_ae", flush=True)
    nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV))
    nova.eval()
    for p in nova.parameters():
        p.requires_grad_(False)
    for p in nova.aggregator.parameters():          # encoder stays frozen (z* fixed)
        p.requires_grad_(False)
    head_params = [p for p in nova.pts3d_head.parameters()]
    for p in head_params:
        p.requires_grad_(True)
    n_head = sum(p.numel() for p in head_params)
    print(f"[model] trainable pts3d_head params: {n_head/1e6:.1f}M", flush=True)

    print("[baseline] oracle before decoder FT:")
    eval_oracle(nova, ncfg, eval_set, args.eval_queries, "before")
    eval_oracle(nova, ncfg, eval_set, args.eval_queries_hi, "before-hi")  # density check

    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    nova.pts3d_head.train()
    rng = torch.Generator().manual_seed(0)
    N = len(win_dirs)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(N, generator=rng)[:args.batch_windows].tolist()
        z = torch.cat([Z[i] for i in idx], 0).to(DEV)          # (B, T, C) frozen encoder tokens
        K = min(p.shape[1] for p in (PN[i] for i in idx))
        surf = torch.cat([PN[i][:, :K] for i in idx], 0).to(DEV)
        xt, tq, v_tgt = fm_to_gt_velocity(surf, args.query_points)
        img0 = torch.zeros(z.shape[0], 1, 3, 1, 1, device=DEV)
        with torch.amp.autocast("cuda", enabled=False):
            v_pred = nova._decode(tokens=z, images=img0, query_points=xt, timestep=tq)["pts3d_xyz"]
            loss = (v_pred - v_tgt).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t0
            print(f"  step {step:5d}/{args.steps} fm_loss={loss.item():.5f} "
                  f"t={el:.0f}s ({el/step:.3f}s/it)", flush=True)

    nova.pts3d_head.eval()
    print("[after] oracle after decoder FT:")
    eval_oracle(nova, ncfg, eval_set, args.eval_queries, "after")

    out = Path(args.head_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in nova.pts3d_head.state_dict().items()}, out)
    print(f"[out] fine-tuned head -> {out}", flush=True)


if __name__ == "__main__":
    main()
