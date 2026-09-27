#!/usr/bin/env python3
"""DIAGNOSTIC 2: the AE ORACLE loses CONCAVE/INTERIOR detail (office4 trash-bin interior collapses to
a convex hull) even though the GT/pts_norm HAS the interior. Is that the DECODER weights under-
extracting it, or the ENCODER discarding it from z*?

Encoder (aggregator) frozen => z* fixed (exactly the adapter's target). Hard-OVERFIT the full
pts3d_head on office4's OWN cached (z*, pts_norm) pairs -- pts_norm is the COMPLETE amodal surface,
which (confirmed) contains the interiors. Then measure a CONCAVITY-SENSITIVE metric: GT->pred recall
at TIGHT normalized thresholds. Interior/concave GT points are precisely the ones a hull-biased
decoder leaves far from every prediction, so tight GT->pred recall is the interior probe. Dense,
chunked, midpoint decode before vs after.

  tight recall JUMPS after overfit      => z* HAS the interior info, decoder under-extracted it
                                           => fix = decoder finetune w/ detail-weighted loss (enc frozen)
  tight recall does NOT move (overfit)  => encoder discarded it from z*
                                           => must touch the encoder (encoder-LoRA / retrain)

Exports before/after dense (200k) oracle clouds + the GT (pts_norm) for named windows so the bin
interior can be inspected visually. All clouds in the same world-scaled frame (× norm_factor/3).
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import torch
import trimesh
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
THRS = [0.005, 0.01, 0.02]          # normalized; 0.005/0.01 are the interior-sensitive ones


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--room", default=str(REPO / "scripts/data/fc_nf16_span24_100_l13_complete/office4"))
    p.add_argument("--n-windows", type=int, default=12, help="windows to overfit AND evaluate (memorization)")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-windows", type=int, default=3)
    p.add_argument("--query-points", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-queries", type=int, default=100000)
    p.add_argument("--export-queries", type=int, default=200000)
    p.add_argument("--chunk", type=int, default=50000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--export-windows", type=int, default=3, help="how many windows to dump before/after plys")
    p.add_argument("--out-dir", default=str(REPO / "outputs/replica/bin_interior_probe_office4"))
    p.add_argument("--head-out", default=str(REPO / "outputs/consecutive_windows/diag_bin_interior_head.pt"))
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
def decode_dense(nova, ncfg, z_star, pts_norm, nq, chunk, seed=42):
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04)
    T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    outs = []
    for ci, s in enumerate(range(0, nq, chunk)):
        n = min(chunk, nq - s)
        torch.manual_seed(seed + ci)
        x_init = torch.rand(1, n, 3, device=DEV) * 2 - 1
        with torch.amp.autocast("cuda", enabled=False):
            sol = solver.sample(time_grid=T, x_init=x_init, method=ncfg.get("fm_sampling", "midpoint"),
                                step_size=step, return_intermediates=False,
                                images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                                token_mask=None, encoder_data={"tokens": z_star.to(DEV)},
                                pointmaps=pts_norm.to(DEV))
        outs.append((sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(outs, 0)


def metrics(pred, gt):
    """Symmetric Chamfer + GT->pred recall at tight thresholds (interior-sensitive)."""
    tp, tg = cKDTree(pred), cKDTree(gt)
    d_pg, _ = tg.query(pred, k=1)   # pred -> gt (accuracy)
    d_gp, _ = tp.query(gt, k=1)     # gt  -> pred (completeness / interior recall)
    out = {"chamfer": float((d_pg.mean() + d_gp.mean()) / 2),
           "acc": float(d_pg.mean()), "comp": float(d_gp.mean())}
    for t in THRS:
        out[f"rec{int(t*1000)}"] = float((d_gp < t).mean())   # GT->pred recall = interior probe
    return out


def eval_set(nova, ncfg, windows, nq, chunk, tag):
    agg = {k: [] for k in ["chamfer", "comp"] + [f"rec{int(t*1000)}" for t in THRS]}
    for z, pn in windows:
        m = metrics(decode_dense(nova, ncfg, z, pn, nq, chunk), pn[0].numpy())
        for k in agg:
            agg[k].append(m[k])
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    rec = "  ".join(f"rec@{int(t*1000)}mm={mean[f'rec{int(t*1000)}']*100:5.1f}%" for t in THRS)
    print(f"    [{tag}] q={nq}  chamfer(norm)={mean['chamfer']:.4f} comp={mean['comp']:.4f}  {rec}", flush=True)
    return mean


def main():
    args = parse_args()
    room = Path(args.room)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    win_dirs = sorted([d for d in room.iterdir() if d.is_dir()])[:args.n_windows]
    print(f"[data] {room.name}: overfit+eval on {len(win_dirs)} windows (memorization)", flush=True)

    Z, PN, SCALE, NAME = [], [], [], []
    for wd in win_dirs:
        Z.append(torch.load(wd / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float())
        PN.append(torch.load(wd / "pts_norm.pt", map_location="cpu", weights_only=True).float())
        meta = torch.load(wd / "meta.pt", weights_only=False)
        SCALE.append(float(meta["norm_factor"]) / 3.0)
        NAME.append(f"{meta['room']}_{wd.name}")
    ev = list(zip(Z, PN))

    print("[load] NOVA3R scene_ae", flush=True)
    nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
    from omegaconf import OmegaConf
    OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
    for p in nova.parameters():
        p.requires_grad_(False)
    head_params = [p for p in nova.pts3d_head.parameters()]
    for p in head_params:
        p.requires_grad_(True)
    print(f"[model] trainable pts3d_head params: {sum(p.numel() for p in head_params)/1e6:.1f}M "
          f"(aggregator frozen => z* fixed)", flush=True)

    base_head = {k: v.detach().clone() for k, v in nova.pts3d_head.state_dict().items()}

    def export(tag):
        for i in range(min(args.export_windows, len(win_dirs))):
            cloud = decode_dense(nova, ncfg, Z[i], PN[i], args.export_queries, args.chunk) * SCALE[i]
            trimesh.PointCloud(cloud).export(out_dir / f"{NAME[i]}_oracle_{tag}.ply")
            if tag == "before":                                   # GT once
                trimesh.PointCloud(PN[i][0].numpy() * SCALE[i]).export(out_dir / f"{NAME[i]}_gt_complete.ply")

    print("[baseline] oracle BEFORE overfit:")
    m_before = eval_set(nova, ncfg, ev, args.eval_queries, args.chunk, "before")
    export("before")

    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    nova.pts3d_head.train()
    rng = torch.Generator().manual_seed(0)
    N = len(win_dirs); t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(N, generator=rng)[:args.batch_windows].tolist()
        z = torch.cat([Z[i] for i in idx], 0).to(DEV)
        K = min(PN[i].shape[1] for i in idx)
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
            print(f"  step {step:5d}/{args.steps} fm_loss={loss.item():.5f} t={el:.0f}s ({el/step:.3f}s/it)", flush=True)

    nova.pts3d_head.eval()
    print("[after] oracle AFTER overfit-to-death:")
    m_after = eval_set(nova, ncfg, ev, args.eval_queries, args.chunk, "after")
    export("after")

    print("\n[VERDICT] tight GT->pred recall (interior probe), before -> after overfit:")
    for t in THRS:
        k = f"rec{int(t*1000)}"
        d = (m_after[k] - m_before[k]) * 100
        print(f"    rec@{int(t*1000)}mm: {m_before[k]*100:5.1f}% -> {m_after[k]*100:5.1f}%  ({d:+.1f} pts)", flush=True)
    dcomp = (m_before["comp"] - m_after["comp"]) / m_before["comp"] * 100
    print(f"    comp(norm): {m_before['comp']:.4f} -> {m_after['comp']:.4f}  ({dcomp:+.1f}%)", flush=True)
    print("    -> big tight-recall jump = z* HAS interiors, decoder-weights problem (finetune wins).")
    print("    -> no movement = encoder discards interiors from z* (must touch encoder).", flush=True)

    out = Path(args.head_out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in nova.pts3d_head.state_dict().items()}, out)
    print(f"[out] overfit head -> {out}\n[out] clouds -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
