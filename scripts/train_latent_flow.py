#!/usr/bin/env python3
"""PoC: LATENT FLOW-MATCHING adapter over z*  (a.k.a. "latent diffusion" prior).

Premise (from the diagnostics): the deterministic DA3->z* regressor averages the many z*
consistent with a few posed frames -> a "puffy", low-precision cloud. A conditional
flow-matching model SAMPLES a sharp, self-consistent z* instead of regressing the mean.

Model:   v_theta(z_t, t | DA3 tokens) -> velocity in z* space (768x128).
Path:    straight rectified flow, z0 ~ N(0,I), z1 = z* (encoder mean tokens),
         Hungarian-coupled per token (768<->768) so the field is low-variance and
         doesn't collapse to identity (same lesson as the point-flow corrector).
Inference: z0 ~ N(0,I) -> integrate ODE -> z_pred -> NOVA3R decode (permutation-invariant
         over the 768 tokens, so the generated ORDER is irrelevant).

First experiment: train on office0-3+room0-2, hold out office4. On held-out windows, decode
and report per-window precision/recall/F@2 (normalized) for:
   ORACLE  = decode(z*)                (ceiling)
   DET     = decode(deterministic adapter z_pred)   (the regression baseline)
   FLOW    = decode(FM-sampled z_pred)              (this method)
The hypothesis is validated iff FLOW precision > DET precision (closes toward ORACLE).
"""
import argparse, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

REPO = Path("/usr/prakt/s0016/vc3r"); NOVA = REPO / "nova3r_lib"
for p in [str(NOVA / "third_party"), str(NOVA), str(NOVA / "demo"), str(REPO / "da3" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
from vc3r.alignment import CrossAttentionBlock, DA3ToNOVA3RAlignment  # noqa: E402
from demo_nova3r import load_model as load_nova3r_model      # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper    # noqa: E402
from nova3r.flow_matching.solver import ODESolver            # noqa: E402
from omegaconf import OmegaConf                              # noqa: E402

DEV = torch.device("cuda")
NOVA_CKPT = REPO / "checkpoints" / "nova3r" / "scene_ae" / "checkpoint-last.pth"
CROOT = REPO / "scripts/data/fc_nf16_span24_100_l13_complete"

# ---- Hungarian coupling (lapjv, threaded) ----
try:
    import lap
    def _solve(c): return lap.lapjv(c, extend_cost=True)[1]
except ImportError:
    from scipy.optimize import linear_sum_assignment
    def _solve(c): return linear_sum_assignment(c)[1]
_POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 8))
def batch_match(a, b):
    costs = torch.cdist(a.detach().float(), b.detach().float()).cpu().numpy()
    return list(_POOL.map(_solve, costs))


class LatentFlowAdapter(torch.nn.Module):
    """Conditional velocity field over z*: reuses the adapter's cross-attention blocks,
    but the query stream is the NOISY latent z_t + time (instead of learnable queries)."""
    def __init__(self, source_dim, hidden=512, tokens=768, dim=128, depth=6, heads=8, t_bands=64):
        super().__init__()
        self.t_bands = t_bands
        self.source_proj = torch.nn.Linear(source_dim, hidden)
        self.z_proj = torch.nn.Linear(dim, hidden)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * t_bands, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden))
        self.blocks = torch.nn.ModuleList([CrossAttentionBlock(hidden, heads) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(hidden)
        self.out = torch.nn.Linear(hidden, dim)
        self.out.weight.data.mul_(0.1); self.out.bias.data.zero_()

    def _temb(self, t):
        f = (2.0 ** torch.arange(self.t_bands, device=t.device, dtype=t.dtype)) * math.pi
        a = t[..., None] * f
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, z_t, t, src):
        s = self.source_proj(src)
        h = self.z_proj(z_t) + self.time_mlp(self._temb(t)).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, s)
        return self.out(self.norm(h))


def fm_loss(model, z_star, src, gen):
    B, T, C = z_star.shape
    z0 = torch.randn(B, T, C, device=DEV, generator=gen)
    cols = batch_match(z0, z_star)                       # match noise tokens -> target tokens
    z1 = torch.stack([z_star[i][torch.as_tensor(cols[i], device=DEV, dtype=torch.long)]
                      for i in range(B)])
    t = torch.rand(B, device=DEV, generator=gen)
    zt = (1 - t)[:, None, None] * z0 + t[:, None, None] * z1
    v_star = z1 - z0
    return ((model(zt, t, src) - v_star) ** 2).mean()


@torch.no_grad()
def sample_z(model, src, steps, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(1, 768, 128, device=DEV)
    ts = torch.linspace(0, 1, steps + 1, device=DEV)
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        k1 = model(z, ts[i].expand(1), src)
        zm = z + 0.5 * dt * k1
        z = z + dt * model(zm, (ts[i] + 0.5 * dt).expand(1), src)
    return z


@torch.no_grad()
def decode(nova, ncfg, tokens, pts_norm, nq, seed=0):
    torch.manual_seed(seed)
    xi = torch.rand(1, nq, 3, device=DEV) * 2 - 1
    solver = ODESolver(velocity_model=BatchModelWrapper(model=nova))
    step = ncfg.get("fm_step_size", 0.04); T = torch.linspace(0, 1, int(1 // step)).to(DEV)
    with torch.amp.autocast("cuda", enabled=False):
        sol = solver.sample(time_grid=T, x_init=xi, method="midpoint", step_size=step,
                            return_intermediates=False, images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
                            token_mask=None, encoder_data={"tokens": tokens.to(DEV)},
                            pointmaps=pts_norm.to(DEV))
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def metrics(pred, gt, thr=0.02):
    tp, tg = cKDTree(pred), cKDTree(gt)
    d_pg = tg.query(pred)[0]; d_gp = tp.query(gt)[0]
    prec = float((d_pg < thr).mean()); rec = float((d_gp < thr).mean())
    return {"chamfer": float((d_pg.mean() + d_gp.mean()) / 2), "prec": prec, "rec": rec,
            "F": 2 * prec * rec / (prec + rec + 1e-9)}


def load_windows(rooms, cap):
    out = []
    for room in rooms:
        wd = sorted([d for d in (CROOT / room).iterdir() if d.is_dir()])
        out += [(room, d) for d in (wd if cap is None else wd[:cap])]
    return out


def load_item(d):
    da3 = torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True).float()
    z = torch.load(d / "z_star_online_mean.pt", map_location="cpu", weights_only=True).float()
    pn = torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True).float()
    return da3, z, pn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rooms", nargs="+", default=["office0", "office1", "office2", "office3", "room0", "room1", "room2"])
    p.add_argument("--val-room", default="office4")
    p.add_argument("--max-per-room", type=int, default=None)
    p.add_argument("--val-windows", type=int, default=6)
    p.add_argument("--det-ckpt", type=Path, default=REPO / "outputs/consecutive_windows/complete_bestcombo_pca32_best.pt")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--eval-queries", type=int, default=50000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-out", type=Path, default=REPO / "outputs/consecutive_windows/latent_flow_poc.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    print(f"[load] NOVA3R decoder"); nova, ncfg = load_nova3r_model(str(NOVA_CKPT), str(DEV)); nova.eval()
    OmegaConf.set_struct(ncfg, False); ncfg["fm_sampling"] = "midpoint"
    for p_ in nova.parameters(): p_.requires_grad_(False)

    train_w = load_windows(args.rooms, args.max_per_room)
    val_w = load_windows([args.val_room], None)[:args.val_windows]
    print(f"[data] train={len(train_w)} windows  val({args.val_room})={len(val_w)}", flush=True)

    DA, Z = [], []
    for _, d in train_w:
        da3, z, _ = load_item(d); DA.append(da3); Z.append(z)
    src_dim = DA[0].shape[-1]
    Z = torch.stack([z[0] for z in Z]).to(DEV)                 # [N,768,128]
    print(f"[data] source_dim={src_dim}  z*={tuple(Z.shape)}", flush=True)

    model = LatentFlowAdapter(src_dim, hidden=args.hidden, depth=args.depth).to(DEV)
    print(f"[model] LatentFlowAdapter hidden={args.hidden} depth={args.depth} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)

    # deterministic baseline adapter (for the head-to-head)
    det = None
    if args.det_ckpt.exists():
        sd = torch.load(args.det_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        tt, hd = sd["target_queries"].shape
        det = DA3ToNOVA3RAlignment(source_dim=sd["source_proj.weight"].shape[1], hidden_dim=hd,
                                   target_tokens=tt, target_dim=sd["out_proj.weight"].shape[0],
                                   depth=sum(1 for k in sd if k.endswith(".query_norm.weight")),
                                   num_heads=hd // 64).to(DEV)
        det.load_state_dict(sd); det.eval()
        print(f"[det] loaded deterministic baseline {args.det_ckpt.name}", flush=True)

    # preload val (da3 + z* + pts_norm)
    VAL = [(load_item(d)) for _, d in val_w]

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        agg = {k: {m: [] for m in ["prec", "rec", "F", "chamfer"]} for k in ["ORACLE", "DET", "FLOW"]}
        for da3, z, pn in VAL:
            gt = pn[0].numpy(); src = da3.to(DEV)
            cand = {"ORACLE": z.to(DEV)}
            cand["FLOW"] = sample_z(model, src, args.sample_steps, seed=1234)
            if det is not None:
                cand["DET"] = det(src).float()
            for k, tok in cand.items():
                m = metrics(decode(nova, ncfg, tok, pn, args.eval_queries), gt)
                for mm in agg[k]: agg[k][mm].append(m[mm])
        print(f"\n[eval@{step}]  (held-out {args.val_room}, {len(VAL)} win, {args.eval_queries} q, F@2 normalized)")
        for k in ["ORACLE", "DET", "FLOW"]:
            if not agg[k]["F"]: continue
            print(f"    {k:6s}  F@2={np.mean(agg[k]['F']):.4f}  prec={np.mean(agg[k]['prec']):.3f}  "
                  f"rec={np.mean(agg[k]['rec']):.3f}  chamfer={np.mean(agg[k]['chamfer']):.4f}", flush=True)
        model.train()

    gen = torch.Generator(device=DEV).manual_seed(args.seed)
    N = len(DA); bs = min(args.batch_size, N); t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (bs,), generator=torch.Generator().manual_seed(args.seed + step)).tolist()
        src = torch.cat([DA[i] for i in idx], 0).to(DEV)
        z1 = Z[idx]
        loss = fm_loss(model, z1, src, gen)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            el = time.time() - t0
            print(f"[train] step {step}/{args.steps} fm_loss={loss.item():.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e} t={el:.0f}s ({el/step:.2f}s/it)", flush=True)
        if step % args.eval_every == 0:
            evaluate(step)
    evaluate(args.steps)
    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.ckpt_out)
    print(f"[save] {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
