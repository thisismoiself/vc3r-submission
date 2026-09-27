#!/usr/bin/env python3
"""
Per-scene test-time adaptation: fine-tune the adapter on a single scene
using only FM loss (no z_star needed at training time).

Improvements over v1:
  - t-sampling bias: t ~ U[t_min, 1] to focus on informative timesteps
  - More FM query points (1024 default) for denser geometric supervision
  - Short ODE rollout + CD loss to directly penalise collapse-to-mean
  - Lower ODE step size at eval (0.01) for sharper decoded geometry
  - Velocity scaling at eval to correct for undershooting spread
  - Longer training (20k steps), lower eta_min

Usage:
  python scene_adapt.py --rooms room0 room1 office0 room2
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT   = Path(__file__).resolve().parents[2]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.flow_matching.path import AffineProbPath             # noqa: E402
from nova3r.flow_matching.path.scheduler import CosineScheduler  # noqa: E402
from nova3r.inference import amp_dtype_mapping                   # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402

import wandb  # noqa: E402

_fm_path = AffineProbPath(scheduler=CosineScheduler())


# ── IP-Adapter parallel cross-attention branch ───────────────────────────────

class IPCrossAttention(torch.nn.Module):
    """Parallel cross-attention branch — paper-faithful IP-Adapter design.

    The Q is shared from the frozen branch (same to_q, same normalized input).
    Only K' and V' are new trainable projections for the adapter tokens.
    Hooked on the inner Attention module so inp[0] is already the normalized,
    time-modulated query x1 that the frozen branch used.
    """
    def __init__(self, query_dim: int = 128, ip_dim: int = 128, num_heads: int = 8):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = query_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        # No to_q — Q reuses the frozen branch's to_q (passed at call time)
        self.to_k   = torch.nn.Linear(ip_dim,    query_dim, bias=False)
        self.to_v   = torch.nn.Linear(ip_dim,    query_dim, bias=False)
        self.to_out = torch.nn.Linear(query_dim, query_dim, bias=False)

    def forward(self, q: torch.Tensor, ip_tokens: torch.Tensor) -> torch.Tensor:
        # q: frozen to_q(x1) already computed — detached so no grad through frozen weights
        B, N, _ = q.shape
        def split(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q_h = split(q)
        k   = split(self.to_k(ip_tokens))
        v   = split(self.to_v(ip_tokens))
        attn = torch.softmax(q_h @ k.transpose(-1, -2) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


# ── PLY writer (binary little-endian) ────────────────────────────────────────

def write_ply(path: Path, pts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    N = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(pts.astype(np.float32).tobytes())


# ── chamfer ───────────────────────────────────────────────────────────────────

def chamfer(x: torch.Tensor, y: torch.Tensor) -> float:
    """Non-differentiable bidirectional CD for eval (returns Python float)."""
    x, y = x.float(), y.float()
    chunk = 512
    d_fwd = [( x[i:i+chunk].unsqueeze(1) - y.unsqueeze(0) ).pow(2).sum(-1).min(1).values
             for i in range(0, x.shape[0], chunk)]
    d_bwd = [( y[j:j+chunk].unsqueeze(1) - x.unsqueeze(0) ).pow(2).sum(-1).min(1).values
             for j in range(0, y.shape[0], chunk)]
    return float((torch.cat(d_fwd).mean() + torch.cat(d_bwd).mean()) / 2)


def chamfer_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Differentiable bidirectional CD for training (gradients flow through x)."""
    y = y.detach().float()
    x = x.float()
    chunk = 256
    d_fwd = [( x[i:i+chunk].unsqueeze(1) - y.unsqueeze(0) ).pow(2).sum(-1).min(1).values
             for i in range(0, x.shape[0], chunk)]
    d_bwd = [( y[j:j+chunk].unsqueeze(1) - x.unsqueeze(0) ).pow(2).sum(-1).min(1).values
             for j in range(0, y.shape[0], chunk)]
    return (torch.cat(d_fwd).mean() + torch.cat(d_bwd).mean()) / 2


def hungarian_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant MSE between two token sets.

    For each sample, optimally match predicted tokens to target tokens with the
    Hungarian algorithm (on a detached L2 cost — no gradient through the argmin),
    reorder the targets accordingly, then plain MSE. This is the correct loss for
    the DA3->NOVA3R adapter because the decoder is permutation-invariant over the
    tokens and the z_star slot index is arbitrary sampling order (see
    same_window_variance.py / latent_analysis.py).

    pred, target: (B, T, D).
    """
    B, T, D = pred.shape
    pd = pred.detach()
    col = torch.empty(B, T, dtype=torch.long, device=pred.device)
    for b in range(B):
        C = torch.cdist(pd[b], target[b])           # (T, T) L2 cost
        _, ci = linear_sum_assignment(C.cpu().numpy())
        col[b] = torch.as_tensor(ci, device=pred.device)
    tgt = torch.gather(target, 1, col.unsqueeze(-1).expand(-1, -1, D))
    return F.mse_loss(pred, tgt)


# ── velocity-scaled wrapper for ODE eval ─────────────────────────────────────

class _ScaledWrapper(torch.nn.Module):
    """Wraps BatchModelWrapper to multiply output velocity by a scalar."""
    def __init__(self, base: BatchModelWrapper, scale: float):
        super().__init__()
        self._base  = base
        self._scale = scale

    @torch.no_grad()
    def forward(self, x, t, **kw):
        return self._base(x, t, **kw) * self._scale


# ── NOVA3R decode ─────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor,
                  pts_norm: torch.Tensor, device: torch.device,
                  num_queries: int = 4096, seed: int = 42,
                  ode_step_size: float = 0.01,
                  velocity_scale: float = 1.0) -> torch.Tensor:
    torch.manual_seed(seed)
    B = tokens.shape[0]
    encoder_data = {"tokens": tokens.to(device)}
    images = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init = torch.rand(B, num_queries, 3, device=device) * 2 - 1
    base_wrapper = BatchModelWrapper(model=nova_model)
    vel_model    = _ScaledWrapper(base_wrapper, velocity_scale) if velocity_scale != 1.0 else base_wrapper
    solver   = ODESolver(velocity_model=vel_model)
    method   = nova_cfg.get("fm_sampling", "euler")
    amp_dt   = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid   = torch.linspace(0, 1, int(1 / ode_step_size) + 1).to(device)
    with torch.cuda.amp.autocast(enabled=(device.type != "cpu"), dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=ode_step_size, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    pts3d = sol[-1] if isinstance(sol, list) else sol
    return pts3d.cpu()


# ── load scene samples ─────────────────────────────────────────────────────────

def load_scene(room: str, n_samples: int, data_root: Path, device: torch.device):
    room_dir = data_root / room
    dirs, da3_list, zstar_list, pnorm_list = [], [], [], []
    for si in range(n_samples):
        d = room_dir / f"sample_{si:03d}"
        if not d.exists():
            break
        dirs.append(d)
        da3_list.append(  torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True))
        zstar_list.append(torch.load(d / "z_star.pt",     map_location="cpu", weights_only=True))
        pnorm_list.append(torch.load(d / "pts_norm.pt",   map_location="cpu", weights_only=True))

    da3   = torch.cat(da3_list,   dim=0)
    zstar = torch.cat(zstar_list, dim=0)
    pnorm = torch.cat(pnorm_list, dim=0)
    print(f"  [{room}] loaded {len(dirs)} samples — da3={tuple(da3.shape)}  pnorm={tuple(pnorm.shape)}")
    return dirs, da3, zstar, pnorm


# ── per-scene adaptation ───────────────────────────────────────────────────────

def adapt_scene(room: str, da3: torch.Tensor, zstar: torch.Tensor,
                pnorm: torch.Tensor, nova_model, nova_cfg,
                cfg, args, device: torch.device, wandb_run,
                ply_dir: Path | None = None) -> dict:

    N = da3.shape[0]
    if getattr(args, "overfit", False):
        # Memorization test: val == train (eval reconstruction on training data)
        tr_da3, tr_zstar, tr_pnorm = da3, zstar, pnorm
        val_da3, val_pnorm         = da3, pnorm
        n_train, n_val = N, N
        print(f"\n[{room}]  OVERFIT MODE  train=val={N}")
    else:
        n_val   = max(1, int(N * args.val_frac))
        n_train = N - n_val
        tr_da3, tr_zstar, tr_pnorm = da3[:n_train], zstar[:n_train], pnorm[:n_train]
        val_da3, val_pnorm         = da3[n_train:],                   pnorm[n_train:]
        print(f"\n[{room}]  train={n_train}  val={n_val}")

    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
        drop          = 0.0,
        use_xyz       = False,
    ).to(device)


    # ── IP-Adapter branches: one per self_point2virtual_blocks layer ──────────
    n_ip_layers = len(nova_model.pts3d_head.self_point2virtual_blocks)
    ip_attns = torch.nn.ModuleList([
        IPCrossAttention(query_dim=128, ip_dim=128, num_heads=8)
        for _ in range(n_ip_layers)
    ]).to(device)
    n_ip = sum(p.numel() for p in ip_attns.parameters())
    print(f"  [{room}]  IP branches: {n_ip_layers} layers × {n_ip//n_ip_layers} params = {n_ip} total")

    # mutable ref so hooks can read the current batch's adapter tokens
    ip_state = [None]

    # Hook on block.cross_attn (the inner Attention module) so inp[0] = x1
    # (the normalized, time-modulated query), matching the paper's shared-Q design.
    hooks = []
    for i, block in enumerate(nova_model.pts3d_head.self_point2virtual_blocks):
        def make_hook(ip_attn, frozen_to_q):
            def hook(module, inp, out):
                x1 = inp[0]                             # normalized+modulated query
                q  = frozen_to_q(x1).detach()           # shared frozen Q, no grad
                return out + ip_attn(q, ip_state[0])
            return hook
        hooks.append(block.cross_attn.register_forward_hook(
            make_hook(ip_attns[i], block.cross_attn.to_q)
        ))

    optimizer = AdamW([
        {"params": adapter.parameters(), "lr": args.lr,        "weight_decay": 1e-4},
        {"params": ip_attns.parameters(),"lr": args.decoder_lr,"weight_decay": 0.0},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 1e-3)

    amp_dt = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)

    tr_da3_dev   = tr_da3.to(device)
    tr_pnorm_dev = tr_pnorm.to(device)
    tr_zstar_dev = tr_zstar.to(device)
    tr_N         = tr_da3_dev.shape[0]

    # (hooks are registered above; they are removed at end of adapt_scene)

    cd_rollout_configured = args.cd_rollout_steps > 0 and args.cd_rollout_weight > 0.0

    history    = []
    best_cd    = float("inf")
    best_state = None
    t0 = time.time()

    for step in range(1, args.steps + 1):
        adapter.train()

        idx_b   = torch.randperm(tr_N, device=device)[:args.batch_size]
        da3_b   = tr_da3_dev[idx_b]
        pnorm_b = tr_pnorm_dev[idx_b]
        zstar_b = tr_zstar_dev[idx_b]
        B       = da3_b.shape[0]

        adapter_out = adapter(da3_b)   # (B, 768, 128)
        ip_state[0] = adapter_out      # IP branch gets adapter output directly
        pred        = adapter_out      # frozen branch also gets adapter output directly

        # ── permutation-invariant z_star MSE (Hungarian) ──────────────────────
        hung_loss = torch.zeros(1, device=device)
        if args.hungarian_weight > 0.0:
            hung_loss = hungarian_mse(adapter_out, zstar_b.float())

        # ── FM loss (t biased toward informative regime) ───────────────────────
        fm_loss = torch.zeros(1, device=device)
        if args.fm_weight > 0.0:
            # token noise on FM path only
            pred_fm = pred + torch.randn_like(pred) * args.token_noise_sigma
            N_full = pnorm_b.shape[1]
            N_q    = min(args.fm_num_queries, N_full)
            idx_q  = torch.stack([torch.randperm(N_full, device=device)[:N_q] for _ in range(B)])
            x_1    = pnorm_b.float().gather(1, idx_q.unsqueeze(-1).expand(-1, -1, 3))
            x_0    = torch.randn(B, N_q, 3, device=device, dtype=torch.float32)
            t_s    = args.t_min + torch.rand(B, device=device) * (1.0 - args.t_min)
            ps     = _fm_path.sample(x_0=x_0, x_1=x_1, t=t_s)
            x_t, u_t = ps.x_t, ps.dx_t
            timestep = t_s.unsqueeze(1).expand(B, N_q)

            images_dummy = torch.zeros(B, 1, 3, 1, 1, device=device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dt):
                decode_out = nova_model._decode(
                    tokens=pred_fm.float(), images=images_dummy,
                    query_points=x_t, timestep=timestep,
                )
            v_pred  = decode_out["pts3d_xyz"].float()
            fm_loss = F.mse_loss(v_pred, u_t)

        # ── short ODE rollout + CD loss (only after warmup) ───────────────────
        cd_loss = torch.zeros(1, device=device)
        use_cd_rollout = cd_rollout_configured and step >= args.cd_rollout_start
        if use_cd_rollout:
            B_cd  = min(args.cd_rollout_batch, B)
            N_cd  = args.cd_rollout_queries
            dt_r  = 1.0 / args.cd_rollout_steps
            pred_cd = pred[:B_cd]                          # (B_cd, 768, 128)
            img_cd  = torch.zeros(B_cd, 1, 3, 1, 1, device=device)
            x_curr  = torch.randn(B_cd, N_cd, 3, device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dt):
                for ri in range(args.cd_rollout_steps):
                    t_r   = torch.full((B_cd, N_cd), ri * dt_r, device=device)
                    v_r   = nova_model._decode(
                        tokens=pred_cd.float(), images=img_cd,
                        query_points=x_curr.detach(), timestep=t_r,
                    )["pts3d_xyz"].float()
                    x_curr = x_curr.detach() + dt_r * v_r  # gradient through v_r only

            gt_cd = pnorm_b[:B_cd].float()
            cd_vals = [chamfer_diff(x_curr[b], gt_cd[b]) for b in range(B_cd)]
            cd_loss  = torch.stack(cd_vals).mean()

        total_loss = (args.fm_weight * fm_loss
                      + args.cd_rollout_weight * cd_loss
                      + args.hungarian_weight * hung_loss)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            print(f"  [{room}] step {step:5d}/{args.steps}"
                  f"  fm={fm_loss.item():.4f}  hung={hung_loss.item():.4f}"
                  f"  cd_roll={cd_loss.item():.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.2e}  t={time.time()-t0:.1f}s")
            wandb_run.log({
                f"{room}/fm_loss":    fm_loss.item(),
                f"{room}/hungarian":  hung_loss.item(),
                f"{room}/cd_rollout": cd_loss.item(),
                f"{room}/lr":         scheduler.get_last_lr()[0],
            }, step=step)

        # ── periodic val eval ──────────────────────────────────────────────────
        if step % args.eval_every == 0 or step == args.steps:
            adapter.eval()
            cd_list = []
            with torch.no_grad():
                for vi in range(len(val_da3)):
                    da3_v  = val_da3[vi:vi+1].to(device)
                    pnrm_v = val_pnorm[vi:vi+1].to(device)
                    adapter_out_v = adapter(da3_v)
                    ip_state[0]   = adapter_out_v
                    tok_v         = adapter_out_v
                    pts    = decode_tokens(
                        nova_model, nova_cfg, tok_v, pnrm_v, device,
                        num_queries=args.num_queries,
                        seed=42 + vi,
                        ode_step_size=args.ode_step_size,
                        velocity_scale=args.velocity_scale,
                    )[0]
                    cd_list.append(chamfer(pts, val_pnorm[vi]))

            mean_cd = float(np.mean(cd_list))
            history.append({"step": step, "val_cd": mean_cd})
            if mean_cd < best_cd:
                best_cd    = mean_cd
                best_state = {
                    "adapter":   copy.deepcopy(adapter.state_dict()),
                    "ip_attns":  copy.deepcopy(ip_attns.state_dict()),
                }
                print(f"  [{room}] *** val_CD={mean_cd:.4f}  (n={len(cd_list)}) ***  [new best]")
            else:
                print(f"  [{room}] *** val_CD={mean_cd:.4f}  (n={len(cd_list)}) ***")
            wandb_run.log({f"{room}/val_cd": mean_cd}, step=step)

    # ── export PLYs using best checkpoint ─────────────────────────────────────
    if ply_dir is not None and best_state is not None:
        adapter.load_state_dict(best_state["adapter"])
        ip_attns.load_state_dict(best_state["ip_attns"])
        adapter.eval()
        ip_attns.eval()
        out_dir   = ply_dir / room
        out_dir.mkdir(parents=True, exist_ok=True)
        n_export  = min(getattr(args, "export_n", 5), len(val_da3))
        print(f"  [{room}] exporting {n_export} val PLYs → {out_dir}")
        with torch.no_grad():
            for vi in range(n_export):
                da3_v  = val_da3[vi:vi+1].to(device)
                pnrm_v = val_pnorm[vi:vi+1].to(device)
                adp_v  = adapter(da3_v)
                ip_state[0] = adp_v
                tok_v  = adp_v
                pts    = decode_tokens(
                    nova_model, nova_cfg, tok_v, pnrm_v, device,
                    num_queries=args.num_queries,
                    seed=42 + vi,
                    ode_step_size=args.ode_step_size,
                    velocity_scale=args.velocity_scale,
                )[0].numpy()
                gt_pts = val_pnorm[vi].numpy()
                write_ply(out_dir / f"val_{vi:03d}_pred.ply", pts)
                write_ply(out_dir / f"val_{vi:03d}_gt.ply",   gt_pts)
        print(f"  [{room}] PLY export done (best_CD={best_cd:.4f})")

        # ── save best checkpoint (adapter + decoder KV) ────────────────────────
        if getattr(args, "save_ckpt", True):
            ckpt_path = out_dir / "best_adapter.pt"
            torch.save(best_state, ckpt_path)
            print(f"  [{room}] saved checkpoint → {ckpt_path}")

        # ── token averaging eval ───────────────────────────────────────────────
        K = getattr(args, "token_avg_k", 0)
        if K > 0:
            K = min(K, tr_N)
            print(f"\n  [{room}] Token averaging: K={K} training windows ...")
            adapter.eval()
            with torch.no_grad():
                # Average adapter outputs over K training windows
                residuals = [adapter(tr_da3_dev[j:j+1]) for j in range(K)]
                res_avg   = torch.cat(residuals, dim=0).mean(dim=0, keepdim=True)
                ip_state[0] = res_avg
                tok_avg   = res_avg

                pts_avg = decode_tokens(
                    nova_model, nova_cfg, tok_avg,
                    tr_pnorm[:1].to(device), device,
                    num_queries=args.num_queries, seed=42,
                    ode_step_size=args.ode_step_size, velocity_scale=args.velocity_scale,
                )[0]  # (N_q, 3)

                # Per-val-sample: token_avg CD vs single-window CD
                cd_avg_list    = []
                cd_single_list = []
                for vi in range(len(val_da3)):
                    gt_vi = val_pnorm[vi]
                    cd_avg_list.append(chamfer(pts_avg, gt_vi))
                    adp_vi = adapter(val_da3[vi:vi+1].to(device))
                    ip_state[0] = adp_vi
                    tok_vi = adp_vi
                    pts_vi = decode_tokens(
                        nova_model, nova_cfg, tok_vi,
                        val_pnorm[vi:vi+1].to(device), device,
                        num_queries=args.num_queries, seed=42 + vi,
                        ode_step_size=args.ode_step_size, velocity_scale=args.velocity_scale,
                    )[0]
                    cd_single_list.append(chamfer(pts_vi, gt_vi))

                mean_cd_avg    = float(np.mean(cd_avg_list))
                mean_cd_single = float(np.mean(cd_single_list))

                # Scene-level: token_avg vs merged val GT
                gt_val_merged  = val_pnorm.view(-1, 3)
                cd_avg_global  = chamfer(pts_avg, gt_val_merged)

                print(f"  [{room}] token_avg    CD (per-val, K={K}) = {mean_cd_avg:.4f}")
                print(f"  [{room}] single_window CD (per-val)       = {mean_cd_single:.4f}")
                print(f"  [{room}] token_avg    CD (vs merged GT)   = {cd_avg_global:.4f}")
                print(f"  [{room}] improvement per-val: {(mean_cd_single - mean_cd_avg)/mean_cd_single*100:.1f}%")

                wandb_run.log({
                    f"{room}/token_avg_cd_perval":  mean_cd_avg,
                    f"{room}/single_window_cd":     mean_cd_single,
                    f"{room}/token_avg_cd_global":  cd_avg_global,
                })

                write_ply(out_dir / f"token_avg_K{K}.ply",   pts_avg.numpy())
                write_ply(out_dir / "gt_val_merged.ply",      gt_val_merged.numpy())
                print(f"  [{room}] token_avg PLY saved.")

    # ── remove hooks so decoder is clean for the next scene ───────────────────
    for h in hooks:
        h.remove()
    ip_state[0] = None

    return history


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rooms",      nargs="+", default=["room0", "room1", "office0", "room2"])
    parser.add_argument("--n-samples",  type=int,   default=200)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    parser.add_argument("--steps",      type=int,   default=20000)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)

    # FM loss
    parser.add_argument("--fm-num-queries",     type=int,   default=1024,
                        help="Query points per FM step (was 512)")
    parser.add_argument("--token-noise-sigma",  type=float, default=0.02)
    parser.add_argument("--t-min",              type=float, default=0.3,
                        help="Sample t ~ U[t_min, 1] to skip uninformative near-noise regime")

    # CD rollout loss
    parser.add_argument("--cd-rollout-steps",   type=int,   default=5,
                        help="Euler steps for ODE rollout CD loss (0 = disabled)")
    parser.add_argument("--cd-rollout-weight",  type=float, default=0.005,
                        help="Weight on CD rollout loss relative to FM loss")
    parser.add_argument("--cd-rollout-start",   type=int,   default=5000,
                        help="Warmup: only activate CD rollout loss after this many steps")
    parser.add_argument("--cd-rollout-queries", type=int,   default=256,
                        help="Query points for rollout (kept small for memory)")
    parser.add_argument("--cd-rollout-batch",   type=int,   default=4,
                        help="Batch size subset used for rollout")

    # eval / decode
    parser.add_argument("--eval-every",    type=int,   default=2000)
    parser.add_argument("--num-queries",   type=int,   default=4096)
    parser.add_argument("--ode-step-size", type=float, default=0.01,
                        help="ODE step size for eval decoding (was 0.04)")
    parser.add_argument("--velocity-scale",type=float, default=1.0,
                        help=">1 pushes points further from centre; try 1.2-1.5")

    # Decoder KV fine-tuning
    parser.add_argument("--decoder-lr",  type=float, default=1e-4,
                        help="LR for IP-Adapter branch params")

    parser.add_argument("--log-every",    type=int,   default=200)
    parser.add_argument("--export-n",    type=int,   default=10)
    parser.add_argument("--save-ckpt",   action="store_true", default=True,
                        help="Save best adapter checkpoint alongside PLYs")
    parser.add_argument("--token-avg-k", type=int,   default=0,
                        help="Average adapter output over K training windows at inference (0=disabled)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config",     type=Path,
                        default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--data-cache", default="multi_scene_N200_s20")
    parser.add_argument("--overfit", action="store_true", default=False,
                        help="Memorization test: eval on training data (val==train)")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)

    EXP_DIR   = Path(__file__).parent
    data_root = EXP_DIR / "data" / args.data_cache
    ply_dir   = EXP_DIR / "ply_export" / "scene_adapt_v3" if args.export_n > 0 else None

    print("Loading NOVA3R ...")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    run = wandb.init(
        project="da3-nova3r-adapter",
        name=f"scene_adapt_v3_{'_'.join(args.rooms)}",
        config=vars(args),
        tags=["scene-adapt", "fm-only", "residual", "cd-rollout", "t-bias"],
    )
    print(f"[wandb] {run.url}")

    results = {}
    for room in args.rooms:
        print(f"\n{'='*60}\n  Scene: {room}\n{'='*60}")
        dirs, da3, zstar, pnorm = load_scene(room, args.n_samples, data_root, device)
        if len(dirs) < 4:
            print(f"  [{room}] too few samples, skipping")
            continue
        history = adapt_scene(
            room=room, da3=da3, zstar=zstar, pnorm=pnorm,
            nova_model=nova_model, nova_cfg=nova_cfg,
            cfg=cfg, args=args, device=device, wandb_run=run,
            ply_dir=ply_dir,
        )
        results[room] = history

    print(f"\n{'='*60}\n  SUMMARY — val CD per scene\n{'='*60}")
    print(f"  {'room':<12}  {'best_CD':>9}  {'@step':>7}  {'final_CD':>9}")
    print(f"  {'-'*46}")
    for room, history in results.items():
        if not history:
            continue
        best  = min(history, key=lambda x: x["val_cd"])
        final = history[-1]
        print(f"  {room:<12}  {best['val_cd']:9.4f}  {best['step']:7d}  {final['val_cd']:9.4f}")
    print(f"{'='*60}")

    wandb.finish()


if __name__ == "__main__":
    main()
