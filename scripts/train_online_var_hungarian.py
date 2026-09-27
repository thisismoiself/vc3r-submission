#!/usr/bin/env python3
"""Train DA3→NOVA3R adapter with Hungarian matching and variance-weighted MSE.

Targets: z_star_online_mean.pt (1,768,128) and z_star_online_var_eff.pt (1,768,128)
from cache_online_hungarian_zstar_windows.py.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_ROOT / "third_party"), str(NOVA3R_ROOT),
           str(REPO_ROOT / "da3" / "src"), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402

DATA_ROOT = REPO_ROOT / "scripts" / "data" / "online_hungarian_zstar_windows"
CFG_PATH  = OVERFIT_SRC / "config.yaml"
N_FRAMES  = 8

_STORAGE_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rooms", type=str, nargs="*", default=None,
                   help="Train roots (bare path = all windows, or root:f0,f1,... by start frame).")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help="Single-root cache path used when --rooms is omitted.")
    p.add_argument("--train-starts", type=int, nargs="*", default=None)
    p.add_argument("--val-root", type=Path, nargs="*", default=None)
    p.add_argument("--val-rooms", type=str, nargs="*", default=None,
                   help="Validation roots (root:s0,s1,...); overrides --val-root/--val-starts.")
    p.add_argument("--val-starts", type=int, nargs="+", default=None)
    p.add_argument("--max-per-root", type=int, default=None,
                   help="Cap training windows loaded per root (RAM control).")
    p.add_argument("--max-val-per-root", type=int, default=None,
                   help="Cap validation windows loaded per root.")
    p.add_argument("--lazy-da3", action="store_true",
                   help="Stream train DA3 tokens per-batch from a local float16 prestage "
                        "dir instead of preloading all into RAM (enables full-data training).")
    p.add_argument("--da3-cache-dir", type=Path,
                   default=Path("/tmp/claude-264016/da3_f16_cache"),
                   help="Local prestage dir for --lazy-da3.")
    p.add_argument("--lru-windows", type=int, default=300,
                   help="Number of DA3 windows kept in RAM by the lazy loader.")
    p.add_argument("--loss-mode", choices=("var_weighted", "plain", "sqrt_var", "huber"),
                   default="var_weighted",
                   help="Token weighting in the training loss. 'plain'/'sqrt_var' stop "
                        "down-weighting high-variance detail tokens; 'huber' is a robust "
                        "(smooth-L1) unweighted loss that caps the influence of blown-up tokens.")
    p.add_argument("--huber-delta", type=float, default=1.0,
                   help="Transition point for --loss-mode huber (per-element diff units).")
    p.add_argument("--mse-weight", type=float, default=1.0,
                   help="Weight on the Hungarian token-MSE base loss. Set 0 to train PURELY on the "
                        "flow-matching signal (requires --vel-loss-weight>0 and/or --endpoint-loss-weight>0). "
                        "NOTE: with 0, token-MSE (val_w) is NOT the objective and rises during training, so "
                        "val_w-based 'best' is misleading -- evaluate the *_last.pt endpoint geometrically.")
    p.add_argument("--vel-loss-weight", type=float, default=0.0,
                   help="Weight of the geometry-aware velocity-matching aux loss (0=off). Matches "
                        "the frozen NOVA3R decoder's velocity field under z_pred vs the target tokens "
                        "on NOVA3R's cosine flow-matching path. Permutation-invariant (no Hungarian).")
    p.add_argument("--vel-points", type=int, default=1024,
                   help="Query points per sample for the velocity loss.")
    p.add_argument("--endpoint-loss-weight", type=float, default=0.0,
                   help="Weight of the endpoint-matching aux loss (Option B, 0=off). Integrates the "
                        "frozen decoder's ODE for z_pred vs the target tokens from a shared noise init "
                        "and L2s the final points per-point (correspondence for free). Loads the decoder "
                        "like --vel-loss-weight; can be combined with it.")
    p.add_argument("--endpoint-points", type=int, default=256,
                   help="Query points per sample for the endpoint loss (kept small: backprop runs "
                        "through --endpoint-steps decoder forwards).")
    p.add_argument("--endpoint-steps", type=int, default=5,
                   help="Euler ODE steps for the endpoint loss (more=faithfuller endpoint, more memory).")
    p.add_argument("--nova-ckpt", type=Path,
                   default=REPO_ROOT / "nova3r_lib" / "checkpoints" / "scene_ae" / "checkpoint-last.pth",
                   help="NOVA3R decoder checkpoint, loaded only when --vel-loss-weight>0.")
    p.add_argument("--decoder-head", type=Path, default=None,
                   help="Optional fine-tuned pts3d_head state_dict to load INTO the frozen NOVA3R "
                        "decoder used by the vel/endpoint aux losses. Aligns the adapter's tokens to "
                        "the exact decoder used at inference (stitch --decoder-ckpt). Encoder stays "
                        "frozen so the z* target is unchanged; only the geometry-aware aux signal moves.")
    p.add_argument("--geom-cond", action="store_true",
                   help="Feed explicit 3D geometry into the adapter: a subsample of the window's "
                        "normalized point cloud is Fourier-embedded and appended to the DA3 source "
                        "tokens so the target queries attend over metric geometry, not just appearance. "
                        "Source is pts_norm (the same visible cloud the decoder is conditioned on).")
    p.add_argument("--geom-points", type=int, default=512,
                   help="Number of geometry points sampled per window per step for --geom-cond.")
    p.add_argument("--geom-bands", type=int, default=16,
                   help="Fourier frequency bands for the geometry positional embedding.")
    p.add_argument("--geom-source", choices=["pts_norm", "da3"], default="pts_norm",
                   help="Geometry cloud for --geom-cond: 'pts_norm' (GT-derived, already cached, "
                        "upper-bound) or 'da3' (GT-free DA3 depth points, needs cache_da3_points.py "
                        "-> da3_pts_norm.pt; the deployable version). Both live in the pts_norm frame.")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--cosine-t0", type=int, default=None,
                   help="First restart cycle length; omit for single cosine sweep.")
    p.add_argument("--cosine-t-mult", type=int, default=2)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay for shadow weights at val/checkpoint. 0 to disable.")
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--token-drop", type=float, default=0.0,
                   help="Fraction of source tokens to drop each step.")
    p.add_argument("--cache-dtype", choices=list(_STORAGE_DTYPES), default="float32")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pca-project", type=Path, default=None,
                   help="If set, project the z* TARGET onto its top-K PCA subspace (tail dims -> "
                        "pooled mean) before use as the Hungarian + velocity target. Basis file is a "
                        "dict {'mean':(128,), 'V':(128,128)} from the training z* pool.")
    p.add_argument("--pca-dims", type=int, default=32,
                   help="K for --pca-project: number of top PCA dims kept (rest set to the mean).")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="Save a periodic *_step{N}.pt snapshot every N steps (0=off) for mid-training "
                        "geometric eval, independent of the val_w-based 'best'. Must be a multiple of "
                        "--log-every.")
    p.add_argument("--brightness", type=float, nargs="?", const=1.2, default=None, metavar="F",
                   help="Token brightness jitter; multiplier (default 1.2), 25%%up/25%%down/50%%off.")
    p.add_argument("--contrast", type=float, nargs="?", const=1.2, default=None, metavar="F",
                   help="Token contrast jitter; scales deviations from token mean.")
    p.add_argument("--hue", type=float, nargs="?", const=20.0, default=None, metavar="DEG",
                   help="Token hue approx; 2D rotation in random feature pair, 50%% prob.")
    p.add_argument("--saturation", type=float, nargs="?", const=1.2, default=None, metavar="F",
                   help="Token saturation jitter; scales per-feature deviations from feature mean.")
    p.add_argument("--aug-seed", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-ckpt", type=Path, default=None)
    p.add_argument("--load-ckpt", type=Path, default=None, help="Load checkpoint and evaluate only.")
    p.add_argument("--ckpt-out", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _parse_window_start(d: Path) -> int:
    return int(d.name.split("_")[0])


def discover_cached_windows(data_root: Path) -> list[Path]:
    windows = sorted(data_root.glob("[0-9]*_[0-9]*"))
    if not windows:
        raise FileNotFoundError(f"No cached windows found under {data_root}")
    return windows


def find_window_by_start(root: Path, start: int) -> Path:
    matches = sorted(root.glob(f"{start}_*"))
    if not matches:
        raise FileNotFoundError(f"No window with start frame {start} under {root}")
    return matches[0]


def parse_root_configs(items: list[str]) -> list[tuple[Path, list[Path]]]:
    configs = []
    for item in items:
        if ":" in item:
            root_str, starts_str = item.rsplit(":", 1)
            root = Path(root_str)
            windows = [find_window_by_start(root, int(s)) for s in starts_str.split(",") if s]
        else:
            root = Path(item)
            windows = discover_cached_windows(root)
        configs.append((root, windows))
    return configs


def load_window(win_dir: Path):
    if not win_dir.exists():
        raise FileNotFoundError(f"Window directory not found: {win_dir}")
    meta    = torch.load(win_dir / "meta.pt",                 map_location="cpu", weights_only=False)
    da3     = torch.load(win_dir / "da3_tokens.pt",           map_location="cpu", weights_only=True)
    mean    = torch.load(win_dir / "z_star_online_mean.pt",   map_location="cpu", weights_only=True)
    var_eff = torch.load(win_dir / "z_star_online_var_eff.pt",map_location="cpu", weights_only=True)
    return meta, da3, mean[0].float(), var_eff[0].float()


def load_pts_norm(win_dir: Path):
    """Normalized surface points (data manifold x_1 for the velocity loss)."""
    return torch.load(win_dir / "pts_norm.pt", map_location="cpu", weights_only=True)[0].float()


def load_geom(win_dir: Path, source: str):
    """Geometry cloud for --geom-cond, in the pts_norm frame. 'pts_norm' = GT-derived
    (already cached); 'da3' = GT-free DA3 depth points from cache_da3_points.py."""
    fname = "pts_norm.pt" if source == "pts_norm" else "da3_pts_norm.pt"
    path = win_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run scripts/cache_da3_points.py for --geom-source da3.")
    return torch.load(path, map_location="cpu", weights_only=True)[0].float()


def sample_geom(pts_batch: torch.Tensor, m: int, generator: torch.Generator | None):
    """Subsample m points per window from a (B, K, 3) batch for geometry conditioning.

    A fresh random subset each step doubles as light augmentation; pass a fixed
    generator (or None) for deterministic validation.
    """
    b, k, _ = pts_batch.shape
    if k <= m:
        return pts_batch
    if generator is None:
        idx = torch.stack([torch.linspace(0, k - 1, m).long() for _ in range(b)])
    else:
        idx = torch.stack([torch.randperm(k, generator=generator)[:m] for _ in range(b)])
    return torch.gather(pts_batch, 1, idx.unsqueeze(-1).expand(-1, -1, 3).to(pts_batch.device))


def load_targets(win_dir: Path):
    """Load only the small targets/meta (no da3 tokens) for lazy training."""
    meta    = torch.load(win_dir / "meta.pt",                 map_location="cpu", weights_only=False)
    mean    = torch.load(win_dir / "z_star_online_mean.pt",   map_location="cpu", weights_only=True)
    var_eff = torch.load(win_dir / "z_star_online_var_eff.pt",map_location="cpu", weights_only=True)
    return meta, mean[0].float(), var_eff[0].float()


class LazyDA3:
    """DA3 tokens streamed per-window from a local float16 prestage dir.

    RAM stays bounded by an LRU instead of scaling with the window count, so we
    can train on the full window set on a memory-limited box. Each source window
    is converted to local float16 once (avoids slow repeated NFS reads).
    """

    def __init__(self, win_dirs, cache_dir: Path, dtype, lru: int):
        self.dtype = dtype
        self.lru = max(1, lru)
        self.paths = []
        cache_dir.mkdir(parents=True, exist_ok=True)
        staged = 0
        for wd in win_dirs:
            lp = cache_dir / f"{wd.parent.name}__{wd.name}.pt"
            if not lp.exists():
                t = torch.load(wd / "da3_tokens.pt", map_location="cpu",
                               weights_only=True).to(dtype).contiguous()
                tmp = lp.with_suffix(".tmp")
                torch.save(t, tmp); tmp.rename(lp); staged += 1
            self.paths.append(lp)
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        print(f"[lazy] {len(self.paths)} windows prestaged to {cache_dir} "
              f"({staged} new), RAM LRU={self.lru}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        c = self._cache
        if i in c:
            c.move_to_end(i)
            return c[i]
        t = torch.load(self.paths[i], map_location="cpu", weights_only=True)
        c[i] = t
        if len(c) > self.lru:
            c.popitem(last=False)
        return t


def matched_loss(pred, mean, var_eff, mode: str, huber_delta: float = 1.0):
    """Hungarian-matched MSE with selectable per-token weighting.

    mode='var_weighted' divides by var_eff (down-weights high-variance/detail
    tokens — the production loss); 'plain' is unweighted; 'sqrt_var' is a gentler
    1/sqrt(var) compromise; 'huber' is an unweighted smooth-L1 that caps the
    influence of a few blown-up tokens (robustness).
    """
    cols = _batch_match_cols(pred, mean)
    losses = []
    for i, col_ind in enumerate(cols):
        col = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)
        tgt = mean[i].to(pred.device)[col]
        if mode == "huber":
            losses.append(torch.nn.functional.huber_loss(pred[i], tgt, delta=huber_delta, reduction="mean"))
            continue
        diff = (pred[i] - tgt).square()
        if mode == "var_weighted":
            diff = diff / var_eff[i].to(pred.device)[col]
        elif mode == "sqrt_var":
            diff = diff / var_eff[i].to(pred.device)[col].sqrt()
        elif mode != "plain":
            raise ValueError(f"unknown loss mode {mode}")
        losses.append(diff.mean())
    return torch.stack(losses).mean()


def velocity_match_loss(nova, z_pred, z_tgt, surf_pts, n_q):
    """Geometry-aware aux loss: match the frozen NOVA3R decoder's velocity field
    under z_pred vs the target tokens, sampled on NOVA3R's cosine flow-matching path
    (alpha_t=sin(pi/2 t), sigma_t=cos(pi/2 t); x_t = sigma_t*x_0 + alpha_t*x_1).
    x_1 = real normalized surface; x_0 = uniform prior (matches the rand*2-1 decode init).
    Decoder consumes tokens as a permutation-invariant set, so no Hungarian is needed:
    this drives decode(z_pred) -> decode(z_tgt) directly in geometry space.
    """
    device = z_pred.device
    B = z_pred.shape[0]
    K = surf_pts.shape[1]
    sel = torch.randint(K, (B, n_q), device=device)
    x1  = torch.gather(surf_pts, 1, sel.unsqueeze(-1).expand(B, n_q, 3))
    x0  = torch.rand(B, n_q, 3, device=device) * 2 - 1
    t   = torch.rand(B, 1, 1, device=device)
    a   = torch.sin(math.pi / 2 * t)
    s   = torch.cos(math.pi / 2 * t)
    xt  = s * x0 + a * x1
    tq  = t.view(B, 1).expand(B, n_q)
    img0 = torch.zeros(B, 1, 3, 1, 1, device=device)
    v_pred = nova._decode(tokens=z_pred, images=img0, query_points=xt, timestep=tq)["pts3d_xyz"]
    with torch.no_grad():
        v_tgt = nova._decode(tokens=z_tgt, images=img0, query_points=xt, timestep=tq)["pts3d_xyz"]
    return (v_pred - v_tgt).square().mean()


def endpoint_match_loss(nova, z_pred, z_tgt, n_q, n_steps):
    """Per-point endpoint-matching loss (Option B): integrate the frozen decoder's ODE
    for z_pred and z_tgt from the SAME noise init, then L2 the final points *per point*.
    The shared init gives correspondence for free (point i of both starts at the same
    noise), so this directly says 'the point born at noise n_i must land exactly where the
    oracle sends it' -- a stronger, correspondence-based signal than set losses (Chamfer/F)
    and than velocity matching (which only constrains the field at sampled x_t). Decoder
    frozen; gradient flows only through z_pred. Euler integration matches the ODESolver.
    """
    device = z_pred.device
    B = z_pred.shape[0]
    img0 = torch.zeros(B, 1, 3, 1, 1, device=device)
    x0 = torch.rand(B, n_q, 3, device=device) * 2 - 1          # uniform prior (decode init)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

    def integrate(z):
        x = x0
        for i in range(n_steps):
            tq = ts[i].expand(B, n_q)
            v = nova._decode(tokens=z, images=img0, query_points=x, timestep=tq)["pts3d_xyz"]
            x = x + (ts[i + 1] - ts[i]) * v
        return x

    x_pred = integrate(z_pred)
    with torch.no_grad():
        x_tgt = integrate(z_tgt)
    return (x_pred - x_tgt).square().mean()


def token_dropout(tokens: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return tokens
    keep = max(1, int(tokens.shape[1] * (1.0 - p)))
    return tokens[:, torch.randperm(tokens.shape[1], device=tokens.device)[:keep], :]


def augment_tokens(tokens, rng, brightness, contrast, hue_deg, saturation):
    out = tokens
    if brightness is not None:
        c = int(torch.randint(4, (1,), generator=rng).item())
        if c == 2: out = out * brightness
        elif c == 3: out = out * (1.0 / brightness)
    if contrast is not None:
        c = int(torch.randint(4, (1,), generator=rng).item())
        if c in (2, 3):
            f = contrast if c == 2 else 1.0 / contrast
            m = out.mean(dim=1, keepdim=True)
            out = m + f * (out - m)
    if hue_deg is not None and torch.randint(2, (1,), generator=rng).item():
        angle = (torch.rand(1, generator=rng).item() * 2 - 1) * hue_deg
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        i = int(torch.randint(out.shape[-1] - 1, (1,), generator=rng).item())
        xi, xj = out[..., i].clone(), out[..., i + 1].clone()
        out = out.clone()
        out[..., i], out[..., i + 1] = c * xi - s * xj, s * xi + c * xj
    if saturation is not None:
        c = int(torch.randint(4, (1,), generator=rng).item())
        if c in (2, 3):
            f = saturation if c == 2 else 1.0 / saturation
            m = out.mean(dim=-1, keepdim=True)
            out = m + f * (out - m)
    return out


# The z_star cost matrices are near-degenerate (tokens cluster tightly), which
# makes scipy's linear_sum_assignment pathologically slow (~0.5s/match → 13s/step).
# lapjv (Jonker-Volgenant) is ~5x faster and releases the GIL, so threading the
# per-sample matches across the batch gives ~20x over threaded scipy.
try:
    import lap  # lapx
    def _solve_lsap(c):
        return lap.lapjv(c, extend_cost=True)[1]   # x[i] = column assigned to row i
except ImportError:                                # fallback (much slower here)
    def _solve_lsap(c):
        return linear_sum_assignment(c)[1]
_LSAP_POOL = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 8)))


def _batch_match_cols(pred, mean):
    """Optimal column assignment per batch element, computed in parallel."""
    costs = torch.cdist(pred.detach(), mean).cpu().numpy()   # [B, N, N]
    return list(_LSAP_POOL.map(_solve_lsap, costs))


def variance_weighted_hungarian_mse(pred, mean, var_eff):
    cols = _batch_match_cols(pred, mean)
    losses = []
    for i, col_ind in enumerate(cols):
        col = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)
        losses.append(((pred[i] - mean[i].to(pred.device)[col]).square()
                       / var_eff[i].to(pred.device)[col]).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def plain_hungarian_mse(pred, mean):
    cols = _batch_match_cols(pred, mean)
    losses = []
    for i, col_ind in enumerate(cols):
        col = torch.as_tensor(col_ind, device=pred.device, dtype=torch.long)
        losses.append((pred[i] - mean[i].to(pred.device)[col]).square().mean())
    return torch.stack(losses).mean()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(CFG_PATH)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    val_starts = list(args.val_starts) if args.val_starts is not None else [56]
    val_root_auto = args.val_root is not None and args.val_starts is None

    if args.rooms:
        train_configs = parse_root_configs(args.rooms)
    else:
        windows = (discover_cached_windows(args.data_root)
                   if args.train_starts is None
                   else [find_window_by_start(args.data_root, s) for s in args.train_starts])
        train_configs = [(args.data_root, windows)]

    if args.val_rooms:
        val_configs = parse_root_configs(args.val_rooms)
    else:
        val_roots = list(args.val_root) if args.val_root else [r for r, _ in train_configs]
        val_configs = [
            (root, discover_cached_windows(root) if val_root_auto
             else [find_window_by_start(root, s) for s in val_starts])
            for root in val_roots
        ]

    if args.max_per_root is not None:
        train_configs = [(r, w[:args.max_per_root]) for r, w in train_configs]
    if args.max_val_per_root is not None:
        val_configs = [(r, w[:args.max_val_per_root]) for r, w in val_configs]

    print(f"[setup] train_roots={len(train_configs)} val_roots={len(val_configs)}")
    for root, windows in train_configs:
        print(f"  train root={root} n={len(windows)}")
    for root, windows in val_configs:
        print(f"  val   root={root} n={len(windows)}")

    da3_dtype = _STORAGE_DTYPES[args.cache_dtype]
    use_vel = args.vel_loss_weight > 0.0
    use_endpoint = args.endpoint_loss_weight > 0.0
    use_decoder = use_vel or use_endpoint          # both aux losses need the frozen NOVA3R decoder
    if args.mse_weight == 0.0 and not use_decoder:
        raise SystemExit("--mse-weight 0 needs a flow-matching signal: set --vel-loss-weight>0 "
                         "and/or --endpoint-loss-weight>0.")
    pca_proj = None
    if args.pca_project is not None:
        _b = torch.load(args.pca_project, map_location="cpu", weights_only=False)
        _pm = _b["mean"].float(); _V = _b["V"].float()
        _P = _V[:, :args.pca_dims] @ _V[:, :args.pca_dims].T          # (128,128) top-K projector
        pca_proj = lambda m: _pm + (m - _pm) @ _P
        print(f"[pca-project] top-{args.pca_dims} PCA subspace from {args.pca_project.name} "
              f"(tail dims -> pooled mean); applied to Hungarian + velocity targets")
    train_da3, train_mean, train_var, train_pts, train_geom = [], [], [], [], []
    train_win_dirs = []
    for root, windows in train_configs:
        for win_dir in windows:
            if args.lazy_da3:
                meta, mean, var_eff = load_targets(win_dir)
            else:
                meta, da3, mean, var_eff = load_window(win_dir)
                train_da3.append(da3.to(dtype=da3_dtype))
            train_win_dirs.append(win_dir)
            train_mean.append(pca_proj(mean) if pca_proj is not None else mean)
            train_var.append(var_eff)
            if use_vel:                                       # pts_norm = velocity-loss manifold x_1
                train_pts.append(load_pts_norm(win_dir))
            if args.geom_cond:                                # geometry conditioning source
                train_geom.append(load_geom(win_dir, args.geom_source))
    _uses = ([f"pts_norm for velocity loss"] if use_vel else []) + \
            ([f"{args.geom_source} for geom cond"] if args.geom_cond else [])
    print(f"  loaded targets for {len(train_win_dirs)} train windows"
          + (f" (+{', '.join(_uses)})" if _uses else ""))
    if args.lazy_da3:
        train_da3 = LazyDA3(train_win_dirs, args.da3_cache_dir, da3_dtype, args.lru_windows)

    val_da3s, val_means, val_vars, val_geom = [], [], [], []
    for root, windows in val_configs:
        for win_dir in windows:
            meta, da3, mean, var_eff = load_window(win_dir)
            val_da3s.append(da3.to(dtype=da3_dtype))
            val_means.append((pca_proj(mean) if pca_proj is not None else mean).unsqueeze(0))
            val_vars.append(var_eff.unsqueeze(0))
            if args.geom_cond:
                val_geom.append(load_geom(win_dir, args.geom_source))
            s = meta.get("start_idx", _parse_window_start(win_dir))
            print(f"  [{root.name}] val   start={s:4d} {win_dir.name:<20} "
                  f"frames {meta['frame_ids'][0]}-{meta['frame_ids'][-1]} "
                  f"var_floor={float(meta.get('var_floor', float('nan'))):.6g}")

    # Stack mean/var but keep DA tokens as a list — concatenating all doubles peak RAM.
    mean_train = torch.stack(train_mean)
    var_train  = torch.stack(train_var)
    del train_mean, train_var

    pts_train = torch.stack(train_pts) if use_vel else None      # (N, K, 3) velocity-loss manifold
    if use_vel:
        del train_pts
    geom_train = torch.stack(train_geom) if args.geom_cond else None  # (N, M, 3) geom-cond cloud
    if args.geom_cond:
        del train_geom

    nova = None
    if use_decoder:
        _NOVA = REPO_ROOT / "nova3r_lib"
        for _p in [str(_NOVA / "third_party"), str(_NOVA)]:
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from demo_nova3r import load_model as _load_nova
        nova, _ = _load_nova(str(args.nova_ckpt), "cuda")
        if args.decoder_head is not None:
            _hd = torch.load(args.decoder_head, map_location="cpu", weights_only=True)
            _miss, _unexp = nova.pts3d_head.load_state_dict(_hd, strict=False)
            print(f"  [decoder-head] loaded fine-tuned pts3d_head {args.decoder_head.name} "
                  f"(missing={len(_miss)} unexpected={len(_unexp)}) -> aux loss now targets the FT decoder")
        nova.eval()
        for _pp in nova.parameters():
            _pp.requires_grad_(False)
        _dparts = ([f"vel(w={args.vel_loss_weight},p={args.vel_points})"] if use_vel else []) + \
                  ([f"endpoint(w={args.endpoint_loss_weight},p={args.endpoint_points},s={args.endpoint_steps})"]
                   if use_endpoint else [])
        print(f"  [decoder] loaded NOVA3R decoder from {args.nova_ckpt} for {', '.join(_dparts)}")

    n_train = len(train_da3)
    if n_train == 0:    raise RuntimeError("No training windows loaded")
    if not val_da3s:    raise RuntimeError("No validation windows loaded")

    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim), hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens), target_dim=int(cfg.target_dim),
        depth=int(cfg.depth), num_heads=int(cfg.num_heads), drop=args.drop,
        geom_cond=args.geom_cond, geom_bands=args.geom_bands,
    ).to(device)
    if args.geom_cond:
        print(f"  [geom] geometry conditioning ON: {args.geom_points} pts/window "
              f"(pts_norm), {args.geom_bands} Fourier bands")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)  # torch 2.2 API

    post_warmup = max(1, args.steps - args.warmup_steps)
    if args.cosine_t0 is not None:
        cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_t0, T_mult=args.cosine_t_mult, eta_min=args.lr_min)
    else:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=post_warmup, eta_min=args.lr_min)
    if args.warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_steps), cosine],
            milestones=[args.warmup_steps],
        )
    else:
        scheduler = cosine

    rng        = torch.Generator().manual_seed(args.seed)
    batch_size = n_train if args.batch_size is None else min(int(args.batch_size), n_train)
    run_tag    = "online_var_hungarian"
    ckpt_path  = args.ckpt_out or (REPO_ROOT / "outputs" / "consecutive_windows" / f"{run_tag}_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    init_step, init_val_loss, init_state = 0, float("inf"), None
    if args.init_ckpt is not None:
        d = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(d["state_dict"])
        init_step, init_val_loss = int(d.get("step", 0)), float(d.get("val_loss", float("inf")))
        init_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
        if d.get("scheduler_state"):
            scheduler.load_state_dict(d["scheduler_state"])
        print(f"[init] {args.init_ckpt} step={init_step} val_loss={init_val_loss}")

    def checkpoint_payload(state_dict, step, val_loss, val_plain_mse, sched_state=None):
        return {
            "state_dict": state_dict, "step": step, "val_loss": val_loss,
            "scheduler_state": sched_state,
            "metadata": {
                "loss": "hungarian_inverse_variance_weighted_mse",
                "val_plain_hungarian_mse": val_plain_mse,
                "n_train_windows": n_train, "n_val_windows": len(val_da3s),
                "batch_size": batch_size, "lr": args.lr, "steps": args.steps, "seed": args.seed,
            },
        }

    use_ema   = args.ema_decay > 0.0
    ema_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()} if use_ema else None

    if args.load_ckpt is not None:
        d = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
        adapter.load_state_dict(d["state_dict"])
        adapter.eval()
        print(f"[ckpt] loaded {args.load_ckpt} step={d.get('step')} val_loss={d.get('val_loss'):.6f}")
        best_state = d["state_dict"]
    else:
        adapter.train()
        best_val_loss, best_step, best_state, saved = init_val_loss, init_step, init_state, False
        aug_rng = torch.Generator().manual_seed(
            args.aug_seed if args.aug_seed is not None else args.seed + 100_000)
        csv_path   = ckpt_path.parent / f"{run_tag}_metrics.csv"
        csv_file   = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file, fieldnames=["step", "lr", "train_loss", "val_weighted", "val_plain_mse", "is_best"])
        csv_writer.writeheader()

        sched_str = (f"cosine restarts T0={args.cosine_t0} Tmult={args.cosine_t_mult}"
                     if args.cosine_t0 is not None else "cosine annealing")
        aug_parts = ([f"b={args.brightness}"] if args.brightness else []) + \
                    ([f"c={args.contrast}"]    if args.contrast   else []) + \
                    ([f"h={args.hue}°"]        if args.hue        else []) + \
                    ([f"s={args.saturation}"]  if args.saturation else [])
        print(f"\n[train] {n_train}w steps={args.steps} lr={args.lr}→{args.lr_min} "
              f"warmup={args.warmup_steps} drop={args.drop} tdrop={args.token_drop} "
              f"wd={args.weight_decay} batch={batch_size} amp={use_amp} "
              f"ema={args.ema_decay if use_ema else 'off'} val_every={args.log_every}"
              + (f" aug=[{','.join(aug_parts)}]" if aug_parts else "")
              + f" ({sched_str})")

        t0 = time.time()
        for step in range(1, args.steps + 1):
            idx       = torch.randperm(n_train, generator=rng)[:batch_size]
            da3_batch = torch.cat([train_da3[i] for i in idx.tolist()], dim=0)
            src = augment_tokens(
                token_dropout(da3_batch.to(device), args.token_drop),
                aug_rng, args.brightness, args.contrast, args.hue, args.saturation,
            )
            geom = None
            if args.geom_cond:
                geom = sample_geom(geom_train[idx], args.geom_points, aug_rng).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = adapter(src, geom_xyz=geom)
            tgt = mean_train[idx].to(device)
            loss = args.mse_weight * matched_loss(
                pred.float(), tgt, var_train[idx].to(device),
                args.loss_mode, args.huber_delta)
            if use_vel:
                vloss = velocity_match_loss(
                    nova, pred.float(), tgt, pts_train[idx].to(device), args.vel_points)
                loss = loss + args.vel_loss_weight * vloss
            if use_endpoint:
                eloss = endpoint_match_loss(
                    nova, pred.float(), tgt, args.endpoint_points, args.endpoint_steps)
                loss = loss + args.endpoint_loss_weight * eloss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if use_ema:
                for k, v in adapter.state_dict().items():
                    ema_state[k].mul_(args.ema_decay).add_(v.cpu(), alpha=1.0 - args.ema_decay)

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                if use_ema:
                    live = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
                    adapter.load_state_dict({k: v.to(device) for k, v in ema_state.items()})
                adapter.eval()
                with torch.no_grad():
                    vl, vp = [], []
                    for vi, (vd, vm, vv) in enumerate(zip(val_da3s, val_means, val_vars)):
                        vgeom = None
                        if args.geom_cond:
                            vgeom = sample_geom(val_geom[vi].unsqueeze(0), args.geom_points, None).to(device)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            out = adapter(vd.to(device), geom_xyz=vgeom).float()
                        vl.append(variance_weighted_hungarian_mse(out, vm.to(device), vv.to(device)).item())
                        vp.append(plain_hungarian_mse(out, vm.to(device)).item())
                    val_loss, val_plain_mse = float(np.mean(vl)), float(np.mean(vp))
                if use_ema:
                    adapter.load_state_dict({k: v.to(device) for k, v in live.items()})
                adapter.train()

                is_best = val_loss < best_val_loss
                lr_now  = scheduler.get_last_lr()[0]
                el = time.time() - t0
                print(f"  step {step:5d}/{args.steps} lr={lr_now:.2e} train={loss.item():.6f} "
                      f"val_w={val_loss:.6f} val_mse={val_plain_mse:.6f} "
                      f"t={el:.0f}s ({el/step:.3f}s/it)"
                      + (" <- best" if is_best else ""), flush=True)
                csv_writer.writerow({"step": step, "lr": lr_now, "train_loss": loss.item(),
                                     "val_weighted": val_loss, "val_plain_mse": val_plain_mse,
                                     "is_best": is_best})
                csv_file.flush()
                if is_best:
                    best_val_loss = val_loss
                    best_step = init_step + step
                    best_state = ({k: v.clone() for k, v in ema_state.items()} if use_ema
                                  else {k: v.cpu().clone() for k, v in adapter.state_dict().items()})
                    torch.save(checkpoint_payload(best_state, best_step, best_val_loss,
                                                  val_plain_mse, scheduler.state_dict()), ckpt_path)
                    saved = True
                # Periodic snapshot (independent of val_w 'best') for mid-training geometric eval.
                # Requires ckpt_every to be a multiple of log_every (this block gates on log_every).
                if args.ckpt_every > 0 and step % args.ckpt_every == 0:
                    snap = ({k: v.clone() for k, v in ema_state.items()} if use_ema
                            else {k: v.cpu().clone() for k, v in adapter.state_dict().items()})
                    snap_path = ckpt_path.with_name(f"{ckpt_path.stem}_step{init_step + step}{ckpt_path.suffix}")
                    torch.save(checkpoint_payload(snap, init_step + step, val_loss,
                                                  val_plain_mse, scheduler.state_dict()), snap_path)
                    print(f"  [snapshot] step {init_step + step} -> {snap_path.name}", flush=True)

        # Always persist the FINAL (EMA) endpoint: for pure-FM runs val_w rises during
        # training, so the val_w-based *_best.pt is the wrong pick -- the geometrically
        # good adapter is this endpoint. Evaluate *_last.pt with the stitch/chamfer eval.
        last_state = ({k: v.clone() for k, v in ema_state.items()} if use_ema
                      else {k: v.cpu().clone() for k, v in adapter.state_dict().items()})
        last_path = ckpt_path.with_name(ckpt_path.stem + "_last" + ckpt_path.suffix)
        torch.save(checkpoint_payload(last_state, init_step + args.steps, best_val_loss,
                                      val_plain_mse, scheduler.state_dict()), last_path)
        print(f"[last] endpoint saved -> {last_path}")

        if best_state is None:
            best_state = last_state
        adapter.load_state_dict(best_state)
        adapter.eval()
        csv_file.close()
        print(f"\n[best] step={best_step} val_weighted={best_val_loss:.6f}")
        print(f"[best] {'saved -> ' + str(ckpt_path) if saved else 'no new checkpoint saved'}")
        print(f"[metrics] {csv_path}")

    with torch.no_grad():
        tl, tp = [], []
        for start in range(0, n_train, batch_size):
            sl = slice(start, min(start + batch_size, n_train))
            da3_e = torch.cat([train_da3[j] for j in range(sl.start, sl.stop)], dim=0)
            geom_e = sample_geom(geom_train[sl], args.geom_points, None).to(device) if args.geom_cond else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = adapter(da3_e.to(device), geom_xyz=geom_e).float()
            tl.append(variance_weighted_hungarian_mse(pred, mean_train[sl].to(device), var_train[sl].to(device)).item())
            tp.append(plain_hungarian_mse(pred, mean_train[sl].to(device)).item())
        vl, vp = [], []
        for vi, (vd, vm, vv) in enumerate(zip(val_da3s, val_means, val_vars)):
            vgeom = sample_geom(val_geom[vi].unsqueeze(0), args.geom_points, None).to(device) if args.geom_cond else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = adapter(vd.to(device), geom_xyz=vgeom).float()
            vl.append(variance_weighted_hungarian_mse(pred, vm.to(device), vv.to(device)).item())
            vp.append(plain_hungarian_mse(pred, vm.to(device)).item())

    print(f"[result] train_weighted={float(np.mean(tl)):.6f} val_weighted={float(np.mean(vl)):.6f} "
          f"train_plain_mse={float(np.mean(tp)):.6f} val_plain_mse={float(np.mean(vp)):.6f}")


if __name__ == "__main__":
    main()
