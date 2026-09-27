#!/usr/bin/env python3
"""
Multi-sample DA3→NOVA3R adapter overfitting experiment.

Each 'sample' = one 8-frame window from Replica room0.
N starting frames are evenly distributed across the full trajectory.
The adapter is trained jointly on all N samples via mini-batch SGD,
then evaluated with NOVA3R ODE decoder (Chamfer distance).

Data is cached per-sample in:
  data/multi_N{n}_s{stride}/sample_{i:03d}/{da3_tokens,z_star,pts_norm,meta}.pt

Usage:
  python multi_sample_train.py --n-samples 2
  python multi_sample_train.py --n-samples 5
  python multi_sample_train.py --n-samples 50 --batch-size 8 --steps 15000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from omegaconf import OmegaConf
from PIL import Image

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
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from depth_anything_3.cfg import create_object                             # noqa: E402
from safetensors.torch import load_file                                     # noqa: E402


# ── DA3 helpers (from extract_data.py) ────────────────────────────────────────

def load_da3_model(model_name: str, device: torch.device):
    if not model_name.startswith("depth-anything/"):
        snapshot = Path(model_name)
    else:
        repo = model_name.replace("/", "--")
        for cache_root in [REPO_ROOT / ".hf_cache", Path.home() / ".cache" / "huggingface"]:
            model_dir = cache_root / "hub" / f"models--{repo}"
            if model_dir.exists():
                break
        revision = (model_dir / "refs" / "main").read_text().strip()
        snapshot  = model_dir / "snapshots" / revision
    with (snapshot / "config.json").open() as f:
        payload = json.load(f)
    model = create_object(OmegaConf.create(payload["config"]))
    state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std  = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    return (images - mean[None, None, :, None, None]) / std[None, None, :, None, None]


def normalize_extrinsics(poses_c2w: torch.Tensor) -> torch.Tensor:
    transform    = torch.linalg.inv(poses_c2w[:, :1])
    normalized   = poses_c2w @ transform
    c2ws         = torch.linalg.inv(normalized)
    median_dist  = c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1)
    normalized[..., :3, 3] /= median_dist
    return normalized


def select_tokens(tokens: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or tokens.shape[1] <= max_n:
        return tokens
    idx = torch.linspace(0, tokens.shape[1] - 1, max_n, device=tokens.device).long()
    return tokens[:, idx]


def crop_frustum_world_points(points_world: torch.Tensor, camera_to_world: torch.Tensor,
                               intrinsics: torch.Tensor, image_hw: tuple) -> torch.Tensor:
    """Keep mesh points inside the camera frustum. Does NOT filter occluded points."""
    H, W = image_hw
    w2c = torch.linalg.inv(camera_to_world)
    ones = torch.ones(points_world.shape[0], 1, dtype=points_world.dtype)
    pts_cam = (torch.cat([points_world, ones], dim=1) @ w2c.T)[:, :3]
    in_front = pts_cam[:, 2] > 0
    pts_cam = pts_cam[in_front]
    u = pts_cam[:, 0] / pts_cam[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]
    v = pts_cam[:, 1] / pts_cam[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]
    in_frustum = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return points_world[in_front][in_frustum]


@contextmanager
def deterministic_numpy_default_rng(seed: int):
    """Patch np.random.default_rng so unseeded calls inside NOVA3R._encode are reproducible."""
    original = np.random.default_rng
    np.random.default_rng = lambda s=None: original(seed if s is None else s)
    try:
        yield
    finally:
        np.random.default_rng = original


@torch.no_grad()
def extract_da3_tokens(model, images: torch.Tensor, poses_c2w: torch.Tensor,
                        intrinsics: torch.Tensor, layer_idx: int,
                        max_tokens: int, device: torch.device) -> torch.Tensor:
    images_bt = images.unsqueeze(0).to(device)
    poses_bt  = poses_c2w.unsqueeze(0).to(device)
    K_bt      = intrinsics.unsqueeze(0).to(device)
    images_norm    = imagenet_normalize(images_bt)
    extrinsics_w2c = normalize_extrinsics(torch.linalg.inv(poses_bt[0]).unsqueeze(0))
    amp_dt = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
              else torch.float16)
    with torch.autocast(device_type=device.type, enabled=False):
        cam_token = model.cam_enc(extrinsics_w2c, K_bt, images_bt.shape[-2:])
    with torch.autocast(device_type=device.type, dtype=amp_dt, enabled=(device.type == "cuda")):
        backbone_out, _ = model.backbone(images_norm, cam_token=cam_token)
    raw  = backbone_out[layer_idx][0].float()              # (T, H_p, W_p, D)
    flat = raw.reshape(raw.shape[0], -1, raw.shape[-1])    # (T, P, D)
    per_frame = select_tokens(flat, max_tokens)             # (T, max_tokens, D)
    return per_frame.reshape(1, -1, per_frame.shape[-1]).cpu()  # (1, T*max_tokens, D)


def world_to_first_camera(pts_world: torch.Tensor,
                           first_pose_c2w: torch.Tensor) -> torch.Tensor:
    w2c  = torch.linalg.inv(first_pose_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_points(pts: torch.Tensor, count: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    if pts.shape[0] >= count:
        idx = torch.randperm(pts.shape[0], generator=gen)[:count]
    else:
        pad = torch.randint(pts.shape[0], (count - pts.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pts.shape[0]), pad])
    return pts[idx]


@torch.no_grad()
def encode_nova3r(model, cfg, pts_first_cam: torch.Tensor,
                   device: torch.device, seed: int = 42):
    """Returns (z_star, pts_norm) both on CPU. pts_first_cam: (1, N, 3)."""
    pts   = pts_first_cam.to(device).float()
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
    norm_mode = cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    with deterministic_numpy_default_rng(seed):
        z = model._encode(pointmaps=pts_norm, test=True)["tokens"].float()
    return z.cpu(), pts_norm.cpu()


# ── NOVA3R ODE decode ──────────────────────────────────────────────────────────

@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor,
                   pts_norm: torch.Tensor, device: torch.device,
                   num_queries: int = 8192, seed: int = 42) -> torch.Tensor:
    """tokens: (B, 768, 128), pts_norm: (B, M, 3). Returns (B, num_queries, 3) on CPU."""
    torch.manual_seed(seed)
    B = tokens.shape[0]
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(B, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(B, num_queries, 3, device=device) * 2 - 1
    wrapper = BatchModelWrapper(model=nova_model)
    solver  = ODESolver(velocity_model=wrapper)
    step_sz = nova_cfg.get("fm_step_size", 0.04)
    method  = nova_cfg.get("fm_sampling", "euler")
    amp_dt  = amp_dtype_mapping.get(nova_cfg.get("amp_dtype", "bf16"), torch.float32)
    T_grid  = torch.linspace(0, 1, int(1 // step_sz)).to(device)
    use_amp = device.type != "cpu"
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    pts3d = sol[-1] if isinstance(sol, list) else sol
    return pts3d.cpu()


# ── Chamfer distance ───────────────────────────────────────────────────────────

def chamfer(x: torch.Tensor, y: torch.Tensor) -> float:
    from pytorch3d.loss import chamfer_distance as cd3
    loss, _ = cd3(x.unsqueeze(0).float(), y.unsqueeze(0).float())
    return float(loss)


# ── Visualization ──────────────────────────────────────────────────────────────

def render_pts(pts: np.ndarray, color_vals: np.ndarray, cmap: str, title: str,
               vmin=None, vmax=None, cbar_label: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    vn  = vmin if vmin is not None else np.percentile(color_vals, 2)
    vx  = vmax if vmax is not None else np.percentile(color_vals, 98)
    norm = Normalize(vmin=vn, vmax=vx)
    cols = plt.get_cmap(cmap)(norm(color_vals))
    step = max(1, len(pts) // 15_000)
    ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1],
               c=cols[::step], s=0.8, linewidths=0, alpha=0.75)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1).set_label(cbar_label, fontsize=8)
    plt.tight_layout()
    return fig


def fig_to_wandb(fig, caption: str = ""):
    import wandb, io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buf).copy(), caption=caption)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples",   type=int, required=True,
                        help="Number of 8-frame windows (2, 5, or 50)")
    parser.add_argument("--stride",      type=int, default=20,
                        help="Frame stride within each 8-frame window")
    parser.add_argument("--steps",       type=int, default=None)
    parser.add_argument("--batch-size",  type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=8192)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)
    N      = args.n_samples
    stride = args.stride
    n_frames   = int(cfg.num_frames)                          # 8
    steps      = args.steps or max(3000, N * 300)
    batch_size = args.batch_size or min(N, 8)

    EXP_DIR   = REPO_ROOT / "experiments" / "overfit_8frames"
    data_root = EXP_DIR / "data"        / f"multi_N{N}_s{stride}"
    ckpt_dir  = EXP_DIR / "checkpoints" / f"multi_N{N}_s{stride}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── scene paths ───────────────────────────────────────────────────────────
    replica_root = Path(str(cfg.replica_root))
    room         = str(cfg.room)
    room_dir     = replica_root / room
    results_dir  = room_dir / "results"
    mesh_ply     = replica_root / f"{room}_mesh.ply"

    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    depth_scale = float(cam_p["scale"])
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                       [0., cam_p["fy"], cam_p["cy"]],
                       [0., 0., 1.]], dtype=np.float32)

    poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    total_frames  = len(all_frame_ids)
    print(f"[multi_sample] Total frames: {total_frames}")

    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    K_proc = K_nat.copy()
    K_proc[0] *= W_proc / W_nat
    K_proc[1] *= H_proc / H_nat
    K_nat_t  = torch.from_numpy(K_nat)
    K_proc_t = torch.from_numpy(K_proc)

    # ── evenly-spaced start indices ───────────────────────────────────────────
    window_idx_span = (n_frames - 1) * stride          # e.g. 7*20=140
    max_start_idx   = total_frames - 1 - window_idx_span
    assert max_start_idx >= 0, \
        f"Not enough frames ({total_frames}) for {n_frames} frames at stride {stride}"
    if N == 1:
        start_indices = [0]
    else:
        start_indices = np.linspace(0, max_start_idx, N, dtype=int).tolist()

    print(f"[multi_sample] N={N} samples, stride={stride}, steps={steps}, bs={batch_size}")
    print(f"[multi_sample] Start indices: {start_indices}")

    # ── determine which samples need extraction ───────────────────────────────
    needs_extract = [
        si for si in range(N)
        if args.force_extract or not all(
            (data_root / f"sample_{si:03d}" / f).exists()
            for f in ["da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"]
        )
    ]

    if needs_extract:
        print(f"[multi_sample] Extracting {len(needs_extract)} samples …")

        # Load mesh (heavy, done once)
        print("[multi_sample] Loading mesh …")
        mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
        mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
        mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

        # Load NOVA3R AE
        print("[multi_sample] Loading NOVA3R AE …")
        nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
        nova_model.eval()
        for p in nova_model.parameters():
            p.requires_grad_(False)
        OmegaConf.set_struct(nova_cfg, False)

        # Load DA3
        print("[multi_sample] Loading DA3 …")
        da3_model = load_da3_model(str(cfg.da3_model), device)

        for si in needs_extract:
            start_idx = start_indices[si]
            sample_dir = data_root / f"sample_{si:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            s_frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
            s_poses     = torch.from_numpy(
                np.stack([poses_all[fid] for fid in s_frame_ids])
            ).float()   # (T, 4, 4)

            print(f"  [sample {si:03d}] start={start_idx}  frames={s_frame_ids}")

            def load_rgb(fid: int) -> torch.Tensor:
                img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
                img = img.resize((W_proc, H_proc), Image.BILINEAR)
                return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

            images_t   = torch.stack([load_rgb(fid) for fid in s_frame_ids])   # (T,3,H,W)
            intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames,1,1))).float()

            # Frustum points (includes occluded geometry) → first-camera frame
            first_c2w    = s_poses[0]
            pts_cam_list = []
            for i, fid in enumerate(s_frame_ids):
                frust = crop_frustum_world_points(
                    points_world    = mesh_pts,
                    camera_to_world = s_poses[i],
                    intrinsics      = K_nat_t,
                    image_hw        = (H_nat, W_nat),
                )
                pts_cam = world_to_first_camera(frust, first_c2w)
                pts_cam_list.append(pts_cam)
                print(f"    frame {fid:06d}: {len(frust):,} pts")

            all_frust = torch.cat(pts_cam_list, dim=0)
            tok_pts   = sample_points(all_frust, int(cfg.mesh_sample_points), seed=si)
            tok_pts_b = tok_pts.unsqueeze(0)  # (1, 8192, 3)

            nova_seed = 1000 + si * 1009 + start_idx
            z_star, pts_norm = encode_nova3r(nova_model, nova_cfg, tok_pts_b, device,
                                              seed=nova_seed)

            da3_tokens = extract_da3_tokens(
                da3_model, images_t, s_poses, intrinsics,
                layer_idx=int(cfg.da3_source_layer_index),
                max_tokens=int(cfg.da3_max_source_tokens),
                device=device,
            )

            torch.save(da3_tokens, sample_dir / "da3_tokens.pt")
            torch.save(z_star,     sample_dir / "z_star.pt")
            torch.save(pts_norm,   sample_dir / "pts_norm.pt")
            torch.save({
                "frame_ids": s_frame_ids,
                "poses_c2w": s_poses,
                "start_idx": start_idx,
                "stride":    stride,
                "room":      room,
            }, sample_dir / "meta.pt")

        del nova_model, da3_model
        torch.cuda.empty_cache()
        print("[multi_sample] Extraction done.")

    # ── load all samples into memory ──────────────────────────────────────────
    all_da3_list    = []
    all_zstar_list  = []
    all_pnorm_list  = []
    all_meta_list   = []

    for si in range(N):
        d = data_root / f"sample_{si:03d}"
        all_da3_list.append(   torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True))
        all_zstar_list.append( torch.load(d / "z_star.pt",     map_location="cpu", weights_only=True))
        all_pnorm_list.append( torch.load(d / "pts_norm.pt",   map_location="cpu", weights_only=True))
        all_meta_list.append(  torch.load(d / "meta.pt",       map_location="cpu", weights_only=False))

    # all shapes: da3=(1, T*N, D), zstar=(1,768,128), pnorm=(1,8192,3)
    all_da3   = torch.cat(all_da3_list,   dim=0)  # (N, T*N_tok, D)
    all_zstar = torch.cat(all_zstar_list, dim=0)  # (N, 768, 128)
    all_pnorm = torch.cat(all_pnorm_list, dim=0)  # (N, 8192, 3)

    print(f"[multi_sample] all_da3:   {tuple(all_da3.shape)}")
    print(f"[multi_sample] all_zstar: {tuple(all_zstar.shape)}")
    print(f"[multi_sample] all_pnorm: {tuple(all_pnorm.shape)}")

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
    ).to(device)
    n_params  = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(adapter.parameters(),
                                   lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # ── W&B init ──────────────────────────────────────────────────────────────
    import wandb
    run_name = f"overfit_multi{N}_s{stride}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        project = str(cfg.wandb_project),
        name    = run_name,
        config  = {
            "mode":           "multi_sample",
            "n_samples":      N,
            "stride":         stride,
            "steps":          steps,
            "batch_size":     batch_size,
            "lr":             float(cfg.lr),
            "weight_decay":   float(cfg.weight_decay),
            "hidden_dim":     int(cfg.hidden_dim),
            "depth":          int(cfg.depth),
            "num_heads":      int(cfg.num_heads),
            "source_dim":     int(cfg.source_dim),
            "target_tokens":  int(cfg.target_tokens),
            "target_dim":     int(cfg.target_dim),
            "n_params":       n_params,
            "num_queries":    args.num_queries,
            "room":           room,
        },
        tags = ["overfit", "multi-sample"],
    )
    print(f"[wandb] {run.url}")

    # ── log sample frame grids (first 8 samples) ──────────────────────────────
    n_show = min(N, 8)
    for si in range(n_show):
        meta = all_meta_list[si]
        fids = meta["frame_ids"]
        fig, axes = plt.subplots(1, n_frames, figsize=(3 * n_frames, 3))
        for ax, fid in zip(axes, fids):
            img = Image.open(results_dir / f"frame{fid:06d}.jpg")
            ax.imshow(img); ax.set_title(f"{fid}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Sample {si:03d}  start_idx={meta['start_idx']}", fontsize=10)
        plt.tight_layout()
        wandb.log({f"samples/frames_{si:03d}": fig_to_wandb(fig, f"sample {si:03d}")}, step=0)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"[multi_sample] Training {steps} steps, bs={batch_size} …")
    adapter.train()
    t0  = time.time()
    rng = torch.Generator().manual_seed(42)

    for step in range(1, steps + 1):
        if N <= batch_size:
            idx = torch.arange(N)
        else:
            idx = torch.randint(0, N, (batch_size,), generator=rng)

        da3_b   = all_da3[idx].to(device)    # (bs, T*N_tok, D)
        zstar_b = all_zstar[idx].to(device)  # (bs, 768, 128)

        pred = adapter(da3_b)
        loss = F.mse_loss(pred, zstar_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % int(cfg.log_every) == 0 or step == 1 or step == steps:
            print(f"  step {step:6d}/{steps}  loss={loss.item():.6f}  "
                  f"t={time.time()-t0:.1f}s")
            wandb.log({"train/loss": loss.item()}, step=step)

    final_loss = loss.item()
    print(f"[multi_sample] Training done. Final loss: {final_loss:.6f}")

    torch.save({
        "adapter_state_dict": adapter.state_dict(),
        "config": OmegaConf.to_container(cfg),
        "n_samples": N, "stride": stride, "steps": steps, "final_loss": final_loss,
    }, ckpt_dir / "adapter_final.pt")
    wandb.log({"train/final_loss": final_loss}, step=steps)

    # ── evaluation ────────────────────────────────────────────────────────────
    print(f"[multi_sample] Evaluating all {N} samples …")
    adapter.eval()

    # Batched adapter forward (avoids OOM for large N)
    pred_all_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            b = all_da3[i:i+batch_size].to(device)
            pred_all_list.append(adapter(b).cpu())
    pred_all = torch.cat(pred_all_list, dim=0)   # (N, 768, 128)

    mse_per = F.mse_loss(pred_all.float(), all_zstar.float(),
                          reduction="none").mean(dim=[1, 2])   # (N,)

    # Load NOVA3R for decoding
    print("[multi_sample] Loading NOVA3R for ODE decode …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    cd_pg_list, cd_ai_list, cd_pi_list = [], [], []
    vis_limit = min(N, 8)   # log images for first 8 samples

    for si in range(N):
        z_i  = all_zstar[si:si+1]      # (1, 768, 128)
        pr_i = pred_all[si:si+1]       # (1, 768, 128)
        pn_i = all_pnorm[si:si+1]      # (1, 8192, 3)

        print(f"  sample {si:03d}: decoding …", end=" ", flush=True)
        gt_dec = decode_tokens(nova_model, nova_cfg, z_i,  pn_i, device,
                                args.num_queries, seed=42+si)[0]  # (Q, 3)
        pr_dec = decode_tokens(nova_model, nova_cfg, pr_i, pn_i, device,
                                args.num_queries, seed=42+si)[0]  # (Q, 3)

        gt_np  = gt_dec.numpy()
        pr_np  = pr_dec.numpy()
        in_np  = all_pnorm[si].numpy()   # (8192, 3)

        cd_pg = chamfer(torch.from_numpy(pr_np), torch.from_numpy(gt_np))
        cd_ai = chamfer(torch.from_numpy(gt_np), torch.from_numpy(in_np))
        cd_pi = chamfer(torch.from_numpy(pr_np), torch.from_numpy(in_np))
        cd_pg_list.append(cd_pg)
        cd_ai_list.append(cd_ai)
        cd_pi_list.append(cd_pi)

        print(f"CD(pg)={cd_pg:.4f}  CD(ai)={cd_ai:.4f}  CD(pi)={cd_pi:.4f}")

        wandb.log({
            f"eval/sample{si:03d}/mse":            float(mse_per[si]),
            f"eval/sample{si:03d}/cd_pred_vs_gt":  cd_pg,
            f"eval/sample{si:03d}/cd_ae_vs_input": cd_ai,
            f"eval/sample{si:03d}/cd_pred_vs_in":  cd_pi,
        }, step=steps)

        # Side-by-side comparison figure (first vis_limit samples)
        if si < vis_limit:
            meta = all_meta_list[si]
            fids = meta["frame_ids"]
            fig_c, axs = plt.subplots(1, 3, figsize=(27, 7),
                                       subplot_kw={"projection": "3d"})
            for ax_c, pts_c, cmap_c, ttl_c in zip(
                axs,
                [in_np,  gt_np,  pr_np],
                ["cividis", "viridis", "plasma"],
                ["Input (normalized)",
                 f"GT decoded  CD_ae={cd_ai:.4f}",
                 f"Pred decoded  CD={cd_pg:.4f}"],
            ):
                clr  = pts_c[:, 1]
                nn   = Normalize(np.percentile(clr, 2), np.percentile(clr, 98))
                cols = plt.get_cmap(cmap_c)(nn(clr))
                stp  = max(1, len(pts_c) // 10_000)
                ax_c.scatter(pts_c[::stp, 0], pts_c[::stp, 2], pts_c[::stp, 1],
                             c=cols[::stp], s=1.0, alpha=0.75)
                ax_c.set_title(ttl_c, fontsize=9, fontweight="bold")
                ax_c.set_xlabel("X"); ax_c.set_ylabel("Z"); ax_c.set_zlabel("Y")
                ax_c.view_init(elev=25, azim=-60)
                ax_c.tick_params(labelsize=7)
            fig_c.suptitle(
                f"Sample {si:03d}  frames {fids[0]:06d}–{fids[-1]:06d} | "
                f"MSE={float(mse_per[si]):.2e}  N_samples={N}",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            wandb.log({
                f"pointclouds/comparison_{si:03d}": fig_to_wandb(fig_c, f"sample {si:03d}")
            }, step=steps)

            # Interactive 3D viewer
            wandb.log({
                f"3d/sample{si:03d}_input":
                    wandb.Object3D({"type":"lidar/beta",
                                    "points": in_np[:, [0,2,1]].astype(np.float32)}),
                f"3d/sample{si:03d}_gt_decoded":
                    wandb.Object3D({"type":"lidar/beta",
                                    "points": gt_np[:, [0,2,1]].astype(np.float32)}),
                f"3d/sample{si:03d}_pred_decoded":
                    wandb.Object3D({"type":"lidar/beta",
                                    "points": pr_np[:, [0,2,1]].astype(np.float32)}),
            }, step=steps)

    # ── aggregate metrics ─────────────────────────────────────────────────────
    cd_pg_mean = float(np.mean(cd_pg_list))
    cd_ai_mean = float(np.mean(cd_ai_list))
    cd_pi_mean = float(np.mean(cd_pi_list))
    mse_mean   = float(mse_per.mean())

    wandb.log({
        "eval_agg/cd_pred_vs_gt_mean":   cd_pg_mean,
        "eval_agg/cd_ae_vs_input_mean":  cd_ai_mean,
        "eval_agg/cd_pred_vs_in_mean":   cd_pi_mean,
        "eval_agg/mse_mean":             mse_mean,
        "eval_agg/final_train_loss":     final_loss,
    }, step=steps)

    # Bar chart: per-sample Chamfer
    fig_w = max(12, N // 2)
    fig_bar, ax_bar = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(N); w = 0.27
    ax_bar.bar(x - w, cd_pg_list, w, label="CD pred↔gt",  color="#e76f51")
    ax_bar.bar(x,     cd_ai_list, w, label="CD ae↔input", color="#2a9d8f")
    ax_bar.bar(x + w, cd_pi_list, w, label="CD pred↔in",  color="#457b9d")
    ax_bar.set_xlabel("Sample index"); ax_bar.set_ylabel("Chamfer distance")
    ax_bar.set_title(f"Per-sample Chamfer  (N={N}, stride={stride})", fontweight="bold")
    ax_bar.legend(); ax_bar.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    wandb.log({"eval_agg/chamfer_barchart": fig_to_wandb(fig_bar)}, step=steps)

    # MSE bar chart
    fig_mse, ax_mse = plt.subplots(figsize=(fig_w, 5))
    ax_mse.bar(np.arange(N), mse_per.numpy(), color="#264653")
    ax_mse.set_xlabel("Sample index"); ax_mse.set_ylabel("MSE (token space)")
    ax_mse.set_title(f"Per-sample MSE  (N={N}, stride={stride})", fontweight="bold")
    ax_mse.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    wandb.log({"eval_agg/mse_barchart": fig_to_wandb(fig_mse)}, step=steps)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  MULTI-SAMPLE  N={N}  STRIDE={stride}  RESULTS")
    print(f"{'='*75}")
    print(f"  {'si':>5}  {'MSE':>10}  {'CD pred↔gt':>12}  {'CD ae↔in':>10}  {'CD pred↔in':>12}")
    for si in range(N):
        print(f"  {si:5d}  {float(mse_per[si]):10.6f}  "
              f"{cd_pg_list[si]:12.6f}  {cd_ai_list[si]:10.6f}  {cd_pi_list[si]:12.6f}")
    print(f"  {'MEAN':>5}  {mse_mean:10.6f}  "
          f"{cd_pg_mean:12.6f}  {cd_ai_mean:10.6f}  {cd_pi_mean:12.6f}")
    print(f"{'='*75}")
    print(f"  W&B: {run.url}")
    print(f"{'='*75}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
