#!/usr/bin/env python3
"""
Multi-scene DA3→NOVA3R adapter training with held-out scene evaluation.

Trains the Q-Former adapter on 8-frame windows drawn from multiple Replica rooms,
then evaluates on a held-out room never seen during training.

Regularization (all opt-in via CLI flags, OFF by default):
  - Random window starts per room (not evenly-spaced) for more intra-room
    diversity — always on, default 200 samples/room
  - Q-Former dropout: --drop 0.1 (default 0.0)
  - Random DA3 token subsampling per step: --tok-subsample 6144 (default 0=all)
  - z_star label noise: --label-noise 0.01 (default 0.0)

A first attempt stacking drop=0.1 + tok-subsample=4096 (50% of tokens) +
label-noise=0.01 caused underfitting: train CD got ~50x worse and test CD
got 3x worse (loss plateaued ~0.35, never converged). Defaults were reset
to OFF so the effect of window diversity alone (50 -> 200 samples/room)
can be isolated first; re-enable regularizers one at a time and weaker
(e.g. tok-subsample 6144 instead of 4096) if memorisation persists.

Data cache layout:
  data/multi_scene_N{n}_s{stride}/{room}/sample_{si:03d}/{da3_tokens,z_star,pts_norm,meta}.pt

Usage:
  python multi_scene_train.py                         # 200 samples/room, no reg.
  python multi_scene_train.py --drop 0.05 --tok-subsample 6144
  python multi_scene_train.py --force-extract
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
import wandb

REPO_ROOT   = Path(__file__).resolve().parents[2]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch.utils.data import Dataset, DataLoader                 # noqa: E402
import itertools                                                 # noqa: E402

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from depth_anything_3.cfg import create_object                             # noqa: E402
from safetensors.torch import load_file                                     # noqa: E402


# ── helpers (shared with multi_sample_train.py) ───────────────────────────────

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
    transform   = torch.linalg.inv(poses_c2w[:, :1])
    normalized  = poses_c2w @ transform
    c2ws        = torch.linalg.inv(normalized)
    median_dist = c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1)
    normalized[..., :3, 3] /= median_dist
    return normalized


def select_tokens(tokens: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or tokens.shape[1] <= max_n:
        return tokens
    idx = torch.linspace(0, tokens.shape[1] - 1, max_n, device=tokens.device).long()
    return tokens[:, idx]


def crop_frustum_world_points(points_world: torch.Tensor, camera_to_world: torch.Tensor,
                               intrinsics: torch.Tensor, image_hw: tuple) -> torch.Tensor:
    H, W = image_hw
    w2c  = torch.linalg.inv(camera_to_world)
    ones = torch.ones(points_world.shape[0], 1, dtype=points_world.dtype)
    pts_cam = (torch.cat([points_world, ones], dim=1) @ w2c.T)[:, :3]
    in_front = pts_cam[:, 2] > 0
    pts_cam  = pts_cam[in_front]
    u = pts_cam[:, 0] / pts_cam[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]
    v = pts_cam[:, 1] / pts_cam[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]
    in_frustum = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return points_world[in_front][in_frustum]


@contextmanager
def deterministic_numpy_default_rng(seed: int):
    original = np.random.default_rng
    np.random.default_rng = lambda s=None: original(seed if s is None else s)
    try:
        yield
    finally:
        np.random.default_rng = original


@torch.no_grad()
def extract_da3_tokens(model, images: torch.Tensor, poses_c2w: torch.Tensor,
                        intrinsics: torch.Tensor, layer_idx: int | list[int],
                        max_tokens: int, device: torch.device,
                        use_poses: bool = True) -> torch.Tensor:
    """DA3 backbone tokens. With use_poses=False the camera token is dropped
    (cam_token=None), i.e. the GT-free path DA3 itself uses when extrinsics are
    not supplied (da3.py forward). Tokens then depend only on the images, so an
    adapter trained on these needs no GT poses at deployment."""
    images_bt = images.unsqueeze(0).to(device)
    K_bt      = intrinsics.unsqueeze(0).to(device)
    images_norm    = imagenet_normalize(images_bt)
    amp_dt = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
              else torch.float16)
    if use_poses:
        poses_bt  = poses_c2w.unsqueeze(0).to(device)
        extrinsics_w2c = normalize_extrinsics(torch.linalg.inv(poses_bt[0]).unsqueeze(0))
        with torch.autocast(device_type=device.type, enabled=False):
            cam_token = model.cam_enc(extrinsics_w2c, K_bt, images_bt.shape[-2:])
    else:
        cam_token = None
    with torch.autocast(device_type=device.type, dtype=amp_dt, enabled=(device.type == "cuda")):
        backbone_out, _ = model.backbone(images_norm, cam_token=cam_token)
    indices = [layer_idx] if isinstance(layer_idx, int) else layer_idx
    parts = []
    for i in indices:
        raw  = backbone_out[i][0].float()
        flat = raw.reshape(raw.shape[0], -1, raw.shape[-1])
        parts.append(select_tokens(flat, max_tokens).reshape(1, -1, raw.shape[-1]))
    return torch.cat(parts, dim=1).cpu()


def world_to_first_camera(pts_world: torch.Tensor,
                           first_pose_c2w: torch.Tensor) -> torch.Tensor:
    w2c  = torch.linalg.inv(first_pose_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_points(pts: torch.Tensor, count: int, seed: int) -> torch.Tensor:
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
    pts   = pts_first_cam.to(device).float()
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
    norm_mode = cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    with deterministic_numpy_default_rng(seed):
        z = model._encode(pointmaps=pts_norm, test=True)["tokens"].float()
    return z.cpu(), pts_norm.cpu()


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg, tokens: torch.Tensor,
                   pts_norm: torch.Tensor, device: torch.device,
                   num_queries: int = 8192, seed: int = 42) -> torch.Tensor:
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
    with torch.cuda.amp.autocast(enabled=(device.type != "cpu"), dtype=amp_dt):
        sol = solver.sample(
            time_grid=T_grid, x_init=x_init, method=method,
            step_size=step_sz, return_intermediates=False,
            images=images, token_mask=None,
            encoder_data=encoder_data, pointmaps=pts_norm.to(device),
        )
    pts3d = sol[-1] if isinstance(sol, list) else sol
    return pts3d.cpu()


def chamfer(x: torch.Tensor, y: torch.Tensor) -> float:
    from pytorch3d.loss import chamfer_distance as cd3
    loss, _ = cd3(x.unsqueeze(0).float(), y.unsqueeze(0).float())
    return float(loss)


def fig_to_wandb(fig, caption: str = ""):
    if wandb.Image is None:
        plt.close(fig)
        return None
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buf).copy(), caption=caption)


# ── per-room extraction ────────────────────────────────────────────────────────

def extract_room(room: str, n_samples: int, stride: int, cfg, replica_root: Path,
                 data_root: Path, da3_model, nova_model, nova_cfg,
                 K_nat: np.ndarray, K_proc: np.ndarray,
                 H_nat: int, W_nat: int, H_proc: int, W_proc: int,
                 device: torch.device, force: bool):
    """Extract and cache n_samples windows from `room`. Returns immediately if cached."""
    room_data = data_root / room
    n_frames  = int(cfg.num_frames)

    room_dir    = replica_root / room
    results_dir = room_dir / "results"
    mesh_ply    = replica_root / f"{room}_mesh.ply"

    K_nat_t  = torch.from_numpy(K_nat)
    K_proc_t = torch.from_numpy(K_proc)

    poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    total_frames  = len(all_frame_ids)

    window_span   = (n_frames - 1) * stride
    max_start_idx = total_frames - 1 - window_span
    assert max_start_idx >= n_samples, \
        f"{room}: not enough frames ({total_frames}) for {n_samples} samples at stride {stride}"

    # Random unique start indices – more intra-room diversity than linspace
    rng_room = np.random.default_rng(abs(hash(room)) % (2**32))
    start_indices = sorted(rng_room.choice(max_start_idx + 1, size=n_samples,
                                            replace=False).tolist())

    needs_extract = [
        si for si in range(n_samples)
        if force or not all(
            (room_data / f"sample_{si:03d}" / f).exists()
            for f in ["da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"]
        )
    ]

    if not needs_extract:
        print(f"  [{room}] All {n_samples} samples cached, skipping extraction.")
        return

    print(f"  [{room}] Extracting {len(needs_extract)}/{n_samples} samples …")
    print(f"  [{room}] Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    for si in needs_extract:
        start_idx  = start_indices[si]
        sample_dir = room_data / f"sample_{si:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        s_frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
        s_poses = torch.from_numpy(
            np.stack([poses_all[fid] for fid in s_frame_ids])
        ).float()

        print(f"    [{room}] sample {si:03d} start={start_idx} frames={s_frame_ids}")

        def load_rgb(fid: int) -> torch.Tensor:
            img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
            img = img.resize((W_proc, H_proc), Image.BILINEAR)
            return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

        images_t   = torch.stack([load_rgb(fid) for fid in s_frame_ids])
        intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames, 1, 1))).float()

        first_c2w = s_poses[0]
        pts_cam_list = []
        for i in range(n_frames):
            frust = crop_frustum_world_points(mesh_pts, s_poses[i], K_nat_t, (H_nat, W_nat))
            pts_cam_list.append(world_to_first_camera(frust, first_c2w))

        all_frust = torch.cat(pts_cam_list, dim=0)
        tok_pts   = sample_points(all_frust, int(cfg.mesh_sample_points),
                                   seed=si + hash(room) % (2**31))
        tok_pts_b = tok_pts.unsqueeze(0)

        nova_seed  = abs(hash(room)) % (2**20) + si * 1009 + start_idx
        z_star, pts_norm = encode_nova3r(nova_model, nova_cfg, tok_pts_b, device, seed=nova_seed)

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

    print(f"  [{room}] Extraction done.")


def training_frame_ids(room: str, n_samples: int, data_root: Path) -> set[int]:
    used = set()
    for si in range(n_samples):
        meta_path = data_root / room / f"sample_{si:03d}" / "meta.pt"
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        used.update(int(fid) for fid in meta["frame_ids"])
    return used


def first_disjoint_window(room: str, n_samples: int, stride: int, cfg,
                          replica_root: Path, data_root: Path) -> tuple[int, list[int]]:
    used_frames = training_frame_ids(room, n_samples, data_root)
    room_dir = replica_root / room
    results_dir = room_dir / "results"
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    n_frames = int(cfg.num_frames)
    max_start_idx = len(all_frame_ids) - 1 - (n_frames - 1) * stride

    for start_idx in range(max_start_idx + 1):
        frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
        if used_frames.isdisjoint(frame_ids):
            return start_idx, frame_ids

    raise RuntimeError(
        f"No stride-{stride} {n_frames}-frame window in {room} is disjoint from "
        f"the {len(used_frames)} training frame IDs."
    )


@torch.no_grad()
def extract_single_window(room: str, start_idx: int, stride: int, cfg,
                          replica_root: Path, out_dir: Path,
                          da3_model, nova_model, nova_cfg,
                          K_nat: np.ndarray, K_proc: np.ndarray,
                          H_nat: int, W_nat: int, H_proc: int, W_proc: int,
                          device: torch.device, force: bool = False):
    expected = ["da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"]
    if not force and all((out_dir / f).exists() for f in expected):
        meta = torch.load(out_dir / "meta.pt", map_location="cpu", weights_only=False)
        print(f"  [{room}] Held-out cache exists: start={meta['start_idx']} frames={meta['frame_ids']}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(cfg.num_frames)
    room_dir = replica_root / room
    results_dir = room_dir / "results"
    mesh_ply = replica_root / f"{room}_mesh.ply"

    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
    frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
    poses_c2w = torch.from_numpy(np.stack([poses_all[fid] for fid in frame_ids])).float()

    print(f"  [{room}] Extracting held-out start={start_idx} frames={frame_ids}")

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((W_proc, H_proc), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

    images = torch.stack([load_rgb(fid) for fid in frame_ids])
    intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames, 1, 1))).float()

    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))
    K_nat_t = torch.from_numpy(K_nat)

    first_c2w = poses_c2w[0]
    pts_cam_list = []
    for i in range(n_frames):
        frust = crop_frustum_world_points(mesh_pts, poses_c2w[i], K_nat_t, (H_nat, W_nat))
        pts_cam_list.append(world_to_first_camera(frust, first_c2w))

    all_frust = torch.cat(pts_cam_list, dim=0)
    tok_pts = sample_points(all_frust, int(cfg.mesh_sample_points), seed=start_idx)
    z_star, pts_norm = encode_nova3r(
        nova_model, nova_cfg, tok_pts.unsqueeze(0), device, seed=start_idx * 1009
    )
    da3_tokens = extract_da3_tokens(
        da3_model, images, poses_c2w, intrinsics,
        layer_idx=int(cfg.da3_source_layer_index),
        max_tokens=int(cfg.da3_max_source_tokens),
        device=device,
    )

    torch.save(da3_tokens, out_dir / "da3_tokens.pt")
    torch.save(z_star, out_dir / "z_star.pt")
    torch.save(pts_norm, out_dir / "pts_norm.pt")
    torch.save({
        "frame_ids": frame_ids,
        "poses_c2w": poses_c2w,
        "start_idx": start_idx,
        "stride": stride,
        "room": room,
        "split": "heldout_disjoint",
    }, out_dir / "meta.pt")


def load_room_samples(room: str, n_samples: int, data_root: Path):
    """Load z_star, pts_norm, meta for a room into RAM. DA3 tokens are NOT loaded
    here — they are read lazily per-sample via MultiSceneDataset to avoid OOM."""
    zstar_list, pnorm_list, meta_list = [], [], []
    room_data = data_root / room
    for si in range(n_samples):
        d = room_data / f"sample_{si:03d}"
        zstar_list.append(torch.load(d / "z_star.pt",   map_location="cpu", weights_only=True))
        pnorm_list.append(torch.load(d / "pts_norm.pt", map_location="cpu", weights_only=True))
        meta_list.append( torch.load(d / "meta.pt",     map_location="cpu", weights_only=False))
    return (torch.cat(zstar_list, dim=0),
            torch.cat(pnorm_list, dim=0),
            meta_list)


class MultiSceneDataset(Dataset):
    """Loads DA3 tokens lazily from disk; keeps z_star in RAM.

    Each call to __getitem__ reads one da3_tokens.pt file (~68 MB) and
    optionally subsamples tokens for augmentation. This avoids loading the
    full dataset (~95 GB) into memory at once.
    """

    def __init__(self, sample_dirs: list[Path], zstar: torch.Tensor,
                 tok_subsample: int = 0):
        self.sample_dirs  = sample_dirs          # list of Path, one per sample
        self.zstar        = zstar                # (N, 768, 128) pre-loaded in RAM
        self.tok_subsample = tok_subsample

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        da3 = torch.load(self.sample_dirs[idx] / "da3_tokens.pt",
                         map_location="cpu", weights_only=True)[0]  # (N_tok, D)
        if self.tok_subsample > 0 and self.tok_subsample < da3.shape[0]:
            perm = torch.randperm(da3.shape[0])[:self.tok_subsample]
            da3  = da3[perm]
        return da3, self.zstar[idx]


# ── evaluation on one room ────────────────────────────────────────────────────

def evaluate_room(room: str, sample_dirs: list, zstar: torch.Tensor,
                  pnorm: torch.Tensor, meta_list: list,
                  adapter, nova_model, nova_cfg, device: torch.device,
                  num_queries: int, batch_size: int, step: int, tag: str):
    N = zstar.shape[0]
    adapter.eval()

    # Load DA3 tokens from disk in batches to avoid OOM
    pred_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_dirs = sample_dirs[i:i+batch_size]
            da3_b = torch.cat([
                torch.load(d / "da3_tokens.pt", map_location="cpu", weights_only=True)
                for d in batch_dirs
            ], dim=0).to(device)
            pred_list.append(adapter(da3_b).cpu())
    pred_all = torch.cat(pred_list, dim=0)

    mse_per = F.mse_loss(pred_all.float(), zstar.float(),
                          reduction="none").mean(dim=[1, 2])

    cd_pg_list, cd_ai_list, cd_pi_list = [], [], []
    vis_limit = min(N, 4)

    for si in range(N):
        z_i  = zstar[si:si+1]
        pr_i = pred_all[si:si+1]
        pn_i = pnorm[si:si+1]

        gt_dec = decode_tokens(nova_model, nova_cfg, z_i,  pn_i, device,
                                num_queries, seed=42+si)[0].numpy()
        pr_dec = decode_tokens(nova_model, nova_cfg, pr_i, pn_i, device,
                                num_queries, seed=42+si)[0].numpy()
        in_np  = pnorm[si].numpy()

        cd_pg = chamfer(torch.from_numpy(pr_dec), torch.from_numpy(gt_dec))
        cd_ai = chamfer(torch.from_numpy(gt_dec), torch.from_numpy(in_np))
        cd_pi = chamfer(torch.from_numpy(pr_dec), torch.from_numpy(in_np))
        cd_pg_list.append(cd_pg)
        cd_ai_list.append(cd_ai)
        cd_pi_list.append(cd_pi)

        print(f"    [{room}] sample {si:03d}: MSE={float(mse_per[si]):.4e}  "
              f"CD(pg)={cd_pg:.4f}  CD(ai)={cd_ai:.4f}")

        wandb.log({
            f"eval_{tag}/{room}/sample{si:03d}/mse":   float(mse_per[si]),
            f"eval_{tag}/{room}/sample{si:03d}/cd_pg": cd_pg,
            f"eval_{tag}/{room}/sample{si:03d}/cd_ai": cd_ai,
        }, step=step)

        if si < vis_limit:
            fig_c, axs = plt.subplots(1, 3, figsize=(24, 6),
                                       subplot_kw={"projection": "3d"})
            for ax_c, pts_c, cmap_c, ttl_c in zip(
                axs,
                [in_np,   gt_dec,  pr_dec],
                ["cividis", "viridis", "plasma"],
                ["Input", f"GT decoded  CD_ae={cd_ai:.4f}",
                 f"Pred decoded  CD={cd_pg:.4f}"],
            ):
                clr  = pts_c[:, 1]
                nn   = Normalize(np.percentile(clr, 2), np.percentile(clr, 98))
                cols = plt.get_cmap(cmap_c)(nn(clr))
                stp  = max(1, len(pts_c) // 10_000)
                ax_c.scatter(pts_c[::stp, 0], pts_c[::stp, 2], pts_c[::stp, 1],
                             c=cols[::stp], s=1.0, alpha=0.75)
                ax_c.set_title(ttl_c, fontsize=9, fontweight="bold")
                ax_c.view_init(elev=25, azim=-60)
            fids = meta_list[si]["frame_ids"]
            fig_c.suptitle(f"[{room}] sample {si:03d}  frames {fids[0]}–{fids[-1]}  "
                            f"MSE={float(mse_per[si]):.2e}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            wb_img = fig_to_wandb(fig_c, f"{room} {si:03d}")
            if wb_img is not None:
                wandb.log({
                    f"pointclouds/{tag}_{room}_{si:03d}": wb_img
                }, step=step)

    cd_pg_mean = float(np.mean(cd_pg_list))
    cd_ai_mean = float(np.mean(cd_ai_list))
    mse_mean   = float(mse_per.mean())
    wandb.log({
        f"eval_{tag}/{room}/cd_pg_mean": cd_pg_mean,
        f"eval_{tag}/{room}/cd_ai_mean": cd_ai_mean,
        f"eval_{tag}/{room}/mse_mean":   mse_mean,
    }, step=step)
    print(f"  [{room}] {tag}  MSE={mse_mean:.4e}  CD(pg)={cd_pg_mean:.4f}  CD(ai)={cd_ai_mean:.4f}")
    return mse_mean, cd_pg_mean


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-room", type=int, default=None,
                        help="8-frame windows per room (default: cfg.samples_per_room or 200)")
    parser.add_argument("--stride",        type=int,   default=20)
    parser.add_argument("--steps",         type=int,   default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    parser.add_argument("--num-queries",   type=int,   default=None)
    parser.add_argument("--eval-samples",  type=int,   default=None,
                        help="Limit decoded eval to the first N cached samples per room")
    parser.add_argument("--heldout-eval", action="store_true",
                        help="Evaluate one room0 window whose frames are disjoint from the training cache")
    parser.add_argument("--heldout-room", default=None,
                        help="Room for --heldout-eval (default: first train room)")
    parser.add_argument("--heldout-start-idx", type=int, default=None,
                        help="Override held-out start index; must be frame-disjoint from training")
    parser.add_argument("--drop",          type=float, default=0.0,
                        help="Q-Former dropout probability (default 0.0 — off; try 0.1)")
    parser.add_argument("--tok-subsample", type=int,   default=0,
                        help="Random DA3 tokens to sample per step (default 0=all; try 6144)")
    parser.add_argument("--label-noise",   type=float, default=0.0,
                        help="Std of Gaussian noise added to z_star targets (default 0.0 — off)")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)
    stride = args.stride

    train_rooms = list(cfg.get("train_rooms", [
        "room0", "room1", "office0", "office1", "office2", "office3", "office4"
    ]))
    test_rooms = list(cfg.get("test_rooms", ["room2"]))
    all_rooms  = train_rooms + [r for r in test_rooms if r not in train_rooms]

    n_per_room   = args.samples_per_room or int(cfg.get("samples_per_room", 200))
    n_train      = n_per_room * len(train_rooms)
    steps        = args.steps or int(cfg.get("steps", max(30000, n_train * 50)))
    batch_size   = args.batch_size or int(cfg.get("batch_size", min(n_train, 16)))
    best_ckpt_every_samples = int(cfg.get("best_checkpoint_every_samples", 0))
    num_queries  = args.num_queries or int(cfg.get("eval_num_queries", 8192))
    eval_samples = args.eval_samples
    if eval_samples is None and cfg.get("eval_samples", None) is not None:
        eval_samples = int(cfg.eval_samples)
    heldout_eval = args.heldout_eval or bool(cfg.get("heldout_eval", False))
    heldout_room_cfg = cfg.get("heldout_room", None)
    heldout_room = args.heldout_room or (str(heldout_room_cfg) if heldout_room_cfg is not None else None)
    heldout_start_idx = args.heldout_start_idx
    if heldout_start_idx is None and cfg.get("heldout_start_idx", None) is not None:
        heldout_start_idx = int(cfg.heldout_start_idx)
    drop         = args.drop
    tok_sub      = args.tok_subsample   # 0 → use all tokens
    label_noise  = args.label_noise

    EXP_DIR   = REPO_ROOT / "experiments" / "overfit_8frames"
    data_root = EXP_DIR / "data" / f"multi_scene_N{n_per_room}_s{stride}"
    heldout_root = EXP_DIR / "data" / f"heldout_disjoint_N{n_per_room}_s{stride}"
    ckpt_dir  = EXP_DIR / "checkpoints" / f"multi_scene_N{n_per_room}_s{stride}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    replica_root = Path(str(cfg.replica_root))
    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    depth_scale  = float(cam_p["scale"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                       [0., cam_p["fy"], cam_p["cy"]],
                       [0., 0., 1.]], dtype=np.float32)
    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    K_proc = K_nat.copy()
    K_proc[0] *= W_proc / W_nat
    K_proc[1] *= H_proc / H_nat

    print(f"[multi_scene] Train rooms: {train_rooms}")
    print(f"[multi_scene] Test  rooms: {test_rooms}")
    print(f"[multi_scene] {n_per_room} samples/room  →  {n_train} train samples")
    print(f"[multi_scene] steps={steps}  batch_size={batch_size}")

    # ── extraction ────────────────────────────────────────────────────────────
    needs_any = any(
        not all(
            (data_root / room / f"sample_{si:03d}" / f).exists()
            for f in ["da3_tokens.pt", "z_star.pt", "pts_norm.pt", "meta.pt"]
        )
        for room in all_rooms
        for si in range(n_per_room)
    ) or args.force_extract

    if needs_any:
        print("[multi_scene] Loading models for extraction …")
        nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
        nova_model.eval()
        for p in nova_model.parameters(): p.requires_grad_(False)
        OmegaConf.set_struct(nova_cfg, False)
        da3_model = load_da3_model(str(cfg.da3_model), device)

        for room in all_rooms:
            print(f"\n[multi_scene] ── {room} ──")
            extract_room(
                room=room, n_samples=n_per_room, stride=stride, cfg=cfg,
                replica_root=replica_root, data_root=data_root,
                da3_model=da3_model, nova_model=nova_model, nova_cfg=nova_cfg,
                K_nat=K_nat, K_proc=K_proc,
                H_nat=H_nat, W_nat=W_nat, H_proc=H_proc, W_proc=W_proc,
                device=device, force=args.force_extract,
            )

        del nova_model, da3_model
        torch.cuda.empty_cache()
        print("[multi_scene] All extraction done.")

    # ── load train data (z_star into RAM; DA3 tokens lazy via Dataset) ───────
    print("[multi_scene] Loading train data (z_star only, DA3 tokens lazy) …")
    train_dirs, train_zstar_list = [], []
    for room in train_rooms:
        zstar, _, _ = load_room_samples(room, n_per_room, data_root)
        train_zstar_list.append(zstar)
        for si in range(n_per_room):
            train_dirs.append(data_root / room / f"sample_{si:03d}")
        print(f"  {room}: {n_per_room} samples")

    train_zstar = torch.cat(train_zstar_list, dim=0)   # (n_train, 768, 128)
    train_dataset = MultiSceneDataset(train_dirs, train_zstar, tok_subsample=tok_sub)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=4, pin_memory=True,
                               persistent_workers=True)
    train_iter = itertools.cycle(train_loader)
    print(f"[multi_scene] Dataset: {len(train_dataset)} samples, z_star RAM: "
          f"{train_zstar.numel()*4/1e9:.2f} GB")

    # ── load test data (z_star + pnorm into RAM; DA3 tokens lazy) ────────────
    print("[multi_scene] Loading test data …")
    test_data = {}
    for room in test_rooms:
        zstar, pnorm, meta = load_room_samples(room, n_per_room, data_root)
        test_dirs = [data_root / room / f"sample_{si:03d}" for si in range(n_per_room)]
        test_data[room] = (test_dirs, zstar, pnorm, meta)
        print(f"  {room}: {n_per_room} samples (test)")

    # ── adapter ───────────────────────────────────────────────────────────────
    adapter = DA3ToNOVA3RAlignment(
        source_dim    = int(cfg.source_dim),
        hidden_dim    = int(cfg.hidden_dim),
        target_tokens = int(cfg.target_tokens),
        target_dim    = int(cfg.target_dim),
        depth         = int(cfg.depth),
        num_heads     = int(cfg.num_heads),
        drop          = drop,
    ).to(device)
    n_params  = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(adapter.parameters(),
                                   lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    # cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_prefix = str(cfg.get(
        "wandb_run_name",
        f"multi_scene_N{n_per_room}x{len(train_rooms)}rooms_drop{drop}_toksub{tok_sub}",
    ))
    run_name = f"{run_prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    run = wandb.init(
        project = str(cfg.wandb_project),
        name    = run_name,
        config  = {
            "mode":            "multi_scene_reg",
            "train_rooms":     train_rooms,
            "test_rooms":      test_rooms,
            "n_per_room":      n_per_room,
            "n_train_total":   n_train,
            "stride":          stride,
            "steps":           steps,
            "batch_size":      batch_size,
            "best_checkpoint_every_samples": best_ckpt_every_samples,
            "lr":              float(cfg.lr),
            "weight_decay":    float(cfg.weight_decay),
            "hidden_dim":      int(cfg.hidden_dim),
            "depth":           int(cfg.depth),
            "num_heads":       int(cfg.num_heads),
            "target_tokens":   int(cfg.target_tokens),
            "target_dim":      int(cfg.target_dim),
            "n_params":        n_params,
            "drop":            drop,
            "tok_subsample":   tok_sub,
            "label_noise":     label_noise,
            "eval_samples":      eval_samples,
            "eval_num_queries":  num_queries,
            "heldout_eval":      heldout_eval,
            "heldout_room":      heldout_room,
            "heldout_start_idx": heldout_start_idx,
        },
        tags = ["multi-scene", "generalization", "regularized"],
    )
    print(f"[wandb] {run.url}")

    # ── training ─────────────────────────────────────────────────────────────
    print(f"[multi_scene] Training {steps} steps, bs={batch_size} …")
    adapter.train()
    t0 = time.time()
    samples_seen = 0
    next_best_ckpt_samples = best_ckpt_every_samples if best_ckpt_every_samples > 0 else None
    best_loss = float("inf")

    def save_adapter_checkpoint(path: Path, step: int, loss_value: float) -> None:
        torch.save({
            "adapter_state_dict": adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": OmegaConf.to_container(cfg),
            "train_rooms": train_rooms,
            "test_rooms": test_rooms,
            "n_per_room": n_per_room,
            "stride": stride,
            "steps": steps,
            "step": step,
            "samples_seen": samples_seen,
            "loss": loss_value,
        }, path)

    for step in range(1, steps + 1):
        da3_b, zstar_b = next(train_iter)
        batch_samples = int(da3_b.shape[0])
        da3_b   = da3_b.to(device)    # (bs, N_tok, D)  – token subsample done in Dataset
        zstar_b = zstar_b.to(device)  # (bs, 768, 128)

        # Label noise on z_star targets
        if label_noise > 0:
            zstar_b = zstar_b + torch.randn_like(zstar_b) * label_noise

        pred = adapter(da3_b)
        loss = F.mse_loss(pred, zstar_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        samples_seen += batch_samples

        if step % int(cfg.log_every) == 0 or step == 1 or step == steps:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:6d}/{steps}  loss={loss.item():.6f}  lr={lr_now:.2e}  "
                  f"samples={samples_seen}  t={time.time()-t0:.1f}s")
            wandb.log({"train/loss": loss.item(), "train/lr": lr_now,
                       "train/samples_seen": samples_seen}, step=step)

        if next_best_ckpt_samples is not None and samples_seen >= next_best_ckpt_samples:
            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                ckpt_path = ckpt_dir / "adapter_best.pt"
                save_adapter_checkpoint(ckpt_path, step, loss_value)
                print(f"[multi_scene] Saved best checkpoint: {ckpt_path} "
                      f"(loss={best_loss:.6f}, step={step}, samples_seen={samples_seen})")
                wandb.log({"train/best_loss": best_loss}, step=step)
                wandb.save(str(ckpt_path), base_path=str(REPO_ROOT), policy="now")
            while next_best_ckpt_samples <= samples_seen:
                next_best_ckpt_samples += best_ckpt_every_samples

    final_loss = loss.item()
    print(f"[multi_scene] Training done. Final loss: {final_loss:.6f}")

    save_adapter_checkpoint(ckpt_dir / "adapter_final.pt", steps, final_loss)
    wandb.log({"train/final_loss": final_loss}, step=steps)

    # ── evaluate ─────────────────────────────────────────────────────────────
    print("[multi_scene] Loading NOVA3R for evaluation …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters(): p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # train rooms (check for overfitting / lower bound)
    print("\n[multi_scene] Evaluating train rooms …")
    train_eval_data = {}
    for room in train_rooms[:2]:   # first 2 to keep eval fast
        zstar, pnorm, meta = load_room_samples(room, n_per_room, data_root)
        sample_dirs = [data_root / room / f"sample_{si:03d}" for si in range(n_per_room)]
        if eval_samples is not None:
            eval_n = min(eval_samples, len(sample_dirs))
            sample_dirs = sample_dirs[:eval_n]
            zstar = zstar[:eval_n]
            pnorm = pnorm[:eval_n]
            meta = meta[:eval_n]
        mse, cd = evaluate_room(room, sample_dirs, zstar, pnorm, meta, adapter,
                                 nova_model, nova_cfg, device,
                                 num_queries, batch_size, steps, tag="train")
        train_eval_data[room] = (mse, cd)

    # test rooms (generalization)
    print("\n[multi_scene] Evaluating test rooms …")
    test_eval_data = {}
    if heldout_eval:
        print("  Skipping cached test samples because --heldout-eval provides the validation window.")
    else:
        for room, (sample_dirs, zstar, pnorm, meta) in test_data.items():
            if eval_samples is not None:
                eval_n = min(eval_samples, len(sample_dirs))
                sample_dirs = sample_dirs[:eval_n]
                zstar = zstar[:eval_n]
                pnorm = pnorm[:eval_n]
                meta = meta[:eval_n]
            mse, cd = evaluate_room(room, sample_dirs, zstar, pnorm, meta, adapter,
                                     nova_model, nova_cfg, device,
                                     num_queries, batch_size, steps, tag="test")
            test_eval_data[room] = (mse, cd)

    heldout_eval_data = {}
    if heldout_eval:
        heldout_room = heldout_room or train_rooms[0]
        if heldout_start_idx is None:
            heldout_start, heldout_frame_ids = first_disjoint_window(
                heldout_room, n_per_room, stride, cfg, replica_root, data_root
            )
        else:
            heldout_start = heldout_start_idx
            used_frames = training_frame_ids(heldout_room, n_per_room, data_root)
            room_dir = replica_root / heldout_room
            frame_files = sorted((room_dir / "results").glob("frame*.jpg"))
            all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
            heldout_frame_ids = [
                all_frame_ids[heldout_start + i * stride]
                for i in range(int(cfg.num_frames))
            ]
            overlap = sorted(used_frames.intersection(heldout_frame_ids))
            if overlap:
                raise ValueError(
                    f"--heldout-start-idx {heldout_start} overlaps training frames: {overlap}"
                )

        heldout_dir = heldout_root / heldout_room / f"start_{heldout_start:06d}"
        print("\n[multi_scene] Evaluating held-out disjoint window …")
        print(f"  [{heldout_room}] start={heldout_start} frames={heldout_frame_ids}")
        print("  Loading DA3 for held-out extraction …")
        da3_model = load_da3_model(str(cfg.da3_model), device)
        extract_single_window(
            room=heldout_room, start_idx=heldout_start, stride=stride, cfg=cfg,
            replica_root=replica_root, out_dir=heldout_dir,
            da3_model=da3_model, nova_model=nova_model, nova_cfg=nova_cfg,
            K_nat=K_nat, K_proc=K_proc,
            H_nat=H_nat, W_nat=W_nat, H_proc=H_proc, W_proc=W_proc,
            device=device, force=args.force_extract,
        )
        del da3_model
        zstar_h = torch.load(heldout_dir / "z_star.pt", map_location="cpu", weights_only=True)
        pnorm_h = torch.load(heldout_dir / "pts_norm.pt", map_location="cpu", weights_only=True)
        meta_h = torch.load(heldout_dir / "meta.pt", map_location="cpu", weights_only=False)
        mse, cd = evaluate_room(
            heldout_room, [heldout_dir], zstar_h, pnorm_h, [meta_h], adapter,
            nova_model, nova_cfg, device, num_queries, batch_size, steps,
            tag="heldout_disjoint",
        )
        heldout_eval_data[heldout_room] = (mse, cd, heldout_start, heldout_frame_ids)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MULTI-SCENE  {len(train_rooms)} train rooms × {n_per_room} samples")
    print(f"{'='*70}")
    print(f"  {'room':>12}  {'split':>6}  {'MSE':>12}  {'CD pred↔gt':>12}")
    for room, (mse, cd) in train_eval_data.items():
        print(f"  {room:>12}  {'train':>6}  {mse:12.4e}  {cd:12.6f}")
    for room, (mse, cd) in test_eval_data.items():
        print(f"  {room:>12}  {'TEST':>6}  {mse:12.4e}  {cd:12.6f}  ← generalization")
    for room, (mse, cd, start_idx, frame_ids) in heldout_eval_data.items():
        print(f"  {room:>12}  {'HELD':>6}  {mse:12.4e}  {cd:12.6f}  "
              f"start={start_idx} frames={frame_ids}")
    print(f"{'='*70}")
    print(f"  W&B: {run.url}")
    print(f"{'='*70}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
