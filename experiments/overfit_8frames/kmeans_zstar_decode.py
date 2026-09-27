#!/usr/bin/env python3
"""
Encode the start=0 stride-20 window with 10 different random subsampling seeds,
cluster the resulting 7680 z_star tokens (10 × 768) into 768 centroids via
k-means, decode the consensus z_star, and compare to the disjoint validation
slice in Rerun.

Usage:
  ! python experiments/overfit_8frames/kmeans_zstar_decode.py --replica-root datasets/replica
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from sklearn.cluster import KMeans

REPO_ROOT   = Path(__file__).resolve().parents[2]
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"
DA3_SRC     = REPO_ROOT / "da3" / "src"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from demo_nova3r import load_model as load_nova3r_model          # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper        # noqa: E402
from nova3r.flow_matching.solver import ODESolver                # noqa: E402
from nova3r.inference import normalize_input, amp_dtype_mapping  # noqa: E402
from vc3r.replica import crop_visible_world_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",          type=Path,
                   default=Path(__file__).parent / "config.yaml")
    p.add_argument("--replica-root",    type=Path, default=None)
    p.add_argument("--num-seeds",       type=int,  default=10)
    p.add_argument("--depth-tolerance", type=float, default=0.05)
    p.add_argument("--num-queries",     type=int,  default=8192)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--rrd-out",         type=Path,
                   default=REPO_ROOT / "outputs" / "kmeans_zstar" / "kmeans_vs_val.rrd")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def world_to_first_camera(pts_world: torch.Tensor,
                           first_c2w: torch.Tensor) -> torch.Tensor:
    w2c  = torch.linalg.inv(first_c2w)
    ones = torch.ones(*pts_world.shape[:-1], 1, dtype=pts_world.dtype)
    return (torch.cat([pts_world, ones], dim=-1) @ w2c.T)[..., :3]


def sample_pts(pool: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if pool.shape[0] >= k:
        idx = torch.randperm(pool.shape[0], generator=gen)[:k]
    else:
        pad = torch.randint(pool.shape[0], (k - pool.shape[0],), generator=gen)
        idx = torch.cat([torch.arange(pool.shape[0]), pad])
    return pool[idx]


@torch.no_grad()
def encode(model, cfg, pts: torch.Tensor, device: torch.device):
    """Returns (z_star (768,128), pts_norm (1,k,3), norm_factor float)."""
    pts   = pts.unsqueeze(0).to(device).float()   # (1, k, 3)
    valid = torch.ones(pts.shape[:2], dtype=torch.bool, device=device)
    norm_mode = cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    pts_norm, _ = normalize_input(pts, valid, pts, valid, mode=norm_mode)
    tokens = model._encode(pointmaps=pts_norm, test=True)["tokens"].float().cpu()
    # norm_factor: median distance of raw pts from origin (before normalisation)
    norm_factor = float(pts.cpu()[0].norm(dim=-1).median().clamp(0.01, 100.0))
    return tokens[0], pts_norm.cpu(), norm_factor


@torch.no_grad()
def decode_tokens(nova_model, nova_cfg,
                  tokens: torch.Tensor,   # (1, 768, 128)
                  pts_norm: torch.Tensor, # (1, k, 3)
                  device: torch.device,
                  num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    encoder_data = {"tokens": tokens.to(device)}
    images  = torch.zeros(1, 1, 3, 1, 1, device=device)
    x_init  = torch.rand(1, num_queries, 3, device=device) * 2 - 1
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
    return (sol[-1] if isinstance(sol, list) else sol)[0].cpu().float().numpy()


def norm_to_world(pts_norm_np: np.ndarray, norm_factor: float,
                  c2w: np.ndarray) -> np.ndarray:
    pts_cam = pts_norm_np / 3.0 * norm_factor
    ones    = np.ones((len(pts_cam), 1), dtype=np.float32)
    return (c2w @ np.hstack([pts_cam, ones]).T).T[:, :3].astype(np.float32)


def main() -> None:
    args   = parse_args()
    cfg    = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    device = torch.device(args.device)

    replica_root = (args.replica_root if args.replica_root is not None
                    else Path(str(cfg.replica_root)))
    room_dir    = replica_root / cfg.room
    results_dir = room_dir / "results"
    mesh_ply    = replica_root / f"{cfg.room}_mesh.ply"

    stride   = int(cfg.frame_stride)
    n_frames = int(cfg.num_frames)
    k        = int(cfg.mesh_sample_points)

    with (replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    H_nat = int(cam["h"]); W_nat = int(cam["w"])
    depth_scale = float(cam["scale"])
    K_native = torch.tensor([
        [cam["fx"], 0., cam["cx"]],
        [0., cam["fy"], cam["cy"]],
        [0., 0., 1.]], dtype=torch.float32)

    poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    frame_ids   = [int(p.stem.replace("frame", "")) for p in frame_files]
    sampled_ids = [frame_ids[i * stride] for i in range(n_frames)]
    sampled_poses = torch.from_numpy(
        np.stack([poses_all[fid] for fid in sampled_ids])).float()
    first_c2w = sampled_poses[0]

    print(f"Window: room={cfg.room}  stride={stride}  frames {sampled_ids[0]}–{sampled_ids[-1]}")
    print(f"Num seeds: {args.num_seeds}  k={k}")

    def load_depth(fid: int) -> torch.Tensor:
        from PIL import Image
        d = Image.open(results_dir / f"depth{fid:06d}.png")
        return torch.from_numpy(np.asarray(d, dtype=np.float32) / depth_scale)

    print("Loading mesh …")
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    print("Building visible point pool …")
    all_cam_pts: list[torch.Tensor] = []
    for i, fid in enumerate(sampled_ids):
        depth = load_depth(fid)
        vis = crop_visible_world_points(
            points_world=mesh_pts, camera_to_world=sampled_poses[i],
            intrinsics=K_native, depth=depth,
            depth_tolerance=args.depth_tolerance)
        pts_cam = world_to_first_camera(vis["points_world"], first_c2w)
        all_cam_pts.append(pts_cam)
        print(f"  frame {fid:06d}: {pts_cam.shape[0]:>6} pts")

    all_cam = torch.cat(all_cam_pts, dim=0)
    base_gen = torch.Generator().manual_seed(0xDEADBEEF)
    shuffle  = torch.randperm(all_cam.shape[0], generator=base_gen)
    all_cam  = all_cam[shuffle]
    train_pool = all_cam[:-k]
    val_pool   = all_cam[-k:]
    print(f"Total pts: {all_cam.shape[0]:,}  train pool: {len(train_pool):,}  val: {len(val_pool):,}")

    print("Loading NOVA3R …")
    ckpt = str(REPO_ROOT / "nova3r" / "checkpoints" / "scene_ae" / "checkpoint-last.pth")
    nova_model, nova_cfg = load_nova3r_model(ckpt, str(device))
    nova_model.eval()
    for p in nova_model.parameters():
        p.requires_grad_(False)
    OmegaConf.set_struct(nova_cfg, False)

    # ── encode training seeds ──────────────────────────────────────────────────
    print(f"Encoding {args.num_seeds} training seeds …")
    all_tokens: list[torch.Tensor] = []  # list of (768, 128)
    seed0_pts_norm, seed0_norm_factor = None, None

    for seed in range(args.num_seeds):
        pts = sample_pts(train_pool, k, seed=seed)
        z, pnorm, nf = encode(nova_model, nova_cfg, pts, device)
        all_tokens.append(z)
        if seed == 0:
            seed0_pts_norm  = pnorm   # (1, k, 3)
            seed0_norm_factor = nf
        print(f"  seed={seed:02d}  l2={z.norm():.2f}  norm_factor={nf:.3f}m")

    # ── encode val ─────────────────────────────────────────────────────────────
    print("Encoding val slice …")
    val_z, val_pts_norm, val_norm_factor = encode(nova_model, nova_cfg, val_pool, device)
    print(f"  val  l2={val_z.norm():.2f}  norm_factor={val_norm_factor:.3f}m")

    # ── k-means on all training tokens ────────────────────────────────────────
    stacked = torch.stack(all_tokens)           # (S, 768, 128)
    S, T, D = stacked.shape
    flat    = stacked.reshape(S * T, D).numpy() # (S*768, 128)
    print(f"Running k-means: {flat.shape[0]} tokens → {T} clusters …")
    km = KMeans(n_clusters=T, random_state=0, n_init=5, max_iter=300, verbose=0)
    km.fit(flat)
    centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32))  # (768, 128)
    kmeans_z = centroids.unsqueeze(0)  # (1, 768, 128)
    print(f"  centroid l2 mean: {centroids.norm(dim=-1).mean():.2f}")

    # ── decode all three ──────────────────────────────────────────────────────
    print("Decoding k-means z_star …")
    km_norm  = decode_tokens(nova_model, nova_cfg, kmeans_z, seed0_pts_norm,
                             device, args.num_queries, seed=args.seed)
    print("Decoding seed-0 z_star …")
    s0_norm  = decode_tokens(nova_model, nova_cfg, all_tokens[0].unsqueeze(0),
                             seed0_pts_norm, device, args.num_queries, seed=args.seed)
    print("Decoding val z_star …")
    val_norm = decode_tokens(nova_model, nova_cfg, val_z.unsqueeze(0), val_pts_norm,
                             device, args.num_queries, seed=args.seed)

    c2w_np = first_c2w.numpy()
    km_world  = norm_to_world(km_norm,  seed0_norm_factor, c2w_np)
    s0_world  = norm_to_world(s0_norm,  seed0_norm_factor, c2w_np)
    val_world = norm_to_world(val_norm, val_norm_factor,   c2w_np)

    # input pts for reference
    input_train = seed0_pts_norm[0].numpy() / 3.0 * seed0_norm_factor
    input_val   = val_pts_norm[0].numpy()   / 3.0 * val_norm_factor
    ones = np.ones((k, 1), dtype=np.float32)
    train_world_in = (c2w_np @ np.hstack([input_train, ones]).T).T[:, :3]
    val_world_in   = (c2w_np @ np.hstack([input_val,   ones]).T).T[:, :3]

    print(f"[centroid] k-means:  {km_world.mean(0).round(3)}")
    print(f"[centroid] seed-0:   {s0_world.mean(0).round(3)}")
    print(f"[centroid] val:      {val_world.mean(0).round(3)}")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    import rerun as rr
    args.rrd_out.parent.mkdir(parents=True, exist_ok=True)
    rr.init("kmeans_zstar", spawn=False)
    rr.save(str(args.rrd_out))
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    BLUE   = np.array([ 50, 150, 255], dtype=np.uint8)
    GREEN  = np.array([ 60, 220,  90], dtype=np.uint8)
    GREY   = np.array([160, 160, 160], dtype=np.uint8)
    ORANGE = np.array([255, 140,  30], dtype=np.uint8)
    PINK   = np.array([220,  80, 200], dtype=np.uint8)

    def sub(pts: np.ndarray, n: int = 100_000) -> np.ndarray:
        if len(pts) <= n:
            return pts
        return pts[np.random.default_rng(0).choice(len(pts), n, replace=False)]

    rr.log("world/kmeans_zstar",
           rr.Points3D(sub(km_world),
                       colors=np.tile(BLUE,  (min(len(km_world),  100_000), 1)),
                       radii=0.012))
    rr.log("world/val_zstar",
           rr.Points3D(sub(val_world),
                       colors=np.tile(GREEN, (min(len(val_world), 100_000), 1)),
                       radii=0.012))
    rr.log("world/seed0_zstar",
           rr.Points3D(sub(s0_world),
                       colors=np.tile(GREY,  (min(len(s0_world),  100_000), 1)),
                       radii=0.010))
    rr.log("world/input_train",
           rr.Points3D(sub(train_world_in, n=30_000),
                       colors=np.tile(ORANGE, (min(len(train_world_in), 30_000), 1)),
                       radii=0.007))
    rr.log("world/input_val",
           rr.Points3D(sub(val_world_in, n=30_000),
                       colors=np.tile(PINK,   (min(len(val_world_in), 30_000), 1)),
                       radii=0.007))

    rr.log("legend", rr.TextDocument(
        "# k-means z_star vs val slice\n\n"
        f"- **Blue** `world/kmeans_zstar`: k-means({T}) on {S}×{T} training tokens, decoded\n"
        f"- **Green** `world/val_zstar`: disjoint val slice z_star, decoded\n"
        f"- **Grey** `world/seed0_zstar`: single seed-0 z_star, decoded\n"
        f"- **Orange** `world/input_train`: seed-0 training input pts\n"
        f"- **Pink** `world/input_val`: val input pts (disjoint)\n\n"
        f"Window: {cfg.room}  stride={stride}  frames {sampled_ids[0]}–{sampled_ids[-1]}\n\n"
        f"Seeds: {args.num_seeds}  |  k={k}  |  KMeans clusters={T}\n\n"
        f"Centroids: k-means {km_world.mean(0).round(3)}  "
        f"val {val_world.mean(0).round(3)}",
        media_type=rr.MediaType.MARKDOWN,
    ))

    print(f"\n[rerun] saved → {args.rrd_out}")
    print(f"  open with:  rerun {args.rrd_out}")


if __name__ == "__main__":
    main()
