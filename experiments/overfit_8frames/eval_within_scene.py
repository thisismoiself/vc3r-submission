#!/usr/bin/env python3
"""
Within-scene generalisation diagnostic.

Tests whether the trained adapter works on 8-frame windows from the TRAINING
rooms that were NOT in the training set (held-out intra-scene windows), to
separate two failure modes:

  A) Memorisation of exact training windows only
     → held-out same-scene windows perform as badly as room2 (unseen scene)

  B) Scene-level memorisation (learns room structure, fails on new scenes)
     → held-out same-scene windows perform well, room2 fails

Usage:
  python eval_within_scene.py                   # 50 held-out samples/room, rooms 0+1
  python eval_within_scene.py --n-eval 20       # faster, fewer samples
  python eval_within_scene.py --rooms room0 room1 office0
  python eval_within_scene.py --split train-seen --rooms room0 --n-eval 8

Results are printed as a summary table for easy comparison with training and test results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

# ── path setup (mirrors multi_scene_train.py) ──────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
EXP_DIR     = REPO_ROOT / "experiments" / "overfit_8frames"
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
NOVA3R_3P   = NOVA3R_ROOT / "third_party"

for _p in [str(NOVA3R_3P), str(NOVA3R_ROOT), str(DA3_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import all helper functions from multi_scene_train (safe: protected by __name__ guard)
from multi_scene_train import (                     # noqa: E402
    load_da3_model, encode_nova3r, extract_da3_tokens,
    decode_tokens, chamfer,
    crop_frustum_world_points, world_to_first_camera, sample_points,
)
from demo_nova3r import load_model as load_nova3r_model   # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402


# ── load training start indices for a room ────────────────────────────────────

def get_train_start_indices(room: str, n_train: int, data_root: Path) -> set[int]:
    starts = set()
    for si in range(n_train):
        meta_path = data_root / room / f"sample_{si:03d}" / "meta.pt"
        if meta_path.exists():
            m = torch.load(meta_path, weights_only=False)
            starts.add(int(m["start_idx"]))
    return starts


# ── extract one held-out sample on-the-fly (no caching) ──────────────────────

def extract_sample_live(start_idx: int, room: str, cfg, replica_root: Path,
                        K_proc: np.ndarray, H_proc: int, W_proc: int,
                        da3_model, nova_model, nova_cfg, device: torch.device):
    """Extract DA3 tokens and z_star for a single window without caching."""
    n_frames = int(cfg.num_frames)
    stride   = int(cfg.frame_stride)

    room_dir    = replica_root / room
    results_dir = room_dir / "results"
    mesh_ply    = replica_root / f"{room}_mesh.ply"

    # Read camera params
    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                      [0., cam_p["fy"], cam_p["cy"]],
                      [0., 0., 1.]], dtype=np.float32)
    K_nat_t  = torch.from_numpy(K_nat)

    poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_files = sorted(results_dir.glob("frame*.jpg"))
    all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]

    s_frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
    s_poses     = torch.from_numpy(
        np.stack([poses_all[fid] for fid in s_frame_ids])
    ).float()

    def load_rgb(fid: int) -> torch.Tensor:
        img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
        img = img.resize((W_proc, H_proc), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

    images_t   = torch.stack([load_rgb(fid) for fid in s_frame_ids])
    intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames, 1, 1))).float()

    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
    mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

    first_c2w   = s_poses[0]
    pts_cam_list = []
    for i in range(n_frames):
        frust = crop_frustum_world_points(mesh_pts, s_poses[i], K_nat_t, (H_nat, W_nat))
        pts_cam_list.append(world_to_first_camera(frust, first_c2w))

    all_frust = torch.cat(pts_cam_list, dim=0)
    tok_pts   = sample_points(all_frust, int(cfg.mesh_sample_points), seed=start_idx)
    tok_pts_b = tok_pts.unsqueeze(0)

    nova_seed = abs(hash(room)) % (2**20) + start_idx * 1009
    z_star, pts_norm = encode_nova3r(nova_model, nova_cfg, tok_pts_b, device, seed=nova_seed)

    da3_tokens = extract_da3_tokens(
        da3_model, images_t, s_poses, intrinsics,
        layer_idx=int(cfg.da3_source_layer_index),
        max_tokens=int(cfg.da3_max_source_tokens),
        device=device,
    )

    return da3_tokens, z_star, pts_norm


# ── evaluate a list of (da3, zstar, pnorm) samples ────────────────────────────

def evaluate_samples(da3_list, zstar_list, pnorm_list, adapter, nova_model,
                     nova_cfg, device, num_queries: int, tag: str):
    adapter.eval()
    cd_pg_list, cd_ai_list, mse_list = [], [], []

    with torch.no_grad():
        for i, (da3, zstar, pnorm) in enumerate(zip(da3_list, zstar_list, pnorm_list)):
            da3_b   = da3.to(device)
            pred    = adapter(da3_b).cpu()
            mse     = float(F.mse_loss(pred, zstar))

            gt_dec  = decode_tokens(nova_model, nova_cfg, zstar,  pnorm, device,
                                     num_queries, seed=42 + i)[0].numpy()
            pr_dec  = decode_tokens(nova_model, nova_cfg, pred,   pnorm, device,
                                     num_queries, seed=42 + i)[0].numpy()
            in_np   = pnorm[0].numpy()

            cd_pg = chamfer(torch.from_numpy(pr_dec), torch.from_numpy(gt_dec))
            cd_ai = chamfer(torch.from_numpy(gt_dec), torch.from_numpy(in_np))
            cd_pg_list.append(cd_pg)
            cd_ai_list.append(cd_ai)
            mse_list.append(mse)
            print(f"    [{tag}] {i:03d}  MSE={mse:.4e}  CD(pg)={cd_pg:.4f}  CD(ai)={cd_ai:.4f}")

    return float(np.mean(mse_list)), float(np.mean(cd_pg_list)), float(np.mean(cd_ai_list))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=str(EXP_DIR / "config.yaml"))
    parser.add_argument("--ckpt",     default=None,
                        help="Path to adapter checkpoint")
    parser.add_argument("--rooms",    nargs="+",
                        default=None,
                        help="Rooms to evaluate. Defaults to cfg.train_rooms.")
    parser.add_argument("--split", choices=["held-out", "train-seen"], default="held-out",
                        help="Evaluate held-out same-scene windows or windows from the training cache.")
    parser.add_argument("--n-eval",   type=int, default=50,
                        help="Windows to evaluate per room")
    parser.add_argument("--n-train",  type=int, default=None,
                        help="Number of training samples per room (to find used indices)")
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(args.device)
    stride = int(cfg.frame_stride)
    n_train = args.n_train or int(cfg.get("samples_per_room", 200))
    rooms = args.rooms or list(cfg.get("train_rooms", [str(cfg.room)]))

    replica_root = Path(str(cfg.replica_root))
    data_root    = EXP_DIR / "data" / f"multi_scene_N{n_train}_s{stride}"
    ckpt_path    = Path(args.ckpt) if args.ckpt is not None else (
        EXP_DIR / "checkpoints" / f"multi_scene_N{n_train}_s{stride}" / "adapter_final.pt"
    )

    # Intrinsics
    with (replica_root / "cam_params.json").open() as f:
        cam_p = json.load(f)["camera"]
    H_nat, W_nat = int(cam_p["h"]), int(cam_p["w"])
    K_nat = np.array([[cam_p["fx"], 0., cam_p["cx"]],
                      [0., cam_p["fy"], cam_p["cy"]],
                      [0., 0., 1.]], dtype=np.float32)
    H_proc, W_proc = int(cfg.image_height), int(cfg.image_width)
    K_proc = K_nat.copy()
    K_proc[0] *= W_proc / W_nat
    K_proc[1] *= H_proc / H_nat

    n_frames   = int(cfg.num_frames)
    window_span   = (n_frames - 1) * stride
    total_frames  = 2000
    max_start_idx = total_frames - 1 - window_span  # = 1859

    # ── load models ──────────────────────────────────────────────────────────
    print("[eval_within_scene] Loading NOVA3R …")
    nova_model, nova_cfg = load_nova3r_model(str(cfg.nova3r_ckpt), str(device))
    nova_model.eval()
    for p in nova_model.parameters(): p.requires_grad_(False)

    print("[eval_within_scene] Loading DA3 …")
    da3_model = load_da3_model(str(cfg.da3_model), device)

    print("[eval_within_scene] Loading adapter checkpoint …")
    adapter = DA3ToNOVA3RAlignment(
        source_dim=int(cfg.source_dim),
        hidden_dim=int(cfg.hidden_dim),
        target_tokens=int(cfg.target_tokens),
        target_dim=int(cfg.target_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        drop=0.0,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["adapter_state_dict"] if "adapter_state_dict" in ckpt else ckpt
    adapter.load_state_dict(state)
    adapter.eval()
    print(f"  Loaded from {ckpt_path}")

    num_queries = int(cfg.target_tokens)

    # ── per-room eval ────────────────────────────────────────────────────────
    results = {}  # room -> (mse, cd_pg, cd_ai)

    for room in rooms:
        print(f"\n[{room}] Collecting training start indices …")
        train_starts = get_train_start_indices(room, n_train, data_root)
        print(f"  {len(train_starts)} training windows used out of {max_start_idx+1} possible")
        if not train_starts:
            raise RuntimeError(
                f"No training metadata found for {room} under {data_root}. "
                "Run multi_scene_train.py first, or pass --n-train/--ckpt matching an existing cache."
            )

        all_possible = set(range(max_start_idx + 1))
        if args.split == "train-seen":
            eval_pool = sorted(train_starts)
        else:
            eval_pool = sorted(all_possible - train_starts)
        if args.n_eval > len(eval_pool):
            raise ValueError(
                f"--n-eval {args.n_eval} exceeds available {args.split} windows "
                f"for {room}: {len(eval_pool)}"
            )
        rng = np.random.default_rng(seed=42)
        eval_starts = sorted(rng.choice(eval_pool, size=args.n_eval, replace=False).tolist())
        print(f"  Evaluating {len(eval_starts)} {args.split} windows: {eval_starts[:5]} …")

        print(f"  Loading mesh for {room} …")
        mesh_ply = replica_root / f"{room}_mesh.ply"
        mesh     = trimesh.load(str(mesh_ply), force="mesh", process=False)
        mesh_pts_np, _ = trimesh.sample.sample_surface(mesh, 2_000_000)
        mesh_pts = torch.from_numpy(mesh_pts_np.astype(np.float32))

        room_dir    = replica_root / room
        results_dir = room_dir / "results"
        poses_all   = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
        frame_files = sorted(results_dir.glob("frame*.jpg"))
        all_frame_ids = [int(p.stem.replace("frame", "")) for p in frame_files]
        K_nat_t = torch.from_numpy(K_nat)

        print(f"  Extracting + evaluating {len(eval_starts)} {args.split} samples …")
        da3_list, zstar_list, pnorm_list = [], [], []
        t0 = time.time()

        for idx, start_idx in enumerate(eval_starts):
            s_frame_ids = [all_frame_ids[start_idx + i * stride] for i in range(n_frames)]
            s_poses     = torch.from_numpy(
                np.stack([poses_all[fid] for fid in s_frame_ids])
            ).float()

            def load_rgb(fid: int) -> torch.Tensor:
                img = Image.open(results_dir / f"frame{fid:06d}.jpg").convert("RGB")
                img = img.resize((W_proc, H_proc), Image.BILINEAR)
                return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.

            images_t   = torch.stack([load_rgb(fid) for fid in s_frame_ids])
            intrinsics = torch.from_numpy(np.tile(K_proc[None], (n_frames, 1, 1))).float()
            first_c2w  = s_poses[0]

            pts_cam_list = []
            for i in range(n_frames):
                frust = crop_frustum_world_points(mesh_pts, s_poses[i], K_nat_t, (H_nat, W_nat))
                pts_cam_list.append(world_to_first_camera(frust, first_c2w))

            all_frust = torch.cat(pts_cam_list, dim=0)
            tok_pts   = sample_points(all_frust, int(cfg.mesh_sample_points), seed=start_idx)
            tok_pts_b = tok_pts.unsqueeze(0)

            nova_seed = abs(hash(room)) % (2**20) + start_idx * 1009
            z_star, pts_norm = encode_nova3r(nova_model, nova_cfg, tok_pts_b, device,
                                              seed=nova_seed)
            da3_tokens = extract_da3_tokens(
                da3_model, images_t, s_poses, intrinsics,
                layer_idx=int(cfg.da3_source_layer_index),
                max_tokens=int(cfg.da3_max_source_tokens),
                device=device,
            )

            da3_list.append(da3_tokens)
            zstar_list.append(z_star)
            pnorm_list.append(pts_norm)

            elapsed = time.time() - t0
            rate    = (idx + 1) / elapsed
            eta     = (len(eval_starts) - idx - 1) / rate
            print(f"  [{room}] extracted {idx+1}/{len(eval_starts)}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        print(f"\n  [{room}] Running adapter eval …")
        mse, cd_pg, cd_ai = evaluate_samples(
            da3_list, zstar_list, pnorm_list,
            adapter, nova_model, nova_cfg, device, num_queries,
            tag=f"{room}_{args.split}"
        )
        results[room] = (mse, cd_pg, cd_ai)
        print(f"\n  [{room}] {args.split.upper()}  MSE={mse:.4e}  CD(pg)={cd_pg:.4f}  CD(ai)={cd_ai:.4f}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Within-Scene Evaluation ({args.split})")
    print(f"{'='*72}")
    print(f"  {'room':>10}  {'split':>14}  {'MSE':>12}  {'CD pred↔gt':>12}  {'CD ae↔input':>12}")

    # Known train/test results from v3 run
    known = {
        "room0": {"train_seen": (5.44e-02, 5.77e-03, 8.0e-03)},
        "room1": {"train_seen": (5.33e-02, 6.11e-03, 8.1e-03)},
        "room2": {"test_unseen": (2.25e+00, 5.67e-02, 8.5e-03)},
    }
    for room in rooms:
        if room in known:
            mse_k, cd_pg_k, cd_ai_k = known[room]["train_seen"]
            print(f"  {room:>10}  {'train (seen)':>14}  {mse_k:12.4e}  {cd_pg_k:12.4f}  {cd_ai_k:12.4f}")
            if room in results:
                mse_h, cd_pg_h, cd_ai_h = results[room]
                print(f"  {room:>10}  {args.split:>14}  {mse_h:12.4e}  {cd_pg_h:12.4f}  {cd_ai_h:12.4f}")

    mse_k, cd_pg_k, cd_ai_k = known["room2"]["test_unseen"]
    print(f"  {'room2':>10}  {'test (unseen)':>14}  {mse_k:12.4e}  {cd_pg_k:12.4f}  {cd_ai_k:12.4f}")
    print(f"{'='*72}")
    print()
    print("Interpretation:")
    print("  If held-out ≈ train(seen)   → scene-level memorisation (learns room structure)")
    print("  If held-out ≈ test(unseen)  → exact window memorisation only")


if __name__ == "__main__":
    main()
