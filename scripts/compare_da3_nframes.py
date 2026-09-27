#!/usr/bin/env python3
"""Compare DA3 tokens extracted from the same temporal span with 8 vs 16 frames.

Both windows share the same first and last frame (same temporal extent).
The 16-frame window uses half the stride, inserting one intermediate frame
between each pair of 8-frame keyframes.

Outputs three measurements:
  1. Sanity check — re-extracted 8-frame tokens vs cached tokens (should be ~0 diff).
  2. 8-frame vs 16-frame nearest-neighbour cosine similarity distribution.
  3. Stats on how normalize_extrinsics scale changes with N_FRAMES.

Usage:
    python scripts/compare_da3_nframes.py \\
        --cache-root scripts/data/online_hungarian_matching_zstar_windows_s20_l23 \\
        --room office0 --start 30 \\
        --replica-root datasets/replica
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT   = Path(__file__).resolve().parents[1]
DA3_SRC     = REPO_ROOT / "da3" / "src"
NOVA3R_ROOT = REPO_ROOT / "nova3r"
OVERFIT_SRC = REPO_ROOT / "experiments" / "overfit_8frames"

for _p in [str(NOVA3R_ROOT / "third_party"), str(NOVA3R_ROOT), str(DA3_SRC), str(OVERFIT_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multi_scene_train import (  # noqa: E402
    load_da3_model, extract_da3_tokens, normalize_extrinsics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path,
                   default=REPO_ROOT / "scripts/data/online_hungarian_matching_zstar_windows_s20_l23")
    p.add_argument("--room",           default="office0")
    p.add_argument("--start",          type=int, default=30,
                   help="Window start frame index in the cached data.")
    p.add_argument("--replica-root",   type=Path,
                   default=REPO_ROOT / "datasets/replica")
    p.add_argument("--da3-model",      default="depth-anything/DA3-LARGE-1.1")
    p.add_argument("--da3-layer",      type=int, nargs="+", default=[2, 3])
    p.add_argument("--da3-max-tokens", type=int, default=2048)
    p.add_argument("--image-height",   type=int, default=392)
    p.add_argument("--image-width",    type=int, default=518)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_rgb(path: Path, h: int, w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0


def nn_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """For each token in a, find cosine similarity to its nearest neighbour in b."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    sims = a_n @ b_n.T          # (Na, Nb)
    return sims.max(dim=1).values


def extrinsics_scale(poses_c2w: torch.Tensor) -> float:
    """Return the median-distance normalisation scale used by normalize_extrinsics."""
    poses_bt = poses_c2w.unsqueeze(0)
    transform = torch.linalg.inv(poses_bt[:, :1])
    normalized = poses_bt @ transform
    c2ws = torch.linalg.inv(normalized)
    return float(c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # ── load cached window ────────────────────────────────────────────────────
    win_dir  = args.cache_root / args.room / f"start_{args.start:04d}"
    meta     = torch.load(win_dir / "meta.pt", map_location="cpu", weights_only=False)
    cached   = torch.load(win_dir / "da3_tokens.pt", map_location="cpu", weights_only=True)

    frame_ids_8 = meta["frame_ids"]    # list of 8 frame indices
    poses_c2w_8 = meta["poses_c2w"]    # (8, 4, 4)
    stride      = int(meta["stride"])  # e.g. 20

    print(f"[window] room={args.room}  start={args.start}  stride={stride}")
    print(f"         frames: {frame_ids_8}")
    print(f"         cached tokens: {tuple(cached.shape)}\n")

    # ── load scene data ───────────────────────────────────────────────────────
    room_dir    = args.replica_root / args.room
    results_dir = room_dir / "results"
    with (args.replica_root / "cam_params.json").open() as f:
        cam = json.load(f)["camera"]
    h_nat, w_nat = int(cam["h"]), int(cam["w"])
    k_native = torch.tensor([
        [cam["fx"], 0.0, cam["cx"]],
        [0.0, cam["fy"], cam["cy"]],
        [0.0, 0.0, 1.0]], dtype=torch.float32)
    k_proc = k_native.clone()
    k_proc[0] *= args.image_width  / w_nat
    k_proc[1] *= args.image_height / h_nat

    poses_all = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    all_frame_ids = [int(p.stem.replace("frame", ""))
                     for p in sorted(results_dir.glob("frame*.jpg"))]

    # ── build 15-frame window: same 8 keyframes + 1 midpoint per gap ─────────
    # stride_half = stride // 2; 15 = 2*8 - 1; every 8-frame index is a subset.
    start_fid    = frame_ids_8[0]
    half         = stride // 2
    frame_ids_15 = [start_fid + i * half for i in range(15)]

    poses_c2w_15 = torch.from_numpy(
        np.stack([poses_all[fid] for fid in frame_ids_15])).float()

    print(f"[8-frame]  frames: {frame_ids_8[0]} … {frame_ids_8[-1]}  (stride={stride})")
    print(f"[15-frame] frames: {frame_ids_15[0]} … {frame_ids_15[-1]}  (stride={half})\n")

    # ── normalisation scale comparison ────────────────────────────────────────
    scale_8  = extrinsics_scale(poses_c2w_8)
    scale_15 = extrinsics_scale(poses_c2w_15)
    print(f"[extrinsics normalisation scale]")
    print(f"  8-frame:  {scale_8:.5f}")
    print(f"  15-frame: {scale_15:.5f}")
    print(f"  ratio:    {scale_15/scale_8:.4f}\n")

    # ── load images ───────────────────────────────────────────────────────────
    print("[loading images …]")

    def rgb(fid: int):
        return load_rgb(results_dir / f"frame{fid:06d}.jpg",
                        args.image_height, args.image_width)

    images_8  = torch.stack([rgb(f) for f in frame_ids_8])
    images_15 = torch.stack([rgb(f) for f in frame_ids_15])
    K_bt_8    = torch.from_numpy(np.tile(k_proc.numpy()[None], (len(frame_ids_8),  1, 1))).float()
    K_bt_15   = torch.from_numpy(np.tile(k_proc.numpy()[None], (len(frame_ids_15), 1, 1))).float()

    # ── load DA3 model ────────────────────────────────────────────────────────
    print(f"[loading DA3 {args.da3_model} …]")
    da3_model = load_da3_model(args.da3_model, device)

    layer_arg = args.da3_layer if len(args.da3_layer) > 1 else args.da3_layer[0]

    # ── re-extract 8-frame tokens (sanity check) ──────────────────────────────
    print("[extracting 8-frame tokens …]")
    with torch.no_grad():
        tok_8 = extract_da3_tokens(
            da3_model, images_8, poses_c2w_8, K_bt_8,
            layer_idx=layer_arg, max_tokens=args.da3_max_tokens, device=device,
        )

    tok_8_sq = tok_8[0]           # (N_tok, dim)
    cached_sq = cached[0]         # (N_tok, dim)
    l2_sanity = (tok_8_sq - cached_sq).norm(dim=-1).mean().item()
    cos_sanity = nn_cosine_sim(tok_8_sq, cached_sq).mean().item()
    print(f"\n[sanity: re-extracted 8-frame vs cached]")
    print(f"  mean token L2:          {l2_sanity:.6f}  (expect ~0)")
    print(f"  mean NN cosine sim:     {cos_sanity:.6f}  (expect ~1)")

    # ── extract 15-frame tokens ───────────────────────────────────────────────
    print("\n[extracting 15-frame tokens …]")
    with torch.no_grad():
        tok_15 = extract_da3_tokens(
            da3_model, images_15, poses_c2w_15, K_bt_15,
            layer_idx=layer_arg, max_tokens=args.da3_max_tokens, device=device,
        )

    tok_15_sq = tok_15[0]         # (N_tok, dim)

    print(f"\n[8-frame vs 15-frame token comparison]")
    print(f"  8-frame  token tensor:  {tuple(tok_8.shape)}")
    print(f"  15-frame token tensor:  {tuple(tok_15.shape)}")

    # L2 between matched (same index) tokens
    l2_direct = (tok_8_sq - tok_15_sq).norm(dim=-1)
    print(f"\n  direct (index-aligned) L2 per token:")
    print(f"    mean={l2_direct.mean():.4f}  median={l2_direct.median():.4f}  "
          f"max={l2_direct.max():.4f}")

    # Nearest-neighbour cosine: for each 8-frame token, best match in 16-frame set
    nn_sim_8_to_15 = nn_cosine_sim(tok_8_sq, tok_15_sq)
    print(f"\n  NN cosine sim (8-frame → 15-frame):")
    print(f"    mean={nn_sim_8_to_15.mean():.4f}  median={nn_sim_8_to_15.median():.4f}  "
          f"min={nn_sim_8_to_15.min():.4f}")

    # Fraction of tokens with NN cosine > threshold
    for t in [0.90, 0.95, 0.99]:
        frac = (nn_sim_8_to_15 > t).float().mean().item()
        print(f"    fraction with NN cos > {t:.2f}: {frac:.3f}")

    # Global representation: mean token vector
    mean_8  = tok_8_sq.mean(0)
    mean_15 = tok_15_sq.mean(0)
    global_cos = F.cosine_similarity(mean_8.unsqueeze(0), mean_15.unsqueeze(0)).item()
    global_l2  = (mean_8 - mean_15).norm().item()
    print(f"\n  global mean-token cosine sim:  {global_cos:.6f}")
    print(f"  global mean-token L2:          {global_l2:.4f}")

    # Feature-channel statistics
    std_8  = tok_8_sq.std(0).mean().item()
    std_15 = tok_15_sq.std(0).mean().item()
    print(f"\n  mean per-channel token std:")
    print(f"    8-frame:  {std_8:.4f}")
    print(f"    15-frame: {std_15:.4f}")

    print("\n[done]")


if __name__ == "__main__":
    main()
