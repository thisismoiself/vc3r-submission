#!/usr/bin/env python3
"""Joint UMAP of scene_ae, scene_n1, and DA3-adapter NOVA3R tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
for path in [
    REPO / "nova3r" / "third_party",
    REPO / "nova3r",
    REPO / "da3" / "src",
    REPO / "scripts",
    REPO / "experiments" / "overfit_8frames",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from demo_nova3r import load_model as load_nova3r_model  # noqa: E402
from vc3r.replica import project_world_points  # noqa: E402
from vc3r.alignment import DA3ToNOVA3RAlignment  # noqa: E402
from nova3r.inference import normalize_input  # noqa: E402
from multi_scene_train import load_da3_model, extract_da3_tokens, world_to_first_camera  # noqa: E402

DEV = torch.device("cuda")
IMG_H, IMG_W = 392, 518
K_SAMPLE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-ckpt", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--room", default="office4")
    p.add_argument("--scene-ae-ckpt", type=Path,
                   default=REPO / "checkpoints/nova3r/scene_ae/checkpoint-last.pth")
    p.add_argument("--scene-n1-ckpt", type=Path,
                   default=REPO / "nova3r/checkpoints/scene_n1/checkpoint-last.pth")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-windows", type=int, default=25)
    p.add_argument("--max-tokens-per-source", type=int, default=15000)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def adapter_cfg(sd: dict[str, torch.Tensor]) -> dict:
    target_tokens, hidden = sd["target_queries"].shape
    cfg = {
        "source_dim": sd["source_proj.weight"].shape[1],
        "hidden_dim": hidden,
        "target_tokens": target_tokens,
        "target_dim": sd["out_proj.weight"].shape[0],
        "depth": sum(k.endswith(".query_norm.weight") for k in sd),
        "num_heads": hidden // 64,
    }
    if "geom_embed.0.weight" in sd:
        raise ValueError("This comparison currently supports non-geometry-conditioned adapters only.")
    return cfg


def balanced_sample(x: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    x = x.reshape(-1, x.shape[-1])
    if len(x) <= limit:
        return x
    return x[rng.choice(len(x), limit, replace=False)]


def main() -> None:
    args = parse_args()
    try:
        import umap
    except ImportError as exc:
        raise SystemExit("Install umap-learn in this Python environment.") from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[load] scene_ae", flush=True)
    scene_ae, ae_cfg = load_nova3r_model(str(args.scene_ae_ckpt), str(DEV))
    scene_ae.eval()
    OmegaConf.set_struct(ae_cfg, False)
    norm_mode = ae_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")

    print("[load] scene_n1", flush=True)
    scene_n1, _ = load_nova3r_model(str(args.scene_n1_ckpt), str(DEV))
    scene_n1.eval()

    print("[load] DA3 + adapter", flush=True)
    da3 = load_da3_model("depth-anything/DA3-LARGE-1.1", DEV)
    ckpt = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter = DA3ToNOVA3RAlignment(**adapter_cfg(ckpt["state_dict"]), drop=0.0).to(DEV)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()

    camera = json.loads((args.data_root / "cam_params.json").read_text())["camera"]
    native_h, native_w = int(camera["h"]), int(camera["w"])
    K_native = torch.tensor([
        [camera["fx"], 0.0, camera["cx"]],
        [0.0, camera["fy"], camera["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    K_proc = K_native.clone()
    K_proc[0] *= IMG_W / native_w
    K_proc[1] *= IMG_H / native_h

    room_dir = args.data_root / args.room
    results = room_dir / "results"
    poses = np.loadtxt(room_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    frame_ids = [int(p.stem.replace("frame", "")) for p in sorted(results.glob("frame*.jpg"))]
    mesh = trimesh.load(str(args.data_root / f"{args.room}_mesh.ply"), force="mesh", process=False)
    mesh_points = torch.from_numpy(
        trimesh.sample.sample_surface(mesh, 2_000_000)[0].astype(np.float32)
    )

    span = (args.n_frames - 1) * args.stride
    windows, base = [], 0
    while base + span < len(frame_ids) and len(windows) < args.max_windows:
        windows.append([base + i * args.stride for i in range(args.n_frames)])
        base += span + args.stride
    print(f"[data] {len(windows)} matched windows", flush=True)

    def load_rgb(fid: int) -> torch.Tensor:
        image = Image.open(results / f"frame{fid:06d}.jpg").convert("RGB")
        image = image.resize((IMG_W, IMG_H), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image, np.float32)).permute(2, 0, 1) / 255.0

    ae_all, n1_all, adapter_all = [], [], []
    for wi, fids in enumerate(windows):
        poses_c2w = torch.from_numpy(np.stack([poses[f] for f in fids])).float()
        first_c2w = poses_c2w[0]
        pool = []
        for pose in poses_c2w:
            projection = project_world_points(
                points_world=mesh_points,
                world_to_camera=torch.linalg.inv(pose),
                intrinsics=K_native,
                image_hw=(native_h, native_w),
            )
            pool.append(world_to_first_camera(mesh_points[projection["inside"]], first_c2w))
        pool = torch.cat(pool)
        generator = torch.Generator().manual_seed(0)
        if len(pool) >= K_SAMPLE:
            indices = torch.randperm(len(pool), generator=generator)[:K_SAMPLE]
        else:
            indices = torch.randint(len(pool), (K_SAMPLE,), generator=generator)
        points = pool[indices].unsqueeze(0).to(DEV)
        valid = torch.ones(points.shape[:2], dtype=torch.bool, device=DEV)
        points_norm, _ = normalize_input(points, valid, points, valid, mode=norm_mode)

        images = torch.stack([load_rgb(fid) for fid in fids])
        intrinsics = K_proc.unsqueeze(0).expand(args.n_frames, -1, -1).clone()
        da3_tokens = extract_da3_tokens(
            da3, images, poses_c2w, intrinsics,
            layer_idx=[1, 3], max_tokens=2048, device=DEV,
        )
        n1_image = (images[0:1].to(DEV) * 2.0 - 1.0).unsqueeze(0)
        with torch.no_grad():
            z_ae = scene_ae._encode(pointmaps=points_norm, test=True)["tokens"]
            z_n1 = scene_n1._encode(images=n1_image, test=True)["tokens"]
            z_adapter = adapter(da3_tokens.to(DEV))
        shapes = (tuple(z_ae.shape), tuple(z_n1.shape), tuple(z_adapter.shape))
        if not (z_ae.shape == z_n1.shape == z_adapter.shape):
            raise RuntimeError(f"Latent shapes do not match: {shapes}")
        ae_all.append(z_ae[0].cpu().float().numpy())
        n1_all.append(z_n1[0].cpu().float().numpy())
        adapter_all.append(z_adapter[0].cpu().float().numpy())
        print(f"  window {wi + 1:02d}/{len(windows)} frames {fids[0]}-{fids[-1]} {shapes[0]}", flush=True)

    arrays = {
        "scene_ae": np.stack(ae_all),
        "scene_n1": np.stack(n1_all),
        "adapter": np.stack(adapter_all),
    }
    np.savez_compressed(args.out_dir / "latents.npz", **arrays)

    rng = np.random.default_rng(args.seed)
    sampled = {
        name: balanced_sample(value, args.max_tokens_per_source, rng)
        for name, value in arrays.items()
    }
    features = np.concatenate(list(sampled.values()))
    labels = np.concatenate([
        np.full(len(value), name) for name, value in sampled.items()
    ])
    features = StandardScaler().fit_transform(features)
    pca = PCA(n_components=50, random_state=args.seed)
    features_50 = pca.fit_transform(features)
    embedding = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
        low_memory=True,
    ).fit_transform(features_50)

    colors = {"scene_ae": "#277da1", "scene_n1": "#43aa8b", "adapter": "#f3722c"}
    names = {
        "scene_ae": "NOVA3R scene_ae",
        "scene_n1": "NOVA3R scene_n1",
        "adapter": "DA3 adapter",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for key in sampled:
        mask = labels == key
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=5, alpha=0.24, edgecolors="none", color=colors[key], label=names[key],
        )
    ax.set_title(f"Joint NOVA3R latent-token UMAP — {args.room}, {len(windows)} windows")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    figure = args.out_dir / "three_way_latent_umap.png"
    fig.savefig(figure, dpi=240)
    plt.close(fig)
    np.savez_compressed(args.out_dir / "umap_coordinates.npz", embedding=embedding, labels=labels)
    print(f"[out] {figure}", flush=True)


if __name__ == "__main__":
    main()
