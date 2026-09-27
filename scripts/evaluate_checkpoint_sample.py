#!/usr/bin/env python3
"""Comprehensive one-window evaluation of a DA3-to-NOVA3R adapter checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import umap
from omegaconf import OmegaConf
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
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
from nova3r.flow_matching.solver import ODESolver  # noqa: E402
from nova3r.inference import normalize_input  # noqa: E402
from nova3r.models.model_wrapper import BatchModelWrapper  # noqa: E402
from multi_scene_train import load_da3_model, extract_da3_tokens, world_to_first_camera  # noqa: E402

DEV = torch.device("cuda")
IMAGE_H, IMAGE_W = 392, 518
POINT_SAMPLE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-ckpt", type=Path, required=True)
    p.add_argument("--scene-ae-ckpt", type=Path,
                   default=REPO / "checkpoints/nova3r/scene_ae/checkpoint-last.pth")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--sequence", type=Path, required=True,
                   help="Sequence directory relative to data-root; must contain traj.txt and results/.")
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--color-ext", choices=["jpg", "png"], default="jpg")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--num-queries", type=int, default=50000)
    p.add_argument("--gt-points", type=int, default=200000)
    p.add_argument("--mesh-samples", type=int, default=2000000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def infer_adapter_cfg(state_dict: dict[str, torch.Tensor]) -> dict:
    target_tokens, hidden = state_dict["target_queries"].shape
    cfg = {
        "source_dim": state_dict["source_proj.weight"].shape[1],
        "hidden_dim": hidden,
        "target_tokens": target_tokens,
        "target_dim": state_dict["out_proj.weight"].shape[0],
        "depth": sum(k.endswith(".query_norm.weight") for k in state_dict),
        "num_heads": hidden // 64,
    }
    if "geom_embed.0.weight" in state_dict:
        raise ValueError("Geometry-conditioned adapters are not supported by this evaluator.")
    return cfg


@torch.no_grad()
def decode(model, cfg, tokens, pointmaps, num_queries: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    initial = torch.rand(1, num_queries, 3, device=DEV) * 2 - 1
    step = float(cfg.get("fm_step_size", 0.04))
    times = torch.linspace(0, 1, int(1 // step), device=DEV)
    solver = ODESolver(velocity_model=BatchModelWrapper(model=model))
    with torch.amp.autocast("cuda", enabled=False):
        solution = solver.sample(
            time_grid=times,
            x_init=initial,
            method="midpoint",
            step_size=step,
            return_intermediates=False,
            images=torch.zeros(1, 1, 3, 1, 1, device=DEV),
            token_mask=None,
            encoder_data={"tokens": tokens.to(DEV)},
            pointmaps=pointmaps.to(DEV),
        )
    return (solution[-1] if isinstance(solution, list) else solution)[0].cpu().float().numpy()


def point_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    gt_tree = cKDTree(ground_truth)
    pred_tree = cKDTree(prediction)
    pred_to_gt = gt_tree.query(prediction, k=1)[0]
    gt_to_pred = pred_tree.query(ground_truth, k=1)[0]
    accuracy = float(pred_to_gt.mean())
    completeness = float(gt_to_pred.mean())
    metrics = {
        "accuracy_m": accuracy,
        "completeness_m": completeness,
        "chamfer_m": 0.5 * (accuracy + completeness),
    }
    for threshold_cm in (1, 2, 5):
        threshold = threshold_cm / 100.0
        precision = float((pred_to_gt < threshold).mean())
        recall = float((gt_to_pred < threshold).mean())
        fscore = 2 * precision * recall / (precision + recall + 1e-9)
        metrics[f"precision@{threshold_cm}cm"] = precision
        metrics[f"recall@{threshold_cm}cm"] = recall
        metrics[f"F@{threshold_cm}cm"] = fscore
    return metrics


def save_umap_plots(scene_tokens: np.ndarray, adapter_tokens: np.ndarray,
                    output: Path, title: str, seed: int) -> None:
    scaler = StandardScaler().fit(scene_tokens)
    scene_scaled = scaler.transform(scene_tokens)
    adapter_scaled = scaler.transform(adapter_tokens)
    pca = PCA(n_components=50, random_state=seed).fit(scene_scaled)
    scene_reduced = pca.transform(scene_scaled)
    adapter_reduced = pca.transform(adapter_scaled)
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine",
        random_state=seed, transform_seed=seed, low_memory=True,
    ).fit(scene_reduced)
    scene_embedding = reducer.embedding_
    adapter_embedding = reducer.transform(adapter_reduced)
    np.savez_compressed(
        output / "umap_coordinates.npz",
        scene_ae=scene_embedding.astype(np.float32),
        adapter=adapter_embedding.astype(np.float32),
    )

    both = np.concatenate([scene_embedding, adapter_embedding])
    lo, hi = both.min(0), both.max(0)
    padding = np.maximum((hi - lo) * 0.04, 1e-8)
    limits = ((lo[0] - padding[0], hi[0] + padding[0]),
              (lo[1] - padding[1], hi[1] + padding[1]))
    sources = {
        "scene_ae": (scene_embedding, "scene_ae", "#277da1"),
        "adapter": (adapter_embedding, "adapter", "#f3722c"),
    }
    for key, (embedding, label, color) in sources.items():
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(embedding[:, 0], embedding[:, 1], s=14, alpha=0.55,
                   edgecolors="none", color=color)
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="UMAP 1", ylabel="UMAP 2",
               title=f"{title} — {label}")
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(output / f"{key}_umap.png", dpi=240)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for embedding, label, color in sources.values():
        ax.scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.45,
                   edgecolors="none", color=color, label=label)
    ax.set(xlim=limits[0], ylim=limits[1], xlabel="UMAP 1", ylabel="UMAP 2",
           title=f"{title} — scene_ae reference UMAP")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "combined_umap.png", dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    sequence_dir = args.data_root / args.sequence
    results_dir = sequence_dir / "results"
    frame_ids = [args.start_frame + i * args.stride for i in range(args.n_frames)]

    camera = json.loads((args.data_root / "cam_params.json").read_text())["camera"]
    native_h, native_w = int(camera["h"]), int(camera["w"])
    intrinsics_native = torch.tensor([
        [camera["fx"], 0.0, camera["cx"]],
        [0.0, camera["fy"], camera["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    intrinsics_processed = intrinsics_native.clone()
    intrinsics_processed[0] *= IMAGE_W / native_w
    intrinsics_processed[1] *= IMAGE_H / native_h

    all_poses = np.loadtxt(sequence_dir / "traj.txt", dtype=np.float32).reshape(-1, 4, 4)
    poses = torch.from_numpy(np.stack([all_poses[f] for f in frame_ids])).float()
    first_pose = poses[0]

    images = []
    for index, frame_id in enumerate(frame_ids):
        source = results_dir / f"frame{frame_id:06d}.{args.color_ext}"
        image = Image.open(source).convert("RGB")
        image.save(output / f"input_{index:02d}_frame{frame_id:06d}.png")
        resized = image.resize((IMAGE_W, IMAGE_H), Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(np.asarray(resized, np.float32)).permute(2, 0, 1) / 255.0)
    images = torch.stack(images)

    np.random.seed(args.seed)
    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    mesh_world = torch.from_numpy(
        trimesh.sample.sample_surface(mesh, args.mesh_samples)[0].astype(np.float32)
    )
    complete_pools = []
    for pose in poses:
        projection = project_world_points(
            points_world=mesh_world,
            world_to_camera=torch.linalg.inv(pose),
            intrinsics=intrinsics_native,
            image_hw=(native_h, native_w),
        )
        complete_pools.append(mesh_world[projection["inside"]])
    complete_world = torch.unique(torch.cat(complete_pools), dim=0)
    complete_first_camera = world_to_first_camera(complete_world, first_pose)

    generator = torch.Generator().manual_seed(args.seed)
    if len(complete_first_camera) >= POINT_SAMPLE:
        indices = torch.randperm(len(complete_first_camera), generator=generator)[:POINT_SAMPLE]
    else:
        indices = torch.randint(len(complete_first_camera), (POINT_SAMPLE,), generator=generator)
    sampled_points = complete_first_camera[indices].unsqueeze(0).to(DEV)

    print("[load] scene_ae", flush=True)
    scene_ae, scene_cfg = load_nova3r_model(str(args.scene_ae_ckpt), str(DEV))
    scene_ae.eval()
    OmegaConf.set_struct(scene_cfg, False)
    norm_mode = scene_cfg.model.params.cfg.pts3d_head.params.get("norm_mode", "none")
    valid = torch.ones(sampled_points.shape[:2], dtype=torch.bool, device=DEV)
    points_norm, _ = normalize_input(
        sampled_points, valid, sampled_points, valid, mode=norm_mode
    )
    with torch.no_grad():
        scene_tokens = scene_ae._encode(pointmaps=points_norm, test=True)["tokens"].float()
    norm_factor = float(sampled_points[0].norm(dim=-1).median().clamp(0.01, 100.0))

    print("[load] DA3 + adapter", flush=True)
    da3 = load_da3_model("depth-anything/DA3-LARGE-1.1", DEV)
    adapter_data = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    adapter = DA3ToNOVA3RAlignment(
        **infer_adapter_cfg(adapter_data["state_dict"]), drop=0.0
    ).to(DEV)
    adapter.load_state_dict(adapter_data["state_dict"])
    adapter.eval()
    processed_intrinsics = intrinsics_processed.unsqueeze(0).expand(args.n_frames, -1, -1).clone()
    da3_tokens = extract_da3_tokens(
        da3, images, poses, processed_intrinsics,
        layer_idx=[1, 3], max_tokens=2048, device=DEV,
    )
    with torch.no_grad():
        predicted_tokens = adapter(da3_tokens.to(DEV)).float()

    torch.save(scene_tokens[0].cpu(), output / "scene_ae_tokens.pt")
    torch.save(predicted_tokens[0].cpu(), output / "adapter_tokens.pt")
    np.savez_compressed(
        output / "latents.npz",
        scene_ae=scene_tokens[0].cpu().numpy(),
        adapter=predicted_tokens[0].cpu().numpy(),
    )

    print("[decode] scene_ae + adapter", flush=True)
    scene_decoded_norm = decode(
        scene_ae, scene_cfg, scene_tokens, points_norm, args.num_queries, args.seed
    )
    adapter_decoded_norm = decode(
        scene_ae, scene_cfg, predicted_tokens, points_norm, args.num_queries, args.seed
    )
    rotation, translation = first_pose[:3, :3].numpy(), first_pose[:3, 3].numpy()
    to_world = lambda points: (
        rotation @ (points / 3.0 * norm_factor).T
    ).T + translation
    scene_decoded_world = to_world(scene_decoded_norm)
    adapter_decoded_world = to_world(adapter_decoded_norm)

    gt_generator = torch.Generator().manual_seed(args.seed)
    if len(complete_world) > args.gt_points:
        gt_indices = torch.randperm(len(complete_world), generator=gt_generator)[:args.gt_points]
        ground_truth = complete_world[gt_indices].numpy()
    else:
        ground_truth = complete_world.numpy()
    trimesh.PointCloud(scene_decoded_world).export(output / "scene_ae_decoded.ply")
    trimesh.PointCloud(adapter_decoded_world).export(output / "adapter_decoded.ply")
    trimesh.PointCloud(ground_truth).export(output / "ground_truth.ply")

    scene_np = scene_tokens[0].cpu().numpy()
    adapter_np = predicted_tokens[0].cpu().numpy()
    cost = (
        np.square(scene_np).sum(1)[:, None]
        + np.square(adapter_np).sum(1)[None, :]
        - 2 * scene_np @ adapter_np.T
    )
    rows, columns = linear_sum_assignment(cost)
    latent_metrics = {
        "raw_paired_mse": float(np.mean(np.square(scene_np - adapter_np))),
        "hungarian_mse": float(cost[rows, columns].mean() / scene_np.shape[1]),
    }
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "sequence": str(args.sequence),
        "frame_ids": frame_ids,
        "adapter_checkpoint": str(args.adapter_ckpt),
        "adapter_checkpoint_sha256": hashlib.sha256(args.adapter_ckpt.read_bytes()).hexdigest(),
        "norm_factor_m": norm_factor,
        "latent": latent_metrics,
        "adapter_pointcloud": point_metrics(adapter_decoded_world, ground_truth),
        "scene_ae_oracle_pointcloud": point_metrics(scene_decoded_world, ground_truth),
        "counts": {
            "complete_pool": len(complete_world),
            "ground_truth_saved": len(ground_truth),
            "decoded": len(adapter_decoded_world),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    title = f"{args.dataset} - {args.split}"
    save_umap_plots(scene_np, adapter_np, output, title, args.seed)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
