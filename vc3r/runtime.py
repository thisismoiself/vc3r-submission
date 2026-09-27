"""Reusable model-loading and inference helpers for VC3R.

This module deliberately has no dependency on training, cache-generation,
plotting, logging, or interactive demo code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from scipy.spatial import cKDTree

from depth_anything_3.cfg import create_object
from vc3r.alignment import CrossAttentionBlock
from nova3r.models.nova3r_pts_cond import Nova3rPtsCond
from vc3r.artifacts import DA3_REVISION, resolve_da3_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_nova3r_model(ckpt_path: str | Path, device: str | torch.device):
    """Load the NOVA3R point autoencoder and its adjacent Hydra config."""
    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No .hydra/config.yaml found at {config_path}. "
            "Keep the NOVA3R Hydra config beside its checkpoint."
        )

    cfg = OmegaConf.load(config_path).experiment
    model_config = cfg.model
    model_classes = {"Nova3rPtsCond": Nova3rPtsCond}
    try:
        model_class = model_classes[model_config["name"]]
    except KeyError as exc:
        supported = ", ".join(sorted(model_classes))
        raise ValueError(
            f"Unsupported NOVA3R model {model_config['name']!r}; supported: {supported}"
        ) from exc

    model = model_class(**model_config["params"]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
    return model, cfg




def load_da3_model(
    model_name: str | Path,
    device: torch.device,
    revision: str = DA3_REVISION,
    local_files_only: bool = False,
):
    """Load DA3 from a local snapshot or an exact Hugging Face revision."""
    snapshot = resolve_da3_snapshot(model_name, revision, local_files_only)

    with (snapshot / "config.json").open() as f:
        payload = json.load(f)
    model = create_object(OmegaConf.create(payload["config"]))
    state = load_file(str(snapshot / "model.safetensors"), device="cpu")
    state = {key.removeprefix("model."): value
             for key, value in state.items() if key.startswith("model.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    return (images - mean[None, None, :, None, None]) / std[None, None, :, None, None]


def _normalize_extrinsics(poses_c2w: torch.Tensor) -> torch.Tensor:
    transform = torch.linalg.inv(poses_c2w[:, :1])
    normalized = poses_c2w @ transform
    c2ws = torch.linalg.inv(normalized)
    median_dist = c2ws[..., :3, 3].norm(dim=-1).median().clamp(min=1e-1)
    normalized[..., :3, 3] /= median_dist
    return normalized


def _select_tokens(tokens: torch.Tensor, max_n: int) -> torch.Tensor:
    if max_n <= 0 or tokens.shape[1] <= max_n:
        return tokens
    idx = torch.linspace(0, tokens.shape[1] - 1, max_n, device=tokens.device).long()
    return tokens[:, idx]


@torch.no_grad()
def extract_da3_tokens(
    model,
    images: torch.Tensor,
    poses_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    layer_idx: int | list[int],
    max_tokens: int,
    device: torch.device,
    use_poses: bool = True,
) -> torch.Tensor:
    """Extract DA3 backbone tokens exactly as in adapter training."""
    images_bt = images.unsqueeze(0).to(device)
    intrinsics_bt = intrinsics.unsqueeze(0).to(device)
    images_norm = _imagenet_normalize(images_bt)
    amp_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)

    if use_poses:
        poses_bt = poses_c2w.unsqueeze(0).to(device)
        extrinsics_w2c = _normalize_extrinsics(torch.linalg.inv(poses_bt[0]).unsqueeze(0))
        with torch.autocast(device_type=device.type, enabled=False):
            cam_token = model.cam_enc(extrinsics_w2c, intrinsics_bt, images_bt.shape[-2:])
    else:
        cam_token = None

    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                        enabled=(device.type == "cuda")):
        backbone_out, _ = model.backbone(images_norm, cam_token=cam_token)

    indices = [layer_idx] if isinstance(layer_idx, int) else layer_idx
    parts = []
    for index in indices:
        raw = backbone_out[index][0].float()
        flat = raw.reshape(raw.shape[0], -1, raw.shape[-1])
        parts.append(_select_tokens(flat, max_tokens).reshape(1, -1, raw.shape[-1]))
    return torch.cat(parts, dim=1).cpu()


def world_to_first_camera(
    points_world: torch.Tensor,
    first_pose_c2w: torch.Tensor,
) -> torch.Tensor:
    world_to_camera = torch.linalg.inv(first_pose_c2w)
    ones = torch.ones(*points_world.shape[:-1], 1, dtype=points_world.dtype)
    return (torch.cat([points_world, ones], dim=-1) @ world_to_camera.T)[..., :3]


def _fps_numpy(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = rng.integers(len(points))
    distances = np.full(len(points), np.inf)
    for index in range(1, count):
        distances = np.minimum(
            distances,
            ((points - points[selected[index - 1]]) ** 2).sum(1),
        )
        selected[index] = int(distances.argmax())
    return points[selected]


def to_exact(
    points: np.ndarray,
    count: int,
    use_fps: bool,
    fps_cap: int = 16384,
) -> np.ndarray:
    """Return exactly ``count`` points, matching the training cache sampler."""
    if len(points) == 0:
        return np.zeros((count, 3), np.float32)
    if len(points) >= count:
        if not use_fps:
            return points[np.random.default_rng(0).choice(len(points), count, replace=False)]
        candidates = (points if len(points) <= fps_cap else
                      points[np.random.default_rng(0).choice(len(points), fps_cap, replace=False)])
        return _fps_numpy(candidates, count)
    pad = np.random.default_rng(0).choice(len(points), count - len(points), replace=True)
    return np.concatenate([points, points[pad]], 0)


class PointFlowCorrector(torch.nn.Module):
    """Conditional point-space velocity field used by the optional corrector."""

    def __init__(
        self,
        token_dim: int = 128,
        hidden: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        fourier_bands: int = 8,
        time_bands: int = 64,
        local_knn: int = 0,
    ):
        super().__init__()
        self.fb = fourier_bands
        self.tb = time_bands
        self.local_knn = local_knn
        self.point_in = torch.nn.Linear(3 * 2 * fourier_bands, hidden)
        self.time_in = torch.nn.Linear(2 * time_bands, hidden)
        self.token_proj = torch.nn.Linear(token_dim, hidden)
        if local_knn > 0:
            self.local_encoder = torch.nn.Sequential(
                torch.nn.Linear(4, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, hidden),
            )
        self.blocks = torch.nn.ModuleList(
            [CrossAttentionBlock(dim=hidden, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = torch.nn.LayerNorm(hidden)
        self.out = torch.nn.Linear(hidden, 3)
        self.out.weight.data.mul_(0.1)
        self.out.bias.data.zero_()

    def _fourier(self, values: torch.Tensor, bands: int) -> torch.Tensor:
        frequencies = (2.0 ** torch.arange(
            bands, device=values.device, dtype=values.dtype
        )) * math.pi
        angles = values[..., None] * frequencies
        encoded = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return encoded.reshape(*values.shape[:-1], -1)

    def forward(self, x_t, time, tokens, neighbors=None):
        hidden = (
            self.point_in(self._fourier(x_t, self.fb))
            + self.time_in(self._fourier(time[..., None], self.tb))
        )
        if self.local_knn > 0 and neighbors is not None:
            distance = neighbors.norm(dim=-1, keepdim=True)
            local = self.local_encoder(
                torch.cat([neighbors, distance], dim=-1)
            ).amax(dim=2)
            hidden = hidden + local
        memory = self.token_proj(tokens)
        for block in self.blocks:
            hidden = block(hidden, memory)
        return self.out(self.norm(hidden))


def _knn_offsets_cpu(cloud: torch.Tensor, count: int) -> torch.Tensor:
    points = cloud.detach().cpu().numpy()
    _, indices = cKDTree(points).query(points, k=count + 1)
    neighbors = points[indices[:, 1:]] - points[:, None, :]
    return torch.from_numpy(neighbors).float()


@torch.no_grad()
def integrate_point_corrector(
    model: PointFlowCorrector,
    initial_points: torch.Tensor,
    tokens: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Integrate the point corrector with the training-time midpoint solver."""
    points = initial_points
    neighbors = None
    if model.local_knn > 0:
        neighbors = torch.stack([
            _knn_offsets_cpu(initial_points[index], model.local_knn)
            for index in range(initial_points.shape[0])
        ]).to(initial_points.device)

    times = torch.linspace(0, 1, steps + 1, device=initial_points.device)
    for index in range(steps):
        time = times[index].expand(points.shape[:2])
        dt = times[index + 1] - times[index]
        velocity = model(points, time, tokens, neighbors)
        midpoint = points + 0.5 * dt * velocity
        midpoint_time = (times[index] + 0.5 * dt).expand(points.shape[:2])
        points = points + dt * model(midpoint, midpoint_time, tokens, neighbors)
    return points
