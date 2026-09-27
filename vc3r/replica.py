"""Replica room geometry and dataset utilities used by VC3R."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ReplicaRoomConfig:
    data_root: Path
    room: str = "room0"
    frame_stride: int = 20
    num_frames: int = 8
    load_depth: bool = True
    image_size: tuple[int, int] | None = None
    visible_points_dir: Path | None = None
    nova3r_tokens_dir: Path | None = None
    load_nova3r_tokens: bool = False
    mesh_points_path: Path | None = None
    mesh_candidate_points: int | None = 200_000
    points_per_frame: int | None = None
    depth_tolerance: float = 0.05

    @property
    def room_dir(self) -> Path:
        return self.data_root / self.room

    @property
    def results_dir(self) -> Path:
        return self.room_dir / "results"

    @property
    def traj_path(self) -> Path:
        return self.room_dir / "traj.txt"

    @property
    def camera_path(self) -> Path:
        return self.data_root / "cam_params.json"


class ReplicaRoomDataset(Dataset):
    """Samples fixed-length 8-frame windows from a Replica room.

    When points_per_frame is set, targets are sampled from precomputed
    visible_points/frameXXXXXX.pt files. If those files are missing and
    mesh_points_path is provided, the dataset falls back to on-the-fly projection.
    """

    def __init__(self, config: ReplicaRoomConfig) -> None:
        self.config = config
        self.intrinsics, self.depth_scale, self.image_hw = self._load_camera()
        self.poses = self._load_poses()
        self.frame_ids = self._discover_frame_ids()
        self.mesh_points = self._load_mesh_points()
        self.visible_points_dir = self.config.visible_points_dir or visible_points_dir_for_room(
            self.config.data_root, self.config.room
        )
        self.nova3r_tokens_dir = self.config.nova3r_tokens_dir or nova3r_tokens_dir_for_room(
            self.config.data_root, self.config.room
        )

        if len(self.frame_ids) != len(self.poses):
            raise ValueError(
                f"{config.room} has {len(self.frame_ids)} RGB frames but "
                f"{len(self.poses)} poses in {config.traj_path}"
            )

        self._frame_id_set = set(self.frame_ids)

    def __len__(self) -> int:
        max_start = len(self.frame_ids) - (self.config.num_frames - 1) * self.config.frame_stride
        return max(0, max_start)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        num_frames = self.config.num_frames
        max_start = len(self.frame_ids) - 1 - (num_frames - 1) * self.config.frame_stride
        if max_start < 0:
            raise IndexError(
                f"Cannot sample {num_frames} frames with stride {self.config.frame_stride} "
                f"from only {len(self.frame_ids)} frames"
            )

        start = index % (max_start + 1)
        ids = [start + i * self.config.frame_stride for i in range(num_frames)]
        self._validate_window(ids)

        images = torch.stack([self._load_image(frame_id) for frame_id in ids])
        poses = torch.from_numpy(np.stack([self.poses[frame_id] for frame_id in ids])).float()
        intrinsics = torch.from_numpy(np.repeat(self.intrinsics[None], num_frames, axis=0)).float()

        sample: dict[str, torch.Tensor | str] = {
            "images": images,
            "poses": poses,
            "intrinsics": intrinsics,
            "frame_ids": torch.tensor(ids, dtype=torch.long),
            "room": self.config.room,
        }

        depths = None
        if self.config.load_depth:
            depths = torch.stack([self._load_depth(frame_id) for frame_id in ids])
            sample["depths"] = depths

        if self.config.points_per_frame is not None:
            targets = [
                self._sample_precomputed_visible_points(
                    frame_id, depths[i] if depths is not None else None
                )
                for i, frame_id in enumerate(ids)
            ]
            sample["target_points_world"] = torch.stack([target["points_world"] for target in targets])
            sample["target_uv"] = torch.stack([target["uv"] for target in targets])
            sample["target_depth"] = torch.stack([target["z"] for target in targets])
            sample["target_mask"] = torch.stack([target["mask"] for target in targets])

        if self.config.load_nova3r_tokens:
            token_payload = self._load_nova3r_tokens(start)
            sample["target_nova3r_tokens"] = token_payload["tokens"]
            sample["target_nova3r_frame_ids"] = token_payload["frame_ids"]

        return sample

    def _load_camera(self) -> tuple[np.ndarray, float, tuple[int, int]]:
        with self.config.camera_path.open("r", encoding="utf-8") as f:
            params = json.load(f)["camera"]

        intrinsics = np.array(
            [
                [params["fx"], 0.0, params["cx"]],
                [0.0, params["fy"], params["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return intrinsics, float(params["scale"]), (int(params["h"]), int(params["w"]))

    def _load_poses(self) -> np.ndarray:
        poses = np.loadtxt(self.config.traj_path, dtype=np.float32)
        if poses.ndim == 1:
            poses = poses[None]
        if poses.shape[1] != 16:
            raise ValueError(f"Expected 16 pose values per row in {self.config.traj_path}")
        return poses.reshape(-1, 4, 4)

    def _discover_frame_ids(self) -> list[int]:
        frame_paths = sorted(self.config.results_dir.glob("frame*.jpg"))
        if not frame_paths:
            raise FileNotFoundError(f"No frame*.jpg files found in {self.config.results_dir}")
        return [int(path.stem.removeprefix("frame")) for path in frame_paths]

    def _load_mesh_points(self) -> torch.Tensor | None:
        if self.config.mesh_points_path is None:
            return None

        payload = torch.load(self.config.mesh_points_path, map_location="cpu")
        points = payload["points"] if isinstance(payload, dict) else payload
        points = points.float()
        if (
            self.config.mesh_candidate_points is not None
            and points.shape[0] > self.config.mesh_candidate_points
        ):
            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(points.shape[0], generator=generator)[
                : self.config.mesh_candidate_points
            ]
            points = points[indices]
        return points

    def _sample_visible_points(self, frame_id: int, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.mesh_points is not None
        assert self.config.points_per_frame is not None

        visible = crop_visible_world_points(
            self.mesh_points,
            torch.from_numpy(self.poses[frame_id]).float(),
            torch.from_numpy(self.intrinsics).float(),
            depth,
            depth_tolerance=self.config.depth_tolerance,
        )

        return self._sample_target_points(visible)

    def _sample_precomputed_visible_points(
        self, frame_id: int, depth: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        visible_path = self.visible_points_dir / f"frame{frame_id:06d}.pt"
        if visible_path.exists():
            payload = torch.load(visible_path, map_location="cpu")
            visible = {
                "points_world": payload["points_world"].float(),
                "uv": payload["uv"].float(),
                "z": payload["z"].float(),
            }
        elif self.mesh_points is not None:
            if depth is None:
                depth = self._load_depth(frame_id)
            return self._sample_visible_points(frame_id, depth)
        else:
            raise FileNotFoundError(
                f"Missing precomputed visible points for frame {frame_id}: {visible_path}"
            )

        return self._sample_target_points(visible)

    def _sample_target_points(self, visible: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self.config.points_per_frame is not None

        target_count = self.config.points_per_frame
        visible_count = visible["points_world"].shape[0]
        if visible_count == 0:
            return {
                "points_world": torch.zeros((target_count, 3), dtype=torch.float32),
                "uv": torch.zeros((target_count, 2), dtype=torch.float32),
                "z": torch.zeros((target_count,), dtype=torch.float32),
                "mask": torch.zeros((target_count,), dtype=torch.bool),
            }

        if visible_count >= target_count:
            indices = torch.randperm(visible_count)[:target_count]
            mask = torch.ones((target_count,), dtype=torch.bool)
        else:
            padding = torch.randint(visible_count, (target_count - visible_count,))
            indices = torch.cat([torch.arange(visible_count), padding])
            mask = torch.zeros((target_count,), dtype=torch.bool)
            mask[:visible_count] = True

        return {
            "points_world": visible["points_world"][indices],
            "uv": visible["uv"][indices],
            "z": visible["z"][indices],
            "mask": mask,
        }

    def _load_nova3r_tokens(self, window_index: int) -> dict[str, torch.Tensor]:
        token_path = nova3r_tokens_path_for_window(self.nova3r_tokens_dir, window_index)
        if not token_path.exists():
            raise FileNotFoundError(f"Missing NOVA3R token target for window {window_index}: {token_path}")

        payload = torch.load(token_path, map_location="cpu")
        tokens = payload["tokens"].float()
        frame_ids = payload["frame_ids"].long()
        expected_ids = torch.tensor(
            [window_index + i * self.config.frame_stride for i in range(self.config.num_frames)],
            dtype=torch.long,
        )
        if not torch.equal(frame_ids, expected_ids):
            raise ValueError(
                f"NOVA3R token target {token_path} frame ids {frame_ids.tolist()} "
                f"do not match dataset window ids {expected_ids.tolist()}"
            )
        return {
            "tokens": tokens,
            "frame_ids": frame_ids,
        }

    def _validate_window(self, frame_ids: list[int]) -> None:
        missing = [
            frame_id
            for frame_id in frame_ids
            if frame_id not in self._frame_id_set
            or not self._image_path(frame_id).exists()
            or (self.config.load_depth and not self._depth_path(frame_id).exists())
        ]
        if missing:
            raise FileNotFoundError(f"Missing Replica files for frame ids: {missing}")

    def _load_image(self, frame_id: int) -> torch.Tensor:
        image = Image.open(self._image_path(frame_id)).convert("RGB")
        if self.config.image_size is not None:
            image = image.resize(self.config.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    def _load_depth(self, frame_id: int) -> torch.Tensor:
        depth = Image.open(self._depth_path(frame_id))
        if self.config.image_size is not None:
            depth = depth.resize(self.config.image_size, Image.NEAREST)
        array = np.asarray(depth, dtype=np.float32) / self.depth_scale
        return torch.from_numpy(array)

    def _image_path(self, frame_id: int) -> Path:
        return self.config.results_dir / f"frame{frame_id:06d}.jpg"

    def _depth_path(self, frame_id: int) -> Path:
        return self.config.results_dir / f"depth{frame_id:06d}.png"


def mesh_path_for_room(data_root: Path, room: str, *, high_res: bool = False) -> Path:
    suffix = "_mesh_ground_truth_10m.ply" if high_res else "_mesh.ply"
    return data_root / f"{room}{suffix}"


def mesh_points_path_for_room(data_root: Path, room: str, num_points: int) -> Path:
    return data_root / room / f"mesh_points_{num_points}.pt"


def visible_points_dir_for_room(data_root: Path, room: str) -> Path:
    return data_root / room / "visible_points"


def visible_points_path_for_frame(data_root: Path, room: str, frame_id: int) -> Path:
    return visible_points_dir_for_room(data_root, room) / f"frame{frame_id:06d}.pt"


def nova3r_tokens_dir_for_room(data_root: Path, room: str) -> Path:
    return data_root / room / "nova3r_tokens_first_camera"


def nova3r_tokens_path_for_window(tokens_dir: Path, window_index: int) -> Path:
    return tokens_dir / f"window{window_index:06d}.pt"


def project_world_points(
    points_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Project world-space points into one camera.

    Args:
        points_world: Tensor shaped [N, 3].
        world_to_camera: Transform shaped [4, 4].
        intrinsics: Camera intrinsics shaped [3, 3].
        image_hw: Image height and width.

    Returns:
        A dictionary with projected pixels, camera-space depth, and validity masks.
    """

    height, width = image_hw
    ones = torch.ones((points_world.shape[0], 1), dtype=points_world.dtype, device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=-1)
    points_camera = (world_to_camera.to(points_world.device, points_world.dtype) @ points_h.T).T[:, :3]

    z = points_camera[:, 2]
    positive_z = z > 0
    safe_z = torch.where(positive_z, z, torch.ones_like(z))

    intrinsics = intrinsics.to(points_world.device, points_world.dtype)
    u = intrinsics[0, 0] * points_camera[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / safe_z + intrinsics[1, 2]
    uv = torch.stack([u, v], dim=-1)

    inside = positive_z & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return {
        "uv": uv,
        "z": z,
        "points_camera": points_camera,
        "positive_z": positive_z,
        "inside": inside,
    }


def crop_visible_world_points(
    points_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    *,
    depth_tolerance: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Crop mesh points to those visible in one Replica frame.

    Replica trajectories are camera-to-world, so this function inverts the pose
    before projection. A point is considered visible if it projects inside the
    image and its projected depth agrees with the rendered depth map.
    """

    world_to_camera = torch.linalg.inv(camera_to_world)
    projection = project_world_points(
        points_world=points_world,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        image_hw=(depth.shape[-2], depth.shape[-1]),
    )

    inside = projection["inside"]
    uv_inside = projection["uv"][inside]
    z_inside = projection["z"][inside]
    points_inside = points_world[inside]

    if uv_inside.numel() == 0:
        empty_points = points_world.new_empty((0, 3))
        return {
            "points_world": empty_points,
            "uv": points_world.new_empty((0, 2)),
            "z": points_world.new_empty((0,)),
            "depth_error": points_world.new_empty((0,)),
        }

    pixels = uv_inside.round().long()
    pixels[:, 0].clamp_(0, depth.shape[-1] - 1)
    pixels[:, 1].clamp_(0, depth.shape[-2] - 1)
    sampled_depth = depth[pixels[:, 1], pixels[:, 0]].to(z_inside.device, z_inside.dtype)
    depth_error = (z_inside - sampled_depth).abs()
    visible = (sampled_depth > 0) & (depth_error <= depth_tolerance)

    return {
        "points_world": points_inside[visible],
        "uv": uv_inside[visible],
        "z": z_inside[visible],
        "depth_error": depth_error[visible],
    }
