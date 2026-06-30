#!/usr/bin/env python3
"""Convert SurRoL BiPegTransfer pickle demos to single-view point cloud data."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "collected_data/peg_block_pick_psm2_cropped_v3"
DEFAULT_NUM_POINTS = 1024
DEFAULT_OUTPUT_PATH = REPO_ROOT / "collected_data/peg_block_pick_psm2_cropped_v3_pointcloud.npz"
DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256
DEFAULT_FOV_DEG = 25
DEFAULT_MASK_MODE = "target"
DEFAULT_POINT_NOISE_STD = 0.0
DEFAULT_POINT_NOISE_CLIP = 0.0
DEFAULT_DEPTH_NOISE_STD = 0.0
DEFAULT_DEPTH_NOISE_CLIP = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build single-view camera-frame point cloud trajectories from SurRoL pickle demos."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-format", choices=("npz", "zarr", "episode_npz"), default=None)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--mask-mode", choices=("target", "no_arm", "all"), default=DEFAULT_MASK_MODE)
    parser.add_argument("--action-mode", choices=("absolute", "dq"), default="absolute")
    parser.add_argument("--waypoints", default="all", help="Use 'all' or a comma list such as 0,1,2.")
    parser.add_argument("--episode-limit", "--max-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--point-noise-std", type=float, default=DEFAULT_POINT_NOISE_STD)
    parser.add_argument("--point-noise-clip", type=float, default=DEFAULT_POINT_NOISE_CLIP)
    parser.add_argument("--depth-noise-std", type=float, default=DEFAULT_DEPTH_NOISE_STD)
    parser.add_argument("--depth-noise-clip", type=float, default=DEFAULT_DEPTH_NOISE_CLIP)
    parser.add_argument("--include-rgb", action="store_true")
    return parser.parse_args()


def episode_index(path: Path) -> int:
    match = re.search(r"image_action_qpos_(\d+)\.pkl$", path.name)
    if not match:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def find_episode_paths(input_dir: Path, episode_limit: Optional[int]) -> List[Path]:
    episode_paths = sorted(input_dir.glob("image_action_qpos_*.pkl"), key=episode_index)
    if episode_limit is not None:
        episode_paths = episode_paths[:episode_limit]
    if not episode_paths:
        raise FileNotFoundError(
            f"No image_action_qpos_*.pkl files found in {input_dir}. "
            "If you are running from Methods/dp3-surrol, relative paths are "
            "resolved from the repository root when possible."
        )
    return episode_paths


def resolve_input_dir(input_dir: Path) -> Path:
    if input_dir.is_absolute() or input_dir.exists():
        return input_dir

    repo_candidate = REPO_ROOT / input_dir
    if repo_candidate.exists():
        return repo_candidate

    return input_dir


def resolve_output_path(output_path: Path) -> Path:
    if output_path.is_absolute() or output_path.parent.exists():
        return output_path

    repo_candidate = REPO_ROOT / output_path
    if repo_candidate.parent.exists():
        return repo_candidate

    return output_path


def build_camera_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal_length = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.asarray(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def parse_waypoint_filter(raw_waypoints: str, data: Dict[str, Any]) -> List[str]:
    if raw_waypoints == "all":
        return sorted(data["images"].keys(), key=lambda waypoint: int(waypoint))
    return [waypoint.strip() for waypoint in raw_waypoints.split(",") if waypoint.strip()]


def get_mask(
    data: Dict[str, Any],
    waypoint: str,
    frame_index: int,
    mask_mode: str,
) -> Optional[np.ndarray]:
    if mask_mode == "all":
        return None
    mask_key = "masks_target" if mask_mode == "target" else "masks_no_arm"
    mask_sequence = data.get(mask_key, {}).get(waypoint)
    if mask_sequence is None or frame_index >= len(mask_sequence):
        return None
    return np.asarray(mask_sequence[frame_index], dtype=bool)


def depth_to_camera_points(
    depth_image: np.ndarray,
    rgb_image: np.ndarray,
    mask_image: Optional[np.ndarray],
    camera_intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image_height, image_width = depth_image.shape
    pixel_x_grid, pixel_y_grid = np.meshgrid(
        np.arange(image_width, dtype=np.float32),
        np.arange(image_height, dtype=np.float32),
    )

    finite_depth_mask = np.isfinite(depth_image) & (depth_image > 0.0)
    if mask_image is not None:
        valid_mask = finite_depth_mask & mask_image
        if not np.any(valid_mask):
            valid_mask = finite_depth_mask
    else:
        valid_mask = finite_depth_mask

    depth_values = depth_image[valid_mask].astype(np.float32)
    point_x_values = (
        (pixel_x_grid[valid_mask] - camera_intrinsic[0, 2]) * depth_values / camera_intrinsic[0, 0]
    )
    point_y_values = (
        (pixel_y_grid[valid_mask] - camera_intrinsic[1, 2]) * depth_values / camera_intrinsic[1, 1]
    )
    point_z_values = depth_values

    points = np.stack([point_x_values, point_y_values, point_z_values], axis=-1).astype(np.float32)
    colors = rgb_image[valid_mask].astype(np.float32) / 255.0
    return points, colors


def sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    num_points: int,
    random_generator: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        raise ValueError("No valid depth points were available for sampling.")

    should_replace = points.shape[0] < num_points
    sampled_indices = random_generator.choice(points.shape[0], size=num_points, replace=should_replace)
    return points[sampled_indices], colors[sampled_indices]


def add_point_noise(
    points: np.ndarray,
    noise_std: float,
    noise_clip: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    if noise_std <= 0.0:
        return points.astype(np.float32, copy=False)
    noise = random_generator.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    if noise_clip > 0.0:
        noise = np.clip(noise, -noise_clip, noise_clip)
    return (points + noise).astype(np.float32)


def add_depth_noise(
    depth_image: np.ndarray,
    noise_std: float,
    noise_clip: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    if noise_std <= 0.0:
        return depth_image.astype(np.float32, copy=False)

    noisy_depth = depth_image.astype(np.float32, copy=True)
    valid_mask = np.isfinite(noisy_depth) & (noisy_depth > 0.0)
    if not np.any(valid_mask):
        return noisy_depth

    noise = random_generator.normal(0.0, noise_std, size=noisy_depth.shape).astype(np.float32)
    if noise_clip > 0.0:
        noise = np.clip(noise, -noise_clip, noise_clip)

    noisy_depth[valid_mask] = np.maximum(noisy_depth[valid_mask] + noise[valid_mask], 1e-6)
    return noisy_depth


def required_frame_count(data: Dict[str, Any], waypoint: str, action_key: str) -> int:
    required_sequences = [
        data["images"][waypoint],
        data["depths"][waypoint],
        data[action_key][waypoint],
        data["qpos"][waypoint],
    ]
    return min(len(sequence) for sequence in required_sequences)


def build_pointcloud_dataset(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    episode_paths = find_episode_paths(args.input_dir, args.episode_limit)
    camera_intrinsic = build_camera_intrinsic(args.image_width, args.image_height, args.fov_deg)
    random_generator = np.random.default_rng(args.seed)
    action_key = "dq_actions" if args.action_mode == "dq" else "actions"

    point_cloud_frames: List[np.ndarray] = []
    point_color_frames: List[np.ndarray] = []
    action_frames: List[np.ndarray] = []
    agent_pos_frames: List[np.ndarray] = []
    waypoint_frames: List[int] = []
    episode_id_frames: List[int] = []
    episode_ends: List[int] = []

    for episode_path in episode_paths:
        with episode_path.open("rb") as pickle_file:
            data = pickle.load(pickle_file)

        episode_frame_count = 0
        selected_waypoints = parse_waypoint_filter(args.waypoints, data)
        for waypoint in selected_waypoints:
            if waypoint not in data.get("images", {}) or waypoint not in data.get("depths", {}):
                continue
            if waypoint not in data.get(action_key, {}) or waypoint not in data.get("qpos", {}):
                continue

            frame_count = required_frame_count(data, waypoint, action_key)
            for frame_index in range(frame_count):
                rgb_pair = data["images"][waypoint][frame_index]
                rgb_image = np.asarray(rgb_pair[0], dtype=np.uint8)
                depth_image = np.asarray(data["depths"][waypoint][frame_index], dtype=np.float32)
                depth_image = add_depth_noise(
                    depth_image=depth_image,
                    noise_std=args.depth_noise_std,
                    noise_clip=args.depth_noise_clip,
                    random_generator=random_generator,
                )
                mask_image = get_mask(data, waypoint, frame_index, args.mask_mode)

                points, colors = depth_to_camera_points(
                    depth_image=depth_image,
                    rgb_image=rgb_image,
                    mask_image=mask_image,
                    camera_intrinsic=camera_intrinsic,
                )
                sampled_points, sampled_colors = sample_points(
                    points=points,
                    colors=colors,
                    num_points=args.num_points,
                    random_generator=random_generator,
                )
                sampled_points = add_point_noise(
                    points=sampled_points,
                    noise_std=args.point_noise_std,
                    noise_clip=args.point_noise_clip,
                    random_generator=random_generator,
                )

                point_cloud_frames.append(sampled_points)
                if args.include_rgb:
                    point_color_frames.append(sampled_colors)
                action_frames.append(np.asarray(data[action_key][waypoint][frame_index], dtype=np.float32))
                agent_pos_frames.append(np.asarray(data["qpos"][waypoint][frame_index], dtype=np.float32))
                waypoint_frames.append(int(waypoint))
                episode_id_frames.append(episode_index(episode_path))
                episode_frame_count += 1

        if episode_frame_count > 0:
            episode_ends.append(len(point_cloud_frames))

    if not point_cloud_frames:
        raise RuntimeError("No frames were converted. Check input files, waypoints, and action mode.")

    dataset = {
        "point_cloud": np.stack(point_cloud_frames, axis=0).astype(np.float32),
        "action": np.stack(action_frames, axis=0).astype(np.float32),
        "agent_pos": np.stack(agent_pos_frames, axis=0).astype(np.float32),
        "waypoint": np.asarray(waypoint_frames, dtype=np.int32),
        "episode_id": np.asarray(episode_id_frames, dtype=np.int32),
        "episode_ends": np.asarray(episode_ends, dtype=np.int64),
        "camera_intrinsic": camera_intrinsic,
    }
    if args.include_rgb:
        dataset["point_color"] = np.stack(point_color_frames, axis=0).astype(np.float32)

    metadata = {
        "source": "SurRoL BiPegTransfer pickle demonstrations",
        "input_dir": str(args.input_dir),
        "num_episodes": len(episode_ends),
        "num_frames": int(dataset["point_cloud"].shape[0]),
        "num_points": args.num_points,
        "point_cloud_frame": "camera",
        "mask_mode": args.mask_mode,
        "action_mode": args.action_mode,
        "action_key": action_key,
        "waypoints": args.waypoints,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "fov_deg": args.fov_deg,
        "camera_intrinsic": camera_intrinsic.tolist(),
        "includes_point_color": args.include_rgb,
        "point_noise_std": args.point_noise_std,
        "point_noise_clip": args.point_noise_clip,
        "depth_noise_std": args.depth_noise_std,
        "depth_noise_clip": args.depth_noise_clip,
    }
    return dataset, metadata


def infer_output_format(output_path: Path, output_format: Optional[str]) -> str:
    if output_format is not None:
        return output_format
    if output_path.suffix == ".zarr":
        return "zarr"
    return "npz"


def save_npz_dataset(output_path: Path, dataset: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **dataset, metadata=np.asarray(json.dumps(metadata)))
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_zarr_dataset(output_path: Path, dataset: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> None:
    try:
        import zarr
    except ImportError as error:
        raise ImportError("zarr is not installed. Install zarr or use --output-format npz.") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    for key in ("point_cloud", "point_color", "action", "agent_pos", "waypoint", "episode_id"):
        if key in dataset:
            data_group.create_dataset(key, data=dataset[key], shape=dataset[key].shape, dtype=dataset[key].dtype)

    meta_group.create_dataset(
        "episode_ends",
        data=dataset["episode_ends"],
        shape=dataset["episode_ends"].shape,
        dtype=dataset["episode_ends"].dtype,
    )
    meta_group.attrs["metadata_json"] = json.dumps(metadata)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def episode_output_dir(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix("")
    return output_path


def save_episode_npz_dataset(
    output_path: Path,
    dataset: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> None:
    output_dir = episode_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_ends = dataset["episode_ends"]
    episode_starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    total_frames = dataset["point_cloud"].shape[0]
    frame_keys = [
        "point_cloud",
        "point_color",
        "action",
        "agent_pos",
        "waypoint",
        "episode_id",
    ]

    for episode_number, (episode_start, episode_end) in enumerate(zip(episode_starts, episode_ends)):
        source_episode_id = int(dataset["episode_id"][episode_start])
        episode_dataset: Dict[str, np.ndarray] = {
            "camera_intrinsic": dataset["camera_intrinsic"],
        }
        for key in frame_keys:
            if key in dataset and dataset[key].shape[0] == total_frames:
                episode_dataset[key] = dataset[key][episode_start:episode_end]

        episode_metadata = dict(metadata)
        episode_metadata.update(
            {
                "episode_number": int(episode_number),
                "source_episode_id": source_episode_id,
                "num_frames": int(episode_end - episode_start),
                "episode_start": int(episode_start),
                "episode_end": int(episode_end),
            }
        )
        episode_path = output_dir / f"episode_{source_episode_id:04d}.npz"
        np.savez_compressed(
            episode_path,
            **episode_dataset,
            metadata=np.asarray(json.dumps(episode_metadata)),
        )

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.input_dir = resolve_input_dir(args.input_dir)
    args.output = resolve_output_path(args.output)
    dataset, metadata = build_pointcloud_dataset(args)
    output_format = infer_output_format(args.output, args.output_format)

    if output_format == "zarr":
        save_zarr_dataset(args.output, dataset, metadata)
    elif output_format == "episode_npz":
        save_episode_npz_dataset(args.output, dataset, metadata)
    else:
        save_npz_dataset(args.output, dataset, metadata)

    print(f"Saved {metadata['num_frames']} frames from {metadata['num_episodes']} episodes")
    print(f"point_cloud: {dataset['point_cloud'].shape}")
    print(f"action: {dataset['action'].shape}")
    print(f"agent_pos: {dataset['agent_pos'].shape}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
