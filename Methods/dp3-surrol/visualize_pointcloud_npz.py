#!/usr/bin/env python3
"""Visualize SurRoL DP3 point cloud NPZ files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "collected_data/bipeg_transfer_dp3_pointcloud.npz"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "collected_data/bipeg_transfer_dp3_pointcloud_preview.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DP3 point cloud NPZ data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", default=None, help="Comma-separated frame ids inside the selected episode.")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-rgb", action="store_true")
    parser.add_argument(
        "--display-y-up",
        action="store_true",
        help="Flip camera-frame Y only for visualization; the NPZ data is not modified.",
    )
    parser.add_argument(
        "--no-invert-y",
        action="store_true",
        help="Keep the raw camera-frame Y axis in XY projections.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path

    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists() or repo_candidate.parent.exists():
        return repo_candidate

    return path


def episode_bounds(data: np.lib.npyio.NpzFile, episode: int) -> Tuple[int, int]:
    if "episode_ends" not in data:
        return 0, int(data["point_cloud"].shape[0])

    episode_ends = data["episode_ends"]
    if episode < 0 or episode >= len(episode_ends):
        raise IndexError(f"Episode {episode} is out of range for {len(episode_ends)} episodes.")

    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def selected_frame_offsets(raw_frames: Optional[str], num_frames: int, episode_len: int) -> List[int]:
    if raw_frames:
        frames = [int(frame.strip()) for frame in raw_frames.split(",") if frame.strip()]
        return [frame for frame in frames if 0 <= frame < episode_len]

    if episode_len <= num_frames:
        return list(range(episode_len))

    return np.linspace(0, episode_len - 1, num_frames, dtype=int).tolist()


def sample_points(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.shape[0] <= max_points:
        return points, colors
    random_generator = np.random.default_rng(seed)
    sampled_indices = random_generator.choice(points.shape[0], size=max_points, replace=False)
    sampled_colors = colors[sampled_indices] if colors is not None else None
    return points[sampled_indices], sampled_colors


def display_points(points: np.ndarray, display_y_up: bool) -> np.ndarray:
    if not display_y_up:
        return points
    transformed_points = points.copy()
    transformed_points[:, 1] *= -1.0
    return transformed_points


def equal_axis_limits(points_by_frame: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate(points_by_frame, axis=0)
    axis_min = stacked.min(axis=0)
    axis_max = stacked.max(axis=0)
    axis_center = (axis_min + axis_max) / 2.0
    axis_radius = np.max(axis_max - axis_min) / 2.0
    axis_radius = max(float(axis_radius), 1e-6)
    return axis_center - axis_radius, axis_center + axis_radius


def metadata_summary(data: np.lib.npyio.NpzFile) -> str:
    if "metadata" not in data:
        return ""
    try:
        metadata = json.loads(str(data["metadata"]))
    except json.JSONDecodeError:
        return ""
    return (
        f"mask={metadata.get('mask_mode', 'unknown')}, "
        f"action={metadata.get('action_mode', 'unknown')}, "
        f"points={metadata.get('num_points', 'unknown')}"
    )


def plot_preview(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    with np.load(input_path) as data:
        start, end = episode_bounds(data, args.episode)
        episode_len = end - start
        frame_offsets = selected_frame_offsets(args.frames, args.num_frames, episode_len)
        if not frame_offsets:
            raise ValueError("No valid frames selected for visualization.")

        global_indices = [start + frame_offset for frame_offset in frame_offsets]
        point_frames: List[np.ndarray] = []
        color_frames: List[Optional[np.ndarray]] = []
        for frame_number, global_index in enumerate(global_indices):
            raw_colors = data["point_color"][global_index] if args.use_rgb and "point_color" in data else None
            sampled_points, sampled_colors = sample_points(
                data["point_cloud"][global_index],
                raw_colors,
                args.max_points,
                args.seed + frame_number,
            )
            sampled_points = display_points(sampled_points, args.display_y_up)
            point_frames.append(sampled_points)
            color_frames.append(sampled_colors)

        axis_min, axis_max = equal_axis_limits(point_frames)
        invert_y = not args.no_invert_y and not args.display_y_up
        y_axis_label = "Y (display up)" if args.display_y_up else "Y"
        figure = plt.figure(figsize=(4.0 * len(global_indices), 7.0))
        axes_3d = [
            figure.add_subplot(2, len(global_indices), plot_index + 1, projection="3d")
            for plot_index in range(len(global_indices))
        ]
        axes_xy = [
            figure.add_subplot(2, len(global_indices), len(global_indices) + plot_index + 1)
            for plot_index in range(len(global_indices))
        ]

        for plot_index, (global_index, frame_offset, points) in enumerate(
            zip(global_indices, frame_offsets, point_frames)
        ):
            sampled_colors = color_frames[plot_index]
            colors = sampled_colors if sampled_colors is not None else points[:, 2]

            axis_3d = axes_3d[plot_index]
            axis_3d.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                c=colors,
                s=args.point_size,
                cmap=None if sampled_colors is not None else "viridis",
                depthshade=False,
            )
            axis_3d.set_xlim(axis_min[0], axis_max[0])
            axis_3d.set_ylim(axis_min[1], axis_max[1])
            axis_3d.set_zlim(axis_min[2], axis_max[2])
            axis_3d.set_xlabel("X")
            axis_3d.set_ylabel(y_axis_label)
            axis_3d.set_zlabel("Z")
            axis_3d.view_init(elev=20, azim=-65)
            axis_3d.set_title(f"episode {args.episode}, frame {frame_offset}\nindex {global_index}")

            axis_xy = axes_xy[plot_index]
            axis_xy.scatter(
                points[:, 0],
                points[:, 1],
                c=colors,
                s=args.point_size,
                cmap=None if sampled_colors is not None else "viridis",
            )
            axis_xy.set_xlim(axis_min[0], axis_max[0])
            axis_xy.set_ylim(axis_min[1], axis_max[1])
            axis_xy.set_xlabel("X")
            axis_xy.set_ylabel(y_axis_label)
            if invert_y:
                axis_xy.invert_yaxis()
                axis_xy.set_ylabel("Y (image down)")
            axis_xy.set_aspect("equal", adjustable="box")
            axis_xy.set_title("XY projection")
            axis_xy.grid(True, alpha=0.35)

        figure.suptitle(f"{input_path.name} | {metadata_summary(data)}", fontsize=14)
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

    print(f"Saved preview to {output_path}")


def main() -> None:
    args = parse_args()
    plot_preview(args)


if __name__ == "__main__":
    main()
