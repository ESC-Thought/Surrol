#!/usr/bin/env python3
"""Render SurRoL DP3 point cloud NPZ episodes as videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "collected_data/bipeg_transfer_dp3_pointcloud.npz"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "collected_data/bipeg_transfer_dp3_pointcloud_episode0.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a DP3 point cloud NPZ episode to video.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive frame index inside the episode.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None, help="Uniformly subsample to at most this many frames.")
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=-65.0)
    parser.add_argument("--spin", action="store_true", help="Slowly rotate the 3D camera during rendering.")
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


def selected_frame_offsets(
    episode_len: int,
    start_frame: int,
    end_frame: Optional[int],
    stride: int,
    max_frames: Optional[int],
) -> List[int]:
    if stride <= 0:
        raise ValueError("--stride must be positive.")

    start_frame = max(0, start_frame)
    end_frame = episode_len if end_frame is None else min(end_frame, episode_len)
    frame_offsets = list(range(start_frame, end_frame, stride))
    if not frame_offsets:
        raise ValueError("No frames selected for video rendering.")

    if max_frames is not None and len(frame_offsets) > max_frames:
        selected_indices = np.linspace(0, len(frame_offsets) - 1, max_frames, dtype=int)
        frame_offsets = [frame_offsets[index] for index in selected_indices]

    return frame_offsets


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


def render_frame(
    figure: plt.Figure,
    axis_3d: plt.Axes,
    axis_xy: plt.Axes,
    points: np.ndarray,
    colors: Optional[np.ndarray],
    axis_min: np.ndarray,
    axis_max: np.ndarray,
    title: str,
    point_size: float,
    elev: float,
    azim: float,
    invert_y: bool,
    y_axis_label: str,
) -> np.ndarray:
    axis_3d.cla()
    axis_xy.cla()

    color_values = colors if colors is not None else points[:, 2]
    cmap = None if colors is not None else "viridis"

    axis_3d.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color_values,
        s=point_size,
        cmap=cmap,
        depthshade=False,
    )
    axis_3d.set_xlim(axis_min[0], axis_max[0])
    axis_3d.set_ylim(axis_min[1], axis_max[1])
    axis_3d.set_zlim(axis_min[2], axis_max[2])
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel(y_axis_label)
    axis_3d.set_zlabel("Z")
    axis_3d.view_init(elev=elev, azim=azim)
    axis_3d.set_title("3D point cloud")

    axis_xy.scatter(
        points[:, 0],
        points[:, 1],
        c=color_values,
        s=point_size,
        cmap=cmap,
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

    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()


def write_video(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    with np.load(input_path) as data:
        start, end = episode_bounds(data, args.episode)
        episode_len = end - start
        frame_offsets = selected_frame_offsets(
            episode_len,
            args.start_frame,
            args.end_frame,
            args.stride,
            args.max_frames,
        )
        global_indices = [start + frame_offset for frame_offset in frame_offsets]
        point_cloud = data["point_cloud"]
        point_color = data["point_color"] if args.use_rgb and "point_color" in data else None

        sampled_points_by_frame: List[np.ndarray] = []
        sampled_colors_by_frame: List[Optional[np.ndarray]] = []
        for frame_number, global_index in enumerate(global_indices):
            raw_colors = point_color[global_index] if point_color is not None else None
            sampled_points, sampled_colors = sample_points(
                point_cloud[global_index],
                raw_colors,
                args.max_points,
                seed=frame_number,
            )
            sampled_points = display_points(sampled_points, args.display_y_up)
            sampled_points_by_frame.append(sampled_points)
            sampled_colors_by_frame.append(sampled_colors)

        axis_min, axis_max = equal_axis_limits(sampled_points_by_frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        figure = plt.figure(figsize=(12.8, 7.2), dpi=args.dpi)
        axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
        axis_xy = figure.add_subplot(1, 2, 2)
        summary = metadata_summary(data)
        if args.display_y_up:
            summary = f"{summary}, display=y-up" if summary else "display=y-up"
        invert_y = not args.no_invert_y and not args.display_y_up

        with imageio.get_writer(output_path, fps=args.fps, macro_block_size=16) as writer:
            for render_index, (global_index, frame_offset, points, colors) in enumerate(
                zip(global_indices, frame_offsets, sampled_points_by_frame, sampled_colors_by_frame)
            ):
                azim = args.azim
                if args.spin and len(frame_offsets) > 1:
                    azim += 360.0 * render_index / len(frame_offsets)
                title = (
                    f"{input_path.name} | episode {args.episode}, frame {frame_offset}, "
                    f"index {global_index}"
                )
                if summary:
                    title = f"{title}\n{summary}"
                frame = render_frame(
                    figure,
                    axis_3d,
                    axis_xy,
                    points,
                    colors,
                    axis_min,
                    axis_max,
                    title,
                    args.point_size,
                    args.elev,
                    azim,
                    invert_y,
                    "Y (display up)" if args.display_y_up else "Y",
                )
                writer.append_data(frame)

        plt.close(figure)

    print(f"Saved point cloud video to {output_path}")


def main() -> None:
    args = parse_args()
    write_video(args)


if __name__ == "__main__":
    main()
