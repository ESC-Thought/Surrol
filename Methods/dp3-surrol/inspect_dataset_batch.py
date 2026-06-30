#!/usr/bin/env python3
"""Inspect DP3-style SurRoL BiPegTransfer dataset batches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import DataLoader

from surrol_bipeg_dataset import DEFAULT_DATA_PATH, DEFAULT_NUM_POINTS, SurrolBipegSequenceDataset, make_train_val_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SurRoL DP3 point-cloud dataloader output.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", choices=["train", "val", "both", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--include-rgb", action="store_true")
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    return parser.parse_args()


def tensor_summary(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bool:
        true_ratio = tensor.float().mean().item()
        return f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} true_ratio={true_ratio:.4f}"
    if not tensor.is_floating_point():
        return f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} first={tensor.flatten()[0].item()}"
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"min={tensor.min().item():.4f} max={tensor.max().item():.4f} "
        f"mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}"
    )


def print_summary(prefix: str, values: Mapping[str, object]) -> None:
    print(f"\n[{prefix}]")
    for key, value in values.items():
        if key == "selected_episode_indices":
            print(f"{key}: {value}")
        elif key == "metadata":
            metadata = value if isinstance(value, dict) else {}
            print(
                "metadata: "
                f"mask={metadata.get('mask_mode', 'unknown')}, "
                f"action={metadata.get('action_mode', 'unknown')}, "
                f"points={metadata.get('num_points', 'unknown')}"
            )
        else:
            print(f"{key}: {value}")


def inspect_dataset(name: str, dataset: SurrolBipegSequenceDataset, args: argparse.Namespace) -> None:
    print_summary(f"{name} dataset", dataset.summary())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    batch = next(iter(dataloader))
    print(f"\n[{name} batch]")
    print(tensor_summary("obs.point_cloud", batch["obs"]["point_cloud"]))
    print(tensor_summary("obs.agent_pos", batch["obs"]["agent_pos"]))
    print(tensor_summary("action", batch["action"]))
    if "episode_index" in batch:
        print(tensor_summary("episode_index", batch["episode_index"]))
        print(tensor_summary("timestep", batch["timestep"]))
        print(tensor_summary("global_index", batch["global_index"]))


def main() -> None:
    args = parse_args()
    if args.split in ("train", "val", "both"):
        train_dataset, val_dataset = make_train_val_datasets(
            data_path=args.data,
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            val_ratio=args.val_ratio,
            seed=args.seed,
            sample_stride=args.sample_stride,
            normalize=args.normalize,
            include_rgb=args.include_rgb,
            return_info=True,
            num_points=args.num_points,
        )
        if args.split in ("train", "both"):
            inspect_dataset("train", train_dataset, args)
        if args.split in ("val", "both"):
            inspect_dataset("val", val_dataset, args)
    else:
        dataset = SurrolBipegSequenceDataset(
            data_path=args.data,
            split="all",
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            val_ratio=args.val_ratio,
            seed=args.seed,
            sample_stride=args.sample_stride,
            normalize=args.normalize,
            include_rgb=args.include_rgb,
            return_info=True,
            num_points=args.num_points,
        )
        inspect_dataset("all", dataset, args)


if __name__ == "__main__":
    main()
