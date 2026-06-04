#!/usr/bin/env python3
"""PyTorch dataset adapter for SurRoL BiPegTransfer DP3 point-cloud data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "collected_data/bipeg_transfer_dp3_pointcloud.npz"

SplitName = Literal["train", "val", "all"]
Normalizer = Dict[str, Dict[str, np.ndarray]]


@dataclass(frozen=True)
class EpisodeRange:
    episode_index: int
    start: int
    end: int


@dataclass(frozen=True)
class SequenceSample:
    episode_index: int
    episode_start: int
    episode_end: int
    current_index: int


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path

    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists() or repo_candidate.parent.exists():
        return repo_candidate

    return path


def parse_metadata(raw_metadata: object) -> Dict[str, object]:
    if raw_metadata is None:
        return {}
    try:
        return json.loads(str(raw_metadata))
    except json.JSONDecodeError:
        return {}


def build_episode_ranges(episode_ends: np.ndarray) -> List[EpisodeRange]:
    ranges: List[EpisodeRange] = []
    episode_start = 0
    for episode_index, episode_end in enumerate(episode_ends.tolist()):
        episode_end = int(episode_end)
        if episode_end <= episode_start:
            raise ValueError(f"Episode {episode_index} has invalid end index {episode_end}.")
        ranges.append(EpisodeRange(episode_index=episode_index, start=episode_start, end=episode_end))
        episode_start = episode_end
    return ranges


def split_episode_indices(num_episodes: int, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")
    episode_indices = np.arange(num_episodes)
    random_generator = np.random.default_rng(seed)
    random_generator.shuffle(episode_indices)

    val_count = int(round(num_episodes * val_ratio))
    if val_ratio > 0.0:
        val_count = max(1, val_count)
    val_count = min(val_count, num_episodes - 1) if num_episodes > 1 else 0

    val_indices = sorted(int(episode_index) for episode_index in episode_indices[:val_count])
    train_indices = sorted(int(episode_index) for episode_index in episode_indices[val_count:])
    return train_indices, val_indices


def window_indices(window_start: int, horizon: int, episode_start: int, episode_end: int) -> np.ndarray:
    raw_indices = np.arange(window_start, window_start + horizon, dtype=np.int64)
    return np.clip(raw_indices, episode_start, episode_end - 1)


def compute_mean_std(values: np.ndarray, axes: Tuple[int, ...], eps: float) -> Dict[str, np.ndarray]:
    mean = values.mean(axis=axes).astype(np.float32)
    std = values.std(axis=axes).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    return {"mean": mean, "std": std}


class SurrolBipegSequenceDataset(Dataset):
    def __init__(
        self,
        data_path: Path = DEFAULT_DATA_PATH,
        split: SplitName = "train",
        obs_horizon: int = 2,
        action_horizon: int = 16,
        val_ratio: float = 0.2,
        seed: int = 0,
        sample_stride: int = 1,
        normalize: bool = False,
        normalizer: Optional[Normalizer] = None,
        include_rgb: bool = False,
        return_info: bool = False,
        eps: float = 1e-6,
    ) -> None:
        if obs_horizon <= 0:
            raise ValueError("obs_horizon must be positive.")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if sample_stride <= 0:
            raise ValueError("sample_stride must be positive.")
        if split not in ("train", "val", "all"):
            raise ValueError("split must be one of: train, val, all.")

        self.data_path = resolve_path(Path(data_path))
        self.split = split
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.val_ratio = val_ratio
        self.seed = seed
        self.sample_stride = sample_stride
        self.normalize = normalize
        self.return_info = return_info
        self.include_rgb = include_rgb
        self.eps = eps

        with np.load(self.data_path, allow_pickle=False) as data:
            self.point_cloud = data["point_cloud"].astype(np.float32, copy=False)
            self.action = data["action"].astype(np.float32, copy=False)
            self.agent_pos = data["agent_pos"].astype(np.float32, copy=False)
            episode_ends = data["episode_ends"].astype(np.int64, copy=False)
            raw_metadata = data["metadata"] if "metadata" in data else None
            self.metadata = parse_metadata(raw_metadata)
            if include_rgb:
                if "point_color" not in data:
                    raise KeyError("include_rgb=True requires point_color in the NPZ.")
                point_color = data["point_color"].astype(np.float32, copy=False)
                self.point_cloud = np.concatenate([self.point_cloud, point_color], axis=-1)

        if self.point_cloud.shape[0] != self.action.shape[0] or self.action.shape[0] != self.agent_pos.shape[0]:
            raise ValueError("point_cloud, action, and agent_pos must have the same first dimension.")

        self.all_episode_ranges = build_episode_ranges(episode_ends)
        train_episode_indices, val_episode_indices = split_episode_indices(
            len(self.all_episode_ranges),
            val_ratio,
            seed,
        )
        if split == "train":
            selected_episode_indices = train_episode_indices
        elif split == "val":
            selected_episode_indices = val_episode_indices
        else:
            selected_episode_indices = list(range(len(self.all_episode_ranges)))
        if not selected_episode_indices:
            raise ValueError(f"No episodes selected for split={split}.")

        selected_episode_set = set(selected_episode_indices)
        self.episode_ranges = [
            episode_range
            for episode_range in self.all_episode_ranges
            if episode_range.episode_index in selected_episode_set
        ]
        self.selected_episode_indices = [episode_range.episode_index for episode_range in self.episode_ranges]
        self.samples = self.build_samples()
        self.normalizer = normalizer
        if normalize and self.normalizer is None:
            self.normalizer = self.compute_normalizer()

    def build_samples(self) -> List[SequenceSample]:
        samples: List[SequenceSample] = []
        for episode_range in self.episode_ranges:
            for current_index in range(episode_range.start, episode_range.end, self.sample_stride):
                samples.append(
                    SequenceSample(
                        episode_index=episode_range.episode_index,
                        episode_start=episode_range.start,
                        episode_end=episode_range.end,
                        current_index=current_index,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, sample_index: int) -> Dict[str, object]:
        sample = self.samples[sample_index]
        obs_start = sample.current_index - self.obs_horizon + 1
        obs_indices = window_indices(obs_start, self.obs_horizon, sample.episode_start, sample.episode_end)
        action_indices = window_indices(sample.current_index, self.action_horizon, sample.episode_start, sample.episode_end)

        point_cloud = self.point_cloud[obs_indices].astype(np.float32, copy=True)
        agent_pos = self.agent_pos[obs_indices].astype(np.float32, copy=True)
        action = self.action[action_indices].astype(np.float32, copy=True)

        if self.normalize:
            point_cloud = self.normalize_array("point_cloud", point_cloud)
            agent_pos = self.normalize_array("agent_pos", agent_pos)
            action = self.normalize_array("action", action)

        item: Dict[str, object] = {
            "obs": {
                "point_cloud": torch.from_numpy(point_cloud),
                "agent_pos": torch.from_numpy(agent_pos),
            },
            "action": torch.from_numpy(action),
        }
        if self.return_info:
            item["episode_index"] = torch.tensor(sample.episode_index, dtype=torch.long)
            item["timestep"] = torch.tensor(sample.current_index - sample.episode_start, dtype=torch.long)
            item["global_index"] = torch.tensor(sample.current_index, dtype=torch.long)
        return item

    def normalize_array(self, key: str, values: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            raise RuntimeError("normalize=True requires a normalizer.")
        stats = self.normalizer[key]
        return ((values - stats["mean"]) / stats["std"]).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            raise RuntimeError("No normalizer is available.")
        stats = self.normalizer["action"]
        return (action * stats["std"] + stats["mean"]).astype(np.float32)

    def set_normalizer(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def get_normalizer(self) -> Normalizer:
        if self.normalizer is None:
            self.normalizer = self.compute_normalizer()
        return self.normalizer

    def selected_frame_indices(self) -> np.ndarray:
        frame_indices = [
            np.arange(episode_range.start, episode_range.end, dtype=np.int64)
            for episode_range in self.episode_ranges
        ]
        return np.concatenate(frame_indices, axis=0)

    def compute_normalizer(self) -> Normalizer:
        frame_indices = self.selected_frame_indices()
        return {
            "point_cloud": compute_mean_std(self.point_cloud[frame_indices], axes=(0, 1), eps=self.eps),
            "agent_pos": compute_mean_std(self.agent_pos[frame_indices], axes=(0,), eps=self.eps),
            "action": compute_mean_std(self.action[frame_indices], axes=(0,), eps=self.eps),
        }

    def summary(self) -> Dict[str, object]:
        return {
            "data_path": str(self.data_path),
            "split": self.split,
            "num_total_episodes": len(self.all_episode_ranges),
            "num_selected_episodes": len(self.episode_ranges),
            "selected_episode_indices": self.selected_episode_indices,
            "num_samples": len(self),
            "obs_horizon": self.obs_horizon,
            "action_horizon": self.action_horizon,
            "point_cloud_shape": tuple(self.point_cloud.shape[1:]),
            "agent_pos_shape": tuple(self.agent_pos.shape[1:]),
            "action_shape": tuple(self.action.shape[1:]),
            "normalize": self.normalize,
            "metadata": self.metadata,
        }


def make_train_val_datasets(
    data_path: Path = DEFAULT_DATA_PATH,
    obs_horizon: int = 2,
    action_horizon: int = 16,
    val_ratio: float = 0.2,
    seed: int = 0,
    sample_stride: int = 1,
    normalize: bool = False,
    include_rgb: bool = False,
    return_info: bool = False,
) -> Tuple[SurrolBipegSequenceDataset, SurrolBipegSequenceDataset]:
    train_dataset = SurrolBipegSequenceDataset(
        data_path=data_path,
        split="train",
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        val_ratio=val_ratio,
        seed=seed,
        sample_stride=sample_stride,
        normalize=False,
        include_rgb=include_rgb,
        return_info=return_info,
    )
    shared_normalizer = train_dataset.get_normalizer()
    train_dataset.normalize = normalize
    val_dataset = SurrolBipegSequenceDataset(
        data_path=data_path,
        split="val",
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        val_ratio=val_ratio,
        seed=seed,
        sample_stride=sample_stride,
        normalize=normalize,
        normalizer=shared_normalizer,
        include_rgb=include_rgb,
        return_info=return_info,
    )
    return train_dataset, val_dataset
