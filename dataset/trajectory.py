"""Dataset-neutral trajectory windowing; Push-T is one possible data source."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Trajectory:
    frames: torch.Tensor  # [T, C, H, W], float in [0, 1]
    actions: torch.Tensor  # [T-1, A]


def _to_frames(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(value)
    if x.ndim != 4:
        raise ValueError(f"frames must be [T,C,H,W] or [T,H,W,C], got {tuple(x.shape)}")
    if x.shape[-1] in (1, 3, 4) and x.shape[1] not in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    x = x.float()
    if x.max() > 1:
        x = x / 255.0
    return x.contiguous()


class TrajectoryDataset(Dataset):
    """Turns trajectories into causal prediction windows.

    Each item contains K history frames, K-1 actions between those frames,
    a current action, and the next frame. The shifted history is provided for
    computing the self-supervised next-agent-state target.
    """

    def __init__(self, trajectories: Sequence[Trajectory], history_length: int = 4, rollout_length: int = 1):
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if rollout_length < 1:
            raise ValueError("rollout_length must be positive")
        self.trajectories = list(trajectories)
        self.history_length = history_length
        self.rollout_length = rollout_length
        self.indices: list[tuple[int, int]] = []
        for trajectory_id, trajectory in enumerate(self.trajectories):
            if len(trajectory.actions) != len(trajectory.frames) - 1:
                raise ValueError("actions must describe every frame-to-frame transition")
            for current in range(history_length - 1, len(trajectory.frames) - rollout_length):
                self.indices.append((trajectory_id, current))

    @classmethod
    def from_npz(cls, paths: Sequence[str | Path], history_length: int = 4,
                 rollout_length: int = 1) -> "TrajectoryDataset":
        trajectories = []
        for path in paths:
            with np.load(path) as data:
                trajectories.append(Trajectory(_to_frames(data["frames"]), torch.as_tensor(data["actions"]).float()))
        return cls(trajectories, history_length, rollout_length)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory_id, current = self.indices[index]
        trajectory = self.trajectories[trajectory_id]
        start = current - self.history_length + 1
        sample = {
            "history_frames": trajectory.frames[start : current + 1],
            "history_actions": trajectory.actions[start:current],
            "current_action": trajectory.actions[current],
            "current_frame": trajectory.frames[current],
            "next_frame": trajectory.frames[current + 1],
            "next_history_frames": trajectory.frames[start + 1 : current + 2],
            "next_history_actions": trajectory.actions[start + 1 : current + 1],
        }
        if self.rollout_length > 1:
            sample["rollout_actions"] = trajectory.actions[current : current + self.rollout_length]
            sample["goal_frame"] = trajectory.frames[current + self.rollout_length]
        return sample


def collate_trajectory_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
