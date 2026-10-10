"""Observed action branches sharing one complete simulator start; no states in batches."""
import glob

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class ActionBranchDataset(Dataset):
    def __init__(self, paths, image_size=96):
        self.paths = sorted({p for pattern in paths for p in glob.glob(str(pattern))})
        if not self.paths:
            raise ValueError("No branch data: run scripts/prepare_action_matching.py first")
        self.image_size, self.cache, self.indices, self.episode_ids = image_size, {}, [], set()
        self.ids = []
        k = None
        for file_id, path in enumerate(self.paths):
            with np.load(path) as data:
                actions, future = data["actions"], data["future_frames"]
                assert actions.ndim == 3 and actions.shape[-1] == 2 and actions.shape[1] >= 2
                assert future.shape[:2] == actions.shape[:2] and len(data["current_frames"]) == len(actions)
                assert data["positive_mask"].shape == (*actions.shape[:2], actions.shape[1])
                assert data["positive_mask"].dtype == np.bool_ and data["positive_mask"].any(-1).all()
                assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.00001
                assert int(data["frame_skip"]) == 1 and str(data["action_mode"]) == "relative"
                if k is not None and actions.shape[1] != k:
                    raise ValueError("all branch files must use the same K")
                k = actions.shape[1]
                self.episode_ids.add(int(data["source_episode"]))
                self.ids.append(int(data["source_episode"]))
                self.indices.extend((file_id, i) for i in range(len(actions)))
        if not self.indices:
            raise ValueError("empty branch dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        file_id, i = self.indices[index]
        if file_id not in self.cache:
            with np.load(self.paths[file_id]) as data:
                # Explicit whitelist: simulator metadata cannot enter training.
                self.cache[file_id] = {key: data[key] for key in ("current_frames", "future_frames", "actions", "positive_mask", "timesteps")}
        data = self.cache[file_id]
        frames = np.concatenate((data["current_frames"][i:i+1], data["future_frames"][i]), 0)
        images = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()/255
        images = F.interpolate(images, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return {"current_frame": images[0], "future_frames": images[1:],
                "actions": torch.from_numpy(data["actions"][i].copy()),
                "positive_mask": torch.from_numpy(data["positive_mask"][i].copy()),
                "timestep": int(data["timesteps"][i]), "episode": self.ids[file_id]}


def equivalent_futures(frames, tolerance=0.0):
    """Pixel-only equivalence, never latent-based (which would reward collapse)."""
    if tolerance < 0:
        raise ValueError("pixel equivalence tolerance must be nonnegative")
    x = frames.astype(np.float32)/255
    # Connected components provide a consistent equivalence relation even when
    # a nonzero threshold gives a non-transitive pairwise proximity relation.
    close = np.abs(x[:, None]-x[None]).mean((2, 3, 4)) <= tolerance
    for i in range(len(frames)):
        close |= close[:, i:i+1] & close[i:i+1, :]
    np.fill_diagonal(close, True)
    return close
