"""Raw-step transitions; states stay in NPZ for environment reset, never in model batches."""
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F


class ActionContrastiveDataset(Dataset):
    def __init__(self, paths, image_size=96, horizon=1, min_action_norm=0.0):
        if horizon < 1 or image_size < 1 or min_action_norm < 0:
            raise ValueError("horizon/image_size must be positive and min_action_norm nonnegative")
        self.paths = sorted({p for pattern in paths for p in glob.glob(str(pattern))})
        if not self.paths:
            raise ValueError(f"no NPZ files matched {paths}")
        self.image_size, self.horizon = image_size, horizon
        # Cache uint8 episodes per dataset/worker, avoiding decompression per shuffled
        # transition. Instance ownership releases memory when a diagnostic loader ends.
        self._episodes = {}
        self.indices = []
        for episode, path in enumerate(self.paths):
            with np.load(path) as data:
                a = data["actions"]
                if a.ndim != 2 or a.shape[1] != 2 or len(a) != len(data["frames"]) - 1:
                    raise ValueError(f"{path}: requires raw-step [T-1,2] actions; prepare the new dedicated dataset")
                if int(data.get("frame_skip", -1)) != 1 or str(data.get("action_mode", "")) != "relative":
                    raise ValueError(f"{path}: missing raw-step relative-action metadata")
                if not len(a):
                    continue
                if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.00001:
                    raise ValueError(f"{path}: relative actions must be finite and in [-1,1]")
                self.indices.extend((episode, t) for t in range(len(a)-horizon+1)
                                    if np.linalg.norm(a[t]) >= min_action_norm)
        if not self.indices:
            raise ValueError("no eligible transitions")

    def episode(self, index):
        if index not in self._episodes:
            with np.load(self.paths[index]) as data:
                self._episodes[index] = (data["frames"], data["actions"])
        return self._episodes[index]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        episode, t = self.indices[index]
        frames, actions = self.episode(episode)
        image = torch.from_numpy(frames[t:t+self.horizon+1].copy())
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = image.float() / 255.0
        image = F.interpolate(image, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return {"frames": image, "actions": torch.from_numpy(actions[t:t+self.horizon].copy()),
                "episode": episode, "timestep": t}
