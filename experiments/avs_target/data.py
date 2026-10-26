"""Episode-disjoint sliding windows; no simulator state in model minibatches."""
from collections import OrderedDict
from copy import deepcopy
import glob
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from dataset.action_contrastive import ActionContrastiveDataset


def merge(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    parent = config.pop("base", None)
    return merge(load_config(path.parent/parent), config) if parent else config


class TrajectoryWindows(ActionContrastiveDataset):
    def __init__(self, paths, horizon, cache_episodes=8, action_scale=None):
        super().__init__(paths, image_size=96, horizon=horizon)
        self._episodes = OrderedDict()
        self.cache_episodes = cache_episodes
        for path in self.paths:
            with np.load(path) as data:
                if action_scale is not None and "action_scale" in data and not np.isclose(float(data["action_scale"]), action_scale):
                    raise ValueError("action_scale metadata mismatch; do not silently reinterpret old actions")

    def episode(self, index):
        if index in self._episodes:
            self._episodes.move_to_end(index)
        else:
            self._episodes[index] = super().episode(index)
            if self.cache_episodes > 0:
                while len(self._episodes) > self.cache_episodes:
                    self._episodes.popitem(last=False)
        return self._episodes[index]


def source_signature(paths):
    return [{"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
             "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(config):
    paths, ids = {}, {}
    for split in ("train", "val", "test"):
        paths[split] = sorted({str(Path(p).resolve()) for pattern in config["data"][f"{split}_paths"] for p in glob.glob(pattern)})
        if not paths[split]:
            raise ValueError(f"no {split} episodes; run prepare_avs_target.py or configure raw-step NPZ paths")
        ids[split] = set()
        for path in paths[split]:
            with np.load(path) as episode:
                ids[split].add(str(episode["source_episode"]) if "source_episode" in episode else Path(path).name)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(paths[a]) & set(paths[b]) or ids[a] & ids[b]:
            raise ValueError(f"episode leakage between {a} and {b}")
    return paths


def loader(config, split, horizon=None, limit=None, shuffle=False):
    settings = config["data"]
    dataset = TrajectoryWindows(settings[f"{split}_paths"], horizon or config["model"]["horizon"],
                                settings["cache_episodes"], settings["action_scale"])
    chosen = dataset
    if limit and len(dataset) > limit:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]))[:limit]
        chosen = Subset(dataset, indices.tolist())
    return DataLoader(chosen, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], pin_memory=config["device"].startswith("cuda")), dataset
