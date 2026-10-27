"""Episode-disjoint sliding windows; no simulator state in model minibatches."""
from collections import OrderedDict
from collections import Counter
from copy import deepcopy
import glob
import hashlib
import warnings
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


def source_signature(paths, content_hash=True):
    records = []
    for path in paths:
        info = Path(path).stat()
        record = {"path": str(Path(path).resolve()), "size": info.st_size, "mtime_ns": info.st_mtime_ns}
        if content_hash:
            record["sha256"] = file_sha256(path)
        records.append(record)
    return records


def check_source_provenance(recorded, current, context, training=True):
    """Content identity for new runs; legacy metadata migration, not mtime identity.

    Evaluation does not train on these files and must not require them to retain
    their original filesystem metadata. Target/operator compatibility is separate.
    """
    if recorded == current:
        return
    hashed = bool(recorded and current) and all("sha256" in row for row in recorded+current)
    if hashed:
        identical = Counter(row["sha256"] for row in recorded) == Counter(row["sha256"] for row in current)
    else:
        # Old checkpoints do not contain enough information to prove byte identity.
        # Ignore unstable mtime ONLY; keep legacy path/size guards for training.
        key = lambda row: (str(Path(row["path"]).resolve()), int(row["size"]))
        identical = Counter(map(key, recorded)) == Counter(map(key, current))
    if training and not identical:
        raise ValueError(f"{context}: training data content/provenance differs; use the original data or a new output directory")
    if training and not hashed:
        warnings.warn(f"{context}: legacy provenance has no content hash; paths/sizes match, ignoring modification times. "
                      "Content identity cannot be proven for this old checkpoint/statistics file.", UserWarning)
    elif not training and not identical:
        warnings.warn(f"{context}: configured training-file metadata differs; evaluation keeps the checkpoint's "
                      "fixed target normalization and does not retrain/refit it.", UserWarning)


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
