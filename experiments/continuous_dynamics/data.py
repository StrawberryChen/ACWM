"""Reuse existing raw-step preprocessing; never read state labels during training."""
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from experiments.avs_target.data import TrajectoryWindows, load_config, validate_splits


def loader(config, split, limit=None, shuffle=False):
    settings = config["data"]
    if settings["action_mode"] != "relative":
        raise ValueError("current prepared NPZ actions are relative, not absolute target positions")
    dataset = TrajectoryWindows(settings[f"{split}_paths"], horizon=1,
                                cache_episodes=settings["cache_episodes"], action_scale=settings["action_scale"])
    for path in dataset.paths:
        with np.load(path) as episode:
            if "action_scale" not in episode:
                raise ValueError(f"{path}: explicit action_scale metadata required; do not guess action units")
            if episode["frames"].dtype != np.uint8:
                raise ValueError(f"{path}: prepared raw RGB must be uint8; avoid double division by 255")
    dataset.image_size = config["model"]["image_size"]
    selected = dataset
    if limit and len(dataset) > limit:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]))[:limit]
        selected = Subset(dataset, indices.tolist())
    return DataLoader(selected, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], pin_memory=config["device"].startswith("cuda")), dataset


def resolve(config_path, data_config=None, output_dir=None):
    from pathlib import Path
    config = load_config(config_path)
    if data_config:
        prepared = load_config(data_config)
        # Borrow only paths/action units/runtime, NOT old loss/model/validation settings.
        config["data"].update(prepared["data"])
        config["device"] = prepared["device"]
        config["output_dir"] = str(Path(prepared["output_dir"]).parent/"continuous_dynamics"/config["loss"]["mode"])
    if output_dir:
        config["output_dir"] = output_dir
    return config
