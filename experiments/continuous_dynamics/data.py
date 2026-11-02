"""Reuse existing raw-step preprocessing; never read state labels during training."""
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from torch.nn import functional as F
from experiments.avs_target.data import TrajectoryWindows, load_config, validate_splits


class RawTransitionDataset(TrajectoryWindows):
    """One shared image conversion for training, probes and episode plots.

    Infer floating-point range PER EPISODE, never per minibatch/window: a dark
    byte-scale window must not suddenly be interpreted as normalized RGB.
    """
    def __init__(self, paths, image_size, cache_episodes, action_scale, frame_value_range="auto", horizon=1):
        if frame_value_range not in {"auto", "unit", "byte"}:
            raise ValueError("frame_value_range must be auto, unit, or byte")
        super().__init__(paths, horizon=horizon, cache_episodes=cache_episodes, action_scale=action_scale)
        self.image_size, self.frame_scales = image_size, []
        for path in self.paths:
            with np.load(path) as episode:
                if "action_scale" not in episode:
                    raise ValueError(f"{path}: explicit action_scale metadata required; do not guess action units")
                frames = episode["frames"]
                if frames.ndim != 4 or not (frames.shape[-1] == 3 or frames.shape[1] == 3):
                    raise ValueError(f"{path}: expected RGB [T,H,W,3] or [T,3,H,W]")
                is_byte = frames.dtype == np.uint8
                if not is_byte and not np.issubdtype(frames.dtype, np.floating):
                    raise ValueError(f"{path}: RGB must be uint8 or floating point, got {frames.dtype}")
                if not np.isfinite(frames).all():
                    raise ValueError(f"{path}: RGB contains NaN or Inf")
                low, high = float(frames.min()), float(frames.max())
                if low < 0 or high > 255:
                    raise ValueError(f"{path}: RGB range [{low}, {high}] is outside [0,255]")
                if frame_value_range == "unit":
                    if is_byte or high > 1:
                        raise ValueError(f"{path}: unit range requires floating RGB in [0,1]")
                    scale = 1.
                elif frame_value_range == "byte":
                    scale = 255.
                else:
                    scale = 255. if is_byte or high > 1 else 1.
                self.frame_scales.append(scale)

    def prepare_frames(self, frames, episode):
        # Raw [T,H,W,3] or [T,3,H,W] -> float32 [T,3,H,W] in [0,1].
        image = torch.from_numpy(frames.astype(np.float32, copy=True))
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = image.float()/self.frame_scales[episode]
        return F.interpolate(image, (self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def __getitem__(self, index):
        episode, t = self.indices[index]
        frames, actions = self.episode(episode)
        return {"frames": self.prepare_frames(frames[t:t+self.horizon+1], episode),
                "actions": torch.from_numpy(actions[t:t+self.horizon].copy()).float(),
                "episode": episode, "timestep": t}


def loader(config, split, limit=None, shuffle=False, horizon=None):
    settings = config["data"]
    if settings["action_mode"] != "relative":
        raise ValueError("current prepared NPZ actions are relative, not absolute target positions")
    dataset = RawTransitionDataset(settings[f"{split}_paths"], config["model"]["image_size"],
                                   settings["cache_episodes"], settings["action_scale"],
                                   settings.get("frame_value_range", "auto"),
                                   config["model"].get("horizon", 1) if horizon is None else horizon)
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
        if config.get("experiment") == "continuous_trajectory":
            config["output_dir"] = str(Path(prepared["output_dir"]).parent/"continuous_trajectory"/config["model"]["direction_mode"])
        else:
            config["output_dir"] = str(Path(prepared["output_dir"]).parent/"continuous_dynamics"/config["loss"]["mode"])
    if output_dir:
        config["output_dir"] = output_dir
    return config
