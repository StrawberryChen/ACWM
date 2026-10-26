"""LeWorld Push-T planning settings adapted to existing single-raw-step models.

Official 5 blocks * 5 actions == 25 independent raw actions. We flatten those
variables and recursively apply the unchanged single-step Predictor 25 times.
This aligns the control budget, not the learned five-action-block architecture.
"""
import glob

import numpy as np


def align_raw_step_config(config):
    if config["model"]["version"] not in {"action_contrastive", "action_matching_v2"}:
        raise ValueError("this profile is only for the two single-raw-step experiments")
    config["planning_protocol"] = "leworld_raw_step"
    config["planner"].update(horizon=25, action_dim=2, population=300, elites=30,
                             iterations=30, var_scale=1.0, clamp_actions=False,
                             cost_type="latent_mse_sum", seed=42,
                             std_correction=0, min_std=0.0)
    config["environment"].update(max_steps=50, replan_interval=25, action_block=1,
                                 goal_offset_steps=25, seed=42)
    config["validation"]["planning_episodes"] = 50
    return config


def training_action_statistics(paths):
    """StandardScaler-compatible population statistics; never fit val/test actions."""
    paths = sorted({p for pattern in paths for p in glob.glob(str(pattern))})
    if not paths:
        raise ValueError("planning standardization requires the model's train trajectories")
    n, total, squared = 0, np.zeros(2, np.float64), np.zeros(2, np.float64)
    for path in paths:
        with np.load(path) as data:
            actions = data["actions"].astype(np.float64)
            if actions.ndim != 2 or actions.shape[-1] != 2 or not np.isfinite(actions).all():
                raise ValueError("expected finite raw-step training actions [T,2]")
            n += len(actions)
            total += actions.sum(0)
            squared += np.square(actions).sum(0)
    if n == 0:
        raise ValueError("empty training action statistics")
    mean = total/n
    std = np.sqrt(np.maximum(squared/n-mean**2, 0))
    std[std < 1e-12] = 1.0
    return mean.tolist(), std.tolist()
