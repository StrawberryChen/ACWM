import torch

from planner import CEMPlanner
from trainer import ACWMTrainer
from utils.factory import build_model
from tests.test_model import configuration, motion_configuration


def batch():
    frames = torch.rand(2, 4, 3, 32, 32)
    return {
        "history_frames": frames,
        "history_actions": torch.rand(2, 3, 2),
        "current_action": torch.rand(2, 2),
        "current_frame": frames[:, -1],
        "next_frame": torch.rand(2, 3, 32, 32),
        "next_history_frames": torch.rand(2, 4, 3, 32, 32),
        "next_history_actions": torch.rand(2, 3, 2),
    }


def motion_batch():
    frames = torch.rand(2, 3, 3, 32, 32)
    return {
        "history_frames": frames,
        "history_actions": torch.rand(2, 2, 2),
        "current_action": torch.rand(2, 2),
        "current_frame": frames[:, -1],
        "next_frame": torch.rand(2, 3, 32, 32),
        "next_history_frames": torch.rand(2, 3, 3, 32, 32),
        "next_history_actions": torch.rand(2, 2, 2),
    }


def test_one_step_training():
    model = build_model(configuration())
    trainer = ACWMTrainer(model, torch.optim.Adam(model.parameters()), {"agent": 1, "environment": 1})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert {"loss", "agent_loss", "environment_loss", "agent_sigreg_loss",
            "environment_sigreg_loss", "agent_latent_std", "environment_latent_std"} == set(metrics)


def test_cem_planner_shape_and_bounds():
    model = build_model(configuration()).eval()
    planner = CEMPlanner(horizon=3, action_dim=2, population=8, elites=2, iterations=2)
    data = batch()
    actions = planner.plan(model, data["history_frames"], data["history_actions"],
                           data["current_frame"], data["next_frame"])
    assert actions.shape == (2, 3, 2)
    assert torch.all(actions <= 1) and torch.all(actions >= -1)


def test_motion_token_training_and_planner():
    model = build_model(motion_configuration())
    trainer = ACWMTrainer(model, torch.optim.Adam(model.parameters()), {"prediction": 1, "flow": 1, "sigreg": 0.01})
    metrics = trainer.train_step(motion_batch())
    assert metrics["loss"] > 0
    assert {"loss", "prediction_loss", "flow_loss", "sigreg_loss", "latent_std",
            "environment_loss", "environment_sigreg_loss"} == set(metrics)
    planner = CEMPlanner(horizon=3, action_dim=2, population=8, elites=2, iterations=2)
    data = motion_batch()
    actions = planner.plan(model.eval(), data["history_frames"], data["history_actions"],
                           data["current_frame"], data["next_frame"])
    assert actions.shape == (2, 3, 2)
