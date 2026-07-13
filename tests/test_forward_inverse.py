import torch

from planner import CEMPlanner
from trainer import ACWMTrainer
from utils.factory import build_model


def forward_inverse_configuration():
    return {
        "data": {"history_length": 3},
        "planner": {"action_dim": 2},
        "model": {
            "version": "forward_inverse",
            "latent_dim": 24,
            "history_size": 3,
            "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 16, "feature_dim": 12},
            "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 24},
            "agent_transition": {"name": "mlp", "agent_dim": 16, "action_dim": 2, "hidden_dim": 32},
            "environment_transition": {"name": "mlp", "environment_dim": 24, "agent_dim": 16, "hidden_dim": 32},
            "predictor": {"type": "forward_inverse", "action_dim": 2},
            "forward_predictor": {"num_layers": 1, "num_heads": 3, "hidden_dim": 48, "dropout": 0.0},
            "inverse_head": {"enabled": True, "hidden_dim": 48, "num_layers": 2},
        },
    }


def batch():
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


def test_forward_inverse_shapes():
    model = build_model(forward_inverse_configuration())
    data = batch()
    frame_latents, current = model.encode(data["history_frames"], data["history_actions"], data["current_frame"])
    assert frame_latents.shape == (2, 3, 24)
    assert current.shape == (2, 24)
    prediction = model.step(frame_latents, current, data["current_action"])
    assert prediction.environment.shape == (2, 24)
    target = model.environment_encoder(data["next_frame"])
    action_pred = model.inverse_action(current, target)
    assert action_pred.shape == (2, 2)


def test_forward_inverse_backward_and_inverse_weight_zero():
    model = build_model(forward_inverse_configuration())
    trainer = ACWMTrainer(model, torch.optim.Adam(model.parameters()), {"forward": 1.0, "inverse": 0.0, "sigreg": 0.01})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert {"loss", "total_loss", "forward_loss", "inverse_loss", "sigreg_loss",
            "latent_mean", "latent_std", "action_mse"} == set(metrics)
    assert model.environment_encoder.network.network[0].weight.grad is not None
    assert model.predictor.forward_predictor.forward_mlp[0].weight.grad is not None


def test_forward_inverse_backward_with_inverse_enabled():
    model = build_model(forward_inverse_configuration())
    trainer = ACWMTrainer(model, torch.optim.Adam(model.parameters()), {"forward": 1.0, "inverse": 0.1, "sigreg": 0.01})
    trainer.train_step(batch())
    assert model.predictor.inverse_head.net[0].weight.grad is not None


def test_planning_does_not_call_inverse_head():
    model = build_model(forward_inverse_configuration()).eval()

    def fail_inverse(*_args, **_kwargs):
        raise AssertionError("planning must not call inverse head")

    model.predictor.inverse = fail_inverse
    planner = CEMPlanner(horizon=3, action_dim=2, population=8, elites=2, iterations=2)
    data = batch()
    actions = planner.plan(model, data["history_frames"], data["history_actions"],
                           data["current_frame"], data["next_frame"])
    assert actions.shape == (2, 3, 2)
