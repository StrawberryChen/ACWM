import torch

from losses import MomentSIGRegLoss
from models.v3_n1 import V3N1GaussianWorldModel
from trainer import ACWMTrainer
from utils.factory import build_model


def config():
    return {
        "data": {"history_length": 3},
        "planner": {"action_dim": 10},
        "model": {
            "version": "v3_n1",
            "history_size": 3,
            "latent_dim": 192,
            "image_size": 224,
            "patch_size": 14,
            "vit_depth": 1,
            "vit_heads": 3,
            "mlp_ratio": 4.0,
            "predictor": {"type": "v3_n1", "action_dim": 2, "action_block": 5},
            "action_consistency_head": {"hidden_dim": 64},
            "temporal_encoder": {"num_layers": 1, "num_heads": 3, "dropout": 0.0},
            "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 192, "feature_dim": 192},
            "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 192},
            "agent_transition": {"name": "mlp", "agent_dim": 192, "action_dim": 2, "hidden_dim": 384},
            "environment_transition": {"name": "mlp", "environment_dim": 192, "agent_dim": 192, "hidden_dim": 384},
        },
    }


def batch(batch_size=2):
    frames = torch.rand(batch_size, 3, 3, 32, 32)
    return {
        "history_frames": frames,
        "history_actions": torch.rand(batch_size, 2, 5, 2) * 512,
        "current_action": torch.rand(batch_size, 5, 2) * 512,
        "current_frame": frames[:, -1],
        "next_frame": torch.rand(batch_size, 3, 32, 32),
    }


def test_v3_shapes_and_action_normalization():
    model = V3N1GaussianWorldModel(vit_depth=1, action_block=5)
    model.set_action_stats(torch.zeros(5, 2), torch.full((5, 2), 512.0))
    latent = model.encode_gaussian(torch.rand(2, 3, 32, 32))
    assert latent.mu.shape == (2, 192)
    assert latent.logvar.shape == (2, 192)
    history = model.encode_gaussian_sequence(torch.rand(2, 3, 3, 32, 32))
    assert history.mu.shape == (2, 3, 192)
    assert history.logvar.shape == (2, 3, 192)
    raw_block = torch.stack((torch.zeros(5, 2), torch.full((5, 2), 512.0)))
    action_norm = model.normalize_action(raw_block)
    assert action_norm.shape == (2, 10)
    assert torch.allclose(action_norm[0], torch.full((10,), -1.0))
    assert torch.allclose(action_norm[1], torch.full((10,), 1.0))
    assert model.denormalize_action(action_norm).shape == (2, 5, 2)
    pred = model.predict_next(history.mu, torch.rand(2, 5, 2) * 512)
    assert pred.shape == (2, 192)
    delta = pred - history.mu[:, -1]
    action_hat = model.predict_action_from_delta(delta)
    assert action_hat.shape == (2, 10)


def test_v3_training_backward():
    model = build_model(config())
    model.set_action_stats(torch.zeros(5, 2), torch.full((5, 2), 512.0))
    trainer = ACWMTrainer(model, torch.optim.AdamW(model.parameters()),
                          {"prediction": 1.0, "beta_kl": 1e-4, "lambda_sig": 1.0,
                           "lambda_action_consistency": 1.0})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert "loss_pred" in metrics and "loss_kl" in metrics and "loss_sig_total" in metrics
    assert "loss_action_consistency" in metrics and "delta_pred_norm" in metrics
    assert model.predictor.mean_head.weight.grad is not None
    assert model.predictor.dynamics_predictor[0].weight.grad is not None
    assert model.predictor.action_consistency_head.net[0].weight.grad is not None


def test_v3_action_consistency_can_be_disabled():
    model = build_model(config())
    model.set_action_stats(torch.zeros(5, 2), torch.full((5, 2), 512.0))
    trainer = ACWMTrainer(model, torch.optim.AdamW(model.parameters()),
                          {"prediction": 1.0, "beta_kl": 1e-4, "lambda_sig": 1.0,
                           "lambda_action_consistency": 0.0})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert "loss_action_consistency" in metrics


def test_v3_predict_next_does_not_call_action_consistency_head(monkeypatch):
    model = V3N1GaussianWorldModel(vit_depth=1, action_block=5)

    def forbidden(_):
        raise AssertionError("ActionConsistencyHead must not be called by predict_next/rollout/planning")

    monkeypatch.setattr(model.action_consistency_head, "forward", forbidden)
    pred = model.predict_next(torch.randn(2, 3, 192), torch.randn(2, 10), action_is_normalized=True)
    assert pred.shape == (2, 192)


def test_v3_world_model_rollout_maintains_history_window(monkeypatch):
    model = build_model(config())

    def forbidden(_):
        raise AssertionError("ActionConsistencyHead must not be called by rollout/planning")

    monkeypatch.setattr(model.predictor.action_consistency_head, "forward", forbidden)
    history = torch.randn(2, 3, 192)
    current = history[:, -1]
    actions = torch.randn(2, 5, 10).clamp(-1, 1)
    predictions = model.rollout(history, current, actions)
    assert len(predictions) == 5
    assert predictions[-1].agent.shape == (2, 3, 192)
    assert predictions[-1].environment.shape == (2, 192)


def test_v3_action_target_shape_matches_head():
    model = V3N1GaussianWorldModel(vit_depth=1, action_block=5)
    model.set_action_stats(torch.zeros(5, 2), torch.full((5, 2), 512.0))
    delta = torch.randn(4, 192)
    action_hat = model.predict_action_from_delta(delta)
    target_action = model.normalize_action(torch.rand(4, 5, 2) * 512)
    assert action_hat.shape == target_action.shape
    assert target_action.ndim == 2


def test_moment_sigreg_components():
    loss, metrics = MomentSIGRegLoss()(torch.randn(8, 192))
    assert loss.requires_grad is False
    assert {"loss_sig_mean", "loss_sig_var", "loss_sig_cov"} <= set(metrics)
