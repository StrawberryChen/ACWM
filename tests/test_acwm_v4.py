import torch

from trainer import ACWMTrainer
from utils.factory import build_model


def config():
    return {
        "data": {"history_length": 3},
        "planner": {"action_dim": 10},
        "model": {
            "version": "acwm_v4",
            "history_size": 3,
            "latent_dim": 192,
            "image_size": 224,
            "patch_size": 14,
            "vit_depth": 1,
            "vit_heads": 3,
            "mlp_ratio": 4.0,
            "transition_hidden_dim": 64,
            "predictor": {"type": "acwm_v4", "action_dim": 2, "action_block": 5},
            "temporal_encoder": {"num_layers": 1, "num_heads": 3, "dropout": 0.0},
            "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 192, "feature_dim": 192},
            "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 192},
            "agent_transition": {"name": "mlp", "agent_dim": 192, "action_dim": 2, "hidden_dim": 384},
            "environment_transition": {"name": "mlp", "environment_dim": 192, "agent_dim": 192, "hidden_dim": 384},
        },
    }


def markov_config():
    cfg = config()
    cfg["model"]["transition_mode"] = "markov"
    return cfg


def batch(batch_size=2, rollout_length=5):
    frames = torch.rand(batch_size, 3, 3, 32, 32)
    return {
        "history_frames": frames,
        "history_actions": torch.rand(batch_size, 2, 5, 2) * 512,
        "current_action": torch.rand(batch_size, 5, 2) * 512,
        "current_frame": frames[:, -1],
        "next_frame": torch.rand(batch_size, 3, 32, 32),
        "rollout_actions": torch.rand(batch_size, rollout_length, 5, 2) * 512,
        "rollout_frames": torch.rand(batch_size, rollout_length, 3, 32, 32),
    }


def test_acwm_v4_shapes_and_residual_prediction():
    model = build_model(config())
    model.set_action_stats(torch.tensor([256.0, 256.0]), torch.tensor([128.0, 128.0]))
    predictor = model.predictor
    history = predictor.encode_sequence(torch.rand(2, 3, 3, 32, 32))
    assert history.shape == (2, 3, 192)
    pred = predictor.predict_next(history, torch.rand(2, 5, 2) * 512)
    assert pred.shape == (2, 192)
    pred_seq = predictor.predict_next_sequence(history, torch.rand(2, 2, 5, 2) * 512, torch.rand(2, 5, 2) * 512)
    assert pred_seq.shape == (2, 3, 192)
    delta = pred - history[:, -1]
    assert delta.shape == (2, 192)


def test_acwm_v4_training_backward_with_terminal_loss():
    model = build_model(config())
    model.set_action_stats(torch.tensor([256.0, 256.0]), torch.tensor([128.0, 128.0]))
    trainer = ACWMTrainer(model, torch.optim.AdamW(model.parameters()),
                          {"prediction": 1.0, "terminal": 1.0, "sigreg": 0.01},
                          sigreg_config={"num_projections": 8, "knots": 5})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert "loss_pred" in metrics and "loss_terminal" in metrics and "loss_sigreg" in metrics
    assert model.predictor.projection.net[0].weight.grad is not None
    assert model.predictor.transition_predictor[0].weight.grad is not None
    assert model.predictor.action_encoder[0].weight.grad is not None


def test_acwm_v4_terminal_loss_can_be_disabled():
    model = build_model(config())
    model.set_action_stats(torch.tensor([256.0, 256.0]), torch.tensor([128.0, 128.0]))
    trainer = ACWMTrainer(model, torch.optim.AdamW(model.parameters()),
                          {"prediction": 1.0, "terminal": 0.0, "sigreg": 0.01},
                          sigreg_config={"num_projections": 8, "knots": 5})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert metrics["loss_terminal"] == 0.0


def test_acwm_v4_rollout_maintains_history_window():
    model = build_model(config())
    history = torch.randn(2, 3, 192)
    current = history[:, -1]
    actions = torch.randn(2, 5, 10).clamp(-1, 1)
    predictions = model.rollout(history, current, actions)
    assert len(predictions) == 5
    assert predictions[-1].agent.shape == (2, 3, 192)
    assert predictions[-1].environment.shape == (2, 192)


def test_acwm_v4_markov_predictor_ignores_temporal_encoder(monkeypatch):
    model = build_model(markov_config())

    def forbidden(*_, **__):
        raise AssertionError("markov transition must not call temporal_encoder")

    monkeypatch.setattr(model.predictor.temporal_encoder, "forward", forbidden)
    history = torch.randn(2, 3, 192)
    current = history[:, -1]
    pred = model.step(history, current, torch.randn(2, 10), action_is_normalized=True)
    assert pred.environment.shape == (2, 192)
    actions = torch.randn(2, 5, 10).clamp(-1, 1)
    rollout = model.rollout(history, current, actions)
    assert rollout[-1].environment.shape == (2, 192)
