import torch

from losses import MomentSIGRegLoss
from models.v3_n1 import V3N1GaussianWorldModel
from trainer import ACWMTrainer
from utils.factory import build_model


def config():
    return {
        "planner": {"action_dim": 2},
        "model": {
            "version": "v3_n1",
            "latent_dim": 192,
            "image_size": 224,
            "patch_size": 14,
            "vit_depth": 1,
            "vit_heads": 3,
            "mlp_ratio": 4.0,
            "predictor": {"type": "v3_n1", "action_dim": 2},
            "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 192, "feature_dim": 192},
            "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 192},
            "agent_transition": {"name": "mlp", "agent_dim": 192, "action_dim": 2, "hidden_dim": 384},
            "environment_transition": {"name": "mlp", "environment_dim": 192, "agent_dim": 192, "hidden_dim": 384},
        },
    }


def batch(batch_size=2):
    frames = torch.rand(batch_size, 1, 3, 32, 32)
    return {
        "history_frames": frames,
        "history_actions": torch.empty(batch_size, 0, 2),
        "current_action": torch.rand(batch_size, 2) * 512,
        "current_frame": frames[:, -1],
        "next_frame": torch.rand(batch_size, 3, 32, 32),
    }


def test_v3_shapes_and_action_normalization():
    model = V3N1GaussianWorldModel(vit_depth=1)
    model.set_action_stats(torch.tensor([0.0, 0.0]), torch.tensor([512.0, 512.0]))
    latent = model.encode_gaussian(torch.rand(2, 3, 32, 32))
    assert latent.mu.shape == (2, 192)
    assert latent.logvar.shape == (2, 192)
    action_norm = model.normalize_action(torch.tensor([[0.0, 512.0]]))
    assert torch.allclose(action_norm, torch.tensor([[-1.0, 1.0]]))
    pred = model.predict_next(latent.mu, torch.rand(2, 2) * 512)
    assert pred.shape == (2, 192)


def test_v3_training_backward():
    model = build_model(config())
    model.set_action_stats(torch.tensor([0.0, 0.0]), torch.tensor([512.0, 512.0]))
    trainer = ACWMTrainer(model, torch.optim.AdamW(model.parameters()),
                          {"prediction": 1.0, "beta_kl": 1e-4, "lambda_sig": 0.05})
    metrics = trainer.train_step(batch())
    assert metrics["loss"] > 0
    assert "loss_pred" in metrics and "loss_kl" in metrics and "loss_sig_total" in metrics
    assert model.predictor.mean_head.weight.grad is not None
    assert model.predictor.dynamics_predictor[0].weight.grad is not None


def test_moment_sigreg_components():
    loss, metrics = MomentSIGRegLoss()(torch.randn(8, 192))
    assert loss.requires_grad is False
    assert {"loss_sig_mean", "loss_sig_var", "loss_sig_cov"} <= set(metrics)
