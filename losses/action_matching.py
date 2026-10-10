"""Gaussian positive NLL + row-only consequence matching + existing SIGReg."""
import torch
from torch.nn import functional as F

from models.action_matching import matching_metrics


def compute_matching_loss(model, batch, sigreg, config):
    images = torch.cat((batch["current_frame"][:, None], batch["future_frames"]), 1)
    b, steps, c, h, w = images.shape
    # Same encoder for current and every REAL observed consequence [B,K+1,D].
    real = model.encode_frame(images.reshape(b*steps, c, h, w)).reshape(b, steps, -1)
    current, targets = real[:, 0], real[:, 1:]
    mean, sigma = model.branch_distribution(current, batch["actions"])
    # Every action has its own observed target; no synthetic opposite target.
    nll = -model.log_density(mean, sigma, targets).mean()
    scores = model.matching_scores(mean, sigma, targets)
    metrics = matching_metrics(scores, batch["positive_mask"], config["matching_temperature"])
    # [1, B*(K+1), D]: no temporal model; regularize only real encoder outputs.
    regularization = sigreg(real.reshape(1, -1, model.latent_dim))
    metrics.update({"nll_loss": nll, "sigreg_loss": regularization,
                    "loss": config["nll_weight"]*nll+config["matching_weight"]*metrics["matching_loss"]+config["sigreg_weight"]*regularization,
                    "positive_cosine": F.cosine_similarity(mean, targets, dim=-1).mean(),
                    # Diagnostic only: neither MSE nor sigma norms are extra losses.
                    "prediction_mse": (mean-targets).square().mean(),
                    "predicted_sigma_mean": sigma.mean(), "predicted_latent_norm": mean.norm(dim=-1).mean(),
                    "target_latent_norm": targets.norm(dim=-1).mean(),
                    "sigma_at_floor_fraction": (sigma <= model.sigma_min*1.01).float().mean(),
                    "sigma_at_ceiling_fraction": (sigma >= model.sigma_max).float().mean()})
    return metrics, real
