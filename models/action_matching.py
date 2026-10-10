"""V2: shared deterministic observation encoder, conditional Gaussian dynamics."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.action_contrastive import ActionContrastiveWorldModel


class ActionMatchingWorldModel(ActionContrastiveWorldModel):
    predictor_type = "action_matching_v2"

    def __init__(self, latent_dim=128, image_size=96, architecture="residual_mlp",
                 sigma_min=0.05, sigma_max=5.0):
        super().__init__(latent_dim, image_size, architecture)
        if not 0 < sigma_min < sigma_max:
            raise ValueError("require 0 < sigma_min < sigma_max")
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.uncertainty_head = nn.Sequential(nn.Linear(latent_dim+32, 256), nn.SiLU(),
                                              nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, latent_dim))

    def distribution(self, observation, action):
        # O [B,D], A [B,2] -> conditional mean/std [B,D]. Encoder stays deterministic.
        assert observation.shape == (len(action), self.latent_dim) and action.shape == (len(observation), 2)
        mean = super().transition(observation, action)
        inputs = torch.cat((observation, self.action_encoder(action)), -1)
        sigma = (F.softplus(self.uncertainty_head(inputs))+self.sigma_min).clamp_max(self.sigma_max)
        return mean, sigma

    def branch_distribution(self, observation, actions):
        # Shared start O [B,D]; actual executed branch actions [B,K,2].
        assert actions.ndim == 3 and actions.shape[0] == len(observation) and actions.shape[-1] == 2
        b, k, _ = actions.shape
        mean, sigma = self.distribution(observation[:, None].expand(-1, k, -1).reshape(b*k, -1), actions.reshape(b*k, 2))
        return mean.reshape(b, k, -1), sigma.reshape(b, k, -1)

    @staticmethod
    def log_density(mean, sigma, target):
        # Broadcastable [...,D] -> [...]. Average log density per latent dimension.
        # Full Gaussian log p = D * this value; matching temperature uses this scale.
        return -0.5*((target-mean).square()/sigma.square()+2*sigma.log()+math.log(2*math.pi)).mean(-1)

    def matching_scores(self, mean, sigma, targets):
        # [B,K,D] -> [B,K,K]: row i predicts action i, column j is observed consequence j.
        assert mean.shape == sigma.shape == targets.shape
        return self.log_density(mean[:, :, None], sigma[:, :, None], targets[:, None])

    # Existing CEM/MPC sees deterministic conditional means, same action bounds,
    # same cosine goal cost and rollout interface as v1. No Gaussian sampling there.


def matching_metrics(scores, positives, temperature=0.1):
    """Row-only multi-positive CE. Image-equivalent futures are acceptable matches."""
    assert scores.ndim == 3 and scores.shape == positives.shape and scores.shape[1] == scores.shape[2]
    assert positives.dtype == torch.bool and positives.any(-1).all()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = scores/temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    loss = (torch.logsumexp(logits, -1)-torch.logsumexp(positive_logits, -1)).mean()
    positive_score = scores.masked_fill(~positives, -torch.inf).max(-1).values
    negative_score = scores.masked_fill(positives, -torch.inf).max(-1).values
    distinguishable = (~positives).any(-1)
    gap = positive_score-negative_score
    # Ties never count as discrimination. Uninformative all-positive rows do not
    # inflate accuracy, but retain their genuine NLL supervision.
    if distinguishable.any():
        accuracy = (gap[distinguishable] > 0).float().mean()
        rank = (1+((scores >= positive_score[..., None]) & ~positives).sum(-1))[distinguishable].float().mean()
        score_gap = gap[distinguishable].mean()
    else:
        accuracy = rank = score_gap = scores.new_tensor(float("nan"))
    return {"matching_loss": loss, "matching_acc": accuracy, "mean_rank": rank,
            "score_gap": score_gap, "informative_rows": distinguishable.float().sum(),
            "ambiguous_row_fraction": (positives.sum(-1) > 1).float().mean()}
