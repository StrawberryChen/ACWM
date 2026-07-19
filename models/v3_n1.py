from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GaussianLatent:
    mu: torch.Tensor
    logvar: torch.Tensor


class V3N1Projection(nn.Module):
    """Linear(192,192) + BatchNorm1d(192), no activation."""

    def __init__(self, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        assert cls_token.ndim == 2, f"CLS token must be [B,192], got {tuple(cls_token.shape)}"
        return self.net(cls_token)


class ActionConsistencyHead(nn.Module):
    """Predict the normalized action label from predicted latent displacement.

    Input shape:  delta_pred [B, 192]
    Output shape: action_hat [B, action_dim]
    """

    def __init__(self, latent_dim: int = 192, action_dim: int = 2, hidden_dim: int = 384):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, delta_pred: torch.Tensor) -> torch.Tensor:
        assert delta_pred.ndim == 2, f"delta_pred must be [B,{self.latent_dim}], got {tuple(delta_pred.shape)}"
        assert delta_pred.shape[-1] == self.latent_dim, (
            f"delta_pred latent dim must be {self.latent_dim}, got {delta_pred.shape[-1]}"
        )
        # action_hat: [B, action_dim]
        return self.net(delta_pred)


class V3N1GaussianWorldModel(nn.Module):
    """ACWM v3-N1: one-frame action-conditioned Gaussian-mean world model."""

    def __init__(self, image_channels: int = 3, latent_dim: int = 192, action_dim: int = 2,
                 image_size: int = 224, patch_size: int = 14, vit_depth: int = 12,
                 vit_heads: int = 3, mlp_ratio: float = 4.0,
                 logvar_min: float = -10.0, logvar_max: float = 10.0,
                 action_consistency_hidden_dim: int = 384):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as error:
            raise ImportError("ACWM v3-N1 requires timm>=1.0") from error
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.encoder = VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=image_channels,
            num_classes=0,
            embed_dim=latent_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=mlp_ratio,
        )
        self.projection = V3N1Projection(latent_dim)
        self.mean_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.action_consistency_head = ActionConsistencyHead(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=action_consistency_hidden_dim,
        )
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("action_min", torch.full((action_dim,), -1.0), persistent=True)
        self.register_buffer("action_max", torch.full((action_dim,), 1.0), persistent=True)

    def set_action_stats(self, action_min: torch.Tensor, action_max: torch.Tensor) -> None:
        assert action_min.shape == self.action_min.shape
        assert action_max.shape == self.action_max.shape
        self.action_min.copy_(action_min.to(self.action_min))
        self.action_max.copy_(action_max.to(self.action_max))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B,2] raw Push-T action; output [B,2] in [-1,1].
        denom = (self.action_max - self.action_min).clamp_min(1e-6)
        return 2.0 * (action - self.action_min) / denom - 1.0

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        # action_norm: [B,2] in [-1,1]; output raw Push-T action coordinates.
        return (action_norm + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min

    def encode_gaussian(self, images: torch.Tensor) -> GaussianLatent:
        assert images.ndim == 4, f"images must be [B,3,H,W], got {tuple(images.shape)}"
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        if images.shape[1] == 3:
            images = (images - self.image_mean) / self.image_std
        # cls: [B,192]
        cls = self.encoder(images)
        # projected: [B,192]
        projected = self.projection(cls)
        # mu/logvar: [B,192]
        mu = self.mean_head(projected)
        logvar = self.logvar_head(projected).clamp(self.logvar_min, self.logvar_max)
        return GaussianLatent(mu, logvar)

    def encode_mean(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_gaussian(images).mu

    def predict_next(self, mu_current: torch.Tensor, action: torch.Tensor,
                     action_is_normalized: bool = False) -> torch.Tensor:
        assert mu_current.ndim == 2, f"mu_current must be [B,192], got {tuple(mu_current.shape)}"
        assert action.ndim == 2 and action.shape[-1] == self.action_dim, (
            f"action must be [B,{self.action_dim}], got {tuple(action.shape)}"
        )
        # a_norm: [B,2]
        a_norm = action if action_is_normalized else self.normalize_action(action)
        # e_action: [B,192]
        e_action = self.action_encoder(a_norm)
        # predictor_input: [B,384]
        predictor_input = torch.cat((mu_current, e_action), dim=-1)
        # mu_pred_next: [B,192]
        return self.dynamics_predictor(predictor_input)

    def predict_action_from_delta(self, delta_pred: torch.Tensor) -> torch.Tensor:
        assert delta_pred.ndim == 2, f"delta_pred must be [B,{self.latent_dim}], got {tuple(delta_pred.shape)}"
        assert delta_pred.shape[-1] == self.latent_dim, (
            f"delta_pred latent dim must be {self.latent_dim}, got {delta_pred.shape[-1]}"
        )
        # action_hat: [B, action_dim], target is normalized real action in [-1, 1].
        return self.action_consistency_head(delta_pred)

    @staticmethod
    def kl_loss(latent: GaussianLatent) -> torch.Tensor:
        return -0.5 * (1.0 + latent.logvar - latent.mu.square() - latent.logvar.exp()).mean()
