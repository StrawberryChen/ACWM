from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ObservationProjection(nn.Module):
    """LeWorld-style 1-layer MLP + BatchNorm projector.

    Input shape:  cls_token [B, 192]
    Output shape: latent    [B, 192]
    """

    def __init__(self, dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim))

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        assert cls_token.ndim == 2, f"CLS token must be [B,192], got {tuple(cls_token.shape)}"
        return self.net(cls_token)


class TimeEmbedding(nn.Module):
    """Sinusoidal scalar time/ratio embedding followed by an MLP."""

    def __init__(self, dim: int = 192, frequencies: int = 16):
        super().__init__()
        self.frequencies = frequencies
        self.net = nn.Sequential(
            nn.Linear(frequencies * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # value: [B,1] or scalar-like tensor; output: [B,dim].
        if value.ndim == 0:
            value = value[None, None]
        elif value.ndim == 1:
            value = value[:, None]
        assert value.ndim == 2 and value.shape[-1] == 1, f"value must be [B,1], got {tuple(value.shape)}"
        freqs = torch.arange(1, self.frequencies + 1, device=value.device, dtype=value.dtype)[None]
        angles = value * freqs * math.pi
        return self.net(torch.cat((angles.sin(), angles.cos()), dim=-1))


class TemporalEdgeEncoder(nn.Module):
    """Learned directed edge representation between two latent states.

    Input shape:  z_i [B,192], z_j [B,192], delta_t [B,1]
    Output shape: e_ij [B,192]
    """

    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimeEmbedding(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, delta_t: torch.Tensor | float) -> torch.Tensor:
        assert z_i.ndim == 2 and z_i.shape[-1] == self.latent_dim, (
            f"z_i must be [B,{self.latent_dim}], got {tuple(z_i.shape)}"
        )
        assert z_j.shape == z_i.shape, f"z_j shape {tuple(z_j.shape)} must match z_i {tuple(z_i.shape)}"
        if not torch.is_tensor(delta_t):
            delta_t = torch.full((z_i.shape[0], 1), float(delta_t), device=z_i.device, dtype=z_i.dtype)
        else:
            delta_t = delta_t.to(device=z_i.device, dtype=z_i.dtype)
            if delta_t.ndim == 0:
                delta_t = delta_t.expand(z_i.shape[0])[:, None]
            elif delta_t.ndim == 1:
                delta_t = delta_t[:, None]
        # diff/prod/time: [B,192]
        diff = z_j - z_i
        prod = z_i * z_j
        time = self.time_embed(delta_t)
        # residual edge keeps the obvious displacement but lets the model learn a nonlinear relation.
        return diff + self.net(torch.cat((z_i, z_j, diff, prod, time), dim=-1))


class ObservationGraphBridgeModel(nn.Module):
    """Observation-only temporal graph representation learner.

    It learns frame latents z_t and directed edge representations e_ij without
    actions, rewards, or simulator states.
    """

    def __init__(self, image_channels: int = 3, latent_dim: int = 192,
                 image_size: int = 224, patch_size: int = 14, vit_depth: int = 12,
                 vit_heads: int = 3, mlp_ratio: float = 4.0, hidden_dim: int = 512):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as error:
            raise ImportError("ObservationGraphBridgeModel requires timm>=1.0") from error
        self.latent_dim = latent_dim
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
        self.projection = ObservationProjection(latent_dim)
        self.edge_encoder = TemporalEdgeEncoder(latent_dim, hidden_dim)
        self.alpha_embed = TimeEmbedding(latent_dim)
        self.bridge_net = nn.Sequential(
            nn.Linear(latent_dim * 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,3,H,W] in [0,1]; output z: [B,192].
        assert images.ndim == 4, f"images must be [B,3,H,W], got {tuple(images.shape)}"
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        cls = self.encoder(images)
        return self.projection(cls)

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W]; output latents: [B,T,192].
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {tuple(frames.shape)}"
        batch, steps = frames.shape[:2]
        flat = self.encode(frames.flatten(0, 1))
        return flat.view(batch, steps, self.latent_dim)

    def edge(self, z_i: torch.Tensor, z_j: torch.Tensor, delta_t: torch.Tensor | float) -> torch.Tensor:
        return self.edge_encoder(z_i, z_j, delta_t)

    def bridge(self, z_start: torch.Tensor, z_end: torch.Tensor, alpha: torch.Tensor | float,
               edge: torch.Tensor | None = None) -> torch.Tensor:
        """Predict the intermediate latent between two endpoints.

        z_start/z_end: [B,192]
        alpha:         [B,1] interpolation ratio in [0,1]
        output:        [B,192]
        """
        assert z_start.ndim == 2 and z_start.shape[-1] == self.latent_dim
        assert z_end.shape == z_start.shape
        if not torch.is_tensor(alpha):
            alpha = torch.full((z_start.shape[0], 1), float(alpha), device=z_start.device, dtype=z_start.dtype)
        else:
            alpha = alpha.to(device=z_start.device, dtype=z_start.dtype)
            if alpha.ndim == 0:
                alpha = alpha.expand(z_start.shape[0])[:, None]
            elif alpha.ndim == 1:
                alpha = alpha[:, None]
        if edge is None:
            edge = self.edge(z_start, z_end, torch.ones_like(alpha))
        linear = (1.0 - alpha) * z_start + alpha * z_end
        diff = z_end - z_start
        prod = z_start * z_end
        alpha_emb = self.alpha_embed(alpha)
        residual = self.bridge_net(torch.cat((z_start, z_end, diff, prod, edge, alpha_emb), dim=-1))
        return linear + residual
