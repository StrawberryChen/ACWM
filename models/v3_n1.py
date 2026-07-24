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
    Output shape: action_hat [B, action_block * action_dim]
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
        # action_hat: [B, action_block * action_dim]
        return self.net(delta_pred)


class V3N1TemporalEncoder(nn.Module):
    """Causal temporal encoder over LeWorld-style state-action tokens.

    Input shape:  frame_latents  [B, T, 192]
                  action_latents [B, T, 192]
    Output shape: h_t           [B, 192]
    """

    def __init__(self, latent_dim: int = 192, history_size: int = 3,
                 num_layers: int = 2, num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_size, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def encode_tokens(self, frame_latents: torch.Tensor, action_latents: torch.Tensor) -> torch.Tensor:
        assert frame_latents.ndim == 3, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert action_latents.ndim == 3, (
            f"action_latents must be [B,T,{self.latent_dim}], got {tuple(action_latents.shape)}"
        )
        assert action_latents.shape == frame_latents.shape, (
            f"action_latents shape {tuple(action_latents.shape)} must match frame_latents {tuple(frame_latents.shape)}"
        )
        assert frame_latents.shape[1] <= self.history_size, (
            f"history window {frame_latents.shape[1]} exceeds configured history_size={self.history_size}"
        )
        assert frame_latents.shape[-1] == self.latent_dim, (
            f"frame latent dim must be {self.latent_dim}, got {frame_latents.shape[-1]}"
        )
        steps = frame_latents.shape[1]
        # tokens: [B,T,192]
        tokens = frame_latents + action_latents + self.temporal_pos[:, -steps:]
        # causal_mask: [T,T], token i can attend only to tokens <= i.
        causal_mask = torch.triu(torch.ones(steps, steps, device=frame_latents.device, dtype=torch.bool), diagonal=1)
        # encoded: [B,T,192]
        return self.transformer(tokens, mask=causal_mask)

    def forward(self, frame_latents: torch.Tensor, action_latents: torch.Tensor) -> torch.Tensor:
        # encoded: [B,T,192]
        encoded = self.encode_tokens(frame_latents, action_latents)
        # h_t: [B,192]
        return encoded[:, -1]


class V3N1GaussianWorldModel(nn.Module):
    """ACWM v3-N1: history-window action-conditioned Gaussian-mean world model."""

    def __init__(self, image_channels: int = 3, latent_dim: int = 192, action_dim: int = 2,
                 image_size: int = 224, patch_size: int = 14, vit_depth: int = 12,
                 vit_heads: int = 3, mlp_ratio: float = 4.0,
                 logvar_min: float = -10.0, logvar_max: float = 10.0,
                 action_consistency_hidden_dim: int = 384, history_size: int = 3,
                 temporal_layers: int = 2, temporal_heads: int = 3, temporal_dropout: float = 0.0,
                 action_block: int = 1):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as error:
            raise ImportError("ACWM v3-N1 requires timm>=1.0") from error
        self.latent_dim = latent_dim
        self.raw_action_dim = action_dim
        self.action_block = action_block
        self.action_dim = action_dim * action_block
        self.history_size = history_size
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
            nn.Linear(self.action_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.temporal_encoder = V3N1TemporalEncoder(
            latent_dim=latent_dim,
            history_size=history_size,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
        )
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(latent_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        self.action_consistency_head = ActionConsistencyHead(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            hidden_dim=action_consistency_hidden_dim,
        )
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)

    def set_action_stats(self, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        """Set LeWorld-style z-score action statistics.

        action_mean/action_std: [2] raw Push-T action statistics shared by all
        action_block positions. The flattened predictor action is still
        [B, action_block * 2].
        """
        action_mean = action_mean.flatten()
        action_std = action_std.flatten().clamp_min(1e-6)
        assert action_mean.shape == self.action_mean.shape, (
            f"action_mean shape {tuple(action_mean.shape)} must be [{self.raw_action_dim}]"
        )
        assert action_std.shape == self.action_std.shape, (
            f"action_std shape {tuple(action_std.shape)} must be [{self.raw_action_dim}]"
        )
        self.action_mean.copy_(action_mean.to(self.action_mean))
        self.action_std.copy_(action_std.to(self.action_std))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B,action_block,2] raw Push-T action block; output [B,action_block*2] z-scored.
        action = self._as_action_block(action)
        action_norm = (action - self.action_mean) / self.action_std
        return action_norm.flatten(1)

    def normalize_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: [B,T,action_block,2] raw action blocks; output [B,T,action_block*2].
        assert actions.ndim == 4, (
            f"actions must be [B,T,{self.action_block},{self.raw_action_dim}], got {tuple(actions.shape)}"
        )
        batch, steps = actions.shape[:2]
        flat = self.normalize_action(actions.flatten(0, 1))
        return flat.view(batch, steps, self.action_dim)

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        # action_norm: [B,action_block*2] z-scored; output raw Push-T action block [B,action_block,2].
        action_norm = self._as_action_block(action_norm)
        return action_norm * self.action_std + self.action_mean

    def action_sequence(self, history_actions: torch.Tensor, current_action: torch.Tensor,
                        action_is_normalized: bool = False) -> torch.Tensor:
        """Build LeWorld-style action history ending in the current action.

        history_actions raw shape: [B,T-1,action_block,2]
        current_action raw shape:  [B,action_block,2]
        output normalized shape:   [B,T,action_block*2]
        """
        if action_is_normalized:
            if current_action.ndim == 2:
                current = current_action
            else:
                current = current_action.flatten(1)
            if history_actions is None or history_actions.numel() == 0:
                previous = current[:, None].expand(-1, self.history_size - 1, -1)
            else:
                previous = (self.normalize_action_sequence(history_actions)
                            if history_actions.ndim == 4 else history_actions)
            assert previous.ndim == 3 and previous.shape[-1] == self.action_dim, (
                f"normalized history_actions must be [B,T-1,{self.action_dim}], got {tuple(previous.shape)}"
            )
            seq = torch.cat((previous, current[:, None]), dim=1)
        else:
            current = self.normalize_action(current_action)
            if history_actions is None or history_actions.numel() == 0:
                previous = current[:, None].expand(-1, self.history_size - 1, -1)
            else:
                previous = (history_actions if history_actions.ndim == 3
                            else self.normalize_action_sequence(history_actions))
                assert previous.ndim == 3 and previous.shape[-1] == self.action_dim, (
                    f"history_actions must be normalized [B,T-1,{self.action_dim}] or raw "
                    f"[B,T-1,{self.action_block},{self.raw_action_dim}], got {tuple(history_actions.shape)}"
                )
            seq = torch.cat((previous, current[:, None]), dim=1)
        assert seq.ndim == 3 and seq.shape[1] == self.history_size and seq.shape[-1] == self.action_dim, (
            f"action sequence must be [B,{self.history_size},{self.action_dim}], got {tuple(seq.shape)}"
        )
        return seq

    def _as_action_block(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 2 and action.shape[-1] == self.action_dim:
            return action.view(action.shape[0], self.action_block, self.raw_action_dim)
        if action.ndim == 2 and action.shape[-1] == self.raw_action_dim and self.action_block == 1:
            return action[:, None]
        assert action.ndim == 3, (
            f"action must be [B,{self.action_dim}] or [B,{self.action_block},{self.raw_action_dim}], "
            f"got {tuple(action.shape)}"
        )
        assert action.shape[1:] == (self.action_block, self.raw_action_dim), (
            f"action block must be [B,{self.action_block},{self.raw_action_dim}], got {tuple(action.shape)}"
        )
        return action

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

    def encode_gaussian_sequence(self, frames: torch.Tensor) -> GaussianLatent:
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {tuple(frames.shape)}"
        batch, steps = frames.shape[:2]
        assert steps <= self.history_size, (
            f"history window {steps} exceeds configured history_size={self.history_size}"
        )
        # flat_frames: [B*T,3,H,W]
        flat_frames = frames.flatten(0, 1)
        flat = self.encode_gaussian(flat_frames)
        # mu/logvar: [B,T,192]
        mu = flat.mu.view(batch, steps, self.latent_dim)
        logvar = flat.logvar.view(batch, steps, self.latent_dim)
        return GaussianLatent(mu, logvar)

    def encode_mean_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode_gaussian_sequence(frames).mu

    def predict_next(self, mu_current: torch.Tensor, action: torch.Tensor,
                     history_actions: torch.Tensor | None = None,
                     action_is_normalized: bool = False) -> torch.Tensor:
        assert mu_current.ndim in {2, 3}, (
            f"mu_current must be [B,192] or history [B,T,192], got {tuple(mu_current.shape)}"
        )
        assert action.ndim in {2, 3}, f"action must be [B,A] or [B,K,A], got {tuple(action.shape)}"
        if mu_current.ndim == 2:
            # Backwards-compatible single-frame path: [B,192] -> [B,1,192].
            frame_latents = mu_current[:, None]
            z_current = mu_current
        else:
            assert mu_current.shape[-1] == self.latent_dim, (
                f"history latent dim must be {self.latent_dim}, got {mu_current.shape[-1]}"
            )
            assert mu_current.shape[1] <= self.history_size, (
                f"history window {mu_current.shape[1]} exceeds configured history_size={self.history_size}"
            )
            # frame_latents: [B,T,192], z_current: [B,192]
            frame_latents = mu_current
            z_current = mu_current[:, -1]
        # action_seq: [B,T,action_block*2], matching LeWorld act_trunc.
        action_seq = self.action_sequence(history_actions, action, action_is_normalized)
        # e_action_seq: [B,T,192]
        e_action_seq = self.action_encoder(action_seq.flatten(0, 1)).view(action_seq.shape[0], action_seq.shape[1], -1)
        # h_t: [B,192] summarizes paired [z_{t-N+1:t}, a_{t-N+1:t}].
        h_t = self.temporal_encoder(frame_latents, e_action_seq)
        # e_action: [B,192] current action block embedding.
        e_action = e_action_seq[:, -1]
        # predictor_input: [B,576] = concat(h_t, action, interaction).
        predictor_input = torch.cat((h_t, e_action, h_t * e_action), dim=-1)
        # delta_pred: [B,192]
        delta_pred = self.dynamics_predictor(predictor_input)
        # mu_pred_next: [B,192] residual prediction around current frame latent.
        return z_current + delta_pred

    def predict_next_sequence(self, frame_latents: torch.Tensor, history_actions: torch.Tensor,
                              current_action: torch.Tensor,
                              action_is_normalized: bool = False) -> torch.Tensor:
        """LeWorld-style multi-position one-step prediction over a real latent window.

        frame_latents:    [B,T,192] real encoded latents [z0,z1,z2]
        history_actions:  [B,T-1,action_block,2] raw actions [a0,a1], or normalized [B,T-1,10]
        current_action:   [B,action_block,2] raw action a2, or normalized [B,10]
        returns:          [B,T,192] predictions [z1_hat,z2_hat,z3_hat]
        """
        assert frame_latents.ndim == 3, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert frame_latents.shape[1] == self.history_size, (
            f"LeWorld-style v3 prediction expects T={self.history_size}, got {frame_latents.shape[1]}"
        )
        assert frame_latents.shape[-1] == self.latent_dim, (
            f"frame latent dim must be {self.latent_dim}, got {frame_latents.shape[-1]}"
        )
        # action_seq: [B,T,action_block*2], aligned as [a0,a1,a2].
        action_seq = self.action_sequence(history_actions, current_action, action_is_normalized)
        assert action_seq.shape[:2] == frame_latents.shape[:2], (
            f"action_seq [B,T] {tuple(action_seq.shape[:2])} must match frame_latents "
            f"{tuple(frame_latents.shape[:2])}"
        )
        # e_action_seq: [B,T,192]
        e_action_seq = self.action_encoder(action_seq.flatten(0, 1)).view(
            action_seq.shape[0], action_seq.shape[1], self.latent_dim
        )
        # h_seq: [B,T,192], token i only sees tokens <= i.
        h_seq = self.temporal_encoder.encode_tokens(frame_latents, e_action_seq)
        # predictor_input: [B,T,576] = concat(h_i, e_a_i, interaction).
        predictor_input = torch.cat((h_seq, e_action_seq, h_seq * e_action_seq), dim=-1)
        # delta_pred: [B,T,192]
        delta_pred = self.dynamics_predictor(predictor_input.flatten(0, 1)).view(
            frame_latents.shape[0], frame_latents.shape[1], self.latent_dim
        )
        # z_pred_next_seq: [B,T,192] = [z1_hat,z2_hat,z3_hat].
        return frame_latents + delta_pred

    def predict_action_from_delta(self, delta_pred: torch.Tensor) -> torch.Tensor:
        assert delta_pred.ndim == 2, f"delta_pred must be [B,{self.latent_dim}], got {tuple(delta_pred.shape)}"
        assert delta_pred.shape[-1] == self.latent_dim, (
            f"delta_pred latent dim must be {self.latent_dim}, got {delta_pred.shape[-1]}"
        )
        # action_hat: [B, action_block * action_dim], target is z-scored real action.
        return self.action_consistency_head(delta_pred)

    @staticmethod
    def kl_loss(latent: GaussianLatent) -> torch.Tensor:
        return -0.5 * (1.0 + latent.logvar - latent.mu.square() - latent.logvar.exp()).mean()
