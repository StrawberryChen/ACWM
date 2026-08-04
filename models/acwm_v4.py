from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class V4Projection(nn.Module):
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


class V4TemporalActionEncoder(nn.Module):
    """Causal encoder over paired frame/action history tokens.

    Input shape:  frame_latents  [B, T, 192]
                  action_latents [B, T, 192]
    Output shape: encoded        [B, T, 192]
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

    def forward(self, frame_latents: torch.Tensor, action_latents: torch.Tensor) -> torch.Tensor:
        assert frame_latents.ndim == 3, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert action_latents.shape == frame_latents.shape, (
            f"action_latents {tuple(action_latents.shape)} must match frame_latents {tuple(frame_latents.shape)}"
        )
        assert frame_latents.shape[1] <= self.history_size, (
            f"history window {frame_latents.shape[1]} exceeds configured history_size={self.history_size}"
        )
        steps = frame_latents.shape[1]
        # tokens: [B,T,192] = state token + current transition action token + temporal position.
        tokens = frame_latents + action_latents + self.temporal_pos[:, -steps:]
        # causal_mask: [T,T], token i sees only tokens <= i.
        causal_mask = torch.triu(torch.ones(steps, steps, device=frame_latents.device, dtype=torch.bool), diagonal=1)
        # encoded: [B,T,192]
        return self.transformer(tokens, mask=causal_mask)


class ACWMV4ResidualWorldModel(nn.Module):
    """ACWM v4: deterministic residual latent transition model.

    The predictor is explicitly parameterized as:
        delta_z = f(z_history, action_history)
        z_next  = z_current + delta_z

    It keeps the LeWorld-style deterministic ViT-Tiny frame encoder/projector and
    removes the v3 Gaussian mean/logvar/KL path so the experiment isolates the
    residual transition idea.
    """

    def __init__(self, image_channels: int = 3, latent_dim: int = 192, action_dim: int = 2,
                 image_size: int = 224, patch_size: int = 14, vit_depth: int = 12,
                 vit_heads: int = 3, mlp_ratio: float = 4.0, history_size: int = 3,
                 temporal_layers: int = 2, temporal_heads: int = 3, temporal_dropout: float = 0.0,
                 action_block: int = 5, hidden_dim: int = 512):
        super().__init__()
        try:
            from timm.models.vision_transformer import VisionTransformer
        except ImportError as error:
            raise ImportError("ACWM v4 requires timm>=1.0") from error
        self.latent_dim = latent_dim
        self.raw_action_dim = action_dim
        self.action_block = action_block
        self.action_dim = action_dim * action_block
        self.history_size = history_size
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
        self.projection = V4Projection(latent_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.temporal_encoder = V4TemporalActionEncoder(
            latent_dim=latent_dim,
            history_size=history_size,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
        )
        self.transition_predictor = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.register_buffer("image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("action_mean", torch.zeros(action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(action_dim), persistent=True)

    def set_action_stats(self, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
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

    def _as_action_block(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 2 and action.shape[-1] == self.action_dim:
            return action.view(action.shape[0], self.action_block, self.raw_action_dim)
        if action.ndim == 2 and action.shape[-1] == self.raw_action_dim and self.action_block == 1:
            return action[:, None]
        assert action.ndim == 3, (
            f"action must be [B,{self.action_dim}] or [B,{self.action_block},{self.raw_action_dim}], "
            f"got {tuple(action.shape)}"
        )
        assert tuple(action.shape[1:]) == (self.action_block, self.raw_action_dim), (
            f"action block must be [B,{self.action_block},{self.raw_action_dim}], got {tuple(action.shape)}"
        )
        return action

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B,action_block,2] raw Push-T action block; output [B,action_block*2].
        action = self._as_action_block(action)
        action_norm = (action - self.action_mean) / self.action_std
        return action_norm.flatten(1)

    def normalize_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: [B,T,action_block,2] raw blocks; output [B,T,action_block*2].
        assert actions.ndim == 4, (
            f"actions must be [B,T,{self.action_block},{self.raw_action_dim}], got {tuple(actions.shape)}"
        )
        batch, steps = actions.shape[:2]
        flat = self.normalize_action(actions.flatten(0, 1))
        return flat.view(batch, steps, self.action_dim)

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        # action_norm: [B,action_block*2]; output raw Push-T action block [B,action_block,2].
        action_norm = self._as_action_block(action_norm)
        return action_norm * self.action_std + self.action_mean

    def action_sequence(self, history_actions: torch.Tensor | None, current_action: torch.Tensor,
                        action_is_normalized: bool = False) -> torch.Tensor:
        """Build action history ending in the current transition action.

        history_actions raw: [B,T-1,action_block,2], current_action raw: [B,action_block,2]
        output normalized:   [B,T,action_block*2]
        """
        if action_is_normalized:
            current = current_action if current_action.ndim == 2 else current_action.flatten(1)
            if history_actions is None or history_actions.numel() == 0:
                previous = current[:, None].expand(-1, self.history_size - 1, -1)
            else:
                previous = self.normalize_action_sequence(history_actions) if history_actions.ndim == 4 else history_actions
        else:
            current = self.normalize_action(current_action)
            if history_actions is None or history_actions.numel() == 0:
                previous = current[:, None].expand(-1, self.history_size - 1, -1)
            else:
                previous = self.normalize_action_sequence(history_actions) if history_actions.ndim == 4 else history_actions
        assert previous.ndim == 3 and previous.shape[-1] == self.action_dim, (
            f"history_actions must be [B,T-1,{self.action_dim}], got {tuple(previous.shape)}"
        )
        seq = torch.cat((previous, current[:, None]), dim=1)
        assert seq.shape[1:] == (self.history_size, self.action_dim), (
            f"action sequence must be [B,{self.history_size},{self.action_dim}], got {tuple(seq.shape)}"
        )
        return seq

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B,3,H,W] in [0,1]; output z: [B,192].
        assert images.ndim == 4, f"images must be [B,3,H,W], got {tuple(images.shape)}"
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        images = (images - self.image_mean) / self.image_std
        # cls: [B,192]
        cls = self.encoder(images)
        # z: [B,192]
        return self.projection(cls)

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,T,3,H,W]; output latents: [B,T,192].
        assert frames.ndim == 5, f"frames must be [B,T,3,H,W], got {tuple(frames.shape)}"
        batch, steps = frames.shape[:2]
        assert steps <= self.history_size, (
            f"history window {steps} exceeds configured history_size={self.history_size}"
        )
        flat = self.encode(frames.flatten(0, 1))
        return flat.view(batch, steps, self.latent_dim)

    def _predict_delta_from_sequence(self, frame_latents: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        assert frame_latents.ndim == 3 and frame_latents.shape[-1] == self.latent_dim, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert action_seq.ndim == 3 and action_seq.shape[:2] == frame_latents.shape[:2], (
            f"action_seq [B,T,A] must align with frame_latents [B,T,D], got "
            f"{tuple(action_seq.shape)} vs {tuple(frame_latents.shape)}"
        )
        # action_latents: [B,T,192]
        action_latents = self.action_encoder(action_seq.flatten(0, 1)).view(
            action_seq.shape[0], action_seq.shape[1], self.latent_dim
        )
        # h_seq: [B,T,192]
        h_seq = self.temporal_encoder(frame_latents, action_latents)
        # transition_input: [B,T,576] = concat(state summary, action, interaction).
        transition_input = torch.cat((h_seq, action_latents, h_seq * action_latents), dim=-1)
        # delta_seq: [B,T,192]
        delta_seq = self.transition_predictor(transition_input.flatten(0, 1)).view(
            frame_latents.shape[0], frame_latents.shape[1], self.latent_dim
        )
        return delta_seq

    def predict_next(self, frame_latents: torch.Tensor, action: torch.Tensor,
                     history_actions: torch.Tensor | None = None,
                     action_is_normalized: bool = False) -> torch.Tensor:
        """Predict one residual transition from the current history.

        frame_latents: [B,T,192] or [B,192]
        action:        [B,action_block,2] raw or [B,action_block*2] normalized
        output:        [B,192] = z_current + delta_z
        """
        assert frame_latents.ndim in {2, 3}, (
            f"frame_latents must be [B,192] or [B,T,192], got {tuple(frame_latents.shape)}"
        )
        if frame_latents.ndim == 2:
            frame_latents = frame_latents[:, None]
        z_current = frame_latents[:, -1]
        action_seq = self.action_sequence(history_actions, action, action_is_normalized)
        if action_seq.shape[1] != frame_latents.shape[1]:
            action_seq = action_seq[:, -frame_latents.shape[1]:]
        delta = self._predict_delta_from_sequence(frame_latents, action_seq)[:, -1]
        # z_next: [B,192]
        return z_current + delta

    def predict_next_sequence(self, frame_latents: torch.Tensor, history_actions: torch.Tensor,
                              current_action: torch.Tensor,
                              action_is_normalized: bool = False) -> torch.Tensor:
        """LeWorld-style shifted one-step residual supervision.

        frame_latents:   [B,T,192] = [z0,z1,z2]
        history_actions: [B,T-1,5,2] raw [a0,a1] or normalized [B,T-1,10]
        current_action:  [B,5,2] raw a2 or normalized [B,10]
        returns:         [B,T,192] = [z1_hat,z2_hat,z3_hat]
        """
        assert frame_latents.ndim == 3, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert frame_latents.shape[1] == self.history_size, (
            f"v4 expects history_size={self.history_size}, got {frame_latents.shape[1]}"
        )
        action_seq = self.action_sequence(history_actions, current_action, action_is_normalized)
        delta_seq = self._predict_delta_from_sequence(frame_latents, action_seq)
        return frame_latents + delta_seq
