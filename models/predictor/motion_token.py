import torch
from torch import nn


class MotionEncoder(nn.Module):
    """Encode a 3-step action history into one motion token.

    Input shape:  history_actions [B, 3, action_dim]
    Output shape: motion_token    [B, 1, 192]
    """

    def __init__(self, action_dim: int, hidden_dim: int = 192, num_layers: int = 2,
                 num_heads: int = 3, dropout: float = 0.0, history_size: int = 3):
        super().__init__()
        self.history_size = history_size
        self.hidden_dim = hidden_dim
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, history_size, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.motion_pool = nn.Linear(hidden_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        nn.init.xavier_uniform_(self.action_projection.weight)
        nn.init.zeros_(self.action_projection.bias)
        nn.init.xavier_uniform_(self.motion_pool.weight)
        nn.init.zeros_(self.motion_pool.bias)

    def forward(self, history_actions: torch.Tensor) -> torch.Tensor:
        assert history_actions.ndim == 3, f"history_actions must be [B,3,A], got {tuple(history_actions.shape)}"
        assert history_actions.shape[1] == self.history_size, (
            f"MotionEncoder expects {self.history_size} actions, got {history_actions.shape[1]}"
        )
        # action_tokens: [B, 3, 192]
        action_tokens = self.action_projection(history_actions)
        action_tokens = action_tokens + self.temporal_embedding
        # encoded_actions: [B, 3, 192]
        encoded_actions = self.encoder(action_tokens)
        # motion_token: [B, 1, 192]; the three action tokens are pooled away.
        motion_token = self.motion_pool(encoded_actions.mean(dim=1, keepdim=True))
        return motion_token


class StateActionTransformer(nn.Module):
    """Bidirectional self-attention over 3 frame tokens plus 1 motion token."""

    def __init__(self, hidden_dim: int = 192, num_layers: int = 2,
                 num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        assert tokens.ndim == 3 and tokens.shape[1] == 4, f"tokens must be [B,4,192], got {tuple(tokens.shape)}"
        # Bidirectional attention: no causal mask is passed.
        return self.encoder(tokens)


class FlowHead(nn.Module):
    """Predict latent flow delta_z from current-frame and motion outputs."""

    def __init__(self, hidden_dim: int = 192, mlp_hidden_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, frame_out: torch.Tensor, motion_out: torch.Tensor) -> torch.Tensor:
        assert frame_out.shape == motion_out.shape, (
            f"frame_out and motion_out must have same shape, got {tuple(frame_out.shape)} and {tuple(motion_out.shape)}"
        )
        # features: [B, 4*192]
        features = torch.cat(
            (frame_out, motion_out, frame_out * motion_out, frame_out - motion_out),
            dim=-1,
        )
        # delta_z: [B, 192]
        return self.net(features)


class MotionTokenPredictor(nn.Module):
    """LeWorld-style latent predictor driven by one pooled motion token.

    frame_latents:    [B, 3, 192] = z_{t-2}, z_{t-1}, z_t
    history_actions:  [B, 3, A]   = a_{t-2}, a_{t-1}, a_t
    z_pred_next:      [B, 192]     = z_t + delta_z
    """

    def __init__(self, action_dim: int, hidden_dim: int = 192, history_size: int = 3,
                 motion_layers: int = 2, transformer_layers: int = 2,
                 num_heads: int = 3, dropout: float = 0.0, flow_hidden_dim: int = 384):
        super().__init__()
        self.history_size = history_size
        self.hidden_dim = hidden_dim
        self.motion_encoder = MotionEncoder(action_dim, hidden_dim, motion_layers, num_heads, dropout, history_size)
        self.frame_temporal_embedding = nn.Parameter(torch.zeros(1, history_size, hidden_dim))
        self.token_type_embedding = nn.Embedding(2, hidden_dim)
        self.state_action_transformer = StateActionTransformer(hidden_dim, transformer_layers, num_heads, dropout)
        self.flow_head = FlowHead(hidden_dim, flow_hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.frame_temporal_embedding, std=0.02)
        nn.init.trunc_normal_(self.token_type_embedding.weight, std=0.02)

    def forward(self, frame_latents: torch.Tensor, history_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert frame_latents.ndim == 3, f"frame_latents must be [B,3,192], got {tuple(frame_latents.shape)}"
        assert frame_latents.shape[1] == self.history_size, (
            f"MotionTokenPredictor expects {self.history_size} frame tokens, got {frame_latents.shape[1]}"
        )
        assert frame_latents.shape[-1] == self.hidden_dim, (
            f"frame latent dim must be {self.hidden_dim}, got {frame_latents.shape[-1]}"
        )
        assert history_actions.ndim == 3 and history_actions.shape[:2] == frame_latents.shape[:2], (
            f"history_actions must be [B,3,A] aligned with frame_latents, got {tuple(history_actions.shape)}"
        )

        batch = frame_latents.shape[0]
        device = frame_latents.device
        frame_type = torch.zeros(batch, self.history_size, dtype=torch.long, device=device)
        motion_type = torch.ones(batch, 1, dtype=torch.long, device=device)

        # frame_tokens: [B, 3, 192]
        frame_tokens = (
            frame_latents
            + self.frame_temporal_embedding
            + self.token_type_embedding(frame_type)
        )
        # motion_token: [B, 1, 192]
        motion_token = self.motion_encoder(history_actions) + self.token_type_embedding(motion_type)
        # tokens: [B, 4, 192]
        tokens = torch.cat((frame_tokens, motion_token), dim=1)
        # outputs: [B, 4, 192]
        outputs = self.state_action_transformer(tokens)
        # frame_out: [B, 192], motion_out: [B, 192]
        frame_out, motion_out = outputs[:, 2], outputs[:, 3]
        # delta_z: [B, 192]
        delta_z = self.flow_head(frame_out, motion_out)
        # z_pred_next: [B, 192]
        z_pred_next = frame_latents[:, 2] + delta_z
        return z_pred_next, delta_z
