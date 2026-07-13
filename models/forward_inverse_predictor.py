import torch
from torch import nn

from .inverse_dynamics_head import InverseDynamicsHead


class ActionEncoder(nn.Module):
    """Shared action encoder applied to every action_t.

    action:       [B, action_dim]
    action_token: [B, 192]
    """

    def __init__(self, action_dim: int = 2, latent_dim: int = 192,
                 hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or latent_dim * 4
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        assert action.ndim == 2, f"action must be [B,A], got {tuple(action.shape)}"
        return self.net(action.float())


class TemporalActionForwardPredictor(nn.Module):
    """Forward branch: history latents + action_t -> z_pred_{t+1}.

    frame_latents: [B, 3, 192] = [z_{t-2}, z_{t-1}, z_t]
    action:        [B, A]      = action_t for frame_t -> frame_{t+1}
    z_pred_next:   [B, 192]
    """

    def __init__(self, action_dim: int = 2, latent_dim: int = 192, history_size: int = 3,
                 num_layers: int = 2, num_heads: int = 3, hidden_dim: int = 384,
                 dropout: float = 0.0, action_hidden_dim: int | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_size = history_size
        self.action_encoder = ActionEncoder(action_dim, latent_dim, action_hidden_dim)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, history_size, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.forward_mlp = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        for module in self.forward_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, frame_latents: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert frame_latents.ndim == 3, f"frame_latents must be [B,3,192], got {tuple(frame_latents.shape)}"
        assert frame_latents.shape[1] == self.history_size, (
            f"expected history_size={self.history_size}, got {frame_latents.shape[1]}"
        )
        assert frame_latents.shape[-1] == self.latent_dim, (
            f"expected latent_dim={self.latent_dim}, got {frame_latents.shape[-1]}"
        )
        assert action.shape[0] == frame_latents.shape[0], "action batch must match frame_latents batch"
        # temporal_input: [B, 3, 192]
        temporal_input = frame_latents + self.temporal_embedding
        causal_mask = torch.triu(
            torch.ones(self.history_size, self.history_size, dtype=torch.bool, device=frame_latents.device),
            diagonal=1,
        )
        # h: [B, 3, 192]
        h = self.temporal_transformer(temporal_input, mask=causal_mask)
        # h_t: [B, 192]
        h_t = h[:, -1, :]
        # u_t: [B, 192]
        u_t = self.action_encoder(action)
        # forward_input: [B, 3*192]
        forward_input = torch.cat((h_t, u_t, h_t * u_t), dim=-1)
        # z_pred_next: [B, 192]
        return self.forward_mlp(forward_input)


class ForwardInversePredictor(nn.Module):
    """Forward world model plus inverse dynamics head used only for training."""

    def __init__(self, action_dim: int = 2, latent_dim: int = 192, history_size: int = 3,
                 forward_num_layers: int = 2, forward_num_heads: int = 3,
                 forward_hidden_dim: int = 384, dropout: float = 0.0,
                 action_hidden_dim: int | None = None, inverse_enabled: bool = True,
                 inverse_hidden_dim: int = 384, inverse_num_layers: int = 2):
        super().__init__()
        self.forward_predictor = TemporalActionForwardPredictor(
            action_dim=action_dim,
            latent_dim=latent_dim,
            history_size=history_size,
            num_layers=forward_num_layers,
            num_heads=forward_num_heads,
            hidden_dim=forward_hidden_dim,
            dropout=dropout,
            action_hidden_dim=action_hidden_dim,
        )
        self.inverse_enabled = inverse_enabled
        self.inverse_head = (
            InverseDynamicsHead(latent_dim, action_dim, inverse_hidden_dim, inverse_num_layers)
            if inverse_enabled
            else None
        )

    def forward(self, frame_latents: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward_predictor(frame_latents, action)

    def inverse(self, z_current: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        if self.inverse_head is None:
            raise RuntimeError("inverse head is disabled")
        return self.inverse_head(z_current, z_next)
