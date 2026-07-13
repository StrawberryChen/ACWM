import torch
from torch import nn


class InverseDynamicsHead(nn.Module):
    """Predict action_t from true adjacent frame latents z_t and z_{t+1}.

    z_current:   [B, 192]
    z_next:      [B, 192]
    action_pred: [B, action_dim]
    """

    def __init__(self, latent_dim: int = 192, action_dim: int = 2,
                 hidden_dim: int = 384, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("InverseDynamicsHead.num_layers must be >= 1")
        layers: list[nn.Module] = []
        input_dim = latent_dim * 4
        for layer_index in range(num_layers - 1):
            layers += [nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim if num_layers > 1 else input_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z_current: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        assert z_current.ndim == 2, f"z_current must be [B,D], got {tuple(z_current.shape)}"
        assert z_next.shape == z_current.shape, (
            f"z_next must match z_current, got {tuple(z_next.shape)} vs {tuple(z_current.shape)}"
        )
        # inverse_input: [B, 4*192]
        inverse_input = torch.cat(
            (z_current, z_next, z_next - z_current, z_current * z_next),
            dim=-1,
        )
        # action_pred: [B, action_dim]
        return self.net(inverse_input)
