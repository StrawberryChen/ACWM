import torch
from torch import nn

from .common import ImageBackbone


class GRUAgentEncoder(nn.Module):
    """Encodes controllable state from visual and control history."""

    def __init__(self, image_channels: int, action_dim: int, state_dim: int, feature_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.visual = ImageBackbone(image_channels, feature_dim)
        self.action_embedding = nn.Linear(action_dim, feature_dim)
        self.temporal = nn.GRU(feature_dim * 2, state_dim, batch_first=True)

    def forward(self, history_frames: torch.Tensor, history_actions: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = history_frames.shape
        if history_actions.shape[1] != steps - 1:
            raise ValueError("history_actions must contain K-1 transitions for K frames")
        visual = self.visual(history_frames.reshape(batch * steps, channels, height, width)).view(batch, steps, -1)
        zero = history_actions.new_zeros(batch, 1, self.action_dim)
        aligned_actions = torch.cat((zero, history_actions), dim=1)
        control = self.action_embedding(aligned_actions)
        _, hidden = self.temporal(torch.cat((visual, control), dim=-1))
        return hidden[-1]

