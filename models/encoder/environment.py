import torch
from torch import nn

from .common import ImageBackbone


class CNNEnvironmentEncoder(nn.Module):
    """Encodes the complete environment from only the current observation."""

    def __init__(self, image_channels: int, state_dim: int):
        super().__init__()
        self.network = ImageBackbone(image_channels, state_dim)

    def forward(self, current_frame: torch.Tensor) -> torch.Tensor:
        return self.network(current_frame)

