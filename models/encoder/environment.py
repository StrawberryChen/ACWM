import torch
from torch import nn

from .common import ImageBackbone, TimmViTBackbone


class CNNEnvironmentEncoder(nn.Module):
    """Encodes the complete environment from only the current observation."""

    def __init__(self, image_channels: int, state_dim: int):
        super().__init__()
        self.network = ImageBackbone(image_channels, state_dim)

    def forward(self, current_frame: torch.Tensor) -> torch.Tensor:
        return self.network(current_frame)


class ViTEnvironmentEncoder(nn.Module):
    """Independent pretrained ViT-Tiny for task-environment state."""

    def __init__(self, image_channels: int, state_dim: int,
                 model_name: str = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, image_size: int = 224, trainable: bool = True):
        super().__init__()
        self.network = TimmViTBackbone(image_channels, state_dim, model_name, pretrained, image_size, trainable)

    def forward(self, current_frame: torch.Tensor) -> torch.Tensor:
        return self.network(current_frame)
