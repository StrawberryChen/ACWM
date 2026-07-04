import torch
from torch import nn


class ImageBackbone(nn.Module):
    """Small resolution-agnostic CNN used by either independent encoder."""

    def __init__(self, in_channels: int, output_dim: int, channels: tuple[int, ...] = (32, 64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for width in channels:
            layers += [nn.Conv2d(current, width, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            current = width
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(current, output_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

