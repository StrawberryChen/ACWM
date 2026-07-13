import torch
from torch import nn

from .common import ImageBackbone, TimmViTBackbone


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


class GRUViTAgentEncoder(GRUAgentEncoder):
    """Temporal agent encoder using an independent pretrained ViT-Tiny."""

    def __init__(self, image_channels: int, action_dim: int, state_dim: int,
                 feature_dim: int = 192,
                 model_name: str | None = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, image_size: int = 224, trainable: bool = True,
                 patch_size: int = 16, projector_hidden_dim: int = 2048):
        super().__init__(image_channels, action_dim, state_dim, feature_dim)
        self.visual = TimmViTBackbone(image_channels, feature_dim, model_name, pretrained, image_size, trainable,
                                      patch_size, projector_hidden_dim)
