import torch
from torch import nn
from torch.nn import functional as F


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


class LeWorldProjector(nn.Module):
    """Official LeWorld-style projector: Linear -> BatchNorm1d -> GELU -> Linear.

    Input shape:  embeddings [B, D]
    Output shape: projected  [B, 192]
    """

    def __init__(self, input_dim: int, output_dim: int = 192, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        assert embeddings.ndim == 2, f"projector expects [B,D], got {tuple(embeddings.shape)}"
        return self.net(embeddings)


class TimmViTBackbone(nn.Module):
    """ViT-Tiny with LeWM-style CLS-token embedding.

    Push-T observations are resized to the configured resolution. A projection
    keeps the downstream latent dimension independently configurable.
    """

    def __init__(self, image_channels: int, output_dim: int,
                 model_name: str | None = "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
                 pretrained: bool = True, image_size: int = 224,
                 trainable: bool = True, patch_size: int = 16,
                 projector_hidden_dim: int = 2048):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("ViT encoder requires timm>=1.0") from error
        self.image_size = image_size
        if model_name is None:
            from timm.models.vision_transformer import VisionTransformer
            self.backbone = VisionTransformer(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=image_channels,
                num_classes=0,
                embed_dim=192,
                depth=12,
                num_heads=3,
            )
        else:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, in_chans=image_channels
            )
        feature_dim = self.backbone.num_features
        self.projection = LeWorldProjector(feature_dim, output_dim, projector_hidden_dim)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)
        if not trainable:
            self.backbone.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if images.shape[1] == 3:
            images = (images - self.mean) / self.std
        return self.projection(self.backbone(images))
