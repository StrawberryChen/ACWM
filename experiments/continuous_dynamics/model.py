"""Shared, randomly initialized ViT and autonomous action-conditioned dynamics."""
import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, image_size=96, patch_size=16, latent_dim=192):
        super().__init__()
        from timm.models.vision_transformer import VisionTransformer
        if image_size % patch_size or latent_dim != 192:
            raise ValueError("v1 requires divisible image/patch sizes and latent_dim=192")
        # Random ViT-Tiny: 12 layers, 3 heads, final CLS; standard ViT LayerNorm,
        # but NO BatchNorm, pretrained weights, teacher, or distribution constraint.
        self.backbone = VisionTransformer(img_size=image_size, patch_size=patch_size,
                                          in_chans=3, num_classes=0, embed_dim=192,
                                          depth=12, num_heads=3, global_pool="token")
        self.projection = nn.Linear(192, latent_dim)
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.image_size = image_size

    def forward(self, images):
        assert images.ndim == 4 and images.shape[1:] == (3, self.image_size, self.image_size)
        # images [B,3,H,W] in [0,1] -> final CLS [B,192] -> O [B,192].
        return self.projection(self.backbone((images-self.mean)/self.std))


class VelocityField(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.action_encoder = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, latent_dim))
        self.network = nn.Sequential(nn.Linear(2*latent_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, latent_dim))

    def forward(self, state, action):
        assert state.ndim == 2 and action.shape == (len(state), 2)
        # O [B,D], A [B,2], embedding [B,D], concatenation [B,2D], V [B,D].
        return self.network(torch.cat((state, self.action_encoder(action)), -1))


class EulerIntegrator(nn.Module):
    def __init__(self, steps=4):
        super().__init__()
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("integration steps must be a positive integer")
        self.steps = steps

    def forward(self, field, state, action):
        states, velocities = [state], []
        for _ in range(self.steps):
            velocity = field(state, action)  # [B,D]; SAME A throughout one raw step
            velocities.append(velocity)
            state = state+velocity/self.steps  # complete autograd through ALL K steps
            states.append(state)
        return state, torch.stack(states, 1), torch.stack(velocities, 1)
        # endpoint [B,D], internal states [B,K+1,D], velocities [B,K,D]


class ContinuousWorldModel(nn.Module):
    def __init__(self, image_size=96, patch_size=16, latent_dim=192, hidden_dim=512, integration_steps=4):
        super().__init__()
        self.encoder = FrameEncoder(image_size, patch_size, latent_dim)
        self.field = VelocityField(latent_dim, hidden_dim)
        self.integrator = EulerIntegrator(integration_steps)

    def forward(self, frames, actions):
        assert frames.ndim == 5 and frames.shape[1] == 2  # [B,2,3,H,W]
        assert actions.shape == (len(frames), 1, 2)  # [B,1,2], A_t: X_t -> X_{t+1}
        encoded = self.encoder(frames.flatten(0, 1)).reshape(len(frames), 2, -1)
        start, target = encoded.unbind(1)  # both [B,192], BOTH receive gradients
        prediction, states, velocities = self.integrator(self.field, start, actions[:, 0])
        return dict(start=start, target=target, prediction=prediction, states=states, velocities=velocities)

    def rollout(self, state, actions):
        assert state.ndim == 2 and actions.ndim == 3 and actions.shape[0] == len(state)
        assert actions.shape[-1] == 2 and actions.shape[1] > 0
        results = []
        for action in actions.unbind(1):
            state, _, _ = self.integrator(self.field, state, action)
            results.append(state)
        return torch.stack(results, 1)  # [B,H,D]; no image GT or planner involved
