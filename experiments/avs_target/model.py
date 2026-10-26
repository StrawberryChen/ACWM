"""Random CNN CSE and noncausal trajectory velocity Transformer."""
import math

import torch
from torch import nn


def sinusoidal(times, width):
    assert width % 2 == 0
    frequency = torch.exp(-math.log(10000)*torch.arange(width//2, device=times.device)/(width//2))
    phase = times[..., None]*frequency
    return torch.cat((phase.sin(), phase.cos()), -1)


class CurrentStateEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        layers, channels = [], 3
        for width in (32, 64, 128, 256):
            layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1), nn.SiLU()))
            channels = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, latent_dim))

    def forward(self, image):
        assert image.ndim == 4 and image.shape[1:] == (3, 96, 96)
        return self.net(image)  # [B,D], no L2/LayerNorm/BatchNorm on CSE latent.


class FlowPredictor(nn.Module):
    def __init__(self, motion_dim, latent_dim=128, max_horizon=16, model_dim=256,
                 layers=4, heads=8, dropout=0.0):
        super().__init__()
        assert model_dim % heads == 0 and model_dim % 2 == 0 and max_horizon > 0
        self.motion_dim, self.max_horizon = motion_dim, max_horizon
        self.motion_projection = nn.Linear(motion_dim, model_dim)
        self.action_encoder = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 128))
        self.action_projection = nn.Linear(128, model_dim)
        self.state_projection = nn.Linear(latent_dim, model_dim)
        self.time_mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.model_dim = model_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, heads, 4*model_dim, dropout,
                                       activation="gelu", batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        # TransformerEncoder clones initial weights; independently initialize each layer.
        for layer in self.transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.output = nn.Linear(model_dim, motion_dim)

    def forward(self, noisy_motion, flow_time, state, actions):
        # X_s [B,H,Dm], s [B], O [B,D], A [B,H,2] -> velocity [B,H,Dm].
        batch, horizon, _ = noisy_motion.shape
        assert 1 <= horizon <= self.max_horizon
        assert noisy_motion.shape[-1] == self.motion_dim and actions.shape == (batch, horizon, 2)
        assert flow_time.shape == (batch,) and state.ndim == 2 and state.shape[0] == batch
        # Flow time s (noise->measurement) is distinct from world transition index h.
        flow_embedding = self.time_mlp(sinusoidal(flow_time*1000, self.model_dim))
        world_embedding = sinusoidal(torch.arange(horizon, device=state.device).float(), self.model_dim)
        future = (self.motion_projection(noisy_motion)+self.action_projection(self.action_encoder(actions))
                  +world_embedding[None]+flow_embedding[:, None])
        tokens = torch.cat((self.state_projection(state)[:, None], future), 1)
        encoded = self.transformer(tokens)  # bidirectional, no causal mask
        return self.output(encoded[:, 1:])


class AVSFlowWorldModel(nn.Module):
    def __init__(self, motion_dim, latent_dim=128, horizon=16, model_dim=256, layers=4, heads=8, dropout=0.0):
        super().__init__()
        self.horizon, self.motion_dim = horizon, motion_dim
        self.cse = CurrentStateEncoder(latent_dim)
        self.predictor = FlowPredictor(motion_dim, latent_dim, horizon, model_dim, layers, heads, dropout)

    def forward(self, frames, actions, noisy_motion, flow_time):
        return self.predictor(noisy_motion, flow_time, self.cse(frames), actions)
