"""Direct finite-difference velocity supervision at raw or observation scale."""
import torch
from torch import nn

from .model import EulerIntegrator, FrameEncoder


def velocity_head(latent_dim, action_embedding_dim, hidden_dim):
    return nn.Sequential(nn.Linear(latent_dim + action_embedding_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, latent_dim))


class ControlledVelocityField(nn.Module):
    def __init__(self, latent_dim, action_embedding_dim, hidden_dim, action_dim=2):
        super().__init__()
        self.action_dim = action_dim
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, action_embedding_dim), nn.SiLU(),
                                            nn.Linear(action_embedding_dim, action_embedding_dim))
        self.network = velocity_head(latent_dim, action_embedding_dim, hidden_dim)

    def forward(self, state, action):
        # All leading axes are independent examples (no action-prefix lookahead).
        assert state.shape[:-1] == action.shape[:-1] and action.shape[-1] == self.action_dim
        return self.network(torch.cat((state, self.action_encoder(action)), dim=-1))


class CrossAttentionVelocityField(nn.Module):
    """One state query reads the ordered raw-action tokens of one block."""

    def __init__(self, latent_dim, action_embedding_dim, hidden_dim, action_block, heads):
        super().__init__()
        if action_block < 2:
            raise ValueError("cross_attention needs multiple raw actions per block")
        if type(heads) is not int or heads < 1 or action_embedding_dim % heads:
            raise ValueError("attention_heads must be positive and divide action_embedding_dim")
        self.latent_dim = latent_dim
        self.action_block = action_block
        self.action_dim = 2 * action_block  # Same flat block input as the baseline and diagnostics.
        self.action_encoder = nn.Sequential(nn.Linear(2, action_embedding_dim), nn.SiLU(),
                                            nn.Linear(action_embedding_dim, action_embedding_dim))
        self.action_position = nn.Parameter(torch.empty(action_block, action_embedding_dim))
        nn.init.normal_(self.action_position, std=0.02)
        self.state_query = nn.Linear(latent_dim, action_embedding_dim)
        self.attention = nn.MultiheadAttention(action_embedding_dim, heads, batch_first=True)
        self.network = velocity_head(latent_dim, action_embedding_dim, hidden_dim)

    def forward(self, state, action):
        # Training: leading axes [B,H_obs]; rollout: [B] (or CEM candidates [N]).
        assert state.shape[:-1] == action.shape[:-1] and action.shape[-1] == self.action_dim
        assert state.shape[-1] == self.latent_dim
        raw = action.reshape(-1, self.action_block, 2)  # [N_blocks,5,2]
        tokens = self.action_encoder(raw) + self.action_position  # [N_blocks,5,64]
        query = self.state_query(state.reshape(-1, self.latent_dim))[:, None]  # [N_blocks,1,64]
        fused, _ = self.attention(query, tokens, tokens, need_weights=False)
        fused = fused.squeeze(1).reshape(*state.shape[:-1], -1)  # [...,64]
        return self.network(torch.cat((state, fused), dim=-1))  # [...,192]


class SequentialVelocityWorldModel(nn.Module):
    def __init__(self, image_size=96, patch_size=16, latent_dim=192, hidden_dim=512,
                 action_embedding_dim=64, horizon=25, readout_stride=5, integration_steps=2,
                 start_fraction=0.5, action_block=1, batchnorm_projector=False,
                 action_fusion="mlp", attention_heads=4):
        super().__init__()
        if not 1 <= readout_stride <= horizon or horizon % readout_stride or not 0 < start_fraction <= 1:
            raise ValueError("Choose a positive horizon divisible by readout_stride and a positive start fraction")
        if type(action_block) is not int or action_block < 1 or readout_stride % action_block:
            raise ValueError("action_block must be positive and divide readout_stride")
        self.encoder = FrameEncoder(image_size, patch_size, latent_dim)
        if batchnorm_projector:
            from models.encoder.common import LeWorldProjector
            # Last-layer ViT CLS -> Linear(192,2048) -> BN -> GELU -> Linear(2048,192).
            self.encoder.projection = LeWorldProjector(latent_dim, latent_dim)
        if action_fusion == "mlp":
            self.field = ControlledVelocityField(latent_dim, action_embedding_dim, hidden_dim, 2 * action_block)
        elif action_fusion == "cross_attention":
            self.field = CrossAttentionVelocityField(latent_dim, action_embedding_dim, hidden_dim, action_block,
                                                     attention_heads)
        else:
            raise ValueError(f"Unknown velocity action_fusion: {action_fusion}")
        self.integrator = EulerIntegrator(integration_steps)
        self.horizon, self.readout_stride, self.start_fraction = horizon, readout_stride, start_fraction
        self.rollout_stride = action_block  # Number of executed raw actions per predicted endpoint.

    def action_blocks(self, actions):
        # [B,H_raw,2] -> [B,H_obs,block*2]; preserve chronological order.
        assert actions.ndim == 3 and actions.shape[1:] == (self.horizon, 2)
        return actions.reshape(len(actions), self.horizon // self.rollout_stride, 2 * self.rollout_stride)

    def forward(self, frames, actions, *, generator=None):
        blocks = self.action_blocks(actions)  # [B,5,10] for 25 raw steps and blocks of 5.
        assert frames.ndim == 5 and len(frames) == len(actions)
        if frames.shape[1] == self.horizon + 1:
            frames = frames[:, ::self.rollout_stride]  # Raw inputs accepted for historical/fixture loaders.
        assert frames.shape[1] == blocks.shape[1] + 1
        real = self.encoder(frames.flatten(0, 1)).reshape(len(frames), frames.shape[1], -1)  # [B,6,192].
        source, target = real[:, :-1], real[:, 1:]  # [B,5,192].
        # Time is local to each action block, whose duration is one. Train explicitly
        # at the real starting state as well as inside each straight bridge.
        r = torch.rand((*blocks.shape[:2], 1), device=real.device, generator=generator)
        phase = ((r-self.start_fraction)/max(1-self.start_fraction, 1e-8)).clamp(0, 1)
        velocity_target = target-source
        velocity_prediction = self.field(source+phase*velocity_target, blocks)
        result = dict(start=real[:, 0], target=target, real_latents=real,
                      velocity_prediction=velocity_prediction, velocity_target=velocity_target)
        if not self.training:
            prediction, states, velocities = self._roll(real[:, 0], actions, collect_paths=True)
            result.update(prediction=prediction, states=states, velocities=velocities)
        return result

    def _roll(self, state, actions, *, collect_paths=False):
        assert state.ndim == 2 and len(state) == len(actions)
        blocks = self.action_blocks(actions)
        predictions, paths, velocities = [], [], []
        for action in blocks.unbind(1):
            if collect_paths:
                state, path, speed = self.integrator(self.field, state, action)
                paths.append(path)
                velocities.append(speed)
            else:
                for _ in range(self.integrator.steps):
                    state = state+self.field(state, action)/self.integrator.steps
            predictions.append(state)
        predicted = torch.stack(predictions, dim=1)
        if collect_paths:
            return predicted, torch.stack(paths, dim=1), torch.stack(velocities, dim=1)
        return predicted

    def rollout(self, state, actions):
        """Endpoints at raw offsets stride, 2*stride, ..., horizon."""
        return self._roll(state, actions)

    def predict_checkpoints(self, state, actions):
        """Public macro trajectory: [B,5,192] for horizon=25, stride=5."""
        readout = self.readout_stride // self.rollout_stride
        return self.rollout(state, actions)[:, readout-1::readout]
