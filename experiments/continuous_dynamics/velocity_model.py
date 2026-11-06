"""Direct finite-difference velocity supervision, with a causal raw-step rollout."""
import torch
from torch import nn

from .model import EulerIntegrator, FrameEncoder


class ControlledVelocityField(nn.Module):
    def __init__(self, latent_dim, action_embedding_dim, hidden_dim):
        super().__init__()
        self.action_encoder = nn.Sequential(nn.Linear(2, action_embedding_dim), nn.SiLU(),
                                            nn.Linear(action_embedding_dim, action_embedding_dim))
        self.network = nn.Sequential(nn.Linear(latent_dim + action_embedding_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, latent_dim))

    def forward(self, state, action):
        # All leading axes are independent examples (no action-prefix lookahead).
        assert state.shape[:-1] == action.shape[:-1] and action.shape[-1] == 2
        return self.network(torch.cat((state, self.action_encoder(action)), dim=-1))


class SequentialVelocityWorldModel(nn.Module):
    def __init__(self, image_size=96, patch_size=16, latent_dim=192, hidden_dim=512,
                 action_embedding_dim=64, horizon=25, readout_stride=5, integration_steps=2,
                 start_fraction=0.5):
        super().__init__()
        if not 1 <= readout_stride <= horizon or horizon % readout_stride or not 0 < start_fraction <= 1:
            raise ValueError("Choose a positive horizon divisible by readout_stride and a positive start fraction")
        self.encoder = FrameEncoder(image_size, patch_size, latent_dim)
        self.field = ControlledVelocityField(latent_dim, action_embedding_dim, hidden_dim)
        self.integrator = EulerIntegrator(integration_steps)
        self.horizon, self.readout_stride, self.start_fraction = horizon, readout_stride, start_fraction

    def forward(self, frames, actions, *, generator=None):
        assert actions.shape == (len(frames), self.horizon, 2)
        assert frames.ndim == 5 and frames.shape[1] == self.horizon + 1
        real = self.encoder(frames.flatten(0, 1)).reshape(len(frames), self.horizon + 1, -1)
        source, target = real[:, :-1], real[:, 1:]  # [B,H,192]; labels from the aligned raw frames
        # Time is local to each raw step, whose duration is one. Train explicitly
        # at the real starting state as well as inside each straight bridge.
        r = torch.rand((*actions.shape[:2], 1), device=real.device, generator=generator)
        phase = ((r-self.start_fraction)/max(1-self.start_fraction, 1e-8)).clamp(0, 1)
        velocity_target = target-source
        velocity_prediction = self.field(source+phase*velocity_target, actions)
        result = dict(start=real[:, 0], target=target, real_latents=real,
                      velocity_prediction=velocity_prediction, velocity_target=velocity_target)
        if not self.training:
            prediction, states, velocities = self._roll(real[:, 0], actions, collect_paths=True)
            result.update(prediction=prediction, states=states, velocities=velocities)
        return result

    def _roll(self, state, actions, *, collect_paths=False):
        assert state.ndim == 2 and actions.shape == (len(state), self.horizon, 2)
        predictions, paths, velocities = [], [], []
        for action in actions.unbind(1):
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
        """All raw endpoints for CEM's per-executed-action diagnostics."""
        return self._roll(state, actions)

    def predict_checkpoints(self, state, actions):
        """Public macro trajectory: [B,5,192] for horizon=25, stride=5."""
        return self.rollout(state, actions)[:, self.readout_stride-1::self.readout_stride]
