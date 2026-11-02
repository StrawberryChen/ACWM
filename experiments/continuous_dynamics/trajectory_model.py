"""Full open-loop trajectory rollout with an explicitly TRAINABLE direction head."""
import torch
from torch import nn
from .model import FrameEncoder, EulerIntegrator


class ActionTrajectoryEncoder(nn.Module):
    def __init__(self, horizon=16, latent_dim=192, layers=2, heads=3):
        super().__init__()
        self.horizon = horizon
        self.action_encoder = nn.Sequential(nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, latent_dim))
        self.positions = nn.Parameter(torch.empty(1, horizon, latent_dim))
        nn.init.normal_(self.positions, std=.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(latent_dim, heads, 4*latent_dim, dropout=0.,
                                       activation="gelu", batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        for layer in self.transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def forward(self, actions):
        assert actions.ndim == 3 and actions.shape[-1] == 2
        assert 1 <= actions.shape[1] <= self.horizon
        embedded = self.action_encoder(actions)  # [B,H,192], same action MLP at all h
        tokens = embedded+self.positions[:, :actions.shape[1]]
        return self.transformer(tokens).mean(1), embedded  # context [B,192], actions [B,H,192]


class TrajectoryVelocityField(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4*latent_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, latent_dim))

    def forward(self, state, action_embedding, context, direction):
        assert state.ndim == 2 and state.shape == action_embedding.shape == context.shape == direction.shape
        # Four [B,192] conditions -> [B,768] -> V [B,192]. No clock/phase input.
        return self.network(torch.cat((state, action_embedding, context, direction), -1))


class TrajectoryWorldModel(nn.Module):
    def __init__(self, image_size=96, patch_size=16, latent_dim=192, hidden_dim=512,
                 integration_steps=4, horizon=16, direction_mode="action_only",
                 action_layers=2, action_heads=3, direction_eps=1e-8):
        super().__init__()
        if direction_mode not in {"action_only", "state_action"}:
            raise ValueError("direction_mode must be action_only or state_action")
        if horizon < 1 or action_layers < 1 or direction_eps <= 0:
            raise ValueError("positive horizon, action_layers and direction_eps required")
        # Common modules initialized before the variant head for matched initial encoders.
        self.encoder = FrameEncoder(image_size, patch_size, latent_dim)
        self.field = TrajectoryVelocityField(latent_dim, hidden_dim)
        self.action_trajectory = ActionTrajectoryEncoder(horizon, latent_dim, action_layers, action_heads)
        self.direction_head = nn.Sequential(
            nn.Linear(latent_dim if direction_mode == "action_only" else 2*latent_dim, latent_dim),
            nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.integrator = EulerIntegrator(integration_steps)
        self.horizon, self.direction_mode, self.direction_eps = horizon, direction_mode, direction_eps

    def condition(self, start, actions):
        context, embeddings = self.action_trajectory(actions)
        inputs = context if self.direction_mode == "action_only" else torch.cat((start, context), -1)
        raw = self.direction_head(inputs)  # [B,192]; trainable, NOT detached
        direction = raw/(raw.norm(dim=-1, keepdim=True)+self.direction_eps)
        return context, embeddings, raw, direction

    def predict_trajectory(self, start, actions):
        assert start.ndim == 2 and actions.shape[0] == len(start)
        context, embeddings, raw, direction = self.condition(start, actions)
        predictions, paths, velocities = [], [], []
        state = start
        for action_embedding in embeddings.unbind(1):
            # Freeze the arrow value over this WHOLE horizon, but do not sever its
            # autograd path through the velocity field. Only the loss reference detaches.
            def conditioned_field(z, a):
                return self.field(z, a, context, direction)
            state, path, velocity = self.integrator(conditioned_field, state, action_embedding)
            predictions.append(state)
            paths.append(path)
            velocities.append(velocity)
        return dict(prediction=torch.stack(predictions, 1), states=torch.stack(paths, 1),
                    velocities=torch.stack(velocities, 1), direction=direction, direction_raw=raw)
        # prediction [B,H,192], states [B,H,K+1,192], velocities [B,H,K,192]

    def forward(self, frames, actions):
        assert frames.ndim == 5 and actions.ndim == 3
        assert frames.shape[:2] == (len(actions), actions.shape[1]+1)
        encoded = self.encoder(frames.flatten(0, 1)).reshape(len(frames), frames.shape[1], -1)
        # True future latents are LABELS ONLY; not passed to condition/rollout.
        start, targets = encoded[:, 0], encoded[:, 1:]
        return {"start": start, "target": targets, **self.predict_trajectory(start, actions)}

    def rollout(self, start, actions):
        return self.predict_trajectory(start, actions)["prediction"]
