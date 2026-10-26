"""One CSE, one conditional Transformer, separate motion/terminal velocity heads."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import CurrentStateEncoder, sinusoidal


class JointFlowPredictor(nn.Module):
    def __init__(self, terminal_dim, motion_dim=None, latent_dim=128, horizon=16,
                 model_dim=256, layers=4, heads=8, dropout=0.0):
        super().__init__()
        assert model_dim % heads == 0 and model_dim % 2 == 0
        self.terminal_dim, self.motion_dim = terminal_dim, motion_dim
        self.horizon, self.model_dim = horizon, model_dim
        self.action_encoder = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 128))
        self.action_projection = nn.Linear(128, model_dim)
        self.state_projection = nn.Linear(latent_dim, model_dim)
        self.time_mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.terminal_projection = nn.Linear(terminal_dim, model_dim)
        self.motion_projection = nn.Linear(motion_dim, model_dim) if motion_dim is not None else None
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, heads, 4*model_dim, dropout, activation="gelu",
                                       batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        for layer in self.transformer.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.motion_velocity_head = nn.Linear(model_dim, motion_dim) if motion_dim is not None else None
        self.terminal_velocity_head = nn.Linear(model_dim, terminal_dim)

    def forward(self, noisy, time, state, actions):
        # Noisy motion [B,H,Dm] (optional), terminal [B,Dg]; shared s [B].
        batch, horizon, action_dim = actions.shape
        assert action_dim == 2 and 1 <= horizon <= self.horizon and time.shape == (batch,)
        assert noisy["terminal"].shape == (batch, self.terminal_dim)
        assert ("motion" in noisy) == (self.motion_dim is not None)
        time_embedding = self.time_mlp(sinusoidal(time*1000, self.model_dim))
        positions = sinusoidal(torch.arange(horizon+1, device=state.device).float(), self.model_dim)
        future = self.action_projection(self.action_encoder(actions))+positions[None, :horizon]+time_embedding[:, None]
        if self.motion_projection is not None:
            assert noisy["motion"].shape == (batch, horizon, self.motion_dim)
            future = future+self.motion_projection(noisy["motion"])
        terminal = self.terminal_projection(noisy["terminal"])+positions[horizon]+time_embedding
        tokens = torch.cat((self.state_projection(state)[:, None], future, terminal[:, None]), 1)
        encoded = self.transformer(tokens)  # [B,H+2,256], no causal mask
        result = {"terminal": self.terminal_velocity_head(encoded[:, -1])}
        if self.motion_velocity_head is not None:
            result["motion"] = self.motion_velocity_head(encoded[:, 1:1+horizon])
        return result


class JointFlowWorldModel(nn.Module):
    def __init__(self, terminal_dim, motion_dim=None, latent_dim=128, **settings):
        super().__init__()
        self.cse = CurrentStateEncoder(latent_dim)
        self.predictor = JointFlowPredictor(terminal_dim, motion_dim, latent_dim, **settings)
        self.terminal_dim, self.motion_dim = terminal_dim, motion_dim
        self.horizon = self.predictor.horizon


def joint_training_pair(targets, noise, time):
    assert set(targets) == set(noise)
    noisy, velocity = {}, {}
    for name, target in targets.items():
        assert not target.requires_grad and target.shape == noise[name].shape
        s = time.reshape(len(time), *([1]*(target.ndim-1)))
        noisy[name] = (1-s)*noise[name]+s*target
        velocity[name] = target-noise[name]
    return noisy, velocity


def joint_flow_loss(predicted, target_velocity, motion_weight=1.0, terminal_weight=1.0):
    assert set(predicted) == set(target_velocity)
    if motion_weight < 0 or terminal_weight < 0 or terminal_weight == 0:
        raise ValueError("nonnegative motion weight and positive terminal weight required")
    for name in predicted:
        assert predicted[name].shape == target_velocity[name].shape and not target_velocity[name].requires_grad
    # Each branch reduced to a MEAN before weighting, regardless of dimensions.
    terminal = F.mse_loss(predicted["terminal"], target_velocity["terminal"])
    motion = F.mse_loss(predicted["motion"], target_velocity["motion"]) if "motion" in predicted else terminal.new_zeros(())
    return {"loss": motion_weight*motion+terminal_weight*terminal,
            "motion_flow_loss": motion, "terminal_flow_loss": terminal}


@torch.no_grad()
def joint_euler_sample(predictor, state, actions, noise, steps=16):
    if steps < 1:
        raise ValueError("positive Euler steps required")
    values = {name: value.clone() for name, value in noise.items()}
    for k in range(steps):
        time = state.new_full((len(state),), k/steps)
        velocities = predictor(values, time, state, actions)
        values = {name: value+velocities[name]/steps for name, value in values.items()}
    return values
