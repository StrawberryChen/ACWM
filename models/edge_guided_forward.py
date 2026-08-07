from __future__ import annotations

import torch
from torch import nn

from models.acwm import Prediction
from models.observation_graph import ObservationGraphBridgeModel


class EdgeGuidedTemporalPredictor(nn.Module):
    """Action-conditioned residual predictor in a frozen obs-graph latent space.

    Input shape:  frame_latents [B,T,192]
                  action_seq    [B,T,10]
    Output shape: z_pred_next   [B,T,192]
    """

    def __init__(self, latent_dim: int = 192, action_dim: int = 10, history_size: int = 3,
                 hidden_dim: int = 512, num_layers: int = 2, num_heads: int = 3, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_size = history_size
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim),
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_size, latent_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, frame_latents: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        assert frame_latents.ndim == 3, (
            f"frame_latents must be [B,T,{self.latent_dim}], got {tuple(frame_latents.shape)}"
        )
        assert action_seq.ndim == 3 and action_seq.shape[:2] == frame_latents.shape[:2], (
            f"action_seq must align with frame_latents, got {tuple(action_seq.shape)} and {tuple(frame_latents.shape)}"
        )
        assert action_seq.shape[-1] == self.action_dim, (
            f"action_seq last dim must be {self.action_dim}, got {action_seq.shape[-1]}"
        )
        steps = frame_latents.shape[1]
        assert steps <= self.history_size, f"steps {steps} exceeds history_size={self.history_size}"
        # action_latents: [B,T,192]
        action_latents = self.action_encoder(action_seq.flatten(0, 1)).view(
            action_seq.shape[0], action_seq.shape[1], self.latent_dim
        )
        # tokens: [B,T,192]
        tokens = frame_latents + action_latents + self.temporal_pos[:, -steps:]
        # causal_mask: [T,T]
        causal_mask = torch.triu(torch.ones(steps, steps, device=frame_latents.device, dtype=torch.bool), diagonal=1)
        # h: [B,T,192]
        h = self.temporal(tokens, mask=causal_mask)
        # delta_input: [B,T,576]
        delta_input = torch.cat((h, action_latents, h * action_latents), dim=-1)
        # delta: [B,T,192]
        delta = self.delta_head(delta_input.flatten(0, 1)).view(
            frame_latents.shape[0], frame_latents.shape[1], self.latent_dim
        )
        return frame_latents + delta


class EdgeGuidedForwardModel(nn.Module):
    """Forward world model trained with frozen observation-graph edge supervision."""

    predictor_type = "edge_guided_forward"

    def __init__(self, obs_model: ObservationGraphBridgeModel, latent_dim: int = 192,
                 raw_action_dim: int = 2, action_block: int = 5, history_size: int = 3,
                 hidden_dim: int = 512, temporal_layers: int = 2, temporal_heads: int = 3,
                 temporal_dropout: float = 0.0, planning_z_weight: float = 1.0,
                 planning_edge_weight: float = 1.0, planning_delta_t: float = 5.0):
        super().__init__()
        self.obs_model = obs_model
        for parameter in self.obs_model.parameters():
            parameter.requires_grad_(False)
        self.obs_model.eval()
        self.latent_dim = latent_dim
        self.raw_action_dim = raw_action_dim
        self.action_block = action_block
        self.action_dim = raw_action_dim * action_block
        self.history_size = history_size
        self.forward_predictor = EdgeGuidedTemporalPredictor(
            latent_dim=latent_dim,
            action_dim=self.action_dim,
            history_size=history_size,
            hidden_dim=hidden_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=temporal_dropout,
        )
        self.planning_z_weight = planning_z_weight
        self.planning_edge_weight = planning_edge_weight
        self.planning_delta_t = planning_delta_t
        self.register_buffer("action_mean", torch.zeros(raw_action_dim), persistent=True)
        self.register_buffer("action_std", torch.ones(raw_action_dim), persistent=True)
        self._planning_start: torch.Tensor | None = None

    @property
    def predictor(self):
        # PlanningEvaluator expects model.predictor.normalize_action_sequence for
        # LeWorld-style action-block models. Keep the property non-Module to avoid
        # registering self as a child module.
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.obs_model.eval()
        return self

    def set_action_stats(self, action_mean: torch.Tensor, action_std: torch.Tensor) -> None:
        action_mean = action_mean.flatten()
        action_std = action_std.flatten().clamp_min(1e-6)
        assert action_mean.shape == self.action_mean.shape
        assert action_std.shape == self.action_std.shape
        self.action_mean.copy_(action_mean.to(self.action_mean))
        self.action_std.copy_(action_std.to(self.action_std))

    def _as_action_block(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 2 and action.shape[-1] == self.action_dim:
            return action.view(action.shape[0], self.action_block, self.raw_action_dim)
        assert action.ndim == 3 and tuple(action.shape[1:]) == (self.action_block, self.raw_action_dim), (
            f"action must be [B,{self.action_dim}] or [B,{self.action_block},{self.raw_action_dim}], "
            f"got {tuple(action.shape)}"
        )
        return action

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        # action: [B,5,2] raw -> [B,10] normalized.
        action = self._as_action_block(action)
        return ((action - self.action_mean) / self.action_std).flatten(1)

    def normalize_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: [B,T,5,2] raw -> [B,T,10] normalized.
        if actions.ndim == 3 and actions.shape[-1] == self.action_dim:
            return actions
        assert actions.ndim == 4, (
            f"actions must be [B,T,{self.action_block},{self.raw_action_dim}], got {tuple(actions.shape)}"
        )
        batch, steps = actions.shape[:2]
        return self.normalize_action(actions.flatten(0, 1)).view(batch, steps, self.action_dim)

    def denormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        # action_norm: [B,10] normalized -> [B,5,2] raw.
        action_norm = self._as_action_block(action_norm)
        return action_norm * self.action_std + self.action_mean

    def denormalize_planner_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.denormalize_action(action)

    def action_sequence(self, history_actions: torch.Tensor | None, current_action: torch.Tensor,
                        action_is_normalized: bool = False) -> torch.Tensor:
        current = current_action if action_is_normalized and current_action.ndim == 2 else (
            current_action.flatten(1) if action_is_normalized else self.normalize_action(current_action)
        )
        if history_actions is None or history_actions.numel() == 0:
            previous = current[:, None].expand(-1, self.history_size - 1, -1)
        else:
            previous = self.normalize_action_sequence(history_actions)
        assert previous.ndim == 3 and previous.shape[-1] == self.action_dim
        seq = torch.cat((previous, current[:, None]), dim=1)
        assert seq.shape[1:] == (self.history_size, self.action_dim), (
            f"action sequence must be [B,{self.history_size},{self.action_dim}], got {tuple(seq.shape)}"
        )
        return seq

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.obs_model.encode_sequence(frames)

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.obs_model.encode(frame)

    def edge(self, z_i: torch.Tensor, z_j: torch.Tensor, delta_t: float | torch.Tensor):
        return self.obs_model.edge(z_i, z_j, delta_t)

    def encode(self, history_frames: torch.Tensor, history_actions: torch.Tensor,
               current_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.encode_sequence(history_frames)
        current = history[:, -1]
        self._planning_start = current.detach()
        return history, current

    def encode_goal(self, goal_frame: torch.Tensor) -> torch.Tensor:
        return self.encode_frame(goal_frame)

    def predict_next(self, frame_latents: torch.Tensor, action: torch.Tensor,
                     history_actions: torch.Tensor | None = None,
                     action_is_normalized: bool = False) -> torch.Tensor:
        if frame_latents.ndim == 2:
            frame_latents = frame_latents[:, None]
        action_seq = self.action_sequence(history_actions, action, action_is_normalized)
        if action_seq.shape[1] != frame_latents.shape[1]:
            action_seq = action_seq[:, -frame_latents.shape[1]:]
        return self.forward_predictor(frame_latents, action_seq)[:, -1]

    def predict_next_sequence(self, frame_latents: torch.Tensor, history_actions: torch.Tensor,
                              current_action: torch.Tensor,
                              action_is_normalized: bool = False) -> torch.Tensor:
        action_seq = self.action_sequence(history_actions, current_action, action_is_normalized)
        return self.forward_predictor(frame_latents, action_seq)

    def step(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
             action: torch.Tensor, history_actions: torch.Tensor | None = None,
             action_is_normalized: bool = False) -> Prediction:
        next_environment = self.predict_next(agent_state, action, history_actions, action_is_normalized)
        next_agent = torch.cat((agent_state[:, 1:], next_environment[:, None]), dim=1)
        return Prediction(next_agent, next_environment, next_environment - agent_state[:, -1])

    def rollout(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
                actions: torch.Tensor, history_actions: torch.Tensor | None = None) -> list[Prediction]:
        if history_actions is None:
            history_actions = torch.zeros(actions.shape[0], self.history_size - 1, self.action_dim,
                                          device=actions.device, dtype=actions.dtype)
        elif history_actions.ndim == 4:
            history_actions = self.normalize_action_sequence(history_actions)
        if history_actions.shape[0] != actions.shape[0]:
            assert actions.shape[0] % history_actions.shape[0] == 0
            history_actions = history_actions.repeat_interleave(actions.shape[0] // history_actions.shape[0], dim=0)
        actions_are_normalized = actions.ndim == 3 and actions.shape[-1] == self.action_dim
        predictions = []
        for action in actions.unbind(dim=1):
            prediction = self.step(agent_state, environment_state, action, history_actions, actions_are_normalized)
            action_norm = action if actions_are_normalized else self.normalize_action(action)
            history_actions = torch.cat((history_actions[:, 1:], action_norm[:, None]), dim=1)
            predictions.append(prediction)
            agent_state, environment_state = prediction.agent, prediction.environment
        return predictions

    def planning_cost(self, final_environment: torch.Tensor, goal_environment: torch.Tensor) -> torch.Tensor:
        z_cost = (final_environment - goal_environment).square().mean(dim=-1)
        if self.planning_edge_weight <= 0 or self._planning_start is None:
            return z_cost
        if final_environment.ndim == 3:
            batch, population, dim = final_environment.shape
            start = self._planning_start[:, None].expand(batch, population, dim).flatten(0, 1)
            final = final_environment.flatten(0, 1)
            goal = goal_environment.expand(batch, population, dim).flatten(0, 1)
            pred_edge = self.edge(start, final, self.planning_delta_t)
            goal_edge = self.edge(start, goal, self.planning_delta_t)
            edge_cost = (pred_edge - goal_edge).square().mean(dim=-1).view(batch, population)
        else:
            start = self._planning_start
            pred_edge = self.edge(start, final_environment, self.planning_delta_t)
            goal_edge = self.edge(start, goal_environment, self.planning_delta_t)
            edge_cost = (pred_edge - goal_edge).square().mean(dim=-1)
        return self.planning_z_weight * z_cost + self.planning_edge_weight * edge_cost
