from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Prediction:
    agent: torch.Tensor
    environment: torch.Tensor
    delta: torch.Tensor | None = None


class AgentCentricWorldModel(nn.Module):
    """Composition root; individual components remain independently replaceable."""

    def __init__(self, agent_encoder: nn.Module, environment_encoder: nn.Module,
                 agent_transition: nn.Module, environment_transition: nn.Module,
                 predictor: nn.Module | None = None, predictor_type: str = "adaln",
                 history_size: int = 3):
        super().__init__()
        self.agent_encoder = agent_encoder
        self.environment_encoder = environment_encoder
        self.agent_transition = agent_transition
        self.environment_transition = environment_transition
        self.predictor = predictor
        self.predictor_type = predictor_type
        self.history_size = history_size
        self._last_history_actions: torch.Tensor | None = None

    def encode_frames(self, history_frames: torch.Tensor) -> torch.Tensor:
        """Encode frame windows without changing the Frame Encoder itself.

        Input shape:  history_frames [B, T, C, H, W]
        Output shape: frame_latents  [B, T, 192]
        """
        assert history_frames.ndim == 5, f"history_frames must be [B,T,C,H,W], got {tuple(history_frames.shape)}"
        batch, steps = history_frames.shape[:2]
        # flat_frames: [B*T, C, H, W]
        flat_frames = history_frames.flatten(0, 1)
        # flat_latents: [B*T, 192]
        flat_latents = self.environment_encoder(flat_frames)
        # frame_latents: [B, T, 192]
        return flat_latents.view(batch, steps, -1)

    def encode(self, history_frames: torch.Tensor, history_actions: torch.Tensor,
               current_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_history_actions = history_actions
        if self.predictor_type == "v3_n1":
            assert self.predictor is not None, "v3_n1 predictor is not configured"
            # frame_latents: [B,T,192], mu_current: [B,192].
            frame_latents = self.predictor.encode_mean_sequence(history_frames)
            assert frame_latents.shape[1] == self.history_size, (
                f"v3_n1 expects history_size={self.history_size}, got {frame_latents.shape[1]}"
            )
            return frame_latents, frame_latents[:, -1]
        if self.predictor_type in {"motion_token", "forward_inverse"}:
            frame_latents = self.encode_frames(history_frames)
            assert frame_latents.shape[1] == self.history_size, (
                f"{self.predictor_type} expects history_size={self.history_size}, got {frame_latents.shape[1]}"
            )
            # Return full frame history as the rollout state, plus current z_t.
            return frame_latents, frame_latents[:, -1]
        return self.agent_encoder(history_frames, history_actions), self.environment_encoder(current_frame)

    def step(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
             action: torch.Tensor, history_actions: torch.Tensor | None = None,
             action_is_normalized: bool = False) -> Prediction:
        if self.predictor_type == "v3_n1":
            assert self.predictor is not None, "v3_n1 predictor is not configured"
            # next_environment: [B,192] = mu_pred_next.
            history = agent_state if agent_state.ndim == 3 else environment_state
            next_environment = self.predictor.predict_next(history, action, history_actions, action_is_normalized)
            if agent_state.ndim == 3:
                # next_agent/history: [B,T,192] = [z_{t-T+2},...,z_t,z_pred].
                next_agent = torch.cat((agent_state[:, 1:], next_environment[:, None]), dim=1)
            else:
                # Backwards-compatible single-frame state.
                next_agent = next_environment
            return Prediction(next_agent, next_environment)
        if self.predictor_type == "motion_token":
            assert self.predictor is not None, "motion_token predictor is not configured"
            assert agent_state.ndim == 3, f"motion_token agent_state must be frame history [B,3,192], got {tuple(agent_state.shape)}"
            action_window = self._action_window(action, agent_state.shape[0], history_actions)
            # next_environment: [B, 192], delta: [B, 192]
            next_environment, delta = self.predictor(agent_state, action_window)
            # next_agent/history: [B, 3, 192] = [z_{t-1}, z_t, z_pred]
            next_agent = torch.cat((agent_state[:, 1:], next_environment[:, None]), dim=1)
            return Prediction(next_agent, next_environment, delta)
        if self.predictor_type == "forward_inverse":
            assert self.predictor is not None, "forward_inverse predictor is not configured"
            assert agent_state.ndim == 3, (
                f"forward_inverse agent_state must be frame history [B,3,192], got {tuple(agent_state.shape)}"
            )
            # next_environment: [B, 192] = z_pred_{t+1}; inverse head is intentionally unused here.
            next_environment = self.predictor(agent_state, action)
            # next_agent/history: [B, 3, 192] = [z_{t-1}, z_t, z_pred]
            next_agent = torch.cat((agent_state[:, 1:], next_environment[:, None]), dim=1)
            return Prediction(next_agent, next_environment)
        next_agent = self.agent_transition(agent_state, action)
        next_environment = self.environment_transition(environment_state, next_agent)
        return Prediction(next_agent, next_environment)

    def rollout(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
                actions: torch.Tensor, history_actions: torch.Tensor | None = None) -> list[Prediction]:
        if self.predictor_type in {"motion_token", "v3_n1"}:
            history_actions = self._initial_previous_actions(actions, history_actions)
        v3_actions_are_normalized = (
            self.predictor_type == "v3_n1"
            and actions.ndim == 3
            and self.predictor is not None
            and actions.shape[-1] == self.predictor.action_dim
        )
        predictions = []
        for action in actions.unbind(dim=1):
            if self.predictor_type == "motion_token":
                prediction = self.step(agent_state, environment_state, action, history_actions)
                # Keep only the two previous actions; step() appends the current action.
                # history_actions: [B, 2, A] = [a_{prev1}, a_current]
                history_actions = torch.cat((history_actions[:, 1:], action[:, None]), dim=1)
            elif self.predictor_type == "v3_n1":
                prediction = self.step(agent_state, environment_state, action, history_actions,
                                       action_is_normalized=v3_actions_are_normalized)
                # history_actions: [B, history_size-1, action_block*action_dim]
                assert self.predictor is not None
                action_norm = action if v3_actions_are_normalized else self.predictor.normalize_action(action)
                history_actions = torch.cat((history_actions[:, 1:], action_norm[:, None]), dim=1)
            else:
                prediction = self.step(agent_state, environment_state, action,
                                       action_is_normalized=(self.predictor_type == "v3_n1"))
            predictions.append(prediction)
            agent_state, environment_state = prediction.agent, prediction.environment
        return predictions

    def inverse_action(self, z_current: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        assert self.predictor_type == "forward_inverse", "inverse_action is only available for forward_inverse"
        assert self.predictor is not None, "forward_inverse predictor is not configured"
        return self.predictor.inverse(z_current, z_next)

    def encode_goal(self, goal_frame: torch.Tensor) -> torch.Tensor:
        if self.predictor_type == "v3_n1":
            assert self.predictor is not None
            return self.predictor.encode_mean(goal_frame)
        return self.environment_encoder(goal_frame)

    def denormalize_planner_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.predictor_type == "v3_n1":
            assert self.predictor is not None
            return self.predictor.denormalize_action(action)
        return action

    def set_action_stats(self, action_min: torch.Tensor, action_max: torch.Tensor) -> None:
        if self.predictor_type == "v3_n1":
            assert self.predictor is not None
            self.predictor.set_action_stats(action_min, action_max)

    def _action_window(self, action: torch.Tensor, batch: int,
                       history_actions: torch.Tensor | None) -> torch.Tensor:
        if history_actions is None:
            history_actions = self._initial_previous_actions(action[:, None], None)
        if history_actions.shape[1] == self.history_size - 1:
            # Training dataset provides [a_{t-2}, a_{t-1}] plus current action a_t.
            history_actions = torch.cat((history_actions, action[:, None]), dim=1)
        assert history_actions.ndim == 3, f"history_actions must be [B,3,A], got {tuple(history_actions.shape)}"
        assert history_actions.shape[0] == batch, (
            f"history_actions batch {history_actions.shape[0]} does not match state batch {batch}"
        )
        assert history_actions.shape[1] == self.history_size, (
            f"motion_token expects {self.history_size} actions, got {history_actions.shape[1]}"
        )
        return history_actions

    def _initial_previous_actions(self, actions: torch.Tensor,
                                  history_actions: torch.Tensor | None) -> torch.Tensor:
        if history_actions is None:
            history_actions = self._last_history_actions
        if history_actions is not None and history_actions.shape[1] == self.history_size:
            history_actions = history_actions[:, -self.history_size + 1:]
        if history_actions is None:
            action_dim = self.predictor.action_dim if self.predictor_type == "v3_n1" and self.predictor is not None else actions.shape[-1]
            zeros = torch.zeros(actions.shape[0], self.history_size - 1, action_dim,
                                device=actions.device, dtype=actions.dtype)
            return zeros
        if self.predictor_type == "v3_n1" and history_actions.ndim == 4:
            # Raw action block history [B,T-1,action_block,2] -> normalized [B,T-1,action_block*2].
            assert self.predictor is not None
            history_actions = self.predictor.normalize_action_sequence(history_actions)
        if history_actions.shape[0] != actions.shape[0]:
            assert actions.shape[0] % history_actions.shape[0] == 0, (
                "cannot broadcast encoded history actions to rollout population"
            )
            repeat = actions.shape[0] // history_actions.shape[0]
            history_actions = history_actions.repeat_interleave(repeat, dim=0)
        return history_actions.to(device=actions.device, dtype=actions.dtype)

    @staticmethod
    def planning_cost(final_environment: torch.Tensor, goal_environment: torch.Tensor) -> torch.Tensor:
        """Task cost intentionally excludes Agent State.

        Agent remains the causal mediator inside rollout, but its final position
        is never part of goal scoring.
        """
        return (final_environment - goal_environment).square().mean(dim=-1)
