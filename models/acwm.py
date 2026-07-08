from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Prediction:
    agent: torch.Tensor
    environment: torch.Tensor


class AgentCentricWorldModel(nn.Module):
    """Composition root; individual components remain independently replaceable."""

    def __init__(self, agent_encoder: nn.Module, environment_encoder: nn.Module,
                 agent_transition: nn.Module, environment_transition: nn.Module):
        super().__init__()
        self.agent_encoder = agent_encoder
        self.environment_encoder = environment_encoder
        self.agent_transition = agent_transition
        self.environment_transition = environment_transition

    def encode(self, history_frames: torch.Tensor, history_actions: torch.Tensor,
               current_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.agent_encoder(history_frames, history_actions), self.environment_encoder(current_frame)

    def step(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
             action: torch.Tensor) -> Prediction:
        next_agent = self.agent_transition(agent_state, action)
        next_environment = self.environment_transition(environment_state, next_agent)
        return Prediction(next_agent, next_environment)

    def rollout(self, agent_state: torch.Tensor, environment_state: torch.Tensor,
                actions: torch.Tensor) -> list[Prediction]:
        predictions = []
        for action in actions.unbind(dim=1):
            prediction = self.step(agent_state, environment_state, action)
            predictions.append(prediction)
            agent_state, environment_state = prediction.agent, prediction.environment
        return predictions

    @staticmethod
    def planning_cost(final_environment: torch.Tensor, goal_environment: torch.Tensor) -> torch.Tensor:
        """Task cost intentionally excludes Agent State.

        Agent remains the causal mediator inside rollout, but its final position
        is never part of goal scoring.
        """
        return (final_environment - goal_environment).square().mean(dim=-1)
