import torch
from torch import nn


class AgentPredictionLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(prediction, target.detach())


class EnvironmentPredictionLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(prediction, target.detach())


class GoalConsistencyLoss(nn.Module):
    """Interface for matching a final rollout state to an encoded goal frame."""

    def forward(self, final_environment: torch.Tensor, goal_environment: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(final_environment, goal_environment.detach())

