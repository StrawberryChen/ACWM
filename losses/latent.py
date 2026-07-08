import torch
from torch import nn


class AgentPredictionLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LeWM-style end-to-end JEPA: gradients flow through predictor and target encoder.
        return nn.functional.mse_loss(prediction, target)


class EnvironmentPredictionLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(prediction, target)


class GoalConsistencyLoss(nn.Module):
    """Interface for matching a final rollout state to an encoded goal frame."""

    def forward(self, final_environment: torch.Tensor, goal_environment: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(final_environment, goal_environment.detach())


class SIGRegLoss(nn.Module):
    """Sketched Isotropic Gaussian Regularizer used by LeWorldModel.

    Random unit projections are compared with a standard normal distribution
    using the Epps-Pulley characteristic-function statistic. Input is [T,B,D],
    so regularization is applied independently at each trajectory step.
    """

    def __init__(self, knots: int = 17, num_projections: int = 1024):
        super().__init__()
        if knots < 2 or num_projections < 1:
            raise ValueError("SIGReg requires knots >= 2 and num_projections >= 1")
        self.num_projections = num_projections
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        trapezoid = torch.full((knots,), 2 * dt, dtype=torch.float32)
        trapezoid[[0, -1]] = dt
        gaussian_cf = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("gaussian_cf", gaussian_cf)
        self.register_buffer("weights", trapezoid * gaussian_cf)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("SIGReg embeddings must have shape [T,B,D]")
        # Characteristic-function statistics are sensitive to fp16 rounding.
        embeddings = embeddings.float()
        directions = torch.randn(
            embeddings.size(-1), self.num_projections,
            device=embeddings.device, dtype=torch.float32,
        )
        directions = torch.nn.functional.normalize(directions, dim=0)
        x_t = (embeddings @ directions).unsqueeze(-1) * self.t
        error = ((x_t.cos().mean(dim=1) - self.gaussian_cf).square()
                 + x_t.sin().mean(dim=1).square())
        statistic = (error @ self.weights) * embeddings.size(1)
        return statistic.mean()
