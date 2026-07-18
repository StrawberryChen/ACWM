from .latent import AgentPredictionLoss, EnvironmentPredictionLoss, GoalConsistencyLoss, SIGRegLoss
from .moment_sigreg import MomentSIGRegLoss

__all__ = ["AgentPredictionLoss", "EnvironmentPredictionLoss", "GoalConsistencyLoss", "SIGRegLoss",
           "MomentSIGRegLoss"]
