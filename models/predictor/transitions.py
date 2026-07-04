import torch
from torch import nn


def _mlp(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))


class AgentTransition(nn.Module):
    """The only transition allowed to consume an action."""

    def __init__(self, agent_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.delta = _mlp(agent_dim + action_dim, agent_dim, hidden_dim)

    def forward(self, agent_state: torch.Tensor, current_action: torch.Tensor) -> torch.Tensor:
        return agent_state + self.delta(torch.cat((agent_state, current_action), dim=-1))


class EnvironmentTransition(nn.Module):
    """Updates environment through next agent state; it has no action input."""

    def __init__(self, environment_dim: int, agent_dim: int, hidden_dim: int):
        super().__init__()
        self.delta = _mlp(environment_dim + agent_dim, environment_dim, hidden_dim)

    def forward(self, environment_state: torch.Tensor, next_agent_state: torch.Tensor) -> torch.Tensor:
        return environment_state + self.delta(torch.cat((environment_state, next_agent_state), dim=-1))

