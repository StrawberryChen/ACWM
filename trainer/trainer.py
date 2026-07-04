from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from losses import AgentPredictionLoss, EnvironmentPredictionLoss, GoalConsistencyLoss


class ACWMTrainer:
    """Supports one-step batches and optional latent multi-step rollout batches."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_weights: dict[str, float], device: str | torch.device = "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.weights = loss_weights
        self.agent_loss = AgentPredictionLoss()
        self.environment_loss = EnvironmentPredictionLoss()
        self.goal_loss = GoalConsistencyLoss()

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def compute_one_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._move(batch)
        agent, environment = self.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        prediction = self.model.step(agent, environment, batch["current_action"])
        target_agent = self.model.agent_encoder(batch["next_history_frames"], batch["next_history_actions"])
        target_environment = self.model.environment_encoder(batch["next_frame"])
        agent_loss = self.agent_loss(prediction.agent, target_agent)
        environment_loss = self.environment_loss(prediction.environment, target_environment)
        total = self.weights.get("agent", 1.0) * agent_loss + self.weights.get("environment", 1.0) * environment_loss
        return {"loss": total, "agent_loss": agent_loss, "environment_loss": environment_loss}

    def compute_rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expected extra fields: rollout_actions [B,L,A], goal_frame [B,C,H,W]."""
        batch = self._move(batch)
        agent, environment = self.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        predictions = self.model.rollout(agent, environment, batch["rollout_actions"])
        goal = self.model.environment_encoder(batch["goal_frame"])
        loss = self.goal_loss(predictions[-1].environment, goal)
        return {"loss": self.weights.get("goal", 1.0) * loss, "goal_loss": loss}

    def train_step(self, batch: dict[str, torch.Tensor], mode: str = "one_step") -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if mode not in {"one_step", "rollout"}:
            raise ValueError("mode must be 'one_step' or 'rollout'")
        losses = self.compute_one_step(batch) if mode == "one_step" else self.compute_rollout(batch)
        losses["loss"].backward()
        self.optimizer.step()
        return {key: value.detach().item() for key, value in losses.items()}

    def fit_epoch(self, loader: Iterable[dict[str, torch.Tensor]], mode: str = "one_step",
                  description: str = "Training") -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        count = 0
        progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=True)
        for batch in progress:
            count += 1
            step_metrics = self.train_step(batch, mode)
            for key, value in step_metrics.items():
                totals[key] += value
            progress.set_postfix(loss=f"{step_metrics['loss']:.5f}")
        return {key: value / max(count, 1) for key, value in totals.items()}
