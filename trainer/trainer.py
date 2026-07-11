from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from losses import AgentPredictionLoss, EnvironmentPredictionLoss, GoalConsistencyLoss, SIGRegLoss


class ACWMTrainer:
    """Supports one-step batches and optional latent multi-step rollout batches."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_weights: dict[str, float], device: str | torch.device = "cpu",
                 sigreg_config: dict | None = None, amp: bool = False,
                 gradient_clip_norm: float | None = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.weights = loss_weights
        self.agent_loss = AgentPredictionLoss()
        self.environment_loss = EnvironmentPredictionLoss()
        self.goal_loss = GoalConsistencyLoss()
        self.sigreg_loss = SIGRegLoss(**(sigreg_config or {})).to(device)
        self.amp_enabled = bool(amp and self.device.type == "cuda")
        self.gradient_clip_norm = gradient_clip_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def compute_one_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._move(batch)
        agent, environment = self.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        target_environment = self.model.environment_encoder(batch["next_frame"])
        if getattr(self.model, "predictor_type", "adaln") == "motion_token":
            # history_actions: [B, 2, A], current_action: [B, A]
            # action_window:    [B, 3, A] = [a_{t-2}, a_{t-1}, a_t]
            action_window = torch.cat((batch["history_actions"], batch["current_action"][:, None]), dim=1)
            assert action_window.shape[1] == 3, f"motion_token expects [B,3,A], got {tuple(action_window.shape)}"
            prediction = self.model.step(agent, environment, batch["current_action"], action_window)
            # prediction.environment: [B, 192] = z_pred_next
            prediction_loss = self.environment_loss(prediction.environment, target_environment)
            # delta_target: [B, 192] = z_{t+1} - z_t
            delta_target = target_environment - environment
            assert prediction.delta is not None, "motion_token predictor must return delta_z"
            flow_loss = nn.functional.mse_loss(prediction.delta, delta_target)
            # sigreg embeddings: [T=2, B, 192]
            embeddings = torch.stack((environment, target_environment), dim=0)
            sigreg = self.sigreg_loss(embeddings)
            total = (self.weights.get("prediction", self.weights.get("environment", 1.0)) * prediction_loss
                     + self.weights.get("flow", 1.0) * flow_loss
                     + self.weights.get("sigreg", self.weights.get("environment_sigreg", 0.1)) * sigreg)
            return {
                "loss": total,
                "prediction_loss": prediction_loss,
                "flow_loss": flow_loss,
                "sigreg_loss": sigreg,
                "latent_std": embeddings.std(dim=(0, 1)).mean(),
                "environment_loss": prediction_loss,
                "environment_sigreg_loss": sigreg,
            }

        prediction = self.model.step(agent, environment, batch["current_action"])
        target_agent = self.model.agent_encoder(batch["next_history_frames"], batch["next_history_actions"])
        agent_loss = self.agent_loss(prediction.agent, target_agent)
        environment_loss = self.environment_loss(prediction.environment, target_environment)
        agent_embeddings = torch.stack((agent, target_agent), dim=0)
        environment_embeddings = torch.stack((environment, target_environment), dim=0)
        agent_sigreg = self.sigreg_loss(agent_embeddings)
        environment_sigreg = self.sigreg_loss(environment_embeddings)
        total = (self.weights.get("agent", 1.0) * agent_loss
                 + self.weights.get("environment", 1.0) * environment_loss
                 + self.weights.get("agent_sigreg", 0.1) * agent_sigreg
                 + self.weights.get("environment_sigreg", 0.1) * environment_sigreg)
        return {
            "loss": total,
            "agent_loss": agent_loss,
            "environment_loss": environment_loss,
            "agent_sigreg_loss": agent_sigreg,
            "environment_sigreg_loss": environment_sigreg,
            "agent_latent_std": agent_embeddings.std(dim=(0, 1)).mean(),
            "environment_latent_std": environment_embeddings.std(dim=(0, 1)).mean(),
        }

    def compute_rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expected extra fields: rollout_actions [B,L,A], goal_frame [B,C,H,W]."""
        batch = self._move(batch)
        agent, environment = self.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        predictions = self.model.rollout(agent, environment, batch["rollout_actions"], batch["history_actions"])
        goal = self.model.environment_encoder(batch["goal_frame"])
        loss = self.goal_loss(predictions[-1].environment, goal)
        return {"loss": self.weights.get("goal", 1.0) * loss, "goal_loss": loss}

    def train_step(self, batch: dict[str, torch.Tensor], mode: str = "one_step") -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if mode not in {"one_step", "rollout"}:
            raise ValueError("mode must be 'one_step' or 'rollout'")
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled):
            losses = self.compute_one_step(batch) if mode == "one_step" else self.compute_rollout(batch)
        self.scaler.scale(losses["loss"]).backward()
        if self.gradient_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
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
