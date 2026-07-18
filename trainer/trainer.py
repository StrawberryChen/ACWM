from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import nn
from tqdm.auto import tqdm

from losses import AgentPredictionLoss, EnvironmentPredictionLoss, GoalConsistencyLoss, MomentSIGRegLoss, SIGRegLoss


class ACWMTrainer:
    """Supports one-step batches and optional latent multi-step rollout batches."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 loss_weights: dict[str, float], device: str | torch.device = "cpu",
                 sigreg_config: dict | None = None, amp: bool = False,
                 gradient_clip_norm: float | None = None, scheduler=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.weights = loss_weights
        self.agent_loss = AgentPredictionLoss()
        self.environment_loss = EnvironmentPredictionLoss()
        self.goal_loss = GoalConsistencyLoss()
        self.sigreg_loss = SIGRegLoss(**(sigreg_config or {})).to(device)
        self.moment_sigreg_loss = MomentSIGRegLoss().to(device)
        self.amp_enabled = bool(amp and self.device.type == "cuda")
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def compute_one_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = self._move(batch)
        if getattr(self.model, "predictor_type", "adaln") == "v3_n1":
            # o_t: [B,3,H,W], a_t raw: [B,2], o_next: [B,3,H,W]
            current = self.model.predictor.encode_gaussian(batch["current_frame"])
            target = self.model.predictor.encode_gaussian(batch["next_frame"])
            # mu_pred_next: [B,192]
            prediction = self.model.step(current.mu, current.mu, batch["current_action"])
            pred_loss = self.environment_loss(prediction.environment, target.mu)
            kl_current = self.model.predictor.kl_loss(current)
            kl_next = self.model.predictor.kl_loss(target)
            kl_loss = 0.5 * (kl_current + kl_next)
            # sig_input: [2B,192]
            sig_input = torch.cat((current.mu, target.mu), dim=0)
            sig_loss, sig_metrics = self.moment_sigreg_loss(sig_input)
            total = (self.weights.get("prediction", 1.0) * pred_loss
                     + self.weights.get("beta_kl", self.weights.get("kl", 1e-4)) * kl_loss
                     + self.weights.get("lambda_sig", self.weights.get("sigreg", 0.05)) * sig_loss)
            permutation = torch.randperm(batch["current_action"].shape[0], device=self.device)
            shuffled = self.model.step(current.mu, current.mu, batch["current_action"][permutation]).environment
            shuffle_mse = self.environment_loss(shuffled, target.mu)
            normal_mse = pred_loss
            metrics = {
                "loss": total,
                "loss_total": total,
                "loss_pred": pred_loss,
                "loss_kl": kl_loss,
                "loss_kl_current": kl_current,
                "loss_kl_next": kl_next,
                "loss_sig_total": sig_loss,
                "mu_current_mean_abs": current.mu.abs().mean(),
                "mu_next_mean_abs": target.mu.abs().mean(),
                "mu_current_std_mean": current.mu.std(dim=0).mean(),
                "mu_next_std_mean": target.mu.std(dim=0).mean(),
                "mu_combined_std_mean": sig_input.std(dim=0).mean(),
                "mu_pred_std_mean": prediction.environment.std(dim=0).mean(),
                "logvar_current_mean": current.logvar.mean(),
                "logvar_next_mean": target.logvar.mean(),
                "variance_current_mean": current.logvar.exp().mean(),
                "variance_next_mean": target.logvar.exp().mean(),
                "action_shuffle_mse_normal": normal_mse,
                "action_shuffle_mse_shuffle": shuffle_mse,
                "action_sensitivity": shuffle_mse - normal_mse,
            }
            metrics.update(sig_metrics)
            return metrics

        agent, environment = self.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        target_environment = self.model.environment_encoder(batch["next_frame"])
        if getattr(self.model, "predictor_type", "adaln") == "forward_inverse":
            assert agent.ndim == 3, f"forward_inverse expects frame latents [B,3,D], got {tuple(agent.shape)}"
            assert batch["history_actions"].shape[1] == agent.shape[1] - 1, (
                "history_actions must be [a_{t-2}, a_{t-1}] for frames [t-2,t-1,t]"
            )
            # prediction.environment: [B, 192] = z_pred_{t+1}; no true future frame is input to forward branch.
            prediction = self.model.step(agent, environment, batch["current_action"])
            forward_loss = self.environment_loss(prediction.environment, target_environment)
            # Inverse branch uses only true adjacent latents z_t and z_{t+1}.
            action_pred = self.model.inverse_action(environment, target_environment)
            inverse_loss = nn.functional.mse_loss(action_pred, batch["current_action"])
            # SIGReg applies only to real Frame Encoder latents: [z_{t-2}, z_{t-1}, z_t, z_{t+1}].
            real_latents = torch.cat((agent, target_environment[:, None]), dim=1)
            sigreg_embeddings = real_latents.transpose(0, 1)
            sigreg = self.sigreg_loss(sigreg_embeddings)
            total = (self.weights.get("forward", self.weights.get("prediction", 1.0)) * forward_loss
                     + self.weights.get("inverse", self.weights.get("inverse_weight", 0.1)) * inverse_loss
                     + self.weights.get("sigreg", self.weights.get("environment_sigreg", 0.1)) * sigreg)
            return {
                "loss": total,
                "total_loss": total,
                "forward_loss": forward_loss,
                "inverse_loss": inverse_loss,
                "sigreg_loss": sigreg,
                "latent_mean": real_latents.mean(),
                "latent_std": real_latents.std(dim=(0, 1)).mean(),
                "action_mse": inverse_loss,
            }

        if getattr(self.model, "predictor_type", "adaln") == "motion_token":
            # history_actions: [B, 2, A], current_action: [B, A]
            # action_window:    [B, 3, A] = [a_{t-2}, a_{t-1}, a_t]
            action_window = torch.cat((batch["history_actions"], batch["current_action"][:, None]), dim=1)
            assert action_window.shape[1] == 3, f"motion_token expects [B,3,A], got {tuple(action_window.shape)}"
            prediction = self.model.step(agent, environment, batch["current_action"], action_window)
            # prediction.environment: [B, 192] = z_pred_next
            prediction_loss = self.environment_loss(prediction.environment, target_environment)
            # sigreg embeddings: [T=2, B, 192]
            embeddings = torch.stack((environment, target_environment), dim=0)
            sigreg = self.sigreg_loss(embeddings)
            total = (self.weights.get("prediction", self.weights.get("environment", 1.0)) * prediction_loss
                     + self.weights.get("sigreg", self.weights.get("environment_sigreg", 0.1)) * sigreg)
            return {
                "loss": total,
                "prediction_loss": prediction_loss,
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
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.gradient_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        output = {key: value.detach().item() for key, value in losses.items()}
        output["gradient_norm"] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        output["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        output["global_step"] = float(self.global_step)
        return output

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
