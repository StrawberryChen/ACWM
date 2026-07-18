"""Minimal training entry point for NPZ trajectory files."""
from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from evaluation import PlanningEvaluator, validate_prediction
from evaluation.prediction import save_prediction_animation
from experiment_tracking import ExperimentLogger, append_metrics
from planner import CEMPlanner
from trainer import ACWMTrainer
from utils import build_model, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Agent-Centric World Model")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.get("seed", 42))
    if config.get("device") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA, but no GPU is available; select a GPU Colab runtime")
    if torch.cuda.is_available() and config.get("device") == "cpu":
        print("Warning: CUDA is available but config.device=cpu; planning and training will be much slower.")
    data_config = config["data"]
    train_paths = data_config.get("train_paths", data_config.get("paths", []))
    val_paths = data_config.get("val_paths", [])
    if not train_paths or not val_paths:
        raise ValueError("data.train_paths and data.val_paths must both contain trajectory NPZ files")
    mode = config["training"].get("mode", "one_step")
    model_version = config.get("model", {}).get("version", "")
    train_rollout = config["training"].get("rollout_length", 1) if mode == "rollout" else 1
    train_dataset = TrajectoryDataset.from_npz(train_paths, data_config["history_length"], train_rollout)
    val_rollout = 5 if model_version == "v3_n1" else 1
    val_dataset = TrajectoryDataset.from_npz(val_paths, data_config["history_length"], val_rollout)
    loader_options = {"batch_size": data_config["batch_size"], "num_workers": data_config.get("num_workers", 0)}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = build_model(config)
    training = config["training"]
    optimizer_name = training.get("optimizer", "adam").lower()
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_type is None:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    if model_version == "v3_n1":
        action_min, action_max = _action_stats(train_dataset)
        model.set_action_stats(action_min, action_max)
        config["action_min"] = action_min.tolist()
        config["action_max"] = action_max.tolist()
    optimizer_kwargs = {"lr": training["learning_rate"], "weight_decay": training.get("weight_decay", 0.0)}
    if optimizer_name == "adamw" and "betas" in training:
        optimizer_kwargs["betas"] = tuple(training["betas"])
    optimizer = optimizer_type(model.parameters(), **optimizer_kwargs)
    scheduler = _build_scheduler(optimizer, training, len(train_loader)) if training.get("scheduler") else None
    loss_weights = dict(training.get("loss_weights", {}))
    loss_weights.update(config.get("loss", {}))
    trainer = ACWMTrainer(model, optimizer, loss_weights, config.get("device", "cpu"),
                          training.get("sigreg"), training.get("amp", False),
                          training.get("gradient_clip_norm"), scheduler)
    planner = CEMPlanner(**config["planner"])
    planning_evaluator = PlanningEvaluator(planner, config["environment"], data_config["history_length"],
                                           config["planner"]["action_dim"], config.get("device", "cpu"),
                                           dataset_paths=val_paths)
    logger = ExperimentLogger(config.get("wandb", {"enabled": False}), config)
    validation = config["validation"]
    best_success_rate = -1.0
    best_val_pred = float("inf")
    best_rollout_step5 = float("inf")
    for epoch in range(1, training["epochs"] + 1):
        print(f"\nEpoch {epoch}/{training['epochs']}")
        train_metrics = trainer.fit_epoch(train_loader, mode, f"Training {epoch}/{training['epochs']}")
        val_metrics = validate_prediction(trainer, val_loader)
        metrics = {**{f"train/{key}": value for key, value in train_metrics.items()},
                   **{f"validation/{key}": value for key, value in val_metrics.items()}}
        if epoch % validation.get("prediction_video_interval", 1) == 0:
            prediction_path = Path(validation["video_dir"]) / f"prediction_epoch_{epoch:04d}.gif"
            save_prediction_animation(trainer, val_loader, prediction_path,
                                      validation.get("prediction_video_samples", 32))
            metrics["validation/prediction_animation"] = logger.video(prediction_path, fps=4)
        if epoch % validation.get("planning_interval", 10) == 0:
            print(f"Running {validation.get('planning_episodes', 50)} Push-T planning episodes...")
            result = planning_evaluator.evaluate(
                model,
                episodes=validation.get("planning_episodes", 50),
                video_dir=Path(validation["video_dir"]) / f"epoch_{epoch:04d}",
                videos_to_save=validation.get("planning_videos", 3),
            )
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics[f"planning/{key}"] = value
            metrics["planning/evaluation_episodes"] = validation.get("planning_episodes", 50)
            for index, video_path in enumerate(result["videos"]):
                metrics[f"planning/video_{index}"] = logger.video(video_path, config["environment"].get("video_fps", 20))
            if result["success_rate"] > best_success_rate:
                best_success_rate = result["success_rate"]
                best_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
                _save_checkpoint(training["best_checkpoint"], model, optimizer, config, epoch, best_metrics)
        numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        val_pred = numeric_metrics.get("validation/loss_pred", numeric_metrics.get("validation/prediction_loss"))
        if val_pred is not None and val_pred < best_val_pred and training.get("best_val_pred_checkpoint"):
            best_val_pred = val_pred
            _save_checkpoint(training["best_val_pred_checkpoint"], model, optimizer, config, epoch, numeric_metrics)
        rollout_step5 = numeric_metrics.get("validation/rollout_mse_step_5")
        if rollout_step5 is not None and rollout_step5 < best_rollout_step5 and training.get("best_rollout_step5_checkpoint"):
            best_rollout_step5 = rollout_step5
            _save_checkpoint(training["best_rollout_step5_checkpoint"], model, optimizer, config, epoch, numeric_metrics)
        append_metrics(training["metrics_file"], epoch, numeric_metrics)
        logger.log(metrics, step=epoch)
        _save_checkpoint(training["checkpoint"], model, optimizer, config, epoch, numeric_metrics)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.6f}" for key, value in numeric_metrics.items()))
    logger.finish()


def _action_stats(dataset: TrajectoryDataset) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.cat([trajectory.actions for trajectory in dataset.trajectories], dim=0)
    return actions.min(dim=0).values, actions.max(dim=0).values


def _build_scheduler(optimizer, training: dict, steps_per_epoch: int):
    warmup_steps = max(1, int(training.get("warmup_epochs", 5)) * max(steps_per_epoch, 1))
    total_steps = max(warmup_steps + 1, int(training["epochs"]) * max(steps_per_epoch, 1))
    min_lr = float(training.get("min_learning_rate", 1e-6))
    base_lr = float(training["learning_rate"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_checkpoint(path, model, optimizer, config, epoch, metrics) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "epoch": epoch, "metrics": metrics,
                "action_min": config.get("action_min"), "action_max": config.get("action_max")},
               checkpoint)


if __name__ == "__main__":
    main()
