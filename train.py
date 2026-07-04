"""Minimal training entry point for NPZ trajectory files."""
from __future__ import annotations

import argparse
from pathlib import Path

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
    data_config = config["data"]
    train_paths = data_config.get("train_paths", data_config.get("paths", []))
    val_paths = data_config.get("val_paths", [])
    if not train_paths or not val_paths:
        raise ValueError("data.train_paths and data.val_paths must both contain trajectory NPZ files")
    mode = config["training"].get("mode", "one_step")
    train_rollout = config["training"].get("rollout_length", 1) if mode == "rollout" else 1
    train_dataset = TrajectoryDataset.from_npz(train_paths, data_config["history_length"], train_rollout)
    val_dataset = TrajectoryDataset.from_npz(val_paths, data_config["history_length"], 1)
    loader_options = {"batch_size": data_config["batch_size"], "num_workers": data_config.get("num_workers", 0)}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = build_model(config)
    training = config["training"]
    optimizer_name = training.get("optimizer", "adam").lower()
    optimizer_type = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(optimizer_name)
    if optimizer_type is None:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    optimizer = optimizer_type(model.parameters(), lr=training["learning_rate"],
                               weight_decay=training.get("weight_decay", 0.0))
    trainer = ACWMTrainer(model, optimizer, training["loss_weights"], config.get("device", "cpu"))
    planner = CEMPlanner(**config["planner"])
    planning_evaluator = PlanningEvaluator(planner, config["environment"], data_config["history_length"],
                                           config["planner"]["action_dim"], config.get("device", "cpu"))
    logger = ExperimentLogger(config.get("wandb", {"enabled": False}), config)
    validation = config["validation"]
    best_success_rate = -1.0
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
        if epoch % validation.get("planning_interval", 5) == 0:
            print(f"Running {validation.get('planning_episodes', 100)} Push-T planning episodes...")
            result = planning_evaluator.evaluate(
                model,
                episodes=validation.get("planning_episodes", 100),
                video_dir=Path(validation["video_dir"]) / f"epoch_{epoch:04d}",
                videos_to_save=validation.get("planning_videos", 3),
            )
            metrics["planning/success_rate"] = result["success_rate"]
            metrics["planning/mean_max_reward"] = result["mean_max_reward"]
            for index, video_path in enumerate(result["videos"]):
                metrics[f"planning/video_{index}"] = logger.video(video_path, config["environment"].get("video_fps", 20))
            if result["success_rate"] > best_success_rate:
                best_success_rate = result["success_rate"]
                best_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
                _save_checkpoint(training["best_checkpoint"], model, optimizer, config, epoch, best_metrics)
        numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        append_metrics(training["metrics_file"], epoch, numeric_metrics)
        logger.log(metrics, step=epoch)
        _save_checkpoint(training["checkpoint"], model, optimizer, config, epoch, numeric_metrics)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.6f}" for key, value in numeric_metrics.items()))
    logger.finish()


def _save_checkpoint(path, model, optimizer, config, epoch, metrics) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "epoch": epoch, "metrics": metrics}, checkpoint)


if __name__ == "__main__":
    main()
