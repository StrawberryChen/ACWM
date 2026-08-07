"""Train an action-conditioned predictor supervised by frozen obs-graph edges."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import TrajectoryDataset
from evaluation import PlanningEvaluator
from experiment_tracking import ExperimentLogger, append_metrics
from models.edge_guided_forward import EdgeGuidedForwardModel
from models.observation_graph import ObservationGraphBridgeModel
from planner import CEMPlanner
from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train edge-guided forward model")
    parser.add_argument("--config", default="configs/edge_guided_forward.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA, but no GPU is available")
    data_config = config["data"]
    train_paths = data_config.get("train_paths", [])
    val_paths = data_config.get("val_paths", [])
    if not train_paths or not val_paths:
        raise ValueError("data.train_paths and data.val_paths must both contain trajectory NPZ files")
    rollout_length = int(config["training"].get("rollout_length", 5))
    train_dataset = TrajectoryDataset.from_npz(train_paths, data_config["history_length"], rollout_length)
    val_dataset = TrajectoryDataset.from_npz(val_paths, data_config["history_length"], rollout_length)
    loader_options = {"batch_size": data_config["batch_size"], "num_workers": data_config.get("num_workers", 0)}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    obs_model = _load_obs_model(config["model"]["obs_checkpoint"], device)
    model_config = config["model"]
    model = EdgeGuidedForwardModel(
        obs_model=obs_model,
        latent_dim=model_config.get("latent_dim", 192),
        raw_action_dim=model_config.get("raw_action_dim", 2),
        action_block=model_config.get("action_block", 5),
        history_size=model_config.get("history_size", data_config["history_length"]),
        hidden_dim=model_config.get("hidden_dim", 512),
        temporal_layers=model_config.get("temporal_layers", 2),
        temporal_heads=model_config.get("temporal_heads", 3),
        temporal_dropout=model_config.get("temporal_dropout", 0.0),
        planning_z_weight=model_config.get("planning_z_weight", 1.0),
        planning_edge_weight=model_config.get("planning_edge_weight", 1.0),
        planning_delta_t=float(config["planner"].get("horizon", 5)),
    ).to(device)
    action_mean, action_std = _action_stats(train_dataset)
    model.set_action_stats(action_mean.to(device), action_std.to(device))
    config["action_mean"] = action_mean.tolist()
    config["action_std"] = action_std.tolist()

    optimizer = torch.optim.AdamW(
        model.forward_predictor.parameters(),
        lr=config["training"].get("learning_rate", 3e-4),
        weight_decay=config["training"].get("weight_decay", 0.05),
        betas=tuple(config["training"].get("betas", [0.9, 0.95])),
    )
    planner = CEMPlanner(**config["planner"])
    planning_paths = train_paths if config["environment"].get("planning_dataset", "train") == "train" else val_paths
    planning_evaluator = PlanningEvaluator(
        planner,
        config["environment"],
        data_config["history_length"],
        config["planner"]["action_dim"],
        device,
        dataset_paths=planning_paths,
    )
    logger = ExperimentLogger(config.get("wandb", {"enabled": False}), config)
    weights = config.get("loss", {})
    best_val = float("inf")
    best_success = -1.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\nEdge-guided Epoch {epoch}/{config['training']['epochs']}")
        train_metrics = _run_epoch(model, train_loader, optimizer, device, weights, train=True)
        val_metrics = _run_epoch(model, val_loader, None, device, weights, train=False)
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in val_metrics.items()},
        }
        if epoch % config["validation"].get("planning_interval", 20) == 0:
            print(f"Running {config['validation'].get('planning_episodes', 50)} Push-T planning episodes...")
            result = planning_evaluator.evaluate(
                model,
                episodes=config["validation"].get("planning_episodes", 50),
                video_dir=Path(config["validation"]["video_dir"]) / f"epoch_{epoch:04d}",
                videos_to_save=config["validation"].get("planning_videos", 3),
            )
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics[f"planning/{key}"] = value
            if result["success_rate"] > best_success:
                best_success = result["success_rate"]
                _save_checkpoint(config["training"]["best_planning_checkpoint"], model, optimizer, config, epoch, metrics)
        numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        append_metrics(config["training"]["metrics_file"], epoch, numeric)
        logger.log(metrics, step=epoch)
        _save_checkpoint(config["training"]["checkpoint"], model, optimizer, config, epoch, numeric)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            _save_checkpoint(config["training"]["best_checkpoint"], model, optimizer, config, epoch, numeric)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.6f}" for key, value in numeric.items()))
    logger.finish()


def _load_obs_model(path: str | Path, device: torch.device) -> ObservationGraphBridgeModel:
    checkpoint = torch.load(path, map_location=device)
    obs_config = checkpoint.get("config", {}).get("model", {})
    model = ObservationGraphBridgeModel(
        image_channels=obs_config.get("image_channels", 3),
        latent_dim=obs_config.get("latent_dim", 192),
        image_size=obs_config.get("image_size", 224),
        patch_size=obs_config.get("patch_size", 14),
        vit_depth=obs_config.get("vit_depth", 12),
        vit_heads=obs_config.get("vit_heads", 3),
        mlp_ratio=obs_config.get("mlp_ratio", 4.0),
        hidden_dim=obs_config.get("hidden_dim", 512),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _run_epoch(model: EdgeGuidedForwardModel, loader, optimizer, device: torch.device,
               weights: dict[str, float], train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    progress = tqdm(loader, desc="Edge-guided training" if train else "Edge-guided validation",
                    dynamic_ncols=True, leave=True)
    with context:
        for raw_batch in progress:
            count += 1
            batch = {key: value.to(device) for key, value in raw_batch.items()}
            losses = _compute_losses(model, batch, weights)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if weights.get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(model.forward_predictor.parameters(), float(weights["gradient_clip_norm"]))
                optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach().item())
            progress.set_postfix(loss=f"{totals['loss'] / count:.5f}")
    if count == 0:
        raise ValueError("loader is empty")
    return {key: value / count for key, value in totals.items()}


def _compute_losses(model: EdgeGuidedForwardModel, batch: dict[str, torch.Tensor],
                    weights: dict[str, float]) -> dict[str, torch.Tensor]:
    # Frozen obs latents.
    history = model.encode_sequence(batch["history_frames"])
    target_next = model.encode_frame(batch["next_frame"])
    pred_sequence = model.predict_next_sequence(history, batch["history_actions"], batch["current_action"])
    target_sequence = torch.cat((history[:, 1:], target_next[:, None]), dim=1)
    z_loss_per_step = (pred_sequence - target_sequence).square().mean(dim=(0, 2))
    z_loss = z_loss_per_step.mean()

    # Edge supervision: Edge(z_i, z_pred_{i+1}) should match Edge(z_i, z_{i+1}).
    pred_edges, true_edges = [], []
    for index in range(history.shape[1]):
        pred_edges.append(model.edge(history[:, index], pred_sequence[:, index], 1.0))
        with torch.no_grad():
            true_edges.append(model.edge(history[:, index], target_sequence[:, index], 1.0))
    pred_edge = torch.stack(pred_edges, dim=1)
    true_edge = torch.stack(true_edges, dim=1)
    edge_loss_per_step = (pred_edge - true_edge).square().mean(dim=(0, 2))
    edge_loss = edge_loss_per_step.mean()

    terminal_z_loss = torch.zeros((), device=history.device)
    terminal_edge_loss = torch.zeros((), device=history.device)
    if "rollout_actions" in batch and weights.get("terminal", 1.0) > 0:
        predictions = model.rollout(history, history[:, -1], batch["rollout_actions"], batch["history_actions"])
        pred_terminal = predictions[-1].environment
        target_terminal = model.encode_frame(batch["rollout_frames"][:, -1])
        terminal_z_loss = (pred_terminal - target_terminal).square().mean()
        pred_terminal_edge = model.edge(history[:, -1], pred_terminal, batch["rollout_actions"].shape[1])
        with torch.no_grad():
            true_terminal_edge = model.edge(history[:, -1], target_terminal, batch["rollout_actions"].shape[1])
        terminal_edge_loss = (pred_terminal_edge - true_terminal_edge).square().mean()

    total = (
        weights.get("z", 1.0) * z_loss
        + weights.get("edge", 1.0) * edge_loss
        + weights.get("terminal", 1.0) * (terminal_z_loss + terminal_edge_loss)
    )
    final_pred = pred_sequence[:, -1]
    delta_pred = final_pred - history[:, -1]
    shuffled = model.step(
        history,
        history[:, -1],
        batch["current_action"][torch.randperm(batch["current_action"].shape[0], device=history.device)],
        batch["history_actions"],
    ).environment
    shuffle_mse = (shuffled - target_next).square().mean()
    return {
        "loss": total,
        "loss_z": z_loss,
        "loss_z_step_1": z_loss_per_step[0],
        "loss_z_step_2": z_loss_per_step[1],
        "loss_z_step_3": z_loss_per_step[2],
        "loss_edge": edge_loss,
        "loss_edge_step_1": edge_loss_per_step[0],
        "loss_edge_step_2": edge_loss_per_step[1],
        "loss_edge_step_3": edge_loss_per_step[2],
        "loss_terminal_z": terminal_z_loss,
        "loss_terminal_edge": terminal_edge_loss,
        "delta_pred_norm": delta_pred.norm(dim=-1).mean(),
        "pred_latent_std_mean": final_pred.std(dim=0).mean(),
        "action_shuffle_mse_normal": z_loss_per_step[-1],
        "action_shuffle_mse_shuffle": shuffle_mse,
        "action_sensitivity": shuffle_mse - z_loss_per_step[-1],
    }


def _action_stats(dataset: TrajectoryDataset) -> tuple[torch.Tensor, torch.Tensor]:
    actions = torch.cat([trajectory.actions for trajectory in dataset.trajectories], dim=0).float()
    actions = actions.reshape(-1, actions.shape[-1])
    return actions.mean(dim=0), actions.std(dim=0).clamp_min(1e-6)


def _save_checkpoint(path, model, optimizer, config: dict[str, Any], epoch: int, metrics: dict[str, Any]) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }, checkpoint)


if __name__ == "__main__":
    main()
