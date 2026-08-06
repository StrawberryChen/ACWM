"""Train observation-only temporal graph/bridge representations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import TrajectoryDataset
from experiment_tracking import ExperimentLogger, append_metrics
from models.observation_graph import ObservationGraphBridgeModel
from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train observation-only temporal graph representation")
    parser.add_argument("--config", default="configs/obs_graph.yaml")
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
    history_length = int(data_config.get("history_length", 3))
    if history_length < 3:
        raise ValueError("observation graph training requires data.history_length >= 3")
    train_dataset = TrajectoryDataset.from_npz(train_paths, history_length, rollout_length=1)
    val_dataset = TrajectoryDataset.from_npz(val_paths, history_length, rollout_length=1)
    loader_options = {
        "batch_size": data_config.get("batch_size", 256),
        "num_workers": data_config.get("num_workers", 0),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model_config = config["model"]
    model = ObservationGraphBridgeModel(
        image_channels=model_config.get("image_channels", 3),
        latent_dim=model_config.get("latent_dim", 192),
        image_size=model_config.get("image_size", 224),
        patch_size=model_config.get("patch_size", 14),
        vit_depth=model_config.get("vit_depth", 12),
        vit_heads=model_config.get("vit_heads", 3),
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        hidden_dim=model_config.get("hidden_dim", 512),
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.get("learning_rate", 3e-4),
        weight_decay=training.get("weight_decay", 0.05),
        betas=tuple(training.get("betas", [0.9, 0.95])),
    )
    weights = config.get("loss", training.get("loss_weights", {}))
    logger = ExperimentLogger(config.get("wandb", {"enabled": False}), config)
    best_val = float("inf")
    for epoch in range(1, training["epochs"] + 1):
        print(f"\nObservation Graph Epoch {epoch}/{training['epochs']}")
        train_metrics = _run_epoch(model, train_loader, optimizer, device, weights, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer=None, device=device, weights=weights, train=False)
        metrics = {
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in val_metrics.items()},
        }
        numeric = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        append_metrics(training["metrics_file"], epoch, numeric)
        logger.log(metrics, step=epoch)
        _save_checkpoint(training["checkpoint"], model, optimizer, config, epoch, numeric)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            _save_checkpoint(training["best_checkpoint"], model, optimizer, config, epoch, numeric)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.6f}" for key, value in numeric.items()))
    logger.finish()


def _run_epoch(model: nn.Module, loader, optimizer, device: torch.device,
               weights: dict[str, float], train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    description = "Observation graph training" if train else "Observation graph validation"
    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=True)
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for raw_batch in progress:
            count += 1
            frames = raw_batch["history_frames"].to(device)
            losses = _compute_losses(model, frames, weights)
            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if weights.get("gradient_clip_norm"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(weights["gradient_clip_norm"]))
                optimizer.step()
            for key, value in losses.items():
                totals[key] += float(value.detach().item())
            progress.set_postfix(loss=f"{totals['loss'] / count:.5f}")
    if count == 0:
        raise ValueError("loader is empty")
    return {key: value / count for key, value in totals.items()}


def _compute_losses(model: ObservationGraphBridgeModel, frames: torch.Tensor,
                    weights: dict[str, float]) -> dict[str, torch.Tensor]:
    # Use the last three frames in the window as z1,z2,z3.
    # frames: [B,T,3,H,W] -> triple: [B,3,3,H,W]
    triple = frames[:, -3:]
    # z: [B,3,192]
    z = model.encode_sequence(triple)
    z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
    # learned directed edges: [B,192]
    e12 = model.edge(z1, z2, 1.0)
    e23 = model.edge(z2, z3, 1.0)
    e13 = model.edge(z1, z3, 2.0)
    # path composition: direct edge should match composed two-step path.
    path_loss = (e12 + e23 - e13).square().mean()
    # bridge midpoint: z1,z3,alpha=0.5 -> z2.
    z2_hat = model.bridge(z1, z3, 0.5, e13)
    bridge_loss = (z2_hat - z2).square().mean()
    linear_mid = 0.5 * (z1 + z3)
    linear_midpoint_mse = (linear_mid - z2).square().mean()
    var_loss, cov_loss, var_metrics = _vicreg_terms(z.flatten(0, 1))
    v12 = z2 - z1
    v23 = z3 - z2
    velocity_cosine = nn.functional.cosine_similarity(v12, v23, dim=-1, eps=1e-8).mean()
    velocity_loss = (1.0 - velocity_cosine)
    acceleration_norm = (z3 - 2.0 * z2 + z1).norm(dim=-1).mean()
    dist_k1 = 0.5 * (v12.norm(dim=-1).mean() + v23.norm(dim=-1).mean())
    dist_k2 = (z3 - z1).norm(dim=-1).mean()
    local_order_acc = (v12.norm(dim=-1) < (z3 - z1).norm(dim=-1)).float().mean()
    total = (
        weights.get("bridge", 1.0) * bridge_loss
        + weights.get("path", 1.0) * path_loss
        + weights.get("var", 1.0) * var_loss
        + weights.get("cov", 1.0) * cov_loss
        + weights.get("velocity", 0.01) * velocity_loss
    )
    return {
        "loss": total,
        "loss_bridge": bridge_loss,
        "loss_path": path_loss,
        "loss_var": var_loss,
        "loss_cov": cov_loss,
        "loss_velocity": velocity_loss,
        "temporal/dist_k1": dist_k1,
        "temporal/dist_k2": dist_k2,
        "temporal/local_order_acc_1_2": local_order_acc,
        "temporal/velocity_cosine_mean": velocity_cosine,
        "temporal/acceleration_norm_mean": acceleration_norm,
        "temporal/linear_midpoint_mse_k2": linear_midpoint_mse,
        "temporal/bridge_improvement_over_linear": linear_midpoint_mse - bridge_loss,
        **var_metrics,
    }


def _vicreg_terms(z: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    # z: [N,192]. This is anti-collapse only; it does not force Gaussian normality.
    assert z.ndim == 2, f"VICReg terms expect [N,D], got {tuple(z.shape)}"
    centered = z - z.mean(dim=0)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    var_loss = torch.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(cov.diag())
    cov_loss = off_diag.square().sum() / (z.shape[1] * (z.shape[1] - 1))
    return var_loss, cov_loss, {
        "latent/std_mean": std.mean(),
        "latent/std_min": std.min(),
        "latent/std_max": std.max(),
        "latent/cov_offdiag_abs_mean": off_diag.abs().sum() / (z.shape[1] * (z.shape[1] - 1)),
    }


def _save_checkpoint(path, model, optimizer, config, epoch, metrics) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "epoch": epoch, "metrics": metrics}, checkpoint)


if __name__ == "__main__":
    main()
