"""Independent trainer: two loss terms only, diagnostics every epoch, no planner."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from experiment_tracking import ExperimentLogger, append_metrics
from experiments.avs_target.data import source_signature, check_source_provenance
from .checkpoint_io import read_checkpoint, require_mounted_drive, save_checkpoint
from .data import resolve, loader, validate_splits
from .leworld_data import provenance as leworld_provenance, split_counts, uses_leworld
from .model import ContinuousWorldModel
from .losses import objective
from .diagnostics import evaluate
from .visualize import visualize


def build_model(config):
    if config.get("experiment") == "continuous_velocity":
        from .velocity_model import SequentialVelocityWorldModel
        return SequentialVelocityWorldModel(**config["model"])
    if config.get("experiment") == "continuous_trajectory":
        settings = dict(config["model"])
        scope = settings.pop("direction_scope", "global")
        if scope == "per_step":
            from .per_step_model import PerStepTrajectoryWorldModel
            return PerStepTrajectoryWorldModel(**settings)
        if scope != "global":
            raise ValueError("direction_scope must be global or per_step")
        from .trajectory_model import TrajectoryWorldModel
        return TrajectoryWorldModel(**settings)
    return ContinuousWorldModel(**config["model"])


def train_epoch(model, batches, optimizer, config, device):
    model.train()
    sums, count = {}, 0
    progress = tqdm(batches, desc="continuous dynamics train", dynamic_ncols=True)
    for batch in progress:
        result = model(batch["frames"].to(device), batch["actions"].to(device))
        losses = objective(result, **config["loss"])
        if not torch.isfinite(losses["loss_total"]):
            raise FloatingPointError("non-finite continuous dynamics loss")
        optimizer.zero_grad(set_to_none=True)
        losses["loss_total"].backward()
        gradient_metrics = {}
        for name in (("field",) if "loss_velocity" in losses else ("direction_head", "action_trajectory")):
            if hasattr(model, name):
                norm = sum(float(p.grad.detach().square().sum()) for p in getattr(model, name).parameters() if p.grad is not None)**.5
                gradient_metrics[f"{name}_gradient_norm"] = norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        n = len(batch["frames"])
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.)+float(value.detach())*n
        for key, value in gradient_metrics.items():
            sums[key] = sums.get(key, 0.)+value*n
        count += n
        if "loss_velocity" in losses:
            progress.set_postfix(velocity=f"{sums['loss_velocity']/count:.6g}", sigreg=f"{sums['loss_sigreg']/count:.6g}")
        else:
            progress.set_postfix(pred=f"{sums['loss_pred']/count:.6g}", progress=f"{sums['loss_progress']/count:.6g}")
    return {f"train/{key}": value/count for key, value in sums.items()}


def load_checkpoint(path, config, model, *, saved=None):
    if saved is None:
        saved = read_checkpoint(path)
    if (saved["config"]["model"] != config["model"] or saved["config"]["loss"] != config["loss"]
            or saved["config"].get("experiment") != config.get("experiment")):
        raise ValueError("checkpoint model/loss mismatch; use matching saved config")
    for key in ("action_mode", "action_scale", "dataset"):
        if saved["config"]["data"].get(key) != config["data"].get(key):
            raise ValueError("checkpoint action semantics mismatch")
    model.load_state_dict(saved["model"])
    return saved


def log_figures(logger, paths, metrics):
    if logger.run is not None:
        metrics.update({key: logger.wandb.Image(str(path)) for key, path in paths.items()})


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config")
    source.add_argument("--workflow", help="Colab workflow YAML")
    parser.add_argument("--data-config", help="borrow existing raw-step NPZ paths/action units/device")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int, help="Pause after this absolute epoch without changing the training schedule")
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--device")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.workflow:
        if args.data_config or args.output_dir or args.epochs is not None:
            parser.error("edit the workflow YAML instead of overriding its data, output or epochs")
        from .workflow import Workflow
        config = Workflow.from_yaml(args.workflow).config
    else:
        config = resolve(args.config or "configs/continuous_dynamics.yaml", args.data_config, args.output_dir)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.integration_steps is not None:
        config["model"]["integration_steps"] = args.integration_steps
    if args.margin is not None:
        config["loss"]["margin"] = args.margin
    if args.device:
        config["device"] = args.device
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    if args.resume and (args.mode != "train" or not args.checkpoint):
        parser.error("--resume requires --mode train and --checkpoint")
    if args.mode == "train" and args.checkpoint and not args.resume:
        parser.error("training a checkpoint requires explicit --resume")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("independent evaluation requires --checkpoint")
    if args.stop_after_epoch is not None and (
        args.mode != "train" or not 1 <= args.stop_after_epoch <= config["training"]["epochs"]
    ):
        parser.error("--stop-after-epoch must be within the training schedule")
    if config["training"]["epochs"] < 1 or config["validation"]["visualization_interval"] < 1:
        raise ValueError("positive epochs/visualization interval required")
    val_interval = config["validation"].setdefault("interval", 1)
    if type(val_interval) is not int or val_interval < 1:
        raise ValueError("validation.interval must be a positive integer")
    if config["data"]["batch_size"] < 1 or config["validation"]["samples"] < 2:
        raise ValueError("positive batch size and >=2 validation samples required")
    if config.get("experiment") == "continuous_velocity" and config["loss"]["mode"] != "velocity_sigreg":
        raise ValueError("continuous_velocity requires loss.mode=velocity_sigreg")
    require_mounted_drive(config["output_dir"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    paths = None if uses_leworld(config) else validate_splits(config)
    if paths is None:
        print(f"Official LeWorld episodes/windows by split: {split_counts(config)}", flush=True)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("select a GPU Colab runtime, or explicitly --device cpu for smoke tests")
    model = build_model(config).to(device)
    saved = load_checkpoint(args.checkpoint, config, model) if args.checkpoint else None
    output = Path(config["output_dir"])
    if args.mode == "evaluate":
        output /= "evaluation_"+args.split
    if args.mode == "train" and not args.resume and (output/"checkpoints/latest.pt").exists():
        raise FileExistsError("output already contains a run; --resume or choose a NEW --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        print(f"Training checkpoint: {(output / 'checkpoints' / 'latest.pt').resolve()}", flush=True)
    (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger_config = deepcopy(config["wandb"])
    if args.mode == "evaluate":
        logger_config["name"] += "_evaluate_"+args.split
        logger_config.pop("id", None)
    elif args.resume and logger_config.get("id") and logger_config.get("mode", "online") == "online":
        logger_config["resume"] = "allow"
    logger = ExperimentLogger(logger_config, config)
    print(f"Raw-step relative actions; no planning. Training validation every {val_interval} epoch(s) and on the final epoch.")
    try:
        val_batches, val_dataset = loader(config, args.split if args.mode == "evaluate" else "val", config["validation"]["samples"])
        if args.mode == "evaluate":
            metrics, values = evaluate(model, val_batches, config, device)
            figure_paths = visualize(model, val_dataset, config, output/"visualizations", saved["epoch"], device)
            (output/"metrics.json").write_text(json.dumps(metrics, indent=2))
            torch.save(values, output/"diagnostics.pt")
            log_figures(logger, figure_paths, metrics)
            logger.log(metrics, step=0)
            print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (float, int))}, indent=2))
            return
        provenance = leworld_provenance(config) if paths is None else source_signature(paths["train"])
        batches, train_dataset = loader(config, "train", shuffle=True)
        print(f"Training windows: {len(train_dataset):,}; optimizer steps per epoch: {len(batches):,}", flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        # Constant LR: no implicit scheduler changes across K/margin experiments.
        start, best = 0, float("inf")
        initial_encoder = ({k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
                           if saved is None else saved["initial_encoder"])
        if saved:
            for section in ("training", "seed"):
                if saved["config"][section] != config[section]:
                    raise ValueError("resume settings differ; preserve original training settings")
            check_source_provenance(saved["train_sources"], provenance, "continuous resume", training=True)
            optimizer.load_state_dict(saved["optimizer"])
            start, best = saved["epoch"], saved["best"]
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            if saved["cuda_rng"] is not None and device.type == "cuda":
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
        checkpoint_dir = output/"checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        epochs = config["training"]["epochs"]
        end = args.stop_after_epoch or epochs
        for epoch in range(start+1, end+1):
            metrics = train_epoch(model, batches, optimizer, config, device)
            validation, improved = {}, False
            message = f"Epoch {epoch}/{epochs} total={metrics['train/loss_total']:.6g}"
            if epoch % val_interval == 0 or epoch == epochs:
                validation, values = evaluate(model, val_batches, config, device)
                metrics.update(validation)
                improved = validation["val/loss_total"] < best
                best = min(best, validation["val/loss_total"])
                message += (f" val_pred={metrics['val/loss_pred']:.6g} std={metrics['collapse/feature_std']:.5g} "
                            f"rank={metrics['collapse/effective_rank']:.3g}")
                if "val/loss_velocity" in metrics:
                    message += f" vel={metrics['val/loss_velocity']:.6g} shuffle_delta={metrics['action/shuffle_loss_delta']:.4g}"
                else:
                    message += f" violation={metrics['dynamics/progress_violation_ratio']:.3g}"
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                              epoch=epoch, best=best, metrics=validation, initial_encoder=initial_encoder,
                              train_sources=provenance, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                              numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
            if improved:
                save_checkpoint(checkpoint_dir/"best_loss.pt", checkpoint, verify=epoch == end)
            verify = epoch in (1, end)
            save_checkpoint(checkpoint_dir/"latest.pt", checkpoint, verify=verify)
            if verify:
                print(f"Checkpoint read back from mounted filesystem: epoch={epoch}, {checkpoint_dir/'latest.pt'}", flush=True)
            append_metrics(output/"metrics.jsonl", epoch, metrics)
            print(message if validation else message+" (validation skipped)")
            if validation and (epoch == 1 or epoch % config["validation"]["visualization_interval"] == 0 or epoch == epochs):
                torch.save(values, output/"diagnostics.pt")
                figure_paths = visualize(model, val_dataset, config, output/"visualizations", epoch, device)
                log_figures(logger, figure_paths, metrics)
            logger.log(metrics, step=epoch)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
