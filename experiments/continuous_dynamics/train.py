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
from .data import resolve, loader, validate_splits
from .model import ContinuousWorldModel
from .losses import objective
from .diagnostics import evaluate
from .visualize import visualize


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        n = len(batch["frames"])
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.)+float(value.detach())*n
        count += n
        progress.set_postfix(pred=f"{sums['loss_pred']/count:.6g}", progress=f"{sums['loss_progress']/count:.6g}")
    return {f"train/{key}": value/count for key, value in sums.items()}


def load_checkpoint(path, config, model):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"]["model"] != config["model"] or saved["config"]["loss"] != config["loss"]:
        raise ValueError("checkpoint model/loss mismatch; use matching saved config")
    for key in ("action_mode", "action_scale"):
        if saved["config"]["data"][key] != config["data"][key]:
            raise ValueError("checkpoint action semantics mismatch")
    model.load_state_dict(saved["model"])
    return saved


def log_figures(logger, paths, metrics):
    if logger.run is not None:
        metrics.update({key: logger.wandb.Image(str(path)) for key, path in paths.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continuous_dynamics.yaml")
    parser.add_argument("--data-config", help="borrow existing raw-step NPZ paths/action units/device")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--device")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = resolve(args.config, args.data_config, args.output_dir)
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
    if config["training"]["epochs"] < 1 or config["validation"]["visualization_interval"] < 1:
        raise ValueError("positive epochs/visualization interval required")
    if config["data"]["batch_size"] < 1 or config["validation"]["samples"] < 2:
        raise ValueError("positive batch size and >=2 validation samples required")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    paths = validate_splits(config)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("select a GPU Colab runtime, or explicitly --device cpu for smoke tests")
    model = ContinuousWorldModel(**config["model"]).to(device)
    saved = load_checkpoint(args.checkpoint, config, model) if args.checkpoint else None
    output = Path(config["output_dir"])
    if args.mode == "evaluate":
        output /= "evaluation_"+args.split
    if args.mode == "train" and not args.resume and (output/"checkpoints/latest.pt").exists():
        raise FileExistsError("output already contains a run; --resume or choose a NEW --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger_config = deepcopy(config["wandb"])
    if args.mode == "evaluate":
        logger_config["name"] += "_evaluate_"+args.split
    logger = ExperimentLogger(logger_config, config)
    print("No planning. No teacher/target detach/regularizer/noise. Raw-step relative actions; validation EVERY epoch.")
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
        provenance = source_signature(paths["train"])
        batches, _ = loader(config, "train", shuffle=True)
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
        for epoch in range(start+1, config["training"]["epochs"]+1):
            metrics = train_epoch(model, batches, optimizer, config, device)
            validation, values = evaluate(model, val_batches, config, device)
            metrics.update(validation)
            append_metrics(output/"metrics.jsonl", epoch, metrics)
            print(f"Epoch {epoch}/{config['training']['epochs']} total={metrics['train/loss_total']:.6g} "
                  f"val_pred={metrics['val/loss_pred']:.6g} std={metrics['collapse/feature_std']:.5g} "
                  f"rank={metrics['collapse/effective_rank']:.3g} violation={metrics['dynamics/progress_violation_ratio']:.3g}")
            improved = validation["val/loss_total"] < best
            best = min(best, validation["val/loss_total"])
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                              epoch=epoch, best=best, metrics=validation, initial_encoder=initial_encoder,
                              train_sources=provenance, torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                              numpy_rng=np.random.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
            torch.save(checkpoint, checkpoint_dir/"latest.pt")
            if improved:
                torch.save(checkpoint, checkpoint_dir/"best_loss.pt")
            if epoch == 1 or epoch % config["validation"]["visualization_interval"] == 0 or epoch == config["training"]["epochs"]:
                torch.save(values, output/"diagnostics.pt")
                figure_paths = visualize(model, val_dataset, config, output/"visualizations", epoch, device)
                log_figures(logger, figure_paths, metrics)
            logger.log(metrics, step=epoch)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
