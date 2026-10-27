"""Additive terminal-only / AVS+terminal trainer; AVS-only entry stays unchanged."""
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
from .data import check_source_provenance, file_sha256, load_config, loader, source_signature, validate_splits
from .joint_evaluate import evaluate_joint, noise_for, targets_for, visualize_terminal
from .joint_model import JointFlowWorldModel, joint_flow_loss, joint_training_pair
from .targets import FixedMotionTarget, fit_statistics
from .terminal import FixedTerminalOperator
from .visualize import visualize_generation, visualize_targets


def build_joint(config, device):
    if config.get("experiment") not in {"terminal_only", "avs_terminal"}:
        raise ValueError("use terminal_only or avs_terminal config; AVS-only keeps its original trainer")
    terminal = FixedTerminalOperator(**config["terminal"]).to(device)
    motion = FixedMotionTarget(**config["target"]).to(device) if config["experiment"] == "avs_terminal" else None
    model = JointFlowWorldModel(terminal.output_dim, None if motion is None else motion.output_dim,
                                **config["model"]).to(device)
    return model, terminal, motion


def load_weights(saved, config, model, terminal, motion):
    if (saved["config"]["model"] != config["model"] or saved["config"]["experiment"] != config["experiment"]
            or saved["terminal_spec"] != terminal.spec):
        raise ValueError("checkpoint architecture/terminal measurement mismatch")
    if saved["config"]["data"]["action_scale"] != config["data"]["action_scale"]:
        raise ValueError("checkpoint action_scale differs; do not reinterpret normalized actions")
    if (saved["motion_spec"] is None) != (motion is None):
        raise ValueError("checkpoint motion branch mismatch")
    if motion is not None and saved["motion_spec"] != motion.spec:
        raise ValueError("checkpoint AVS target mismatch")
    model.load_state_dict(saved["model"])
    terminal.load_state_dict(saved["terminal"])
    if motion is not None:
        motion.load_state_dict(saved["motion"])


def write_checkpoint(path, model, terminal, motion, optimizer, scheduler, config,
                     epoch, metrics, best, initial_cse, history, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), terminal=terminal.state_dict(), terminal_spec=terminal.spec,
                    motion=None if motion is None else motion.state_dict(), motion_spec=None if motion is None else motion.spec,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), config=config,
                    epoch=epoch, metrics=metrics, best=best, initial_cse=initial_cse, history=history,
                    train_sources=provenance, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), path)


def train_joint_epoch(model, terminal, motion, batches, optimizer, config, device):
    model.train()
    sums, count = {}, 0
    progress = tqdm(batches, desc=config["experiment"]+" train", dynamic_ncols=True)
    for batch in progress:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        target = targets_for(frames, terminal, motion)
        noise = {name: torch.randn_like(value) for name, value in target.items()}
        time = torch.rand(len(frames), device=device)
        noisy, velocity = joint_training_pair(target, noise, time)
        prediction = model.predictor(noisy, time, model.cse(frames[:, 0]), actions)
        metrics = joint_flow_loss(prediction, velocity, config["loss"]["lambda_motion"], config["loss"]["lambda_terminal"])
        if not torch.isfinite(metrics["loss"]):
            raise FloatingPointError("non-finite joint FM loss")
        optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        metrics["cse_gradient_norm"] = sum(float(p.grad.square().sum()) for p in model.cse.parameters() if p.grad is not None)**0.5
        metrics["action_gradient_norm"] = sum(float(p.grad.square().sum()) for p in model.predictor.action_encoder.parameters() if p.grad is not None)**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
        optimizer.step()
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.)+float(value.detach() if torch.is_tensor(value) else value)*len(frames)
        count += len(frames)
        progress.set_postfix(motion=f"{sums['motion_flow_loss']/count:.5f}", terminal=f"{sums['terminal_flow_loss']/count:.5f}")
    return {f"train/{name}": value/count for name, value in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/avs_terminal.yaml")
    parser.add_argument("--data-config", help="reuse existing Colab raw-step dataset configuration")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_config:
        prepared = load_config(args.data_config)
        config["data"], config["device"] = prepared["data"], prepared["device"]
        config["output_dir"] = str(Path(prepared["output_dir"]).parent/config["experiment"])
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if config["validation"]["interval"] < 1 or config["training"]["epochs"] < 1:
        raise ValueError("positive epochs/validation interval required")
    if config["validation"]["generation_samples"] < 2 or min(config["validation"]["flow_steps"]) < 1:
        raise ValueError("need >=2 generation samples and positive flow steps")
    if config["model"]["horizon"] not in config["validation"]["horizons"]:
        raise ValueError("validation horizons must include the training horizon")
    if config["loss"]["lambda_motion"] < 0 or config["loss"]["lambda_terminal"] <= 0:
        raise ValueError("nonnegative motion weight and positive terminal weight required")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    paths = validate_splits(config)
    provenance = source_signature(paths["train"], content_hash=args.mode == "train")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("enable a GPU runtime")
    model, terminal, motion = build_joint(config, device)
    initial_cse = {k: v.detach().cpu().clone() for k, v in model.cse.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    first, best, history = 1, float("inf"), []
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        load_weights(saved, config, model, terminal, motion)
        check_source_provenance(saved["train_sources"], provenance, "checkpoint training dataset", training=args.mode == "train")
        initial_cse = saved["initial_cse"]
        if args.resume:
            if saved["config"]["loss"] != config["loss"]:
                raise ValueError("changed loss weights require a new experiment")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scheduler.T_max = config["training"]["epochs"]
            first, best, history = saved["epoch"]+1, saved["best"], saved["history"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            if device.type == "cuda" and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    elif args.resume or args.mode != "train":
        raise ValueError("--checkpoint required")
    if motion is not None:
        stats_path = output/"avs_stats.json"
        if stats_path.exists():
            before = motion.std.clone(), motion.mean.clone()
            stats = motion.load_statistics(stats_path)
            check_source_provenance(stats["train_sources"], provenance, "AVS target statistics", training=args.mode == "train")
            if args.checkpoint:
                torch.testing.assert_close(before[0], motion.std)
                torch.testing.assert_close(before[1], motion.mean)
        elif args.checkpoint:
            raise FileNotFoundError("keep avs_stats.json alongside the joint checkpoint")
        else:
            stats_config = deepcopy(config)
            stats_config["data"]["batch_size"] = config["data"]["stats_batch_size"]
            stats_loader, _ = loader(stats_config, "train", horizon=1)
            fit_statistics(motion, tqdm(stats_loader, desc="Fixed AVS train statistics"), device, stats_path, provenance)
    metadata_output = output if args.mode == "train" else output/f"evaluation_{args.split}"
    metadata_output.mkdir(parents=True, exist_ok=True)
    (metadata_output/"terminal_spec.json").write_text(json.dumps(terminal.spec, indent=2))
    (metadata_output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    split = args.split if args.mode == "evaluate" else "val"
    validation, _ = loader(config, split, limit=config["validation"]["samples"])
    if len(validation.dataset) < 2:
        raise ValueError("need at least two validation windows")
    logger = ExperimentLogger(config["wandb"], config)
    try:
        if args.mode == "train":
            first_batch = next(iter(validation))["frames"].to(device)
            y = terminal(first_batch[:, -1])
            visualize_terminal(terminal, first_batch.cpu(), y.cpu(), None, output/"terminal_target.png")
            if motion is not None:
                visualize_targets(motion, validation, output/"target_visualizations", device,
                                  min(config["validation"]["visualization_examples"], len(validation.dataset)))
            if logger.run is not None:
                media = {"terminal/target_visualization": logger.wandb.Image(str(output/"terminal_target.png"))}
                if motion is not None:
                    media["avs/target_visualizations"] = [logger.wandb.Image(str(path))
                        for path in sorted((output/"target_visualizations").glob("*.png"))]
                logger.log(media, step=0)
            train, _ = loader(config, "train", shuffle=True)
        epochs = range(first, config["training"]["epochs"]+1) if args.mode == "train" else [0]
        for epoch in epochs:
            media = {}
            metrics = train_joint_epoch(model, terminal, motion, train, optimizer, config, device) if args.mode == "train" else {}
            improved = False
            if args.mode == "evaluate" or epoch % config["validation"]["interval"] == 0 or epoch == config["training"]["epochs"]:
                measured, examples = evaluate_joint(model, terminal, motion, validation, config, device)
                for key in ("loss", "motion_flow_loss", "terminal_flow_loss"):
                    measured[f"{split}/{key}"] = measured.pop(key)
                metrics.update(measured)
                improved = measured["generation/terminal_mse"] < best
                best = min(best, measured["generation/terminal_mse"])
                destination = output/f"evaluation_{split}" if args.mode == "evaluate" else output/f"validation/epoch_{epoch:04d}"
                visualize_terminal(terminal, examples["frames"], examples["target_terminal"], examples["predicted_terminal"], destination/"terminal.png")
                if motion is not None:
                    # Reuse original motion visualization, no change to AVS-only.
                    motion_examples = {"frames": examples["frames"], "prediction": examples["predicted_motion"],
                                       "target": motion(examples["frames"].to(device)).cpu()}
                    visualize_generation(motion, motion_examples, destination/"motion_videos", config["validation"]["comparison_videos"])
                if logger.run is not None:
                    media["terminal/generated_visualization"] = logger.wandb.Image(str(destination/"terminal.png"))
                if args.mode == "evaluate":
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination/"report.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
                        "checkpoint_sha256": file_sha256(args.checkpoint), "split": split, "metrics": metrics}, indent=2))
            if args.mode == "train":
                scheduler.step()
                history.append({"epoch": epoch, **metrics})
                for name in (["latest.pt", "best_terminal.pt"] if improved else ["latest.pt"]):
                    write_checkpoint(output/"checkpoints"/name, model, terminal, motion, optimizer, scheduler,
                                     config, epoch, metrics, best, initial_cse, history, provenance)
            append_metrics(output/"logs"/f"{args.mode}_{split}.jsonl", epoch, metrics)
            logger.log({**metrics, **media}, step=epoch)
            print(json.dumps(metrics, indent=2))
        print("Prediction phase complete. Planning is NOT automatic: inspect prediction and frozen probes first.")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
