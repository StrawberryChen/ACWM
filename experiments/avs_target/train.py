"""Independent AVS-target experiment. No planning, teacher or auxiliary losses."""
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
from .data import load_config, loader, source_signature, validate_splits
from .evaluate import evaluate
from .flow import flow_loss, flow_training_pair
from .model import AVSFlowWorldModel
from .targets import FixedMotionTarget, fit_statistics
from .visualize import visualize_generation, visualize_targets


def save_checkpoint(path, model, operator, optimizer, scheduler, config, epoch, metrics, best, initial_cse, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), target=operator.state_dict(), target_spec=operator.spec,
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), config=config,
                    epoch=epoch, metrics=metrics, best=best, initial_cse=initial_cse, history=history,
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None), path)


def train_epoch(model, operator, batches, optimizer, device, clip):
    model.train()
    total, count, state_gradient, action_gradient = 0., 0, 0., 0.
    progress = tqdm(batches, desc="AVS Flow train", dynamic_ncols=True)
    for batch in progress:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        target = operator(frames)  # no gradients, no trainable target projection
        noise, time = torch.randn_like(target), torch.rand(len(target), device=device)
        noisy, velocity = flow_training_pair(target, noise, time)
        predicted = model(frames[:, 0], actions, noisy, time)
        loss = flow_loss(predicted, velocity)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Flow Matching loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        state_gradient += sum(float(p.grad.square().sum()) for p in model.cse.parameters() if p.grad is not None)**0.5
        action_gradient += sum(float(p.grad.square().sum()) for p in model.predictor.action_encoder.parameters() if p.grad is not None)**0.5
        assert not any(p.requires_grad for p in operator.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += float(loss.detach())*len(target)
        count += len(target)
        progress.set_postfix(flow_loss=f"{total/count:.6f}")
    if not count:
        raise ValueError("empty train loader")
    return {"train/flow_loss": total/count, "gradient/cse_norm": state_gradient/len(batches),
            "gradient/action_encoder_norm": action_gradient/len(batches)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/avs_flow.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--target", choices=("avs", "pixel_diff", "random_projection"))
    parser.add_argument("--output-dir")
    parser.add_argument("--validation-interval", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("experiment") in {"terminal_only", "avs_terminal"}:
        raise ValueError("terminal experiments use experiments.avs_target.joint_train; AVS-only is preserved")
    if "baseline" in config:
        raise ValueError("run experiments.avs_target.baseline for the separate MSE+SIGReg control")
    if args.target:
        config["target"]["kind"] = args.target
        config["output_dir"] = str(Path(config["output_dir"]).parent/args.target)
        config["wandb"]["name"] = f"{args.target}_flow_seed{config['seed']}"
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.validation_interval is not None:
        config["validation"]["interval"] = args.validation_interval
    if config["validation"]["interval"] < 1 or config["training"]["epochs"] < 1:
        raise ValueError("positive training epochs and validation interval required")
    if config["validation"]["generation_samples"] < 2 or min(config["validation"]["flow_steps"]) < 1:
        raise ValueError("at least two generation samples and positive Euler steps required")
    if config["model"]["horizon"] not in config["validation"]["horizons"]:
        raise ValueError("validation horizons must include model horizon")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    paths = validate_splits(config)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Colab GPU runtime")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    operator = FixedMotionTarget(**config["target"]).to(device)
    model = AVSFlowWorldModel(operator.output_dim, **config["model"]).to(device)
    initial_cse = {key: value.detach().cpu().clone() for key, value in model.cse.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    best, first, history = float("inf"), 1, []
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if saved["config"]["model"] != config["model"] or saved["target_spec"] != operator.spec:
            raise ValueError("checkpoint architecture/fixed target mismatch")
        model.load_state_dict(saved["model"])
        operator.load_state_dict(saved["target"])
        initial_cse = saved["initial_cse"]
        if args.resume:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scheduler.T_max = config["training"]["epochs"]
            first, best, history = saved["epoch"]+1, saved["best"], saved["history"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda" and saved["cuda_rng"]:
                torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    elif args.resume or args.mode != "train":
        raise ValueError("--checkpoint required")
    provenance = source_signature(paths["train"])
    stats_path = output/"avs_stats.json"
    if stats_path.exists():
        before = deepcopy(operator.state_dict())
        stats = operator.load_statistics(stats_path)
        if stats["train_sources"] != provenance:
            raise ValueError("train sources changed; use a new experiment output directory")
        if args.checkpoint:
            for key in ("mean", "std"):
                torch.testing.assert_close(operator.state_dict()[key], before[key])
    elif args.checkpoint:
        raise FileNotFoundError("preserve avs_stats.json alongside checkpoint for normalization provenance")
    else:
        stats_config = deepcopy(config)
        stats_config["data"]["batch_size"] = config["data"]["stats_batch_size"]
        stats_loader, _ = loader(stats_config, "train", horizon=1)
        print("Fitting FIXED target statistics on unique training transitions only...")
        fit_statistics(operator, tqdm(stats_loader, desc="Target train statistics"), device, stats_path, provenance)
    config["target_output_dim"] = operator.output_dim
    (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger = ExperimentLogger(config["wandb"], config)
    try:
        split = args.split if args.mode == "evaluate" else "val"
        val_loader, val_dataset = loader(config, split, limit=config["validation"]["samples"])
        if len(val_loader.dataset) < 2:
            raise ValueError("validation needs at least two windows")
        if args.mode == "evaluate":
            metrics, examples = evaluate(model, operator, val_loader, config, device, config["seed"]+100)
            metrics[f"{split}/flow_loss"] = metrics.pop("flow_loss")
            visualize_generation(operator, examples, output/f"evaluation_{split}/videos", config["validation"]["comparison_videos"])
            append_metrics(output/f"evaluation_{split}/metrics.jsonl", 0, metrics)
            logger.log(metrics, step=0)
            print(json.dumps(metrics, indent=2))
            return
        actual = min(config["validation"]["visualization_examples"], len(val_loader.dataset))
        count = visualize_targets(operator, val_loader, output/"target_visualizations", device, actual)
        print(f"Saved {count} validation target diagnostics (requested {config['validation']['visualization_examples']}).")
        if logger.run is not None:
            images = [logger.wandb.Image(str(image)) for image in sorted((output/"target_visualizations").glob("*.png"))]
            logger.log({"avs/target_visualizations": images}, step=0)
        train_loader, _ = loader(config, "train", shuffle=True)
        for epoch in range(first, config["training"]["epochs"]+1):
            print(f"\nAVS-target {operator.kind}: epoch {epoch}/{config['training']['epochs']}; ONLY loss = Flow Matching MSE")
            metrics = train_epoch(model, operator, train_loader, optimizer, device, config["training"]["gradient_clip_norm"])
            improved = False
            if epoch % config["validation"]["interval"] == 0 or epoch == config["training"]["epochs"]:
                extra, examples = evaluate(model, operator, val_loader, config, device, config["seed"]+100)
                extra["val/flow_loss"] = extra.pop("flow_loss")
                metrics.update(extra)
                improved = metrics["generation/mse"] < best
                best = min(best, metrics["generation/mse"])
                visualize_generation(operator, examples, output/f"videos/epoch_{epoch:04d}", config["validation"]["comparison_videos"])
                if logger.run is not None:
                    metrics["generation/visualization"] = logger.wandb.Image(str(output/f"videos/epoch_{epoch:04d}/trajectory_000.png")) if config["validation"]["comparison_videos"] else None
            scheduler.step()
            serializable = {key: value for key, value in metrics.items() if isinstance(value, (float, int))}
            history.append({"epoch": epoch, **serializable})
            for name in (["latest.pt", "best_generation.pt"] if improved else ["latest.pt"]):
                save_checkpoint(output/"checkpoints"/name, model, operator, optimizer, scheduler,
                                config, epoch, serializable, best, initial_cse, history)
            append_metrics(output/"logs/metrics.jsonl", epoch, serializable)
            logger.log(metrics, step=epoch)
            print(json.dumps(serializable, indent=2))
        print("Weights saved. No planning: future motion is not a future state latent. Run evaluate and probes next.")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
