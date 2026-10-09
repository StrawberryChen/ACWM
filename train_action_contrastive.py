"""Independent Action-Contrastive training, held-out evaluation and optional planning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from evaluation.action_contrastive import diagnostics, make_loader
from evaluation.planning import PlanningEvaluator
from experiment_tracking import ExperimentLogger, append_metrics
from models.action_contrastive import ActionContrastiveWorldModel, negative_actions, ranking_metrics
from planner import CEMPlanner
from visualization.action_contrastive import save_plots


def run_epoch(model, loader, device, settings, optimizer=None, clip=1.0, seed=42):
    training = optimizer is not None
    model.train(training)
    totals, count = defaultdict(float), 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc="Train" if training else "Validation", dynamic_ncols=True)
        for batch in progress:
            frames, actions = batch["frames"].to(device), batch["actions"][:, 0].to(device)
            # Shared E on both sides; gradients flow through O_t and O_next.
            b = len(frames)
            encoded = model.encode_frame(frames.flatten(0, 1)).reshape(b, 2, model.latent_dim)
            candidates = negative_actions(actions, settings["num_negatives"], settings["negative_type"], generator)
            scores = model.candidate_scores(encoded[:, 0], candidates, encoded[:, 1])
            metrics = ranking_metrics(scores, settings["temperature"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            for key, value in metrics.items():
                totals[key] += float(value.detach())*b
            count += b
            progress.set_postfix(loss=f"{totals['loss']/count:.4f}", acc=f"{totals['ranking_acc']/count:.3f}")
    return {key: value/count for key, value in totals.items()}


def checkpoint(path, model, optimizer, scheduler, epoch, config, metrics, history, best_ranking, best_planning):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "observation_encoder": model.observation_encoder.state_dict(),
                "action_encoder": model.action_encoder.state_dict(), "transition_model": model.transition_model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                "config": config, "metrics": metrics, "history": history,
                "best_ranking": best_ranking, "best_planning": best_planning,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


@torch.no_grad()
def planning(model, config, device, output, split):
    if config["data"]["action_scale"] != 100.0:
        raise ValueError("swm relative=True requires action_scale=100; use 100 for planning")
    planner = CEMPlanner(**config["planner"])
    evaluator = PlanningEvaluator(planner, config["environment"], 1, 2, device,
                                  dataset_paths=config["data"][f"{split}_paths"])
    model.eval()
    result = evaluator.evaluate(model, episodes=config["validation"]["planning_episodes"],
                                videos_to_save=config["validation"]["planning_videos"], video_dir=output)
    metrics = {f"planning/{key}": value for key, value in result.items() if isinstance(value, (int, float))}
    # Measure latent-only throughput, excluding rendering/encoding/environment time.
    b, h = config["planner"]["population"], config["planner"]["horizon"]
    o = torch.randn(b, model.latent_dim, device=device)
    a = torch.rand(b, h, 2, device=device)*2-1
    for _ in range(3):
        model.rollout_latents(o, a)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        model.rollout_latents(o, a)
    if device.type == "cuda":
        torch.cuda.synchronize()
    metrics["planning/imagined_transitions_per_second"] = b*h*10/(time.perf_counter()-start)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_contrastive.yaml")
    parser.add_argument("--mode", choices=("train", "evaluate", "plan"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-interval", type=int, help="Validate every N epochs (also on final epoch)")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.validation_interval is not None:
        config["validation"]["interval"] = args.validation_interval
    validation_interval = int(config["validation"].get("interval", 1))
    if validation_interval < 1:
        raise ValueError("validation interval must be positive")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config["device"])
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    # Guard against trajectory leakage, including glob patterns pointing to the same files.
    import glob
    splits = {key: {str(Path(p).resolve()) for pattern in config["data"][f"{key}_paths"]
                    for p in glob.glob(pattern)} for key in ("train", "val", "test")}
    if not all(splits.values()):
        raise ValueError("prepare the dedicated train/val/test dataset before starting")
    if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("train/val/test episode files overlap")
    model_config = {key: value for key, value in config["model"].items() if key != "version"}
    model = ActionContrastiveWorldModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    history, start_epoch, best_ranking, best_planning = [], 1, -1.0, -1.0
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if saved["config"]["model"] != config["model"]:
            raise ValueError("checkpoint model config differs from experiment config")
        model.load_state_dict(saved["model"])
        if args.resume:
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history = saved["history"]
            start_epoch = saved["epoch"] + 1
            best_ranking, best_planning = saved["best_ranking"], saved["best_planning"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    elif args.mode != "train" or args.resume:
        raise ValueError("--checkpoint is required for evaluate, plan, or resume")
    with (output / ("config.yaml" if args.mode == "train" else f"{args.mode}_{args.split}_config.yaml")).open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    logger = ExperimentLogger(config["wandb"], config)
    try:
        if args.mode == "plan":
            metrics = planning(model, config, device, output / f"videos/{args.split}", args.split)
            append_metrics(output / f"planning_{args.split}.jsonl", 0, metrics)
            logger.log(metrics, step=0)
            if args.split == "val" and metrics["planning/success_rate"] > best_planning:
                # Test results never select checkpoints.
                destination = output / "checkpoints/best_planning.pt"
                previous = torch.load(destination, map_location="cpu", weights_only=False).get("best_planning", -1) if destination.exists() else -1
                if metrics["planning/success_rate"] > previous:
                    saved["best_planning"] = metrics["planning/success_rate"]
                    saved["metrics"] = {**saved.get("metrics", {}), **metrics}
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(saved, destination)
            print(json.dumps(metrics, indent=2))
            return
        val_loader, _ = make_loader(config, args.split)
        if args.mode == "evaluate":
            metrics = {f"{args.split}/{k}": v for k, v in run_epoch(model, val_loader, device, config["loss"]).items()}
            extra, observations, metadata = diagnostics(model, config, device, args.split)
            metrics.update(extra)
            save_plots([{"epoch": 0, **metrics}], metrics, observations, metadata, config, output / f"plots/{args.split}")
            append_metrics(output / f"evaluation_{args.split}.jsonl", 0, metrics)
            logger.log(metrics, step=0)
            print(json.dumps(metrics, indent=2))
            return
        train_loader, _ = make_loader(config, "train", shuffle=True)
        # Training always validates on val, even if --split test was passed.
        val_loader, _ = make_loader(config, "val")
        for epoch in range(start_epoch, config["training"]["epochs"]+1):
            print(f"\nAction-Contrastive epoch {epoch}/{config['training']['epochs']}")
            train = run_epoch(model, train_loader, device, config["loss"], optimizer,
                              config["training"]["gradient_clip_norm"], seed+epoch)
            do_validate = epoch % validation_interval == 0 or epoch == config["training"]["epochs"]
            metrics = {f"train/{k}": v for k, v in train.items()}
            val = None
            if do_validate:
                val = run_epoch(model, val_loader, device, config["loss"], seed=seed)
                metrics.update({f"val/{k}": v for k, v in val.items()})
                extra, observations, metadata = diagnostics(model, config, device)
                metrics.update(extra)
            metrics["optimization/learning_rate"] = optimizer.param_groups[0]["lr"]
            do_plan = config["validation"]["planning_enabled"] and epoch % config["validation"]["planning_interval"] == 0
            if do_plan:
                metrics.update(planning(model, config, device, output / f"videos/epoch_{epoch:04d}", "val"))
            history.append({"epoch": epoch, **metrics})
            if do_validate:
                save_plots(history, metrics, observations, metadata, config, output / "plots")
            scheduler.step()
            ranking_improved = val is not None and val["ranking_acc"] > best_ranking
            planning_improved = do_plan and metrics["planning/success_rate"] > best_planning
            if val is not None:
                best_ranking = max(best_ranking, val["ranking_acc"])
            if do_plan:
                best_planning = max(best_planning, metrics["planning/success_rate"])
            names = ["latest.pt"] + (["best_ranking.pt"] if ranking_improved else []) + (["best_planning.pt"] if planning_improved else [])
            for name in names:
                checkpoint(output / "checkpoints" / name, model, optimizer, scheduler, epoch, config,
                           metrics, history, best_ranking, best_planning)
            append_metrics(output / "logs/metrics.jsonl", epoch, metrics)
            images = {f"plots/{p.stem}": logger.wandb.Image(str(p)) for p in (output / "plots").glob("*.png")} if logger.run is not None and do_validate else {}
            logger.log({**metrics, **images}, step=epoch)
            if val is not None:
                print(f"val_acc={val['ranking_acc']:.4f} gap={val['score_gap']:.4f} "
                      f"feature_std={metrics['representation/mean_feature_std']:.4f} "
                      f"O_drop={metrics.get('dependency/O_shuffle_drop', 0):.4f} "
                      f"A_drop={metrics.get('dependency/A_shuffle_drop', 0):.4f}")
            else:
                print(f"train_loss={train['loss']:.4f} train_acc={train['ranking_acc']:.4f}; "
                      f"validation every {validation_interval} epochs")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
