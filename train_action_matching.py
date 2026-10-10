"""Independent V2 training; five-epoch validation and one final CEM/MPC evaluation."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import glob
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from evaluation.action_matching import branch_loader, diagnostics
from experiment_tracking import ExperimentLogger, append_metrics
from losses.action_matching import compute_matching_loss
from losses.latent import SIGRegLoss
from models.action_matching import ActionMatchingWorldModel
from train_action_contrastive import planning as cem_evaluation
from visualization.action_matching import save_plots


@contextmanager
def evaluation_seed(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def run_epoch(model, loader, sigreg, config, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals, denominators = defaultdict(float), defaultdict(float)
    info_keys = {"matching_acc", "mean_rank", "score_gap"}
    rows = 0
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc="V2 train" if training else "V2 validation", dynamic_ncols=True)
        for raw in progress:
            batch = {key: value.to(device) for key, value in raw.items()}
            metrics, _ = compute_matching_loss(model, batch, sigreg, config["loss"])
            if not torch.isfinite(metrics["loss"]):
                raise FloatingPointError("non-finite V2 total loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                optimizer.step()
            n = len(batch["current_frame"])
            rows += int(metrics["informative_rows"])
            for key, value in metrics.items():
                if key == "informative_rows":
                    continue
                value = float(value.detach())
                denominator = int(metrics["informative_rows"]) if key in info_keys else n
                if denominator and math.isfinite(value):
                    totals[key] += value*denominator
                    denominators[key] += denominator
            progress.set_postfix(loss=f"{totals['loss']/denominators['loss']:.4f}",
                                 match=f"{totals['matching_loss']/denominators['matching_loss']:.4f}")
    if not denominators["loss"]:
        raise ValueError("empty V2 loader")
    result = {key: value/denominators[key] for key, value in totals.items()}
    result["informative_rows"] = rows
    # Undefined accuracy is omitted, not logged as a misleading zero/one.
    return result


def save_checkpoint(path, model, sigreg, optimizer, scheduler, config, epoch, metrics, history, best_match, best_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "observation_encoder": model.observation_encoder.state_dict(),
                "action_encoder": model.action_encoder.state_dict(), "transition_model": model.transition_model.state_dict(),
                "uncertainty_head": model.uncertainty_head.state_dict(), "sigreg": sigreg.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch,
                "config": config, "metrics": metrics, "history": history,
                "best_matching_loss": best_match, "best_total_loss": best_loss,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, path)


def verify_splits(config, datasets):
    ids = {split: dataset.episode_ids for split, dataset in datasets.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"source episode leakage between {a} and {b}")
    for split in ids:
        paths = sorted({p for pattern in config["data"][f"{split}_paths"] for p in glob.glob(pattern)})
        if not paths:
            raise ValueError(f"no {split} trajectories: rerun V2 preparation")
        trajectory_ids = set()
        for path in paths:
            with np.load(path) as data:
                trajectory_ids.add(int(data["source_episode"]))
        if trajectory_ids != ids[split]:
            raise ValueError("branch and sequential trajectory source episodes do not match")


def plan_and_save(model, config, device, output, selected_path, split, logger, log_step):
    # Selected ONLY by held-out validation before running test environment episodes.
    saved = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    print(f"Planning checkpoint: {selected_path} (training epoch {saved['epoch']}); split={split}")
    metrics = cem_evaluation(model, config, device, output/f"videos/final_{split}", split)
    append_metrics(output/f"logs/planning_{split}.jsonl", saved["epoch"], metrics)
    logger.log({**metrics, "planning/checkpoint_epoch": saved["epoch"]}, step=log_step)
    # Test metrics are attached for provenance, never used to choose 'best'.
    saved["planning_evaluation"] = {"split": split, "metrics": metrics, "checkpoint_source": str(selected_path)}
    (output/"checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(saved, output/"checkpoints/planning_evaluated.pt")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_matching_v2.yaml")
    parser.add_argument("--mode", choices=("train", "evaluate", "plan"), default="train")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--validation-interval", type=int)
    parser.add_argument("--skip-final-planning", action="store_true")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.validation_interval is not None:
        config["validation"]["interval"] = args.validation_interval
    if config["validation"]["interval"] < 1:
        raise ValueError("validation interval must be positive")
    if config["model"]["version"] != "action_matching_v2":
        raise ValueError("use the independent action_matching_v2 config")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Colab")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model = ActionMatchingWorldModel(**{key: value for key, value in config["model"].items() if key != "version"}).to(device)
    sigreg = SIGRegLoss(config["loss"]["sigreg_knots"], config["loss"]["sigreg_projections"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["training"]["epochs"])
    loaders, datasets = {}, {}
    for split in ("train", "val", "test"):
        loaders[split], datasets[split] = branch_loader(config, split, shuffle=split == "train")
    verify_splits(config, datasets)
    start_epoch, history, best_match, best_loss = 1, [], float("inf"), float("inf")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if saved["config"]["model"] != config["model"]:
            raise ValueError("checkpoint model config differs")
        model.load_state_dict(saved["model"])
        sigreg.load_state_dict(saved["sigreg"])
        if args.resume:
            if saved["config"]["loss"] != config["loss"]:
                raise ValueError("changed loss configuration requires a new experiment")
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scheduler.T_max = config["training"]["epochs"]
            start_epoch, history = saved["epoch"]+1, saved["history"]
            best_match, best_loss = saved["best_matching_loss"], saved["best_total_loss"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if device.type == "cuda" and saved.get("cuda_rng"):
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    elif args.resume or args.mode != "train":
        raise ValueError("--checkpoint is required")
    filename = "config.yaml" if args.mode == "train" else f"{args.mode}_{args.split}_config.yaml"
    with (output/filename).open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    logger = ExperimentLogger(config["wandb"], config)
    try:
        if args.mode == "plan":
            plan_and_save(model, config, device, output, Path(args.checkpoint), args.split, logger, 0)
            return
        if args.mode == "evaluate":
            with evaluation_seed(config["seed"]+100, device):
                val = run_epoch(model, loaders[args.split], sigreg, config, device)
                extra, observations, metadata = diagnostics(model, config, device, args.split)
            metrics = {**{f"{args.split}/{k}": v for k, v in val.items()}, **extra}
            save_plots([{"epoch": 0, **metrics}], metrics, observations, metadata, config, output/f"plots/{args.split}")
            append_metrics(output/f"logs/evaluation_{args.split}.jsonl", 0, metrics)
            logger.log(metrics, step=0)
            print(json.dumps(metrics, indent=2))
            return
        for epoch in range(start_epoch, config["training"]["epochs"]+1):
            print(f"\nAction Matching V2 epoch {epoch}/{config['training']['epochs']}")
            train = run_epoch(model, loaders["train"], sigreg, config, device, optimizer)
            metrics = {f"train/{k}": v for k, v in train.items()}
            validate = epoch % config["validation"]["interval"] == 0 or epoch == config["training"]["epochs"]
            improved_match = improved_total = False
            if validate:
                with evaluation_seed(config["seed"]+100, device):
                    val = run_epoch(model, loaders["val"], sigreg, config, device)
                    extra, observations, metadata = diagnostics(model, config, device)
                metrics.update({f"val/{k}": v for k, v in val.items()})
                metrics.update(extra)
                improved_match, improved_total = val["matching_loss"] < best_match, val["loss"] < best_loss
                best_match, best_loss = min(best_match, val["matching_loss"]), min(best_loss, val["loss"])
            metrics["optimization/learning_rate"] = optimizer.param_groups[0]["lr"]
            history.append({"epoch": epoch, **metrics})
            if validate:
                save_plots(history, metrics, observations, metadata, config, output/"plots")
            scheduler.step()
            names = ["latest.pt"]+(["best_matching.pt"] if improved_match else [])+(["best_loss.pt"] if improved_total else [])
            for name in names:
                save_checkpoint(output/"checkpoints"/name, model, sigreg, optimizer, scheduler, config,
                                epoch, metrics, history, best_match, best_loss)
            append_metrics(output/"logs/metrics.jsonl", epoch, metrics)
            images = {f"plots/{p.stem}": logger.wandb.Image(str(p)) for p in (output/"plots").glob("*.png")} if validate and logger.run is not None else {}
            logger.log({**metrics, **images}, step=epoch)
            print(" ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key in
                           {"train/loss", "val/loss", "val/nll_loss", "val/matching_acc", "val/positive_cosine", "retrieval/top1_h1"}))
        print("Training complete; weights saved.")
        if config["validation"]["planning_after_training"] and not args.skip_final_planning:
            print("Starting final planning evaluation.")
            selection = config["validation"]["planning_checkpoint"]
            if selection not in {"best_matching", "best_loss", "latest"}:
                raise ValueError("invalid planning_checkpoint")
            plan_and_save(model, config, device, output, output/f"checkpoints/{selection}.pt",
                          config["validation"]["planning_split"], logger, config["training"]["epochs"]+1)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
