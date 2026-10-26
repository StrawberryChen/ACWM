"""Separate LeWorld-STYLE control: shared random CNN + deterministic MSE + SIGReg.

Not official LeWorld ViT/AdaLN replication. Never imported by the AVS Flow trainer.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from tqdm.auto import tqdm

from experiment_tracking import ExperimentLogger, append_metrics
from losses.latent import SIGRegLoss
from .data import load_config, loader, validate_splits
from .evaluate import collapse_metrics
from .model import CurrentStateEncoder


class LeWorldStyleControl(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.cse = CurrentStateEncoder(latent_dim)
        self.action = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 128))
        self.predictor = nn.Sequential(nn.Linear(latent_dim+128, 256), nn.SiLU(), nn.Linear(256, latent_dim))

    def losses(self, frames, actions, sigreg, weight):
        b, time = frames.shape[:2]
        real = self.cse(frames.flatten(0, 1)).reshape(b, time, -1)
        predicted = self.predictor(torch.cat((real[:, :-1], self.action(actions)), -1))
        prediction = (predicted-real[:, 1:]).square().mean()
        regularization = sigreg(real.transpose(0, 1))
        return prediction+weight*regularization, prediction, regularization, real[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", help="reuse generated Colab data paths without copying the dataset")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_config:
        data_config = load_config(args.data_config)
        config["data"], config["device"] = data_config["data"], data_config["device"]
        config["output_dir"] = str(Path(data_config["output_dir"]).parent/"leworld_style")
    if "baseline" not in config:
        raise ValueError("use avs_leworld_style config; SIGReg is forbidden in the Flow pipeline")
    validate_splits(config)
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    model = LeWorldStyleControl(config["model"]["latent_dim"]).to(device)
    initial = deepcopy(model.cse.cpu().state_dict())
    model.to(device)
    sigreg = SIGRegLoss(config["baseline"]["knots"], config["baseline"]["num_projections"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.)
    output = Path(args.output_dir or config["output_dir"])
    (output/"checkpoints").mkdir(parents=True, exist_ok=True)
    train, _ = loader(config, "train", shuffle=True)
    val, _ = loader(config, "val", limit=config["validation"]["samples"])
    logger = ExperimentLogger(config["wandb"], config)
    best = float("inf")
    try:
        for epoch in range(1, config["training"]["epochs"]+1):
            metrics = {}
            for split, batches in (("train", train), ("val", val)):
                if split == "val" and epoch % config["validation"]["interval"] and epoch != config["training"]["epochs"]:
                    continue
                training = split == "train"
                model.train(training)
                total, count, features = torch.zeros(3, device=device), 0, []
                with torch.set_grad_enabled(training):
                    for batch in tqdm(batches, desc=f"LeWorld-style {split} epoch {epoch}"):
                        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
                        loss, pred, reg, feature = model.losses(frames, actions, sigreg, config["baseline"]["sigreg_weight"])
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                            optimizer.step()
                        total += torch.stack((loss, pred, reg)).detach()*len(frames)
                        count += len(frames)
                        if not training:
                            features.append(feature.detach().cpu())
                for key, value in zip(("loss", "prediction_loss", "sigreg_loss"), (total/count).tolist()):
                    metrics[f"{split}/{key}"] = value
                if not training:
                    metrics.update(collapse_metrics(torch.cat(features)))
            checkpoint = dict(model=model.state_dict(), config=config, epoch=epoch, optimizer=optimizer.state_dict(),
                              initial_cse=initial, metrics=metrics)
            torch.save(checkpoint, output/"checkpoints/latest.pt")
            if "val/prediction_loss" in metrics and metrics["val/prediction_loss"] < best:
                best = metrics["val/prediction_loss"]
                torch.save(checkpoint, output/"checkpoints/best_prediction.pt")
            append_metrics(output/"metrics.jsonl", epoch, metrics)
            logger.log(metrics, step=epoch)
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
