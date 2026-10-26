"""Evaluation-only state supervision on a frozen CSE, including random controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment_tracking import ExperimentLogger
from .data import file_sha256, load_config, loader, validate_splits
from .model import CurrentStateEncoder


@torch.no_grad()
def extract(cse, config, split, limit, device):
    batches, dataset = loader(config, split, horizon=1, limit=limit)
    states = {}
    for i, path in enumerate(dataset.paths):
        with np.load(path) as episode:
            if "states" not in episode or episode["states"].shape[-1] < 5:
                raise ValueError("GT states are required ONLY for this evaluation script")
            states[i] = episode["states"][:, :5].copy()
    features, labels = [], []
    for batch in batches:
        features.append(cse(batch["frames"][:, 0].to(device)).cpu())
        # Simulator state is read here, never returned to main model training.
        raw = torch.from_numpy(np.stack([states[int(ep)][int(t)] for ep, t in zip(batch["episode"], batch["timestep"])]))
        labels.append(torch.cat((raw[:, :4], raw[:, 4:5].sin(), raw[:, 4:5].cos()), -1))
    return torch.cat(features).float(), torch.cat(labels).float()


def prediction_metrics(prediction, target):
    angle_p = torch.atan2(prediction[:, 4], prediction[:, 5])
    angle_y = torch.atan2(target[:, 4], target[:, 5])
    angle_error = torch.atan2((angle_p-angle_y).sin(), (angle_p-angle_y).cos()).abs()
    return {"agent_position_rmse_pixels": float((prediction[:, :2]-target[:, :2]).square().mean().sqrt()),
            "block_position_rmse_pixels": float((prediction[:, 2:4]-target[:, 2:4]).square().mean().sqrt()),
            "block_orientation_mae_degrees": float(angle_error.mean()*180/torch.pi)}


def fit_probes(features, labels, settings, device):
    x_train, x_val, x_test = features
    y_train, y_val, y_test = labels
    # All normalization fit exclusively on probe train features/labels.
    xm, xs = x_train.mean(0), x_train.std(0, correction=0).clamp_min(1e-8)
    ym, ys = y_train.mean(0), y_train.std(0, correction=0).clamp_min(1e-6)
    x = [(v-xm)/xs for v in features]
    y = [(v-ym)/ys for v in labels]
    design = torch.cat((x[0].double(), torch.ones(len(x[0]), 1)), -1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64)*1e-3
    penalty[-1, -1] = 0
    weight = torch.linalg.solve(design.T@design+penalty, design.T@y[0].double())
    predictions = {}
    for split, index in (("val", 1), ("test", 2)):
        values = torch.cat((x[index].double(), torch.ones(len(x[index]), 1)), -1)@weight
        predictions[f"linear/{split}"] = values.float()*ys+ym
    head = nn.Sequential(nn.Linear(x[0].shape[1], settings["hidden_dim"]), nn.SiLU(), nn.Linear(settings["hidden_dim"], 6)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings["learning_rate"])
    train_x, train_y, val_x, val_y = (v.to(device) for v in (x[0], y[0], x[1], y[1]))
    best_loss, best = float("inf"), deepcopy(head.state_dict())
    for _ in range(settings["epochs"]):
        head.train()
        indices = torch.randperm(len(train_x), device=device)
        for chosen in indices.split(settings["batch_size"]):
            loss = (head(train_x[chosen])-train_y[chosen]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            score = float((head(val_x)-val_y).square().mean())
        if score < best_loss:
            best_loss, best = score, deepcopy(head.state_dict())
    head.load_state_dict(best)  # select by validation, never by test
    head.eval()
    with torch.no_grad():
        for split, index in (("val", 1), ("test", 2)):
            predictions[f"mlp/{split}"] = head(x[index].to(device)).cpu()*ys+ym
    metrics = {}
    for name, prediction in predictions.items():
        target = y_val if name.endswith("val") else y_test
        metrics.update({f"{name}/{k}": value for k, value in prediction_metrics(prediction, target).items()})
    for split, target in (("val", y_val), ("test", y_test)):
        metrics.update({f"constant/{split}/{k}": value for k, value in prediction_metrics(ym.expand_as(target), target).items()})
    return metrics, {"linear_weight": weight, "mlp": best, "feature_mean": xm, "feature_std": xs,
                     "label_mean": ym, "label_std": ys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_splits(config)
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cse = CurrentStateEncoder(config["model"]["latent_dim"]).to(device)
    cse.load_state_dict({k.removeprefix("cse."): v for k, v in saved["model"].items() if k.startswith("cse.")})
    cse.requires_grad_(False).eval()
    before = deepcopy(cse.state_dict())
    result, probe_weights = {}, {}
    for name in ("trained_cse", "random_cse"):
        if name == "random_cse":
            cse.load_state_dict(saved["initial_cse"])
        features, labels = [], []
        for split in ("train", "val", "test"):
            limit = config["probe"]["max_train_samples" if split == "train" else "max_eval_samples"]
            x, y = extract(cse, config, split, limit, device)
            features.append(x)
            labels.append(y)
        torch.manual_seed(config["seed"])  # same head initialization for trained/random controls
        metrics, weights = fit_probes(features, labels, config["probe"], device)
        result.update({f"probe/{name}/{k}": value for k, value in metrics.items()})
        probe_weights[name] = weights
    assert not any(p.requires_grad or p.grad is not None for p in cse.parameters())
    cse.load_state_dict(before)
    output = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent/"probes"
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result, indent=2))
    (output/"metadata.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
                                                   "checkpoint_sha256": file_sha256(args.checkpoint)}, indent=2))
    torch.save(probe_weights, output/"heads.pt")
    logger = ExperimentLogger({**config["wandb"], "name": config["wandb"]["name"]+"_probes"}, config)
    logger.log(result, step=0)
    logger.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
