"""Frozen-encoder state probe ONLY after world-model training."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch

from experiment_tracking import ExperimentLogger
from experiments.avs_target.probe import fit_probes
from experiments.avs_target.data import file_sha256
from .data import load_config, loader, validate_splits
from .model import ContinuousWorldModel
from .train import load_checkpoint


@torch.no_grad()
def extract(encoder, config, split, limit, device):
    import numpy as np
    batches, dataset = loader(config, split, limit=limit)
    labels = {}
    for i, path in enumerate(dataset.paths):
        with np.load(path) as data:
            if "states" not in data or data["states"].shape[-1] < 5:
                raise ValueError("probe needs states [T,>=5]; world-model training does not")
            labels[i] = data["states"][:, :5].copy()
    features, targets = [], []
    for batch in batches:
        features.append(encoder(batch["frames"][:, 0].to(device)).cpu())
        raw = torch.from_numpy(np.stack([labels[int(ep)][int(t)] for ep, t in zip(batch["episode"], batch["timestep"])]))
        targets.append(torch.cat((raw[:, :4], raw[:, 4:5].sin(), raw[:, 4:5].cos()), -1))
    return torch.cat(features).float(), torch.cat(targets).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="saved training config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    if args.no_wandb:
        config["wandb"]["enabled"] = False
    validate_splits(config)
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    model = ContinuousWorldModel(**config["model"]).to(device)
    saved = load_checkpoint(args.checkpoint, config, model)
    encoder = model.encoder.requires_grad_(False).eval()
    before = deepcopy(encoder.state_dict())
    results, weights = {}, {}
    for name in ("trained_encoder", "random_encoder"):
        if name == "random_encoder":
            encoder.load_state_dict(saved["initial_encoder"])
        features, labels = [], []
        for split in ("train", "val", "test"):
            limit = config["probe"]["max_train_samples" if split == "train" else "max_eval_samples"]
            x, y = extract(encoder, config, split, limit, device)
            features.append(x)
            labels.append(y)
        torch.manual_seed(config["seed"])
        metrics, state = fit_probes(features, labels, config["probe"], device)
        results.update({f"probe/{name}/{k}": v for k, v in metrics.items()})
        weights[name] = state
    encoder.load_state_dict(before)
    assert not any(p.requires_grad or p.grad is not None for p in encoder.parameters())
    output = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent/"probes"/Path(args.checkpoint).stem
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(results, indent=2))
    (output/"metadata.json").write_text(json.dumps(dict(checkpoint_sha256=file_sha256(args.checkpoint),
                                                       note="angle decoded as sin/cos; position error in pixels"), indent=2))
    torch.save(weights, output/"heads.pt")
    logger = ExperimentLogger({**config["wandb"], "name": config["wandb"]["name"]+"_probe"}, config)
    try:
        logger.log(results, step=0)
    finally:
        logger.finish()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
