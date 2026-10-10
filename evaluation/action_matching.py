"""V2 conditional matching dependency and cosine rollout retrieval diagnostics."""
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataset.action_matching import ActionBranchDataset
from evaluation.action_contrastive import make_loader as trajectory_loader, representation_metrics
from models.action_contrastive import ranking_metrics
from models.action_matching import matching_metrics


def branch_loader(config, split, limit=None, shuffle=False):
    dataset = ActionBranchDataset(config["data"][f"branch_{split}_paths"], config["model"]["image_size"])
    if limit and len(dataset) > limit:
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]))[:limit].tolist()
        chosen = Subset(dataset, indices)
    else:
        chosen = dataset
    loader = DataLoader(chosen, batch_size=config["data"]["batch_size"], shuffle=shuffle,
                        num_workers=config["data"]["num_workers"], pin_memory=config["device"].startswith("cuda"))
    return loader, dataset


@torch.no_grad()
def diagnostics(model, config, device, split="val"):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config["seed"]+100)
    loader, _ = branch_loader(config, split, config["validation"]["diagnostic_samples"])
    totals, count, observations, metadata = defaultdict(float), 0, [], []
    for batch in loader:
        current = model.encode_frame(batch["current_frame"].to(device))
        futures = batch["future_frames"].to(device)
        targets = model.encode_frame(futures.flatten(0, 1)).reshape(len(current), futures.shape[1], -1)
        actions, mask = batch["actions"].to(device), batch["positive_mask"].to(device)
        n = len(current)
        observations.append(current.cpu())
        metadata.append(torch.stack((batch["timestep"], batch["episode"],
                                     torch.atan2(actions[:, 0, 1], actions[:, 0, 0]).cpu()), -1))
        if n > 1:
            shift = int(torch.randint(1, n, (), generator=generator, device=device))
            shuffled = torch.arange(n, device=device).roll(shift)
            outputs = {}
            for name, o, a in (("normal", current, actions), ("O_shuffle", current[shuffled], actions),
                               ("A_shuffle", current, actions[shuffled])):
                mean, sigma = model.branch_distribution(o, a)
                scores = model.matching_scores(mean, sigma, targets)
                outputs[name] = matching_metrics(scores, mask, config["loss"]["matching_temperature"])
            valid = int(outputs["normal"]["informative_rows"])
            if valid:
                for name, result in outputs.items():
                    totals[f"dependency/{name}_acc"] += float(result["matching_acc"])*valid
                totals["dependency/O_shuffle_drop"] += float(outputs["normal"]["matching_acc"]-outputs["O_shuffle"]["matching_acc"])*valid
                totals["dependency/A_shuffle_drop"] += float(outputs["normal"]["matching_acc"]-outputs["A_shuffle"]["matching_acc"])*valid
                count += valid
    metrics = {key: value/count for key, value in totals.items()} if count else {}
    metrics["dependency/informative_rows"] = count
    observations = torch.cat(observations)
    metrics.update(representation_metrics(observations))
    settings = config["validation"]
    horizons = settings["retrieval_horizons"]
    loader, _ = trajectory_loader(config, split, horizon=max(horizons), limit=settings["retrieval_samples"])
    starts, actions, targets = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        starts.append(model.encode_frame(frames[:, 0]))
        actions.append(batch["actions"].to(device))
        targets.append(torch.stack([model.encode_frame(frames[:, h]) for h in horizons], 1))
    starts, actions, targets = map(torch.cat, (starts, actions, targets))
    n = len(starts)
    k = min(settings["retrieval_candidates"], n)
    if k < 2:
        raise ValueError("at least two retrieval samples are required")
    predictions = model.rollout_latents(starts, actions)
    shuffled_predictions = model.rollout_latents(starts, actions.roll(1, 0))
    random = torch.rand(n, n, device=device, generator=generator)
    random.fill_diagonal_(2)
    candidates = torch.cat((torch.arange(n, device=device)[:, None], random.topk(k-1, largest=False).indices), 1)
    for index, h in enumerate(horizons):
        target_candidates = targets[candidates, index]
        for name, query in (("", predictions[:, h-1]), ("identity_", starts),
                            ("A_shuffle_", shuffled_predictions[:, h-1])):
            scores = F.cosine_similarity(query[:, None], target_candidates, dim=-1)
            result = ranking_metrics(scores)
            metrics[f"retrieval/{name}top1_h{h}"] = float(result["ranking_acc"])
            metrics[f"retrieval/{name}mean_rank_h{h}"] = float(result["mean_rank"])
        metrics[f"rollout/positive_cosine_h{h}"] = float(F.cosine_similarity(predictions[:, h-1], targets[:, index], dim=-1).mean())
    metrics.update({"retrieval/candidates": k, "retrieval/samples": n})
    return metrics, observations, torch.cat(metadata)
