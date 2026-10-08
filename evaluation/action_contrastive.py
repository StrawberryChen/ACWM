"""Ranking, dependency, angular generalization, and future retrieval diagnostics."""
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from dataset.action_contrastive import ActionContrastiveDataset
from models.action_contrastive import negative_actions, ranking_metrics


def make_loader(config, split, horizon=1, limit=None, seed=None, shuffle=False):
    data = config["data"]
    dataset = ActionContrastiveDataset(data[f"{split}_paths"], config["model"]["image_size"],
                                      horizon, data.get("min_action_norm", 0.0))
    if limit and len(dataset) > limit:
        g = torch.Generator().manual_seed(config["seed"] if seed is None else seed)
        dataset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:limit].tolist())
    return DataLoader(dataset, batch_size=data["batch_size"], shuffle=shuffle,
                      num_workers=data.get("num_workers", 0), pin_memory=config["device"].startswith("cuda")), dataset


def representation_metrics(observations):
    # observations [N,D]; covariance spectrum estimated on centered features.
    centered = observations.double() - observations.double().mean(0)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    rank = (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp()
    if singular.sum() < 1e-10:
        rank = torch.zeros_like(rank)
    normalized = F.normalize(observations.double(), dim=-1)
    n = len(normalized)
    pairwise = ((normalized.sum(0).square().sum() - normalized.square().sum()) / max(1, n*(n-1)))
    return {"representation/mean_feature_std": float(observations.std(0, unbiased=False).mean()),
            "representation/effective_rank": float(rank),
            "representation/mean_pairwise_cosine": float(pairwise)}


@torch.no_grad()
def diagnostics(model, config, device, split="val"):
    # Fixed evaluation RNG: compare epochs using the same counterfactuals and samples.
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 100)
    settings = config["validation"]
    loader, _ = make_loader(config, split, limit=settings["diagnostic_samples"])
    totals, observations, metadata = defaultdict(float), [], []
    count = 0
    model.eval()
    for batch in loader:
        images, action = batch["frames"].to(device), batch["actions"][:, 0].to(device)
        o, target = model.encode_frame(images[:, 0]), model.encode_frame(images[:, 1])
        candidates = negative_actions(action, config["loss"]["num_negatives"], config["loss"]["negative_type"], generator)
        normal = ranking_metrics(model.candidate_scores(o, candidates, target), config["loss"]["temperature"])
        n = len(o)
        # A nonzero cyclic shift has no fixed points, unlike a plain random permutation.
        shift = int(torch.randint(1, n, (), device=device, generator=generator)) if n > 1 else 0
        permutation = torch.arange(n, device=device).roll(shift)
        if n > 1:
            shuffled_o = ranking_metrics(model.candidate_scores(o[permutation], candidates, target))
            shuffled_a = ranking_metrics(model.candidate_scores(o, candidates[permutation], target))
            for key, value in {"O_shuffle_acc": shuffled_o["ranking_acc"],
                               "A_shuffle_acc": shuffled_a["ranking_acc"],
                               "normal_acc": normal["ranking_acc"],
                               "O_shuffle_drop": normal["ranking_acc"]-shuffled_o["ranking_acc"],
                               "A_shuffle_drop": normal["ranking_acc"]-shuffled_a["ranking_acc"]}.items():
                totals[f"dependency/{key}"] += float(value)*n
            totals["dependency/samples"] += n
        for angle in settings["angles"]:
            # Restrict angular probes to the unit disk so every rotated action
            # remains in [-1,1]^2 with exactly the original magnitude and angle.
            eligible = action.norm(dim=-1) <= 1
            probe = action[eligible]
            if not len(probe):
                continue
            radians = torch.tensor(angle*torch.pi/180, device=device)
            x, y = probe.unbind(-1)
            rotated = torch.stack((radians.cos()*x-radians.sin()*y,
                                   radians.sin()*x+radians.cos()*y), -1)
            result = ranking_metrics(model.candidate_scores(o[eligible], torch.stack((probe, rotated), 1), target[eligible]))
            totals[f"angle/{angle}/ranking_acc"] += float(result["ranking_acc"])*len(probe)
            totals[f"angle/{angle}/score_gap"] += float(result["score_gap"])*len(probe)
            totals[f"angle/{angle}/samples"] += len(probe)
        totals["diagnostics/zero_action_fraction"] += float((action.norm(dim=-1) < 1e-6).sum())
        count += n
        observations.append(o.cpu())
        metadata.append(torch.stack((batch["timestep"], batch["episode"],
                                    torch.atan2(action[:, 1], action[:, 0]).cpu()), -1))
    dependency_count = totals.pop("dependency/samples", 0)
    metrics = {}
    for key, value in totals.items():
        denominator = count
        if key.startswith("dependency/"):
            denominator = dependency_count
        elif key.startswith("angle/"):
            denominator = 1 if key.endswith("/samples") else totals[f"{key.rsplit('/', 1)[0]}/samples"]
        metrics[key] = value / max(1, denominator)
    for angle in settings["angles"]:
        metrics.setdefault(f"angle/{angle}/samples", 0)
        for key in ("ranking_acc", "score_gap"):
            metrics.setdefault(f"angle/{angle}/{key}", float("nan"))
    o = torch.cat(observations)
    metrics.update(representation_metrics(o))
    # Retrieval uses distinct examples from held-out episodes; all horizons use
    # the same start indices and candidate bank to make the horizon curve comparable.
    horizons = settings["retrieval_horizons"]
    loader, dataset = make_loader(config, split, horizon=max(horizons), limit=settings["retrieval_samples"])
    starts, action_sequences, targets = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        starts.append(model.encode_frame(frames[:, 0]))
        action_sequences.append(batch["actions"].to(device))
        targets.append(torch.stack([model.encode_frame(frames[:, h]) for h in horizons], 1))
    starts, action_sequences, targets = map(torch.cat, (starts, action_sequences, targets))
    predictions = model.rollout_latents(starts, action_sequences)
    n = len(starts)
    k = min(settings["retrieval_candidates"], n)
    if k < 2:
        raise ValueError("future retrieval requires at least two distinct transitions")
    # Sample without replacement per query; never include the correct row as a distractor.
    random_keys = torch.rand(n, n, device=device, generator=generator)
    random_keys.fill_diagonal_(2)
    distractors = random_keys.topk(k-1, largest=False).indices
    indices = torch.cat((torch.arange(n, device=device)[:, None], distractors), 1)
    for i, h in enumerate(horizons):
        scores = F.cosine_similarity(predictions[:, h-1, None], targets[indices, i], dim=-1)
        result = ranking_metrics(scores)
        metrics[f"retrieval/top1_h{h}"] = float(result["ranking_acc"])
        metrics[f"retrieval/mean_rank_h{h}"] = float(result["mean_rank"])
    metrics["retrieval/candidates"] = k
    metrics["retrieval/samples"] = n
    return metrics, o, torch.cat(metadata)
