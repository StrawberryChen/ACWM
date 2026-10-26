"""Held-out fixed-target generation, dependency and collapse diagnostics."""
from collections import defaultdict

import torch
from torch.nn import functional as F

from .flow import euler_sample, flow_loss, flow_training_pair


def collapse_metrics(features):
    # Samples [N,D]; effective rank uses covariance eigenvalues, not their square roots.
    centered = features.double()-features.double().mean(0)
    eigenvalues = torch.linalg.svdvals(centered).square()/max(len(features)-1, 1)
    total = eigenvalues.sum()
    probabilities = eigenvalues/total.clamp_min(1e-30)
    rank = torch.exp(-(probabilities*probabilities.clamp_min(1e-30).log()).sum()) if total > 1e-30 else total*0
    normalized = F.normalize(features.double(), dim=-1)
    similarity = normalized@normalized.T
    mask = ~torch.eye(len(features), device=features.device, dtype=torch.bool)
    return {"representation/feature_std": float(centered.std(0, correction=0).mean()),
            "representation/effective_rank": float(rank),
            "representation/pairwise_cosine": float(similarity[mask].mean()) if mask.any() else 0.,
            "representation/state_variance": float(centered.square().sum(-1).mean())}


def motion_metrics(predicted, target, operator, threshold):
    # Report normalized MSE plus denormalized physical detector-space diagnostics.
    p, y = operator.maps(operator.denormalize(predicted)), operator.maps(operator.denormalize(target))
    result = {"mse": float((predicted-target).square().mean()),
              "raw_mse": float((p-y).square().mean()),
              "cosine": float(F.cosine_similarity(predicted.flatten(1), target.flatten(1), dim=-1).mean())}
    if operator.kind == "avs":
        valid = y.sum(-3) > threshold
        correct = p.argmax(-3) == y.argmax(-3)
        result["direction_accuracy"] = float(correct[valid].float().mean()) if valid.any() else None
        result["direction_valid_pixels"] = int(valid.sum())
        result["magnitude_error"] = float((p.sum(-3)-y.sum(-3)).abs().mean())
    else:
        result["magnitude_error"] = float((p.abs().mean(-3)-y.abs().mean(-3)).abs().mean())
    return result


@torch.no_grad()
def evaluate(model, operator, loader, config, device, seed=100):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    totals, count, features = defaultdict(float), 0, []
    raw_sum, raw_signed_sum, raw_square, raw_count, zero_count, channel_sum = 0., 0., 0., 0, 0, None
    generated_inputs = {k: [] for k in ("state", "actions", "target", "noise", "frames")}
    remaining = config["validation"]["generation_samples"]
    dependency_samples = 0
    threshold = config["validation"]["zero_motion_threshold"]
    for batch in loader:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        target = operator(frames)  # fixed no-grad real trajectory, includes transition zero
        state = model.cse(frames[:, 0])
        features.append(state.cpu())
        noise = torch.randn(target.shape, generator=generator, device=device)
        time = torch.rand(len(target), generator=generator, device=device)
        noisy, velocity = flow_training_pair(target, noise, time)
        prediction = model.predictor(noisy, time, state, actions)
        loss = flow_loss(prediction, velocity)
        n = len(target)
        totals["flow_loss"] += float(loss)*n
        count += n
        if n > 1:
            shift = int(torch.randint(1, n, (), generator=generator, device=device))
            permutation = torch.arange(n, device=device).roll(shift)
            normal = float(loss)
            shuffled_o = float(flow_loss(model.predictor(noisy, time, state[permutation], actions), velocity))
            shuffled_a = float(flow_loss(model.predictor(noisy, time, state, actions[permutation]), velocity))
            totals["state_delta"] += (shuffled_o-normal)*n
            totals["action_delta"] += (shuffled_a-normal)*n
            # Low flow time carries less target leakage through X_s; same noise and
            # times for all three conditions. This is analysis, never an added loss.
            for s in (0.0, 0.1, 0.5, 0.9):
                fixed_time = time.new_full((n,), s)
                x_s, u = flow_training_pair(target, noise, fixed_time)
                base = float(flow_loss(model.predictor(x_s, fixed_time, state, actions), u))
                o = float(flow_loss(model.predictor(x_s, fixed_time, state[permutation], actions), u))
                a = float(flow_loss(model.predictor(x_s, fixed_time, state, actions[permutation]), u))
                for key, value in (("normal", base), ("state_delta", o-base), ("action_delta", a-base)):
                    totals[f"s{s:g}/{key}"] += value*n
            dependency_samples += n
        raw = operator.maps(operator.denormalize(target))
        raw_sum += float(raw.abs().sum())
        raw_signed_sum += float(raw.sum())
        raw_square += float(raw.square().sum())
        raw_count += raw.numel()
        zero_count += int((raw.abs() <= threshold).sum())
        channel_energy = raw.abs().sum((0, 1, 3, 4)).cpu()
        channel_sum = channel_energy if channel_sum is None else channel_sum+channel_energy
        if remaining > 0:
            keep = min(remaining, n)
            for key, value in (("state", state), ("actions", actions), ("target", target),
                               ("noise", noise), ("frames", frames)):
                generated_inputs[key].append(value[:keep].cpu())
            remaining -= keep
    if count == 0:
        raise ValueError("empty evaluation split")
    result = {"flow_loss": totals["flow_loss"]/count, **collapse_metrics(torch.cat(features))}
    result.update({"dependency/state_shuffle_delta": totals["state_delta"]/max(dependency_samples, 1),
                   "dependency/action_shuffle_delta": totals["action_delta"]/max(dependency_samples, 1),
                   "dependency/samples": dependency_samples,
                   "avs/motion_energy": raw_sum/raw_count,
                   "avs/motion_second_moment": raw_square/raw_count,
                   "avs/motion_variance": max(0., raw_square/raw_count-(raw_signed_sum/raw_count)**2),
                   "avs/zero_motion_ratio": zero_count/raw_count})
    # Variance is computed below from raw targets, not normalized values.
    all_inputs = {k: torch.cat(v) for k, v in generated_inputs.items()}
    for key, value in totals.items():
        if key.startswith("s"):
            result[f"dependency/{key}"] = value/max(dependency_samples, 1)
    if channel_sum is not None:
        distribution = channel_sum/channel_sum.sum().clamp_min(1e-30)
        for channel, value in enumerate(distribution):
            result[f"avs/channel_{channel}_fraction"] = float(value)
    state, actions, target, noise = (all_inputs[k].to(device) for k in ("state", "actions", "target", "noise"))
    horizons = [h for h in config["validation"]["horizons"] if h <= actions.shape[1]]
    samples = {}
    for steps in config["validation"]["flow_steps"]:
        for horizon in horizons:
            predicted = euler_sample(model.predictor, state, actions[:, :horizon], noise[:, :horizon], steps)
            measured = motion_metrics(predicted, target[:, :horizon], operator, threshold)
            result.update({f"generation/k{steps}/h{horizon}/{k}": v for k, v in measured.items() if v is not None})
            if horizon == actions.shape[1]:
                samples[steps] = predicted.cpu()
    reference_steps = max(samples)
    reference = samples[reference_steps].to(device)
    for name in ("mse", "cosine", "magnitude_error", "direction_accuracy"):
        key = f"generation/k{reference_steps}/h{actions.shape[1]}/{name}"
        if key in result:
            result[f"generation/{name}"] = result[key]
    if len(state) > 1:
        for name, o, a in (("state", state.roll(1, 0), actions), ("action", state, actions.roll(1, 0))):
            shuffled = euler_sample(model.predictor, o, a, noise, reference_steps)
            result[f"generation/{name}_shuffle_mse_delta"] = float(
                (shuffled-target).square().mean()-(reference-target).square().mean())
    all_inputs["prediction"] = samples[reference_steps]
    return result, all_inputs
