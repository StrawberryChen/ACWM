"""Prediction-first evaluation for terminal-only and joint fixed measurements."""
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from .evaluate import collapse_metrics, motion_metrics
from .joint_model import joint_euler_sample, joint_flow_loss, joint_training_pair


@torch.no_grad()
def targets_for(frames, terminal, motion=None):
    # The terminal is X_t+H, not X_t or the first future frame.
    result = {"terminal": terminal(frames[:, -1])}
    if motion is not None:
        result["motion"] = motion(frames)
    return result


def noise_for(targets, generator):
    return {name: torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
            for name, value in targets.items()}


@torch.no_grad()
def evaluate_joint(model, terminal, motion, batches, config, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config["seed"]+100)
    totals, count, dependency_count, features = defaultdict(float), 0, 0, []
    examples = {"frames": [], "actions": [], "state": [], "noise_terminal": []}
    if motion is not None:
        examples["noise_motion"] = []
    remaining = config["validation"]["generation_samples"]
    motion_count, channel_energy = 0, None
    for batch in batches:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        state = model.cse(frames[:, 0])
        targets = targets_for(frames, terminal, motion)
        if motion is not None:
            raw = motion.maps(motion.denormalize(targets["motion"]))
            totals["raw_motion_abs"] += float(raw.abs().sum())
            totals["raw_motion_sum"] += float(raw.sum())
            totals["raw_motion_square"] += float(raw.square().sum())
            totals["raw_motion_zero"] += int((raw.abs() <= config["validation"]["zero_motion_threshold"]).sum())
            motion_count += raw.numel()
            energy = raw.abs().sum((0, 1, 3, 4)).cpu()
            channel_energy = energy if channel_energy is None else channel_energy+energy
        noise = noise_for(targets, generator)
        time = torch.rand(len(frames), device=device, generator=generator)
        noisy, velocity = joint_training_pair(targets, noise, time)
        normal = joint_flow_loss(model.predictor(noisy, time, state, actions), velocity,
                                 config["loss"]["lambda_motion"], config["loss"]["lambda_terminal"])
        for key, value in normal.items():
            totals[key] += float(value)*len(frames)
        count += len(frames)
        features.append(state.cpu())
        if len(frames) > 1:
            permutation = torch.arange(len(frames), device=device).roll(1)
            for s in (None, 0.0, 0.1, 0.5, 0.9):
                x, v, times = noisy, velocity, time
                if s is not None:
                    times = time.new_full(time.shape, s)
                    x, v = joint_training_pair(targets, noise, times)
                metrics = {}
                for name, o, a in (("normal", state, actions), ("state", state[permutation], actions),
                                   ("action", state, actions[permutation])):
                    metrics[name] = joint_flow_loss(model.predictor(x, times, o, a), v,
                        config["loss"]["lambda_motion"], config["loss"]["lambda_terminal"])
                prefix = "dependency" if s is None else f"dependency/s{s:g}"
                for name in ("state", "action"):
                    for branch in ("loss", "terminal_flow_loss", "motion_flow_loss"):
                        totals[f"{prefix}/{name}_shuffle_{branch}_delta"] += float(metrics[name][branch]-metrics["normal"][branch])*len(frames)
            dependency_count += len(frames)
        if remaining:
            n = min(remaining, len(frames))
            for name, value in (("frames", frames), ("actions", actions), ("state", state)):
                examples[name].append(value[:n].cpu())
            for name, value in noise.items():
                examples[f"noise_{name}"].append(value[:n].cpu())
            remaining -= n
    if count == 0:
        raise ValueError("empty validation")
    result = {key: value/(dependency_count if key.startswith("dependency/") else count)
              for key, value in totals.items() if not key.startswith("raw_motion_")
              and (not key.startswith("dependency/") or dependency_count)}
    if motion_count:
        result["avs/motion_energy"] = totals["raw_motion_abs"]/motion_count
        result["avs/motion_variance"] = max(0., totals["raw_motion_square"]/motion_count-(totals["raw_motion_sum"]/motion_count)**2)
        result["avs/zero_motion_ratio"] = totals["raw_motion_zero"]/motion_count
        for index, value in enumerate(channel_energy/channel_energy.sum().clamp_min(1e-30)):
            result[f"avs/channel_{index}_fraction"] = float(value)
    result.update(collapse_metrics(torch.cat(features)))
    result["dependency/samples"] = dependency_count
    result["dependency/state_shuffle_delta"] = result.get("dependency/state_shuffle_loss_delta", 0.)
    result["dependency/action_shuffle_delta"] = result.get("dependency/action_shuffle_loss_delta", 0.)
    examples = {key: torch.cat(value).to(device) for key, value in examples.items()}
    frames, actions, state = examples["frames"], examples["actions"], examples["state"]
    samples = {}
    max_steps = max(config["validation"]["flow_steps"])
    full_horizon = actions.shape[1]
    for steps in config["validation"]["flow_steps"]:
        for horizon in config["validation"]["horizons"]:
            if horizon > full_horizon:
                continue
            noise = {"terminal": examples["noise_terminal"]}
            if motion is not None:
                noise["motion"] = examples["noise_motion"][:, :horizon]
            predicted = joint_euler_sample(model.predictor, state, actions[:, :horizon], noise, steps)
            actual = targets_for(frames[:, :horizon+1], terminal, motion)
            prefix = f"generation/k{steps}/h{horizon}"
            result[f"{prefix}/terminal_mse"] = float((predicted["terminal"]-actual["terminal"]).square().mean())
            result[f"{prefix}/terminal_cosine"] = float(F.cosine_similarity(predicted["terminal"], actual["terminal"], dim=-1).mean())
            if motion is not None:
                measured = motion_metrics(predicted["motion"], actual["motion"], motion, config["validation"]["zero_motion_threshold"])
                result.update({f"{prefix}/motion_{key}": value for key, value in measured.items() if value is not None})
            if steps == max_steps and horizon == full_horizon:
                samples = predicted
    result["generation/terminal_mse"] = result[f"generation/k{max_steps}/h{full_horizon}/terminal_mse"]
    if motion is not None:
        result["generation/motion_mse"] = result[f"generation/k{max_steps}/h{full_horizon}/motion_mse"]
    goal = terminal(frames[:, -1])
    result["terminal/feature_std"] = float(goal.std(0, correction=0).mean())
    noise = {name: examples[f"noise_{name}"] for name in samples}
    for name, o, a in (("state", state.roll(1, 0), actions), ("action", state, actions.roll(1, 0))):
        shuffled = joint_euler_sample(model.predictor, o, a, noise, max_steps)
        result[f"generation/{name}_shuffle_terminal_mse_delta"] = float(
            (shuffled["terminal"]-goal).square().mean()-(samples["terminal"]-goal).square().mean())
    examples.update({f"predicted_{name}": value for name, value in samples.items()})
    examples["target_terminal"] = goal
    return result, {key: value.cpu() for key, value in examples.items()}


@torch.no_grad()
def visualize_terminal(terminal, frames, target, predicted, path, count=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = terminal.dct_basis.device
    truth_image = terminal.image(target.to(device)).cpu()
    predicted_image = terminal.image(predicted.to(device)).cpu() if predicted is not None else None
    n = min(count, len(frames))
    columns = 4 if predicted_image is not None else 3
    fig, axes = plt.subplots(n, columns, figsize=(3*columns, 3*n), squeeze=False)
    for row in range(n):
        panels = [(frames[row, 0], "Current real frame"), (frames[row, -1], "Terminal real frame"),
                  (truth_image[row], "Fixed terminal target")]
        if predicted_image is not None:
            panels.append((predicted_image[row], "Generated terminal measurement"))
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image.clamp(0, 1).permute(1, 2, 0).numpy())
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
