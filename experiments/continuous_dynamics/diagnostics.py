"""Read-only diagnostics; no diagnostic contributes to the training objective."""
import torch
from torch.nn import functional as F
from .losses import objective, output_progress


def summary(values, prefix):
    values = values.double().flatten()
    return {f"{prefix}/{name}": float(value) for name, value in
            (("mean", values.mean()), ("std", values.std(correction=0)),
             ("min", values.min()), ("max", values.max()))}


class CollapseDiagnostics:
    @staticmethod
    def compute(latents):
        # Across observations, NOT across the coordinates of one observation.
        z = latents.double()  # [N,D]
        centered = z-z.mean(0)
        covariance = centered.T@centered/max(len(z)-1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total = eigenvalues.sum()
        probabilities = eigenvalues/total.clamp_min(1e-30)
        rank = (-(probabilities*probabilities.clamp_min(1e-30).log()).sum()).exp() if total > 1e-24 else total*0
        norms = z.norm(dim=-1)
        pairs = (norms > 1e-12) & (norms.roll(1) > 1e-12)
        cosines = F.cosine_similarity(z, z.roll(1, 0), dim=-1)
        return {"collapse/feature_std": float(z.std(0, correction=0).mean()),
                "collapse/effective_rank": float(rank),
                "collapse/pairwise_cosine": float(cosines[pairs].mean()) if pairs.any() else 0.,
                "collapse/zero_latent_ratio": float((norms <= 1e-12).double().mean()),
                "collapse/state_variance": float(centered.square().sum(-1).mean())}


class ActionSensitivityDiagnostics:
    @staticmethod
    def compute(field, state, config, generator):
        n = config["action_samples"]
        if n < 2 or config["action_high"] <= config["action_low"]:
            raise ValueError("at least two sampled actions and ordered bounds required")
        actions = torch.rand(len(state), n, 2, device=state.device, generator=generator)
        actions = config["action_low"]+(config["action_high"]-config["action_low"])*actions
        velocity = field(state[:, None].expand(-1, n, -1).reshape(-1, state.shape[-1]),
                         actions.reshape(-1, 2)).reshape(len(state), n, -1)
        i, j = torch.triu_indices(n, n, offset=1, device=state.device)
        return {"action/velocity_distance": float((velocity[:, i]-velocity[:, j]).norm(dim=-1).mean()),
                "action/velocity_cosine": float(F.cosine_similarity(velocity[:, i], velocity[:, j], dim=-1).mean())}

    @staticmethod
    def trajectory(model, state, config, generator, horizon):
        n = config["action_samples"]
        if n < 2 or config["action_high"] <= config["action_low"]:
            raise ValueError("at least two actions and ordered bounds required")
        actions = torch.rand(len(state)*n, horizon, 2, device=state.device, generator=generator)
        actions = config["action_low"]+(config["action_high"]-config["action_low"])*actions
        starts = state[:, None].expand(-1, n, -1).reshape(-1, state.shape[-1])
        context, embeddings, _, direction = model.condition(starts, actions)
        velocity = model.field(starts, embeddings[:, 0], context, direction).reshape(len(state), n, -1)
        direction = direction.reshape(len(state), n, -1)
        i, j = torch.triu_indices(n, n, offset=1, device=state.device)
        return {"action/velocity_distance": float((velocity[:, i]-velocity[:, j]).norm(dim=-1).mean()),
                "action/velocity_cosine": float(F.cosine_similarity(velocity[:, i], velocity[:, j], dim=-1).mean()),
                "direction/action_sequence_distance": float((direction[:, i]-direction[:, j]).norm(dim=-1).mean())}


@torch.no_grad()
def evaluate(model, batches, config, device):
    model.eval()
    sums, count, collected = {}, 0, {k: [] for k in ("start", "target", "prediction", "states", "velocities", "actions")}
    trajectory = hasattr(model, "direction_head")
    if trajectory:
        collected.update(direction=[], direction_raw=[])
    generator = torch.Generator(device=device).manual_seed(config["seed"]+100)
    for batch in batches:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        output = model(frames, actions)
        metrics = {f"val/{k}": float(v) for k, v in objective(output, **config["loss"]).items()}
        if trajectory:
            metrics.update(ActionSensitivityDiagnostics.trajectory(model, output["start"], config["validation"], generator, actions.shape[1]))
        else:
            metrics.update(ActionSensitivityDiagnostics.compute(model.field, output["start"], config["validation"], generator))
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.)+value*len(frames)
        count += len(frames)
        for key in collected:
            collected[key].append((actions if key == "actions" else output[key]).cpu())
    if count < 2:
        raise ValueError("validation needs at least two transitions")
    values = {k: torch.cat(v) for k, v in collected.items()}
    metrics = {k: v/count for k, v in sums.items()}
    metrics.update(CollapseDiagnostics.compute(values["start"]))
    if trajectory:
        real_sequence = torch.cat((values["start"][:, None], values["target"]), 1)
        transition = real_sequence.diff(dim=1).norm(dim=-1)
        metrics["val/identity_loss"] = float((values["target"]-values["start"][:, None]).square().mean())
        for h in range(values["target"].shape[1]):
            metrics[f"val/pred_step_{h+1}"] = float((values["prediction"][:, h]-values["target"][:, h]).square().mean())
        metrics["direction/raw_norm"] = float(values["direction_raw"].norm(dim=-1).mean())
        metrics["direction/feature_std"] = float(values["direction"].std(0, correction=0).mean())
        metrics["direction/velocity_cosine"] = float(F.cosine_similarity(values["velocities"], values["direction"][:, None, None], dim=-1).mean())
    else:
        transition = (values["target"]-values["start"]).norm(dim=-1)
    metrics.update(summary(transition, "dynamics/latent_transition_distance"))
    metrics.update(summary(values["velocities"].norm(dim=-1), "dynamics/velocity_norm"))
    progress = output_progress(values, config["loss"]["eps"])
    metrics.update(summary(progress, "dynamics/progress"))
    metrics["dynamics/progress_violation_ratio"] = float((progress < config["loss"]["margin"]).float().mean())
    reference_velocity = values["velocities"][:, :, :1] if trajectory else values["velocities"][:, :1]
    metrics["dynamics/velocity_cosine"] = float(F.cosine_similarity(values["velocities"], reference_velocity, dim=-1).mean())
    reference_norm = values["direction_raw"].norm(dim=-1) if trajectory else values["velocities"][:, 0].norm(dim=-1)
    metrics["dynamics/zero_reference_ratio"] = float((reference_norm <= config["loss"]["eps"]).float().mean())
    # A global random derangement: even final singleton minibatches get a wrong A.
    order = torch.randperm(count, generator=torch.Generator().manual_seed(config["seed"]+101))
    shuffled = torch.empty_like(order)
    shuffled[order] = order.roll(1)
    error = 0.
    for selected in torch.arange(count).split(config["data"]["batch_size"]):
        if trajectory:
            prediction = model.rollout(values["start"][selected].to(device), values["actions"][shuffled[selected]].to(device))
        else:
            prediction, _, _ = model.integrator(model.field, values["start"][selected].to(device),
                                                 values["actions"][shuffled[selected], 0].to(device))
        error += float((prediction-values["target"][selected].to(device)).square().mean())*len(selected)
    metrics["action/shuffle_loss"] = error/count
    metrics["action/shuffle_loss_delta"] = error/count-metrics["val/loss_pred"]
    metrics["action/normal_loss"] = metrics["val/loss_pred"]
    metrics["validation/samples"] = count
    return metrics, values
