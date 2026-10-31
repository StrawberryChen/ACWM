"""Read-only diagnostics; no diagnostic contributes to the training objective."""
import torch
from torch.nn import functional as F
from .losses import objective, reference_progress


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


@torch.no_grad()
def evaluate(model, batches, config, device):
    model.eval()
    sums, count, collected = {}, 0, {k: [] for k in ("start", "target", "prediction", "states", "velocities", "actions")}
    generator = torch.Generator(device=device).manual_seed(config["seed"]+100)
    for batch in batches:
        frames, actions = batch["frames"].to(device), batch["actions"].to(device)
        output = model(frames, actions)
        metrics = {f"val/{k}": float(v) for k, v in objective(output, **config["loss"]).items()}
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
    transition = (values["target"]-values["start"]).norm(dim=-1)
    metrics.update(summary(transition, "dynamics/latent_transition_distance"))
    metrics.update(summary(values["velocities"].norm(dim=-1), "dynamics/velocity_norm"))
    progress = reference_progress(values["velocities"], config["loss"]["eps"])
    metrics.update(summary(progress, "dynamics/progress"))
    metrics["dynamics/progress_violation_ratio"] = float((progress < config["loss"]["margin"]).float().mean())
    metrics["dynamics/velocity_cosine"] = float(F.cosine_similarity(values["velocities"], values["velocities"][:, :1], dim=-1).mean())
    metrics["dynamics/zero_reference_ratio"] = float((values["velocities"][:, 0].norm(dim=-1) <= config["loss"]["eps"]).float().mean())
    # A global random derangement: even final singleton minibatches get a wrong A.
    order = torch.randperm(count, generator=torch.Generator().manual_seed(config["seed"]+101))
    shuffled = torch.empty_like(order)
    shuffled[order] = order.roll(1)
    error = 0.
    for selected in torch.arange(count).split(config["data"]["batch_size"]):
        prediction, _, _ = model.integrator(model.field, values["start"][selected].to(device),
                                             values["actions"][shuffled[selected], 0].to(device))
        error += float((prediction-values["target"][selected].to(device)).square().mean())*len(selected)
    metrics["action/shuffle_loss"] = error/count
    metrics["action/shuffle_loss_delta"] = error/count-metrics["val/loss_pred"]
    metrics["action/normal_loss"] = metrics["val/loss_pred"]
    metrics["validation/samples"] = count
    return metrics, values
