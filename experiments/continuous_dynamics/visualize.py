"""Evaluation-only PCA plots; physical labels never enter model losses."""
from pathlib import Path
import numpy as np
import torch


def pca_fit(latents):
    mean = latents.double().mean(0)
    _, _, right = torch.linalg.svd(latents.double()-mean, full_matrices=False)
    return mean, right[:2].T


@torch.no_grad()
def visualize(model, dataset, config, output, epoch, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = min(config["validation"]["visualization_samples"], len(dataset))
    indices = np.random.default_rng(config["seed"]).choice(len(dataset), count, replace=False)
    samples = [dataset[int(i)] for i in indices]
    images = torch.stack([sample["frames"][0] for sample in samples])
    features = torch.cat([model.encoder(batch.to(device)).cpu() for batch in images.split(config["data"]["batch_size"])])
    mean, basis = pca_fit(features)
    project = lambda z: ((z.cpu().double()-mean)@basis).numpy()
    xy = project(features)
    paths = {}
    # States loaded ONLY in this read-only visualization. Missing labels don't block training.
    states, available = {}, True
    for sample in samples:
        ep = sample["episode"]
        if ep not in states:
            with np.load(dataset.paths[ep]) as data:
                if "states" not in data:
                    available = False
                    break
                states[ep] = data["states"][:, :5].copy()
    if available:
        labels = np.stack([states[s["episode"]][s["timestep"]] for s in samples])
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for i, name in enumerate(("agent_x", "agent_y", "block_x", "block_y", "block_angle")):
            dots = axes[i].scatter(xy[:, 0], xy[:, 1], c=labels[:, i], s=8, cmap="twilight" if i == 4 else "viridis")
            axes[i].set_title(name)
            fig.colorbar(dots, ax=axes[i])
        paths["visualization/physical_pca"] = output/f"physical_pca_{epoch:04d}.png"
        fig.tight_layout()
        fig.savefig(paths["visualization/physical_pca"])
        plt.close(fig)
    else:
        print("No simulator labels: physical-color PCA skipped (training unaffected).")
    # Entire first validation episode in temporal order, not shuffled minibatches.
    raw, _ = dataset.episode(0)
    from torch.nn import functional as F
    image = torch.from_numpy(raw.copy())
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    trajectory = []
    for chunk in image.split(config["data"]["batch_size"]):
        chunk = F.interpolate(chunk.to(device).float()/255., (dataset.image_size, dataset.image_size), mode="bilinear", align_corners=False)
        trajectory.append(model.encoder(chunk).cpu())
    trajectory = torch.cat(trajectory)
    xy_trajectory = project(trajectory)
    fig, ax = plt.subplots()
    ax.plot(xy_trajectory[:, 0], xy_trajectory[:, 1], alpha=.5)
    fig.colorbar(ax.scatter(*xy_trajectory.T, c=np.arange(len(trajectory)), s=12), ax=ax, label="raw frame index")
    ax.set_title("Full episode, validation-fitted PCA")
    paths["visualization/latent_trajectory"] = output/f"latent_trajectory_{epoch:04d}.png"
    fig.savefig(paths["visualization/latent_trajectory"])
    plt.close(fig)
    sample = dataset[0]
    result = model(sample["frames"][None].to(device), sample["actions"][None].to(device))
    internal, target = project(result["states"][0]), project(result["target"])
    fig, ax = plt.subplots()
    ax.plot(*internal.T, "o-", label="Euler internal path")
    for k, point in enumerate(internal):
        ax.annotate(str(k), point)
    ax.scatter(*target.T, marker="*", s=150, label="E(real next frame)")
    ax.set_title("One real step: PCA is diagnostic, not GT for intermediate states")
    ax.legend()
    paths["visualization/internal_integration"] = output/f"internal_{epoch:04d}.png"
    fig.savefig(paths["visualization/internal_integration"])
    plt.close(fig)
    np.savez_compressed(output/f"latents_{epoch:04d}.npz", features=features.numpy(),
                        trajectory=trajectory.numpy(), internal_states=result["states"].cpu().numpy(),
                        target=result["target"].cpu().numpy(), pca_mean=mean.numpy(), pca_basis=basis.numpy())
    return paths
