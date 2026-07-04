from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_loss_curve(history: dict[str, list[float]], output: str | Path | None = None):
    figure, axis = plt.subplots()
    for name, values in history.items():
        axis.plot(values, label=name)
    axis.set(xlabel="step", ylabel="loss")
    axis.legend()
    if output:
        figure.savefig(output, bbox_inches="tight")
    return figure


def plot_latent_similarity(states: torch.Tensor, output: str | Path | None = None):
    normalized = torch.nn.functional.normalize(states.detach().cpu(), dim=-1)
    similarity = normalized @ normalized.T
    figure, axis = plt.subplots()
    image = axis.imshow(similarity.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    figure.colorbar(image, ax=axis)
    if output:
        figure.savefig(output, bbox_inches="tight")
    return figure


def plot_embeddings(states: torch.Tensor, labels=None, output: str | Path | None = None):
    """Dependency-free 2-D PCA view for agent or environment states."""
    centered = states.detach().cpu() - states.detach().cpu().mean(0)
    _, _, components = torch.pca_lowrank(centered, q=2)
    points = centered @ components
    figure, axis = plt.subplots()
    axis.scatter(points[:, 0], points[:, 1], c=labels)
    if output:
        figure.savefig(output, bbox_inches="tight")
    return figure


def plot_latent_rollout(environment_states: torch.Tensor, goal_state: torch.Tensor | None = None,
                        output: str | Path | None = None):
    """PCA projection of a rollout [L,D], optionally alongside one goal state."""
    states = environment_states.detach().cpu()
    combined = torch.cat((states, goal_state.detach().cpu().view(1, -1)), dim=0) if goal_state is not None else states
    centered = combined - combined.mean(0)
    _, _, components = torch.pca_lowrank(centered, q=min(2, *centered.shape))
    points = centered @ components
    figure, axis = plt.subplots()
    axis.plot(points[: len(states), 0], points[: len(states), 1], marker="o", label="prediction")
    if goal_state is not None:
        axis.scatter(points[-1, 0], points[-1, 1], marker="*", s=160, label="goal")
    axis.legend()
    if output:
        figure.savefig(output, bbox_inches="tight")
    return figure
