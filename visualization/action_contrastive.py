"""Persist plots and PCA; analysis metadata never supervises the model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_plots(history, metrics, observations, metadata, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    series = {
        "training_loss": ["train/loss", "val/loss"],
        "ranking_accuracy": ["train/ranking_acc", "val/ranking_acc"],
        "score_gap": ["train/score_gap", "val/score_gap"],
        "feature_std": ["representation/mean_feature_std"],
        "effective_rank": ["representation/effective_rank"],
        "shuffle_dependency": ["dependency/normal_acc", "dependency/O_shuffle_acc", "dependency/A_shuffle_acc"],
    }
    for name, keys in series.items():
        fig, ax = plt.subplots()
        for key in keys:
            if any(key in row for row in history):
                ax.plot([row["epoch"] for row in history], [row.get(key, float("nan")) for row in history], label=key)
        ax.set_xlabel("Epoch")
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png")
        plt.close(fig)
    for name, keys, xlabel in [
        ("action_angle_vs_accuracy", [(a, f"angle/{a}/ranking_acc") for a in config["validation"]["angles"]], "Angle (degrees)"),
        ("action_angle_vs_score_gap", [(a, f"angle/{a}/score_gap") for a in config["validation"]["angles"]], "Angle (degrees)"),
        ("horizon_vs_retrieval_accuracy", [(h, f"retrieval/top1_h{h}") for h in config["validation"]["retrieval_horizons"]], "Raw-step horizon"),
    ]:
        fig, ax = plt.subplots()
        ax.plot([x for x, _ in keys], [metrics[key] for _, key in keys], "o-")
        if name == "horizon_vs_retrieval_accuracy":
            ax.axhline(1/metrics["retrieval/candidates"], ls="--", label="chance")
            ax.legend()
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png")
        plt.close(fig)
    centered = observations - observations.mean(0)
    _, _, v = torch.linalg.svd(centered, full_matrices=False)
    xy = (centered @ v[:2].T).numpy()
    for index, name in enumerate(("timestep", "episode", "action_direction")):
        fig, ax = plt.subplots()
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=metadata[:, index], s=5, cmap="viridis")
        # Connect each sampled trajectory in temporal order to expose state evolution.
        for episode in metadata[:, 1].unique()[:8]:
            ids = (metadata[:, 1] == episode).nonzero().flatten()
            ids = ids[metadata[ids, 0].argsort()]
            ax.plot(xy[ids, 0], xy[ids, 1], alpha=0.2, linewidth=0.5)
        fig.colorbar(scatter, ax=ax, label=name)
        ax.set(xlabel="PC1", ylabel="PC2")
        fig.tight_layout()
        fig.savefig(directory / f"pca_{name}.png")
        plt.close(fig)
