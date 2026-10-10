"""V2 plots with explicit NLL/matching/SIGReg names and retrieval baselines."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def save_plots(history, metrics, observations, metadata, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    series = {"total_loss": ["train/loss", "val/loss"],
              "nll_loss": ["train/nll_loss", "val/nll_loss"],
              "matching_loss": ["train/matching_loss", "val/matching_loss"],
              "sigreg_loss": ["train/sigreg_loss", "val/sigreg_loss"],
              "matching_accuracy": ["train/matching_acc", "val/matching_acc"],
              "score_gap": ["train/score_gap", "val/score_gap"],
              "positive_cosine": ["train/positive_cosine", "val/positive_cosine"],
              "feature_std": ["representation/mean_feature_std"],
              "effective_rank": ["representation/effective_rank"],
              "pairwise_cosine": ["representation/mean_pairwise_cosine"],
              "predicted_sigma": ["train/predicted_sigma_mean", "val/predicted_sigma_mean"],
              "shuffle_dependency": ["dependency/normal_acc", "dependency/O_shuffle_acc", "dependency/A_shuffle_acc"]}
    for name, keys in series.items():
        fig, ax = plt.subplots()
        for key in keys:
            if any(key in row for row in history):
                ax.plot([row["epoch"] for row in history], [row.get(key, float("nan")) for row in history], label=key)
        if ax.lines:
            ax.legend()
        ax.set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(directory/f"{name}.png")
        plt.close(fig)
    fig, ax = plt.subplots()
    horizons = config["validation"]["retrieval_horizons"]
    for prefix, name in (("", "Predicted"), ("identity_", "Keep current O"), ("A_shuffle_", "Shuffle actions")):
        ax.plot(horizons, [metrics[f"retrieval/{prefix}top1_h{h}"] for h in horizons], "o-", label=name)
    ax.axhline(1/metrics["retrieval/candidates"], linestyle="--", label="Chance")
    ax.set(xlabel="Raw-step horizon", ylabel="Future retrieval Top-1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory/"horizon_vs_retrieval_accuracy.png")
    plt.close(fig)
    centered = observations-observations.mean(0)
    _, _, directions = torch.linalg.svd(centered, full_matrices=False)
    if directions.shape[0] >= 2:
        xy = (centered@directions[:2].T).numpy()
        for i, name in enumerate(("timestep", "episode", "action_direction")):
            fig, ax = plt.subplots()
            points = ax.scatter(xy[:, 0], xy[:, 1], c=metadata[:, i], s=5)
            fig.colorbar(points, ax=ax, label=name)
            ax.set(xlabel="PC1", ylabel="PC2")
            fig.tight_layout()
            fig.savefig(directory/f"pca_{name}.png")
            plt.close(fig)
