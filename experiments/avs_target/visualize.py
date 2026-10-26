"""Measurement visualizations, never claimed to reconstruct future images."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def heatmap(x, vmax):
    values = (np.maximum(x, 0)/max(vmax, 1e-12)).clip(0, 1)
    return (plt.get_cmap("magma")(values)[..., :3]*255).astype(np.uint8)


@torch.no_grad()
def visualize_targets(operator, loader, output, device, count=50):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        maps = operator.maps(operator(frames, normalize=False))[:, 0].cpu().numpy()
        for i in range(len(frames)):
            if saved >= count:
                return saved
            response = maps[i]
            channel_count = operator.channels if operator.kind != "random_projection" else 0
            panels = 2+channel_count+(2 if operator.kind == "avs" else 1)
            columns = 4
            fig, axes = plt.subplots(int(np.ceil(panels/columns)), columns, figsize=(12, 3*int(np.ceil(panels/columns))))
            axes = np.asarray(axes).ravel()
            for ax in axes:
                ax.axis("off")
            for j in range(2):
                axes[j].imshow(frames[i, j].permute(1, 2, 0).cpu())
                axes[j].set_title(f"Real frame t+{j}")
            for channel in range(channel_count):
                maximum = max(float(np.abs(response).max()), 1e-12)
                axes[2+channel].imshow(response[channel], cmap="magma" if operator.kind == "avs" else "coolwarm",
                                       vmin=0 if operator.kind == "avs" else -maximum, vmax=maximum)
                label = f"{channel*360/operator.directions:g} degrees (image clockwise)" if operator.kind == "avs" else f"RGB diff {channel}"
                axes[2+channel].set_title(label)
            if operator.kind == "avs":
                direction = response.argmax(0).astype(float)
                direction[response.sum(0) <= 1e-12] = np.nan
                axes[2+channel_count].imshow(direction, cmap="hsv", vmin=0, vmax=operator.directions)
                axes[2+channel_count].set_title("Direction argmax; blank=zero")
                axes[3+channel_count].imshow(response.sum(0), cmap="magma")
                axes[3+channel_count].set_title("Motion magnitude")
            else:
                axes[2+channel_count].plot(response.reshape(-1))
                axes[2+channel_count].axis("on")
                axes[2+channel_count].set_title("Fixed target components")
            fig.tight_layout()
            fig.savefig(output/f"target_{saved:04d}.png")
            plt.close(fig)
            saved += 1
    return saved


@torch.no_grad()
def visualize_generation(operator, examples, output, count=4):
    import imageio.v2 as imageio
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = operator.mean.device
    target = operator.maps(operator.denormalize(examples["target"].to(device))).cpu().numpy()
    predicted = operator.maps(operator.denormalize(examples["prediction"].to(device))).cpu().numpy()
    for i in range(min(count, len(target))):
        frames = []
        vmax = max(float(np.abs(target[i]).max()), 1e-12)
        for h in range(target.shape[1]):
            real = (examples["frames"][i, h+1].permute(1, 2, 0).numpy()*255).astype(np.uint8)
            # For signed/RP targets use absolute component magnitude; no invented direction.
            gt = heatmap(np.abs(target[i, h]).mean(0), vmax)
            pr = heatmap(np.abs(predicted[i, h]).mean(0), vmax)
            gt = np.repeat(np.repeat(gt, max(1, 96//gt.shape[0]), 0), max(1, 96//gt.shape[1]), 1)[:96, :96]
            pr = np.repeat(np.repeat(pr, max(1, 96//pr.shape[0]), 0), max(1, 96//pr.shape[1]), 1)[:96, :96]
            # Dimensions 3/6/12 divide 96; RP is 1x1. Labels documented in README.
            assert gt.shape == pr.shape == real.shape
            frames.append(np.concatenate((real, gt, pr), 1))
        imageio.mimsave(output/f"trajectory_{i:03d}.mp4", frames, fps=10)
        imageio.imwrite(output/f"trajectory_{i:03d}.png", frames[-1])
