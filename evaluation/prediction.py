from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@torch.no_grad()
def validate_prediction(trainer, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
    trainer.model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    for batch in loader:
        count += 1
        for key, value in trainer.compute_one_step(batch).items():
            totals[key] += value.item()
    if count == 0:
        raise ValueError("validation loader is empty")
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def save_prediction_animation(trainer, loader, path: str | Path, max_samples: int = 32, fps: int = 4) -> Path:
    """Animate observations with honest latent errors (ACWM has no pixel decoder)."""
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ImportError("prediction animation requires imageio and Pillow") from error
    trainer.model.eval()
    frames = []
    seen = 0
    for raw_batch in loader:
        batch = trainer._move(raw_batch)
        agent, environment = trainer.model.encode(batch["history_frames"], batch["history_actions"], batch["current_frame"])
        prediction = trainer.model.step(agent, environment, batch["current_action"])
        target_agent = trainer.model.agent_encoder(batch["next_history_frames"], batch["next_history_actions"])
        target_environment = trainer.model.environment_encoder(batch["next_frame"])
        agent_error = (prediction.agent - target_agent).square().mean(-1)
        environment_error = (prediction.environment - target_environment).square().mean(-1)
        for index in range(len(agent_error)):
            image = (batch["next_frame"][index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canvas = Image.fromarray(image).resize((384, 384))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 384, 42), fill=(0, 0, 0))
            draw.text((8, 6), f"agent MSE {agent_error[index]:.5f} | env MSE {environment_error[index]:.5f}", fill="white")
            frames.append(np.asarray(canvas))
            seen += 1
            if seen >= max_samples:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(path, frames, fps=fps, loop=0)
                return path
    raise ValueError("no validation samples available for animation")

