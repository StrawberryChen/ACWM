from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def validate_prediction(trainer, loader: Iterable[dict[str, torch.Tensor]]) -> dict[str, float]:
    trainer.model.eval()
    totals: dict[str, float] = defaultdict(float)
    count = 0
    progress = tqdm(loader, desc="Prediction validation", dynamic_ncols=True, leave=True)
    for batch in progress:
        count += 1
        for key, value in trainer.compute_one_step(batch).items():
            totals[key] += value.item()
        if getattr(trainer.model, "predictor_type", "adaln") == "v3_n1" and "rollout_actions" in batch:
            rollout_metrics = _v3_rollout_metrics(trainer, batch)
            for key, value in rollout_metrics.items():
                totals[key] += value.item()
        progress.set_postfix(loss=f"{totals['loss'] / count:.5f}")
    if count == 0:
        raise ValueError("validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def _v3_rollout_metrics(trainer, raw_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch = trainer._move(raw_batch)
    history = trainer.model.predictor.encode_gaussian_sequence(batch["history_frames"]).mu
    current = history[:, -1]
    history_actions = batch["history_actions"]
    metrics = {}
    for step_index, action in enumerate(batch["rollout_actions"].unbind(dim=1), start=1):
        prediction = trainer.model.step(history, current, action, history_actions)
        history, current = prediction.agent, prediction.environment
        history_actions = torch.cat((history_actions[:, 1:], action[:, None]), dim=1)
        if step_index in {1, 2, 3, 5}:
            target = trainer.model.predictor.encode_gaussian(batch["rollout_frames"][:, step_index - 1]).mu
            metrics[f"rollout_mse_step_{step_index}"] = (current - target).square().mean()
    return metrics


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
        prediction = trainer.model.step(agent, environment, batch["current_action"], batch["history_actions"])
        target_environment = trainer.model.encode_goal(batch["next_frame"]) if hasattr(trainer.model, "encode_goal") else trainer.model.environment_encoder(batch["next_frame"])
        if getattr(trainer.model, "predictor_type", "adaln") in {"motion_token", "forward_inverse", "v3_n1"}:
            agent_error = None
        else:
            target_agent = trainer.model.agent_encoder(batch["next_history_frames"], batch["next_history_actions"])
            agent_error = (prediction.agent - target_agent).square().mean(-1)
        environment_error = (prediction.environment - target_environment).square().mean(-1)
        for index in range(len(environment_error)):
            image = (batch["next_frame"][index].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canvas = Image.fromarray(image).resize((384, 384))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 384, 42), fill=(0, 0, 0))
            if agent_error is None:
                text = f"{trainer.model.predictor_type} pred MSE {environment_error[index]:.5f}"
            else:
                text = f"agent MSE {agent_error[index]:.5f} | env MSE {environment_error[index]:.5f}"
            draw.text((8, 6), text, fill="white")
            frames.append(np.asarray(canvas))
            seen += 1
            if seen >= max_samples:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(path, frames, fps=fps, loop=0)
                return path
    raise ValueError("no validation samples available for animation")
