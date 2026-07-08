from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


def _image_tensor(observation: Any, device: torch.device) -> torch.Tensor:
    if isinstance(observation, dict):
        for key in ("pixels", "image", "observation.image"):
            if key in observation:
                observation = observation[key]
                break
    image = torch.as_tensor(observation, device=device)
    if image.ndim != 3:
        raise ValueError("Push-T planning requires a pixel observation")
    if image.shape[-1] in (1, 3, 4):
        image = image[..., :3].permute(2, 0, 1)
    image = image.float()
    return image / 255.0 if image.max() > 1 else image


class PlanningEvaluator:
    """Closed-loop CEM evaluation in the official gym-pusht environment."""

    def __init__(self, planner, config: dict[str, Any], history_length: int, action_dim: int, device):
        self.planner = planner
        self.config = config
        self.history_length = history_length
        self.action_dim = action_dim
        self.device = torch.device(device)

    def _make_env(self):
        try:
            import gymnasium as gym
            import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0
            import pymunk
        except ImportError as error:
            raise ImportError("planning validation requires gymnasium and gym-pusht") from error
        if not hasattr(pymunk.Space, "add_collision_handler"):
            raise RuntimeError(
                "gym-pusht 0.1.x is incompatible with Pymunk 7+. "
                "Install the compatible version with: pip install 'pymunk>=6.6,<7'"
            )
        return gym.make(
            self.config.get("env_id", "gym_pusht/PushT-v0"),
            obs_type="pixels",
            render_mode="rgb_array",
            observation_width=self.config.get("observation_width", 96),
            observation_height=self.config.get("observation_height", 96),
        )

    def _goal_frame(self, env) -> torch.Tensor:
        if self.config.get("goal_image"):
            try:
                from PIL import Image
            except ImportError as error:
                raise ImportError("loading environment.goal_image requires Pillow") from error
            return _image_tensor(np.asarray(Image.open(self.config["goal_image"]).convert("RGB")), self.device).unsqueeze(0)
        reset_state = self.config.get("goal_reset_state", [256.0, 256.0, 256.0, 256.0, 0.785398])
        goal_observation, _ = env.reset(options={"reset_to_state": reset_state})
        return _image_tensor(goal_observation, self.device).unsqueeze(0)

    @torch.no_grad()
    def evaluate(self, model, episodes: int = 100, video_dir: str | Path | None = None,
                 videos_to_save: int = 3) -> dict[str, Any]:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ImportError("planning videos require imageio") from error
        model.eval()
        goal_env = self._make_env()
        goal = self._goal_frame(goal_env)
        goal_env.close()
        successes, rewards, video_paths = 0, [], []
        progress = tqdm(range(episodes), desc="Push-T planning", dynamic_ncols=True, leave=True)
        for episode in progress:
            env = self._make_env()
            observation, _ = env.reset(seed=self.config.get("seed", 0) + episode)
            current = _image_tensor(observation, self.device)
            frames = deque([current.clone() for _ in range(self.history_length)], maxlen=self.history_length)
            actions = deque([torch.zeros(self.action_dim, device=self.device) for _ in range(self.history_length - 1)],
                            maxlen=max(self.history_length - 1, 1))
            video = [env.render()] if episode < videos_to_save else []
            episode_reward, succeeded = 0.0, False
            for step in range(self.config.get("max_steps", 300)):
                history_frames = torch.stack(tuple(frames)).unsqueeze(0)
                history_actions = (torch.stack(tuple(actions)).unsqueeze(0) if self.history_length > 1
                                   else torch.empty(1, 0, self.action_dim, device=self.device))
                planned = self.planner.plan(model, history_frames, history_actions, current.unsqueeze(0), goal)
                action = planned[0, 0]
                observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                episode_reward = max(episode_reward, float(reward))
                current = _image_tensor(observation, self.device)
                frames.append(current)
                if self.history_length > 1:
                    actions.append(action)
                if video:
                    video.append(env.render())
                succeeded = bool(terminated or info.get("is_success", False) or reward >= self.config.get("success_threshold", 0.95))
                if step % 10 == 0:
                    progress.set_postfix(episode=episode + 1, step=step + 1,
                                         successes=f"{successes}/{episode}", reward=f"{episode_reward:.3f}")
                if terminated or truncated:
                    break
            successes += int(succeeded)
            rewards.append(episode_reward)
            progress.set_postfix(successes=f"{successes}/{episode + 1}",
                                 success_rate=f"{successes / (episode + 1):.1%}")
            if video:
                path = Path(video_dir or "outputs/videos") / f"planning_episode_{episode:03d}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(path, video, fps=self.config.get("video_fps", 20))
                video_paths.append(path)
            env.close()
        return {
            "success_rate": successes / episodes,
            "mean_max_reward": float(np.mean(rewards)),
            "videos": video_paths,
        }
