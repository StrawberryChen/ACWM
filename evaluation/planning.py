from __future__ import annotations

from collections import deque
import glob
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
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


def _resize_image_tensor(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    # image: [3,H,W] -> [3,height,width]
    assert image.ndim == 3, f"image tensor must be [C,H,W], got {tuple(image.shape)}"
    if image.shape[-2:] == (height, width):
        return image
    return F.interpolate(image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)


class PlanningEvaluator:
    """Closed-loop CEM evaluation in the official gym-pusht environment."""

    def __init__(self, planner, config: dict[str, Any], history_length: int, action_dim: int, device,
                 dataset_paths: list[str] | None = None):
        self.planner = planner
        self.config = config
        self.history_length = history_length
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.dataset_paths = dataset_paths or []
        self.observation_height = int(self.config.get("observation_height", 96))
        self.observation_width = int(self.config.get("observation_width", 96))

    def _image(self, observation: Any) -> torch.Tensor:
        image = _image_tensor(observation, self.device)
        return _resize_image_tensor(image, self.observation_height, self.observation_width)

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
            return self._image(np.asarray(Image.open(self.config["goal_image"]).convert("RGB"))).unsqueeze(0)
        reset_state = self.config.get("goal_reset_state", [256.0, 256.0, 256.0, 256.0, 0.785398])
        goal_observation, _ = env.reset(options={"reset_to_state": reset_state})
        self._set_goal_pose(env, np.asarray(reset_state, dtype=np.float32))
        return self._image(goal_observation).unsqueeze(0)

    @staticmethod
    def _set_goal_pose(env, goal_state: np.ndarray) -> None:
        """Set Push-T green goal pose from a dataset state.

        gym-pusht stores state as [agent_x, agent_y, block_x, block_y, block_angle],
        while env.unwrapped.goal_pose is [block_x, block_y, block_angle].
        """
        env.unwrapped.goal_pose = np.asarray(goal_state[2:5], dtype=np.float32)

    def _dataset_eval_cases(self, episodes: int) -> list[dict[str, Any]]:
        """LeWorld-style eval cases sampled from validation episodes.

        Each case starts from a dataset simulator state and uses the same
        episode's state at +goal_offset_steps as the goal.
        """
        paths: list[str] = []
        for pattern in self.dataset_paths:
            matches = sorted(glob.glob(str(pattern)))
            paths.extend(matches if matches else [str(pattern)])
        if not paths:
            raise ValueError("LeWorld dataset-goal planning requires validation dataset paths")
        goal_offset = int(self.config.get("goal_offset_steps", 25))
        cases: list[dict[str, Any]] = []
        for path in paths:
            with np.load(path) as data:
                if "states" not in data:
                    raise KeyError(
                        f"{path} has no 'states'. Re-run scripts/prepare_pusht.py with the latest code "
                        "so planning eval can reset to dataset start/goal states."
                    )
                frames = data["frames"]
                actions = data["actions"]
                states = data["states"].astype(np.float32)
                if len(states) != len(frames) or len(actions) != len(frames) - 1:
                    raise ValueError(f"{path} has inconsistent frames/actions/states lengths")
                for current in range(self.history_length - 1, len(frames) - goal_offset):
                    start = current - self.history_length + 1
                    cases.append({
                        "path": path,
                        "current": current,
                        "history_frames": frames[start: current + 1],
                        "history_actions": actions[start:current].astype(np.float32),
                        "start_state": states[current],
                        "goal_state": states[current + goal_offset],
                    })
        if len(cases) < episodes:
            raise ValueError(f"requested {episodes} planning episodes, but only {len(cases)} valid dataset starts exist")
        rng = np.random.default_rng(int(self.config.get("seed", 0)))
        indices = rng.choice(len(cases), size=episodes, replace=False)
        return [cases[int(index)] for index in indices]

    @torch.no_grad()
    def evaluate(self, model, episodes: int = 100, video_dir: str | Path | None = None,
                 videos_to_save: int = 3) -> dict[str, Any]:
        try:
            import imageio.v2 as imageio
        except ImportError as error:
            raise ImportError("planning videos require imageio") from error
        model.eval()
        use_dataset_goals = bool(self.config.get("eval_from_dataset", True))
        dataset_cases = self._dataset_eval_cases(episodes) if use_dataset_goals else None
        if not use_dataset_goals:
            goal_env = self._make_env()
            fixed_goal = self._goal_frame(goal_env)
            goal_env.close()
        else:
            fixed_goal = None
        successes, rewards, final_rewards, episode_lengths, planning_times, video_paths = 0, [], [], [], [], []
        replan_interval = max(1, int(self.config.get("replan_interval", 1)))
        action_block = max(1, int(self.config.get("action_block", 1)))
        raw_action_dim = max(1, self.action_dim // action_block)
        progress = tqdm(range(episodes), desc="Push-T planning", dynamic_ncols=True, leave=True)
        for episode in progress:
            env = self._make_env()
            if dataset_cases is not None:
                case = dataset_cases[episode]
                observation, _ = env.reset(options={"reset_to_state": case["start_state"]})
                self._set_goal_pose(env, case["goal_state"])
                current = self._image(observation)
                frames = deque([self._image(frame) for frame in case["history_frames"]],
                               maxlen=self.history_length)
                frames[-1] = current
                if getattr(model, "predictor_type", None) == "v3_n1":
                    raw_history_actions = torch.as_tensor(case["history_actions"], device=self.device).float().unsqueeze(0)
                    normalized_history_actions = model.predictor.normalize_action_sequence(raw_history_actions)[0]
                    actions = deque(normalized_history_actions.unbind(0), maxlen=max(self.history_length - 1, 1))
                else:
                    actions = deque([torch.as_tensor(action, device=self.device).float()
                                     for action in case["history_actions"]],
                                    maxlen=max(self.history_length - 1, 1))
                goal_observation, _ = env.reset(options={"reset_to_state": case["goal_state"]})
                self._set_goal_pose(env, case["goal_state"])
                goal = self._image(goal_observation).unsqueeze(0)
                observation, _ = env.reset(options={"reset_to_state": case["start_state"]})
                self._set_goal_pose(env, case["goal_state"])
                current = self._image(observation)
            else:
                observation, _ = env.reset(seed=self.config.get("seed", 0) + episode)
                current = self._image(observation)
                frames = deque([current.clone() for _ in range(self.history_length)], maxlen=self.history_length)
                actions = deque([torch.zeros(self.action_dim, device=self.device) for _ in range(self.history_length - 1)],
                                maxlen=max(self.history_length - 1, 1))
                goal = fixed_goal
            video = [env.render()] if episode < videos_to_save else []
            episode_reward, succeeded = 0.0, False
            action_queue: deque[tuple[torch.Tensor, torch.Tensor | None, bool]] = deque()
            for step in range(self.config.get("max_steps", 300)):
                if not action_queue:
                    history_frames = torch.stack(tuple(frames)).unsqueeze(0)
                    if getattr(model, "predictor_type", None) == "v3_n1":
                        history_actions = (torch.stack(tuple(actions)).unsqueeze(0) if len(actions)
                                           else torch.zeros(1, self.history_length - 1, self.action_dim,
                                                            device=self.device))
                    else:
                        history_actions = (torch.stack(tuple(actions)).unsqueeze(0) if self.history_length > 1
                                           else torch.empty(1, 0, self.action_dim, device=self.device))
                    start_plan = time.perf_counter()
                    planned = self.planner.plan(model, history_frames, history_actions, current.unsqueeze(0), goal)
                    planning_times.append(time.perf_counter() - start_plan)
                    if getattr(model, "predictor_type", None) == "v3_n1" and hasattr(model, "denormalize_planner_action"):
                        # planned block: [action_block * action_dim] normalized.
                        # env queue receives raw single-step actions [action_dim].
                        for block in planned[0, :replan_interval].unbind(0):
                            raw_block = model.denormalize_planner_action(block[None])[0]
                            assert raw_block.ndim == 2 and raw_block.shape[0] == action_block, (
                                f"expected raw action block [{action_block},A], got {tuple(raw_block.shape)}"
                            )
                            for raw_index, raw_action in enumerate(raw_block.unbind(0)):
                                action_queue.append((raw_action, block, raw_index == action_block - 1))
                    else:
                        action_queue.extend((action, None, True) for action in planned[0, :replan_interval].unbind(0))
                action, planned_block, block_done = action_queue.popleft()
                env_action = action if getattr(model, "predictor_type", None) == "v3_n1" else (
                    model.denormalize_planner_action(action[None])[0] if hasattr(model, "denormalize_planner_action") else action
                )
                assert env_action.ndim == 1 and env_action.shape[-1] == raw_action_dim, (
                    f"env action must be [{raw_action_dim}], got {tuple(env_action.shape)}"
                )
                observation, reward, terminated, truncated, info = env.step(env_action.cpu().numpy())
                episode_reward = max(episode_reward, float(reward))
                current = self._image(observation)
                if getattr(model, "predictor_type", None) == "v3_n1":
                    # LeWorld action_block alignment: one latent/frame-history step
                    # advances only after action_block raw env actions have executed.
                    if block_done:
                        frames.append(current)
                        if self.history_length > 1:
                            assert planned_block is not None
                            actions.append(planned_block)
                else:
                    frames.append(current)
                if self.history_length > 1 and getattr(model, "predictor_type", None) != "v3_n1":
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
            final_rewards.append(float(reward))
            episode_lengths.append(step + 1)
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
            "mean_max_overlap": float(np.mean(rewards)),
            "mean_final_overlap": float(np.mean(final_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_planning_time": float(np.mean(planning_times)) if planning_times else 0.0,
            "videos": video_paths,
        }
