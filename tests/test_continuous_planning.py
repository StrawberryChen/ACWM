import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch import nn

from experiments.continuous_dynamics.data import load_config
from experiments.continuous_dynamics.planning import (
    LatentTrajectoryCEM, action_bounds, environment_action, evaluate_planning, discover_runs, main)
from experiments.continuous_dynamics.trajectory_model import TrajectoryWorldModel


class FakeEnv:
    """Small deterministic relative-action environment, NOT a Push-T physics test."""
    def __init__(self):
        self.unwrapped = self
        self.action_space = SimpleNamespace(low=np.full(2, -1., dtype=np.float32), high=np.full(2, 1., dtype=np.float32))
        self.goal_pose = np.array([256., 256., 0.])
        self.state = np.zeros(7, dtype=np.float32)
        self.goal = self.state.copy()
        self.closed = False

    def reset(self, seed=None, options=None):
        if options:
            self.state = options["state"].copy()
            self.goal = options.get("goal_state", self.goal).copy()
        return None, {}

    def _set_goal_state(self, goal):
        self.goal = goal.copy()

    def render(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., :2] = (np.clip(self.state[:2], 0, 512)/512*255).astype(np.uint8)
        return image

    def step(self, action):
        assert action.shape == (2,) and np.max(np.abs(action)) <= 1.00001
        self.state[:2] += 100*action
        distance = float(np.linalg.norm(self.state[:2]-self.goal[:2]))
        return None, -distance, distance < .1, False, {}

    def close(self):
        self.closed = True


class ToyModel(nn.Module):
    horizon = 3
    direction_mode = "action_only"

    def encoder(self, frame):
        return frame[:, :2].mean((2, 3))

    def rollout(self, start, actions):
        return start[:, None]+actions.cumsum(1)


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def settings(self):
        settings = load_config("configs/continuous_trajectory_planning.yaml")
        settings.update(episodes=1, max_steps=2, num_candidates=8, num_elites=2,
                        iterations=2, candidate_batch_size=4, horizon=3, video_episodes=1)
        return settings

    def test_model_env_units_and_bounds(self):
        env = FakeEnv()
        low, high = action_bounds(env, 512., 100.)
        np.testing.assert_allclose(high, [100/512, 100/512])
        np.testing.assert_allclose(low, -high)
        np.testing.assert_allclose(environment_action([.1, -.1], env, 512., 100.), [.512, -.512])
        with self.assertRaises(ValueError):
            environment_action([1., 0.], env, 512., 100.)
        with self.assertRaises(ValueError):
            action_bounds(env, 0., 100.)

    def test_cem_recomputes_candidate_arrows_and_returns_evaluated_cost(self):
        cfg = self.settings()
        for mode in ("action_only", "state_action"):
            torch.manual_seed(42)
            model = TrajectoryWorldModel(image_size=32, hidden_dim=16, horizon=3,
                                         action_layers=1, direction_mode=mode).eval()
            observed = []
            hook = model.direction_head.register_forward_hook(lambda module, args, result: observed.append(len(result)))
            start, goal = torch.randn(1, 192), torch.randn(1, 192)
            planner = LatentTrajectoryCEM(cfg, [-.1, -.1], [.1, .1], "cpu")
            result = planner.plan(model, start, goal)
            hook.remove()
            self.assertEqual(observed, [4, 4, 4, 4])  # 2 CEM iterations, two fresh candidate batches each.
            self.assertTrue((result["actions"].abs() <= .100001).all())
            with torch.no_grad():
                expected = model.rollout(start, result["actions"][None])[0]
            torch.testing.assert_close(result["prediction"], expected)
            self.assertAlmostEqual(result["cost"], float((expected[-1]-goal[0]).square().mean()), places=5)
            self.assertFalse(result["prediction"].requires_grad)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            repeated = LatentTrajectoryCEM(cfg, [-.1, -.1], [.1, .1], "cpu").plan(model, start, goal)
            torch.testing.assert_close(result["actions"], repeated["actions"])

    def test_invalid_planner_settings(self):
        cfg = self.settings()
        cfg["execute_steps"] = 4
        with self.assertRaises(ValueError):
            LatentTrajectoryCEM(cfg, [-1, -1], [1, 1], "cpu")
        cfg = self.settings()
        planner = LatentTrajectoryCEM(cfg, [-1, -1], [1, 1], "cpu")
        cfg["horizon"] = 2
        with self.assertRaises(ValueError):
            planner.plan(ToyModel(), torch.zeros(1, 2), torch.ones(1, 2))

    def test_mpc_baselines_same_cases_outputs_videos_and_environment_success(self):
        settings = self.settings()
        envs = []
        def make_env(evaluator):
            env = FakeEnv()
            envs.append(env)
            return env
        case = dict(path="synthetic_test.npz", current=0,
                    start_state=np.array([100., 100., 256., 256., 0.], dtype=np.float32),
                    goal_state=np.array([110., 100., 256., 256., 0.], dtype=np.float32))
        config = dict(model={"image_size": 32}, data={"action_scale": 512.})
        with tempfile.TemporaryDirectory() as temp:
            with (patch("evaluation.planning.PlanningEvaluator._make_env", make_env),
                  patch("evaluation.planning.PlanningEvaluator._dataset_eval_cases", return_value=[case]),
                  patch("imageio.v2.mimsave") as videos):
                result = evaluate_planning(ToyModel(), config, settings, [], temp, torch.device("cpu"))
            self.assertEqual(set(result), {"cem", "zero", "random"})
            self.assertEqual(videos.call_count, 3)
            for call in videos.call_args_list:
                self.assertEqual(len(call.args[1]), 3)  # initial + two real steps
                self.assertEqual(call.args[1][0].shape, (32, 64, 3))
            self.assertAlmostEqual(result["zero"]["summary"]["mean_final_goal_distance"], 10.)
            self.assertEqual(result["zero"]["summary"]["success_rate_percent"], 0.)
            self.assertTrue(all(env.closed for env in envs))
            self.assertTrue((Path(temp)/"cases.json").exists())
            self.assertTrue((Path(temp)/"metrics.json").exists())
            self.assertEqual(len(list((Path(temp)/"cem").glob("*.npz"))), 2)
            saved = json.loads((Path(temp)/"cem/episode_000.json").read_text())
            self.assertEqual(saved["steps"], 2)
            self.assertIn("executed_endpoint_mse", saved["plans"][0])

    def test_colab_choose_run_entry_and_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root/"continuous_trajectory/action_only"
            (run/"checkpoints").mkdir(parents=True)
            (run/"checkpoints/latest.pt").write_bytes(b"test checkpoint identity")
            config = dict(experiment="continuous_trajectory", device="cpu", loss={"weight": .1},
                          model={"direction_mode": "action_only", "horizon": 3},
                          data={"action_mode": "relative", "action_scale": 512., "val_paths": []})
            (run/"config.yaml").write_text(yaml.safe_dump(config))
            self.assertEqual(discover_runs(root), [run])
            output = root/"results"
            with (patch("sys.argv", ["planning", "--choose", "--runs-root", str(root), "--output-dir", str(output)]),
                  patch("builtins.input", return_value="0"),
                  patch("experiments.continuous_dynamics.planning.build_model", return_value=ToyModel()),
                  patch("experiments.continuous_dynamics.planning.read_checkpoint", return_value={"config": config}),
                  patch("experiments.continuous_dynamics.planning.load_checkpoint"),
                  patch("experiments.continuous_dynamics.planning.evaluate_planning", return_value={}) as evaluate):
                main()
            self.assertEqual(evaluate.call_count, 1)
            metadata = json.loads((output/"metadata.json").read_text())
            self.assertEqual(metadata["settings"]["horizon"], 3)
            self.assertEqual(len(metadata["checkpoint_sha256"]), 64)

    def test_checkpoint_only_planning_without_saved_yaml(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root/"checkpoints/latest.pt"
            checkpoint.parent.mkdir()
            torch.save({"config": dict(experiment="continuous_trajectory", device="cpu",
                                      data={"action_mode": "relative", "action_scale": 512., "val_paths": []})},
                       checkpoint)
            with (patch("sys.argv", ["planning", "--checkpoint", str(checkpoint)]),
                  patch("experiments.continuous_dynamics.planning.build_model", return_value=ToyModel()),
                  patch("experiments.continuous_dynamics.planning.load_checkpoint"),
                  patch("experiments.continuous_dynamics.planning.evaluate_planning", return_value={}) as evaluate):
                main()
            self.assertEqual(evaluate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
