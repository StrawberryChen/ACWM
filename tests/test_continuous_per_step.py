import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from experiments.continuous_dynamics.data import load_config, loader, resolve
from experiments.continuous_dynamics.train import build_model, main, load_checkpoint
from experiments.continuous_dynamics.losses import objective, output_progress
from experiments.continuous_dynamics.diagnostics import evaluate
from experiments.continuous_dynamics.planning import LatentTrajectoryCEM


class PerStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, mode):
        path = ("configs/continuous_trajectory_action_per_step.yaml" if mode == "action_only" else
                "configs/continuous_trajectory_state_action_per_step.yaml")
        config = load_config(path)
        config["model"].update(image_size=32, hidden_dim=16, horizon=3, action_layers=1)
        config["device"], config["wandb"]["enabled"] = "cpu", False
        return config

    def test_each_h_uses_own_arrow_for_all_and_only_its_internal_steps(self):
        # The local loss must allow a real reversal between action intervals.
        direction = torch.tensor([[[1., 0.], [-1., 0.]]], requires_grad=True)
        velocity = direction.detach()[:, :, None].expand(1, 2, 4, 2).clone().requires_grad_()
        progress = output_progress(dict(direction=direction, velocities=velocity))
        torch.testing.assert_close(progress, torch.ones(1, 2, 4))
        progress.sum().backward()
        self.assertIsNone(direction.grad)
        torch.testing.assert_close(velocity.grad, direction.detach()[:, :, None].expand_as(velocity))

    def test_only_scope_changes_not_parameter_shapes_or_initialization(self):
        for mode in ("action_only", "state_action"):
            config = self.config(mode)
            torch.manual_seed(42)
            local = build_model(config)
            config["model"]["direction_scope"] = "global"
            torch.manual_seed(42)
            global_model = build_model(config)
            self.assertEqual(set(local.state_dict()), set(global_model.state_dict()))
            for key, value in local.state_dict().items():
                torch.testing.assert_close(value, global_model.state_dict()[key])

    def test_standalone_yamls_have_fixed_paths_and_independent_parameters(self):
        for path, mode in (("configs/continuous_trajectory_action_per_step.yaml", "action_only"),
                           ("configs/continuous_trajectory_state_action_per_step.yaml", "state_action")):
            raw = yaml.safe_load(Path(path).read_text())
            self.assertEqual(raw["base"], "continuous_dynamics.yaml")
            config = resolve(path)  # no data config or generated temporary file
            self.assertEqual(config["model"]["direction_mode"], mode)
            self.assertEqual(config["model"]["direction_scope"], "per_step")
            for split in ("train", "val", "test"):
                self.assertEqual(config["data"][f"{split}_paths"], [
                    f"/content/drive/MyDrive/ACWM/continuous_trajectory_fresh_20261104_190724/avs_target/data/{split}/episode_*.npz"])
            self.assertEqual(config["validation"]["interval"], 5)
            self.assertIn("training", raw)
            self.assertIn("probe", raw)
            self.assertIn("loss", raw)

    def test_shapes_gradients_and_no_future_image_leak_both_modes(self):
        frames, actions = torch.rand(2, 4, 3, 32, 32), torch.rand(2, 3, 2)
        for mode in ("action_only", "state_action"):
            model = build_model(self.config(mode)).eval()
            for key in ("loss_pred", "loss_progress"):
                model.zero_grad(set_to_none=True)
                source = frames.clone().requires_grad_()
                captured = []
                def hook(module, inputs, value):
                    value.retain_grad()
                    captured.append(value)
                handle = model.direction_head.register_forward_hook(hook)
                output = model(source, actions)
                handle.remove()
                self.assertEqual(output["direction"].shape, (2, 3, 192))
                self.assertEqual(output["states"].shape, (2, 3, 5, 192))
                torch.testing.assert_close(output["direction"].norm(dim=-1), torch.ones(2, 3))
                torch.testing.assert_close(output["states"][:, 1:, 0], output["prediction"][:, :-1])
                output["target"].retain_grad()
                objective(output, margin=10.)[key].backward()
                self.assertEqual(len(captured), 3)
                self.assertTrue(all(value.grad is not None and value.grad.norm() > 0 for value in captured))
                for module in (model.direction_head, model.action_trajectory, model.field, model.encoder):
                    self.assertGreater(sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None), 0)
                if key == "loss_pred":
                    self.assertGreater(output["target"].grad.norm(), 0)
                else:
                    self.assertEqual(float(source.grad[:, 1:].norm()), 0)
            with torch.no_grad():
                original = model(frames, actions)
                changed = frames.clone()
                changed[:, 1:] = torch.rand_like(changed[:, 1:])
                alternative = model(changed, actions)
                torch.testing.assert_close(original["direction"], alternative["direction"])
                torch.testing.assert_close(original["prediction"], alternative["prediction"])
                self.assertGreater((original["direction"][:, 1:]-original["direction"][:, :-1]).abs().max(), 1e-7)

    def test_state_action_heads_use_predicted_state_at_each_interval(self):
        model = build_model(self.config("state_action")).eval()
        frames, actions = torch.rand(2, 4, 3, 32, 32), torch.rand(2, 3, 2)
        inputs = []
        handle = model.direction_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
        with torch.no_grad():
            output = model(frames, actions)
        handle.remove()
        for h in range(3):
            torch.testing.assert_close(inputs[h][:, :192], output["states"][:, h, 0])

    def fixture(self, root, mode):
        config = self.config(mode)
        config["output_dir"] = str(root/(mode+"_per_step"))
        config["data"].update(batch_size=2, num_workers=0)
        config["training"]["epochs"] = 1
        config["validation"].update(samples=4, visualization_samples=4, action_samples=2)
        config["probe"].update(epochs=1, max_train_samples=4, max_eval_samples=4)
        for ep, split in enumerate(("train", "val", "test")):
            path = root/f"{split}.npz"
            if not path.exists():
                rng = np.random.default_rng(ep)
                np.savez(path, frames=rng.random((7, 32, 32, 3), dtype=np.float32),
                         actions=rng.uniform(-1, 1, (6, 2)).astype("float32"),
                         states=rng.random((7, 5), dtype=np.float32), source_episode=ep,
                         frame_skip=1, action_mode="relative", action_scale=512.)
            config["data"][f"{split}_paths"] = [str(path)]
        return config

    def test_diagnostics_and_candidate_planning_recompute_local_arrows(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("action_only", "state_action"):
                config = self.fixture(Path(temp), mode)
                model = build_model(config)
                batches, _ = loader(config, "val")
                metrics, values = evaluate(model, batches, config, torch.device("cpu"))
                self.assertIn("direction/temporal_cosine", metrics)
                self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
                settings = load_config("configs/continuous_trajectory_planning.yaml")
                settings.update(horizon=3, num_candidates=4, num_elites=2, iterations=2, candidate_batch_size=2)
                calls = []
                handle = model.direction_head.register_forward_hook(lambda module, args, value: calls.append(value.shape))
                result = LatentTrajectoryCEM(settings, [-.1, -.1], [.1, .1], "cpu").plan(
                    model, values["start"][:1], values["target"][:1, -1])
                handle.remove()
                self.assertEqual(len(calls), 12)  # 2 iterations × 2 batches × 3 distinct arrows.
                self.assertEqual(result["prediction"].shape, (3, 192))

    def test_train_evaluate_probe_and_checkpoint_scope_isolation(self):
        from experiments.continuous_dynamics.probe import main as probe
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            for mode in ("action_only", "state_action"):
                config = self.fixture(root, mode)
                path = root/f"{mode}.yaml"
                path.write_text(yaml.safe_dump(config))
                with patch("sys.argv", ["train", "--config", str(path)]):
                    main()
                run = Path(config["output_dir"])
                checkpoint = run/"checkpoints/latest.pt"
                with patch("sys.argv", ["eval", "--config", str(path), "--checkpoint", str(checkpoint), "--mode", "evaluate"]):
                    main()
                (run/"config.yaml").unlink()
                with patch("sys.argv", ["probe", "--checkpoint", str(checkpoint)]):
                    probe()
                self.assertTrue((run/"probes/latest/metrics.json").exists())
                with np.load(run/"visualizations/latents_0001.npz") as plot:
                    self.assertEqual(plot["directions"].shape, (1, 3, 192))
                config["model"].pop("direction_scope")
                with self.assertRaises(ValueError):
                    load_checkpoint(checkpoint, config, build_model(config))
                resolved = resolve(path, data_config=path)
                self.assertTrue(resolved["output_dir"].endswith(mode+"_per_step"))


if __name__ == "__main__":
    unittest.main()
