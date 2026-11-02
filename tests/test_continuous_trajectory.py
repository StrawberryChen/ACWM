"""Trainable sequence arrows: isolated modes, full rollout, no future-image leak."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from experiments.continuous_dynamics.data import load_config, loader, resolve
from experiments.continuous_dynamics.trajectory_model import TrajectoryWorldModel
from experiments.continuous_dynamics.losses import objective, output_progress
from experiments.continuous_dynamics.diagnostics import evaluate
from experiments.continuous_dynamics.train import build_model, load_checkpoint, main


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, mode):
        torch.manual_seed(42)
        return TrajectoryWorldModel(image_size=32, hidden_dim=16, horizon=3,
                                    action_layers=1, direction_mode=mode)

    def test_shapes_open_loop_and_future_images_are_labels_only(self):
        frames, actions = torch.rand(2, 4, 3, 32, 32), torch.rand(2, 3, 2)
        for mode in ("action_only", "state_action"):
            model = self.model(mode).eval()
            with torch.no_grad():
                output = model(frames, actions)
                changed = frames.clone()
                changed[:, 1:] = torch.rand_like(changed[:, 1:])
                alternative = model(changed, actions)
            self.assertEqual(output["prediction"].shape, (2, 3, 192))
            self.assertEqual(output["states"].shape, (2, 3, 5, 192))
            self.assertEqual(output["velocities"].shape, (2, 3, 4, 192))
            torch.testing.assert_close(output["prediction"], alternative["prediction"])
            torch.testing.assert_close(output["direction"], alternative["direction"])
            self.assertGreater((output["target"]-alternative["target"]).abs().max(), 0)
            torch.testing.assert_close(output["states"][:, 1:, 0], output["prediction"][:, :-1])
            torch.testing.assert_close(output["direction"].norm(dim=-1), torch.ones(2))

    def test_both_direction_heads_receive_both_loss_gradients(self):
        for mode in ("action_only", "state_action"):
            model = self.model(mode)
            frames = torch.rand(2, 4, 3, 32, 32, requires_grad=True)
            actions = torch.rand(2, 3, 2, requires_grad=True)
            for loss_name in ("loss_pred", "loss_progress"):
                model.zero_grad(set_to_none=True)
                frames.grad, actions.grad = None, None
                output = model(frames, actions)
                output["direction"].retain_grad()
                output["target"].retain_grad()
                # Large margin makes the directional hinge active for this gradient test.
                objective(output, margin=10.)[loss_name].backward()
                for module in (model.direction_head, model.action_trajectory, model.field, model.encoder):
                    norm = sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)
                    self.assertGreater(norm, 0, (mode, loss_name, type(module).__name__))
                self.assertGreater(output["direction"].grad.norm(), 0)
                self.assertGreater(actions.grad.norm(), 0)
                if loss_name == "loss_pred":
                    self.assertGreater(output["target"].grad.norm(), 0)
                    self.assertGreater(frames.grad[:, 1:].norm(), 0)
                else:
                    # Future image targets DO NOT participate in progress or dynamics.
                    self.assertIsNone(output["target"].grad)
                    self.assertEqual(float(frames.grad[:, 1:].norm()), 0.)

    def test_reference_detach_only_not_live_dynamics_direction(self):
        velocity = torch.randn(2, 3, 4, 192, requires_grad=True)
        direction = torch.randn(2, 192, requires_grad=True)
        output_progress(dict(velocities=velocity, direction=direction)).sum().backward()
        self.assertIsNone(direction.grad)  # loss-reference branch detached
        torch.testing.assert_close(velocity.grad, direction.detach()[:, None, None].expand_as(velocity))

    def test_direction_mode_input_contract_and_matched_random_encoder(self):
        models = [self.model(mode) for mode in ("action_only", "state_action")]
        for a, b in zip(models[0].encoder.parameters(), models[1].encoder.parameters()):
            torch.testing.assert_close(a, b)
        actions = torch.rand(2, 3, 2)
        first, other = torch.randn(2, 192), torch.randn(2, 192)
        for model in models:
            arrow1 = model.condition(first, actions)[-1]
            arrow2 = model.condition(other, actions)[-1]
            if model.direction_mode == "action_only":
                torch.testing.assert_close(arrow1, arrow2)
            else:
                self.assertGreater((arrow1-arrow2).abs().max(), 1e-3)
            reordered = model.condition(first, actions.flip(1))[-1]
            self.assertGreater((arrow1-reordered).abs().max(), 1e-7)

    def fixture(self, root, mode):
        config = load_config("configs/continuous_trajectory_action.yaml")
        config["model"].update(image_size=32, hidden_dim=16, horizon=3, action_layers=1, direction_mode=mode)
        config["device"], config["wandb"]["enabled"] = "cpu", False
        config["output_dir"] = str(root/mode)
        config["data"].update(batch_size=2, num_workers=0)
        config["training"]["epochs"] = 1
        config["validation"].update(samples=4, action_samples=2, visualization_samples=4)
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

    def test_dataset_horizon_and_metrics_and_probe_one_frame_override(self):
        with tempfile.TemporaryDirectory() as temp:
            for mode in ("action_only", "state_action"):
                config = self.fixture(Path(temp), mode)
                batches, dataset = loader(config, "val")
                self.assertEqual(len(dataset), 4)
                self.assertEqual(dataset[3]["frames"].shape, (4, 3, 32, 32))
                with np.load(dataset.paths[0]) as data:
                    torch.testing.assert_close(dataset[3]["actions"], torch.from_numpy(data["actions"][3:6]))
                _, probe_dataset = loader(config, "val", horizon=1)
                self.assertEqual(len(probe_dataset), 6)
                model = build_model(config)
                metrics, values = evaluate(model, batches, config, torch.device("cpu"))
                self.assertEqual(values["direction"].shape, (4, 192))
                self.assertIn("direction/action_sequence_distance", metrics)
                self.assertIn("val/identity_loss", metrics)
                self.assertIn("val/pred_step_3", metrics)
                self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_both_variants_train_checkpoint_evaluate_and_probe(self):
        from experiments.continuous_dynamics.probe import main as probe
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            for mode in ("action_only", "state_action"):
                config = self.fixture(root, mode)
                path = root/f"{mode}.yaml"
                path.write_text(yaml.safe_dump(config))
                with patch("sys.argv", ["train", "--config", str(path)]):
                    main()
                checkpoint = root/mode/"checkpoints/latest.pt"
                saved = torch.load(checkpoint, weights_only=False)
                logged = json.loads((root/mode/"metrics.jsonl").read_text().splitlines()[0])
                self.assertGreater(logged["train/direction_head_gradient_norm"], 0)
                self.assertGreater(logged["train/action_trajectory_gradient_norm"], 0)
                self.assertGreater(saved["metrics"]["val/loss_pred"], 0)
                initial = self.model(mode)
                self.assertGreater((saved["model"]["direction_head.2.weight"]-initial.direction_head[2].weight).norm(), 0)
                args = ["evaluate", "--config", str(path), "--checkpoint", str(checkpoint), "--mode", "evaluate"]
                with patch("sys.argv", args):
                    main()
                with patch("sys.argv", ["probe", "--config", str(path), "--checkpoint", str(checkpoint)]):
                    probe()
                self.assertTrue((root/mode/"evaluation_val/metrics.json").exists())
                self.assertTrue((root/mode/"probes/latest/heads.pt").exists())
                self.assertEqual(len(list((root/mode/"visualizations").glob("*.png"))), 3)
                wrong = self.fixture(root, "state_action" if mode == "action_only" else "action_only")
                with self.assertRaises(ValueError):
                    load_checkpoint(checkpoint, wrong, build_model(wrong))


if __name__ == "__main__":
    unittest.main()
