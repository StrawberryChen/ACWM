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

from experiments.continuous_dynamics.data import load_config
from experiments.continuous_dynamics.losses import objective
from experiments.continuous_dynamics.planning import LatentTrajectoryCEM
from experiments.continuous_dynamics.train import build_model, main, read_checkpoint
from experiments.continuous_dynamics.velocity_model import SequentialVelocityWorldModel


class DirectVelocityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_alignment_and_direct_gradient_without_euler(self):
        model = SequentialVelocityWorldModel(image_size=32, hidden_dim=8, horizon=5, readout_stride=5,
                                             start_fraction=1)
        frames, actions = torch.rand(2, 6, 3, 32, 32), torch.rand(2, 5, 2)
        with patch.object(model.integrator, "forward", side_effect=AssertionError("Euler must not run in training")):
            output = model(frames, actions)
            self.assertNotIn("prediction", output)
            real = output["real_latents"]
            torch.testing.assert_close(output["velocity_target"], real.diff(dim=1))
            torch.testing.assert_close(output["velocity_prediction"], model.field(real[:, :-1], actions))
            losses = objective(output, mode="velocity_sigreg", sigreg_projections=4)
            self.assertEqual(set(losses), {"loss_total", "loss_velocity", "loss_sigreg"})
            losses["loss_total"].backward()
        self.assertGreater(model.field.network[0].weight.grad.norm(), 0)
        self.assertGreater(model.encoder.projection.weight.grad.norm(), 0)

    def test_sequential_readout_causality_and_cem(self):
        model = SequentialVelocityWorldModel(image_size=32, hidden_dim=8, horizon=25, integration_steps=2).eval()
        start, actions = torch.randn(2, 192), torch.rand(2, 25, 2)
        changed = actions.clone()
        changed[:, 10:15] *= -1
        with torch.no_grad():
            full = model.rollout(start, actions)
            result = model.predict_checkpoints(start, actions)
            alternate = model.predict_checkpoints(start, changed)
        self.assertEqual(result.shape, (2, 5, 192))
        torch.testing.assert_close(result, full[:, 4::5])
        torch.testing.assert_close(result[:, :2], alternate[:, :2], rtol=0, atol=0)
        self.assertGreater((result[:, 2]-alternate[:, 2]).norm(), 0)
        settings = load_config("configs/continuous_trajectory_planning.yaml")
        settings.update(horizon=25, num_candidates=4, num_elites=2, iterations=2, candidate_batch_size=2)
        plan = LatentTrajectoryCEM(settings, [-.1, -.1], [.1, .1], "cpu").plan(model, start[:1], result[:1, -1])
        self.assertEqual(plan["actions"].shape, (25, 2))
        self.assertEqual(plan["prediction"].shape, (25, 192))

    def test_short_training_validation_and_frozen_encoder_probe(self):
        from experiments.continuous_dynamics.planning import main as planning
        from experiments.continuous_dynamics.probe import main as probe
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = load_config("configs/continuous_velocity_sigreg.yaml")
            config.update(device="cpu", output_dir=str(root/"run"))
            config["model"].update(image_size=32, hidden_dim=8, horizon=5, readout_stride=5)
            config["data"].update(batch_size=2, num_workers=0)
            config["training"].update(epochs=1)
            config["validation"].update(samples=2, visualization_samples=2, action_samples=2)
            config["loss"].update(sigreg_projections=4)
            config["probe"].update(epochs=1, max_train_samples=4, max_eval_samples=4)
            config["wandb"]["enabled"] = False
            for index, split in enumerate(("train", "val", "test")):
                path = root/f"{split}.npz"
                rng = np.random.default_rng(index)
                np.savez(path, frames=rng.random((7, 32, 32, 3), dtype=np.float32),
                         actions=rng.uniform(-.1, .1, (6, 2)).astype("float32"),
                         states=rng.random((7, 5), dtype=np.float32),
                         frame_skip=1, action_mode="relative", action_scale=512., source_episode=index)
                config["data"][f"{split}_paths"] = [str(path)]
            path = root/"config.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch("sys.argv", ["train", "--config", str(path)]):
                main()
            checkpoint = root/"run/checkpoints/latest.pt"
            saved = read_checkpoint(checkpoint)
            self.assertIsInstance(build_model(saved["config"]), SequentialVelocityWorldModel)
            for key in ("val/loss_velocity", "val/velocity_zero_mse", "val/loss_sigreg", "val/pred_step_5",
                        "val/identity_loss", "action/velocity_shuffle_loss_delta", "action/shuffle_loss_delta"):
                self.assertIn(key, saved["metrics"])
            self.assertNotIn("val/loss_progress", saved["metrics"])
            (root/"run/config.yaml").unlink()
            with patch("sys.argv", ["probe", "--checkpoint", str(checkpoint), "--no-wandb"]):
                probe()
            metrics = root/"run/probes/latest/metrics.json"
            self.assertTrue(json.loads(metrics.read_text()))
            with (patch("sys.argv", ["planning", "--checkpoint", str(checkpoint),
                                    "--output-dir", str(root/"planning")]),
                  patch("experiments.continuous_dynamics.planning.evaluate_planning", return_value={}) as run):
                planning()
            self.assertEqual(run.call_args.args[2]["goal_offset_steps"], 5)
            self.assertEqual(json.loads((root/"planning/metadata.json").read_text())["trajectory_readout_stride"], 5)
