"""CPU smoke tests, runnable with unittest as well as pytest."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from experiments.continuous_dynamics.model import ContinuousWorldModel, EulerIntegrator, VelocityField
from experiments.continuous_dynamics.losses import objective, reference_progress, DirectionalProgressLoss
from experiments.continuous_dynamics.diagnostics import CollapseDiagnostics, evaluate
from experiments.continuous_dynamics.data import load_config, loader, validate_splits


class ContinuousDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vit_shapes_shared_encoder_gradients_and_no_bn(self):
        torch.manual_seed(1)
        model = ContinuousWorldModel(image_size=32, hidden_dim=16)
        self.assertFalse(any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()))
        frames = torch.rand(2, 2, 3, 32, 32, requires_grad=True)
        actions = torch.rand(2, 1, 2, requires_grad=True)
        captured = []
        def hook(module, args, result):
            result.retain_grad()
            captured.append(result)
        handle = model.field.register_forward_hook(hook)
        output = model(frames, actions)
        output["target"].retain_grad()
        self.assertEqual(output["prediction"].shape, (2, 192))
        self.assertEqual(output["states"].shape, (2, 5, 192))
        self.assertEqual(output["velocities"].shape, (2, 4, 192))
        objective(output)["loss_total"].backward()
        handle.remove()
        self.assertGreater(output["target"].grad.norm(), 0)
        for side in (0, 1):
            self.assertGreater(frames.grad[:, side].norm(), 0)
        self.assertGreater(actions.grad.norm(), 0)
        self.assertTrue(all(v.grad is not None and v.grad.norm() > 0 for v in captured))
        self.assertTrue(all(any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
                            for module in (model.encoder.backbone, model.encoder.projection, model.field.action_encoder, model.field.network)))
        self.assertEqual(model.rollout(output["start"].detach(), torch.rand(2, 3, 2)).shape, (2, 3, 192))

    def test_exact_euler_and_k_ablations(self):
        class Field(nn.Module):
            def forward(self, z, a):
                return z+a
        for k in (1, 2, 4, 8):
            start, action = torch.ones(2, 2, requires_grad=True), torch.ones(2, 2)
            end, path, velocities = EulerIntegrator(k)(Field(), start, action)
            manual = start
            for _ in range(k):
                manual = manual+(manual+action)/k
            torch.testing.assert_close(end, manual)
            self.assertEqual(velocities.shape[1], k)
            end.sum().backward()
            self.assertGreater(start.grad.min(), 0)

    def test_direction_is_per_sample_and_only_direction_detached(self):
        v = torch.tensor([[[3., 4.], [0., 5.]], [[0., 2.], [3., 0.]]], requires_grad=True)
        p = reference_progress(v)
        torch.testing.assert_close(p, torch.tensor([[5., 4.], [2., 0.]]))
        p.sum().backward()
        torch.testing.assert_close(v.grad, torch.tensor([[[.6, .8], [.6, .8]], [[0., 1.], [0., 1.]]]))

    def test_zero_velocity_stationary_point_is_not_hidden(self):
        v = torch.zeros(2, 4, 192, requires_grad=True)
        loss = DirectionalProgressLoss(.1)(v)
        self.assertAlmostEqual(float(loss), .1, places=6)
        loss.backward()
        self.assertEqual(float(v.grad.abs().max()), 0.)  # positive loss, zero gradient!
        self.assertEqual(float(DirectionalProgressLoss(0)(v)), 0.)

    def test_constant_encoder_can_have_small_nonzero_compromise(self):
        # Constant representation cannot make both terms zero, but MSE averages D.
        start = torch.zeros(2, 192)
        v = torch.zeros(2, 4, 192)
        v[..., 0] = .100001
        output = dict(prediction=start+v.mean(1), target=start, velocities=v)
        losses = objective(output)
        self.assertEqual(float(losses["loss_progress"]), 0.)
        self.assertLess(float(losses["loss_total"]), .000053)

    def test_ablation_modes_backward_and_invalid_settings(self):
        for mode in ("prediction_only", "nonzero_velocity", "directional"):
            z = torch.randn(2, 192, requires_grad=True)
            target = torch.randn_like(z, requires_grad=True)
            v = torch.randn(2, 4, 192, requires_grad=True)
            loss = objective(dict(prediction=z, target=target, velocities=v), mode=mode)
            loss["loss_total"].backward()
            self.assertGreater(target.grad.norm(), 0)
        with self.assertRaises(ValueError):
            EulerIntegrator(0)
        with self.assertRaises(ValueError):
            objective({}, mode="flow_matching")

    def test_collapse_metrics(self):
        values = CollapseDiagnostics.compute(torch.ones(8, 192))
        self.assertEqual(values["collapse/feature_std"], 0)
        self.assertEqual(values["collapse/effective_rank"], 0)
        self.assertAlmostEqual(values["collapse/pairwise_cosine"], 1.)
        self.assertGreater(CollapseDiagnostics.compute(torch.randn(30, 192))["collapse/effective_rank"], 10)

    def fixture(self, root):
        config = load_config("configs/continuous_dynamics.yaml")
        config["device"], config["wandb"]["enabled"] = "cpu", False
        config["output_dir"] = str(root/"output")
        config["model"].update(image_size=32, hidden_dim=16)
        config["training"]["epochs"] = 1
        config["data"].update(batch_size=2, num_workers=0)
        config["validation"].update(samples=4, visualization_samples=4, action_samples=3)
        config["probe"].update(epochs=1, max_train_samples=4, max_eval_samples=4)
        rng = np.random.default_rng(1)
        for ep, split in enumerate(("train", "val", "test")):
            path = root/f"{split}.npz"
            frames = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
            actions = rng.uniform(-1, 1, (4, 2)).astype("float32")
            np.savez(path, frames=frames, actions=actions, states=rng.normal(size=(5, 5)).astype("float32"),
                     source_episode=ep, frame_skip=1, action_mode="relative", action_scale=512.)
            config["data"][f"{split}_paths"] = [str(path)]
        return config

    def test_dataset_alignment_no_state_labels_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.fixture(Path(temp))
            validate_splits(config)
            batches, dataset = loader(config, "val")
            sample = dataset[2]
            with np.load(dataset.paths[0]) as ep:
                torch.testing.assert_close(sample["actions"][0], torch.from_numpy(ep["actions"][2]))
                torch.testing.assert_close(sample["frames"][0], torch.from_numpy(ep["frames"][2]).permute(2, 0, 1).float()/255)
                torch.testing.assert_close(sample["frames"][1], torch.from_numpy(ep["frames"][3]).permute(2, 0, 1).float()/255)
            self.assertNotIn("states", sample)
            model = ContinuousWorldModel(**config["model"])
            metrics, _ = evaluate(model, batches, config, torch.device("cpu"))
            self.assertIn("action/velocity_distance", metrics)
            self.assertIn("action/shuffle_loss", metrics)
            self.assertEqual(metrics["validation/samples"], 4)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            config["data"]["test_paths"] = config["data"]["val_paths"]
            with self.assertRaises(ValueError):
                validate_splits(config)

    def test_end_to_end_train_evaluate_probe_and_resume(self):
        from experiments.continuous_dynamics.train import main
        from experiments.continuous_dynamics.probe import main as probe
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = self.fixture(root)
            path = root/"config.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch("sys.argv", ["train", "--config", str(path)]):
                main()
            checkpoint = root/"output/checkpoints/latest.pt"
            self.assertTrue(checkpoint.exists())
            self.assertTrue((root/"output/checkpoints/best_loss.pt").exists())
            self.assertEqual(len(list((root/"output/visualizations").glob("*.png"))), 3)
            args = ["train", "--config", str(path), "--checkpoint", str(checkpoint)]
            with patch("sys.argv", args+["--mode", "evaluate"]):
                main()
            self.assertTrue((root/"output/evaluation_val/metrics.json").exists())
            with patch("sys.argv", args+["--resume"]):
                main()
            with patch("sys.argv", ["probe", "--config", str(path), "--checkpoint", str(checkpoint)]):
                probe()
            self.assertTrue((root/"output/probes/latest/heads.pt").exists())

    def test_prepared_config_keeps_new_model_and_loss(self):
        from experiments.continuous_dynamics.data import resolve
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = self.fixture(root)
            old["model"] = {"latent_dim": 128, "horizon": 16}
            old["loss"] = {"lambda_motion": 1}
            path = root/"prepared.yaml"
            path.write_text(yaml.safe_dump(old))
            result = resolve("configs/continuous_dynamics.yaml", path)
            self.assertEqual(result["model"]["latent_dim"], 192)
            self.assertEqual(result["model"]["integration_steps"], 4)
            self.assertEqual(result["loss"]["mode"], "directional")
            self.assertEqual(result["data"]["val_paths"], old["data"]["val_paths"])


if __name__ == "__main__":
    unittest.main()
