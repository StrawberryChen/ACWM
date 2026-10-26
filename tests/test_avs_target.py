import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from experiments.avs_target.data import TrajectoryWindows, load_config, loader, source_signature, validate_splits
from experiments.avs_target.evaluate import collapse_metrics, motion_metrics
from experiments.avs_target.flow import euler_sample, flow_loss, flow_training_pair
from experiments.avs_target.model import AVSFlowWorldModel, CurrentStateEncoder
from experiments.avs_target.targets import FixedMotionTarget, fit_statistics, shifted


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_static_and_directional_motion_no_wraparound():
    operator = FixedMotionTarget(resolution=12)
    image = torch.zeros(2, 3, 96, 96)
    image[:, :, 30:60, 20:40] = 1
    stationary = torch.stack((image, image), 1)
    assert operator(stationary, normalize=False).abs().max() == 0
    following = shifted(image, 4, 0)
    value = operator.maps(operator(torch.stack((image, following), 1), normalize=False))
    energy = value.sum((0, 1, 3, 4))
    assert energy[0] > energy[4] and energy[0] > 0
    # Opposite brightness polarity must preserve direction, not reverse it.
    inverse = operator.maps(operator(torch.stack((1-image, 1-following), 1), normalize=False))
    reverse_energy = inverse.sum((0, 1, 3, 4))
    assert reverse_energy[0] > reverse_energy[4]
    edge = torch.zeros(1, 1, 8, 8)
    edge[..., -1] = 1
    assert shifted(edge, 1, 0).sum() == 0


@pytest.mark.parametrize("kind,dimension", [("avs", 288), ("pixel_diff", 108), ("random_projection", 288)])
def test_fixed_targets_shapes_no_gradient_and_reproducibility(kind, dimension):
    operator = FixedMotionTarget(kind=kind)
    frames = torch.rand(2, 5, 3, 96, 96, requires_grad=True)
    result = operator(frames)
    assert result.shape == (2, 4, dimension) and not result.requires_grad
    assert not list(operator.parameters())
    torch.testing.assert_close(result, FixedMotionTarget(kind=kind)(frames))
    assert result[:, 0].var() > 1e-15


def test_flow_linear_path_velocity_and_euler():
    source = torch.tensor([[[-1., 2.]]])
    target = torch.tensor([[[3., 0.]]])
    noisy, velocity = flow_training_pair(target, source, torch.tensor([0.5]))
    torch.testing.assert_close(noisy, torch.tensor([[[1., 1.]]]))
    torch.testing.assert_close(velocity, torch.tensor([[[4., -2.]]]))
    result = euler_sample(lambda x, s, o, a: velocity, torch.zeros(1, 2), torch.zeros(1, 1, 2), source, 4)
    torch.testing.assert_close(result, target)
    assert flow_loss(velocity, velocity) == 0
    with pytest.raises(ValueError):
        euler_sample(None, None, None, source, 0)


def test_forward_backward_cse_action_gradients_and_no_future_encoder():
    torch.manual_seed(42)
    operator = FixedMotionTarget(resolution=3)
    model = AVSFlowWorldModel(operator.output_dim, latent_dim=16, horizon=4, model_dim=32, layers=1, heads=4)
    frames = torch.rand(2, 5, 3, 96, 96, requires_grad=True)
    actions = torch.rand(2, 4, 2)
    target = operator(frames)
    noisy, velocity = flow_training_pair(target, torch.randn_like(target), torch.tensor([0.2, 0.8]))
    calls = []
    hook = model.cse.register_forward_hook(lambda _, args, out: calls.append(args[0].shape))
    loss = flow_loss(model(frames[:, 0], actions, noisy, torch.tensor([0.2, 0.8])), velocity)
    loss.backward()
    assert calls == [torch.Size((2, 3, 96, 96))]
    assert frames.grad[:, 0].abs().sum() > 0 and frames.grad[:, 1:].abs().sum() == 0
    for module in (model.cse, model.predictor.action_encoder, model.predictor.transformer):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    assert not any(isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) for m in model.cse.modules())
    hook.remove()


def fixture_config(tmp_path):
    config = load_config("configs/avs_flow.yaml")
    config.update(device="cpu", output_dir=str(tmp_path/"outputs"))
    config["data"].update(batch_size=2, num_workers=0, stats_batch_size=2)
    config["model"].update(latent_dim=16, horizon=4, model_dim=32, layers=1, heads=4)
    config["target"].update(resolution=3, pair_chunk=8)
    config["training"]["epochs"] = 1
    config["validation"].update(samples=4, generation_samples=4, horizons=[1, 2, 4], flow_steps=[1, 2],
                                 visualization_examples=2, comparison_videos=1)
    config["probe"].update(max_train_samples=8, max_eval_samples=4, epochs=2)
    config["wandb"]["enabled"] = False
    for episode, split in enumerate(("train", "val", "test")):
        path = tmp_path/f"episode_{episode}.npz"
        frames = np.zeros((9, 96, 96, 3), dtype=np.uint8)
        states = np.zeros((9, 5), dtype=np.float32)
        for t in range(9):
            frames[t, 20:40, 20+t:40+t] = 255
            states[t] = [100+t, 100+episode, 256+t, 256, t/10]
        np.savez_compressed(path, frames=frames, actions=np.full((8, 2), .2, np.float32), states=states,
                            source_episode=episode, frame_skip=1, action_mode="relative", action_scale=512.)
        config["data"][f"{split}_paths"] = [str(path)]
    return config


def test_sliding_windows_alignment_first_transition_and_train_only_statistics(tmp_path):
    config = fixture_config(tmp_path)
    validate_splits(config)
    dataset = TrajectoryWindows(config["data"]["train_paths"], 4, action_scale=512.)
    sample = dataset[2]
    assert len(dataset) == 5 and sample["frames"].shape == (5, 3, 96, 96)
    assert set(sample) == {"frames", "actions", "episode", "timestep"}
    operator = FixedMotionTarget(resolution=3)
    value = operator(sample["frames"][None], normalize=False)
    first = operator(sample["frames"][None, :2], normalize=False)
    torch.testing.assert_close(value[:, :1], first)
    batches, _ = loader(config, "train", horizon=1)
    path = tmp_path/"stats.json"
    saved = fit_statistics(operator, batches, "cpu", path, source_signature(dataset.paths))
    assert saved["count"] == 8*3*3  # 8 UNIQUE pairs, not five overlapping H=4 windows
    normalized = operator(sample["frames"][None])
    torch.testing.assert_close(operator.denormalize(normalized), value)
    clone = FixedMotionTarget(resolution=3)
    clone.load_statistics(path)
    torch.testing.assert_close(clone(sample["frames"][None]), normalized)
    with pytest.raises(ValueError, match="different"):
        FixedMotionTarget(resolution=6).load_statistics(path)
    config["data"]["test_paths"] = config["data"]["train_paths"]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(config)


def test_generation_metrics_mask_static_direction_and_scale_collapse():
    operator = FixedMotionTarget(resolution=3)
    target = torch.zeros(2, 4, operator.output_dim)
    metrics = motion_metrics(target, target, operator, 1e-6)
    assert metrics["mse"] == 0 and metrics["direction_accuracy"] is None
    z = torch.randn(32, 16)
    normal, small = collapse_metrics(z), collapse_metrics(z*1e-5)
    assert small["representation/feature_std"] < 1e-4
    assert small["representation/effective_rank"] == pytest.approx(normal["representation/effective_rank"], rel=1e-5)


@pytest.mark.parametrize("kind", ["avs", "pixel_diff", "random_projection"])
def test_full_train_resume_evaluate_visualizations_and_frozen_probes(tmp_path, monkeypatch, kind):
    from experiments.avs_target import train, probe
    config = fixture_config(tmp_path)
    config["target"]["kind"] = kind
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    train.main()
    output = Path(config["output_dir"])
    assert (output/"avs_stats.json").exists()
    checkpoint = output/"checkpoints/latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["epoch"] == 1 and "generation/k2/h4/mse" in saved["metrics"]
    assert "dependency/s0/state_delta" in saved["metrics"]
    assert len(list((output/"target_visualizations").glob("*.png"))) == 2
    assert (output/"videos/epoch_0001/trajectory_000.mp4").exists()
    config["training"]["epochs"] = 2
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--checkpoint", str(checkpoint), "--resume"])
    train.main()
    assert torch.load(checkpoint, weights_only=False)["epoch"] == 2
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(path), "--checkpoint", str(checkpoint), "--mode", "evaluate", "--split", "test"])
    train.main()
    assert (output/"evaluation_test/metrics.jsonl").exists()
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(path), "--checkpoint", str(checkpoint)])
    probe.main()
    metrics = json.loads((output/"probes/metrics.json").read_text())
    assert "probe/random_cse/linear/test/block_orientation_mae_degrees" in metrics
    assert "probe/trained_cse/mlp/test/agent_position_rmse_pixels" in metrics


def test_deterministic_sigreg_control_is_separate(tmp_path):
    from experiments.avs_target.baseline import LeWorldStyleControl
    from losses.latent import SIGRegLoss
    model = LeWorldStyleControl(latent_dim=16)
    frames = torch.rand(2, 5, 3, 96, 96)
    total, pred, reg, _ = model.losses(frames, torch.rand(2, 4, 2), SIGRegLoss(5, 8), .09)
    torch.testing.assert_close(total, pred+.09*reg)
    total.backward()
    assert sum(p.grad.abs().sum() for p in model.cse.parameters() if p.grad is not None) > 0


def test_baseline_entrypoint_and_probe_compatible_checkpoint(tmp_path, monkeypatch):
    from experiments.avs_target import baseline
    config = fixture_config(tmp_path)
    config["baseline"] = dict(sigreg_weight=.09, knots=5, num_projections=8)
    path = tmp_path/"baseline.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["baseline", "--config", str(path)])
    baseline.main()
    saved = torch.load(Path(config["output_dir"])/"checkpoints/best_prediction.pt", weights_only=False)
    assert "initial_cse" in saved and "val/sigreg_loss" in saved["metrics"]
    assert "flow_loss" not in saved["metrics"]


@pytest.mark.parametrize("resolution,directions", [(3, 4), (6, 8), (12, 8)])
def test_target_geometry_ablation(resolution, directions):
    operator = FixedMotionTarget(resolution=resolution, directions=directions)
    frames = torch.rand(1, 2, 3, 96, 96)
    assert operator(frames).shape == (1, 1, directions*resolution**2)


def test_preparation_no_clipping_no_windows_and_ablations_share_data(tmp_path, monkeypatch):
    import scripts.prepare_avs_target as prepare
    states = np.tile(np.array([100, 100, 256, 256, 0], np.float32), (90, 1))
    source = {"data": {"state": states, "action": states[:, :2]+[25, -50],
                       "img": np.zeros((90, 96, 96, 3), np.uint8)},
              "meta": {"episode_ends": np.array([30, 60, 90])}}
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open=lambda *a, **k: source))
    monkeypatch.setattr(prepare, "ensure_zarr", lambda args: tmp_path)
    config_path = tmp_path/"colab.yaml"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-root", str(tmp_path), "--config-output", str(config_path)])
    prepare.main()
    prepare.main()
    config = load_config(config_path)
    assert config["data"]["action_scale"] == 512
    paths = validate_splits(config)
    with np.load(paths["train"][0]) as episode:
        assert len(episode["frames"]) == 30
        np.testing.assert_allclose(episode["actions"][0], [25/512, -50/512])
    for name in ("pixel_diff_flow", "random_projection_flow", "avs_leworld_style"):
        ablation = load_config(f"configs/{name}.yaml")
        assert ablation["model"]["horizon"] == 16
