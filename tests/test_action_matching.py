import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from dataset.action_matching import ActionBranchDataset, equivalent_futures
from losses.action_matching import compute_matching_loss
from losses.latent import SIGRegLoss
from models.action_matching import ActionMatchingWorldModel, matching_metrics


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration():
    with open("configs/action_matching_v2.yaml") as stream:
        config = yaml.safe_load(stream)
    config["device"] = "cpu"
    config["data"].update(batch_size=2, num_workers=0)
    config["model"]["latent_dim"] = 32
    config["loss"].update(sigreg_knots=5, sigreg_projections=16)
    config["wandb"]["enabled"] = False
    config["validation"].update(diagnostic_samples=4, retrieval_samples=4)
    return config


def test_gaussian_row_matching_nll_sigreg_gradients():
    torch.manual_seed(42)
    model = ActionMatchingWorldModel()
    batch = {"current_frame": torch.rand(3, 3, 96, 96), "future_frames": torch.rand(3, 2, 3, 96, 96),
             "actions": torch.rand(3, 2, 2)*2-1, "positive_mask": torch.eye(2, dtype=torch.bool).expand(3, -1, -1)}
    sigreg = SIGRegLoss(5, 16)
    captured = []
    hook = sigreg.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0].detach().clone()))
    metrics, real = compute_matching_loss(model, batch, sigreg, configuration()["loss"])
    assert real.shape == (3, 3, 128)
    torch.testing.assert_close(captured[0], real.detach().reshape(1, 9, 128))
    metrics["loss"].backward()
    for module in (model.observation_encoder, model.action_encoder, model.transition_model, model.uncertainty_head):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    hook.remove()
    mean, sigma = model.branch_distribution(real[:, 0].detach(), batch["actions"])
    assert mean.shape == sigma.shape == (3, 2, 128)
    assert sigma.min() >= model.sigma_min and sigma.max() <= model.sigma_max
    manual = torch.distributions.Normal(mean, sigma).log_prob(real[:, 1:].detach()).mean(-1)
    torch.testing.assert_close(model.log_density(mean, sigma, real[:, 1:].detach()), manual)
    scores = model.matching_scores(mean, sigma, real[:, 1:].detach())
    assert scores.shape == (3, 2, 2)
    torch.testing.assert_close(scores.diagonal(dim1=1, dim2=2), manual)


def test_collapse_matching_penalty_and_real_sigreg_penalty():
    scores = torch.ones(4, 3, 3, requires_grad=True)
    mask = torch.eye(3, dtype=torch.bool).expand(4, -1, -1)
    result = matching_metrics(scores, mask)
    assert float(result["matching_loss"].detach()) == pytest.approx(np.log(3))
    assert result["matching_acc"] == 0
    sigreg = SIGRegLoss(5, 64)
    torch.manual_seed(42)
    collapsed = sigreg(torch.zeros(1, 256, 32))
    torch.manual_seed(42)
    healthy = sigreg(torch.randn(1, 256, 32))
    assert collapsed > healthy


def test_equivalent_futures_are_multi_positive_without_inflated_accuracy():
    frames = np.zeros((3, 96, 96, 3), dtype=np.uint8)
    frames[2] = 255
    mask = torch.from_numpy(equivalent_futures(frames))[None]
    assert mask[0, 0, 1] and not mask[0, 0, 2]
    scores = torch.tensor([[[2., 2., 0.], [2., 2., 0.], [0., 0., 2.]]])
    result = matching_metrics(scores, mask, 1)
    assert result["matching_acc"] == 1
    result = matching_metrics(torch.zeros(1, 2, 2), torch.ones(1, 2, 2, dtype=torch.bool))
    assert result["matching_loss"] == 0 and result["informative_rows"] == 0
    assert torch.isnan(result["matching_acc"])


def write_fixture(root, split, episode):
    rng = np.random.default_rng(episode)
    current = rng.integers(0, 256, (4, 96, 96, 3), dtype=np.uint8)
    futures = rng.integers(0, 256, (4, 2, 96, 96, 3), dtype=np.uint8)
    actions = rng.uniform(-1, 1, (4, 2, 2)).astype(np.float32)
    branch = root/f"{split}_branches.npz"
    np.savez(branch, current_frames=current, future_frames=futures, actions=actions,
             positive_mask=np.tile(np.eye(2, dtype=np.bool_), (4, 1, 1)), timesteps=np.arange(4),
             start_physics=np.zeros((4, 18)), source_episode=episode, frame_skip=1, action_mode="relative")
    trajectory = root/f"{split}_trajectory.npz"
    np.savez(trajectory, frames=rng.integers(0, 256, (20, 96, 96, 3), dtype=np.uint8),
             actions=rng.uniform(-1, 1, (19, 2)).astype(np.float32), states=np.zeros((20, 7)),
             source_episode=episode, frame_skip=1, action_mode="relative", action_scale=100)
    return branch, trajectory


def test_train_validation_resume_evaluate_and_automatic_final_planning(tmp_path, monkeypatch):
    import train_action_matching as train
    config = configuration()
    for episode, split in enumerate(("train", "val", "test")):
        branch, trajectory = write_fixture(tmp_path, split, episode)
        config["data"][f"branch_{split}_paths"] = [str(branch)]
        config["data"][f"{split}_paths"] = [str(trajectory)]
    sample = ActionBranchDataset(config["data"]["branch_train_paths"])[0]
    assert sample["future_frames"].shape == (2, 3, 96, 96)
    assert not any(key in sample for key in ("states", "start_physics", "reward"))
    output = tmp_path/"outputs"
    config["output_dir"] = str(output)
    config["training"]["epochs"] = 2
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    planned = []

    def fake_plan(model, config, device, output, split):
        planned.append((str(output), split))
        return {"planning/success_rate": 0.0}

    monkeypatch.setattr(train, "cem_evaluation", fake_plan)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    train.main()
    saved_path = output/"checkpoints/latest.pt"
    saved = torch.load(saved_path, weights_only=False)
    assert saved["epoch"] == 2
    assert "val/loss" not in saved["history"][0] and "val/loss" in saved["history"][1]
    assert len(planned) == 1 and planned[0][1] == "test"
    assert planned[0][0].endswith("videos/final_test")
    assert (output/"plots/horizon_vs_retrieval_accuracy.png").exists()
    assert (output/"checkpoints/best_matching.pt").exists()
    evaluated = torch.load(output/"checkpoints/planning_evaluated.pt", weights_only=False)
    assert evaluated["planning_evaluation"]["split"] == "test"
    assert "uncertainty_head" in evaluated
    config["training"]["epochs"] = 3
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--checkpoint", str(saved_path), "--resume"])
    train.main()
    assert len(planned) == 2
    assert torch.load(saved_path, weights_only=False)["epoch"] == 3
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--mode", "evaluate", "--split", "test", "--checkpoint", str(saved_path)])
    train.main()
    assert (output/"logs/evaluation_test.jsonl").exists()


def test_source_episode_leakage_is_rejected(tmp_path):
    from train_action_matching import verify_splits
    config = configuration()
    datasets = {}
    for split in ("train", "val", "test"):
        branch, trajectory = write_fixture(tmp_path, split, 0)
        config["data"][f"{split}_paths"] = [str(trajectory)]
        datasets[split] = ActionBranchDataset([str(branch)])
    with pytest.raises(ValueError, match="leakage"):
        verify_splits(config, datasets)


def test_preparation_cli_paths_splits_and_resume(tmp_path, monkeypatch):
    import scripts.prepare_action_matching as prepare
    config = configuration()
    base = tmp_path/"base.yaml"
    base.write_text(yaml.safe_dump(config))
    generated = tmp_path/"colab.yaml"
    old = tmp_path/"data/train/old.npz"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old experiment untouched")
    source = {"data": {"state": np.zeros((60, 5), dtype=np.float32),
                       "action": np.zeros((60, 2), dtype=np.float32)},
              "meta": {"episode_ends": np.array([20, 40, 60])}}
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open=lambda *a, **k: source))
    monkeypatch.setattr(prepare, "ensure_zarr", lambda args: tmp_path)
    monkeypatch.setattr(prepare, "version", lambda name: "test-version")
    collected = []

    def collect(evaluator, initial, goal, actions, config, episode):
        collected.append(episode)
        branch, trajectory = write_fixture(tmp_path, f"source_{episode}", episode)
        with np.load(branch) as b, np.load(trajectory) as t:
            return dict(b), dict(t)

    monkeypatch.setattr(prepare, "collect_episode", collect)
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-root", str(tmp_path),
                                      "--base-config", str(base), "--config-output", str(generated)])
    prepare.main()
    assert len(collected) == 3
    loaded = yaml.safe_load(generated.read_text())
    assert loaded["device"] == "cuda" and loaded["validation"]["interval"] == 5
    assert loaded["validation"]["planning_after_training"]
    datasets = {s: ActionBranchDataset(loaded["data"][f"branch_{s}_paths"])
                for s in ("train", "val", "test")}
    from train_action_matching import verify_splits
    verify_splits(loaded, datasets)
    prepare.main()
    assert len(collected) == 3  # Completed episode pairs are not collected again.
    assert old.read_bytes() == b"old experiment untouched"


@pytest.mark.skipif(os.environ.get("ACWM_TEST_PUSHT") != "1", reason="real environment integration")
def test_real_counterfactual_collection_and_unchanged_cem_mpc(tmp_path):
    from evaluation.planning import PlanningEvaluator
    from scripts.prepare_action_matching import collect_episode, save_npz
    from train_action_contrastive import planning
    config = configuration()
    config["model"]["latent_dim"] = 128
    config["collection"].update(branch_stride=5, num_branches=3)
    evaluator = PlanningEvaluator(None, config["environment"], 1, 2, "cpu")
    initial = np.array([80, 80, 256, 256, 0.785398], dtype=np.float32)
    goal = np.array([180, 80, 256, 256, 0.785398], dtype=np.float32)
    branches, trajectory = collect_episode(evaluator, initial, goal, np.tile([180., 80.], (30, 1)), config, 0)
    assert branches["actions"].shape == (6, 3, 2)
    assert trajectory["states"].shape == (31, 7)
    assert trajectory["actions"].shape == (30, 2)
    np.testing.assert_array_equal(branches["actions"][:, 1], -branches["actions"][:, 0])
    np.testing.assert_array_equal(branches["current_frames"], trajectory["frames"][np.arange(0, 30, 5)])
    np.testing.assert_array_equal(branches["future_frames"][:, 0], trajectory["frames"][np.arange(1, 31, 5)])
    assert not np.array_equal(branches["future_frames"][0, 0], branches["future_frames"][0, 1])
    branch_path, path = tmp_path/"branches.npz", tmp_path/"trajectory.npz"
    save_npz(branch_path, branches)
    save_npz(path, trajectory)
    dataset = ActionBranchDataset([str(branch_path)])
    assert len(dataset) == 6 and dataset[0]["actions"].shape == (3, 2)
    config["data"]["val_paths"] = [str(path)]
    config["data"]["train_paths"] = [str(path)]  # synthetic normalization fixture
    config["planner"].update(horizon=2, population=8, elites=2, iterations=2)
    config["environment"].update(max_steps=2, replan_interval=1)
    config["validation"].update(planning_episodes=1, planning_videos=1)
    model = ActionMatchingWorldModel().eval()
    result = planning(model, config, torch.device("cpu"), tmp_path/"videos", "val")
    assert 0 <= result["planning/success_rate"] <= 100
    assert result["planning/imagined_transitions_per_second"] > 0
    import imageio.v2 as imageio
    reader = imageio.get_reader(tmp_path/"videos/planning_episode_000.mp4")
    assert reader.get_data(0).shape[:2] == (96, 192)
    reader.close()
