import os

import numpy as np
import pytest
import torch

from dataset.action_contrastive import ActionContrastiveDataset
from evaluation.action_contrastive import representation_metrics
from models.action_contrastive import ActionContrastiveWorldModel, negative_actions, ranking_metrics
from planner import CEMPlanner
from scripts.prepare_action_contrastive import convert_actions


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_shapes_and_shared_encoder_gradients():
    torch.manual_seed(42)
    model = ActionContrastiveWorldModel()
    x = torch.rand(4, 3, 96, 96)
    y = torch.rand_like(x)
    a = torch.rand(4, 2)*2-1
    o, target = model.encode_frame(x), model.encode_frame(y)
    o.retain_grad()
    target.retain_grad()
    assert o.shape == target.shape == (4, 128)
    candidates = negative_actions(a)
    assert torch.equal(candidates[:, 1], -a)
    assert model.transition(o, a).shape == (4, 128)
    loss = ranking_metrics(model.candidate_scores(o, candidates, target))["loss"]
    loss.backward()
    assert o.grad.abs().sum() > 0 and target.grad.abs().sum() > 0
    for module in (model.observation_encoder, model.action_encoder, model.transition_model):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0


def test_ranking_and_collapse_diagnostics():
    metrics = ranking_metrics(torch.tensor([[0.9, 0.2], [0.1, 0.5]]))
    assert metrics["ranking_acc"] == 0.5
    assert metrics["mean_rank"] == 1.5
    collapsed = representation_metrics(torch.ones(16, 128))
    assert collapsed["representation/mean_feature_std"] == 0
    assert collapsed["representation/effective_rank"] == 0
    assert collapsed["representation/mean_pairwise_cosine"] == pytest.approx(1)
    assert ranking_metrics(torch.ones(4, 2))["ranking_acc"] == 0  # no tie optimism


def test_raw_step_alignment_and_no_states_in_batch(tmp_path):
    frames = np.zeros((20, 96, 96, 3), dtype=np.uint8)
    for t in range(20):
        frames[t] = t
    actions = np.arange(19, dtype=np.float32)[:, None].repeat(2, 1)/20
    path = tmp_path / "episode.npz"
    np.savez(path, frames=frames, actions=actions, states=np.zeros((20, 5)), frame_skip=1, action_mode="relative")
    dataset = ActionContrastiveDataset([str(path)], horizon=16)
    sample = dataset[2]
    assert sample["frames"][0, 0, 0, 0] == pytest.approx(2/255)
    assert sample["frames"][1, 0, 0, 0] == pytest.approx(3/255)
    assert sample["actions"][0, 0] == pytest.approx(2/20)
    assert "states" not in sample
    np.savez(path, frames=frames, actions=np.zeros((19, 5, 2)))
    with pytest.raises(ValueError, match="raw-step"):
        ActionContrastiveDataset([str(path)])


def test_action_conversion():
    states = np.array([[100, 200, 0, 0, 0], [50, 50, 0, 0, 0]])
    actions = np.array([[150, 175], [500, -100]])
    np.testing.assert_allclose(convert_actions(actions, states, 100), [[0.5, -0.25], [1, -1]])


def test_prepare_disjoint_raw_episodes_and_preserve_old_data(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import yaml
    import scripts.prepare_action_contrastive as prepare
    root = tmp_path / "data/train"
    root.mkdir(parents=True)
    sentinel = root / "existing.npz"
    sentinel.write_bytes(b"old experiment")
    n = 200
    states = np.zeros((n, 5), dtype=np.float32)
    states[:, :2] = [100, 200]
    data = {"data": {"img": np.zeros((n, 12, 12, 3), dtype=np.uint8), "state": states,
                     "action": states[:, :2] + [25, -50]},
            "meta": {"episode_ends": np.arange(20, n+1, 20)}}
    monkeypatch.setitem(sys.modules, "zarr", SimpleNamespace(open=lambda *args, **kwargs: data))
    monkeypatch.setattr(prepare, "ensure_zarr", lambda args: "fake.zarr")
    output_config = tmp_path / "colab.yaml"
    monkeypatch.setattr(sys, "argv", ["prepare", "--output-root", str(tmp_path), "--config-output", str(output_config)])
    prepare.main()
    prepare.main()  # safe reuse with identical manifest
    assert sentinel.read_bytes() == b"old experiment"
    with output_config.open() as stream:
        config = yaml.safe_load(stream)
    import glob
    all_paths = [glob.glob(config["data"][f"{split}_paths"][0]) for split in ("train", "val", "test")]
    assert list(map(len, all_paths)) == [8, 1, 1]
    assert len({p for paths in all_paths for p in paths}) == 10
    with np.load(all_paths[0][0]) as episode:
        assert episode["actions"].shape == (19, 2)
        assert episode["frames"].shape[0] == 20
        np.testing.assert_allclose(episode["actions"], np.tile([0.25, -0.5], (19, 1)))


def test_train_validate_checkpoint_resume_and_evaluate(tmp_path, monkeypatch):
    import sys
    import yaml
    from train_action_contrastive import main
    with open("configs/action_contrastive.yaml") as stream:
        config = yaml.safe_load(stream)
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.npz"
        np.savez(path, frames=rng.integers(0, 256, (20, 96, 96, 3), dtype=np.uint8),
                 actions=rng.uniform(-1, 1, (19, 2)).astype(np.float32), frame_skip=1, action_mode="relative")
        config["data"][f"{split}_paths"] = [str(path)]
    config["device"] = "cpu"
    config["output_dir"] = str(tmp_path / "outputs")
    config["data"].update(batch_size=4, num_workers=0)
    config["model"]["latent_dim"] = 32
    config["wandb"]["enabled"] = False
    config["training"]["epochs"] = 2
    config["validation"].update(diagnostic_samples=8, retrieval_samples=4)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train_action_contrastive.py", "--config", str(path), "--validation-interval", "5"])
    main()
    saved = tmp_path / "outputs/checkpoints/latest.pt"
    checkpoint = torch.load(saved, weights_only=False)
    assert checkpoint["epoch"] == 2
    assert "val/loss" not in checkpoint["history"][0]
    assert "val/loss" in checkpoint["history"][1]  # final epoch validates even off-interval
    assert checkpoint["config"]["validation"]["interval"] == 5
    assert checkpoint["metrics"]["retrieval/candidates"] == 4
    assert (tmp_path / "outputs/plots/horizon_vs_retrieval_accuracy.png").exists()
    assert (tmp_path / "outputs/checkpoints/best_ranking.pt").exists()
    # Continue one more epoch and verify optimizer/scheduler/checkpoint interfaces.
    config["training"]["epochs"] = 3
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train_action_contrastive.py", "--config", str(path), "--resume", "--checkpoint", str(saved)])
    main()
    assert torch.load(saved, weights_only=False)["epoch"] == 3
    monkeypatch.setattr(sys, "argv", ["train_action_contrastive.py", "--config", str(path), "--mode", "evaluate", "--split", "test", "--checkpoint", str(saved)])
    main()
    assert (tmp_path / "outputs/evaluation_test.jsonl").exists()


@pytest.mark.skipif(os.environ.get("ACWM_TEST_PUSHT") != "1", reason="opt-in real Push-T integration")
def test_real_pusht_planning_and_video(tmp_path):
    import yaml
    from train_action_contrastive import planning
    with open("configs/action_contrastive.yaml") as stream:
        config = yaml.safe_load(stream)
    states = np.tile([80., 80., 256., 256., 0.785398], (30, 1))
    states[:, 0] += np.arange(30)
    path = tmp_path / "episode.npz"
    np.savez(path, frames=np.zeros((30, 96, 96, 3), dtype=np.uint8),
             actions=np.zeros((29, 2), dtype=np.float32), states=states,
             frame_skip=1, action_mode="relative")
    config["data"]["val_paths"] = [str(path)]
    config["planner"].update(horizon=2, population=8, elites=2, iterations=2)
    config["environment"].update(max_steps=2)
    config["validation"].update(planning_episodes=1, planning_videos=1)
    model = ActionContrastiveWorldModel().eval()
    result = planning(model, config, torch.device("cpu"), tmp_path / "videos", "val")
    assert 0 <= result["planning/success_rate"] <= 100
    assert result["planning/imagined_transitions_per_second"] > 0
    assert np.isfinite(result["planning/diagnostics/mean_cem_cost"])
    assert (tmp_path / "videos/planning_episode_000.mp4").stat().st_size > 0


@pytest.mark.parametrize("architecture", ["residual_mlp", "concat_mlp", "film"])
def test_rollout_and_cem_only_encode_real_frames(architecture):
    model = ActionContrastiveWorldModel(latent_dim=32, architecture=architecture).eval()
    o = torch.rand(10, 32)
    actions = torch.rand(10, 16, 2)
    out = model.rollout_latents(o, actions)
    assert out.shape == (10, 16, 32)
    state = o
    for i in range(16):
        state = model.transition(state, actions[:, i])
        torch.testing.assert_close(state, out[:, i])
    calls = []
    hook = model.observation_encoder.register_forward_hook(lambda *args: calls.append(1))
    planner = CEMPlanner(horizon=4, action_dim=2, population=16, elites=4, iterations=2)
    image = torch.rand(1, 3, 96, 96)
    planned = planner.plan(model, image[:, None], torch.empty(1, 0, 2), image, image)
    assert planned.shape == (1, 4, 2)
    assert planned.abs().max() <= 1
    assert len(calls) == 2  # current observation and goal, never imagined frames
    hook.remove()
