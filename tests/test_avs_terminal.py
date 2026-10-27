import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from experiments.avs_target.data import file_sha256, load_config
from experiments.avs_target.joint_evaluate import targets_for
from experiments.avs_target.joint_model import JointFlowWorldModel, joint_euler_sample, joint_flow_loss, joint_training_pair
from experiments.avs_target.planning import TerminalCEMPlanner, check_diagnostic_review
from experiments.avs_target.targets import FixedMotionTarget
from experiments.avs_target.terminal import FixedTerminalOperator


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("backend", ["downsample", "dct"])
def test_fixed_terminal_and_goal_operator_identical_no_gradient(backend):
    operator = FixedTerminalOperator(backend=backend)
    image = torch.rand(2, 3, 96, 96, requires_grad=True)
    target, goal = operator(image), operator(image.clone())
    assert target.shape == (2, 1728 if backend == "downsample" else 192)
    assert not list(operator.parameters()) and not target.requires_grad
    torch.testing.assert_close(target, goal)
    assert operator.image(target).shape == (2, 3, 24, 24)
    if backend == "dct":
        torch.testing.assert_close(operator.dct_basis@operator.dct_basis.T, torch.eye(24), atol=3e-6, rtol=1e-5)
        full = FixedTerminalOperator(backend="dct", size=8, dct_coefficients=8)
        expected = torch.nn.functional.interpolate(image.detach(), (8, 8), mode="area")
        torch.testing.assert_close(full.image(full(image)), expected, atol=3e-6, rtol=1e-5)


def test_terminal_is_last_frame_and_avs_first_transition_unchanged():
    frames = torch.zeros(2, 5, 3, 96, 96)
    frames[:, -1] = 1
    terminal, motion = FixedTerminalOperator(), FixedMotionTarget()
    targets = targets_for(frames, terminal, motion)
    assert targets["terminal"].min() == 1
    torch.testing.assert_close(targets["motion"], motion(frames))
    assert targets["motion"].shape == (2, 4, 288)


def test_separate_means_prevent_terminal_dimension_dominating_loss():
    predicted = {"motion": torch.full((2, 4, 288), 2., requires_grad=True),
                 "terminal": torch.full((2, 1728), 3., requires_grad=True)}
    targets = {name: torch.zeros_like(value) for name, value in predicted.items()}
    metrics = joint_flow_loss(predicted, targets)
    assert float(metrics["motion_flow_loss"].detach()) == 4
    assert float(metrics["terminal_flow_loss"].detach()) == 9
    assert float(metrics["loss"].detach()) == 13
    metrics["loss"].backward()
    assert predicted["motion"].grad.abs().sum() > 0 and predicted["terminal"].grad.abs().sum() > 0


@pytest.mark.parametrize("joint", [False, True])
def test_shared_cse_forward_backward_future_operators_detached(joint):
    terminal = FixedTerminalOperator(size=4, dct_coefficients=4)
    motion = FixedMotionTarget(resolution=3) if joint else None
    model = JointFlowWorldModel(terminal.output_dim, None if motion is None else motion.output_dim,
                                latent_dim=16, horizon=4, model_dim=32, layers=1, heads=4)
    frames = torch.rand(2, 5, 3, 96, 96, requires_grad=True)
    actions = torch.rand(2, 4, 2)
    target = targets_for(frames, terminal, motion)
    noise = {name: torch.randn_like(value) for name, value in target.items()}
    time = torch.tensor([.2, .8])
    noisy, velocity = joint_training_pair(target, noise, time)
    prediction = model.predictor(noisy, time, model.cse(frames[:, 0]), actions)
    assert prediction["terminal"].shape == (2, 48)
    assert ("motion" in prediction) == joint
    loss = joint_flow_loss(prediction, velocity)["loss"]
    loss.backward()
    assert frames.grad[:, 0].abs().sum() > 0 and frames.grad[:, 1:].abs().sum() == 0
    for module in (model.cse, model.predictor.action_encoder, model.predictor.terminal_velocity_head):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    if joint:
        assert model.predictor.motion_velocity_head.weight.grad.abs().sum() > 0
    else:
        assert model.predictor.motion_projection is None and model.predictor.motion_velocity_head is None


def test_joint_euler_known_constant_velocity():
    noise = {"motion": torch.zeros(2, 4, 3), "terminal": torch.zeros(2, 7)}
    velocity = {name: torch.full_like(value, 2.) for name, value in noise.items()}
    result = joint_euler_sample(lambda *args: velocity, torch.zeros(2, 3), torch.zeros(2, 4, 2), noise, 4)
    for name in result:
        torch.testing.assert_close(result[name], velocity[name])


def small_config(tmp_path, experiment):
    config = load_config(f"configs/{experiment}.yaml")
    config.update(device="cpu", output_dir=str(tmp_path/experiment))
    config["data"].update(batch_size=2, num_workers=0, stats_batch_size=2)
    config["model"].update(latent_dim=16, horizon=4, model_dim=32, layers=1, heads=4)
    config["target"].update(resolution=3, pair_chunk=8)
    config["terminal"].update(size=4, dct_coefficients=4)
    config["training"]["epochs"] = 1
    config["validation"].update(samples=4, generation_samples=4, horizons=[1, 2, 4], flow_steps=[1, 2],
                                 visualization_examples=2, comparison_videos=1)
    config["probe"].update(max_train_samples=8, max_eval_samples=4, epochs=2)
    config["planning"].update(horizon=4, num_candidates=8, num_elites=2, iterations=2,
                               flow_steps=2, episodes=1, max_steps=2, goal_offset_steps=5, candidate_batch_size=4)
    config["wandb"]["enabled"] = False
    for episode, split in enumerate(("train", "val", "test")):
        path = tmp_path/f"episode_{episode}.npz"
        frames = np.zeros((12, 96, 96, 3), np.uint8)
        states = np.tile(np.array([80., 80., 256., 256., .785398], np.float32), (12, 1))
        for t in range(12):
            frames[t, 20:40, 20+t:40+t] = 255
            states[t, 0] += t
        np.savez_compressed(path, frames=frames, actions=np.full((11, 2), .1, np.float32),
                            states=states, source_episode=episode, frame_skip=1, action_mode="relative", action_scale=512.)
        config["data"][f"{split}_paths"] = [str(path)]
    return config


@pytest.mark.parametrize("experiment", ["terminal_only", "avs_terminal"])
def test_train_resume_evaluate_probe_and_review_provenance(tmp_path, monkeypatch, experiment):
    from experiments.avs_target import joint_train, probe
    config = small_config(tmp_path, experiment)
    path = tmp_path/"config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path)])
    joint_train.main()
    output = Path(config["output_dir"])
    checkpoint = output/"checkpoints/latest.pt"
    saved = torch.load(checkpoint, weights_only=False)
    assert "train/terminal_flow_loss" in saved["metrics"] and "val/terminal_flow_loss" in saved["metrics"]
    assert "generation/k2/h4/terminal_mse" in saved["metrics"]
    assert (saved["motion"] is not None) == (experiment == "avs_terminal")
    assert (output/"terminal_target.png").exists() and (output/"validation/epoch_0001/terminal.png").exists()
    assert not list(output.glob("planning*"))
    config["training"]["epochs"] = 2
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(path), "--checkpoint", str(checkpoint), "--resume"])
    joint_train.main()
    assert torch.load(checkpoint, weights_only=False)["epoch"] == 2
    # Simulate the old Colab checkpoint/stats format and a Drive mtime-only change.
    legacy = torch.load(checkpoint, weights_only=False)
    for record in legacy["train_sources"]:
        record.pop("sha256", None)
    torch.save(legacy, checkpoint)
    if experiment == "avs_terminal":
        stats_path = output/"avs_stats.json"
        stats = json.loads(stats_path.read_text())
        for record in stats["train_sources"]:
            record.pop("sha256", None)
        stats_path.write_text(json.dumps(stats))
    for record in legacy["train_sources"]:
        os.utime(record["path"], ns=(record["mtime_ns"]+10_000_000, record["mtime_ns"]+10_000_000))
    original_config_bytes = (output/"config.yaml").read_bytes()
    original_target = legacy["motion"]
    config["evaluation_note"] = "must not overwrite training configuration"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["eval", "--config", str(path), "--checkpoint", str(checkpoint), "--mode", "evaluate"])
    joint_train.main()
    assert (output/"config.yaml").read_bytes() == original_config_bytes
    after_evaluation = torch.load(checkpoint, weights_only=False)
    assert after_evaluation["epoch"] == 2
    if original_target is not None:
        for key in ("mean", "std"):
            torch.testing.assert_close(after_evaluation["motion"][key], original_target[key])
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(path), "--checkpoint", str(checkpoint)])
    probe.main()
    evaluation = output/"evaluation_val/report.json"
    probes = output/"probes/metrics.json"
    assert check_diagnostic_review(checkpoint, evaluation, probes, True)["checkpoint_sha256"] == file_sha256(checkpoint)
    from experiments.avs_target.compare import read_experiment
    comparison = read_experiment(output)
    assert "generation/terminal_mse" in comparison
    assert not any("/test/" in key for key in comparison)
    with pytest.raises(ValueError, match="review"):
        check_diagnostic_review(checkpoint, evaluation, probes, False)
    metadata = output/"probes/metadata.json"
    data = json.loads(metadata.read_text())
    data["checkpoint_sha256"] = "wrong"
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exact checkpoint"):
        check_diagnostic_review(checkpoint, evaluation, probes, True)


class ToyModel(nn.Module):
    horizon, terminal_dim, motion_dim = 3, 3, 2

    def __init__(self, motion_velocity):
        super().__init__()
        self.motion_velocity, self.cse_calls = motion_velocity, 0

    def cse(self, image):
        self.cse_calls += 1
        assert image.max() == 0  # goal image is white; must NEVER be encoded here
        return torch.zeros(len(image), 4)

    def predictor(self, noisy, time, state, actions):
        signal = actions.mean((1, 2))[:, None].expand(-1, 3)
        return {"terminal": signal, "motion": torch.full_like(noisy["motion"], self.motion_velocity)}


@pytest.mark.parametrize("samples", [1, 3])
def test_cem_expected_terminal_mse_only_and_chunk_invariance(samples):
    settings = load_config("configs/avs_terminal.yaml")["planning"]
    settings.update(horizon=3, num_candidates=8, num_elites=2, iterations=2, future_samples=samples,
                    flow_steps=2, candidate_batch_size=3)
    terminal = FixedTerminalOperator(size=1, dct_coefficients=1)
    current, goal = torch.zeros(1, 3, 96, 96), torch.ones(1, 3, 96, 96)
    first_model, second_model = ToyModel(1e6), ToyModel(-1e6)
    first = TerminalCEMPlanner(settings, "cpu").plan(first_model, terminal, current, goal)
    settings["candidate_batch_size"] = 1
    second = TerminalCEMPlanner(settings, "cpu").plan(second_model, terminal, current, goal)
    torch.testing.assert_close(first["actions"], second["actions"])
    assert first_model.cse_calls == second_model.cse_calls == 1
    assert first["terminal_samples"].shape == (samples, 3)
    assert first["cost"] == pytest.approx(float((first["terminal_samples"]-1).square().mean()))
    assert first["evaluated_candidates"] == 16 and first["candidates_per_second"] > 0
    assert first["actions"].shape == (3, 2) and first["actions"].abs().max() <= 1


def test_original_avs_config_still_same():
    assert load_config("configs/avs_only.yaml") == load_config("configs/avs_flow.yaml")


def test_default_joint_measurement_dimensions_and_single_shared_transformer():
    from experiments.avs_target.joint_train import build_joint
    config = load_config("configs/avs_terminal.yaml")
    model, terminal, motion = build_joint(config, "cpu")
    assert terminal.output_dim == 1728 and motion.output_dim == 288
    assert sum(isinstance(m, torch.nn.TransformerEncoder) for m in model.modules()) == 1
    with torch.no_grad():
        result = model.predictor({"terminal": torch.randn(1, 1728), "motion": torch.randn(1, 16, 288)},
                                  torch.tensor([.5]), torch.randn(1, 128), torch.randn(1, 16, 2))
    assert result["terminal"].shape == (1, 1728) and result["motion"].shape == (1, 16, 288)


def test_render_marker_does_not_change_physical_goal():
    from types import SimpleNamespace
    from experiments.avs_target.planning import observation_frame
    actual_goal = np.arange(7, dtype=np.float32)
    original_pose = np.array([10., 20., .5])
    base = SimpleNamespace(goal_pose=original_pose, goal_state=actual_goal.copy())
    observed = []
    env = SimpleNamespace(unwrapped=base, render=lambda: observed.append(base.goal_pose.copy()) or np.zeros((96, 96, 3), np.uint8))
    evaluator = SimpleNamespace(_image=lambda array: torch.from_numpy(array).permute(2, 0, 1).float()/255)
    frame = observation_frame(env, evaluator, {"render_goal_pose": [256., 256., .785398]})
    assert frame.shape == (1, 3, 96, 96)
    np.testing.assert_array_equal(observed[0], np.array([256., 256., .785398], np.float32))
    np.testing.assert_array_equal(base.goal_pose, original_pose)
    np.testing.assert_array_equal(base.goal_state, actual_goal)


def test_mpc_action_scale_and_execute_first_action_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import experiments.avs_target.planning as module
    config = load_config("configs/terminal_only.yaml")
    config["planning"].update(episodes=1, max_steps=3, video_episodes=0, horizon=3, execute_steps=1)
    executed, warmed, cases = [], [], [{"start_state": np.zeros(7), "goal_state": np.ones(7)}]
    class Environment:
        def step(self, action):
            executed.append(action.copy())
            return None, -1., False, False, {}
        def close(self):
            pass
    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass
        def _dataset_eval_cases(self, count):
            return cases
        def _make_env(self):
            return Environment()
        def _reset_env(self, *args):
            return None, {}
    class Planner:
        def __init__(self, *args):
            pass
        def plan(self, model, terminal, current, goal, previous):
            warmed.append(previous)
            return dict(actions=torch.tensor([[.25, -.5], [.9, .9], [-.8, .8]]),
                        terminal_samples=torch.zeros(1, 3), cost=0., cem_time=.1,
                        world_model_time=.05, candidates_per_second=80.)
    monkeypatch.setattr(module, "PlanningEvaluator", Evaluator)
    monkeypatch.setattr(module, "TerminalCEMPlanner", Planner)
    monkeypatch.setattr(module, "observation_frame", lambda *args: torch.zeros(1, 3, 96, 96))
    terminal = FixedTerminalOperator(size=1, dct_coefficients=1)
    result = module.evaluate_mpc(SimpleNamespace(), terminal, config, [], tmp_path, torch.device("cpu"))
    assert len(executed) == 3 and warmed[0] is None
    np.testing.assert_allclose(executed, np.tile([.25*512/100, -.5*512/100], (3, 1)))
    assert warmed[1].shape == (2, 2)
    assert result["planning/success_rate"] == 0 and "planning/steps_to_success" not in result


@pytest.mark.skipif(os.environ.get("ACWM_TEST_PUSHT") != "1", reason="real Push-T integration")
def test_real_joint_terminal_mpc_video_and_prediction_artifacts(tmp_path):
    from experiments.avs_target.joint_train import build_joint
    from experiments.avs_target.planning import evaluate_mpc
    config = small_config(tmp_path, "avs_terminal")
    model, terminal, _ = build_joint(config, "cpu")
    result = evaluate_mpc(model, terminal, config, config["data"]["test_paths"], tmp_path/"planning", torch.device("cpu"))
    assert 0 <= result["planning/success_rate"] <= 100
    assert result["planning/world_model_time"] > 0 and result["planning/candidates_per_second"] > 0
    assert (tmp_path/"planning/planning_episode_000.mp4").exists()
    assert len(list((tmp_path/"planning").glob("episode_000/plan_*/prediction.npz"))) >= 1
