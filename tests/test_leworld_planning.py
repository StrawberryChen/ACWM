from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from evaluation.leworld_planning import align_raw_step_config, training_action_statistics
from planner import CEMPlanner


def test_profile_preserves_training_and_single_step_interface():
    for filename in ("action_contrastive.yaml", "action_matching_v2.yaml"):
        with open(f"configs/{filename}") as stream:
            config = yaml.safe_load(stream)
        before = yaml.safe_dump({k: config[k] for k in ("model", "loss", "training", "data", "output_dir")})
        align_raw_step_config(config)
        assert before == yaml.safe_dump({k: config[k] for k in ("model", "loss", "training", "data", "output_dir")})
        assert config["planner"]["horizon"] == 25
        assert config["planner"]["action_dim"] == 2  # NOT a fictitious trained 10D Predictor
        assert config["environment"]["replan_interval"] == 25
        assert config["environment"]["max_steps"] == 50
        assert config["planner"]["cost_type"] == "latent_mse_sum"
        assert config["planner"]["clamp_actions"] is False


def test_action_stats_population_and_constant_dimension(tmp_path):
    path = tmp_path/"train.npz"
    np.savez(path, actions=np.array([[0., 0.2], [1., 0.2], [-1., 0.2]], np.float32))
    mean, std = training_action_statistics([str(path)])
    np.testing.assert_allclose(mean, [0., 0.2], atol=1e-7)
    np.testing.assert_allclose(std, [np.sqrt(2/3), 1.])
    with pytest.raises(ValueError):
        training_action_statistics([str(tmp_path/"missing*.npz")])


class RecordingModel:
    def __init__(self):
        self.candidates = []

    def encode(self, frames, actions, current):
        state = torch.zeros(len(current), 2)
        return state, state

    def encode_goal(self, goal):
        return torch.ones(len(goal), 2)

    def rollout(self, agent, environment, actions):
        self.candidates.append(actions.clone())
        return [SimpleNamespace(environment=actions.sum(1))]

    def planning_cost(self, final, goal):
        raise AssertionError("LeWorld cost must not use model cosine criterion")


def test_seeded_normalized_cem_mse_and_raw_execution_actions():
    image = torch.zeros(1, 3, 8, 8)
    settings = dict(horizon=25, action_dim=2, population=20, elites=4, iterations=3,
                    clamp_actions=False, cost_type="latent_mse_sum", seed=42,
                    std_correction=0, min_std=0., action_mean=[0.2, -0.3], action_std=[0.5, 0.25])
    model1, model2 = RecordingModel(), RecordingModel()
    p1, p2 = CEMPlanner(**settings), CEMPlanner(**settings)
    args = (image[:, None], torch.empty(1, 0, 2), image, image)
    result = p1.plan(model1, *args)
    torch.manual_seed(999)  # independent solver RNG must ignore global RNG changes
    torch.testing.assert_close(result, p2.plan(model2, *args))
    assert result.shape == (1, 25, 2)
    raw = model1.candidates[0].reshape(1, 20, 25, 2)
    torch.testing.assert_close(raw[0, 0], torch.tensor([0.2, -0.3]).expand(25, 2))
    assert raw.abs().max() > 1  # official CEM has no +/-1 clipping
    costs = (model1.candidates[-1].sum(1)-1).square().sum(-1)
    assert p1.last_diagnostics["cem_cost"] == pytest.approx(float(costs.topk(4, largest=False).values.mean()))
    p1.plan(model1, *args, init_action=result[:, 5:])
    torch.testing.assert_close(model1.candidates[3][0, :20], result[0, 5:])


def test_mpc_executes_25_distinct_raw_actions_before_replanning(monkeypatch):
    from evaluation.planning import PlanningEvaluator

    class FakeEnv:
        def __init__(self):
            self.executed = []

        def reset(self, **kwargs):
            return np.zeros((8, 8, 3), np.uint8), {}

        def step(self, action):
            self.executed.append(action.copy())
            return np.zeros((8, 8, 3), np.uint8), 0., False, False, {}

        def close(self):
            pass

    class FakePlanner:
        horizon = 25
        last_diagnostics = {}

        def plan(self, *args, **kwargs):
            assert kwargs["init_action"] is None  # all 25 planned actions were consumed
            return torch.arange(50).reshape(1, 25, 2).float()/100

    config = dict(env_id="test", eval_from_dataset=False, max_steps=50,
                  replan_interval=25, action_block=1, observation_width=8, observation_height=8)
    evaluator = PlanningEvaluator(FakePlanner(), config, 1, 2, "cpu")
    env = FakeEnv()
    monkeypatch.setattr(evaluator, "_make_env", lambda: env)
    monkeypatch.setattr(evaluator, "_goal_frame", lambda _: torch.zeros(1, 3, 8, 8))
    result = evaluator.evaluate(SimpleNamespace(eval=lambda: None), episodes=1, videos_to_save=0)
    assert len(env.executed) == 50
    assert result["diagnostics/planning_calls"] == 2
    np.testing.assert_allclose(np.stack(env.executed[:25]), np.arange(50).reshape(25, 2)/100)
    np.testing.assert_array_equal(env.executed[:25], env.executed[25:])
