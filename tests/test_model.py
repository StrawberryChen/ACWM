import inspect

import torch

from utils.factory import build_model


def configuration():
    return {"model": {
        "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 16, "feature_dim": 12},
        "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 24},
        "agent_transition": {"name": "mlp", "agent_dim": 16, "action_dim": 2, "hidden_dim": 32},
        "environment_transition": {"name": "mlp", "environment_dim": 24, "agent_dim": 16, "hidden_dim": 32},
        "predictor": {"type": "adaln"},
    }}


def motion_configuration():
    return {"data": {"history_length": 3}, "model": {
        "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 16, "feature_dim": 12},
        "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 24},
        "agent_transition": {"name": "mlp", "agent_dim": 16, "action_dim": 2, "hidden_dim": 32},
        "environment_transition": {"name": "mlp", "environment_dim": 24, "agent_dim": 16, "hidden_dim": 32},
        "predictor": {
            "type": "motion_token",
            "action_dim": 2,
            "hidden_dim": 24,
            "history_size": 3,
            "motion_layers": 1,
            "transformer_layers": 1,
            "num_heads": 3,
            "flow_hidden_dim": 32,
        },
    }}


def test_information_flow_and_shapes():
    model = build_model(configuration())
    frames = torch.rand(2, 4, 3, 32, 32)
    history_actions = torch.rand(2, 3, 2)
    agent, environment = model.encode(frames, history_actions, frames[:, -1])
    prediction = model.step(agent, environment, torch.rand(2, 2))
    assert agent.shape == prediction.agent.shape == (2, 16)
    assert environment.shape == prediction.environment.shape == (2, 24)
    assert list(inspect.signature(model.environment_transition.forward).parameters) == ["environment_state", "next_agent_state"]


def test_rollout_is_differentiable():
    model = build_model(configuration())
    agent = torch.randn(2, 16)
    environment = torch.randn(2, 24)
    predictions = model.rollout(agent, environment, torch.randn(2, 3, 2))
    assert len(predictions) == 3
    predictions[-1].environment.square().mean().backward()
    assert model.agent_transition.delta[0].weight.grad is not None
    assert model.environment_transition.delta[0].weight.grad is not None


def test_planning_cost_excludes_agent_state():
    model = build_model(configuration())
    final_environment = torch.tensor([[1.0, 2.0]])
    goal_environment = torch.tensor([[1.0, 4.0]])
    assert torch.equal(model.planning_cost(final_environment, goal_environment), torch.tensor([2.0]))


def test_motion_token_predictor_shapes_and_rollout():
    model = build_model(motion_configuration())
    frames = torch.rand(2, 3, 3, 32, 32)
    history_actions = torch.rand(2, 2, 2)
    agent, environment = model.encode(frames, history_actions, frames[:, -1])
    assert agent.shape == (2, 3, 24)
    assert environment.shape == (2, 24)
    prediction = model.step(agent, environment, torch.rand(2, 2), history_actions)
    assert prediction.agent.shape == (2, 3, 24)
    assert prediction.environment.shape == (2, 24)
    assert prediction.delta.shape == (2, 24)
    predictions = model.rollout(agent, environment, torch.rand(2, 4, 2), history_actions)
    assert len(predictions) == 4
    predictions[-1].environment.square().mean().backward()
    assert model.predictor.flow_head.net[0].weight.grad is not None
