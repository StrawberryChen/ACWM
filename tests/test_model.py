import inspect

import torch

from utils.factory import build_model


def configuration():
    return {"model": {
        "agent_encoder": {"name": "gru", "image_channels": 3, "action_dim": 2, "state_dim": 16, "feature_dim": 12},
        "environment_encoder": {"name": "cnn", "image_channels": 3, "state_dim": 24},
        "agent_transition": {"name": "mlp", "agent_dim": 16, "action_dim": 2, "hidden_dim": 32},
        "environment_transition": {"name": "mlp", "environment_dim": 24, "agent_dim": 16, "hidden_dim": 32},
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

