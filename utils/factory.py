from typing import Any

from models import AgentCentricWorldModel
from models.encoder import CNNEnvironmentEncoder, GRUAgentEncoder
from models.predictor import AgentTransition, EnvironmentTransition
from .registry import Registry

AGENT_ENCODERS = Registry()
ENVIRONMENT_ENCODERS = Registry()
AGENT_TRANSITIONS = Registry()
ENVIRONMENT_TRANSITIONS = Registry()
AGENT_ENCODERS.register("gru", GRUAgentEncoder)
ENVIRONMENT_ENCODERS.register("cnn", CNNEnvironmentEncoder)
AGENT_TRANSITIONS.register("mlp", AgentTransition)
ENVIRONMENT_TRANSITIONS.register("mlp", EnvironmentTransition)


def build_model(config: dict[str, Any]) -> AgentCentricWorldModel:
    model = config["model"]
    return AgentCentricWorldModel(
        AGENT_ENCODERS.build(model["agent_encoder"]),
        ENVIRONMENT_ENCODERS.build(model["environment_encoder"]),
        AGENT_TRANSITIONS.build(model["agent_transition"]),
        ENVIRONMENT_TRANSITIONS.build(model["environment_transition"]),
    )

