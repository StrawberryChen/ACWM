from typing import Any

from models import AgentCentricWorldModel
from models.encoder import CNNEnvironmentEncoder, GRUAgentEncoder, GRUViTAgentEncoder, ViTEnvironmentEncoder
from models.predictor import AgentTransition, EnvironmentTransition, MotionTokenPredictor
from .registry import Registry

AGENT_ENCODERS = Registry()
ENVIRONMENT_ENCODERS = Registry()
AGENT_TRANSITIONS = Registry()
ENVIRONMENT_TRANSITIONS = Registry()
PREDICTORS = Registry()
AGENT_ENCODERS.register("gru", GRUAgentEncoder)
AGENT_ENCODERS.register("vit_tiny_gru", GRUViTAgentEncoder)
ENVIRONMENT_ENCODERS.register("cnn", CNNEnvironmentEncoder)
ENVIRONMENT_ENCODERS.register("vit_tiny", ViTEnvironmentEncoder)
AGENT_TRANSITIONS.register("mlp", AgentTransition)
ENVIRONMENT_TRANSITIONS.register("mlp", EnvironmentTransition)
PREDICTORS.register("motion_token", MotionTokenPredictor)


def build_model(config: dict[str, Any]) -> AgentCentricWorldModel:
    model = config["model"]
    predictor_config = dict(model.get("predictor", {"type": "adaln"}))
    predictor_type = predictor_config.pop("type", "adaln")
    if predictor_type not in {"adaln", "motion_token"}:
        raise ValueError("model.predictor.type must be 'adaln' or 'motion_token'")
    predictor = None
    if predictor_type == "motion_token":
        predictor_config.setdefault("hidden_dim", model["environment_encoder"].get("state_dim", 192))
        predictor = PREDICTORS.build({"name": "motion_token", **predictor_config})
    return AgentCentricWorldModel(
        AGENT_ENCODERS.build(model["agent_encoder"]),
        ENVIRONMENT_ENCODERS.build(model["environment_encoder"]),
        AGENT_TRANSITIONS.build(model["agent_transition"]),
        ENVIRONMENT_TRANSITIONS.build(model["environment_transition"]),
        predictor=predictor,
        predictor_type=predictor_type,
        history_size=predictor_config.get("history_size", config.get("data", {}).get("history_length", 3)),
    )
