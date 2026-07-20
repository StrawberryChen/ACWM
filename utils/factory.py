from typing import Any

from models import AgentCentricWorldModel
from models.encoder import CNNEnvironmentEncoder, GRUAgentEncoder, GRUViTAgentEncoder, ViTEnvironmentEncoder
from models.forward_inverse_predictor import ForwardInversePredictor
from models.predictor import AgentTransition, EnvironmentTransition, MotionTokenPredictor
from models.v3_n1 import V3N1GaussianWorldModel
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
PREDICTORS.register("forward_inverse", ForwardInversePredictor)
PREDICTORS.register("v3_n1", V3N1GaussianWorldModel)


def build_model(config: dict[str, Any]) -> AgentCentricWorldModel:
    model = config["model"]
    version = model.get("version")
    predictor_config = dict(model.get("predictor", {}))
    predictor_type = predictor_config.pop("type", None)
    if version is None:
        version = {"adaln": "leworld", "motion_token": "motion_token",
                   "forward_inverse": "forward_inverse", "v3_n1": "v3_n1"}.get(predictor_type or "adaln")
    if predictor_type is None:
        predictor_type = {"leworld": "adaln", "motion_token": "motion_token",
                          "forward_inverse": "forward_inverse", "v3_n1": "v3_n1"}[version]
    if version not in {"leworld", "motion_token", "forward_inverse", "v3_n1"}:
        raise ValueError("model.version must be 'leworld', 'motion_token', 'forward_inverse', or 'v3_n1'")
    if predictor_type not in {"adaln", "motion_token", "forward_inverse", "v3_n1"}:
        raise ValueError("model.predictor.type must be 'adaln', 'motion_token', 'forward_inverse', or 'v3_n1'")
    if version == "leworld" and predictor_type != "adaln":
        raise ValueError("model.version='leworld' expects predictor.type='adaln'")
    if version != "leworld" and predictor_type != version:
        raise ValueError(f"model.version={version!r} expects predictor.type={version!r}")
    predictor = None
    if predictor_type == "motion_token":
        predictor_config.setdefault("hidden_dim", model["environment_encoder"].get("state_dim", 192))
        predictor = PREDICTORS.build({"name": "motion_token", **predictor_config})
    elif predictor_type == "forward_inverse":
        forward_config = dict(model.get("forward_predictor", {}))
        inverse_config = dict(model.get("inverse_head", {}))
        predictor_spec = {
            "name": "forward_inverse",
            "action_dim": predictor_config.get("action_dim", config.get("planner", {}).get("action_dim", 2)),
            "latent_dim": model.get("latent_dim", model["environment_encoder"].get("state_dim", 192)),
            "history_size": model.get("history_size", config.get("data", {}).get("history_length", 3)),
            "forward_num_layers": forward_config.get("num_layers", 2),
            "forward_num_heads": forward_config.get("num_heads", 3),
            "forward_hidden_dim": forward_config.get("hidden_dim", 384),
            "dropout": forward_config.get("dropout", 0.0),
            "action_hidden_dim": forward_config.get("action_hidden_dim"),
            "inverse_enabled": inverse_config.get("enabled", True),
            "inverse_hidden_dim": inverse_config.get("hidden_dim", 384),
            "inverse_num_layers": inverse_config.get("num_layers", 2),
        }
        predictor = PREDICTORS.build(predictor_spec)
    elif predictor_type == "v3_n1":
        action_consistency_config = dict(model.get("action_consistency_head", {}))
        temporal_config = dict(model.get("temporal_encoder", {}))
        predictor = PREDICTORS.build({
            "name": "v3_n1",
            "image_channels": model.get("image_channels", 3),
            "latent_dim": model.get("latent_dim", 192),
            "action_dim": predictor_config.get("action_dim", config.get("planner", {}).get("action_dim", 2)),
            "image_size": model.get("image_size", 224),
            "patch_size": model.get("patch_size", 14),
            "vit_depth": model.get("vit_depth", 12),
            "vit_heads": model.get("vit_heads", 3),
            "mlp_ratio": model.get("mlp_ratio", 4.0),
            "logvar_min": model.get("logvar_min", -10.0),
            "logvar_max": model.get("logvar_max", 10.0),
            "action_consistency_hidden_dim": action_consistency_config.get("hidden_dim", 384),
            "history_size": model.get("history_size", config.get("data", {}).get("history_length", 3)),
            "temporal_layers": temporal_config.get("num_layers", 2),
            "temporal_heads": temporal_config.get("num_heads", 3),
            "temporal_dropout": temporal_config.get("dropout", 0.0),
        })
    return AgentCentricWorldModel(
        AGENT_ENCODERS.build(model["agent_encoder"]),
        ENVIRONMENT_ENCODERS.build(model["environment_encoder"]),
        AGENT_TRANSITIONS.build(model["agent_transition"]),
        ENVIRONMENT_TRANSITIONS.build(model["environment_transition"]),
        predictor=predictor,
        predictor_type=predictor_type,
        history_size=model.get("history_size", predictor_config.get("history_size",
                                                                     config.get("data", {}).get("history_length", 3))),
    )
