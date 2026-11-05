"""One trainable arrow per real transition; only K substeps share an arrow."""
import torch
from .trajectory_model import TrajectoryWorldModel


class PerStepTrajectoryWorldModel(TrajectoryWorldModel):
    def __init__(self, direction_scope="per_step", **settings):
        if direction_scope != "per_step":
            raise ValueError("PerStepTrajectoryWorldModel requires direction_scope=per_step")
        super().__init__(**settings)
        # Exactly the same trainable parameter shapes/init as the matching global
        # version. Only token selection and timing of arrow computation differ.
        self.direction_scope = "per_step"

    def step_direction(self, state, contextual_action):
        inputs = (contextual_action if self.direction_mode == "action_only" else
                  torch.cat((state, contextual_action), -1))
        raw = self.direction_head(inputs)  # [B,192]; state is PREDICTED at this step.
        return raw, raw/(raw.norm(dim=-1, keepdim=True)+self.direction_eps)

    def condition(self, start, actions):
        # Existing sensitivity diagnostic asks only for the FIRST velocity.
        tokens, embeddings = self.action_trajectory.encode_tokens(actions)
        raw, direction = self.step_direction(start, tokens[:, 0])
        return tokens.mean(1), embeddings, raw, direction

    def predict_trajectory(self, start, actions):
        assert start.ndim == 2 and actions.shape[0] == len(start)
        tokens, embeddings = self.action_trajectory.encode_tokens(actions)
        context = tokens.mean(1)  # Full candidate context still available to dynamics.
        predictions, paths, velocities, directions, raw_directions = [], [], [], [], []
        state = start
        for h in range(actions.shape[1]):
            # No pooling before Direction Head: h uses its OWN contextual token.
            # Recompute from current predicted state for state_action; never use GT.
            raw, direction = self.step_direction(state, tokens[:, h])
            def conditioned_field(z, a):
                return self.field(z, a, context, direction)
            state, path, velocity = self.integrator(conditioned_field, state, embeddings[:, h])
            predictions.append(state)
            paths.append(path)
            velocities.append(velocity)
            directions.append(direction)
            raw_directions.append(raw)
        return dict(prediction=torch.stack(predictions, 1), states=torch.stack(paths, 1),
                    velocities=torch.stack(velocities, 1), direction=torch.stack(directions, 1),
                    direction_raw=torch.stack(raw_directions, 1))
        # prediction [B,H,D], states [B,H,K+1,D], V [B,H,K,D], arrows [B,H,D].
