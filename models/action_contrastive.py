"""Single-frame action ranking model. Raw latents are never normalized in dynamics."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F


class ActionContrastiveWorldModel(nn.Module):
    predictor_type = "action_contrastive"

    def __init__(self, latent_dim=128, image_size=96, architecture="residual_mlp"):
        super().__init__()
        if architecture not in {"residual_mlp", "concat_mlp", "film"}:
            raise ValueError(f"unknown architecture: {architecture}")
        self.latent_dim, self.image_size = latent_dim, image_size
        self.architecture = architecture
        layers, channels = [], 3
        for width in (32, 64, 128, 256):
            layers.extend([nn.Conv2d(channels, width, 3, 2, 1), nn.SiLU()])
            channels = width
        self.observation_encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1),
                                                 nn.Flatten(), nn.Linear(256, latent_dim))
        self.action_encoder = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32))
        self.transition_model = nn.Sequential(nn.Linear(latent_dim + 32, 256), nn.SiLU(),
                                              nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, latent_dim))
        if architecture == "film":
            self.modulation = nn.Linear(32, 2 * latent_dim)

    def encode_frame(self, image):
        # image [B,3,H,W] -> raw O [B,D].
        assert image.ndim == 4 and image.shape[1] == 3
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.observation_encoder(image)

    def transition(self, observation, action):
        # O [B,D], A [B,2] -> O_pred [B,D].
        assert observation.ndim == 2 and observation.shape[-1] == self.latent_dim
        assert action.shape == (observation.shape[0], 2)
        embedding = self.action_encoder(action)
        state = observation
        if self.architecture == "film":
            scale, shift = self.modulation(embedding).chunk(2, -1)
            state = observation * (1 + scale) + shift
        delta = self.transition_model(torch.cat((state, embedding), -1))
        return delta if self.architecture == "concat_mlp" else observation + delta

    def candidate_scores(self, observation, candidates, target):
        # candidates [B,K,2]; predictions [B,K,D]; scores [B,K].
        b, k, a = candidates.shape
        assert a == 2 and observation.shape == target.shape == (b, self.latent_dim)
        predictions = self.transition(observation[:, None].expand(-1, k, -1).reshape(b*k, -1),
                                      candidates.reshape(b*k, 2)).reshape(b, k, -1)
        return (F.normalize(predictions, dim=-1) * F.normalize(target, dim=-1)[:, None]).sum(-1)

    def forward(self, image, action):
        return self.transition(self.encode_frame(image), action)

    def rollout_latents(self, observation, actions):
        # Sequential time dependency; all B candidate sequences run in parallel.
        # actions [B,H,2] -> predictions [B,H,D].
        assert actions.ndim == 3 and actions.shape[0] == observation.shape[0] and actions.shape[-1] == 2
        predictions = []
        for action in actions.unbind(1):
            observation = self.transition(observation, action)
            predictions.append(observation)
        return torch.stack(predictions, 1)

    def encode(self, history_frames, history_actions, current_frame):
        observation = self.encode_frame(current_frame)
        return observation, observation

    encode_goal = encode_frame

    def rollout(self, agent, environment, actions):
        predictions = self.rollout_latents(environment, actions)
        return [SimpleNamespace(environment=z, agent=z) for z in predictions.unbind(1)]

    def planning_cost(self, final, goal):
        return 1 - F.cosine_similarity(final, goal, dim=-1)


def negative_actions(action, count=1, kind="opposite", generator=None):
    """Return [B,K,2]; positive is always candidate zero (not passed to F as an index)."""
    if count < 1:
        raise ValueError("at least one negative action is required")
    if kind not in {"opposite", "orthogonal", "random", "mixed"}:
        raise ValueError(kind)
    values = [action]
    for i in range(count):
        choice = ("opposite", "orthogonal", "random")[i % 3] if kind == "mixed" else kind
        if choice == "random":
            negative = torch.rand(action.shape, device=action.device, generator=generator) * 2 - 1
        elif choice == "opposite":
            # Preserve exact -A for the first negative; vary angle for extra negatives.
            if i == 0:
                values.append(-action)
                continue
            angle = torch.as_tensor(torch.pi if i == 0 else torch.pi + i * torch.pi / (count + 1), device=action.device)
            x, y = action.unbind(-1)
            negative = torch.stack((angle.cos()*x-angle.sin()*y, angle.sin()*x+angle.cos()*y), -1)
        else:
            x, y = action.unbind(-1)
            negative = torch.stack((-y, x), -1) * (-1 if i % 2 else 1) * (1 - 0.5*i/count)
        # Rotations of actions in the square can leave the square; retain direction
        # and shrink magnitude only when needed to stay in the valid action range.
        negative = negative / negative.abs().amax(-1, keepdim=True).clamp_min(1)
        values.append(negative)
    return torch.stack(values, 1)


def ranking_metrics(scores, temperature=0.1):
    assert scores.ndim == 2 and scores.shape[1] >= 2 and temperature > 0
    target = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
    gap = scores[:, 0] - scores[:, 1:].max(-1).values
    return {"loss": F.cross_entropy(scores / temperature, target),
            # Ties are not counted as successful action discrimination.
            "ranking_acc": (gap > 0).float().mean(),
            "mean_rank": (1 + (scores[:, 1:] >= scores[:, :1]).sum(-1)).float().mean(),
            "score_gap": gap.mean(), "positive_score": scores[:, 0].mean(),
            "hardest_negative_score": scores[:, 1:].max(-1).values.mean()}
