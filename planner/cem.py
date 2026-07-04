import torch


class CEMPlanner:
    """Replaceable sampling planner operating exclusively in latent space."""

    def __init__(self, horizon: int, action_dim: int, action_low: float | list[float] = -1.0,
                 action_high: float | list[float] = 1.0, population: int = 256,
                 elites: int = 32, iterations: int = 5):
        if not 0 < elites <= population:
            raise ValueError("elites must be between 1 and population")
        self.horizon, self.action_dim = horizon, action_dim
        self.population, self.elites, self.iterations = population, elites, iterations
        self.action_low, self.action_high = action_low, action_high

    @torch.no_grad()
    def plan(self, model, history_frames: torch.Tensor, history_actions: torch.Tensor,
             current_frame: torch.Tensor, goal_frame: torch.Tensor) -> torch.Tensor:
        """Returns [B,H,A]; batching lets each input state get its own search."""
        agent, environment = model.encode(history_frames, history_actions, current_frame)
        goal = model.environment_encoder(goal_frame)
        batch, device = agent.shape[0], agent.device
        low = torch.as_tensor(self.action_low, device=device).expand(self.action_dim)
        high = torch.as_tensor(self.action_high, device=device).expand(self.action_dim)
        mean = ((low + high) / 2).view(1, 1, 1, -1).expand(batch, self.population, self.horizon, -1).clone()
        std = ((high - low) / 2).view(1, 1, 1, -1).expand_as(mean).clone()
        for _ in range(self.iterations):
            actions = (mean + std * torch.randn_like(mean)).clamp(low, high)
            flat_actions = actions.flatten(0, 1)
            states = model.rollout(
                agent[:, None].expand(-1, self.population, -1).flatten(0, 1),
                environment[:, None].expand(-1, self.population, -1).flatten(0, 1),
                flat_actions,
            )
            final = states[-1].environment.view(batch, self.population, -1)
            cost = (final - goal[:, None]).square().mean(dim=-1)
            indices = cost.topk(self.elites, largest=False).indices
            elite = actions.gather(1, indices[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim))
            elite_mean, elite_std = elite.mean(1, keepdim=True), elite.std(1, keepdim=True).clamp_min(1e-4)
            mean, std = elite_mean.expand_as(mean), elite_std.expand_as(std)
        return mean[:, 0]

