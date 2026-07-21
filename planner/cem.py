import torch


class CEMPlanner:
    """Replaceable sampling planner operating exclusively in latent space."""

    def __init__(self, horizon: int, action_dim: int, action_low: float | list[float] = -1.0,
                 action_high: float | list[float] = 1.0, population: int = 256,
                 elites: int = 32, iterations: int = 5, var_scale: float = 1.0,
                 clamp_actions: bool = True):
        if not 0 < elites <= population:
            raise ValueError("elites must be between 1 and population")
        self.horizon, self.action_dim = horizon, action_dim
        self.population, self.elites, self.iterations = population, elites, iterations
        self.action_low, self.action_high = action_low, action_high
        self.var_scale = var_scale
        self.clamp_actions = clamp_actions
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def plan(self, model, history_frames: torch.Tensor, history_actions: torch.Tensor,
             current_frame: torch.Tensor, goal_frame: torch.Tensor,
             init_action: torch.Tensor | None = None) -> torch.Tensor:
        """Returns [B,H,A]; batching lets each input state get its own search."""
        agent, environment = model.encode(history_frames, history_actions, current_frame)
        goal = model.encode_goal(goal_frame) if hasattr(model, "encode_goal") else model.environment_encoder(goal_frame)
        batch, device = agent.shape[0], agent.device
        low = torch.as_tensor(self.action_low, device=device).expand(self.action_dim)
        high = torch.as_tensor(self.action_high, device=device).expand(self.action_dim)
        mean = torch.zeros(batch, self.horizon, self.action_dim, device=device, dtype=environment.dtype)
        if init_action is not None:
            init_action = init_action.to(device=device, dtype=environment.dtype)
            assert init_action.ndim == 3 and init_action.shape[0] == batch and init_action.shape[-1] == self.action_dim, (
                f"init_action must be [B,<=H,{self.action_dim}], got {tuple(init_action.shape)}"
            )
            keep = min(init_action.shape[1], self.horizon)
            mean[:, :keep] = init_action[:, :keep]
        std = torch.full_like(mean, self.var_scale)
        final_cost = None
        for _ in range(self.iterations):
            actions = mean[:, None] + std[:, None] * torch.randn(
                batch, self.population, self.horizon, self.action_dim,
                device=device, dtype=environment.dtype,
            )
            if self.clamp_actions:
                actions = actions.clamp(low, high)
            actions[:, 0] = mean
            flat_actions = actions.flatten(0, 1)
            agent_population = agent[:, None].expand(-1, self.population, *agent.shape[1:]).flatten(0, 1)
            environment_population = environment[:, None].expand(-1, self.population, -1).flatten(0, 1)
            states = model.rollout(agent_population, environment_population, flat_actions)
            final = states[-1].environment.view(batch, self.population, -1)
            # The goal criterion is environment-only. Agent latent participates
            # in causal rollout but is deliberately absent from task scoring.
            cost = model.planning_cost(final, goal[:, None])
            indices = cost.topk(self.elites, largest=False).indices
            elite = actions.gather(1, indices[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim))
            mean, std = elite.mean(1), elite.std(1).clamp_min(1e-4)
            final_cost = cost.gather(1, indices).mean(1)
        plan = mean.detach()
        self.last_diagnostics = {
            "cem_cost": float(final_cost.mean().item()) if final_cost is not None else 0.0,
            "cem_action_abs_mean": float(plan.abs().mean().item()),
            "cem_action_abs_max": float(plan.abs().max().item()),
            "cem_action_std": float(plan.std().item()),
        }
        return plan
