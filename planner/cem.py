import torch


class CEMPlanner:
    """Replaceable sampling planner operating exclusively in latent space."""

    def __init__(self, horizon: int, action_dim: int, action_low: float | list[float] = -1.0,
                 action_high: float | list[float] = 1.0, population: int = 256,
                 elites: int = 32, iterations: int = 5, var_scale: float = 1.0,
                 clamp_actions: bool = True, cost_type: str = "model",
                 seed: int | None = None, std_correction: int = 1,
                 min_std: float = 1e-4,
                 action_mean: list[float] | None = None,
                 action_std: list[float] | None = None):
        if not 0 < elites <= population:
            raise ValueError("elites must be between 1 and population")
        self.horizon, self.action_dim = horizon, action_dim
        self.population, self.elites, self.iterations = population, elites, iterations
        self.action_low, self.action_high = action_low, action_high
        self.var_scale = var_scale
        self.clamp_actions = clamp_actions
        if cost_type not in {"model", "latent_mse_sum"}:
            raise ValueError("unknown planning cost_type")
        if std_correction not in {0, 1} or min_std < 0:
            raise ValueError("invalid CEM std settings")
        if (action_mean is None) != (action_std is None):
            raise ValueError("action_mean and action_std must be supplied together")
        if action_mean is not None and len(action_mean) != action_dim:
            raise ValueError("action_mean must match action_dim")
        if action_std is not None and (len(action_std) != action_dim or min(action_std) <= 0):
            raise ValueError("action_std must contain positive per-action scales")
        self.cost_type, self.seed = cost_type, seed
        self.std_correction, self.min_std = std_correction, min_std
        self.action_mean, self.action_std = action_mean, action_std
        self._generators = {}
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def plan(self, model, history_frames: torch.Tensor, history_actions: torch.Tensor,
             current_frame: torch.Tensor, goal_frame: torch.Tensor,
             init_action: torch.Tensor | None = None) -> torch.Tensor:
        """Returns [B,H,A]; batching lets each input state get its own search."""
        agent, environment = model.encode(history_frames, history_actions, current_frame)
        goal = model.encode_goal(goal_frame) if hasattr(model, "encode_goal") else model.environment_encoder(goal_frame)
        batch, device = agent.shape[0], agent.device
        offset = torch.as_tensor(self.action_mean if self.action_mean is not None else [0.0]*self.action_dim,
                                 device=device, dtype=environment.dtype)
        scale = torch.as_tensor(self.action_std if self.action_std is not None else [1.0]*self.action_dim,
                                device=device, dtype=environment.dtype)
        generator = None
        if self.seed is not None:
            key = str(device)
            if key not in self._generators:
                self._generators[key] = torch.Generator(device=device).manual_seed(self.seed)
            generator = self._generators[key]
        low = torch.as_tensor(self.action_low, device=device).expand(self.action_dim)
        high = torch.as_tensor(self.action_high, device=device).expand(self.action_dim)
        mean = torch.zeros(batch, self.horizon, self.action_dim, device=device, dtype=environment.dtype)
        if init_action is not None:
            init_action = init_action.to(device=device, dtype=environment.dtype)
            assert init_action.ndim == 3 and init_action.shape[0] == batch and init_action.shape[-1] == self.action_dim, (
                f"init_action must be [B,<=H,{self.action_dim}], got {tuple(init_action.shape)}"
            )
            keep = min(init_action.shape[1], self.horizon)
            mean[:, :keep] = (init_action[:, :keep]-offset)/scale
        std = torch.full_like(mean, self.var_scale)
        final_cost = None
        for _ in range(self.iterations):
            actions = mean[:, None] + std[:, None] * torch.randn(
                batch, self.population, self.horizon, self.action_dim,
                device=device, dtype=environment.dtype,
                generator=generator,
            )
            if self.clamp_actions:
                actions = actions.clamp(low, high)
            actions[:, 0] = mean
            # Search in standardized coordinates; raw-step Predictor still receives
            # physical relative actions [B*population,H,2], without changing training.
            flat_actions = (actions*scale+offset).flatten(0, 1)
            agent_population = agent[:, None].expand(-1, self.population, *agent.shape[1:]).flatten(0, 1)
            environment_population = environment[:, None].expand(-1, self.population, -1).flatten(0, 1)
            states = model.rollout(agent_population, environment_population, flat_actions)
            final = states[-1].environment.view(batch, self.population, -1)
            # The goal criterion is environment-only. Agent latent participates
            # in causal rollout but is deliberately absent from task scoring.
            cost = ((final-goal[:, None]).square().sum(-1) if self.cost_type == "latent_mse_sum"
                    else model.planning_cost(final, goal[:, None]))
            indices = cost.topk(self.elites, largest=False).indices
            elite = actions.gather(1, indices[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim))
            mean, std = elite.mean(1), elite.std(1, correction=self.std_correction).clamp_min(self.min_std)
            final_cost = cost.gather(1, indices).mean(1)
        plan = (mean*scale+offset).detach()
        self.last_diagnostics = {
            "cem_cost": float(final_cost.mean().item()) if final_cost is not None else 0.0,
            "cem_action_abs_mean": float(mean.abs().mean().item()),
            "cem_action_abs_max": float(mean.abs().max().item()),
            "cem_action_std": float(mean.std().item()),
        }
        return plan
