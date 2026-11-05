"""Exploratory latent CEM/MPC for the two continuous trajectory models.

No retraining, no future GT in the predictor, no progress term in planning cost.
Real environment outcomes, not predicted cost, determine success.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from evaluation.planning import PlanningEvaluator
from experiments.avs_target.data import load_config, file_sha256
from .train import build_model, load_checkpoint


def action_bounds(env, model_scale, env_scale):
    if model_scale <= 0 or env_scale <= 0:
        raise ValueError("positive model/environment action scales required")
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all() or np.any(low >= high):
        raise ValueError("expected finite, ordered 2D relative-action environment bounds")
    # Intersect training input support [-1,1] with actual environment bounds.
    model_low = np.maximum(-1., low*env_scale/model_scale)
    model_high = np.minimum(1., high*env_scale/model_scale)
    if np.any(model_low >= model_high) or np.any(model_low > 0) or np.any(model_high < 0):
        raise ValueError("invalid model-action support or zero baseline not admissible")
    return model_low, model_high


def environment_action(action, env, model_scale, env_scale):
    # Model A=(absolute_target-agent)/model_scale; environment uses /env_scale.
    result = np.asarray(action, dtype=np.float32)*(model_scale/env_scale)
    assert result.shape == (2,)
    clipped = np.clip(result, env.action_space.low, env.action_space.high).astype(np.float32)
    if not np.allclose(result, clipped, atol=1e-5, rtol=1e-6):
        raise ValueError("planner produced out-of-bounds action; refuse silent execution clipping")
    return clipped


class LatentTrajectoryCEM:
    def __init__(self, settings, low, high, device):
        self.settings, self.device = settings, torch.device(device)
        for key in ("num_candidates", "num_elites", "iterations", "candidate_batch_size", "horizon", "execute_steps"):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if settings["num_elites"] > settings["num_candidates"] or settings["execute_steps"] > settings["horizon"]:
            raise ValueError("invalid elites or execution horizon")
        if settings["initial_std"] <= 0 or settings["min_std"] <= 0:
            raise ValueError("positive CEM standard deviations required")
        self.low, self.high = (torch.as_tensor(v, device=self.device, dtype=torch.float32) for v in (low, high))
        if self.low.shape != (2,) or self.high.shape != (2,) or not (self.low < self.high).all():
            raise ValueError("ordered two-dimensional model action bounds required")
        self.generator = torch.Generator(device=self.device).manual_seed(settings["seed"])

    @torch.no_grad()
    def plan(self, model, start, goal, warm_start=None):
        model.eval()
        cfg = self.settings
        horizon, count = cfg["horizon"], cfg["num_candidates"]
        assert start.ndim == 2 and start.shape == goal.shape and len(start) == 1
        if horizon != model.horizon:
            raise ValueError("use the trained full horizon; action context/arrow depend on its length")
        mean = start.new_zeros(horizon, 2)
        if warm_start is not None:
            mean[:len(warm_start)] = warm_start
        mean = mean.clamp(self.low, self.high)
        std = torch.full_like(mean, cfg["initial_std"])
        best_cost, best_actions, best_prediction = float("inf"), None, None
        begin = time.perf_counter()
        for _ in range(cfg["iterations"]):
            actions = mean[None]+std[None]*torch.randn(count, horizon, 2, device=start.device, generator=self.generator)
            actions = actions.clamp(self.low, self.high)
            actions[0] = mean
            costs, predictions = [], []
            for batch in actions.split(cfg["candidate_batch_size"]):
                # EACH candidate gets its own sequence context and arrow, freshly
                # recomputed on EVERY CEM iteration. Goal is NOT a model input.
                predicted = model.rollout(start.expand(len(batch), -1), batch)  # [N,H,192]
                costs.append((predicted[:, -1]-goal).square().mean(-1))
                predictions.append(predicted)
            costs, predictions = torch.cat(costs), torch.cat(predictions)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("non-finite CEM cost")
            index = int(costs.argmin())
            if float(costs[index]) < best_cost:
                best_cost = float(costs[index])
                best_actions, best_prediction = actions[index].clone(), predictions[index].clone()
            elite = actions[costs.topk(cfg["num_elites"], largest=False).indices]
            mean = elite.mean(0)
            std = elite.std(0, correction=0).clamp_min(cfg["min_std"])
        if start.device.type == "cuda":
            torch.cuda.synchronize(start.device)
        return dict(actions=best_actions, prediction=best_prediction, cost=best_cost,
                    seconds=time.perf_counter()-begin)


def rendered_frame(env, evaluator, settings):
    # Preserve training's fixed green drawing marker, while actual evaluation
    # goal_state remains the sampled dataset state. Never conflate these goals.
    base = env.unwrapped
    previous = getattr(base, "goal_pose", None)
    try:
        if previous is not None:
            base.goal_pose = np.asarray(settings["render_goal_pose"], dtype=np.float32)
        return evaluator._image(env.render())[None]  # [1,3,S,S], float [0,1]
    finally:
        if previous is not None:
            base.goal_pose = previous


def video_frame(current, goal):
    return np.concatenate([(frame[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                           for frame in (current, goal)], axis=1)


@torch.no_grad()
def evaluate_planning(model, config, settings, dataset_paths, output, device):
    import imageio.v2 as imageio
    settings = dict(settings)
    settings["horizon"] = model.horizon
    if settings["episodes"] < 1 or settings["max_steps"] < 1 or settings["goal_offset_steps"] < 1:
        raise ValueError("positive episode count, step budget and dataset goal offset required")
    if settings["video_episodes"] < 0 or settings["video_fps"] < 1:
        raise ValueError("invalid video settings")
    if not settings["policies"] or len(set(settings["policies"])) != len(settings["policies"]) or any(
            policy not in {"cem", "zero", "random"} for policy in settings["policies"]):
        raise ValueError("policies must be a nonempty subset of cem, zero, random")
    size = config["model"]["image_size"]
    evaluator = PlanningEvaluator(None, dict(env_id="swm/PushT-v1", relative_action=True,
                                            observation_width=size, observation_height=size,
                                            with_target=settings["with_target"], seed=settings["seed"],
                                            goal_offset_steps=settings["goal_offset_steps"]),
                                  1, 2, device, dataset_paths=dataset_paths)
    print("Sampling fixed dataset start/goal cases (state labels used ONLY for environment evaluation)...", flush=True)
    cases = evaluator._dataset_eval_cases(settings["episodes"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"cases.json").write_text(json.dumps([
        dict(path=case["path"], timestep=int(case["current"]), start=case["start_state"].tolist(),
             goal=case["goal_state"].tolist()) for case in cases], indent=2))
    scale = float(config["data"]["action_scale"])
    results = {}
    for policy in settings["policies"]:
        policy_dir = output/policy
        policy_dir.mkdir(exist_ok=True)
        rows = []
        for episode, case in enumerate(cases):
            print(f"[{policy}] episode {episode+1}/{len(cases)}: reset", flush=True)
            env = evaluator._make_env()
            try:
                env.reset(seed=settings["seed"]+episode)
                evaluator._reset_env(env, case["goal_state"], case["goal_state"])
                goal_image = rendered_frame(env, evaluator, settings)
                goal = model.encoder(goal_image)  # Goal latent used ONLY by CEM scoring.
                evaluator._reset_env(env, case["start_state"], case["goal_state"])
                current = rendered_frame(env, evaluator, settings)
                low, high = action_bounds(env, scale, settings["env_action_scale"])
                planner = LatentTrajectoryCEM({**settings, "seed": settings["seed"]+episode}, low, high, device)
                rng = np.random.default_rng(settings["seed"]+episode)
                frames = [video_frame(current, goal_image)] if episode < settings["video_episodes"] else []
                step, success, stopped, warm_start = 0, False, False, None
                initial_latent_cost = float((model.encoder(current)-goal).square().mean())
                plans, rewards, executed, real_goal_distances = [], [], [], []
                progress = tqdm(total=settings["max_steps"], desc=f"{policy} {episode+1}/{len(cases)}", dynamic_ncols=True)
                try:
                    while step < settings["max_steps"] and not stopped:
                        start = model.encoder(current)
                        if policy == "cem":
                            plan = planner.plan(model, start, goal, warm_start)
                            actions = plan["actions"]
                            print(f"  plan step={step}: latent_cost={plan['cost']:.6g}, seconds={plan['seconds']:.2f}", flush=True)
                        else:
                            array = (np.zeros((model.horizon, 2), dtype=np.float32) if policy == "zero" else
                                     rng.uniform(low, high, size=(model.horizon, 2)).astype(np.float32))
                            actions = torch.as_tensor(array, device=device)
                            plan = None
                        take = min(settings["execute_steps"], settings["max_steps"]-step)
                        for action in actions[:take]:
                            raw = environment_action(action.cpu().numpy(), env, scale, settings["env_action_scale"])
                            _, reward, terminated, truncated, info = env.step(raw)
                            current = rendered_frame(env, evaluator, settings)
                            rewards.append(float(reward))
                            # Existing swm reward contract: negative state-goal distance.
                            real_goal_distances.append(float(-reward))
                            executed.append(action.cpu().numpy().tolist())
                            step += 1
                            progress.update(1)
                            progress.set_postfix(distance=f"{-reward:.3f}")
                            success = bool(terminated or info.get("is_success", False))
                            stopped = success or bool(truncated)
                            if frames:
                                frames.append(video_frame(current, goal_image))
                            if stopped:
                                break
                        if plan is not None:
                            observed = model.encoder(current)
                            # Number actually executed may be <take on success/truncation.
                            actual = step-(plans[-1]["end_step"] if plans else 0)
                            row = dict(end_step=step, cost=plan["cost"], seconds=plan["seconds"],
                                       executed_endpoint_mse=float((plan["prediction"][actual-1:actual]-observed).square().mean()),
                                       observed_goal_latent_mse=float((observed-goal).square().mean()),
                                       environment_goal_distance=real_goal_distances[-1])
                            plans.append(row)
                            np.savez_compressed(policy_dir/f"episode_{episode:03d}_plan_{step:04d}.npz",
                                                actions=actions.cpu().numpy(), predicted_latents=plan["prediction"].cpu().numpy(),
                                                goal_latent=goal.cpu().numpy(), observed_latent=observed.cpu().numpy())
                            warm_start = actions[take:]
                finally:
                    progress.close()
                distances = np.asarray(real_goal_distances)
                row = dict(episode=episode, success=success, steps=step,
                           final_goal_distance=float(distances[-1]), best_goal_distance=float(distances.min()),
                           reward_sum=sum(rewards), initial_goal_latent_mse=initial_latent_cost,
                           final_goal_latent_mse=float((model.encoder(current)-goal).square().mean()),
                           mean_executed_displacement_pixels=float(np.linalg.norm(np.asarray(executed)*scale, axis=-1).mean()),
                           action_low_model_units=low.tolist(), action_high_model_units=high.tolist(), plans=plans,
                           executed_model_actions=executed)
                rows.append(row)
                (policy_dir/f"episode_{episode:03d}.json").write_text(json.dumps(row, indent=2))
                if frames:
                    imageio.mimsave(policy_dir/f"planning_episode_{episode:03d}.mp4", frames, fps=settings["video_fps"])
                print(f"[{policy}] {episode+1}/{len(cases)} success={success}, final_distance={distances[-1]:.3f}", flush=True)
            finally:
                env.close()
        summary = dict(episodes=len(rows), successful_episodes=sum(row["success"] for row in rows),
                       success_rate_percent=100*sum(row["success"] for row in rows)/len(rows),
                       mean_final_goal_distance=float(np.mean([row["final_goal_distance"] for row in rows])),
                       mean_best_goal_distance=float(np.mean([row["best_goal_distance"] for row in rows])))
        results[policy] = {"summary": summary, "episodes": rows}
        (output/"metrics.json").write_text(json.dumps(results, indent=2))
    return results


def discover_runs(root):
    root = Path(root)
    patterns = ("continuous_trajectory*/runs/*/checkpoints/latest.pt",
                "continuous_trajectory/*/checkpoints/latest.pt",
                "avs_target/outputs/continuous_trajectory/*/checkpoints/latest.pt")
    runs = sorted({p.parent.parent for pattern in patterns for p in root.glob(pattern)
                   if (p.parent.parent/"config.yaml").is_file()})
    return [run for run in runs if load_config(run/"config.yaml").get("experiment") == "continuous_trajectory"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="saved training config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--run", help="experiment directory; infer config.yaml and checkpoints/latest.pt")
    parser.add_argument("--choose", action="store_true", help="interactively choose an existing Drive experiment")
    parser.add_argument("--runs-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--planning-config", default="configs/continuous_trajectory_planning.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.choose:
        if args.run or args.config or args.checkpoint:
            parser.error("use --choose OR explicit --run/config/checkpoint")
        runs = discover_runs(args.runs_root)
        if not runs:
            raise FileNotFoundError("no trajectory runs found; use --run with the actual experiment directory")
        for index, run in enumerate(runs):
            spec = load_config(run/"config.yaml")
            print(f"[{index}] {run}\n    direction={spec['model']['direction_mode']}, "
                  f"scope={spec['model'].get('direction_scope', 'global')}, "
                  f"H={spec['model']['horizon']}, progress_weight={spec['loss']['weight']}", flush=True)
        choice = int(input("输入要做 planning 的实验编号："))
        if not 0 <= choice < len(runs):
            raise ValueError("experiment number out of range")
        args.run = str(runs[choice])
    if args.run:
        if args.config:
            parser.error("--run already selects its saved config; do not combine with --config")
        args.config = str(Path(args.run)/"config.yaml")
        args.checkpoint = args.checkpoint or str(Path(args.run)/"checkpoints/latest.pt")
    if not args.config or not args.checkpoint:
        parser.error("use --choose, --run, or both --config and --checkpoint")
    config, settings = load_config(args.config), load_config(args.planning_config)
    if config.get("experiment") != "continuous_trajectory":
        raise ValueError("use a continuous_trajectory action_only/state_action checkpoint")
    if config["data"]["action_mode"] != "relative":
        raise ValueError("this planning adapter requires relative-action dataset semantics")
    for key in ("episodes", "max_steps"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.device:
        config["device"] = args.device
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("select a GPU Colab runtime")
    torch.manual_seed(settings["seed"])
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, config, model)
    model.eval().requires_grad_(False)
    output = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent/"planning"/f"run_{int(time.time())}"
    if (output/"metadata.json").exists():
        raise FileExistsError("choose a fresh planning output directory")
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
                    direction_mode=model.direction_mode, direction_scope=getattr(model, "direction_scope", "global"),
                    split=args.split, settings={**settings, "horizon": model.horizon},
                    model_action_scale=config["data"]["action_scale"], env_action_scale=settings["env_action_scale"],
                    cost="mean squared terminal predicted latent vs encoded fixed goal image",
                    success="swm environment terminated or info.is_success; not CEM cost or fixed green T overlap",
                    note="exploratory 10-case evaluation, not a claim of official LeWorld benchmark alignment")
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Checkpoint: {args.checkpoint}\nPlanning output: {output}\nVideos: actual execution LEFT, fixed goal RIGHT", flush=True)
    result = evaluate_planning(model, config, settings, config["data"][f"{args.split}_paths"], output, device)
    print(json.dumps({policy: value["summary"] for policy, value in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
