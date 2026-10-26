"""Explicit, diagnostics-gated terminal-measurement CEM + MPC.

Goal images NEVER go through CSE. Motion predictions NEVER contribute to cost.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from evaluation.planning import PlanningEvaluator
from experiment_tracking import ExperimentLogger
from .data import file_sha256, load_config, validate_splits
from .joint_model import joint_euler_sample
from .joint_train import build_joint, load_weights


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def observation_frame(env, evaluator, settings):
    """Keep the dataset's visual marker fixed while physical success goal varies.

    Changing the task goal must not silently change the training image appearance.
    Only the drawing pose is temporarily changed; goal_state/reward are untouched.
    """
    base = env.unwrapped
    previous = getattr(base, "goal_pose", None)
    pose = settings.get("render_goal_pose")
    if pose is not None and previous is not None:
        base.goal_pose = np.asarray(pose, dtype=np.float32)
    try:
        return evaluator._image(env.render())[None]
    finally:
        if previous is not None:
            base.goal_pose = previous


class TerminalCEMPlanner:
    def __init__(self, settings, device):
        self.settings, self.device = settings, torch.device(device)
        required_positive = ("num_candidates", "num_elites", "iterations", "horizon", "execute_steps",
                             "future_samples", "flow_steps", "candidate_batch_size")
        if any(int(settings[k]) < 1 for k in required_positive):
            raise ValueError("positive planning dimensions required")
        if settings["num_elites"] > settings["num_candidates"] or settings["execute_steps"] > settings["horizon"]:
            raise ValueError("invalid elite count or execute_steps")
        if settings["action_low"] >= settings["action_high"] or settings["initial_std"] <= 0 or settings["min_std"] < 0:
            raise ValueError("invalid planning bounds/std")
        self.generator = torch.Generator(device=self.device).manual_seed(settings["seed"])

    @torch.no_grad()
    def plan(self, model, terminal, current, goal_image, initial_actions=None):
        model.eval()
        config = self.settings
        horizon, n, k = config["horizon"], config["num_candidates"], config["future_samples"]
        if horizon > model.horizon:
            raise ValueError("planning horizon exceeds trained model horizon")
        assert current.shape == goal_image.shape == (1, 3, 96, 96)
        synchronize(self.device)
        begin = time.perf_counter()
        state = model.cse(current)  # encode actual current frame ONCE, never goal
        goal = terminal(goal_image)  # exactly the training terminal operator
        mean = state.new_zeros(horizon, 2)
        if initial_actions is not None:
            take = min(horizon, len(initial_actions))
            mean[:take] = initial_actions[:take]
        mean.clamp_(config["action_low"], config["action_high"])
        std = torch.full_like(mean, config["initial_std"])
        # Common random numbers: same K source noises across all candidates and
        # CEM iterations of one plan. A new call draws new noise. Reduces noise bias.
        noise = {"terminal": torch.randn(k, model.terminal_dim, generator=self.generator, device=self.device)}
        if model.motion_dim is not None:
            noise["motion"] = torch.randn(k, horizon, model.motion_dim, generator=self.generator, device=self.device)
        best_cost, best_actions, best_terminals = float("inf"), None, None
        world_time, evaluations = 0., 0
        for _ in range(config["iterations"]):
            candidates = (mean[None]+std[None]*torch.randn(n, horizon, 2, generator=self.generator, device=self.device)).clamp(
                config["action_low"], config["action_high"])
            candidates[0] = mean
            costs, terminals = [], []
            for batch in candidates.split(config["candidate_batch_size"]):
                b = len(batch)
                actions = batch[:, None].expand(b, k, horizon, 2).reshape(b*k, horizon, 2)
                expanded_noise = {name: value[None].expand(b, *value.shape).reshape(b*k, *value.shape[1:])
                                  for name, value in noise.items()}
                synchronize(self.device)
                start = time.perf_counter()
                generated = joint_euler_sample(model.predictor, state.expand(b*k, -1), actions,
                                                expanded_noise, config["flow_steps"])
                synchronize(self.device)
                world_time += time.perf_counter()-start
                terminal_samples = generated["terminal"].reshape(b, k, model.terminal_dim)
                # Expected terminal measurement MSE. No motion or CSE latent cost.
                costs.append((terminal_samples-goal[:, None]).square().mean(-1).mean(-1))
                terminals.append(terminal_samples)
                evaluations += b
            costs, terminals = torch.cat(costs), torch.cat(terminals)
            if not torch.isfinite(costs).all():
                raise FloatingPointError("non-finite terminal planning cost")
            index = int(costs.argmin())
            if float(costs[index]) < best_cost:
                best_cost = float(costs[index])
                best_actions, best_terminals = candidates[index].clone(), terminals[index].clone()
            elite = candidates[costs.topk(config["num_elites"], largest=False).indices]
            mean = elite.mean(0)
            std = elite.std(0, correction=0).clamp_min(config["min_std"])
        synchronize(self.device)
        elapsed = time.perf_counter()-begin
        return {"actions": best_actions, "terminal_samples": best_terminals,
                "cost": best_cost, "cem_time": elapsed, "world_model_time": world_time,
                "candidates_per_second": evaluations/max(elapsed, 1e-12),
                "evaluated_candidates": evaluations, "future_samples": k}


def check_diagnostic_review(checkpoint, evaluation_report, probe_report, reviewed):
    if not reviewed:
        raise ValueError("review prediction/dependency/probe results first; pass --diagnostics-reviewed only after review")
    evaluation_path, probe_path = Path(evaluation_report), Path(probe_report)
    report = json.loads(evaluation_path.read_text())
    probes = json.loads(probe_path.read_text())
    probe_metadata = json.loads(probe_path.with_name("metadata.json").read_text())
    digest = file_sha256(checkpoint)
    if report["checkpoint_sha256"] != digest or probe_metadata["checkpoint_sha256"] != digest:
        raise ValueError("diagnostic reports must evaluate the exact checkpoint being planned")
    if report.get("split") != "val":
        raise ValueError("review validation diagnostics, not test-based model selection")
    required = ("generation/terminal_mse", "representation/feature_std",
                "dependency/state_shuffle_delta", "dependency/action_shuffle_delta")
    if any(key not in report["metrics"] or not math.isfinite(report["metrics"][key]) for key in required):
        raise ValueError("missing/non-finite prediction and representation diagnostics")
    if not any(key.startswith("probe/trained_cse/") and "/val/" in key and isinstance(value, (int, float))
               and math.isfinite(value) for key, value in probes.items()):
        raise ValueError("missing frozen-CSE validation probes")
    # No arbitrary numeric success threshold is fabricated here. Human review
    # decides whether diagnostics are meaningful; provenance only is machine checked.
    return {"checkpoint_sha256": digest, "evaluation_report": str(evaluation_path),
            "probe_report": str(probe_path), "reviewed": True}


@torch.no_grad()
def evaluate_mpc(model, terminal, config, dataset_paths, output, device):
    import imageio.v2 as imageio
    from PIL import Image
    settings = config["planning"]
    if settings["episodes"] < 1 or settings["max_steps"] < 1 or settings["video_episodes"] < 0:
        raise ValueError("positive episode count/budget and nonnegative video count required")
    planner = TerminalCEMPlanner(settings, device)
    # Reuse the existing environment/reset/success definitions, but NOT its latent planner/cost.
    env_config = dict(env_id="swm/PushT-v1", relative_action=True, with_target=settings.get("with_target", True),
                      observation_width=96, observation_height=96, goal_offset_steps=settings["goal_offset_steps"],
                      seed=settings["seed"])
    evaluator = PlanningEvaluator(None, env_config, 1, 2, device, dataset_paths=dataset_paths)
    cases = evaluator._dataset_eval_cases(settings["episodes"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scale = float(config["data"]["action_scale"])
    if scale <= 0:
        raise ValueError("positive action preprocessing scale required")
    episodes, plans = [], []
    for episode, case in enumerate(cases):
        env = evaluator._make_env()
        videos, next_actions, success, reward_sum = [], None, False, 0.
        try:
            obs, info = evaluator._reset_env(env, case["goal_state"], case["goal_state"])
            goal = observation_frame(env, evaluator, settings)
            obs, info = evaluator._reset_env(env, case["start_state"], case["goal_state"])
            current = observation_frame(env, evaluator, settings)
            if episode < settings["video_episodes"]:
                videos.append(np.concatenate(((current[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8),
                                               (goal[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)), 1))
            step = 0
            while step < settings["max_steps"]:
                result = planner.plan(model, terminal, current, goal, next_actions)
                plans.append({key: result[key] for key in ("cost", "cem_time", "world_model_time", "candidates_per_second")})
                directory = output/f"episode_{episode:03d}/plan_{step:04d}"
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory/"prediction.npz", terminal_samples=result["terminal_samples"].cpu().numpy(),
                                     actions=result["actions"].cpu().numpy(), terminal_goal=terminal(goal).cpu().numpy(),
                                     cost=result["cost"])
                picture = terminal.image(result["terminal_samples"].mean(0, keepdim=True))[0].clamp(0, 1)
                Image.fromarray((picture.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)).save(directory/"predicted_terminal.png")
                take = min(settings["execute_steps"], settings["max_steps"]-step)
                for action in result["actions"][:take]:
                    # Dataset uses (target-agent)/action_scale; swm relative control
                    # uses target=agent+100*env_action. Convert units, do NOT silently
                    # feed /512-normalized model actions as /100 environment actions.
                    env_action = action.cpu().numpy()*(scale/100.)
                    obs, reward, terminated, truncated, info = env.step(env_action)
                    reward_sum += float(reward)
                    step += 1
                    current = observation_frame(env, evaluator, settings)
                    success = bool(terminated or info.get("is_success", False))
                    if episode < settings["video_episodes"]:
                        videos.append(np.concatenate(((current[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8),
                                                       (goal[0].permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)), 1))
                    if success or truncated:
                        break
                next_actions = result["actions"][take:]
                if success or truncated:
                    break
            final_distance = float(-reward)  # swm reward is negative state-goal distance
            episodes.append(dict(success=success, final_goal_distance=final_distance, episode_reward=reward_sum,
                                 steps=step, steps_to_success=step if success else None))
            if videos:
                imageio.mimsave(output/f"planning_episode_{episode:03d}.mp4", videos, fps=20)
            print(f"Terminal MPC episode {episode+1}/{len(cases)}: success={success}, distance={final_distance:.3f}")
        finally:
            env.close()
    success_steps = [row["steps_to_success"] for row in episodes if row["success"]]
    metrics = {"planning/success_rate": 100*sum(row["success"] for row in episodes)/len(episodes),
               "planning/final_goal_distance": float(np.mean([row["final_goal_distance"] for row in episodes])),
               "planning/episode_reward": float(np.mean([row["episode_reward"] for row in episodes])),
               "planning/successful_episodes": len(success_steps),
               "planning/mean_episode_steps": float(np.mean([row["steps"] for row in episodes])),
               "planning/horizon": settings["horizon"], "planning/execute_steps": settings["execute_steps"],
               "planning/future_samples": settings["future_samples"], "planning/flow_steps": settings["flow_steps"],
               "planning/action_scale": scale}
    if success_steps:
        metrics["planning/steps_to_success"] = float(np.mean(success_steps))
    for key in ("cem_time", "world_model_time", "candidates_per_second"):
        metrics[f"planning/{key}"] = float(np.mean([row[key] for row in plans]))
    (output/"metrics.json").write_text(json.dumps({"metrics": metrics, "episodes": episodes, "plans": plans}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="resolved output/config.yaml from joint training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics-report", required=True)
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--diagnostics-reviewed", action="store_true")
    parser.add_argument("--enable-planning", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    if not config["planning"]["enabled"] and not args.enable_planning:
        raise ValueError("planning disabled by default; finish and review Phase 1-5 first")
    config["planning"]["enabled"] = True  # record actual explicit runtime selection
    review = check_diagnostic_review(args.checkpoint, args.diagnostics_report, args.probe_report, args.diagnostics_reviewed)
    validate_splits(config)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("enable a GPU runtime")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, terminal, motion = build_joint(config, device)
    load_weights(saved, config, model, terminal, motion)
    output = Path(config["output_dir"])/f"planning_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    (output/"diagnostic_review.json").write_text(json.dumps(review, indent=2))
    import yaml
    (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    logger = ExperimentLogger({**config["wandb"], "name": config["wandb"]["name"]+"_planning"}, config)
    try:
        metrics = evaluate_mpc(model, terminal, config, config["data"][f"{args.split}_paths"], output, device)
        logger.log(metrics, step=0)  # no planning videos uploaded
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
