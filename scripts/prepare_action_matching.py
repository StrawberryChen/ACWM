"""V2 real counterfactual data: deterministic prefix replay, never fabricated images."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.action_matching import equivalent_futures
from evaluation.planning import PlanningEvaluator
from scripts.prepare_pusht_zarr import ensure_zarr, array_from_group


def physical_snapshot(env):
    # Include block velocities/angular velocity, absent from the usual 7D observation.
    values = []
    for body in (env.unwrapped.agent, env.unwrapped.block):
        values.extend((*body.position, *body.velocity, body.angle, body.angular_velocity,
                       *body.force, body.torque))
    return np.asarray(values, dtype=np.float64)


def candidates(action, k, kind, rng):
    if k < 2:
        raise ValueError("num_branches must be at least 2")
    if kind not in {"mixed", "random", "orthogonal"}:
        raise ValueError(kind)
    values = [action, -action]
    for i in range(k-2):
        if kind == "random" or (kind == "mixed" and i % 2):
            candidate = rng.uniform(-1, 1, 2)
        else:
            candidate = np.asarray([-action[1], action[0]]) * (-1 if i % 2 else 1)
            candidate *= 1-0.5*i/max(1, k-2)
        values.append(candidate)
    return np.asarray(values, dtype=np.float32)


def collect_episode(evaluator, initial_state, goal_state, absolute_actions, config, episode_id):
    """Each branch forks the same physical history by exact seeded reset+prefix replay."""
    primary, replay = evaluator._make_env(), evaluator._make_env()
    seed = int(config["seed"])+int(episode_id)
    collection = config["collection"]
    stride, k = int(collection["branch_stride"]), int(collection["num_branches"])
    if stride < 1:
        raise ValueError("branch_stride must be positive")
    options = {"state": evaluator._state_for_env(initial_state),
               "goal_state": evaluator._state_for_env(goal_state)}
    rng = np.random.default_rng(seed)
    current_images, futures, action_sets, masks, timesteps = [], [], [], [], []
    reference_snapshots, states, trajectory_frames, executed = [], [], [], []
    try:
        obs, _ = primary.reset(seed=seed, options=options)
        trajectory_frames.append(primary.render().copy())
        states.append(obs["state"].copy())
        for t, absolute_action in enumerate(tqdm(absolute_actions, desc=f"Collect episode {episode_id}", leave=False)):
            # Only action preprocessing reads agent xy. It is never returned in a model batch.
            action = np.clip((absolute_action[:2]-np.asarray(primary.unwrapped.agent.position))/config["data"]["action_scale"], -1, 1).astype(np.float32)
            branching = t % stride == 0
            if branching:
                branch_actions = candidates(action, k, collection["extra_actions"], rng)
                current = trajectory_frames[-1]
                snapshot = physical_snapshot(primary)
                other_futures = []
                for alternative in branch_actions[1:]:
                    replay.reset(seed=seed, options=options)
                    for previous in executed:
                        replay.step(previous)
                    np.testing.assert_allclose(physical_snapshot(replay), snapshot, rtol=0, atol=1e-6,
                                               err_msg="branch prefix did not recreate full dynamic state")
                    np.testing.assert_array_equal(replay.render(), current,
                                                  err_msg="branch must start from the identical observation")
                    replay.step(alternative)
                    other_futures.append(replay.render().copy())
            obs, _, _, _, _ = primary.step(action)
            next_frame = primary.render().copy()
            trajectory_frames.append(next_frame)
            states.append(obs["state"].copy())
            executed.append(action.copy())
            if branching:
                future = np.stack([next_frame, *other_futures])
                current_images.append(current)
                futures.append(future)
                action_sets.append(branch_actions)
                masks.append(equivalent_futures(future, collection["equivalent_pixel_tolerance"]))
                timesteps.append(t)
                reference_snapshots.append(snapshot)
    finally:
        primary.close()
        replay.close()
    branches = {"current_frames": np.stack(current_images), "future_frames": np.stack(futures),
                "actions": np.stack(action_sets), "positive_mask": np.stack(masks),
                "timesteps": np.asarray(timesteps), "start_physics": np.stack(reference_snapshots),
                "source_episode": episode_id, "frame_skip": 1, "action_mode": "relative"}
    trajectory = {"frames": np.stack(trajectory_frames), "actions": np.stack(executed),
                  "states": np.stack(states), "source_episode": episode_id,
                  "frame_skip": 1, "action_mode": "relative", "action_scale": config["data"]["action_scale"]}
    return branches, trajectory


def save_npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--download-dir", default="/content/ACWM/data_src")
    parser.add_argument("--url", default="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip")
    parser.add_argument("--base-config", default="configs/action_matching_v2.yaml")
    parser.add_argument("--config-output", default="configs/colab_action_matching_v2.yaml")
    args = parser.parse_args()
    import zarr
    with open(args.base_config) as stream:
        config = yaml.safe_load(stream)
    if config["data"]["action_scale"] != 100 or not config["environment"]["relative_action"]:
        raise ValueError("V2 collection requires swm relative actions with scale 100")
    source = zarr.open(str(ensure_zarr(args)), mode="r")
    states = array_from_group(source["data"], ("state",)).astype(np.float32)
    raw_actions = array_from_group(source["data"], ("action",)).astype(np.float32)
    ends = array_from_group(source["meta"], ("episode_ends",)).astype(int)
    assert len(states) == len(raw_actions) and states.shape[1] >= 5 and raw_actions.shape[1] == 2
    order = np.random.default_rng(config["seed"]).permutation(len(ends))
    nv = max(1, round(len(ends)*config["data"]["val_fraction"]))
    nt = max(1, round(len(ends)*config["data"]["test_fraction"]))
    if nv+nt >= len(ends):
        raise ValueError("not enough episodes for three splits")
    splits = {"val": set(order[:nv]), "test": set(order[nv:nv+nt]), "train": set(order[nv+nt:])}
    root = Path(args.output_root)/"action_matching_v2"
    manifest = {"schema": 2, "seed": config["seed"], "url": args.url,
                "collection": config["collection"], "environment": config["environment"],
                "action_scale": config["data"]["action_scale"],
                "stable_worldmodel_version": version("stable-worldmodel"),
                "pymunk_version": version("pymunk"),
                "splits": {name: sorted(map(int, ids)) for name, ids in splits.items()}}
    path = root/"data/manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as stream:
            if yaml.safe_load(stream) != manifest:
                raise ValueError("collection manifest differs; choose another --output-root")
    with path.open("w") as stream:
        yaml.safe_dump(manifest, stream)
    evaluator = PlanningEvaluator(None, config["environment"], 1, 2, "cpu")
    start = 0
    for episode, end in enumerate(tqdm(ends, desc="Simulating observed action branches")):
        split = next(name for name, ids in splits.items() if episode in ids)
        branch_path = root/f"data/branches/{split}/episode_{episode:06d}.npz"
        trajectory_path = root/f"data/trajectories/{split}/episode_{episode:06d}.npz"
        if not (branch_path.exists() and trajectory_path.exists()):
            if end-start < 2:
                start = end
                continue
            branches, trajectory = collect_episode(evaluator, states[start, :5], states[end-1, :5],
                                                   raw_actions[start:end-1], config, episode)
            save_npz(branch_path, branches)
            save_npz(trajectory_path, trajectory)
        start = end
    for split in splits:
        config["data"][f"{split}_paths"] = [str(root/f"data/trajectories/{split}/episode_*.npz")]
        config["data"][f"branch_{split}_paths"] = [str(root/f"data/branches/{split}/episode_*.npz")]
    config["output_dir"] = str(root/"outputs")
    config["device"] = "cuda"
    output = Path(args.config_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Prepared real K={config['collection']['num_branches']} branch groups; old experiments untouched.")
    print(f"Train: python train_action_matching.py --config {output}")


if __name__ == "__main__":
    main()
