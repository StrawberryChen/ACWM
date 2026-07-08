"""Download LeRobot Push-T and convert complete episodes to ACWM trajectory NPZs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lerobot/pusht_image")
    parser.add_argument("--output-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", default="configs/default.yaml")
    parser.add_argument("--config-output", default="configs/colab.yaml")
    return parser.parse_args()


def save_episode(dataset, indices: list[int], path: Path) -> None:
    rows = dataset.select(indices).sort("frame_index")
    frames = np.stack([np.asarray(image.convert("RGB"), dtype=np.uint8) for image in rows["observation.image"]])
    all_actions = np.asarray(rows["action"], dtype=np.float32)
    if len(frames) < 2:
        raise ValueError(f"episode at {path} has fewer than two frames")
    # LeRobot stores an action on every row; the last has no following frame.
    actions = all_actions[:-1]
    np.savez_compressed(path, frames=frames, actions=actions)


def write_colab_config(args: argparse.Namespace, root: Path) -> None:
    with open(args.base_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # This generated configuration is specifically for a GPU Colab runtime.
    config["device"] = "cuda"
    config["data"]["train_paths"] = [str(root / "data/train/*.npz")]
    config["data"]["val_paths"] = [str(root / "data/val/*.npz")]
    config["training"]["checkpoint"] = str(root / "checkpoints/acwm_latest.pt")
    config["training"]["best_checkpoint"] = str(root / "checkpoints/acwm_best.pt")
    config["training"]["metrics_file"] = str(root / "outputs/metrics.jsonl")
    config["validation"]["video_dir"] = str(root / "outputs/videos")
    output = Path(args.config_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    root = Path(args.output_root)
    train_dir, val_dir = root / "data/train", root / "data/val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.dataset} from Hugging Face...")
    dataset = load_dataset(args.dataset, split="train")
    episode_rows: dict[int, list[int]] = defaultdict(list)
    for row_index, episode_index in enumerate(dataset["episode_index"]):
        episode_rows[int(episode_index)].append(row_index)

    episodes = np.asarray(sorted(episode_rows))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    val_count = max(1, round(len(episodes) * args.val_fraction))
    val_episodes = set(episodes[:val_count].tolist())

    for position, episode in enumerate(episodes, start=1):
        split_dir = val_dir if int(episode) in val_episodes else train_dir
        save_episode(dataset, episode_rows[int(episode)], split_dir / f"episode_{int(episode):06d}.npz")
        print(f"\rConverted {position}/{len(episodes)} episodes", end="", flush=True)
    print(f"\nTrain episodes: {len(episodes) - val_count}; validation episodes: {val_count}")
    write_colab_config(args, root)
    print(f"Ready. Train with: python train.py --config {args.config_output}")


if __name__ == "__main__":
    main()
