"""Convert original Diffusion Policy PushT zarr data to ACWM NPZ episodes.

This is the LeWorld-aligned Push-T data path: unlike lerobot/pusht_image,
the original zarr replay buffer contains full simulator states
[agent_x, agent_y, block_x, block_y, block_angle], which are required for
dataset start/goal planning evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import yaml


DEFAULT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-config", default="configs/acwm_v3_n1.yaml")
    parser.add_argument("--config-output", default="configs/colab_acwm_v3_n1.yaml")
    parser.add_argument("--download-dir", default="/content/ACWM/data_src")
    parser.add_argument("--frame-skip", type=int, default=5)
    return parser.parse_args()


def write_colab_config(args: argparse.Namespace, root: Path) -> None:
    with open(args.base_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
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


def ensure_zarr(args: argparse.Namespace) -> Path:
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "pusht.zip"
    if not zip_path.exists():
        print(f"Downloading {args.url} ...")
        urllib.request.urlretrieve(args.url, zip_path)
    extract_dir = download_dir / "pusht"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(download_dir)
    candidates = list(download_dir.rglob("*.zarr"))
    if not candidates:
        raise FileNotFoundError(f"no .zarr directory found under {download_dir}")
    return candidates[0]


def array_from_group(group, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in group:
            return group[name][:]
    raise KeyError(f"none of {names} found; available keys: {list(group.keys())}")


def save_episodes(zarr_path: Path, root: Path, val_fraction: float, seed: int, frame_skip: int) -> None:
    try:
        import zarr
    except ImportError as error:
        raise ImportError("prepare_pusht_zarr.py requires zarr; install with pip install zarr") from error
    dataset = zarr.open(str(zarr_path), mode="r")
    data = dataset["data"]
    meta = dataset["meta"]
    actions = array_from_group(data, ("action",)).astype(np.float32)
    states = array_from_group(data, ("state",)).astype(np.float32)
    frames = array_from_group(data, ("img", "image", "images", "camera_0")).astype(np.uint8)
    episode_ends = array_from_group(meta, ("episode_ends",)).astype(np.int64)
    if states.ndim != 2 or states.shape[1] < 5:
        raise ValueError(f"zarr states must be [N,>=5], got {states.shape}")
    if frames.ndim != 4:
        raise ValueError(f"zarr image array must be [N,H,W,C] or [N,C,H,W], got {frames.shape}")
    if frames.shape[1] in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if len(actions) != len(states) or len(frames) != len(states):
        raise ValueError("zarr action/state/frame arrays must have the same first dimension")

    train_dir, val_dir = root / "data/train", root / "data/val"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    episodes = np.arange(len(episode_ends))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, round(len(episodes) * val_fraction))
    val_episodes = set(episodes[:val_count].tolist())

    start = 0
    for episode, end in enumerate(episode_ends):
        out_dir = val_dir if episode in val_episodes else train_dir
        indices = np.arange(start, int(end), frame_skip)
        if len(indices) < 2:
            start = int(end)
            continue
        # LeWorld action_block alignment:
        # frames[k] -> frames[k+1] spans frame_skip raw env timesteps, supervised by
        # the full consecutive action block actions[indices[k]:indices[k+1]].
        action_blocks = np.stack([actions[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)])
        if action_blocks.shape[1] != frame_skip:
            raise ValueError(f"expected action blocks of length {frame_skip}, got {action_blocks.shape}")
        np.savez_compressed(
            out_dir / f"episode_{episode:06d}.npz",
            frames=frames[indices],
            actions=action_blocks.astype(np.float32),
            states=states[indices, :5],
        )
        start = int(end)
    print(f"Converted {len(episode_ends)} episodes from {zarr_path}")
    print(f"Train episodes: {len(episode_ends) - val_count}; validation episodes: {val_count}")


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between 0 and 1")
    root = Path(args.output_root)
    zarr_path = ensure_zarr(args)
    save_episodes(zarr_path, root, args.val_fraction, args.seed, args.frame_skip)
    write_colab_config(args, root)
    print(f"Ready. Train with: python train.py --config {args.config_output}")


if __name__ == "__main__":
    main()
