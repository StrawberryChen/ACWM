"""Prepare independent raw-step train/val/test episodes without touching existing data."""
import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_pusht_zarr import ensure_zarr, array_from_group


def convert_actions(actions, states, scale):
    # Diffusion Policy's zarr action is an absolute target XY, not a displacement.
    if scale <= 0:
        raise ValueError("action_scale must be positive")
    return np.clip((actions[:, :2] - states[:, :2]) / scale, -1, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--download-dir", default="/content/ACWM/data_src")
    parser.add_argument("--url", default="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip")
    parser.add_argument("--base-config", default="configs/action_contrastive.yaml")
    parser.add_argument("--config-output", default="configs/colab_action_contrastive.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import zarr
    with open(args.base_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = Path(args.output_root) / "action_contrastive"
    source = zarr.open(str(ensure_zarr(args)), mode="r")
    frames = array_from_group(source["data"], ("img", "image", "images"))
    states = array_from_group(source["data"], ("state",)).astype(np.float32)
    raw_actions = array_from_group(source["data"], ("action",)).astype(np.float32)
    ends = array_from_group(source["meta"], ("episode_ends",)).astype(int)
    assert len(frames) == len(states) == len(raw_actions)
    if frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)
    scale = float(config["data"]["action_scale"])
    actions = convert_actions(raw_actions, states, scale)
    order = np.random.default_rng(args.seed).permutation(len(ends))
    nval = max(1, round(len(ends)*config["data"]["val_fraction"]))
    ntest = max(1, round(len(ends)*config["data"]["test_fraction"]))
    if nval + ntest >= len(ends):
        raise ValueError("not enough episodes for three disjoint splits")
    splits = {"val": set(order[:nval]), "test": set(order[nval:nval+ntest]),
              "train": set(order[nval+ntest:])}
    # Reuse only an identical manifest; never leave stale episodes after split changes.
    manifest = {"seed": args.seed, "source_url": args.url, "action_scale": scale,
                "splits": {key: sorted(map(int, ids)) for key, ids in splits.items()}, "frame_skip": 1}
    manifest_path = root / "data/manifest.yaml"
    if manifest_path.exists():
        with manifest_path.open() as stream:
            if yaml.safe_load(stream) != manifest:
                raise ValueError("dataset manifest differs; choose a new --output-root for this experiment")
    start = 0
    for episode, end in enumerate(ends):
        split = next(key for key, ids in splits.items() if episode in ids)
        path = root / f"data/{split}/episode_{episode:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # A[t] supervises image[t] -> image[t+1]; final unpaired action discarded.
            np.savez_compressed(path, frames=frames[start:end], actions=actions[start:end-1],
                                states=states[start:end, :5], frame_skip=1, action_mode="relative",
                                action_scale=scale)
        start = end
    with manifest_path.open("w") as stream:
        yaml.safe_dump(manifest, stream)
    for split in splits:
        config["data"][f"{split}_paths"] = [str(root / f"data/{split}/*.npz")]
    config["device"] = "cuda"
    config["output_dir"] = str(root / "outputs")
    path = Path(args.config_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Raw-step relative actions: (target_xy-agent_xy)/{scale}, clipped [-1,1].")
    print(f"Disjoint episodes: { {key: len(ids) for key, ids in splits.items()} }")
    print(f"Ready: python train_action_contrastive.py --config {path}")


if __name__ == "__main__":
    main()
