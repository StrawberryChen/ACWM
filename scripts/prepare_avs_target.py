"""Prepare raw-step episodes once; windows/AVS maps are never saved as images."""
import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.avs_target.data import load_config
from scripts.prepare_pusht_zarr import ensure_zarr, array_from_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/content/drive/MyDrive/ACWM")
    parser.add_argument("--download-dir", default="/content/ACWM/data_src")
    parser.add_argument("--url", default="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip")
    parser.add_argument("--base-config", default="configs/avs_flow.yaml")
    parser.add_argument("--config-output", default="configs/colab_avs_flow.yaml")
    args = parser.parse_args()
    import zarr
    config = load_config(args.base_config)
    source = zarr.open(str(ensure_zarr(args)), mode="r")
    frames = array_from_group(source["data"], ("img", "image", "images"))
    states = array_from_group(source["data"], ("state",)).astype(np.float32)
    actions = array_from_group(source["data"], ("action",)).astype(np.float32)
    ends = array_from_group(source["meta"], ("episode_ends",)).astype(int)
    assert len(frames) == len(states) == len(actions) and actions.shape[1] == 2
    if frames.shape[1] == 3:
        frames = frames.transpose(0, 2, 3, 1)
    scale = config["data"]["action_scale"]
    actions = (actions-states[:, :2])/scale  # no clipping, no state labels in training
    if not np.isfinite(actions).all() or abs(actions).max() > 1.00001:
        raise ValueError("workspace-normalized actions outside [-1,1]; choose an explicit appropriate action_scale")
    order = np.random.default_rng(config["seed"]).permutation(len(ends))
    count = max(1, round(len(ends)*0.1))
    if 2*count >= len(ends):
        raise ValueError("not enough episodes")
    splits = {"val": set(order[:count]), "test": set(order[count:2*count]), "train": set(order[2*count:])}
    root = Path(args.output_root)/"avs_target"
    manifest = dict(seed=config["seed"], action_scale=scale, clipping=False, source_url=args.url,
                    splits={k: sorted(map(int, v)) for k, v in splits.items()})
    manifest_path = root/"data/manifest.yaml"
    if manifest_path.exists() and yaml.safe_load(manifest_path.read_text()) != manifest:
        raise ValueError("data manifest differs; use a different output-root")
    start = 0
    for episode, end in enumerate(ends):
        split = next(k for k, value in splits.items() if episode in value)
        path = root/f"data/{split}/episode_{episode:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.savez_compressed(path, frames=frames[start:end], actions=actions[start:end-1],
                                states=states[start:end, :5], source_episode=episode,
                                frame_skip=1, action_mode="relative", action_scale=scale)
        start = end
    manifest_path.write_text(yaml.safe_dump(manifest))
    for split in splits:
        config["data"][f"{split}_paths"] = [str(root/f"data/{split}/episode_*.npz")]
    config["device"] = "cuda"
    config["output_dir"] = str(root/"outputs"/(config.get("experiment") or ("leworld_style" if "baseline" in config else config["target"]["kind"])))
    Path(args.config_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config_output).write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Prepared episode-disjoint data; actions=(target-agent)/{scale}, no clipping.")
    entrypoint = "joint_train" if config.get("experiment") in {"terminal_only", "avs_terminal"} else ("baseline" if "baseline" in config else "train")
    print(f"python -m experiments.avs_target.{entrypoint} --config {args.config_output}")


if __name__ == "__main__":
    main()
