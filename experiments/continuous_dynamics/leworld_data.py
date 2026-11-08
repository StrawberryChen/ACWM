"""Raw-step windows over the official LeWorld Push-T HDF5 release."""
from collections import OrderedDict, deque
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler


ARCHIVE = "/content/drive/MyDrive/ACWM/pusht_expert_train.h5.zst"
H5_PATH = "/content/acwm_data/pusht_expert_train.h5"
ARCHIVE_BYTES = 13_136_247_974
H5_BYTES = 46_300_921_856
EPISODE_COUNT = 18_685
ARCHIVE_SHA256 = "7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318"
NATIVE_ACTION_SCALE = 100.0
SPLIT_SEED = 42  # Fixed across training seeds, model variants, and runtime restarts.


def _h5(path, **kwargs):
    import h5py
    import hdf5plugin  # noqa: F401 - registers the official file's Blosc filter
    return h5py.File(path, "r", **kwargs)


def _marker(path):
    return path.with_suffix(".verified.json")


def _verified(path):
    try:
        stamp = json.loads(_marker(path).read_text())
        stat = path.stat()
        return stamp == dict(archive_sha256=ARCHIVE_SHA256, size=H5_BYTES,
                             mtime_ns=stat.st_mtime_ns) and stat.st_size == H5_BYTES
    except (OSError, ValueError):
        return False


def prepare_colab_leworld(archive=ARCHIVE, destination=H5_PATH):
    """Verify and unpack the Drive archive once to local runtime storage."""
    import zstandard

    archive, dest = Path(archive), Path(destination)
    if _verified(dest):
        print(f"Reusing verified local LeWorld dataset: {dest}", flush=True)
        return dest
    if not archive.is_file() or archive.stat().st_size != ARCHIVE_BYTES:
        raise FileNotFoundError(f"Expected the complete official archive at {archive}; mount Drive first.")
    if dest.exists():
        raise FileExistsError(f"Unverified file at {dest}; move it aside before unpacking.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(dest.parent).free < H5_BYTES + 1024**3:
        raise OSError(f"Need about 45 GiB free in the Colab runtime to unpack {dest.name}.")
    temporary = dest.with_suffix(".h5.part")
    if temporary.exists():
        raise FileExistsError(f"Partial extraction exists at {temporary}; check no extraction is running.")
    print(f"Unpacking official Push-T dataset to {dest} (43.1 GiB)…", flush=True)
    digest = hashlib.sha256()

    class HashedReader:
        def __init__(self, source):
            self.source = source

        def read(self, size):
            block = self.source.read(size)
            digest.update(block)
            return block

    output = temporary.open("xb")
    try:
        with output as out, archive.open("rb") as src:
            with zstandard.ZstdDecompressor().stream_reader(HashedReader(src)) as stream:
                done, milestone = 0, 0
                while block := stream.read(16 * 1024**2):
                    done += len(block)
                    if done > H5_BYTES:
                        raise ValueError("uncompressed archive exceeds the official file size")
                    out.write(block)
                    if done - milestone >= 2 * 1024**3:
                        print(f"Unpacked {done / H5_BYTES:.0%}", flush=True)
                        milestone = done
            if done != H5_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
                raise ValueError("official archive size or SHA-256 check failed")
        _, lengths = episode_metadata(temporary)
        with _h5(temporary) as file:
            if len(lengths) != EPISODE_COUNT or file["pixels"][0].shape[-1] != 3:
                raise ValueError("unexpected official episode count or unreadable RGB")
        temporary.replace(dest)
        _marker(dest).write_text(json.dumps(dict(archive_sha256=ARCHIVE_SHA256, size=H5_BYTES,
                                                mtime_ns=dest.stat().st_mtime_ns)))
    finally:
        temporary.unlink(missing_ok=True)
    print(f"LeWorld dataset ready: {dest} ({len(lengths):,} episodes)", flush=True)
    return dest


def uses_leworld(config):
    return config["data"].get("dataset") == "leworld_hdf5"


def episode_metadata(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} missing; run Colab step 01 to unpack the official archive.")
    with _h5(path) as file:
        offsets, lengths = (file[key][:].astype(np.int64) for key in ("ep_offset", "ep_len"))
        if (lengths.ndim != 1 or len(lengths) < 3 or offsets.shape != lengths.shape or
                np.any(lengths < 1) or not np.array_equal(offsets, lengths.cumsum() - lengths)):
            raise ValueError("invalid HDF5 episode boundaries")
        n = int(lengths.sum())
        rgb = file["pixels"]
        if (rgb.ndim != 4 or rgb.shape[0] != n or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or
                file["action"].shape != (n, 2) or file["state"].shape != (n, 7)):
            raise ValueError("expected aligned uint8 RGB, 2D relative action, and 7D state columns")
    return offsets, lengths


def split_episodes(lengths):
    order = np.random.default_rng(SPLIT_SEED).permutation(len(lengths))
    holdout = max(1, round(len(order) * .10))
    return {key: np.sort(ids) for key, ids in (
        ("train", order[2 * holdout:]), ("val", order[:holdout]), ("test", order[holdout:2 * holdout]))}


def split_counts(config):
    _, lengths = episode_metadata(config["data"]["h5_path"])
    horizon = config["model"].get("horizon", 1)
    return {key: (len(ids), int(np.maximum(lengths[ids] - horizon, 0).sum()))
            for key, ids in split_episodes(lengths).items()}


def provenance(config):
    from experiments.avs_target.data import file_sha256
    path = Path(config["data"]["h5_path"])
    _, lengths = episode_metadata(path)
    source = ARCHIVE_SHA256 if _verified(path) else file_sha256(path)
    ids = split_episodes(lengths)["train"].astype("<i8")
    fingerprint = hashlib.sha256(source.encode() + ids.tobytes()).hexdigest()
    return [dict(path=str(path), size=path.stat().st_size, sha256=fingerprint)]


def planning_action_statistics(config):
    """Fit a per-axis action proposal on the checkpoint's training split only."""
    path = config["data"]["h5_path"]
    offsets, lengths = episode_metadata(path)
    train = split_episodes(lengths)["train"]
    selected = np.zeros(len(lengths), dtype=bool)
    selected[train] = True
    keep = np.repeat(selected, lengths)
    keep[offsets + lengths - 1] = False  # No outgoing transition on the last frame.
    with _h5(path) as file:
        native = file["action"][:][keep].astype(np.float64)
    native = native[np.isfinite(native).all(axis=1)]
    scale = float(config["data"]["action_scale"])
    if len(native) < 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("need finite training actions and a positive model action scale")
    factor = NATIVE_ACTION_SCALE / scale
    mean, std = native.mean(axis=0) * factor, native.std(axis=0) * factor
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("dataset-standardized CEM needs positive action spread on both axes")
    return dict(mean=mean.tolist(), std=std.tolist(), fitted_actions=len(native), split="train")


class LeWorldWindows(Dataset):
    """A compact prefix-sum index; no pixel I/O at startup or simulator labels in batches."""

    def __init__(self, config, split, horizon, observation_stride=1):
        self.path = config["data"]["h5_path"]
        self.image_size = config["model"]["image_size"]
        self.action_factor = NATIVE_ACTION_SCALE / float(config["data"]["action_scale"])
        self.horizon = int(horizon)
        self.observation_stride = observation_stride
        if type(observation_stride) is not int or observation_stride < 1 or self.horizon % observation_stride:
            raise ValueError("observation_stride must divide the raw horizon")
        if self.horizon < 1 or not np.isfinite(self.action_factor) or self.action_factor <= 0:
            raise ValueError("positive HDF5 horizon and model action scale required")
        self.offsets, self.lengths = episode_metadata(self.path)
        self.episode_ids = split_episodes(self.lengths)[split]
        self.ends = np.maximum(self.lengths[self.episode_ids] - self.horizon, 0).cumsum()
        self._handle = self._pid = None
        self._episode_cache = OrderedDict()
        self.cache_size = 0  # Evaluation/probe read isolated windows directly.
        if not len(self):
            raise ValueError(f"no {split} episodes support horizon {horizon}")

    def __len__(self):
        return int(self.ends[-1])

    def __getstate__(self):
        return dict(self.__dict__, _handle=None, _pid=None, _episode_cache=OrderedDict())

    def _file(self):
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = _h5(self.path, rdcc_nbytes=96 * 1024**2, rdcc_nslots=1009)
            self._pid = pid
            self._episode_cache.clear()
        return self._handle

    def window_index(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = int(np.searchsorted(self.ends, index, side="right"))
        return int(self.episode_ids[row]), int(index - (self.ends[row - 1] if row else 0))

    def prepare_frames(self, frames, episode):
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.)
        return F.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _actions(self, start, stop):
        # Already relative to the agent. Convert from native displacement/100
        # to the existing model units (normally displacement/512); never clip.
        actions = self._file()["action"][start:stop] * self.action_factor
        if not np.isfinite(actions).all():
            raise ValueError("non-finite action inside an episode transition")
        return actions

    def episode(self, episode):
        start = int(self.offsets[episode])
        stop = start + int(self.lengths[episode])
        return self._file()["pixels"][start:stop], self._actions(start, stop - 1)

    def episode_states(self, episode):
        start = int(self.offsets[episode])
        return self._file()["state"][start:start + int(self.lengths[episode])]

    def _cached_episode(self, episode):
        # The sampler reuses a small cohort for several mixed-episode batches.
        # Decode and resize once per episode per worker, not once per window.
        self._file()
        cached = self._episode_cache.get(episode)
        if cached is None:
            raw, actions = self.episode(episode)
            resized = torch.cat([self.prepare_frames(raw[i:i + 32], episode)
                                 for i in range(0, len(raw), 32)])
            cached = resized, torch.from_numpy(actions)
            self._episode_cache[episode] = cached
            if len(self._episode_cache) > self.cache_size:
                self._episode_cache.popitem(last=False)
        else:
            self._episode_cache.move_to_end(episode)
        return cached

    def __getitem__(self, index):
        episode, t = self.window_index(index)
        if self.cache_size:
            frames, actions = self._cached_episode(episode)
            return dict(frames=frames[t:t + self.horizon + 1:self.observation_stride],
                        actions=actions[t:t + self.horizon], episode=episode, timestep=t)
        start = int(self.offsets[episode]) + t
        stop = start + self.horizon
        frames = self._file()["pixels"][start:stop + 1:self.observation_stride]
        return dict(frames=self.prepare_frames(frames, episode), actions=torch.from_numpy(self._actions(start, stop)),
                    episode=episode, timestep=t)


class EpisodeLocalBatchSampler(Sampler):
    """Visit every window once per epoch while reusing a shuffled cohort of episodes.

    Round-robin the starts of different episodes instead of making a batch from
    one trajectory. Replace exhausted episodes so long trajectories do not leave
    many single-trajectory tail batches. No samples are dropped.
    """

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size, self.seed = batch_size, seed
        self.ends = dataset.ends
        self.starts = np.r_[0, self.ends[:-1]]
        self.cohort_size = min(16, max(4, batch_size // 2))
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (int(self.ends[-1]) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        eligible = np.flatnonzero(self.ends > self.starts)
        episodes = iter(rng.permutation(eligible))

        def window_order(row):
            return iter(rng.permutation(int(self.ends[row] - self.starts[row])) + self.starts[row])

        windows = deque(window_order(next(episodes)) for _ in range(min(self.cohort_size, len(eligible))))
        batch = []
        while windows:
            window = windows.popleft()
            index = next(window, None)
            while index is None and (row := next(episodes, None)) is not None:
                window = window_order(row)
                index = next(window, None)
            if index is None:
                continue
            windows.append(window)
            batch.append(int(index))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def planning_cases(config, split, count, goal_offset, seed):
    dataset = LeWorldWindows(config, split, goal_offset)
    if count > len(dataset):
        raise ValueError(f"requested {count} cases but only {len(dataset)} starts available")
    indices = np.random.default_rng(seed).choice(len(dataset), size=count, replace=False)
    cases = []
    for index in indices:
        episode, t = dataset.window_index(int(index))
        rows = dataset.episode_states(episode)
        cases.append(dict(path=dataset.path, source_episode=episode, current=t,
                          start_state=rows[t], goal_state=rows[t + goal_offset]))
    return cases
