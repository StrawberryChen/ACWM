import hashlib

import numpy as np
import pytest

from experiments.continuous_dynamics import leworld_data as official
from experiments.continuous_dynamics.data import loader


@pytest.fixture
def fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    path = tmp_path / "source.h5"
    lengths = np.arange(4, 14)
    n = lengths.sum()
    rgb = np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((n, 4, 4, 3), np.uint8)
    offsets = lengths.cumsum() - lengths
    actions = np.repeat(np.arange(n, dtype=np.float32)[:, None], 2, axis=1) / 100
    actions[offsets + lengths - 1] = np.nan  # No transition after terminal row.
    with h5py.File(path, "w") as file:
        for key, value in dict(ep_len=lengths, ep_offset=offsets, pixels=rgb, action=actions,
                               state=np.repeat(np.arange(n)[:, None], 7, axis=1)).items():
            file[key] = value
    config = dict(seed=9, device="cpu", data=dict(dataset="leworld_hdf5", h5_path=str(path),
                  action_scale=512., action_mode="relative", batch_size=2, num_workers=0),
                  model=dict(image_size=4, horizon=3))
    return path, config, offsets, lengths


def test_hdf5_window_alignment_split_and_planning(fixture):
    _, config, offsets, lengths = fixture
    splits = official.split_episodes(lengths)
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(lengths)))
    batch, dataset = loader(config, "train")
    assert next(iter(batch))["frames"].shape == (2, 4, 3, 4, 4)
    assert len(dataset) == sum(lengths[splits["train"]] - 3)
    for index in range(len(dataset)):
        item = dataset[index]
        ep, t = item["episode"], item["timestep"]
        start = offsets[ep] + t
        assert t + 3 < lengths[ep]
        assert set(item) == {"frames", "actions", "episode", "timestep"}
        np.testing.assert_allclose(item["frames"][:, 0, 0, 0] * 255, np.arange(start, start + 4))
        np.testing.assert_allclose(item["actions"][:, 0] * 512, np.arange(start, start + 3), rtol=1e-6)
    cases = official.planning_cases(config, "test", 2, 3, 42)
    assert all(c["source_episode"] in splits["test"] for c in cases)
    assert all(c["goal_state"][0] - c["start_state"][0] == 3 for c in cases)
    assert [c["current"] for c in cases] == [c["current"] for c in official.planning_cases(config, "test", 2, 3, 42)]


def test_archive_checks_and_reuse(fixture, tmp_path, monkeypatch):
    zstd = pytest.importorskip("zstandard")
    path, _, _, lengths = fixture
    data = path.read_bytes()
    packed = zstd.ZstdCompressor().compress(data)
    archive, target = tmp_path / "official.zst", tmp_path / "runtime/data.h5"
    archive.write_bytes(packed)
    monkeypatch.setattr(official, "ARCHIVE_BYTES", len(packed))
    monkeypatch.setattr(official, "H5_BYTES", len(data))
    monkeypatch.setattr(official, "EPISODE_COUNT", len(lengths))
    with pytest.raises(ValueError, match="SHA-256"):
        official.prepare_colab_leworld(archive, target)
    assert not list(target.parent.iterdir())
    monkeypatch.setattr(official, "ARCHIVE_SHA256", hashlib.sha256(packed).hexdigest())
    assert official.prepare_colab_leworld(archive, target).read_bytes() == data
    archive.unlink()
    assert official.prepare_colab_leworld(archive, target) == target


def test_observation_stride_keeps_all_raw_actions_and_probe_frames(fixture):
    _, config, offsets, _ = fixture
    config["model"].update(horizon=6, action_block=3)
    _, dataset = loader(config, "train")
    item = dataset[0]
    start = offsets[item["episode"]] + item["timestep"]
    np.testing.assert_allclose(item["frames"][:, 0, 0, 0] * 255, [start, start + 3, start + 6])
    np.testing.assert_allclose(item["actions"][:, 0] * 512, np.arange(start, start + 6), rtol=1e-6)
    _, probe = loader(config, "train", horizon=1)
    assert probe[0]["frames"].shape[0] == 2
