import os
from pathlib import Path
import shutil

import pytest

from experiments.avs_target.data import check_source_provenance, source_signature


def test_content_hash_accepts_timestamp_change_and_relocation(tmp_path):
    original = tmp_path/"episode.npz"
    original.write_bytes(b"same trajectory content")
    before = source_signature([original])
    os.utime(original, ns=(before[0]["mtime_ns"]+1_000_000, before[0]["mtime_ns"]+1_000_000))
    after = source_signature([original])
    assert before != after
    check_source_provenance(before, after, "checkpoint", training=True)
    moved = tmp_path/"new_root/episode.npz"
    moved.parent.mkdir()
    shutil.copyfile(original, moved)
    check_source_provenance(before, source_signature([moved]), "checkpoint", training=True)


def test_content_change_same_size_is_rejected_for_training_only(tmp_path):
    path = tmp_path/"episode.npz"
    path.write_bytes(b"original")
    before = source_signature([path])
    path.write_bytes(b"modified")
    after = source_signature([path])
    assert before[0]["size"] == after[0]["size"]
    with pytest.raises(ValueError, match="training data content"):
        check_source_provenance(before, after, "checkpoint", training=True)
    with pytest.warns(UserWarning, match="does not retrain"):
        check_source_provenance(before, after, "checkpoint", training=False)


def test_legacy_without_hash_ignores_mtime_but_keeps_path_size_guard(tmp_path):
    path = tmp_path/"episode.npz"
    path.write_bytes(b"trajectory")
    legacy = source_signature([path], content_hash=False)
    os.utime(path, ns=(legacy[0]["mtime_ns"]+1_000_000, legacy[0]["mtime_ns"]+1_000_000))
    current = source_signature([path])
    with pytest.warns(UserWarning, match="legacy provenance"):
        check_source_provenance(legacy, current, "legacy checkpoint", training=True)
    check_source_provenance(legacy, source_signature([path], content_hash=False), "legacy eval", training=False)
    path.write_bytes(b"different trajectory with new size")
    with pytest.raises(ValueError, match="provenance differs"):
        check_source_provenance(legacy, source_signature([path]), "legacy checkpoint", training=True)
