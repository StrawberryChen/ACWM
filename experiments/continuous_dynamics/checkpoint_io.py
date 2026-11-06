"""Checkpoint publication and readback on the filesystem visible to training."""
import os
from pathlib import Path
import tempfile

import torch


def read_checkpoint(path):
    # ACWM checkpoints include RNG states; load only trusted experiment files.
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_checkpoint(path, expected_epoch=None):
    saved = read_checkpoint(path)
    if not isinstance(saved, dict) or not {"epoch", "model", "config", "optimizer"} <= saved.keys():
        raise ValueError(f"Not a resumable ACWM continuous checkpoint: {path}")
    if type(saved["epoch"]) is not int or saved["epoch"] < 1:
        raise ValueError(f"Invalid checkpoint epoch: {path}")
    if expected_epoch is not None and saved["epoch"] != expected_epoch:
        raise ValueError(f"Checkpoint {path}: expected epoch {expected_epoch}, got {saved['epoch']}")
    return saved


def save_checkpoint(path, checkpoint, *, verify=False):
    """Write a sibling temp file so a failed write does not truncate latest.pt."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if verify:
            verify_checkpoint(temporary, checkpoint["epoch"])
        os.replace(temporary, path)
        if verify:
            verify_checkpoint(path, checkpoint["epoch"])
    finally:
        Path(temporary).unlink(missing_ok=True)


def require_mounted_drive(path):
    """Do not silently write into local /content/drive before Drive is mounted."""
    mount = Path("/content/drive")
    target = Path(path).resolve()
    if mount in target.parents and not mount.is_mount():
        raise RuntimeError("Mount Google Drive at /content/drive before training; refusing a local lookalike path.")
