"""Locate continuous-model checkpoints; saved YAML is not required for evaluation."""
from pathlib import Path

from .data import resolve


def checkpoint_path(run=None, training_config=None, checkpoint=None):
    if run is not None and training_config is not None:
        raise ValueError("select at most one of --run or --training-config")
    if training_config is not None:
        run = resolve(training_config, None, None)["output_dir"]
    if checkpoint is None:
        if run is None:
            raise ValueError("use --checkpoint, --run or --training-config")
        checkpoint = Path(run) / "checkpoints" / "latest.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "In Colab, mount Drive and use experiments.continuous_dynamics.colab.find_checkpoints "
            "to locate the actual checkpoint."
        )
    return str(checkpoint)
