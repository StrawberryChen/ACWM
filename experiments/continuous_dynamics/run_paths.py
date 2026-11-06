"""Resolve evaluation artifacts without treating a training template as a saved run."""
from pathlib import Path

from .data import resolve


def run_paths(run=None, training_config=None, checkpoint=None):
    if (run is None) == (training_config is None):
        raise ValueError("select exactly one of --run or --training-config")
    if training_config is not None:
        run = resolve(training_config, None, None)["output_dir"]
    run = Path(run)
    config = run / "config.yaml"
    checkpoint = Path(checkpoint) if checkpoint is not None else run / "checkpoints" / "latest.pt"
    missing = [str(path) for path in (config, checkpoint) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Training artifacts not found: {', '.join(missing)}. "
            "In Colab, check that Drive is mounted. If training used a different "
            "output_dir (including --output-dir), select the actual experiment with --run."
        )
    return str(config), str(checkpoint)
