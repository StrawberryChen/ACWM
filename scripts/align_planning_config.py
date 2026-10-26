"""Update an existing Colab YAML without touching data, training, or output paths."""
import argparse
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.leworld_planning import align_raw_step_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    align_raw_step_config(config)
    with path.open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print("Planning updated: 25 raw-action horizon, execute 25 then replan, budget 50, latent MSE sum.")
    print("Encoder, loss, data and Drive output paths unchanged; existing weights remain compatible.")


if __name__ == "__main__":
    main()
