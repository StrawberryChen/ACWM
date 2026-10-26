"""Compare validation diagnostics without training, planning, or test selection."""
import argparse
import json
from pathlib import Path


def read_experiment(path):
    path = Path(path)
    report = json.loads((path/"evaluation_val/report.json").read_text())
    probes = json.loads((path/"probes/metrics.json").read_text())
    metadata = json.loads((path/"probes/metadata.json").read_text())
    if report["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("prediction and probe reports refer to different checkpoints")
    return {**report["metrics"], **{key: value for key, value in probes.items() if "/val/" in key}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-dir", required=True)
    parser.add_argument("--joint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    terminal, joint = read_experiment(args.terminal_dir), read_experiment(args.joint_dir)
    shared = sorted(set(terminal) & set(joint))
    values = {key: {"terminal_only": terminal[key], "avs_terminal": joint[key],
                    "joint_minus_terminal": joint[key]-terminal[key]}
              for key in shared if isinstance(terminal[key], (int, float)) and isinstance(joint[key], (int, float))}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2))
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
