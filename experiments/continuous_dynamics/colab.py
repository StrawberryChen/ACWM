"""Select a saved run in a notebook without typing or guessing its output directory."""
from pathlib import Path
import subprocess
import sys


DEFAULT_ROOTS = ("/content/drive/MyDrive/ACWM", "/content/ACWM")


def find_checkpoints(*roots):
    """Find latest checkpoints, with best_loss as fallback, without loading weights."""
    runs = {}
    for root in roots or DEFAULT_ROOTS:
        root = Path(root)
        print(f"Searching {root}..." if root.is_dir() else f"Unavailable: {root}", flush=True)
        for path in root.rglob("*.pt"):
            if path.parent.name != "checkpoints" or path.name not in ("latest.pt", "best_loss.pt"):
                continue
            if path.is_file() and (path.parent not in runs or path.name == "latest.pt"):
                runs[path.parent] = path
    return sorted(runs.values())


def select_checkpoint(*roots):
    checkpoints = find_checkpoints(*roots)
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints/latest.pt or checkpoints/best_loss.pt in the searched folders. "
            "Confirm the Drive account/mount and check the training log for the saved path."
        )
    for i, checkpoint in enumerate(checkpoints):
        print(f"[{i}] {checkpoint}")
    index = int(input("输入本次实验的编号："))
    if not 0 <= index < len(checkpoints):
        raise ValueError("experiment number out of range")
    return str(checkpoints[index])


def evaluate_run(*tasks, roots=DEFAULT_ROOTS, episodes=10, max_steps=100):
    """Run probe and/or planning on one explicitly selected, trusted checkpoint."""
    if not tasks or any(task not in ("probe", "planning") for task in tasks):
        raise ValueError("choose 'probe', 'planning', or both")
    checkpoint = select_checkpoint(*roots)
    for task in tasks:
        flags = ["--no-wandb"] if task == "probe" else [
            "--split", "val", "--episodes", str(episodes), "--max-steps", str(max_steps)
        ]
        result = subprocess.run([
            sys.executable, "-u", "-m", f"experiments.continuous_dynamics.{task}",
            "--checkpoint", checkpoint, *flags,
        ])
        if result.returncode:
            raise SystemExit(f"{task} failed (exit {result.returncode}); see the error printed above.")
