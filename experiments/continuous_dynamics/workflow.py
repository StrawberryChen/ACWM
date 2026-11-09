"""One YAML-driven workflow for the continuous trajectory experiments."""
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .checkpoint_io import require_mounted_drive, verify_checkpoint
from .data import load_config, validate_splits
from .leworld_data import H5_PATH, split_counts, uses_leworld
from experiments.avs_target.data import merge


DEFAULT_WORKFLOW = "configs/colab_trajectory.yaml"
VELOCITY_WORKFLOW = "configs/colab_velocity_sigreg.yaml"
OBS_VELOCITY_WORKFLOW = "configs/colab_velocity_obs5_bn.yaml"
REPO = Path(__file__).resolve().parents[2]
VARIANTS = {
    "action_per_step": "continuous_trajectory_action_per_step.yaml",
    "state_action_per_step": "continuous_trajectory_state_action_per_step.yaml",
    "action_global": "continuous_trajectory_action.yaml",
    "state_action_global": "continuous_trajectory_state_action.yaml",
    "velocity_sigreg": "continuous_velocity_sigreg.yaml",
    "velocity_obs5_bn": "continuous_velocity_obs5_bn.yaml",
}


def _run_streamed(command, cwd):
    # Child OS file descriptors are not necessarily routed to the notebook cell.
    # Read both streams and print in the kernel process so Colab receives them.
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace", bufsize=1, start_new_session=True) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            return process.wait()
        except KeyboardInterrupt:
            for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, sig)
                    process.wait(timeout=timeout)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise


def configure_colab(variant, run_name, *, overrides=None, planning=None):
    """Persist notebook parameters next to the run; notebook needs no hand-written paths."""
    template = REPO / {"velocity_sigreg": VELOCITY_WORKFLOW,
                       "velocity_obs5_bn": OBS_VELOCITY_WORKFLOW}.get(variant, DEFAULT_WORKFLOW)
    settings = load_config(template)
    settings.update(variant=variant, run_name=run_name)
    if overrides:
        settings["overrides"] = merge(settings["overrides"], overrides)
    if planning:
        settings["planning"] = merge(settings["planning"], planning)
    candidate = Workflow(template, settings)
    target = candidate.run / "workflow.yaml"
    require_mounted_drive(target)
    if target.is_file() and load_config(target) == settings:
        return Workflow.from_yaml(target)
    if candidate.checkpoint.exists():
        candidate.verify()
    import yaml
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return Workflow.from_yaml(target)


def _update_known(settings, overrides, prefix=""):
    if not isinstance(overrides, dict):
        raise ValueError(f"{prefix or 'overrides'} must be a mapping")
    for key, value in overrides.items():
        if key not in settings:
            raise ValueError(f"Unknown setting: {prefix}{key}")
        if isinstance(settings[key], dict):
            _update_known(settings[key], value, f"{prefix}{key}.")
        else:
            settings[key] = value


class Workflow:
    @classmethod
    def from_yaml(cls, path=DEFAULT_WORKFLOW):
        return cls(path, load_config(path))

    def __init__(self, path, settings):
        self.path = Path(path).resolve()
        unknown = set(settings) - {"variant", "run_name", "runs_root", "overrides", "planning", "dataset"}
        if unknown:
            raise ValueError(f"Unknown workflow settings: {sorted(unknown)}")
        variant, name = settings["variant"], settings["run_name"]
        if variant not in VARIANTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Choose variant from {tuple(VARIANTS)} and a simple run_name")
        root = Path(settings["runs_root"])
        if not root.is_absolute():
            raise ValueError("runs_root must be an absolute path")
        self.run = root / variant / name
        self.checkpoint = self.run / "checkpoints" / "latest.pt"
        self.config = load_config(REPO / "configs" / VARIANTS[variant])
        dataset = settings.get("dataset", "legacy_npz")
        if dataset == "leworld_hdf5":
            self.config["data"].update(dataset=dataset, h5_path=H5_PATH)
            for split in ("train", "val", "test"):
                self.config["data"].pop(f"{split}_paths", None)
        elif dataset != "legacy_npz":
            raise ValueError(f"Unknown dataset: {dataset}")
        overrides = settings.get("overrides", {})
        if set(overrides) - {"seed", "device", "data", "model", "loss", "training", "validation", "probe", "wandb"}:
            raise ValueError("Use workflow variant/runs_root/run_name to select model and output")
        if {"direction_mode", "direction_scope"} & overrides.get("model", {}).keys():
            raise ValueError("Select direction mode and scope with workflow variant")
        _update_known(self.config, overrides)
        self.config["output_dir"] = str(self.run)
        self.config["validation"].setdefault("interval", 1)
        self.config["wandb"]["name"] = name
        if self.config["wandb"].get("mode", "online") == "online":
            self.config["wandb"]["id"] = hashlib.sha256(str(self.run).encode()).hexdigest()[:16]
        self.planning = dict(split="val", episodes=10, max_steps=100)
        _update_known(self.planning, settings.get("planning", {}), "planning.")
        if self.planning["split"] not in ("val", "test") or any(
            type(self.planning[k]) is not int or self.planning[k] < 1 for k in ("episodes", "max_steps")
        ):
            raise ValueError("Planning needs val/test and positive integer episodes/max_steps")

    def preflight(self):
        import torch
        require_mounted_drive(self.run)
        if self.config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab before training.")
        if uses_leworld(self.config):
            counts = split_counts(self.config)
            print(f"Official LeWorld (episodes, windows) by split: {counts}", flush=True)
            batch = self.config["data"]["batch_size"]
            if batch < 1:
                raise ValueError("batch_size must be positive")
            print(f"Optimizer steps per full epoch: {(counts['train'][1] + batch - 1) // batch:,}", flush=True)
        else:
            paths = validate_splits(self.config)
            print(f"Legacy dataset episodes: { {key: len(value) for key, value in paths.items()} }")
        print(f"Run directory: {self.run}\nCheckpoint: {self.checkpoint}")
        if self.checkpoint.exists():
            self.verify()

    def verify(self):
        require_mounted_drive(self.run)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {self.checkpoint}. Run the first-epoch cell first.")
        saved = verify_checkpoint(self.checkpoint)
        if saved["config"] != self.config:
            raise ValueError("This checkpoint belongs to different settings. Restore the original YAML or choose a new run_name.")
        epoch = saved["epoch"]
        print(f"Checkpoint readable: epoch {epoch}/{self.config['training']['epochs']}, "
              f"{self.checkpoint.stat().st_size:,} bytes\n{self.checkpoint}", flush=True)
        return epoch

    def _command(self, module, *flags):
        print(f"Starting {module}; live subprocess output follows…", flush=True)
        returncode = _run_streamed([sys.executable, "-u", "-m",
                                    f"experiments.continuous_dynamics.{module}", *flags], REPO)
        if returncode:
            raise RuntimeError(f"{module} exited with code {returncode}; see the original error above.")

    def first_epoch(self):
        self.preflight()
        if self.checkpoint.exists():
            return
        self._command("train", "--workflow", str(self.path), "--stop-after-epoch", "1")
        self.verify()

    def train_remaining(self, *, drive_confirmed=False):
        epoch = self.verify()
        if epoch == self.config["training"]["epochs"]:
            print("Training is already complete.")
            return
        if drive_confirmed is not True:
            raise RuntimeError("Open Drive in a browser and confirm latest.pt is present, then enable drive_confirmed.")
        self._command("train", "--workflow", str(self.path), "--resume", "--checkpoint", str(self.checkpoint))
        self.verify()

    def _completed(self):
        if self.verify() != self.config["training"]["epochs"]:
            raise RuntimeError("Training has not reached the configured final epoch.")

    def probe(self):
        self._completed()
        self._command("probe", "--checkpoint", str(self.checkpoint), "--no-wandb")
        print(f"Probe metrics: {self.run/'probes/latest/metrics.json'}")

    def plan(self, *, preset=None, execute_steps=None, episodes=None, max_steps=None, euler_steps=None):
        if euler_steps is not None and (type(euler_steps) is not int or euler_steps < 1):
            raise ValueError("euler_steps must be a positive integer")
        self._completed()
        if preset is None:
            model = self.config["model"]
            aligned = (uses_leworld(self.config) and self.config["experiment"] == "continuous_velocity"
                       and model.get("action_block") == 5 and model.get("horizon") == 25)
            preset = "flow_jepa" if aligned else "legacy"
        if preset not in {"flow_jepa", "legacy"}:
            raise ValueError("Planning preset must be flow_jepa or legacy.")
        try:
            import pymunk
        except ImportError as error:
            raise RuntimeError("Install .[colab-trajectory] before running planning.") from error
        if not hasattr(pymunk.Space, "on_collision"):
            raise RuntimeError("Planning requires Pymunk 7; reinstall .[colab-trajectory] and restart the runtime if needed.")
        flags = ["--checkpoint", str(self.checkpoint), "--split", self.planning["split"], "--preset", preset]
        for key, value in (("episodes", episodes), ("max_steps", max_steps), ("execute_steps", execute_steps)):
            if value is None and preset == "legacy":
                value = self.planning.get(key)
            if value is not None:
                flags.extend(("--" + key.replace("_", "-"), str(value)))
        if euler_steps is not None:
            flags.extend(("--euler-steps", str(euler_steps)))
        self._command("planning", *flags)
        print(f"Planning metrics and videos: {self.run/'planning'}")
