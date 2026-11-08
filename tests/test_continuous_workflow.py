from pathlib import Path
import contextlib
import io
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from experiments.continuous_dynamics import workflow
from experiments.continuous_dynamics.checkpoint_io import save_checkpoint, verify_checkpoint


class WorkflowTests(TestCase):
    def test_obs5_planning_defaults_to_flow_jepa_without_legacy_budget_overrides(self):
        with TemporaryDirectory() as root:
            flow = workflow.Workflow("unused.yaml", dict(
                variant="velocity_obs5_bn", run_name="test-run", runs_root=root,
                dataset="leworld_hdf5", planning={"split": "test", "episodes": 10, "max_steps": 100},
            ))
            fake_pymunk = SimpleNamespace(Space=SimpleNamespace(on_collision=True))
            with (patch.dict(sys.modules, {"pymunk": fake_pymunk}),
                  patch.object(flow, "_completed"),
                  patch.object(flow, "_command") as command):
                flow.plan()
                self.assertEqual(command.call_args.args[-4:], ("--split", "test", "--preset", "flow_jepa"))
                flow.plan(execute_steps=5, episodes=10)
                self.assertEqual(command.call_args.args[-4:], ("--episodes", "10", "--execute-steps", "5"))
                flow.plan(preset="legacy")
                self.assertEqual(command.call_args.args[-6:],
                                 ("--preset", "legacy", "--episodes", "10", "--max-steps", "100"))

    def test_streamed_subprocess_sends_stdout_stderr_and_exit_code_to_parent(self):
        with TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()) as output:
            code = workflow._run_streamed(
                [sys.executable, "-u", "-c", "import sys; print('batch 1'); print('batch 2', file=sys.stderr); sys.exit(4)"],
                root)
        self.assertEqual(code, 4)
        self.assertEqual(output.getvalue().splitlines(), ["batch 1", "batch 2"])

    def test_failed_checkpoint_does_not_replace_existing_file(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "latest.pt"
            save_checkpoint(path, dict(epoch=1, model={}, optimizer={}, config={}), verify=True)
            with patch("torch.save", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    save_checkpoint(path, dict(epoch=2))
            self.assertEqual(verify_checkpoint(path)["epoch"], 1)
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_workflow_requires_drive_confirmation_and_uses_same_checkpoint_for_evaluation(self):
        with TemporaryDirectory() as root:
            flow = workflow.Workflow("unused.yaml", dict(
                variant="action_per_step", run_name="test-run", runs_root=root,
                overrides={"training": {"epochs": 2}}, planning={"episodes": 3},
            ))
            flow.checkpoint.parent.mkdir(parents=True)
            saved = dict(epoch=1, config=flow.config, optimizer={}, model={"weight": torch.ones(1)})
            save_checkpoint(flow.checkpoint, saved, verify=True)
            with self.assertRaisesRegex(RuntimeError, "Open Drive"):
                flow.train_remaining()
            with self.assertRaisesRegex(RuntimeError, "final epoch"):
                flow.probe()
            with patch.object(flow, "_command") as command:
                command.side_effect = lambda *_: save_checkpoint(flow.checkpoint, {**saved, "epoch": 2})
                flow.train_remaining(drive_confirmed=True)
                self.assertEqual(command.call_args.args[-2:], ("--checkpoint", str(flow.checkpoint)))
                command.side_effect = None
                flow.probe()
                self.assertEqual(command.call_args.args, ("probe", "--checkpoint", str(flow.checkpoint), "--no-wandb"))
