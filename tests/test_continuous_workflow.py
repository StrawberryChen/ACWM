from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from experiments.continuous_dynamics import workflow
from experiments.continuous_dynamics.checkpoint_io import save_checkpoint, verify_checkpoint


class WorkflowTests(TestCase):
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
