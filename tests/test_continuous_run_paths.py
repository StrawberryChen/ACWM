import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from experiments.continuous_dynamics.colab import evaluate_run, find_checkpoints
from experiments.continuous_dynamics.run_paths import checkpoint_path


class RunPathsTests(unittest.TestCase):
    def test_repository_yaml_selects_artifacts_but_is_not_used_for_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "actual experiment"
            template = root / "training.yaml"
            template.write_text(yaml.safe_dump({"output_dir": str(run)}))
            with self.assertRaisesRegex(FileNotFoundError, "Checkpoint not found"):
                checkpoint_path(training_config=template)
            latest = run / "checkpoints" / "latest.pt"
            latest.parent.mkdir(parents=True)
            latest.touch()
            self.assertEqual(checkpoint_path(training_config=template), str(latest))
            best = latest.with_name("best_loss.pt")
            best.touch()
            self.assertEqual(checkpoint_path(run=run, checkpoint=best), str(best))
            self.assertEqual(find_checkpoints(root), [latest])
            with (patch("builtins.input", return_value="0"),
                  patch("experiments.continuous_dynamics.colab.subprocess.run") as run_cmd):
                run_cmd.return_value.returncode = 0
                evaluate_run("probe", "planning", roots=(root,))
            self.assertEqual(run_cmd.call_count, 2)
            for call in run_cmd.call_args_list:
                self.assertIn(str(latest), call.args[0])
                self.assertNotIn("--config", call.args[0])
