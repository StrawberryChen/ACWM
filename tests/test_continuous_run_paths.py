import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.continuous_dynamics.run_paths import run_paths


class RunPathsTests(unittest.TestCase):
    def test_repository_yaml_selects_artifacts_but_is_not_used_for_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "actual experiment"
            template = root / "training.yaml"
            template.write_text(yaml.safe_dump({"output_dir": str(run)}))
            with self.assertRaisesRegex(FileNotFoundError, "Drive is mounted"):
                run_paths(training_config=template)
            config = run / "config.yaml"
            latest = run / "checkpoints" / "latest.pt"
            latest.parent.mkdir(parents=True)
            config.touch()
            latest.touch()
            self.assertEqual(run_paths(training_config=template), (str(config), str(latest)))
            best = latest.with_name("best_loss.pt")
            best.touch()
            self.assertEqual(run_paths(run=run, checkpoint=best), (str(config), str(best)))
