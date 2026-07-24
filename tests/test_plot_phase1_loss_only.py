from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_phase1_loss_only import main


class PlotPhase1LossOnlyTest(unittest.TestCase):
    def test_writes_loss_curves_from_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "metrics.csv").write_text(
                "episode,loss\n1,3.0\n2,2.0\n3,1.0\n",
                encoding="utf-8",
            )

            main([str(output_dir), "--window", "2"])

            self.assertTrue((output_dir / "loss_curve.png").exists())
            self.assertTrue((output_dir / "loss_curve_normalized.png").exists())
            status = json.loads((output_dir / "loss_only_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["metrics_rows"], 3)
            self.assertEqual(status["latest_episode"], 3)

    def test_missing_loss_column_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "metrics.csv").write_text("episode\n1\n", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                main([str(output_dir)])


if __name__ == "__main__":
    unittest.main()
