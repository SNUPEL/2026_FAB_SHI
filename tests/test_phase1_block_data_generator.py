"""Phase 1 block-only synthetic data generation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Utils.phase1.phase1_block_data_generator import (
    PHASE1_BLOCK_FEATURE_COLUMNS,
    generate_phase1_block_data,
    validate_phase1_block_data,
    write_phase1_block_generation_package,
)


class Phase1BlockDataGeneratorTest(unittest.TestCase):
    """Phase 1 training data must not depend on W/O rows."""

    def test_generates_block_only_rows_without_work_order_column(self) -> None:
        actual = self._actual_blocks().drop(columns=["WK_ORD_NO"])

        generated = generate_phase1_block_data(actual, n_blocks=20, seed=11)

        self.assertEqual(len(generated), 20)
        self.assertTrue(set(PHASE1_BLOCK_FEATURE_COLUMNS).issubset(generated.columns))
        self.assertFalse(generated[list(PHASE1_BLOCK_FEATURE_COLUMNS)].isna().any().any())
        self.assertEqual(set(generated["DATA_ROLE"]), {"synthetic_phase1_block_only"})
        self.assertTrue(set(generated["THK"]).issubset(set(actual["THK"])))
        self.assertGreaterEqual(generated["STL_QTY"].min(), 1)
        self.assertGreaterEqual(generated["BV_QTY"].min(), 0)

    def test_requires_bevel_quantity_without_fallback(self) -> None:
        actual = self._actual_blocks().drop(columns=["BV_QTY"])

        with self.assertRaisesRegex(RuntimeError, "missing_required_columns.*BV_QTY"):
            validate_phase1_block_data(actual, source_name="unit-test")

    def test_writes_generation_package_with_correlation_report(self) -> None:
        actual = self._actual_blocks().drop(columns=["WK_ORD_NO"])

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_phase1_block_generation_package(
                actual_blocks=actual,
                output_dir=tmpdir,
                n_blocks=16,
                seed=2026,
            )

            summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
            generated = pd.read_csv(paths["synthetic_csv"])
            distribution = pd.read_csv(paths["distribution_summary_csv"])

        self.assertEqual(summary["generated_row_count"], 16)
        self.assertEqual(summary["source_row_count"], len(actual))
        self.assertIn("correlation_mae", summary)
        self.assertGreaterEqual(summary["correlation_mae"], 0.0)
        self.assertEqual(len(generated), 16)
        self.assertIn("generated_mean", distribution.columns)

    @staticmethod
    def _actual_blocks() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "PROJ_NO": ["P1"] * 10,
                "BLK_NO": [f"B{i}" for i in range(10)],
                "WK_ORD_NO": [f"WO{i}" for i in range(10)],
                "GYEL": ["NP"] * 10,
                "LTH": [8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000],
                "THK": [10, 11, 12, 13, 13, 14, 15, 16, 18, 20],
                "CUT_LTH": [120, 180, 260, 330, 450, 620, 850, 1100, 1400, 1800],
                "MARK_LTH": [60, 90, 130, 170, 230, 310, 420, 540, 710, 900],
                "STL_QTY": [2, 3, 4, 5, 7, 9, 12, 15, 19, 24],
                "BV_QTY": [0, 1, 1, 2, 3, 5, 8, 12, 18, 26],
            }
        )


if __name__ == "__main__":
    unittest.main()
