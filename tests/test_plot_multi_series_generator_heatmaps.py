"""다계열 실적·생성 상관관계 히트맵의 데이터 계약 회귀시험."""

from __future__ import annotations

import unittest
from pathlib import Path

from matplotlib import font_manager
import numpy as np
import pandas as pd

from scripts.plot_multi_series_generator_heatmaps import (
    BLOCK_FEATURES,
    JOINT_FEATURES,
    _correlation_matrix,
    _physical_block_cross_series_metrics,
    _prepare_actual_family_data,
    _resolve_korean_font,
)


def _actual_wo_row(block_no: str, cut_length: float, series: str = "NP") -> dict:
    return {
        "PROJ_NO": "P1",
        "GYEL": series,
        "BLK_NO": block_no,
        "LTH": 1000.0 + cut_length,
        "BTH": 3000.0,
        "THK": 20.0,
        "MARK_LTH": 50.0,
        "CUT_LTH": cut_length,
        "BVL_LTH": 4.0,
        "BV_QTY": 3,
        "PTLST_QTY": 5,
        "STL_QTY": 1,
    }


def _joint_block_metric_row(block_no: str, series: str, base: float) -> dict:
    return {
        "PROJ_NO": "P1",
        "BLK_NO": block_no,
        "GYEL": series,
        "WO_QTY": base,
        "LTH": base,
        "THK": base,
        "BTH": base,
        "CUT_LTH": base,
        "BV_QTY": base,
    }


class MultiSeriesGeneratorHeatmapTest(unittest.TestCase):
    def test_actual_block_derives_wo_count_and_max_tact_time_from_actual_wos(self) -> None:
        actual_wos = pd.DataFrame(
            [
                _actual_wo_row("B1", 100.0),
                _actual_wo_row("B1", 200.0),
                _actual_wo_row("B2", 300.0, series="FN"),
            ]
        )
        actual_blocks = pd.DataFrame(
            [
                {
                    **_actual_wo_row("B1", 300.0),
                    "BVL_LTH": 8.0,
                    "BV_QTY": 6,
                    "PTLST_QTY": 10,
                    "STL_QTY": 2,
                },
                _actual_wo_row("B2", 300.0, series="FN"),
            ]
        )

        prepared_wos, prepared_blocks = _prepare_actual_family_data(
            actual_wos,
            actual_blocks,
            "NP",
        )

        self.assertEqual(len(prepared_wos), 2)
        self.assertEqual(len(prepared_blocks), 1)
        self.assertEqual(int(prepared_blocks.iloc[0]["WO_QTY"]), 2)
        self.assertAlmostEqual(
            float(prepared_blocks.iloc[0]["TACT_TIME"]),
            float(prepared_wos["TACT_TIME"].max()),
        )
        self.assertTrue(set(BLOCK_FEATURES).issubset(prepared_blocks.columns))

    def test_correlation_matrix_rejects_constant_feature(self) -> None:
        frame = pd.DataFrame(
            {
                "LTH": [1.0, 2.0, 3.0],
                "BTH": [10.0, 10.0, 10.0],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "constant_correlation_features"):
            _correlation_matrix(frame, ["LTH", "BTH"], "NP_WO_actual")

    def test_correlation_matrix_is_symmetric_with_unit_diagonal(self) -> None:
        frame = pd.DataFrame(
            {
                "LTH": [1.0, 2.0, 4.0, 8.0],
                "BTH": [8.0, 4.0, 2.0, 1.0],
                "THK": [1.0, 3.0, 2.0, 5.0],
            }
        )

        matrix = _correlation_matrix(frame, ["LTH", "BTH", "THK"], "NP_WO_actual")

        self.assertTrue(np.allclose(matrix.to_numpy(), matrix.to_numpy().T))
        self.assertTrue(np.allclose(np.diag(matrix.to_numpy()), 1.0))

    def test_physical_block_cross_series_metrics_reports_all_relations(self) -> None:
        rows = []
        for block_index, (np_base, fl_base) in enumerate(((1.0, 30.0), (2.0, 20.0), (3.0, 10.0)), start=1):
            rows.append(_joint_block_metric_row(f"B{block_index}", "NP", np_base))
            rows.append(_joint_block_metric_row(f"B{block_index}", "FL", fl_base))
        metrics = _physical_block_cross_series_metrics(pd.DataFrame(rows), min_pairs=3)

        self.assertEqual(len(metrics), 6 * len(JOINT_FEATURES) ** 2)
        np_fl_lth = metrics.loc[
            metrics["series_a"].eq("NP")
            & metrics["series_b"].eq("FL")
            & metrics["feature_a"].eq("LTH")
            & metrics["feature_b"].eq("LTH")
        ].iloc[0]
        self.assertEqual(np_fl_lth["status"], "ok")
        self.assertEqual(int(np_fl_lth["pair_count"]), 3)
        self.assertAlmostEqual(float(np_fl_lth["correlation"]), -1.0)

        fn_nc = metrics.loc[
            metrics["series_a"].eq("FN")
            & metrics["series_b"].eq("NC")
            & metrics["feature_a"].eq("WO_QTY")
            & metrics["feature_b"].eq("WO_QTY")
        ].iloc[0]
        self.assertEqual(fn_nc["status"], "insufficient_pairs")
        self.assertEqual(int(fn_nc["pair_count"]), 0)
        self.assertTrue(pd.isna(fn_nc["correlation"]))

    def test_korean_font_is_registered_for_matplotlib_lookup(self) -> None:
        font_path = Path("/mnt/c/Windows/Fonts/malgun.ttf")
        if not font_path.is_file():
            self.skipTest("Windows Korean font is unavailable in this environment")

        font = _resolve_korean_font(font_path)
        resolved = Path(
            font_manager.findfont(font.get_name(), fallback_to_default=False)
        ).resolve()

        self.assertEqual(resolved, font_path.resolve())


if __name__ == "__main__":
    unittest.main()
