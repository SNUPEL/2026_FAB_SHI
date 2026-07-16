from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from Utils.data.multi_series_cutting_data import MultiSeriesCuttingData
from Utils.phase1.multi_series_planner import (
    build_multi_series_phase1_daily_plans,
    write_multi_series_phase1_daily_plans,
)


class MultiSeriesPhase1PlannerTest(unittest.TestCase):
    def test_writer_rejects_missing_problem_rows(self) -> None:
        with TemporaryDirectory() as output_dir:
            with self.assertRaises(RuntimeError):
                write_multi_series_phase1_daily_plans({}, output_dir)
            self.assertEqual(list(Path(output_dir).iterdir()), [])

    def test_builds_one_joint_five_bay_problem_per_day(self) -> None:
        blocks = pd.DataFrame(
            [
                self._block("P1", "NP", "BLK_1", "20260316", cut=1000, width=3000),
                self._block("P2", "FN", "BLK_2", "20260316", cut=100, width=3000),
                self._block("P3", "FL", "BLK_3", "20260316", cut=100, width=3000),
            ]
        )
        work_orders = pd.DataFrame(
            [
                self._wo("WO_NP_1", "P1", "NP", "BLK_1", cut=600, actual_bay="24"),
                self._wo("WO_NP_2", "P1", "NP", "BLK_1", cut=400, actual_bay="24"),
                self._wo("WO_FN_1", "P2", "FN", "BLK_2", cut=100, actual_bay="22"),
                self._wo("WO_FL_1", "P3", "FL", "BLK_3", cut=100, actual_bay="24"),
            ]
        )
        data = MultiSeriesCuttingData(blocks=blocks, work_orders=work_orders)

        result = build_multi_series_phase1_daily_plans(data)

        self.assertEqual(result["problem_count"], 1)
        problem = result["problems"][0]
        self.assertEqual(problem["workday"], "20260311")
        self.assertEqual(problem["balancing_groups"], ["NP", "FN_FL"])
        self.assertEqual(
            problem["bay_capacity_weights"],
            {"22": 4.0, "23": 3.0, "24": 4.0, "25": 2.0, "trans": 2.0},
        )
        self.assertEqual(set(problem["plan"]["bay_loads"]), {"22", "23", "24", "25", "trans"})
        assignments = problem["plan"]["assignments"]
        np_assignment = next(row for row in assignments if row["series"] == "NP")
        self.assertEqual(np_assignment["block_set_id"], "P1::NP::BLK_1")
        self.assertIn(np_assignment["assigned_bay"], {"22", "23"})
        self.assertEqual(np_assignment["candidate_bays"], "22|23")
        fn_fl_families = {row["series"] for row in assignments if row["series"] in {"FN", "FL"}}
        self.assertEqual(fn_fl_families, {"FN", "FL"})

    @staticmethod
    def _block(project: str, series: str, block: str, reference: str, *, cut: float, width: float) -> dict:
        row = {
            "PROJ_NO": project,
            "GYEL": series,
            "BLK_NO": block,
            "BLOCK_SET_ID": f"{project}::{series}::{block}",
            "ACT_ST_DT": 20260311,
            "ASS_YN": "N",
            "SASS_ACT_STDT": reference,
            "ASS_ACT_STDT": reference,
            "PAN_ST_DT": reference,
            "CUT_LTH": cut,
            "BV_QTY": 0,
            "CURVE_QTY": 0,
            "PTLST_QTY": 1,
            "BVL_LTH": 0,
            "BTH": width,
        }
        return row

    @staticmethod
    def _wo(
        work_order: str,
        project: str,
        series: str,
        block: str,
        *,
        cut: float,
        actual_bay: str,
    ) -> dict:
        return {
            "WK_ORD_NO": work_order,
            "PROJ_NO": project,
            "GYEL": series,
            "BLK_NO": block,
            "BLOCK_SET_ID": f"{project}::{series}::{block}",
            "STL_QTY": 0,
            "CUT_LTH": cut,
            "BV_QTY": 0,
            "LTH": 10000,
            "BTH": 3000,
            "THK": 20,
            "CUT_BAY": actual_bay,
        }


if __name__ == "__main__":
    unittest.main()
