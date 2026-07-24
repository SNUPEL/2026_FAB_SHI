from __future__ import annotations

import unittest
from datetime import date
import json
from pathlib import Path
import tempfile

from Utils.data.cutting_start_date import (
    audit_cutting_start_dates,
    calculate_cutting_start_date,
    korean_public_holiday_dates,
    subtract_working_days,
    summarize_cutting_start_date_audit,
    write_cutting_start_date_audit,
)


class CuttingStartDateTest(unittest.TestCase):
    def test_korean_public_holiday_calendar_excludes_substitute_holiday(self) -> None:
        holidays = korean_public_holiday_dates({2026})

        self.assertIn(date(2026, 3, 2), holidays)
        self.assertEqual(
            subtract_working_days(date(2026, 3, 3), 1, holiday_dates=holidays),
            date(2026, 2, 27),
        )

    def test_np_boundary_conditions_add_both_working_days(self) -> None:
        result = calculate_cutting_start_date(
            {
                "PROJ_NO": "P1",
                "BLK_NO": "B1",
                "GYEL": "NP",
                "ASS_YN": "N",
                "SASS_ACT_STDT": 20260316,
                "ASS_ACT_STDT": 20260320,
                "CUT_LTH": 2000,
                "BV_QTY": 30,
            },
            holiday_dates={date(2026, 3, 12)},
        )

        self.assertEqual(result.lead_working_days, 5)
        self.assertEqual(result.start_date, date(2026, 3, 6))
        self.assertEqual(result.reason_codes, ("np_base_3", "cut_lth_ge_2000", "bv_qty_ge_30"))

    def test_fl_or_condition_and_bevel_condition_are_independent(self) -> None:
        result = calculate_cutting_start_date(
            {
                "PROJ_NO": "P1",
                "BLK_NO": "B1",
                "GYEL": "FL",
                "PAN_ST_DT": 20260316,
                "PTLST_QTY": 13,
                "CUT_LTH": 100,
                "BVL_LTH": 100,
            }
        )

        self.assertEqual(result.lead_working_days, 5)
        self.assertEqual(result.start_date, date(2026, 3, 9))
        self.assertEqual(result.reason_codes, ("fl_base_3", "ptlst_ge_13_or_cut_ge_450", "bvl_lth_ge_100"))

    def test_fn_cnt_and_nc_curve_rules_use_confirmed_reference_dates(self) -> None:
        fn = calculate_cutting_start_date(
            {
                "PROJ_NO": "P1",
                "BLK_NO": "CNT_BLK_1",
                "GYEL": "FN",
                "ASS_ACT_STDT": 20260320,
                "BV_QTY": 0,
                "CURVE_QTY": 0,
            }
        )
        nc = calculate_cutting_start_date(
            {
                "PROJ_NO": "P2",
                "BLK_NO": "CURVE_BLK_1",
                "GYEL": "NC",
                "ASS_ACT_STDT": 20260320,
                "BV_QTY": 0,
                "CURVE_QTY": 10,
            }
        )

        self.assertEqual(fn.lead_working_days, 7)
        self.assertEqual(fn.start_date, date(2026, 3, 11))
        self.assertEqual(nc.lead_working_days, 10)
        self.assertEqual(nc.start_date, date(2026, 3, 6))

    def test_rejects_missing_family_specific_reference_date(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing_required_date"):
            calculate_cutting_start_date(
                {
                    "PROJ_NO": "P1",
                    "BLK_NO": "B1",
                    "GYEL": "FL",
                    "PTLST_QTY": 1,
                    "CUT_LTH": 1,
                    "BVL_LTH": 0,
                }
            )

    def test_audit_preserves_source_date_and_reports_calculated_difference(self) -> None:
        rows = [
            {
                "PROJ_NO": "P1",
                "BLK_NO": "B1",
                "GYEL": "NP",
                "ACT_ST_DT": 20260311,
                "ASS_YN": "N",
                "SASS_ACT_STDT": 20260316,
                "ASS_ACT_STDT": 20260320,
                "CUT_LTH": 100,
                "BV_QTY": 0,
            },
            {
                "PROJ_NO": "P2",
                "BLK_NO": "B2",
                "GYEL": "FL",
                "ACT_ST_DT": 20260310,
                "PAN_ST_DT": 20260316,
                "PTLST_QTY": 13,
                "CUT_LTH": 100,
                "BVL_LTH": 100,
            },
        ]

        audit = audit_cutting_start_dates(rows)
        summary = summarize_cutting_start_date_audit(audit)

        self.assertEqual(audit.loc[0, "BLOCK_SET_ID"], "P1::NP::B1")
        self.assertEqual(audit.loc[0, "SOURCE_ACT_ST_DT"], "20260311")
        self.assertEqual(audit.loc[0, "CALCULATED_ACT_ST_DT"], "20260311")
        self.assertTrue(bool(audit.loc[0, "MATCHED_SOURCE_DATE"]))
        self.assertEqual(audit.loc[1, "SOURCE_ACT_ST_DT"], "20260310")
        self.assertEqual(audit.loc[1, "CALCULATED_ACT_ST_DT"], "20260309")
        self.assertEqual(int(audit.loc[1, "DELTA_CALENDAR_DAYS"]), -1)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["series"]["NP"]["matched_count"], 1)
        self.assertEqual(summary["series"]["FL"]["matched_count"], 0)

    def test_writes_audit_csv_and_summary_json(self) -> None:
        audit = audit_cutting_start_dates(
            [
                {
                    "PROJ_NO": "P1",
                    "BLK_NO": "B1",
                    "GYEL": "NP",
                    "ACT_ST_DT": 20260311,
                    "ASS_YN": "N",
                    "SASS_ACT_STDT": 20260316,
                    "ASS_ACT_STDT": 20260320,
                    "CUT_LTH": 100,
                    "BV_QTY": 0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_cutting_start_date_audit(audit, tmpdir)
            csv_text = Path(paths["csv"]).read_text(encoding="utf-8-sig")
            summary = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))

        self.assertIn("BLOCK_SET_ID", csv_text)
        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(summary["matched_count"], 1)


if __name__ == "__main__":
    unittest.main()
