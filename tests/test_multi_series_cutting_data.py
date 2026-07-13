from __future__ import annotations

import unittest

import pandas as pd

from Utils.data.multi_series_cutting_data import (
    build_block_set_id,
    prepare_multi_series_cutting_data,
)


class MultiSeriesCuttingDataTest(unittest.TestCase):
    def test_block_identity_includes_series_and_wo_count_is_not_steel_quantity(self) -> None:
        block_rows = pd.DataFrame(
            [
                self._block_row(
                    "NP",
                    cut=30.0,
                    steel=1,
                    bevel=3,
                    bevel_length=4.0,
                    part_quantity=8,
                ),
                self._block_row("FL", cut=40.0, steel=2, bevel=1),
            ]
        )
        wo_rows = pd.DataFrame(
            [
                self._wo_row("WO_NP_1", "NP", cut=10.0, steel=1, bevel=1),
                self._wo_row("WO_NP_2", "NP", cut=20.0, steel=0, bevel=2),
                self._wo_row("WO_FL_1", "FL", cut=40.0, steel=2, bevel=1),
            ]
        )

        prepared = prepare_multi_series_cutting_data(block_rows, wo_rows)

        np_key = build_block_set_id("P1", "NP", "B1")
        fl_key = build_block_set_id("P1", "FL", "B1")
        self.assertNotEqual(np_key, fl_key)
        blocks = prepared.blocks.set_index("BLOCK_SET_ID")
        self.assertEqual(int(blocks.loc[np_key, "WO_QTY"]), 2)
        self.assertEqual(int(blocks.loc[np_key, "STL_QTY"]), 1)
        self.assertEqual(float(blocks.loc[np_key, "MARK_LTH"]), 5.0)
        self.assertNotIn("RET_QTY", prepared.work_orders.columns)

    def test_rejects_block_and_wo_aggregate_mismatch(self) -> None:
        block_rows = pd.DataFrame(
            [
                self._block_row(
                    "NP",
                    cut=31.0,
                    steel=1,
                    bevel=3,
                    bevel_length=4.0,
                    part_quantity=8,
                )
            ]
        )
        wo_rows = pd.DataFrame(
            [
                self._wo_row("WO_1", "NP", cut=10.0, steel=1, bevel=1),
                self._wo_row("WO_2", "NP", cut=20.0, steel=0, bevel=2),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "block_wo_aggregate_mismatch"):
            prepare_multi_series_cutting_data(block_rows, wo_rows)

    def test_rejects_missing_required_column_without_alias_fallback(self) -> None:
        block_rows = pd.DataFrame([self._block_row("NP", cut=30.0, steel=1, bevel=3)]).drop(columns=["BTH"])
        wo_rows = pd.DataFrame([self._wo_row("WO_1", "NP", cut=30.0, steel=1, bevel=3)])

        with self.assertRaisesRegex(RuntimeError, "missing_required_columns"):
            prepare_multi_series_cutting_data(block_rows, wo_rows)

    @staticmethod
    def _block_row(
        series: str,
        *,
        cut: float,
        steel: int,
        bevel: int,
        mark: float = 5.0,
        bevel_length: float = 2.0,
        part_quantity: int = 4,
    ) -> dict:
        return {
            "PROJ_NO": "P1",
            "BLK_NO": "B1",
            "ACT_ST_DT": 20260309,
            "ACT_ED_DT": 20260310,
            "RT_CUT_ST_DTM": 202603090800,
            "RT_CUT_ED_DTM": 202603090900,
            "GYEL": series,
            "LTH": 10000,
            "BTH": 3000,
            "THK": 20,
            "WGT": 100,
            "MARK_LTH": mark,
            "CUT_LTH": cut,
            "BVL_LTH": bevel_length,
            "PTLST_QTY": part_quantity,
            "STL_QTY": steel,
            "BV_QTY": bevel,
            "CURVE_QTY": 0,
            "ASS_YN": "N",
            "CUT_BAY": "22" if series == "NP" else "25",
            "EQP_NM": "EQP_1",
            "SASS_WKA": "S1",
            "ASS_WKA": "A1",
            "SASS_ACT_STDT": 20260313,
            "ASS_ACT_STDT": 20260323,
            "PAN_ST_DT": 20260316 if series == "FL" else None,
        }

    @classmethod
    def _wo_row(cls, wo: str, series: str, *, cut: float, steel: int, bevel: int) -> dict:
        row = cls._block_row(series, cut=cut, steel=steel, bevel=bevel)
        row["WK_ORD_NO"] = wo
        row["RET_QTY"] = 1
        row["LTH"] = 10000
        row["BTH"] = 3000
        row["THK"] = 20
        return row


if __name__ == "__main__":
    unittest.main()
