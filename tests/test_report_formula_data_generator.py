"""MIXED 내부 NP 구성요소가 사용하는 PDF 고정 수식 회귀시험."""

from __future__ import annotations

import hashlib
import unittest

import numpy as np

from Utils.data import report_formula_data_generator as report_formula
from Utils.data.report_formula_data_generator import (
    BLOCK_COLUMNS,
    TACT_A_CUT,
    TACT_A_MARK,
    TACT_A_PTLST,
    TACT_A_THK,
    WO_COLUMNS,
    _generate_work_order_rows,
    _scale_non_negative_values_to_total,
    generate_report_formula_data,
)


class ReportFormulaDataGeneratorTest(unittest.TestCase):
    def test_pdf_formula_coefficients_are_fixed(self) -> None:
        expected = {
            "BLOCK_LENGTH_MEAN": 13380.9,
            "BLOCK_LENGTH_STD": 5376.4,
            "BLOCK_LENGTH_MIN": 515.0,
            "BLOCK_LENGTH_MAX": 21995.0,
            "BLOCK_MARK_A": 0.03425,
            "BLOCK_MARK_B": -178.9,
            "BLOCK_MARK_ZERO_INFLATION": 0.0127,
            "BLOCK_MARK_GAMMA_SHAPE": 3.049,
            "BLOCK_MARK_GAMMA_SCALE": 150.8,
            "BLOCK_MARK_GAMMA_SHIFT": -459.9,
            "BLOCK_CUT_A": 1.7056,
            "BLOCK_CUT_B": 59.21,
            "BLOCK_CUT_GAMMA_SHAPE": 3.391,
            "BLOCK_CUT_GAMMA_SCALE": 0.292,
            "BLOCK_STEEL_A": 0.01203,
            "BLOCK_STEEL_B": 2.014,
            "BLOCK_STEEL_GAMMA_SHAPE": 7.616,
            "BLOCK_STEEL_GAMMA_SCALE": 0.128,
            "BLOCK_THICKNESS_A": 3.7816,
            "BLOCK_THICKNESS_B": 13.31,
            "BLOCK_THICKNESS_STD": 4.797,
            "WO_LENGTH_A": 0.95,
            "WO_LENGTH_B": 0.86,
            "WO_LENGTH_POWER": 0.89,
            "WO_THICKNESS_BASE": 12.4,
            "WO_THICKNESS_AMP": 6.7,
            "WO_THICKNESS_DECAY": -2.85,
            "WO_THICKNESS_SKEW_SHAPE": 7.09,
            "WO_THICKNESS_SKEW_LOC": -1.15,
            "WO_THICKNESS_SKEW_SCALE": 1.53,
            "WO_CUT_A": 0.0058,
            "WO_CUT_B": 12.649,
            "WO_CUT_STD": 42.072,
            "WO_MARK_A": 0.0047,
            "WO_MARK_B": -8.293,
            "WO_MARK_STD": 17.452,
            "WO_BEVEL_A_THK": 0.827,
            "WO_BEVEL_A_LTH": 0.0004,
            "WO_BEVEL_A_MARK": 0.0649,
            "WO_BEVEL_B": -12.496,
            "WO_BEVEL_STD": 7.602,
            "WO_BVQ_A_BVL": 0.1447,
            "WO_BVQ_A_CUT": 0.01236,
            "WO_BVQ_B": 1.4136,
            "WO_BVQ_STD": 2.812,
            "WO_PTLST_A_CUT": 0.2156,
            "WO_PTLST_A_LTH": -0.0005,
            "WO_PTLST_A_MARK": -0.1079,
            "WO_PTLST_B": 5.441,
            "WO_PTLST_STD": 7.602,
            "TACT_A_CUT": 0.3037,
            "TACT_A_MARK": 0.1325,
            "TACT_A_THK": 0.4790,
            "TACT_A_PTLST": 0.3840,
        }

        for name, value in expected.items():
            self.assertEqual(getattr(report_formula, name), value, name)
        self.assertFalse(hasattr(report_formula, "TACT_A_BV"))

    def test_generated_np_component_preserves_block_wo_identity(self) -> None:
        generated = generate_report_formula_data(n_blocks=8, seed=77)

        self.assertEqual(tuple(generated.wo_df.columns), WO_COLUMNS)
        self.assertEqual(tuple(generated.block_df.columns), BLOCK_COLUMNS)
        self.assertEqual(len(generated.block_df), 8)
        self.assertFalse(generated.wo_df.isna().any().any())
        grouped = generated.wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"])
        for block in generated.block_df.itertuples(index=False):
            rows = grouped.get_group((block.PROJ_NO, block.GYEL, block.BLK_NO))
            self.assertEqual(block.LTH, rows["LTH"].max())
            self.assertEqual(block.BTH, rows["BTH"].max())
            self.assertEqual(block.THK, rows["THK"].max())
            self.assertEqual(block.STL_QTY, int(rows["STL_QTY"].sum()))
            self.assertAlmostEqual(block.CUT_LTH, float(rows["CUT_LTH"].sum()), places=6)
            self.assertAlmostEqual(block.MARK_LTH, float(rows["MARK_LTH"].sum()), places=6)

    def test_wo_count_and_stl_quantity_are_not_aliased(self) -> None:
        generated = generate_report_formula_data(n_blocks=100, seed=20260715)

        self.assertGreater(generated.wo_df["STL_QTY"].nunique(), 1)
        self.assertNotEqual(len(generated.wo_df), int(generated.block_df["STL_QTY"].sum()))
        wo_counts = generated.wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"]).size()
        block_stl = generated.block_df.set_index(["PROJ_NO", "GYEL", "BLK_NO"])["STL_QTY"]
        self.assertTrue((wo_counts != block_stl).any())

    def test_tact_time_is_shared_case6_formula_without_bevel_term(self) -> None:
        generated = generate_report_formula_data(n_blocks=4, seed=20260714)
        expected = (
            TACT_A_CUT * generated.wo_df["CUT_LTH"]
            + TACT_A_MARK * generated.wo_df["MARK_LTH"]
            + TACT_A_THK * generated.wo_df["THK"]
            + TACT_A_PTLST * generated.wo_df["PTLST_QTY"]
        )

        np.testing.assert_allclose(
            generated.wo_df["TACT_TIME"].to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
        )

    def test_auxiliary_bth_and_stl_streams_do_not_change_pdf_core(self) -> None:
        generated = generate_report_formula_data(n_blocks=5, seed=77)
        wo_core = generated.wo_df[
            [
                "PROJ_NO", "BLK_NO", "WK_ORD_NO", "GYEL", "LTH", "THK",
                "CUT_LTH", "MARK_LTH", "BVL_LTH", "BV_QTY", "TACT_TIME", "PTLST_QTY",
            ]
        ]
        block_core = generated.block_df[
            [
                "PROJ_NO", "BLK_NO", "GYEL", "LTH", "THK", "CUT_LTH",
                "MARK_LTH", "BVL_LTH", "BV_QTY", "TACT_TIME", "PTLST_QTY",
            ]
        ]

        self.assertEqual(
            hashlib.sha256(wo_core.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
            "7dbe81b401985390e97e24b8ce2896e2a527090edc8c3ed811eea414a9727c45",
        )
        self.assertEqual(
            hashlib.sha256(block_core.to_csv(index=False, lineterminator="\n").encode()).hexdigest(),
            "c12890ef7ac2e3337a76cfab78a353997e06acd2d139522a6da3e6eafd3414e0",
        )

    def test_positive_block_total_cannot_fall_back_from_zero_wo_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "positive W/O formula total"):
            _scale_non_negative_values_to_total(
                raw_values=np.zeros(3),
                total=100.0,
                field_name="CUT_LTH",
            )

    def test_wo_index_starts_at_zero_and_keeps_block_max_length(self) -> None:
        rows = _generate_work_order_rows(
            rng=np.random.default_rng(20260714),
            project_no="PPT_PROJ",
            block_no="PPT_BLOCK",
            gyel="NP",
            block_length=15_000.0,
            block_thickness=20.0,
            block_mark_length=731.125,
            block_cut_length=1_824.75,
            wo_count=7,
            thickness_specs=report_formula.DEFAULT_THICKNESS_SPECS,
        )

        self.assertEqual(max(float(row["LTH"]) for row in rows), 15_000.0)
        self.assertAlmostEqual(sum(float(row["MARK_LTH"]) for row in rows), 731.125, places=6)
        self.assertAlmostEqual(sum(float(row["CUT_LTH"]) for row in rows), 1_824.75, places=6)

    def test_np_component_rejects_non_np_public_request(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "NP"):
            generate_report_formula_data(n_blocks=1, seed=7, gyel="FL")


if __name__ == "__main__":
    unittest.main()
