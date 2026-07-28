"""FN/FL/NC 고정 산출식 생성기와 학습 파이프라인 어댑터의 계약 회귀시험.

새 생성기는 실적 Excel을 다시 적합하지 않고 모듈 상수(`BLOCK`/`WO`)의 고정 계수로
생성한다. 어댑터 `ShipyardGenerator`는 MIXED 학습 파이프라인이 요구하는
`from_generation_profile -> generate(wo_counts, block_seeds)` 계약과 확정 TACT_TIME
(Case6), BLK_ID 그룹키, STL_QTY = W/O 수 계약을 만족해야 한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
from pandas.testing import assert_frame_equal


def _load_generator_module():
    source = Path(__file__).parents[1] / "데이터분석" / "shipyard_data_generator.py"
    spec = importlib.util.spec_from_file_location("shipyard_data_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"generator_module_load_failed: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShipyardDataGeneratorTest(unittest.TestCase):
    def test_tact_formula_matches_pdf_case6_and_ignores_bevel_quantity(self) -> None:
        module = _load_generator_module()
        self.assertEqual(module.TACT_FORMULA_SCOPE, "all_series_shared_np_ppt_case6")
        self.assertEqual(module.TACT_CUT, 0.3037)
        self.assertEqual(module.TACT_MARK, 0.1325)
        self.assertEqual(module.TACT_THK, 0.4790)
        self.assertEqual(module.TACT_PT, 0.3840)
        self.assertFalse(hasattr(module, "TACT_BV"))

        base = module._calculate_tact_time(
            cut_length=np.array([100.0]),
            mark_length=np.array([50.0]),
            thickness=np.array([10.0]),
            part_quantity=np.array([2.0]),
            bevel_quantity=np.array([0.0]),
        )
        thicker = module._calculate_tact_time(
            cut_length=np.array([100.0]),
            mark_length=np.array([50.0]),
            thickness=np.array([20.0]),
            part_quantity=np.array([2.0]),
            bevel_quantity=np.array([0.0]),
        )
        beveled = module._calculate_tact_time(
            cut_length=np.array([100.0]),
            mark_length=np.array([50.0]),
            thickness=np.array([10.0]),
            part_quantity=np.array([2.0]),
            bevel_quantity=np.array([5.0]),
        )
        self.assertGreater(float(thicker[0]), float(base[0]))
        self.assertEqual(float(beveled[0]), float(base[0]))
        self.assertAlmostEqual(float(thicker[0] - base[0]), 10.0 * module.TACT_THK)

    def test_calculate_tact_time_rejects_negative_input(self) -> None:
        module = _load_generator_module()
        with self.assertRaisesRegex(RuntimeError, "invalid_tact_formula_input"):
            module._calculate_tact_time(
                cut_length=np.array([-1.0]),
                mark_length=np.array([50.0]),
                thickness=np.array([10.0]),
                part_quantity=np.array([2.0]),
                bevel_quantity=np.array([0.0]),
            )

    def test_adapter_rejects_np_series_without_reading_excel(self) -> None:
        """NP는 canonical report_formula 고정식이 담당하므로 어댑터가 거부한다."""

        module = _load_generator_module()
        with patch.object(module.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(RuntimeError, "np_fixed_formula_required"):
                module.ShipyardGenerator(series="NP")
        read_excel.assert_not_called()

    def test_adapter_requires_explicit_empirical_series(self) -> None:
        module = _load_generator_module()
        with patch.object(module.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(RuntimeError, "empirical_series_required"):
                module.ShipyardGenerator(series=None)
        read_excel.assert_not_called()

    def test_adapter_honors_prescribed_wo_counts(self) -> None:
        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="FN").fit()
        work_orders, blocks = generator.generate(
            n_blocks=3,
            seed=20260716,
            wo_counts=(1, 4, 2),
            block_seeds=(101, 202, 303),
        )
        self.assertEqual(work_orders.groupby("BLK_ID", sort=True).size().tolist(), [1, 4, 2])
        self.assertEqual(blocks.sort_values("BLK_ID")["WO_QTY"].tolist(), [1, 4, 2])

    def test_adapter_rejects_fractional_prescribed_counts(self) -> None:
        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="FN").fit()
        with self.assertRaisesRegex(RuntimeError, "invalid_wo_counts"):
            generator.generate(
                n_blocks=1,
                seed=20260716,
                wo_counts=(1.5,),
                block_seeds=(101,),
            )

    def test_generation_profile_roundtrip_preserves_exact_generation(self) -> None:
        module = _load_generator_module()
        for series in module.EMPIRICAL_SERIES:
            fitted = module.ShipyardGenerator(series=series).fit()
            restored = module.ShipyardGenerator.from_generation_profile(
                fitted.to_generation_profile()
            )
            expected_wo, expected_blocks = fitted.generate(
                n_blocks=3,
                seed=20260720,
                wo_counts=(1, 4, 2),
                block_seeds=(101, 202, 303),
            )
            actual_wo, actual_blocks = restored.generate(
                n_blocks=3,
                seed=20260720,
                wo_counts=(1, 4, 2),
                block_seeds=(101, 202, 303),
            )
            assert_frame_equal(actual_wo, expected_wo, check_exact=True)
            assert_frame_equal(actual_blocks, expected_blocks, check_exact=True)

    def test_from_generation_profile_rejects_schema_mismatch(self) -> None:
        module = _load_generator_module()
        payload = module.ShipyardGenerator(series="FL").fit().to_generation_profile()
        payload["schema"] = "some_other_schema_v9"
        with self.assertRaisesRegex(RuntimeError, "empirical_profile_schema_mismatch"):
            module.ShipyardGenerator.from_generation_profile(payload)

    def test_generated_stl_quantity_equals_wo_count(self) -> None:
        """새 계약: W/O별 STL_QTY=1 → 블록 STL_QTY = WO_QTY."""

        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="NC").fit()
        work_orders, blocks = generator.generate(
            n_blocks=4,
            seed=7,
            wo_counts=(1, 2, 3, 1),
            block_seeds=(11, 22, 33, 44),
        )
        self.assertTrue((work_orders["STL_QTY"] == 1).all())
        merged = blocks.sort_values("BLK_ID")
        self.assertEqual(merged["STL_QTY"].tolist(), merged["WO_QTY"].tolist())

    def test_generate_produces_pipeline_columns_without_missing_values(self) -> None:
        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="FL").fit()
        work_orders, blocks = generator.generate(
            n_blocks=5,
            seed=99,
            wo_counts=(2, 1, 3, 1, 2),
            block_seeds=(1, 2, 3, 4, 5),
        )
        self.assertEqual(
            list(work_orders.columns),
            list(module.ADAPTER_WO_IDENTITY) + list(module.ADAPTER_WO_FEATURES),
        )
        self.assertEqual(list(blocks.columns), list(module.ADAPTER_BLOCK_COLUMNS))
        self.assertFalse(bool(work_orders.isna().any().any()))
        self.assertFalse(bool(blocks.isna().any().any()))
        self.assertTrue((work_orders["TACT_TIME"] > 0).all())
        self.assertTrue((work_orders["GYEL"] == "FL").all())

    def test_generate_is_deterministic_from_block_seeds(self) -> None:
        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="FN").fit()
        first_wo, first_blk = generator.generate(
            n_blocks=2, seed=1, wo_counts=(2, 3), block_seeds=(500, 600)
        )
        second_wo, second_blk = generator.generate(
            n_blocks=2, seed=999, wo_counts=(2, 3), block_seeds=(500, 600)
        )
        assert_frame_equal(first_wo, second_wo, check_exact=True)
        assert_frame_equal(first_blk, second_blk, check_exact=True)

    def test_block_aggregation_uses_max_sum_contract(self) -> None:
        module = _load_generator_module()
        generator = module.ShipyardGenerator(series="FN").fit()
        work_orders, blocks = generator.generate(
            n_blocks=1, seed=3, wo_counts=(5,), block_seeds=(700,)
        )
        block = blocks.iloc[0]
        self.assertEqual(block["WO_QTY"], 5)
        self.assertEqual(block["LTH"], work_orders["LTH"].max())
        self.assertEqual(block["BTH"], work_orders["BTH"].max())
        self.assertEqual(block["THK"], work_orders["THK"].max())
        self.assertEqual(block["CUT_LTH"], work_orders["CUT_LTH"].sum())
        self.assertEqual(block["MARK_LTH"], work_orders["MARK_LTH"].sum())
        self.assertEqual(block["BVL_LTH"], work_orders["BVL_LTH"].sum())
        self.assertEqual(block["BV_QTY"], work_orders["BV_QTY"].sum())
        self.assertEqual(block["PTLST_QTY"], work_orders["PTLST_QTY"].sum())
        self.assertEqual(block["TACT_TIME"], work_orders["TACT_TIME"].max())

    def test_free_functions_are_param_driven(self) -> None:
        module = _load_generator_module()
        rng = np.random.default_rng(2026)
        block_frame = module.generate_blocks(module.BLOCK["FN"], 1, rng, "FN")
        block_frame = block_frame.copy()
        block_frame["STL_QTY"] = 3
        work_orders = module.generate_wos(module.WO["FN"], block_frame, rng, "FN")
        self.assertEqual(len(work_orders), 3)
        self.assertTrue((work_orders["GYEL"] == "FN").all())


if __name__ == "__main__":
    unittest.main()
