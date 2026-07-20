"""발표용 다계열 합성데이터 생성기의 희소구간 적합 회귀시험."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd
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
    def test_empirical_profile_roundtrip_preserves_exact_generation(self) -> None:
        module = _load_generator_module()
        root = Path(__file__).parents[1]
        fitted = module.ShipyardGenerator(
            root / "변경사항" / "절단WO_데이터.xlsx",
            root / "변경사항" / "절단블록_데이터.xlsx",
            mode="spearman",
            series="FN",
        ).fit()
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

    def test_empirical_generator_accepts_prescribed_wo_counts(self) -> None:
        module = _load_generator_module()
        root = Path(__file__).parents[1]
        generator = module.ShipyardGenerator(
            root / "변경사항" / "절단WO_데이터.xlsx",
            root / "변경사항" / "절단블록_데이터.xlsx",
            mode="spearman",
            series="FN",
        ).fit()

        work_orders, blocks = generator.generate(
            n_blocks=3,
            seed=20260716,
            wo_counts=(1, 4, 2),
            block_seeds=(101, 202, 303),
        )

        self.assertEqual(work_orders.groupby("BLK_ID", sort=True).size().tolist(), [1, 4, 2])
        self.assertEqual(blocks.sort_values("BLK_ID")["WO_QTY"].tolist(), [1, 4, 2])

    def test_empirical_generator_rejects_fractional_prescribed_counts(self) -> None:
        module = _load_generator_module()
        root = Path(__file__).parents[1]
        generator = module.ShipyardGenerator(
            root / "변경사항" / "절단WO_데이터.xlsx",
            root / "변경사항" / "절단블록_데이터.xlsx",
            mode="spearman",
            series="FN",
        ).fit()

        with self.assertRaisesRegex(RuntimeError, "invalid_wo_counts"):
            generator.generate(
                n_blocks=1,
                seed=20260716,
                wo_counts=(1.5,),
                block_seeds=(101,),
            )

    def test_conditional_sampler_uses_bth_formula_and_stl_probability(self) -> None:
        module = _load_generator_module()
        rng = np.random.default_rng(17)
        features = rng.uniform(1.0, 100.0, size=(20, len(module.CONDITIONAL_FEATURES)))
        design = np.column_stack([np.ones(len(features)), np.log1p(features)])
        bth = np.exp(design @ np.array([5.0, 0.1, -0.2, 0.3, 0.05, -0.04, 0.02, 0.01]))
        actual = pd.DataFrame(
            features,
            columns=module.CONDITIONAL_FEATURES,
        )
        actual["GYEL"] = "FN"
        actual["BTH"] = bth
        actual["STL_QTY"] = np.arange(len(actual)) % 3
        model = module._fit_conditional_bth_stl_model(actual)
        generated_features = actual.loc[[1], module.CONDITIONAL_FEATURES]

        bth, stl_quantity = module._sample_conditional_bth_stl(
            generated_features,
            model,
            np.random.default_rng(2026),
            neighbor_count=4,
        )

        self.assertIn(float(bth[0]), model["bth_formula"].observed_specs)
        self.assertIn(int(stl_quantity[0]), actual["STL_QTY"].tolist())

    def test_stl_quantity_sampling_preserves_series_marginal_counts(self) -> None:
        module = _load_generator_module()
        rng = np.random.default_rng(19)
        features = rng.uniform(1.0, 100.0, size=(10, len(module.CONDITIONAL_FEATURES)))
        design = np.column_stack([np.ones(len(features)), np.log1p(features)])
        actual = pd.DataFrame(
            features,
            columns=module.CONDITIONAL_FEATURES,
        )
        actual["GYEL"] = "FL"
        actual["BTH"] = np.exp(
            design @ np.array([5.1, 0.1, -0.1, 0.2, 0.03, -0.02, 0.01, 0.04])
        )
        actual["STL_QTY"] = [0, 0] + [1] * 8
        model = module._fit_conditional_bth_stl_model(actual)
        generated_features = pd.concat(
            [actual[module.CONDITIONAL_FEATURES]] * 10,
            ignore_index=True,
        )

        _, stl_quantity = module._sample_conditional_bth_stl(
            generated_features,
            model,
            np.random.default_rng(2026),
            neighbor_count=4,
        )

        values, counts = np.unique(stl_quantity, return_counts=True)
        self.assertEqual(dict(zip(values.tolist(), counts.tolist())), {0: 20, 1: 80})

    def test_generated_block_uses_required_wo_aggregations(self) -> None:
        module = _load_generator_module()
        generated_wos = pd.DataFrame(
            {
                "LTH": [100.0, 200.0],
                "THK": [10.0, 20.0],
                "MARK_LTH": [5.0, 7.0],
                "CUT_LTH": [20.0, 30.0],
                "BVL_LTH": [1.0, 2.0],
                "BV_QTY": [3, 4],
                "PTLST_QTY": [5, 6],
                "BTH": [3000.0, 4000.0],
                "STL_QTY": [1, 2],
                "TACT_TIME": [30.0, 50.0],
            }
        )

        block = module._aggregate_generated_block(generated_wos, mark_aggregation="max")

        self.assertEqual(block["WO_QTY"], 2)
        self.assertEqual(block["STL_QTY"], 3)
        self.assertEqual(block["BTH"], 4000.0)
        self.assertEqual(block["TACT_TIME"], 50.0)
        self.assertEqual(block["MARK_LTH"], 7.0)

    def test_generated_block_can_reproduce_historical_mark_sum_contract(self) -> None:
        module = _load_generator_module()
        generated_wos = pd.DataFrame(
            {
                "LTH": [100.0, 200.0],
                "THK": [10.0, 20.0],
                "MARK_LTH": [5.0, 7.0],
                "CUT_LTH": [20.0, 30.0],
                "BVL_LTH": [1.0, 2.0],
                "BV_QTY": [3, 4],
                "PTLST_QTY": [5, 6],
                "BTH": [3000.0, 4000.0],
                "STL_QTY": [1, 2],
                "TACT_TIME": [30.0, 50.0],
            }
        )

        block = module._aggregate_generated_block(generated_wos, mark_aggregation="sum")

        self.assertEqual(block["MARK_LTH"], 12.0)

    def test_mark_aggregation_contract_is_resolved_from_block_and_wo_data(self) -> None:
        module = _load_generator_module()
        work_orders = pd.DataFrame(
            {
                "PROJ_NO": ["P1", "P1", "P2", "P2"],
                "GYEL": ["NP", "NP", "NP", "NP"],
                "BLK_NO": ["B1", "B1", "B2", "B2"],
                "MARK_LTH": [2.0, 5.0, 3.0, 7.0],
            }
        )
        max_blocks = pd.DataFrame(
            {
                "PROJ_NO": ["P1", "P2"],
                "GYEL": ["NP", "NP"],
                "BLK_NO": ["B1", "B2"],
                "MARK_LTH": [5.0, 7.0],
            }
        )
        sum_blocks = max_blocks.copy()
        sum_blocks["MARK_LTH"] = [7.0, 10.0]
        keys = ["PROJ_NO", "GYEL", "BLK_NO"]

        self.assertEqual(module._resolve_mark_aggregation(work_orders, max_blocks, keys), "max")
        self.assertEqual(module._resolve_mark_aggregation(work_orders, sum_blocks, keys), "sum")

    def test_scale_to_max_preserves_order_floor_and_exact_maximum(self) -> None:
        module = _load_generator_module()

        scaled = module._scale_to_max(
            np.array([2.0, 4.0, 8.0]),
            maximum=10.0,
            lower_bound=1.0,
        )

        self.assertTrue(np.all(np.diff(scaled) > 0))
        self.assertGreaterEqual(float(scaled.min()), 1.0)
        self.assertEqual(float(scaled.max()), 10.0)

    def test_scale_to_max_preserves_zero_mark_block(self) -> None:
        module = _load_generator_module()

        scaled = module._scale_to_max(
            np.array([2.0, 4.0, 8.0]),
            maximum=0.0,
            lower_bound=1.0,
        )

        np.testing.assert_array_equal(scaled, np.zeros(3))

    def test_zero_inflated_floor_preserves_zero_and_positive_support(self) -> None:
        module = _load_generator_module()

        self.assertEqual(module._apply_zero_inflated_floor(-0.1, 0.26), 0.0)
        self.assertEqual(module._apply_zero_inflated_floor(0.0, 0.26), 0.0)
        self.assertEqual(module._apply_zero_inflated_floor(0.2, 0.26), 0.26)
        self.assertEqual(module._apply_zero_inflated_floor(0.5, 0.26), 0.5)

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

    def test_empirical_generator_rejects_np_series_before_reading_input(self) -> None:
        """NP 요청은 Excel을 읽기 전 canonical PDF 생성기로 보내야 한다."""

        module = _load_generator_module()
        with patch.object(module.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(RuntimeError, "np_fixed_formula_required"):
                module.ShipyardGenerator("wo.xlsx", "block.xlsx", series="NP")
        read_excel.assert_not_called()

    def test_empirical_generator_requires_explicit_non_np_series_before_reading_input(self) -> None:
        """계열 없는 과거 NP 파일이 empirical 경로로 우회하지 못하게 한다."""

        module = _load_generator_module()
        with patch.object(module.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(RuntimeError, "empirical_series_required"):
                module.ShipyardGenerator("wo.xlsx", "block.xlsx", series=None)
        read_excel.assert_not_called()

    def test_block_chain_parameters_are_fitted_from_selected_series(self) -> None:
        module = _load_generator_module()
        length = np.arange(100.0, 700.0, 100.0)
        mark = 2.0 * length + 3.0
        cut = 4.0 * mark + 5.0
        wo_quantity = 0.5 * cut + 1.0
        blocks = pd.DataFrame(
            {
                "LTH": length,
                "MARK_LTH": mark,
                "CUT_LTH": cut,
                "WO_QTY": wo_quantity,
            }
        )

        parameters = module._fit_block_chain_parameters(blocks)

        self.assertAlmostEqual(parameters["mark_a"], 2.0)
        self.assertAlmostEqual(parameters["mark_b"], 3.0)
        self.assertAlmostEqual(parameters["cut_a"], 4.0)
        self.assertAlmostEqual(parameters["cut_b"], 5.0)
        self.assertAlmostEqual(parameters["wo_count_a"], 0.5)
        self.assertAlmostEqual(parameters["wo_count_b"], 1.0)
        self.assertAlmostEqual(parameters["mark_residual_sd"], 0.0, places=8)
        self.assertAlmostEqual(parameters["cut_residual_sd"], 0.0, places=8)
        self.assertAlmostEqual(parameters["wo_count_residual_sd"], 0.0, places=8)

    def test_multi_series_input_requires_explicit_series(self) -> None:
        module = _load_generator_module()
        wo = pd.DataFrame({"GYEL": ["NP", "FN"], "value": [1, 2]})
        block = pd.DataFrame({"GYEL": ["NP", "FN"], "value": [1, 2]})

        with self.assertRaisesRegex(RuntimeError, "series_required_for_multi_series_input"):
            module._select_series_rows(wo, block, None)

        selected_wo, selected_block, selected_series = module._select_series_rows(wo, block, "FN")
        self.assertEqual(selected_series, "FN")
        self.assertEqual(selected_wo["value"].tolist(), [2])
        self.assertEqual(selected_block["value"].tolist(), [2])

    def test_sample_wo_count_allows_single_wo_block(self) -> None:
        module = _load_generator_module()
        rng = np.random.default_rng(2026)

        count = module._sample_wo_count(
            expected_count=1.0,
            residual_sd=0.0,
            minimum=1,
            maximum=12,
            rng=rng,
        )

        self.assertEqual(count, 1)

    def test_adjacent_merge_bin_uses_same_zero_based_position_as_fit(self) -> None:
        module = _load_generator_module()

        bins = [module._adjacent_merge_bin(index, 8) for index in range(7)]

        self.assertEqual(bins, [0, 1, 2, 4, 5, 7, 8])
        self.assertNotIn(9, bins)

    def test_thickness_snap_rejects_missing_allowed_spec_without_fallback(self) -> None:
        module = _load_generator_module()
        generator = object.__new__(module.ShipyardGenerator)
        generator.thk_grid = np.array([10.0, 20.0])
        generator.thk_freq = np.array([0.5, 0.5])

        with self.assertRaisesRegex(RuntimeError, "no_allowed_thickness_spec"):
            generator._snap_thk(12.0, 5.0, np.random.default_rng(1))

    def test_sparse_ratio_bins_use_only_observed_bins(self) -> None:
        module = _load_generator_module()
        centers = np.linspace(0.05, 0.95, 10)
        means = np.array([np.nan, np.nan, 18.0, 17.0, 16.0, 15.0, 14.0, 13.5, 13.0, 12.5])
        standard_deviations = np.array([np.nan, np.nan, 2.0, 2.1, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7])

        mean_parameters, standard_deviation_parameters, observed = module._fit_thickness_ratio_profiles(
            centers,
            means,
            standard_deviations,
        )

        self.assertEqual(observed, 8)
        self.assertTrue(np.isfinite(mean_parameters).all())
        self.assertTrue(np.isfinite(standard_deviation_parameters).all())

    def test_sparse_ratio_bins_reject_insufficient_observations(self) -> None:
        module = _load_generator_module()
        centers = np.linspace(0.05, 0.95, 10)
        means = np.array([np.nan] * 8 + [13.0, 12.5])
        standard_deviations = np.array([np.nan] * 8 + [1.8, 1.7])

        with self.assertRaisesRegex(RuntimeError, "insufficient_observed_thickness_ratio_bins"):
            module._fit_thickness_ratio_profiles(centers, means, standard_deviations)


if __name__ == "__main__":
    unittest.main()
