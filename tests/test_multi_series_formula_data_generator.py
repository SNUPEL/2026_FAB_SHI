"""실적 계열 조합을 보존하는 다계열 수식 생성기 회귀시험."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Utils.data import multi_series_formula_data_generator as multi_formula
from Utils.data.multi_series_formula_data_generator import (
    DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    SUPPORTED_SERIES,
    build_multi_series_formula_episode_jobs,
    fit_physical_block_joint_profile,
    generate_multi_series_formula_data,
    select_physical_block_series_allocations,
    validate_multi_series_formula_data,
)
from Utils.data.report_formula_data_generator import (
    build_report_formula_episode_jobs,
    generate_report_formula_block_seeds,
)
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs


def _joint_block_row(project: str, block: str, series: str, base: float) -> dict:
    return {
        "PROJ_NO": project,
        "BLK_NO": block,
        "GYEL": series,
        "LTH": base,
        "THK": base / 10.0,
        "BTH": base * 2.0,
        "CUT_LTH": base * 3.0,
        "BV_QTY": int(base // 100),
    }


class MultiSeriesFormulaDataGeneratorTest(unittest.TestCase):
    def test_python_seed_conversion_preserves_uint32_range(self) -> None:
        values = pd.Series(
            [
                np.uint32(np.iinfo(np.int32).max + 1),
                np.uint32(np.iinfo(np.uint32).max),
            ]
        )

        self.assertEqual(
            multi_formula._python_int_tuple(values),
            (2_147_483_648, 4_294_967_295),
        )

    def test_joint_profile_uses_physical_block_identity_and_wo_counts(self) -> None:
        work_orders = pd.DataFrame(
            [
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "NP"},
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "NP"},
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "FL"},
                {"PROJ_NO": "P1", "BLK_NO": "B2", "GYEL": "NP"},
                {"PROJ_NO": "P2", "BLK_NO": "B1", "GYEL": "NC"},
            ]
        )
        blocks = pd.DataFrame(
            [
                _joint_block_row("P1", "B1", "NP", 200.0),
                _joint_block_row("P1", "B1", "FL", 100.0),
                _joint_block_row("P1", "B2", "NP", 300.0),
                _joint_block_row("P2", "B1", "NC", 400.0),
            ]
        )

        profile = fit_physical_block_joint_profile(work_orders, blocks)

        self.assertEqual(profile.combinations, (("FL", "NP"), ("NC",), ("NP",)))
        np.testing.assert_allclose(profile.probabilities, [1 / 3, 1 / 3, 1 / 3])
        p1_b1 = profile.rows.loc[
            profile.rows["PROJ_NO"].eq("P1") & profile.rows["BLK_NO"].eq("B1")
        ]
        self.assertEqual(
            dict(zip(p1_b1["GYEL"], p1_b1["SERIES_WO_QTY"])),
            {"FL": 1, "NP": 2},
        )
        physical = profile.physical_rows.set_index(["PROJ_NO", "BLK_NO"])
        self.assertEqual(int(physical.loc[("P1", "B1"), "TOTAL_WO_QTY"]), 3)
        self.assertEqual(physical.loc[("P1", "B1"), "SERIES_COMBINATION"], ("FL", "NP"))

    def test_series_allocation_is_conditioned_on_total_and_preserves_count_vector(self) -> None:
        work_orders = pd.DataFrame(
            [
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "NP"},
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "NP"},
                {"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "FL"},
                {"PROJ_NO": "P1", "BLK_NO": "B2", "GYEL": "NP"},
            ]
        )
        blocks = pd.DataFrame(
            [
                _joint_block_row("P1", "B1", "NP", 100.0),
                _joint_block_row("P1", "B1", "FL", 200.0),
                _joint_block_row("P1", "B2", "NP", 300.0),
            ]
        )
        profile = fit_physical_block_joint_profile(work_orders, blocks)
        block_seeds = pd.DataFrame(
            [
                {
                    "physical_index": 1,
                    "LTH": 100.0,
                    "THK": 10.0,
                    "CUT_LTH": 300.0,
                    "WO_QTY": 3,
                    "PHYSICAL_RANDOM_SEED": 101,
                },
                {
                    "physical_index": 2,
                    "LTH": 300.0,
                    "THK": 30.0,
                    "CUT_LTH": 900.0,
                    # 실적 profile에는 총 W/O 수 4가 없다. 그래도 원 수식값 4를
                    # 다른 지원값으로 바꾸지 않고 계열 count vector 합으로 보존해야 한다.
                    "WO_QTY": 4,
                    "PHYSICAL_RANDOM_SEED": 202,
                },
            ]
        )

        first = select_physical_block_series_allocations(
            profile, block_seeds, np.random.default_rng(17)
        )
        second = select_physical_block_series_allocations(
            profile, block_seeds, np.random.default_rng(17)
        )

        assert_frame_equal(first, second, check_exact=True)
        totals = first.groupby("physical_index", sort=True)["SERIES_WO_QTY"].sum()
        self.assertEqual(totals.tolist(), [3, 4])
        formula_totals = first.groupby("physical_index", sort=True)["FORMULA_WO_QTY"].first()
        self.assertEqual(formula_totals.tolist(), [3, 4])
        self.assertTrue(first["FORMULA_WO_QTY"].eq(first["TOTAL_WO_QTY"]).all())
        self.assertNotIn("COUNT_ADJUSTED_TO_SUPPORT", first.columns)
        first_vector = dict(
            zip(
                first.loc[first["physical_index"].eq(1), "GYEL"],
                first.loc[first["physical_index"].eq(1), "SERIES_WO_QTY"],
            )
        )
        self.assertIn(first_vector, ({"FL": 1, "NP": 2}, {"NP": 3}))

    def test_post_generation_block_matching_api_is_removed(self) -> None:
        self.assertFalse(hasattr(multi_formula, "match_generated_blocks_to_joint_targets"))

    def test_series_combination_sampling_tracks_actual_distribution(self) -> None:
        work_orders = pd.read_excel("변경사항/절단WO_데이터.xlsx", sheet_name="Sheet1")
        blocks = pd.read_excel(DEFAULT_MULTI_SERIES_BLOCK_SOURCE, sheet_name="Sheet1")
        profile = fit_physical_block_joint_profile(work_orders, blocks)
        seed_sequence = np.random.SeedSequence(20260716)
        block_seed_child, allocation_child, physical_child, *_ = seed_sequence.spawn(
            3 + len(SUPPORTED_SERIES)
        )
        block_seed = int(block_seed_child.generate_state(1, dtype=np.uint32)[0])
        block_seeds = generate_report_formula_block_seeds(
            profile.physical_block_count,
            block_seed,
        )
        block_seeds["PHYSICAL_RANDOM_SEED"] = np.random.default_rng(
            physical_child
        ).integers(
            1,
            np.iinfo(np.int32).max,
            size=profile.physical_block_count,
            dtype=np.int64,
        )
        allocations = select_physical_block_series_allocations(
            profile,
            block_seeds,
            np.random.default_rng(allocation_child),
        )

        sampled = allocations.groupby("physical_index", sort=True)["GYEL"].agg(
            lambda values: tuple(sorted(set(values)))
        ).value_counts(normalize=True)
        expected = dict(zip(profile.combinations, profile.probabilities))
        observed = sampled.to_dict()
        combinations = set(expected) | set(observed)
        total_variation = 0.5 * sum(
            abs(float(expected.get(combination, 0.0)) - float(observed.get(combination, 0.0)))
            for combination in combinations
        )
        self.assertLess(float(total_variation), 0.05)

    def test_validation_orders_physical_blocks_by_numeric_generated_index(self) -> None:
        rows = []
        for project_no, block_no, series in (
            ("SYNTH_MULTI_PROJ_2", "SYNTH_MULTI_BLK_001001", "NP"),
            ("SYNTH_MULTI_PROJ_9", "SYNTH_MULTI_BLK_008001", "FL"),
            ("SYNTH_MULTI_PROJ_10", "SYNTH_MULTI_BLK_009001", "NC"),
        ):
            rows.append(
                {
                    "PROJ_NO": project_no,
                    "BLK_NO": block_no,
                    "WK_ORD_NO": f"WO_{block_no}",
                    "GYEL": series,
                    "LTH": 100.0,
                    "BTH": 1_000.0,
                    "THK": 10.0,
                    "CUT_LTH": 10.0,
                    "MARK_LTH": 5.0,
                    "BVL_LTH": 1.0,
                    "STL_QTY": 1,
                    "BV_QTY": 1,
                    "TACT_TIME": 9.2575,
                    "PTLST_QTY": 2,
                }
            )
        work_orders = pd.DataFrame(rows)
        blocks = pd.DataFrame(
            [
                {
                    "PROJ_NO": row["PROJ_NO"],
                    "BLK_NO": row["BLK_NO"],
                    "GYEL": row["GYEL"],
                    "LTH": row["LTH"],
                    "BTH": row["BTH"],
                    "THK": row["THK"],
                    "CUT_LTH": row["CUT_LTH"],
                    "MARK_LTH": row["MARK_LTH"],
                    "BVL_LTH": row["BVL_LTH"],
                    "STL_QTY": row["STL_QTY"],
                    "WO_QTY": 1,
                    "BV_QTY": row["BV_QTY"],
                    "TACT_TIME": row["TACT_TIME"],
                    "PTLST_QTY": row["PTLST_QTY"],
                }
                for row in rows
            ]
        )

        validate_multi_series_formula_data(
            work_orders,
            blocks,
            expected_physical_blocks=3,
            expected_series_combinations=(("NP",), ("FL",), ("NC",)),
        )

    def test_series_allocation_rejects_fractional_physical_index(self) -> None:
        work_orders = pd.DataFrame(
            [{"PROJ_NO": "P1", "BLK_NO": "B1", "GYEL": "NP"}]
        )
        blocks = pd.DataFrame([_joint_block_row("P1", "B1", "NP", 100.0)])
        profile = fit_physical_block_joint_profile(work_orders, blocks)
        block_seeds = pd.DataFrame(
            [
                {
                    "physical_index": 1.5,
                    "LTH": 100.0,
                    "THK": 10.0,
                    "CUT_LTH": 300.0,
                    "WO_QTY": 1,
                    "PHYSICAL_RANDOM_SEED": 101,
                }
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "invalid physical block seed values"):
            select_physical_block_series_allocations(
                profile,
                block_seeds,
                np.random.default_rng(17),
            )

    def test_generated_multi_series_data_preserves_physical_and_series_block_contracts(self) -> None:
        seed = 20260714
        generated = generate_multi_series_formula_data(n_physical_blocks=24, seed=seed)

        block_seed_child = np.random.SeedSequence(seed).spawn(3 + len(SUPPORTED_SERIES))[0]
        block_seed = int(block_seed_child.generate_state(1, dtype=np.uint32)[0])
        original_formula = generate_report_formula_block_seeds(24, block_seed)
        formula_totals_by_index = generated.allocation_df.groupby(
            "physical_index", sort=True
        )["FORMULA_WO_QTY"].first()
        self.assertEqual(
            formula_totals_by_index.tolist(),
            original_formula.sort_values("physical_index")["WO_QTY"].astype(int).tolist(),
        )

        self.assertEqual(generated.physical_block_count, 24)
        self.assertEqual(
            generated.block_df.groupby(["PROJ_NO", "BLK_NO"]).ngroups,
            24,
        )
        self.assertGreater(len(generated.block_df), generated.physical_block_count)
        self.assertTrue(set(generated.block_df["GYEL"]).issubset(SUPPORTED_SERIES))
        self.assertGreaterEqual(generated.block_df["GYEL"].nunique(), 2)
        self.assertEqual(
            generated.block_df.duplicated(["PROJ_NO", "GYEL", "BLK_NO"]).sum(),
            0,
        )
        grouped = generated.wo_df.groupby(["PROJ_NO", "GYEL", "BLK_NO"])
        self.assertEqual(grouped.ngroups, len(generated.block_df))
        allocation_counts = generated.allocation_df.groupby(
            ["PROJ_NO", "BLK_NO", "GYEL"], sort=True
        )["SERIES_WO_QTY"].first()
        generated_counts = generated.wo_df.groupby(
            ["PROJ_NO", "BLK_NO", "GYEL"], sort=True
        ).size()
        self.assertEqual(allocation_counts.to_dict(), generated_counts.to_dict())
        physical_allocations = generated.allocation_df.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        )["SERIES_WO_QTY"].sum()
        physical_rows = generated.wo_df.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        ).size()
        self.assertEqual(physical_allocations.to_dict(), physical_rows.to_dict())
        formula_totals = generated.allocation_df.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        )["FORMULA_WO_QTY"].first()
        self.assertEqual(formula_totals.to_dict(), physical_rows.to_dict())
        multi_series_seed_counts = generated.allocation_df.groupby(
            ["PROJ_NO", "BLK_NO"], sort=True
        )["SERIES_RANDOM_SEED"].agg(lambda values: values.nunique() == len(values))
        self.assertTrue(multi_series_seed_counts.all())
        for _, block in generated.block_df.iterrows():
            key = (block["PROJ_NO"], block["GYEL"], block["BLK_NO"])
            rows = grouped.get_group(key)
            self.assertEqual(float(block["BTH"]), float(rows["BTH"].max()))
            expected_tact = (
                0.3037 * rows["CUT_LTH"]
                + 0.1325 * rows["MARK_LTH"]
                + 0.4790 * rows["THK"]
                + 0.3840 * rows["PTLST_QTY"]
            )
            np.testing.assert_allclose(rows["TACT_TIME"], expected_tact, atol=1e-6, rtol=0.0)

    def test_validation_rejects_fractional_allocation_counts(self) -> None:
        generated = generate_multi_series_formula_data(
            n_physical_blocks=3,
            seed=20260714,
        )
        invalid_allocations = generated.allocation_df.copy()
        invalid_allocations["SERIES_WO_QTY"] = invalid_allocations[
            "SERIES_WO_QTY"
        ].astype(float)
        invalid_allocations.loc[invalid_allocations.index[0], "SERIES_WO_QTY"] += 0.5

        with self.assertRaisesRegex(RuntimeError, "invalid expected series allocation values"):
            validate_multi_series_formula_data(
                generated.wo_df,
                generated.block_df,
                expected_physical_blocks=3,
                expected_series_combinations=generated.series_combinations,
                expected_allocations=invalid_allocations,
            )

    def test_phase1_and_phase2_share_exact_mixed_episode_input(self) -> None:
        phase1 = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=8,
            max_blocks=8,
            seed=20260714,
        )[0]
        phase2 = build_report_formula_episode_jobs(
            episode_count=1,
            min_blocks=8,
            max_blocks=8,
            seed=20260714,
            gyel="MIXED",
        )[0]

        self.assertEqual(phase1["seed"], phase2["seed"])
        self.assertEqual(phase1["physical_block_count"], phase2["physical_block_count"])
        self.assertEqual(tuple(phase1["jobs"]), tuple(phase2["jobs"]))
        assert_frame_equal(phase1["wo_df"], phase2["wo_df"], check_exact=True)
        assert_frame_equal(phase1["blocks"], phase2["block_df"], check_exact=True)

    def test_public_mixed_episode_builder_is_deterministic(self) -> None:
        first = build_multi_series_formula_episode_jobs(1, 6, 6, seed=99)[0]
        second = build_multi_series_formula_episode_jobs(1, 6, 6, seed=99)[0]

        self.assertEqual(first["series_combinations"], second["series_combinations"])
        assert_frame_equal(first["wo_df"], second["wo_df"], check_exact=True)
        assert_frame_equal(first["block_df"], second["block_df"], check_exact=True)

if __name__ == "__main__":
    unittest.main()
