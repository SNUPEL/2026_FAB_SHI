"""실적 계열 조합을 보존하는 다계열 수식 생성기 회귀시험."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Utils.data.multi_series_formula_data_generator import (
    DEFAULT_MULTI_SERIES_BLOCK_SOURCE,
    JOINT_BLOCK_FEATURES,
    SUPPORTED_SERIES,
    build_multi_series_formula_episode_jobs,
    fit_physical_block_joint_profile,
    generate_multi_series_formula_data,
    match_generated_blocks_to_joint_targets,
    sample_physical_block_joint_targets,
    validate_multi_series_formula_data,
)
from Utils.data.report_formula_data_generator import build_report_formula_episode_jobs
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


def _generated_joint_block(source_block_id: str, base: float) -> dict:
    return {
        "source_block_id": source_block_id,
        "WO_QTY": int(base // 10),
        "LTH": base,
        "THK": base,
        "BTH": base,
        "CUT_LTH": base,
        "BV_QTY": base,
    }


def _joint_target_rows(series: str, ranks: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for physical_index, rank in enumerate(ranks, start=1):
        row = {"physical_index": physical_index, "GYEL": series}
        row.update({f"{feature}__rank": rank for feature in JOINT_BLOCK_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


class MultiSeriesFormulaDataGeneratorTest(unittest.TestCase):
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
        self.assertEqual(dict(zip(p1_b1["GYEL"], p1_b1["WO_QTY"])), {"FL": 1, "NP": 2})
        for feature in JOINT_BLOCK_FEATURES:
            self.assertIn(f"{feature}__rank", profile.rows.columns)

    def test_joint_target_sampling_is_deterministic_and_never_splits_a_donor(self) -> None:
        work_orders = pd.DataFrame(
            [
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

        first = sample_physical_block_joint_targets(profile, 100, np.random.default_rng(17))
        second = sample_physical_block_joint_targets(profile, 100, np.random.default_rng(17))

        assert_frame_equal(first, second, check_exact=True)
        combinations = first.groupby("physical_index")["GYEL"].agg(lambda values: tuple(sorted(values)))
        self.assertTrue(set(combinations).issubset({("FL", "NP"), ("NP",)}))
        self.assertIn(("FL", "NP"), set(combinations))
        self.assertIn(("NP",), set(combinations))

    def test_joint_rank_matching_reproduces_cross_series_direction_without_changing_marginals(self) -> None:
        generated_np = pd.DataFrame(
            [_generated_joint_block("NP_LOW", 10.0), _generated_joint_block("NP_MID", 20.0), _generated_joint_block("NP_HIGH", 30.0)]
        )
        generated_fl = pd.DataFrame(
            [_generated_joint_block("FL_LOW", 100.0), _generated_joint_block("FL_MID", 200.0), _generated_joint_block("FL_HIGH", 300.0)]
        )
        np_targets = _joint_target_rows("NP", (0.1, 0.5, 0.9))
        fl_targets = _joint_target_rows("FL", (0.9, 0.5, 0.1))

        np_mapping = match_generated_blocks_to_joint_targets(generated_np, np_targets)
        fl_mapping = match_generated_blocks_to_joint_targets(generated_fl, fl_targets)

        self.assertEqual(np_mapping, {"NP_LOW": 1, "NP_MID": 2, "NP_HIGH": 3})
        self.assertEqual(fl_mapping, {"FL_HIGH": 1, "FL_MID": 2, "FL_LOW": 3})
        np_values = generated_np.set_index("source_block_id")["LTH"]
        fl_values = generated_fl.set_index("source_block_id")["LTH"]
        coupled_np = pd.Series({physical: np_values[source] for source, physical in np_mapping.items()})
        coupled_fl = pd.Series({physical: fl_values[source] for source, physical in fl_mapping.items()})
        self.assertAlmostEqual(float(coupled_np.corr(coupled_fl)), -1.0)
        self.assertEqual(sorted(coupled_np), [10.0, 20.0, 30.0])
        self.assertEqual(sorted(coupled_fl), [100.0, 200.0, 300.0])

    def test_joint_target_sampling_reproduces_actual_probabilities(self) -> None:
        work_orders = pd.read_excel("변경사항/절단WO_데이터.xlsx", sheet_name="Sheet1")
        blocks = pd.read_excel(DEFAULT_MULTI_SERIES_BLOCK_SOURCE, sheet_name="Sheet1")
        profile = fit_physical_block_joint_profile(work_orders, blocks)

        targets = sample_physical_block_joint_targets(profile, 100_000, np.random.default_rng(20260714))
        samples = targets.groupby("physical_index")["GYEL"].agg(lambda values: tuple(sorted(values)))
        observed = samples.value_counts(normalize=True)

        self.assertEqual(profile.physical_block_count, 1_044)
        self.assertEqual(len(profile.combinations), 13)
        for combination, expected in zip(profile.combinations, profile.probabilities):
            self.assertLess(abs(float(observed[combination]) - expected), 0.005)

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

    def test_generated_multi_series_data_preserves_physical_and_series_block_contracts(self) -> None:
        generated = generate_multi_series_formula_data(n_physical_blocks=24, seed=20260714)

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
