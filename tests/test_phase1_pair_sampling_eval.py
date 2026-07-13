"""Tests for Phase 1 actual-vs-heuristic-vs-Proposed evaluation."""

from types import SimpleNamespace
import unittest

from scripts.evaluate_phase1_pair_sampling import (
    PPB_LCP6_HEURISTICS,
    _heuristic_candidate_for_eval,
    _label_for,
    _metric_specs,
    _resolve_heuristics,
    _score_row,
)
from Phase1.pair_self_labeling import run_phase1_pair_policy_rollout


class Phase1PairSamplingEvalTest(unittest.TestCase):
    """The presentation comparison must not reuse Proposed's action mask for baselines."""

    def test_ppb_lcp6_expands_to_three_block_orders_by_two_bay_rules(self) -> None:
        self.assertEqual(
            _resolve_heuristics("ppb_lcp6"),
            [
                "steel_first_balanced",
                "cut_first_balanced",
                "bevel_first_balanced",
                "steel_first_long_cut_preferred",
                "cut_first_long_cut_preferred",
                "bevel_first_long_cut_preferred",
            ],
        )
        self.assertEqual(_resolve_heuristics("business6"), PPB_LCP6_HEURISTICS)
        self.assertEqual(_resolve_heuristics("all6"), PPB_LCP6_HEURISTICS)
        self.assertNotIn("long_cut_first_balanced", _resolve_heuristics("ppb_lcp6"))

    def test_presentation_labels_use_ppb_and_lcp_without_long_alias_noise(self) -> None:
        self.assertEqual(_label_for("steel_first_balanced"), "MSF+PPB")
        self.assertEqual(_label_for("cut_first_long_cut_preferred"), "LCF+LCP")
        self.assertEqual(_label_for("bevel_first_long_cut_preferred"), "MBF+LCP")

    def test_eval_heuristic_keeps_bay24_available_for_long_cut_baseline(self) -> None:
        jobs = {
            "WO_LONG": SimpleNamespace(
                job_id="WO_LONG",
                block_set_id="P1::LONG",
                steel_quantity=10,
                plate_length=1200,
                thickness=12,
                cut_length=1200.0,
                bevel_quantity=1,
                cut_bay=None,
                allowed_bay_ids=("24",),
            )
        }

        candidate = _heuristic_candidate_for_eval(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm="steel_first_balanced",
        )

        self.assertEqual(candidate.assignments, {"P1::LONG": "24"})
        self.assertEqual(candidate.bay_loads["24"]["long_cut_bay24_count"], 1)

    def test_proposed_rollout_can_disable_long_cut_hard_mask_for_eval(self) -> None:
        jobs = {
            "WO_LONG": SimpleNamespace(
                job_id="WO_LONG",
                block_set_id="P1::LONG",
                steel_quantity=10,
                plate_length=1200,
                thickness=12,
                cut_length=1200.0,
                bevel_quantity=1,
                cut_bay=None,
                allowed_bay_ids=("24",),
            )
        }

        with self.assertRaises(RuntimeError):
            run_phase1_pair_policy_rollout(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                model=None,
                temperature=1.0,
                seed=1,
                source="masked",
                selection="greedy",
            )

        candidate = run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            model=None,
            temperature=1.0,
            seed=1,
            source="unmasked",
            selection="greedy",
            long_cut_hard_mask=False,
        )

        self.assertEqual(candidate.assignments, {"P1::LONG": "24"})
        self.assertEqual(candidate.bay_loads["24"]["long_cut_bay24_count"], 1)

    def test_score_row_and_plots_use_three_workload_objectives(self) -> None:
        row = _score_row(1, "ACTUAL_20260331", "candidate_workbook", 25, "proposed_sampling256", (0, 1.5, 2), 1)

        self.assertEqual(row["evaluation_input_type"], "candidate_workbook")
        self.assertEqual(row["steel_gap"], 0)
        self.assertEqual(row["cut_gap"], 1.5)
        self.assertEqual(row["bevel_gap"], 2)
        self.assertNotIn("score_3", row)
        self.assertEqual(len(_metric_specs("steel_first")), 3)


if __name__ == "__main__":
    unittest.main()
