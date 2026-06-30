"""Tests for Phase 1 actual-vs-heuristic-vs-Proposed evaluation."""

from types import SimpleNamespace
import unittest

from scripts.evaluate_phase1_pair_sampling import (
    BUSINESS6_HEURISTICS,
    _heuristic_candidate_for_eval,
    _resolve_heuristics,
)


class Phase1PairSamplingEvalTest(unittest.TestCase):
    """The presentation comparison must not reuse Proposed's action mask for baselines."""

    def test_business6_expands_to_three_block_orders_by_two_bay_rules(self) -> None:
        self.assertEqual(
            _resolve_heuristics("business6"),
            [
                "steel_first_balanced",
                "cut_first_balanced",
                "bevel_first_balanced",
                "steel_first_long_cut_preferred",
                "cut_first_long_cut_preferred",
                "bevel_first_long_cut_preferred",
            ],
        )
        self.assertEqual(_resolve_heuristics("all6"), BUSINESS6_HEURISTICS)

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


if __name__ == "__main__":
    unittest.main()
