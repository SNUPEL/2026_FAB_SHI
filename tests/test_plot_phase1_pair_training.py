"""Tests for Phase 1 pair training plot aggregation."""

import unittest

from scripts.plot_phase1_pair_training import (
    _best_agent_rows_by_episode,
    _latest_objective_scope_rows,
    _score_labels,
)


class Phase1PairTrainingPlotTest(unittest.TestCase):
    """Proposed curves must use the best agent candidate, not greedy only."""

    def test_best_agent_rows_choose_best_of_greedy_and_samples(self) -> None:
        rows = [
            {
                "episode": "1",
                "source": "agent_greedy",
                "score_json": "[5, 100, 10]",
                "learning_score_json": "[5, 100, 10]",
            },
            {
                "episode": "1",
                "source": "agent_sample_2",
                "score_json": "[2, 200, 9]",
                "learning_score_json": "[2, 200, 9]",
            },
            {
                "episode": "1",
                "source": "steel_first_balanced",
                "score_json": "[0, 300, 1]",
                "learning_score_json": "[0, 300, 1]",
            },
        ]

        best_by_episode = _best_agent_rows_by_episode(rows)

        self.assertEqual(best_by_episode[1]["source"], "agent_sample_2")

    def test_wo_first_uses_six_shared_and_series_score_labels(self) -> None:
        self.assertEqual(
            _score_labels("wo_first"),
            [
                "shared W/O gap",
                "series W/O gap",
                "shared cut gap",
                "series cut gap",
                "shared bevel gap",
                "series bevel gap",
            ],
        )

    def test_wo_first_series_only_uses_three_series_score_labels(self) -> None:
        self.assertEqual(
            _score_labels("wo_first", "series_only"),
            ["series W/O gap", "series cut gap", "series bevel gap"],
        )

    def test_mixed_objective_history_uses_latest_scope_rows(self) -> None:
        rows = [
            {"episode": "1", "score_mode": "wo_first", "objective_scope": "shared_and_series"},
            {"episode": "2", "score_mode": "wo_first", "objective_scope": "series_only"},
            {"episode": "3", "score_mode": "wo_first", "objective_scope": "series_only"},
        ]

        filtered, objective_scope = _latest_objective_scope_rows(rows)

        self.assertEqual(objective_scope, "series_only")
        self.assertEqual([row["episode"] for row in filtered], ["2", "3"])


if __name__ == "__main__":
    unittest.main()
