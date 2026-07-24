"""Tests for Phase 1 pair training plot aggregation."""

import unittest

from scripts.plot_phase1_pair_training import (
    _best_agent_rows_by_episode,
    _latest_objective_scope_rows,
    _quick_status,
    _score_labels,
    _subproblem_sources,
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


class Phase1AgentBestRateTest(unittest.TestCase):
    """resource_pool_subproblems_v1 composite best_source must be split per subproblem."""

    def test_composite_best_source_splits_into_subproblem_candidates(self) -> None:
        self.assertEqual(
            _subproblem_sources("NP_NC:wo_first_balanced|FN_FL:agent_sample_8"),
            ["wo_first_balanced", "agent_sample_8"],
        )

    def test_bare_best_source_is_kept_for_legacy_single_subproblem_runs(self) -> None:
        self.assertEqual(_subproblem_sources("agent_greedy"), ["agent_greedy"])

    def test_agent_best_rate_counts_subproblems_not_parent_episodes(self) -> None:
        rows = [
            {"episode": "1", "best_source": "NP_NC:wo_first_balanced|FN_FL:agent_sample_8"},
            {"episode": "2", "best_source": "NP_NC:agent_greedy|FN_FL:agent_sample_3"},
        ]
        sources = [row["best_source"] for row in rows]

        status = _quick_status(rows, [1.0, 0.5], sources, ["agent_greedy", "agent_greedy"])

        # 4 subproblems, 3 of them won by an agent candidate.
        self.assertEqual(status["agent_best_count"], 3)
        self.assertAlmostEqual(status["agent_best_rate"], 0.75)
        self.assertAlmostEqual(status["last50_agent_best_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
