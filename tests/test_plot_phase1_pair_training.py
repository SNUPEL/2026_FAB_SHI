"""Tests for Phase 1 pair training plot aggregation."""

import unittest

from scripts.plot_phase1_pair_training import _best_agent_rows_by_episode


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


if __name__ == "__main__":
    unittest.main()
