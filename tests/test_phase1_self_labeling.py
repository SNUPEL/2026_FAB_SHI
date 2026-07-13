"""Phase 1 self-labeling tests.

The Phase 1 self-label trainer must not be fixed-teacher imitation.
It should compare policy-sampled rollouts against heuristic candidates and
learn from the best sequence for the current problem.
"""

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

from Phase1.self_labeling import (
    PHASE1_SELF_LABEL_HEURISTIC_BANK,
    Phase1SelfLabelCandidate,
    _projected_bank_score,
    _select_best_candidate,
    _score_bay_loads,
    run_phase1_policy_rollout,
    train_phase1_pointer_self_labeling,
)
from Utils.phase1.phase1_bay_balancer import Phase1Block


class Phase1SelfLabelingTest(unittest.TestCase):
    """Pin down the requested SLIM-style Phase 1 learning behavior."""

    def test_select_best_candidate_prefers_objective_not_teacher_source(self) -> None:
        """An agent sample can beat LPT and then becomes the pseudo-label."""

        lpt = Phase1SelfLabelCandidate(
            source="lpt",
            transitions=[],
            assignments={"A": "22"},
            bay_loads={
                "22": self._load(steel=10, cut=100.0, bevel=3, long_cut=1),
                "23": self._load(steel=0, cut=0.0, bevel=0, long_cut=0),
            },
        )
        agent = Phase1SelfLabelCandidate(
            source="agent_sample_1",
            transitions=[],
            assignments={"A": "23"},
            bay_loads={
                "22": self._load(steel=5, cut=50.0, bevel=1, long_cut=0),
                "23": self._load(steel=5, cut=50.0, bevel=2, long_cut=0),
            },
        )

        best = _select_best_candidate([lpt, agent], score_mode="steel_first")

        self.assertEqual(best.source, "agent_sample_1")

    def test_policy_rollout_records_select_block_and_select_bay_transitions(self) -> None:
        """A sampled policy rollout should produce trainable pseudo-label steps."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }

        candidate = run_phase1_policy_rollout(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            model=None,
            temperature=1.0,
            seed=7,
            source="agent_sample_1",
        )

        self.assertEqual(candidate.source, "agent_sample_1")
        self.assertEqual(len(candidate.assignments), 2)
        self.assertEqual([step.phase for step in candidate.transitions[0:2]], ["SELECT_BLOCK", "SELECT_BAY"])
        self.assertIn(candidate.assignments["P1::A"], {"22", "23"})
        self.assertEqual(len(_score_bay_loads(candidate.bay_loads, "steel_first")), 3)

    def test_self_label_training_writes_inspectable_learning_data(self) -> None:
        """Training should leave candidate and best-action data for audit."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            "WO_C": self._job("WO_C", "P1::C", steel=3, cut_length=200, bevel_quantity=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pointer_self_labeling(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("steel_lpt_greedy_insertion",),
                hidden_dim=16,
                seed=1,
            )
            candidate_rows = self._read_csv_lines(summary["candidate_summary_csv"])
            best_rows = [
                json.loads(line)
                for line in Path(summary["best_action_table_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]
            manifest_exists = Path(summary["learning_data_manifest_json"]).exists()
            loss_curve_exists = Path(summary["loss_curve_png"]).exists()

        self.assertEqual(len(candidate_rows), 2)
        self.assertEqual(len(best_rows), 6)
        self.assertTrue(manifest_exists)
        self.assertTrue(loss_curve_exists)
        self.assertIn("selected_action_id", best_rows[0])

    def test_self_label_training_uses_variable_episode_jobs_and_eight_heuristics(self) -> None:
        """Each episode can train on a different generated Phase 1 problem."""

        episode_jobs = [
            {
                "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
                "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            },
            {
                "WO_C": self._job("WO_C", "P2::C", steel=3, cut_length=200, bevel_quantity=1),
                "WO_D": self._job("WO_D", "P2::D", steel=6, cut_length=1400, bevel_quantity=4),
                "WO_E": self._job("WO_E", "P2::E", steel=5, cut_length=700, bevel_quantity=3),
            },
        ]
        episode_metadata = [
            {"problem_id": "EP00001", "block_count": 2, "seed": 101},
            {"problem_id": "EP00002", "block_count": 3, "seed": 202},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pointer_self_labeling(
                jobs=None,
                episode_jobs=episode_jobs,
                episode_metadata=episode_metadata,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=2,
                rollout_samples=1,
                heuristic_algorithms=PHASE1_SELF_LABEL_HEURISTIC_BANK,
                hidden_dim=16,
                seed=1,
            )
            candidate_rows = self._read_csv_dicts(summary["candidate_summary_csv"])
            metric_rows = self._read_csv_dicts(summary["metrics_csv"])
            best_rows = [
                json.loads(line)
                for line in Path(summary["best_action_table_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual({row["problem_id"] for row in candidate_rows}, {"EP00001", "EP00002"})
        self.assertEqual({int(row["candidate_count"]) for row in metric_rows}, {9})
        self.assertEqual(len(candidate_rows), 18)
        self.assertEqual({row["block_count"] for row in candidate_rows}, {"2", "3"})
        self.assertEqual({row["problem_id"] for row in best_rows}, {"EP00001", "EP00002"})

    def test_lcp_uses_three_gap_score_after_hard_masking(self) -> None:
        """LCP no longer prepends Bay24 count; Bay24 avoidance is a candidate mask."""

        bay_loads = {
            "22": self._load(steel=40, cut=1000.0, bevel=10, long_cut=0),
            "23": self._load(steel=20, cut=1000.0, bevel=10, long_cut=0),
            "24": self._load(steel=20, cut=0.0, bevel=0, long_cut=0),
        }
        block = Phase1Block(
            block_set_id="P1::LONG",
            project_no="P1",
            block_no="LONG",
            job_ids=("WO_LONG",),
            wo_count=1,
            steel_quantity_sum=10,
            cut_length_sum=10.0,
            bevel_quantity_sum=1,
            long_cut_over_1000=1,
            allowed_bay_ids=("22", "23", "24"),
            length_avg=None,
            thickness_avg=None,
        )

        self.assertEqual(
            _projected_bank_score(bay_loads, "23", block, "long_cut_preferred"),
            (20.0, 1010.0, 11.0, "23"),
        )
        self.assertLess(
            _projected_bank_score(bay_loads, "24", block, "long_cut_preferred"),
            _projected_bank_score(bay_loads, "23", block, "long_cut_preferred"),
        )

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        steel: int,
        cut_length: float,
        bevel_quantity: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            steel_quantity=steel,
            cut_length=cut_length,
            bevel_quantity=bevel_quantity,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            extra={
                "source_project_no": block_set_id.split("::")[0],
                "source_block_no": block_set_id.split("::")[1],
                "source_wk_ord_no": job_id,
            },
        )

    @staticmethod
    def _load(steel: int, cut: float, bevel: int, long_cut: int) -> dict:
        return {
            "steel_quantity_sum": steel,
            "cut_length_sum": cut,
            "bevel_quantity_sum": bevel,
            "long_cut_bay24_count": long_cut,
            "wo_count": 1,
            "block_count": 1,
            "capacity_weight": 1.0,
        }

    @staticmethod
    def _read_csv_lines(path: str) -> list[str]:
        with Path(path).open("r", encoding="utf-8-sig") as file:
            return file.read().splitlines()[1:]

    @staticmethod
    def _read_csv_dicts(path: str) -> list[dict]:
        import csv

        with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))


if __name__ == "__main__":
    unittest.main()
