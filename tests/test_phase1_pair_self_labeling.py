"""Phase 1 pair-action self-labeling tests."""

from types import SimpleNamespace
from pathlib import Path
import csv
import json
import tempfile
import unittest

from Train.algorithm.phase1_pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    build_phase1_pair_candidates,
    run_phase1_pair_policy_rollout,
    train_phase1_pair_self_labeling,
    _candidate_learning_score,
)
from Train.algorithm.phase1_self_labeling import PHASE1_SELF_LABEL_HEURISTIC_BANK


class Phase1PairSelfLabelingTest(unittest.TestCase):
    """Pair MDP should score `(block, bay)` directly, not split the action."""

    def test_pair_candidates_use_gap_only_state_and_hard_mask_long_cut_bay24(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }

        candidates = build_phase1_pair_candidates(jobs=jobs, bay_ids=["22", "23", "24"])

        self.assertEqual(len(candidates), 5)
        self.assertIn("P1::A@22", {candidate["action_id"] for candidate in candidates})
        self.assertNotIn("P1::A@24", {candidate["action_id"] for candidate in candidates})
        self.assertNotIn("block_long_cut_flag", candidates[0]["feature_names"])
        self.assertNotIn("bay_block_ratio_before", candidates[0]["feature_names"])
        self.assertIn("projected_bay22_steel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay23_steel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay24_steel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay22_cut_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay23_cut_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay24_cut_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay22_bevel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay23_bevel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bay24_bevel_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_steel_gap", candidates[0]["feature_names"])
        self.assertIn("projected_cut_gap", candidates[0]["feature_names"])
        self.assertIn("projected_bevel_gap", candidates[0]["feature_names"])
        self.assertNotIn("projected_steel_absdev", candidates[0]["feature_names"])
        self.assertNotIn("projected_cut_absdev", candidates[0]["feature_names"])
        self.assertNotIn("projected_bevel_absdev", candidates[0]["feature_names"])
        self.assertNotIn("long_cut_to_bay24", candidates[0]["feature_names"])
        self.assertNotIn("projected_long_cut_bay24_ratio", candidates[0]["feature_names"])
        self.assertNotIn("block_length_ratio", candidates[0]["feature_names"])
        self.assertNotIn("block_thickness_ratio", candidates[0]["feature_names"])
        self.assertIn("bay_22_flag", candidates[0]["feature_names"])
        self.assertIn("bay_23_flag", candidates[0]["feature_names"])
        self.assertIn("bay_24_flag", candidates[0]["feature_names"])
        self.assertNotIn("block_wo_ratio", candidates[0]["feature_names"])
        self.assertNotIn("block_wo_ratio", PHASE1_PAIR_FEATURE_NAMES)
        self.assertNotIn("bay_id_ratio", candidates[0]["feature_names"])
        self.assertNotIn("bay_id_ratio", PHASE1_PAIR_FEATURE_NAMES)
        self.assertNotIn("assigned_block_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertNotIn("long_cut_bay24_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertIn("current_bay22_steel_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertIn("current_bay23_cut_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertIn("current_bay24_bevel_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertEqual(len(PHASE1_PAIR_FEATURE_NAMES), 18)
        self.assertEqual(len(PHASE1_PAIR_ENV_FEATURE_NAMES), 14)
        self.assertEqual(len(candidates[0]["features"]), len(candidates[0]["feature_names"]))
        bay24_candidate = next(candidate for candidate in candidates if candidate["action_id"] == "P1::B@24")
        feature_map = dict(zip(bay24_candidate["feature_names"], bay24_candidate["features"]))
        self.assertEqual(
            (feature_map["bay_22_flag"], feature_map["bay_23_flag"], feature_map["bay_24_flag"]),
            (0.0, 0.0, 1.0),
        )

    def test_policy_rollout_records_one_select_pair_transition_per_block(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            "WO_C": self._job("WO_C", "P1::C", steel=3, cut_length=200, bevel_quantity=1),
        }

        candidate = run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            model=None,
            temperature=1.0,
            seed=7,
            source="agent_sample_1",
        )

        self.assertEqual(len(candidate.assignments), 3)
        self.assertEqual([step.phase for step in candidate.transitions], ["SELECT_PAIR", "SELECT_PAIR", "SELECT_PAIR"])
        self.assertTrue(all("@" in step.selected_action_id for step in candidate.transitions))

    def test_pair_training_writes_auditable_sampled_learning_data(self) -> None:
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
            summary = train_phase1_pair_self_labeling(
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
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))
            with Path(summary["candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                candidates = list(csv.DictReader(file))
            best_rows = [
                json.loads(line)
                for line in Path(summary["best_action_table_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual({row["problem_id"] for row in metrics}, {"EP00001", "EP00002"})
        self.assertEqual({int(row["candidate_count"]) for row in metrics}, {9})
        self.assertEqual(len(candidates), 18)
        self.assertEqual([row["phase"] for row in best_rows[:2]], ["SELECT_PAIR", "SELECT_PAIR"])

    def test_pair_training_can_generate_episodes_on_the_fly(self) -> None:
        calls = []

        def episode_factory(episode: int):
            calls.append(episode)
            return {
                "jobs": {
                    f"WO_{episode}_A": self._job(f"WO_{episode}_A", f"P{episode}::A", steel=10, cut_length=1200, bevel_quantity=2),
                    f"WO_{episode}_B": self._job(f"WO_{episode}_B", f"P{episode}::B", steel=7, cut_length=400, bevel_quantity=8),
                },
                "metadata": {"problem_id": f"EP{episode:05d}", "block_count": 2, "seed": 100 + episode},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=None,
                episode_metadata=None,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=3,
                rollout_samples=1,
                heuristic_algorithms=PHASE1_SELF_LABEL_HEURISTIC_BANK,
                hidden_dim=16,
                seed=1,
                episode_factory=episode_factory,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))

        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(len(metrics), 3)
        self.assertEqual(summary["episode_mode"], "on_the_fly")

    def test_pair_training_writes_periodic_and_best_validation_checkpoints(self) -> None:
        episode_jobs = [
            {
                "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
                "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            },
            {
                "WO_C": self._job("WO_C", "P2::C", steel=3, cut_length=200, bevel_quantity=1),
                "WO_D": self._job("WO_D", "P2::D", steel=6, cut_length=1400, bevel_quantity=4),
            },
        ]
        episode_metadata = [
            {"problem_id": "EP00001", "block_count": 2, "seed": 101},
            {"problem_id": "EP00002", "block_count": 2, "seed": 202},
        ]

        def validation_episode_factory(episode: int):
            return {
                "jobs": {
                    f"VAL_{episode}_A": self._job(f"VAL_{episode}_A", f"VAL{episode}::A", steel=10, cut_length=1200, bevel_quantity=2),
                    f"VAL_{episode}_B": self._job(f"VAL_{episode}_B", f"VAL{episode}::B", steel=7, cut_length=400, bevel_quantity=8),
                },
                "metadata": {"problem_id": f"VAL{episode:05d}", "block_count": 2, "seed": 900 + episode},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=episode_jobs,
                episode_metadata=episode_metadata,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=2,
                rollout_samples=1,
                heuristic_algorithms=PHASE1_SELF_LABEL_HEURISTIC_BANK,
                hidden_dim=16,
                seed=1,
                checkpoint_every=1,
                validation_every=1,
                validation_episodes=2,
                validation_episode_factory=validation_episode_factory,
            )
            with Path(summary["validation_summary_csv"]).open(encoding="utf-8-sig") as file:
                validation_rows = list(csv.DictReader(file))
            with Path(summary["validation_candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                validation_candidate_rows = list(csv.DictReader(file))

            self.assertTrue(Path(summary["checkpoint_path"]).exists())
            self.assertTrue(Path(summary["best_checkpoint_path"]).exists())
            self.assertTrue(Path(summary["validation_candidate_summary_csv"]).exists())
            self.assertTrue(Path(summary["validation_steel_gap_png"]).exists())
            self.assertTrue(Path(summary["validation_cut_gap_png"]).exists())
            self.assertTrue(Path(summary["validation_bevel_gap_png"]).exists())
            self.assertTrue(Path(summary["validation_long_cut_png"]).exists())
            self.assertTrue(Path(summary["validation_best_source_counts_png"]).exists())
            self.assertTrue((Path(temp_dir) / "checkpoints" / "phase1_pair_pointer_ep00001.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoints" / "phase1_pair_pointer_ep00002.pt").exists())
            self.assertEqual(len(validation_rows), 4)
            self.assertEqual(len(validation_candidate_rows), 36)
            self.assertIn("agent_greedy", {row["source"] for row in validation_candidate_rows})
            self.assertIn("score_3", validation_candidate_rows[0])
            self.assertNotIn("score_4", validation_candidate_rows[0])
            self.assertEqual({row["validation_source"] for row in validation_rows}, {"synthetic"})
            self.assertIn("best_validation_score", summary)

    def test_pair_training_resumes_from_latest_checkpoint(self) -> None:
        episode_jobs = [
            {
                f"WO_{episode}_A": self._job(f"WO_{episode}_A", f"P{episode}::A", steel=10, cut_length=1200, bevel_quantity=2),
                f"WO_{episode}_B": self._job(f"WO_{episode}_B", f"P{episode}::B", steel=7, cut_length=400, bevel_quantity=8),
            }
            for episode in range(1, 5)
        ]
        episode_metadata = [
            {"problem_id": f"EP{episode:05d}", "block_count": 2, "seed": 100 + episode}
            for episode in range(1, 5)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase1_pair_self_labeling(
                episode_jobs=episode_jobs[:2],
                episode_metadata=episode_metadata[:2],
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=2,
                rollout_samples=1,
                heuristic_algorithms=PHASE1_SELF_LABEL_HEURISTIC_BANK,
                hidden_dim=16,
                seed=1,
                checkpoint_every=1,
            )
            summary = train_phase1_pair_self_labeling(
                episode_jobs=episode_jobs,
                episode_metadata=episode_metadata,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=4,
                rollout_samples=1,
                heuristic_algorithms=PHASE1_SELF_LABEL_HEURISTIC_BANK,
                hidden_dim=16,
                seed=1,
                checkpoint_every=1,
                resume_checkpoint="latest",
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))

        self.assertEqual(summary["start_episode"], 3)
        self.assertEqual(summary["resumed_from_episode"], 2)
        self.assertEqual([int(row["episode"]) for row in metrics], [1, 2, 3, 4])

    def test_pair_learning_score_can_prepend_phase2_feedback(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=500, bevel_quantity=2),
        }
        base_better_but_phase2_bad = run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            model=None,
            temperature=1.0,
            seed=1,
            source="bad_for_phase2",
        )
        base_worse_but_phase2_ok = run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            model=None,
            temperature=1.0,
            seed=2,
            source="ok_for_phase2",
        )

        def phase2_feedback_scorer(candidate, _jobs, _bay_ids):
            return (0,) if candidate.source == "ok_for_phase2" else (1,)

        self.assertLess(
            _candidate_learning_score(
                candidate=base_worse_but_phase2_ok,
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                score_mode="steel_first",
                phase2_feedback_scorer=phase2_feedback_scorer,
            ),
            _candidate_learning_score(
                candidate=base_better_but_phase2_bad,
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                score_mode="steel_first",
                phase2_feedback_scorer=phase2_feedback_scorer,
            ),
        )

    def test_pair_training_writes_phase2_feedback_learning_score_when_enabled(self) -> None:
        episode_jobs = [
            {
                "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=500, bevel_quantity=2),
                "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
            },
        ]
        episode_metadata = [{"problem_id": "EP00001", "block_count": 2, "seed": 101}]

        def phase2_feedback_scorer(candidate, _jobs, _bay_ids):
            return (0,) if candidate.source == "bevel_first_balanced" else (1,)

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=episode_jobs,
                episode_metadata=episode_metadata,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("steel_first_balanced", "bevel_first_balanced"),
                hidden_dim=16,
                seed=1,
                phase2_feedback_scorer=phase2_feedback_scorer,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))
            with Path(summary["candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                candidates = list(csv.DictReader(file))

        self.assertEqual(metrics[0]["best_source"], "bevel_first_balanced")
        self.assertEqual(json.loads(metrics[0]["phase2_feedback_score_json"]), [0])
        self.assertIn("phase2_feedback_score_json", candidates[0])
        self.assertEqual(summary["phase2_feedback_score_enabled"], True)
        self.assertEqual(summary["phase2_feedback_score_length"], 1)

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        steel: int,
        cut_length: float,
        bevel_quantity: int,
        length: float = 10000.0,
        thickness: float = 13.0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            steel_quantity=steel,
            cut_length=cut_length,
            bevel_quantity=bevel_quantity,
            plate_length=length,
            thickness=thickness,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            extra={
                "source_project_no": block_set_id.split("::")[0],
                "source_block_no": block_set_id.split("::")[1],
                "source_wk_ord_no": job_id,
            },
        )


if __name__ == "__main__":
    unittest.main()
