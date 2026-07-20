"""MIXED Phase 1 pair-policy rollout, CE 학습, checkpoint 회귀시험."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    build_phase1_pair_candidates,
    run_phase1_pair_policy_rollout,
    train_phase1_pair_self_labeling,
)
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Utils.learning.phase_agent_checkpoints import load_phase1_pair_pointer_checkpoint
from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs


class Phase1PairSelfLabelingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episode = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=3,
            max_blocks=3,
            seed=20260715,
            verbose=False,
        )[0]
        cls.weights = joint_phase1_bay_capacity_weights()
        cls.bay_ids = tuple(cls.weights)

    def test_pair_candidates_use_single_19_plus_8_mixed_schema(self) -> None:
        candidates = build_phase1_pair_candidates(
            jobs=self.episode["jobs"],
            bay_ids=self.bay_ids,
            bay_capacity_weights=self.weights,
        )

        self.assertGreater(len(candidates), 0)
        self.assertEqual(len(PHASE1_PAIR_FEATURE_NAMES), 19)
        self.assertEqual(len(PHASE1_PAIR_ENV_FEATURE_NAMES), 8)
        self.assertEqual(
            PHASE1_PAIR_FEATURE_NAMES[-6:],
            [
                "projected_shared_wo_gap_ratio",
                "projected_series_wo_gap_ratio",
                "projected_shared_cut_gap_ratio",
                "projected_series_cut_gap_ratio",
                "projected_shared_bevel_gap_ratio",
                "projected_series_bevel_gap_ratio",
            ],
        )
        for candidate in candidates:
            self.assertEqual(candidate["feature_names"], PHASE1_PAIR_FEATURE_NAMES)
            self.assertEqual(len(candidate["features"]), 19)
            self.assertIn(candidate["bay_id"], self.bay_ids)

    def test_uniform_policy_rollout_assigns_every_block_once(self) -> None:
        rollout = run_phase1_pair_policy_rollout(
            jobs=self.episode["jobs"],
            bay_ids=self.bay_ids,
            model=None,
            temperature=1.0,
            seed=7,
            source="uniform_test",
            selection="sample",
            bay_capacity_weights=self.weights,
        )

        self.assertEqual(len(rollout.assignments), self.episode["block_count"])
        self.assertEqual(len(rollout.transitions), self.episode["block_count"])
        self.assertEqual(set(rollout.bay_loads), set(self.bay_ids))
        for transition in rollout.transitions:
            self.assertEqual(len(transition.env_features), 8)
            self.assertTrue(0 <= transition.selected_action_index < len(transition.candidate_ids))

    def test_one_episode_training_writes_only_mixed_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_phase1_pair_self_labeling(
                episode_jobs=[self.episode["jobs"]],
                episode_metadata=[self.episode["metadata"]],
                bay_ids=self.bay_ids,
                bay_capacity_weights=self.weights,
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("wo_first_balanced",),
                hidden_dim=8,
                checkpoint_every=1,
                validation_every=100,
                validation_episodes=0,
                seed=11,
            )
            checkpoint = Path(result["checkpoint_path"])
            loaded = load_phase1_pair_pointer_checkpoint(checkpoint)

            self.assertTrue(checkpoint.is_file())
            self.assertTrue((Path(temp_dir) / "metrics.csv").is_file())
            self.assertEqual(result["rule_profile"], "multi_series_260711")
            self.assertEqual(result["score_mode"], "wo_first")
            self.assertEqual(result["objective_scope"], "shared_and_series")
            self.assertEqual(result["episode_scope_version"], "joint_five_bay_v3_shared_pool")
            self.assertEqual(loaded.pair_feature_dim, 19)
            self.assertEqual(loaded.env_feature_dim, 8)
            self.assertEqual(loaded.objective_scope, "shared_and_series")

    def test_series_only_training_persists_three_score_checkpoint_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_metadata = dict(self.episode["metadata"])
            validation_metadata.update(
                {
                    "problem_id": "VAL_SERIES_ONLY",
                    "validation_source": "synthetic",
                }
            )
            result = train_phase1_pair_self_labeling(
                episode_jobs=[self.episode["jobs"]],
                episode_metadata=[self.episode["metadata"]],
                bay_ids=self.bay_ids,
                bay_capacity_weights=self.weights,
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("wo_first_balanced",),
                hidden_dim=8,
                checkpoint_every=1,
                validation_every=1,
                validation_episodes=1,
                validation_episode_factory=lambda _: {
                    "jobs": self.episode["jobs"],
                    "metadata": validation_metadata,
                },
                validation_rollout_samples=1,
                objective_scope="series_only",
                seed=13,
            )
            loaded = load_phase1_pair_pointer_checkpoint(result["checkpoint_path"])
            with (Path(temp_dir) / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                metric = next(csv.DictReader(file))

            self.assertEqual(result["objective_scope"], "series_only")
            self.assertEqual(result["score_field_names"], ["score_0", "score_1", "score_2"])
            self.assertEqual(loaded.objective_scope, "series_only")
            self.assertEqual(metric["objective_scope"], "series_only")
            self.assertEqual(len(json.loads(metric["score_json"])), 3)
            self.assertNotIn("validation_wo_gap_png", result)
            self.assertNotIn("validation_cut_gap_png", result)
            self.assertNotIn("validation_bevel_gap_png", result)
            for key in (
                "validation_series_wo_gap_png",
                "validation_series_cut_gap_png",
                "validation_series_bevel_gap_png",
            ):
                self.assertTrue(Path(result[key]).is_file())

    def test_resume_transfers_checkpoint_to_another_objective_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            validation_metadata = dict(self.episode["metadata"])
            validation_metadata.update(
                {
                    "problem_id": "VAL_SCOPE_TRANSFER",
                    "validation_source": "synthetic",
                }
            )
            first = train_phase1_pair_self_labeling(
                episode_jobs=[self.episode["jobs"]],
                episode_metadata=[self.episode["metadata"]],
                bay_ids=self.bay_ids,
                bay_capacity_weights=self.weights,
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("wo_first_balanced",),
                hidden_dim=8,
                checkpoint_every=1,
                validation_every=1,
                validation_episodes=1,
                validation_episode_factory=lambda _: {
                    "jobs": self.episode["jobs"],
                    "metadata": validation_metadata,
                },
                validation_rollout_samples=1,
                seed=17,
            )

            resumed = train_phase1_pair_self_labeling(
                episode_jobs=[self.episode["jobs"], self.episode["jobs"]],
                episode_metadata=[self.episode["metadata"], self.episode["metadata"]],
                bay_ids=self.bay_ids,
                bay_capacity_weights=self.weights,
                output_dir=temp_dir,
                episodes=2,
                rollout_samples=1,
                heuristic_algorithms=("wo_first_balanced",),
                hidden_dim=8,
                checkpoint_every=1,
                validation_every=100,
                validation_episodes=0,
                objective_scope="series_only",
                resume_checkpoint=first["checkpoint_path"],
                seed=17,
            )
            loaded = load_phase1_pair_pointer_checkpoint(resumed["checkpoint_path"])
            with (Path(temp_dir) / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                scopes = [row["objective_scope"] for row in csv.DictReader(file)]

            self.assertEqual(resumed["start_episode"], 2)
            self.assertTrue(resumed["objective_scope_transition"])
            self.assertEqual(resumed["previous_objective_scope"], "shared_and_series")
            self.assertEqual(resumed["best_validation_score"], [])
            self.assertEqual(loaded.objective_scope, "series_only")
            self.assertEqual(scopes, ["shared_and_series", "series_only"])

    def test_old_or_incomplete_checkpoint_is_rejected(self) -> None:
        model = Phase1PairPointerPolicy(
            pair_feature_dim=16,
            env_feature_dim=5,
            hidden_dim=8,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old_phase1.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pair_feature_names": PHASE1_PAIR_FEATURE_NAMES,
                    "env_feature_names": PHASE1_PAIR_ENV_FEATURE_NAMES,
                    "hidden_dim": 8,
                    "rule_profile": "legacy_np",
                    "score_mode": "steel_first",
                    "episode_scope_version": "legacy_three_bay",
                },
                path,
            )

            with self.assertRaises(RuntimeError):
                load_phase1_pair_pointer_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
