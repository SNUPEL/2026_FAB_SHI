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
    _merge_phase1_resource_pool_candidates,
    build_phase1_pair_candidates,
    run_phase1_pair_policy_resource_pool_best_of_k,
    run_phase1_pair_policy_rollout,
    split_phase1_jobs_by_resource_pool,
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

    def test_pair_candidates_use_single_20_plus_8_resource_pool_schema(self) -> None:
        candidates = build_phase1_pair_candidates(
            jobs=self.episode["jobs"],
            bay_ids=self.bay_ids,
            bay_capacity_weights=self.weights,
        )

        self.assertGreater(len(candidates), 0)
        self.assertEqual(len(PHASE1_PAIR_FEATURE_NAMES), 20)
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
            self.assertEqual(len(candidate["features"]), 20)
            self.assertIn(candidate["bay_id"], self.bay_ids)

    def test_jobs_are_split_into_two_independent_resource_pool_subproblems(self) -> None:
        jobs = self._four_series_jobs()

        subproblems = split_phase1_jobs_by_resource_pool(jobs)

        self.assertEqual(list(subproblems), ["NP_NC", "FN_FL"])
        self.assertEqual(
            {row["family"] for row in subproblems["NP_NC"].values()},
            {"NP", "NC"},
        )
        self.assertEqual(
            {row["family"] for row in subproblems["FN_FL"].values()},
            {"FN", "FL"},
        )
        self.assertEqual(
            set(subproblems["NP_NC"]) | set(subproblems["FN_FL"]),
            set(jobs),
        )

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

    def test_one_parent_episode_performs_one_update_per_present_resource_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = self._four_series_jobs()
            metadata = self._metadata("EP_RESOURCE_POOLS", seed=7)
            result = train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[metadata],
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
            with (Path(temp_dir) / "subproblem_metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                subproblem_rows = list(csv.DictReader(file))
            action_rows = [
                json.loads(line)
                for line in (Path(temp_dir) / "best_action_table.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            checkpoint_solution = (
                Path(temp_dir) / "checkpoints" / "solutions" / "episode_00001"
            )

            self.assertTrue(checkpoint.is_file())
            self.assertTrue((Path(temp_dir) / "metrics.csv").is_file())
            self.assertEqual(result["rule_profile"], "multi_series_260711")
            self.assertEqual(result["score_mode"], "wo_first")
            self.assertEqual(result["objective_scope"], "shared_and_series")
            self.assertEqual(result["episode_scope_version"], "resource_pool_subproblems_v1")
            self.assertEqual(result["optimizer_update_count"], 2)
            self.assertEqual(
                [row["subproblem_id"] for row in subproblem_rows],
                ["NP_NC", "FN_FL"],
            )
            self.assertEqual(loaded.pair_feature_dim, 20)
            self.assertEqual(loaded.env_feature_dim, 8)
            self.assertEqual(loaded.objective_scope, "shared_and_series")
            for role in ("teacher_best", "agent_best"):
                role_dir = checkpoint_solution / role
                self.assertTrue((role_dir / "phase1_block_bay_plan.json").is_file())
                self.assertTrue((role_dir / "phase1_block_assignments.csv").is_file())
                self.assertTrue((role_dir / "phase1_bay_loads.csv").is_file())
                plan = json.loads(
                    (role_dir / "phase1_block_bay_plan.json").read_text(encoding="utf-8")
                )
                self.assertEqual(plan["checkpoint_episode"], 1)
                self.assertEqual(plan["solution_role"], role)
                self.assertEqual(plan["summary"]["assignment_count"], 4)
            agent_plan = json.loads(
                (checkpoint_solution / "agent_best" / "phase1_block_bay_plan.json")
                .read_text(encoding="utf-8")
            )
            self.assertTrue(
                all(
                    source_part.split(":", 1)[1].startswith("agent_")
                    for source_part in agent_plan["algorithm"].split("|")
                )
            )
            expected_bays = {
                "NP_NC": {"22", "23", "24"},
                "FN_FL": {"25", "trans"},
            }
            for row in action_rows:
                candidate_bays = {
                    action_id.rsplit("@", 1)[1] for action_id in row["candidate_ids"]
                }
                self.assertTrue(candidate_bays <= expected_bays[row["subproblem_id"]])

    def test_absent_resource_pool_creates_no_update_or_zero_metric_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = {
                "WO_NP": self._job("WO_NP", "P1::NP::BLK_1", "NP", 100.0, 2),
                "WO_NC": self._job("WO_NC", "P2::NC::BLK_2", "NC", 120.0, 3),
            }
            metadata = self._metadata("EP_NP_NC_ONLY", seed=9)
            metadata["block_count"] = 2
            metadata["balancing_groups"] = ["NP", "NC"]
            result = train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[metadata],
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
                seed=9,
            )
            with (Path(temp_dir) / "subproblem_metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                subproblem_rows = list(csv.DictReader(file))
            with (Path(temp_dir) / "metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                parent_row = next(csv.DictReader(file))

            self.assertEqual(result["optimizer_update_count"], 1)
            self.assertEqual([row["subproblem_id"] for row in subproblem_rows], ["NP_NC"])
            self.assertEqual(parent_row["subproblem_count"], "1")

    def test_resource_pool_best_of_k_merges_complete_disjoint_assignments(self) -> None:
        jobs = self._four_series_jobs()

        candidate = run_phase1_pair_policy_resource_pool_best_of_k(
            jobs=jobs,
            bay_ids=self.bay_ids,
            model=None,
            sample_count=2,
            temperature=1.0,
            seed=19,
            bay_capacity_weights=self.weights,
            objective_scope="series_only",
        )

        self.assertEqual(set(candidate.assignments), {row["block_set_id"] for row in jobs.values()})
        self.assertEqual(len(candidate.transitions), 4)
        self.assertIn("NP_NC:", candidate.source)
        self.assertIn("FN_FL:", candidate.source)

    def test_resource_pool_merge_rejects_candidates_labeled_as_the_wrong_pool(self) -> None:
        jobs = self._four_series_jobs()
        subproblems = split_phase1_jobs_by_resource_pool(jobs)
        np_nc_candidate = run_phase1_pair_policy_rollout(
            jobs=subproblems["NP_NC"],
            bay_ids=self.bay_ids,
            model=None,
            temperature=1.0,
            seed=29,
            source="np_nc",
            selection="greedy",
            bay_capacity_weights=self.weights,
        )
        fn_fl_candidate = run_phase1_pair_policy_rollout(
            jobs=subproblems["FN_FL"],
            bay_ids=self.bay_ids,
            model=None,
            temperature=1.0,
            seed=31,
            source="fn_fl",
            selection="greedy",
            bay_capacity_weights=self.weights,
        )

        with self.assertRaisesRegex(RuntimeError, "assignment scope"):
            _merge_phase1_resource_pool_candidates(
                jobs=jobs,
                bay_ids=self.bay_ids,
                bay_capacity_weights=self.weights,
                selected=[
                    ("NP_NC", fn_fl_candidate),
                    ("FN_FL", np_nc_candidate),
                ],
            )

    def test_validation_writes_six_views_and_marks_absent_series_as_na(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = self._four_series_jobs()
            metadata = self._metadata("VAL_RESOURCE_POOLS", seed=23)
            metadata["validation_source"] = "synthetic"
            train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[metadata],
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
                validation_episode_factory=lambda _: {"jobs": jobs, "metadata": metadata},
                validation_rollout_samples=1,
                objective_scope="series_only",
                seed=23,
            )
            with (Path(temp_dir) / "validation_candidate_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(
                list(dict.fromkeys(row["validation_view"] for row in rows)),
                ["NP", "NC", "NP_NC", "FN", "FL", "FN_FL"],
            )
            np_row = next(row for row in rows if row["validation_view"] == "NP")
            mixed_row = next(row for row in rows if row["validation_view"] == "NP_NC")
            self.assertNotEqual(np_row["np_wo_gap"], "")
            self.assertEqual(np_row["nc_wo_gap"], "")
            self.assertNotEqual(mixed_row["np_wo_gap"], "")
            self.assertNotEqual(mixed_row["nc_wo_gap"], "")
            self.assertNotEqual(mixed_row["np_nc_wo_gap"], "")
            for file_name in (
                "validation_np_wo_gap.png",
                "validation_np_nc_wo_gap.png",
                "validation_fn_fl_bv_gap.png",
            ):
                self.assertTrue((Path(temp_dir) / file_name).is_file())

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

    def test_previous_joint_scope_checkpoint_is_rejected(self) -> None:
        model = Phase1PairPointerPolicy(
            pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
            env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
            hidden_dim=8,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "previous_scope.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pair_feature_names": PHASE1_PAIR_FEATURE_NAMES,
                    "env_feature_names": PHASE1_PAIR_ENV_FEATURE_NAMES,
                    "hidden_dim": 8,
                    "rule_profile": "multi_series_260711",
                    "score_mode": "wo_first",
                    "objective_scope": "series_only",
                    "episode_scope_version": "joint_five_bay_v3_shared_pool",
                },
                path,
            )

            with self.assertRaisesRegex(RuntimeError, "scope version mismatch"):
                load_phase1_pair_pointer_checkpoint(path)

    @classmethod
    def _four_series_jobs(cls) -> dict[str, dict]:
        return {
            "WO_NP": cls._job("WO_NP", "P1::NP::BLK_1", "NP", 100.0, 2),
            "WO_NC": cls._job("WO_NC", "P2::NC::BLK_2", "NC", 120.0, 3),
            "WO_FN": cls._job("WO_FN", "P3::FN::BLK_3", "FN", 140.0, 4),
            "WO_FL": cls._job("WO_FL", "P4::FL::BLK_4", "FL", 160.0, 5),
        }

    @classmethod
    def _metadata(cls, problem_id: str, *, seed: int) -> dict:
        return {
            "problem_id": problem_id,
            "block_count": 4,
            "seed": seed,
            "balancing_groups": ["NP", "NC", "FN", "FL"],
            "bay_ids": list(cls.bay_ids),
            "bay_capacity_weights": dict(cls.weights),
        }

    @staticmethod
    def _job(job_id: str, block_set_id: str, family: str, cut: float, bevel: int) -> dict:
        return {
            "job_id": job_id,
            "block_set_id": block_set_id,
            "family": family,
            "steel_quantity": 1,
            "cut_length": cut,
            "bevel_quantity": bevel,
            "plate_length": 10_000.0,
            "plate_width": 3_000.0,
            "thickness": 20.0,
            "processing_time": 30.0,
            "base_stage_minutes": {"cut": 30.0},
            "cut_bay": None,
            "source_cut_bay": None,
            "allowed_bay_ids": (),
            "allowed_machine_ids": (),
            "prohibited_machine_ids": (),
            "extra": {"source_wk_ord_no": job_id},
        }


if __name__ == "__main__":
    unittest.main()
