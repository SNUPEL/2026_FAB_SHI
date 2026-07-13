"""Phase 1 pair-action self-labeling tests."""

from types import SimpleNamespace
from pathlib import Path
import csv
import json
import tempfile
import unittest

import torch

from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    build_phase1_pair_candidates,
    run_phase1_pair_policy_rollout,
    train_phase1_pair_self_labeling,
    _candidate_learning_score,
    _collapse_agent_samples_for_validation_plot,
    _read_jsonl_rows,
    _score_bay_loads,
    _build_phase1_pair_candidates_from_cache,
    _build_phase1_pair_episode_cache,
    _pair_env_features,
    _sample_index,
)
from Phase1.self_labeling import PHASE1_SELF_LABEL_HEURISTIC_BANK, run_phase1_heuristic_candidate
from Utils.learning.phase_graph_mdp import PHASE1_BLOCK_BAY_EDGE_FEATURES
from Utils.phase1.phase1_bay_balancer import _add_block_load, _empty_phase1_bay_loads


class Phase1PairSelfLabelingTest(unittest.TestCase):
    """Pair MDP should score `(block, bay)` directly, not split the action."""

    def test_pair_candidates_use_graph_edge_features_and_hard_mask_long_cut_bay24(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }

        candidates = build_phase1_pair_candidates(jobs=jobs, bay_ids=["22", "23", "24"])

        self.assertEqual(len(candidates), 5)
        self.assertIn("P1::A@22", {candidate["action_id"] for candidate in candidates})
        self.assertNotIn("P1::A@24", {candidate["action_id"] for candidate in candidates})
        self.assertEqual(candidates[0]["feature_names"], PHASE1_BLOCK_BAY_EDGE_FEATURES)
        self.assertEqual(PHASE1_PAIR_FEATURE_NAMES, PHASE1_BLOCK_BAY_EDGE_FEATURES)
        self.assertNotIn("bay_22_flag", candidates[0]["feature_names"])
        self.assertNotIn("bay_23_flag", candidates[0]["feature_names"])
        self.assertNotIn("bay_24_flag", candidates[0]["feature_names"])
        self.assertIn("projected_steel_gap_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_cut_gap_ratio", candidates[0]["feature_names"])
        self.assertIn("projected_bevel_gap_ratio", candidates[0]["feature_names"])
        self.assertNotIn("projected_long_cut_bay24_count_ratio", candidates[0]["feature_names"])
        self.assertIn("bay_capacity_weight_ratio", candidates[0]["feature_names"])
        self.assertNotIn("current_bay22_steel_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertNotIn("current_bay23_cut_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertNotIn("current_bay24_bevel_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertIn("steel_gap_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertNotIn("long_cut_bay24_count_ratio", PHASE1_PAIR_ENV_FEATURE_NAMES)
        self.assertEqual(len(PHASE1_PAIR_FEATURE_NAMES), 10)
        self.assertEqual(len(PHASE1_PAIR_ENV_FEATURE_NAMES), 5)
        self.assertEqual(len(candidates[0]["features"]), len(candidates[0]["feature_names"]))

    def test_pair_candidates_keep_same_feature_dim_when_bay25_is_added(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=900, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }

        base = build_phase1_pair_candidates(jobs=jobs, bay_ids=["22", "23", "24"])
        expanded = build_phase1_pair_candidates(jobs=jobs, bay_ids=["22", "23", "24", "25"])

        self.assertEqual(len(base[0]["features"]), len(expanded[0]["features"]))
        self.assertIn("P1::A@25", {candidate["action_id"] for candidate in expanded})

    def test_pair_candidates_reject_missing_capacity_weight(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=900, bevel_quantity=2),
        }
        bay_loads = {
            bay_id: {
                "steel_quantity_sum": 0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
                "long_cut_bay24_count": 0,
                "wo_count": 0,
                "block_count": 0,
            }
            for bay_id in ("22", "23", "24")
        }

        with self.assertRaises(RuntimeError):
            build_phase1_pair_candidates(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                bay_loads=bay_loads,
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

    def test_common_planning_state_preserves_legacy_rollout_for_100_blocks(self) -> None:
        jobs = {
            f"WO_{index:03d}": self._job(
                f"WO_{index:03d}",
                f"P{index // 10}::B{index:03d}",
                steel=(index % 9) + 1,
                cut_length=float((index * 37) % 1400),
                bevel_quantity=index % 11,
            )
            for index in range(100)
        }
        bay_ids = ("22", "23", "24")
        weights = {"22": 4.0, "23": 3.0, "24": 4.0}
        seed = 20260710
        cache = _build_phase1_pair_episode_cache(
            jobs=jobs,
            bay_ids=bay_ids,
            long_cut_hard_mask=True,
            bay_capacity_weights=weights,
        )
        legacy_loads = _empty_phase1_bay_loads(cache.bay_ids, cache.bay_capacity_weights)
        remaining = set(cache.block_by_id)
        generator = torch.Generator().manual_seed(seed)
        legacy_steps = []
        legacy_assignments = {}
        for block_step in range(len(cache.blocks)):
            candidates = _build_phase1_pair_candidates_from_cache(
                cache=cache,
                bay_loads=legacy_loads,
                remaining_block_ids=sorted(remaining),
            )
            env_features = _pair_env_features(
                bay_loads=legacy_loads,
                assigned_block_count=block_step,
                total_block_count=len(cache.blocks),
                totals=cache.env_totals,
            )
            selected_index = _sample_index(
                logits=torch.zeros(len(candidates), dtype=torch.float32),
                temperature=1.0,
                generator=generator,
            )
            selected = candidates[selected_index]
            block_id = str(selected["block_set_id"])
            bay_id = str(selected["bay_id"])
            legacy_steps.append(
                (
                    [list(row["features"]) for row in candidates],
                    list(env_features),
                    selected_index,
                    str(selected["action_id"]),
                )
            )
            legacy_assignments[block_id] = bay_id
            _add_block_load(legacy_loads[bay_id], bay_id, cache.block_by_id[block_id])
            remaining.remove(block_id)

        migrated = run_phase1_pair_policy_rollout(
            jobs=jobs,
            bay_ids=bay_ids,
            model=None,
            temperature=1.0,
            seed=seed,
            source="migration_regression",
            selection="sample",
            bay_capacity_weights=weights,
        )

        self.assertEqual(migrated.assignments, legacy_assignments)
        self.assertEqual(migrated.bay_loads, legacy_loads)
        self.assertEqual(
            [
                (
                    transition.candidate_features,
                    transition.env_features,
                    transition.selected_action_index,
                    transition.selected_action_id,
                )
                for transition in migrated.transitions
            ],
            legacy_steps,
        )

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
            self.assertEqual(summary["bay_ids"], ["22", "23", "24"])
            self.assertEqual(summary["bay_capacity_weights"], {"22": 1.0, "23": 1.0, "24": 1.0})
            self.assertEqual(summary["score_field_names"], ["score_0", "score_1", "score_2"])
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
            self.assertTrue(Path(summary["validation_best_source_counts_png"]).exists())
            self.assertTrue((Path(temp_dir) / "checkpoints" / "phase1_pair_pointer_ep00001.pt").exists())
            self.assertTrue((Path(temp_dir) / "checkpoints" / "phase1_pair_pointer_ep00002.pt").exists())
            self.assertEqual(len(validation_rows), 4)
            self.assertEqual(len(validation_candidate_rows), 36)
            self.assertIn("agent_greedy", {row["source"] for row in validation_candidate_rows})
            self.assertIn("score_2", validation_candidate_rows[0])
            self.assertNotIn("score_3", validation_candidate_rows[0])
            self.assertEqual({row["validation_source"] for row in validation_rows}, {"synthetic"})
            self.assertIn("best_validation_score", summary)

    def test_validation_plot_rows_collapse_agent_samples_to_best_of_k(self) -> None:
        rows = [
            {
                "validation_source": "synthetic",
                "evaluation_input_type": "generated",
                "validation_episode": "1",
                "problem_id": "VAL00001",
                "source": "agent_greedy",
                "learning_score_json": "[3, 0, 0]",
                "score_0": "3",
            },
            {
                "validation_source": "synthetic",
                "evaluation_input_type": "generated",
                "validation_episode": "1",
                "problem_id": "VAL00001",
                "source": "agent_sample_2",
                "learning_score_json": "[1, 0, 0]",
                "score_0": "1",
            },
            {
                "validation_source": "synthetic",
                "evaluation_input_type": "generated",
                "validation_episode": "1",
                "problem_id": "VAL00001",
                "source": "steel_first_balanced",
                "learning_score_json": "[2, 0, 0]",
                "score_0": "2",
            },
        ]

        collapsed = _collapse_agent_samples_for_validation_plot(rows)

        self.assertNotIn("agent_greedy", {row["source"] for row in collapsed})
        self.assertNotIn("agent_sample_2", {row["source"] for row in collapsed})
        proposed = [row for row in collapsed if row["source"] == "proposed_best_of_k"]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0]["score_0"], "1")

    def test_actual_8days_validation_adds_actual_assignment_candidate(self) -> None:
        """ACTUAL_ validation must compare the historical Bay assignment as a method."""

        episode_jobs = [
            {
                "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=400, bevel_quantity=2),
                "WO_B": self._job("WO_B", "P1::B", steel=8, cut_length=700, bevel_quantity=5),
            }
        ]
        episode_metadata = [{"problem_id": "EP00001", "block_count": 2, "seed": 101}]

        def validation_episode_factory(_episode: int):
            return {
                "jobs": {
                    "VAL_A": self._job(
                        "VAL_A",
                        "ACT::A",
                        steel=10,
                        cut_length=400,
                        bevel_quantity=2,
                        source_cut_bay="22",
                    ),
                    "VAL_B": self._job(
                        "VAL_B",
                        "ACT::B",
                        steel=8,
                        cut_length=1200,
                        bevel_quantity=5,
                        source_cut_bay="24",
                    ),
                },
                "metadata": {
                    "problem_id": "ACTUAL_20260407",
                    "block_count": 2,
                    "seed": 900,
                    "validation_source": "actual_8days",
                    "evaluation_input_type": "candidate_workbook",
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=episode_jobs,
                episode_metadata=episode_metadata,
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                validation_rollout_samples=3,
                heuristic_algorithms=("steel_first_balanced",),
                hidden_dim=16,
                seed=1,
                validation_every=1,
                validation_episodes=1,
                validation_episode_factory=validation_episode_factory,
                bay_capacity_weights={"22": 4, "23": 3, "24": 4},
            )
            with Path(summary["validation_summary_csv"]).open(encoding="utf-8-sig") as file:
                validation_rows = list(csv.DictReader(file))
            with Path(summary["validation_candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                candidate_rows = list(csv.DictReader(file))

        actual_rows = [row for row in candidate_rows if row["source"] == "actual_assignment"]
        self.assertEqual(len(actual_rows), 1)
        self.assertEqual(summary["rollout_samples"], 1)
        self.assertEqual(summary["validation_rollout_samples"], 3)
        self.assertEqual(validation_rows[0]["candidate_count"], "5")
        self.assertIn("agent_sample_2", {row["source"] for row in candidate_rows})
        self.assertIn("agent_sample_3", {row["source"] for row in candidate_rows})
        self.assertEqual(actual_rows[0]["transition_count"], "0")
        self.assertEqual(actual_rows[0]["validation_source"], "actual_8days")
        bay_loads = json.loads(actual_rows[0]["bay_loads_json"])
        self.assertEqual(bay_loads["22"]["steel_quantity_sum"], 10)
        self.assertEqual(bay_loads["24"]["steel_quantity_sum"], 8)
        self.assertEqual(bay_loads["22"]["capacity_weight"], 4.0)
        self.assertEqual(bay_loads["24"]["capacity_weight"], 4.0)

    def test_split_actual_block_is_reported_without_entering_feedback_ranking(self) -> None:
        train_jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=1, cut_length=400, bevel_quantity=2),
        }

        def validation_episode_factory(_episode: int):
            return {
                "jobs": {
                    "VAL_A": self._job(
                        "VAL_A", "ACT::SPLIT", steel=1, cut_length=400,
                        bevel_quantity=2, source_cut_bay="22",
                    ),
                    "VAL_B": self._job(
                        "VAL_B", "ACT::SPLIT", steel=1, cut_length=700,
                        bevel_quantity=5, source_cut_bay="23",
                    ),
                },
                "metadata": {
                    "problem_id": "ACTUAL_20260424",
                    "block_count": 1,
                    "validation_source": "actual_8days",
                    "evaluation_input_type": "candidate_workbook_wo_expanded",
                },
            }

        def feedback_scorer(candidate, _jobs, _bay_ids):
            if candidate.source == "actual_assignment":
                raise AssertionError("split actual baseline must not enter Phase 2 feedback ranking")
            return (0.0,)

        contract = {
            "checkpoint": "/tmp/phase2.pt",
            "checkpoint_sha256": "a" * 64,
            "run_spec": {"run_spec_schema_version": "phase2_run_spec_v1"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=[train_jobs],
                episode_metadata=[{"problem_id": "EP00001", "block_count": 1, "seed": 101}],
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                validation_rollout_samples=1,
                heuristic_algorithms=("steel_first_balanced",),
                hidden_dim=16,
                validation_every=1,
                validation_episodes=1,
                validation_episode_factory=validation_episode_factory,
                phase2_feedback_scorer=feedback_scorer,
                phase2_feedback_contract=contract,
                bay_capacity_weights={"22": 4, "23": 3, "24": 4},
            )
            with Path(summary["validation_candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))

        actual = next(row for row in rows if row["source"] == "actual_assignment")
        self.assertEqual(actual["rank"], "")
        self.assertEqual(actual["comparison_only"], "1")
        self.assertEqual(actual["constraint_violation_count"], "1")
        self.assertEqual(actual["phase2_feedback_score_json"], "")
        loads = json.loads(actual["bay_loads_json"])
        self.assertEqual(loads["22"]["steel_quantity_sum"], 1)
        self.assertEqual(loads["23"]["steel_quantity_sum"], 1)

    def test_self_label_candidate_preserves_capacity_weight_for_scoring(self) -> None:
        """Validation scores must use per-capacity Bay load, not raw Bay totals."""

        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=40, cut_length=400, bevel_quantity=8),
            "WO_B": self._job("WO_B", "P1::B", steel=30, cut_length=300, bevel_quantity=6),
            "WO_C": self._job("WO_C", "P1::C", steel=40, cut_length=400, bevel_quantity=8),
        }

        candidate = run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm="steel_first_balanced",
            bay_capacity_weights={"22": 4, "23": 3, "24": 4},
        )

        self.assertEqual(candidate.bay_loads["22"]["capacity_weight"], 4.0)
        self.assertEqual(candidate.bay_loads["23"]["capacity_weight"], 3.0)
        self.assertEqual(candidate.bay_loads["24"]["capacity_weight"], 4.0)
        self.assertEqual(_score_bay_loads(candidate.bay_loads, "steel_first"), (0.0, 0.0, 0.0))

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

    def test_feedback_training_checkpoint_stores_exact_phase2_contract(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }
        contract = {
            "checkpoint": "/tmp/phase2.pt",
            "checkpoint_sha256": "a" * 64,
            "run_spec": {"run_spec_schema_version": "phase2_run_spec_v1"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[{"problem_id": "EP00001", "block_count": 2, "seed": 101}],
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("steel_first_balanced",),
                hidden_dim=16,
                checkpoint_every=1,
                phase2_feedback_scorer=lambda _candidate, _jobs, _bays: (0.0,),
                phase2_feedback_contract=contract,
            )
            checkpoint = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=True)

        self.assertEqual(checkpoint["phase2_feedback_contract"], contract)
        self.assertEqual(summary["phase2_feedback_contract"], contract)

    def test_feedback_training_resume_rejects_different_phase2_contract(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
            "WO_B": self._job("WO_B", "P1::B", steel=7, cut_length=400, bevel_quantity=8),
        }
        contract_a = {
            "checkpoint": "/tmp/phase2-a.pt",
            "checkpoint_sha256": "a" * 64,
            "run_spec": {"run_spec_schema_version": "phase2_run_spec_v1"},
        }
        contract_b = {
            **contract_a,
            "checkpoint": "/tmp/phase2-b.pt",
            "checkpoint_sha256": "b" * 64,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[{"problem_id": "EP00001", "block_count": 2, "seed": 101}],
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("steel_first_balanced",),
                hidden_dim=16,
                checkpoint_every=1,
                phase2_feedback_scorer=lambda _candidate, _jobs, _bays: (0.0,),
                phase2_feedback_contract=contract_a,
            )

            with self.assertRaises(RuntimeError):
                train_phase1_pair_self_labeling(
                    episode_jobs=[jobs, jobs],
                    episode_metadata=[
                        {"problem_id": "EP00001", "block_count": 2, "seed": 101},
                        {"problem_id": "EP00002", "block_count": 2, "seed": 102},
                    ],
                    bay_ids=["22", "23", "24"],
                    output_dir=temp_dir,
                    episodes=2,
                    rollout_samples=1,
                    heuristic_algorithms=("steel_first_balanced",),
                    hidden_dim=16,
                    checkpoint_every=1,
                    resume_checkpoint="latest",
                    phase2_feedback_scorer=lambda _candidate, _jobs, _bays: (0.0,),
                    phase2_feedback_contract=contract_b,
                )

    def test_feedback_scorer_without_contract_is_rejected(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut_length=1200, bevel_quantity=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaises(RuntimeError):
            train_phase1_pair_self_labeling(
                episode_jobs=[jobs],
                episode_metadata=[{"problem_id": "EP00001", "block_count": 1, "seed": 101}],
                bay_ids=["22", "23", "24"],
                output_dir=temp_dir,
                episodes=1,
                rollout_samples=1,
                heuristic_algorithms=("steel_first_balanced",),
                hidden_dim=16,
                phase2_feedback_scorer=lambda _candidate, _jobs, _bays: (0.0,),
            )

    def test_read_jsonl_rows_skips_corrupt_resume_audit_line(self) -> None:
        """Interrupted JSONL audit writes must not block checkpoint resume."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best_action_table.jsonl"
            path.write_text(
                '{"episode": 1, "step": 0}\n'
                '{"episode": 2, "step": \n'
                '\x00{"episode": 3, "step": 0}\n'
                '{"episode": 4, "step": 0}\n',
                encoding="utf-8",
            )

            rows = _read_jsonl_rows(path, "episode", start_episode=4)

        self.assertEqual([row["episode"] for row in rows], [1, 3])

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
        feedback_call_count = 0

        def phase2_feedback_scorer(candidate, _jobs, _bay_ids):
            nonlocal feedback_call_count
            feedback_call_count += 1
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
                phase2_feedback_contract={
                    "checkpoint": "/tmp/phase2.pt",
                    "checkpoint_sha256": "a" * 64,
                    "run_spec": {"run_spec_schema_version": "phase2_run_spec_v1"},
                },
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
        self.assertEqual(feedback_call_count, 3)

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        steel: int,
        cut_length: float,
        bevel_quantity: int,
        length: float = 10000.0,
        thickness: float = 13.0,
        source_cut_bay: str | None = None,
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
            source_cut_bay=source_cut_bay,
            allowed_bay_ids=(),
            extra={
                "source_project_no": block_set_id.split("::")[0],
                "source_block_no": block_set_id.split("::")[1],
                "source_wk_ord_no": job_id,
            },
        )


if __name__ == "__main__":
    unittest.main()
