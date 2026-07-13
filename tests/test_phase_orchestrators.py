"""Integration tests for Phase 1/Phase 2 orchestrators.

The orchestrators must reuse existing graph, heuristic, communication, and
scenario contracts instead of becoming another isolated implementation.
"""

from types import SimpleNamespace
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from Environment.constraints.profiles import default_phase2_constraint_profile
from Phase1.orchestrator import run_phase1_graph_workflow
from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    _build_phase1_pair_candidates_from_cache,
    _build_phase1_pair_episode_cache,
    build_phase1_pair_candidates,
)
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Phase2.orchestrator import (
    _job_processing_time,
    run_phase2_full_graph_workflow,
    run_phase2_graph_workflow,
    write_phase2_workflow_outputs,
)
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase2.run_spec import build_phase2_run_spec
from Phase2.state import PHASE2_STATE_SCHEMA_VERSION, phase2_state_feature_schema
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
    _build_phase2_common_environment,
    build_phase2_batch_machine_candidate_bank,
    run_phase2_batch_machine_candidate,
    train_phase2_batch_machine_self_labeling,
)
from Phase1.self_labeling import (
    PHASE1_SELF_LABEL_HEURISTIC_BANK,
    _score_bay_loads,
    run_phase1_heuristic_candidate,
)
from Utils.phase1.phase1_bay_balancer import _add_block_load, _empty_phase1_bay_loads
from Utils.learning.phase_agent_checkpoints import (
    load_phase2_checkpoint_run_spec,
    load_phase2_set_pointer_checkpoint,
)
from main import _phase1_agent_assignment_builder


class PhaseOrchestratorTest(unittest.TestCase):
    """End-to-end slices across existing Phase 1/2 modules."""

    def test_phase1_workflow_builds_graph_candidates_and_best_plan(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }

        result = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23", "24", "25"],
            heuristic_algorithms=("steel_first_balanced", "cut_first_balanced"),
        )

        self.assertEqual(result["graph"]["metadata"]["block_count"], 2)
        self.assertEqual(result["candidate_count"], 2)
        self.assertIn(result["best_source"], {"steel_first_balanced", "cut_first_balanced"})
        self.assertEqual(len(result["plan"]["assignments"]), 2)
        self.assertIn("score", result["best_candidate"])

    def test_phase1_workflow_preserves_existing_heuristic_results(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::C", steel=8, cut=1200.0, bevel=9),
            "WO_D": self._job("WO_D", "P1::D", steel=4, cut=500.0, bevel=0),
        }
        bay_ids = ["22", "23", "24"]

        for algorithm in PHASE1_SELF_LABEL_HEURISTIC_BANK:
            with self.subTest(algorithm=algorithm):
                expected = run_phase1_heuristic_candidate(
                    jobs=jobs,
                    bay_ids=bay_ids,
                    algorithm=algorithm,
                    long_cut_hard_mask=True,
                )
                result = run_phase1_graph_workflow(
                    jobs=jobs,
                    bay_ids=bay_ids,
                    heuristic_algorithms=(algorithm,),
                    long_cut_hard_mask=True,
                )
                actual_assignments = {
                    row["block_set_id"]: row["assigned_bay"]
                    for row in result["plan"]["assignments"]
                }

                self.assertEqual(actual_assignments, expected.assignments)
                self.assertEqual(result["plan"]["bay_loads"], expected.bay_loads)
                self.assertEqual(result["plan"]["score"], list(_score_bay_loads(expected.bay_loads, "steel_first")))

    def test_phase1_pair_cache_preserves_candidate_features(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::C", steel=8, cut=1200.0, bevel=9),
        }
        bay_ids = ["22", "23", "24"]
        bay_capacity_weights = {"22": 4, "23": 3, "24": 4}
        cache = _build_phase1_pair_episode_cache(
            jobs=jobs,
            bay_ids=bay_ids,
            long_cut_hard_mask=True,
            bay_capacity_weights=bay_capacity_weights,
        )

        slow_initial = build_phase1_pair_candidates(
            jobs=jobs,
            bay_ids=bay_ids,
            bay_capacity_weights=bay_capacity_weights,
        )
        fast_initial = _build_phase1_pair_candidates_from_cache(
            cache=cache,
            bay_loads=_empty_phase1_bay_loads(cache.bay_ids, cache.bay_capacity_weights),
            remaining_block_ids=sorted(cache.block_by_id),
        )
        self.assertEqual(fast_initial, slow_initial)

        bay_loads = _empty_phase1_bay_loads(cache.bay_ids, cache.bay_capacity_weights)
        _add_block_load(bay_loads["22"], "22", cache.block_by_id["P1::A"])
        remaining = sorted(set(cache.block_by_id) - {"P1::A"})
        slow_after_one = build_phase1_pair_candidates(
            jobs=jobs,
            bay_ids=bay_ids,
            bay_loads=bay_loads,
            remaining_block_ids=remaining,
            assigned_block_count=1,
            bay_capacity_weights=bay_capacity_weights,
        )
        fast_after_one = _build_phase1_pair_candidates_from_cache(
            cache=cache,
            bay_loads=bay_loads,
            remaining_block_ids=remaining,
        )
        self.assertEqual(fast_after_one, slow_after_one)

    def test_phase2_workflow_consumes_phase1_plan_and_builds_machine_graph(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("steel_first_balanced",),
        )
        scenario = {
            "jobs": [self._job_dict(job) for job in jobs.values()],
            "machines": [
                self._machine_dict("PLS21", "22"),
                self._machine_dict("PLS22", "22"),
                self._machine_dict("PLS31", "23"),
            ],
        }

        phase2 = run_phase2_graph_workflow(
            scenario=scenario,
            phase1_plan=phase1["plan"],
            assignment_mode="allowed_bay_ids",
        )

        assigned_bays = {
            row["block_set_id"]: row["assigned_bay"]
            for row in phase1["plan"]["assignments"]
        }
        edge_bays = {
            edge["source_block_id"]: edge["target_bay_id"]
            for edge in phase2["graph"]["candidate_edges"]
        }

        self.assertEqual(phase2["phase1_message_count"], 2)
        self.assertEqual(phase2["phase2_feedback_count"], 2)
        self.assertEqual(phase2["applied_summary"]["assigned_job_count"], 2)
        self.assertTrue(edge_bays)
        self.assertTrue(all(edge_bays[block_id] == assigned_bay for block_id, assigned_bay in assigned_bays.items()))

    def test_phase2_merged_candidate_uses_graph_feasibility_and_builds_timeline(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("steel_first_balanced",),
        )
        phase2 = run_phase2_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [self._machine_dict("PLS21", "22"), self._machine_dict("PLS31", "23")],
            },
            phase1_plan=phase1["plan"],
        )
        candidate = run_phase2_batch_machine_candidate(
            jobs={
                row["job_id"]: row
                for row in phase2["applied_scenario"]["jobs"]
            },
            machines={
                row["machine_id"]: row
                for row in phase2["applied_scenario"]["machines"]
            },
            phase1_assignments={
                message["block_set_id"]: message["assigned_bay"]
                for message in phase2["phase1_messages"]
            },
            source="min_makespan",
            model=None,
        )

        self.assertEqual(set(candidate.machine_assignments), {"WO_A", "WO_B"})
        self.assertEqual(sum(batch["wo_count"] for batch in candidate.batches), 2)
        self.assertEqual(len(candidate.timeline), len(candidate.batches))

    def test_phase2_merged_candidate_selects_batch_and_machine_without_prior_assignment(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=1),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=9),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }
        jobs["WO_A"].base_stage_minutes = {"cut": 40.0}
        jobs["WO_B"].base_stage_minutes = {"cut": 10.0}
        jobs["WO_C"].base_stage_minutes = {"cut": 20.0}

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
            },
            phase1_assignments={"P1::A": "22"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
        )

        self.assertEqual(PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES[0], "hard_violation_count")
        self.assertEqual(PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES[1], "makespan")
        self.assertEqual(candidate.score_tuple[0], 0)
        self.assertEqual(set(candidate.machine_assignments), {"WO_A", "WO_B", "WO_C"})
        self.assertTrue(all(1 <= batch["wo_count"] <= 3 for batch in candidate.batches))
        self.assertTrue(all(batch["length_sum"] <= 55000.0 for batch in candidate.batches))
        self.assertEqual(candidate.transitions[0].action_type, "select_machine")
        self.assertFalse(candidate.transitions[0].selected_job_ids)
        self.assertEqual(candidate.transitions[1].action_type, "select_wo")
        self.assertTrue(candidate.transitions[1].selected_job_ids)
        self.assertEqual(
            len([row for row in candidate.event_log if row["event_type"] == "PROCESS_START"]),
            len(jobs),
        )
        self.assertEqual(
            len([row for row in candidate.event_log if row["event_type"] == "PROCESS_FINISH"]),
            len(jobs),
        )
        timeline_by_batch = {row["batch_id"]: row for row in candidate.timeline}
        for event in candidate.event_log:
            if event["event_type"] not in {"PROCESS_START", "PROCESS_FINISH"}:
                continue
            batch_id = event["payload"]["batch_id"]
            expected_time = (
                timeline_by_batch[batch_id]["start_time"]
                if event["event_type"] == "PROCESS_START"
                else timeline_by_batch[batch_id]["finish_time"]
            )
            self.assertEqual(event["time_min"], expected_time)

    def test_phase2_merged_candidate_uses_machine_count_aware_batch_size(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=1),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=9),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
            "WO_D": self._job("WO_D", "P1::A", steel=7, cut=400.0, bevel=4),
        }

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
                "PLS23": self._machine_dict("PLS23", "22"),
            },
            phase1_assignments={"P1::A": "22"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
        )

        self.assertEqual([batch["wo_count"] for batch in candidate.batches], [2, 1, 1])
        self.assertEqual(sum(batch["wo_count"] for batch in candidate.batches), 4)

    def test_phase2_merged_candidate_uses_remaining_machine_slots_for_eight_jobs(self) -> None:
        jobs = {
            f"WO_{index}": self._job(f"WO_{index}", "P1::A", steel=index + 1, cut=100.0 + index, bevel=index % 3)
            for index in range(8)
        }

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
                "PLS23": self._machine_dict("PLS23", "22"),
            },
            phase1_assignments={"P1::A": "22"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
        )

        self.assertEqual([batch["wo_count"] for batch in candidate.batches], [3, 3, 2])
        self.assertEqual(sum(batch["wo_count"] for batch in candidate.batches), 8)

    def test_phase2_merged_batch_size_is_calculated_per_bay(self) -> None:
        jobs = {
            f"WO_A{index}": self._job(f"WO_A{index}", "P1::A", steel=index + 1, cut=100.0 + index, bevel=index % 3)
            for index in range(8)
        }
        jobs["WO_B0"] = self._job("WO_B0", "P2::B", steel=1, cut=50.0, bevel=0)

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
                "PLS23": self._machine_dict("PLS23", "22"),
                "PLS31": self._machine_dict("PLS31", "23"),
                "PLS32": self._machine_dict("PLS32", "23"),
                "PLS33": self._machine_dict("PLS33", "23"),
                "PLS34": self._machine_dict("PLS34", "23"),
            },
            phase1_assignments={"P1::A": "22", "P2::B": "23"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
        )

        self.assertEqual(candidate.batches[0]["wo_count"], 3)
        self.assertEqual(sum(batch["wo_count"] for batch in candidate.batches), 9)

    def test_phase2_merged_candidate_solves_each_bay_as_subproblem(self) -> None:
        jobs = {
            f"WO_A{index}": self._job(f"WO_A{index}", "P1::A", steel=index + 1, cut=100.0 + index, bevel=index % 3)
            for index in range(4)
        }
        jobs.update(
            {
                f"WO_B{index}": self._job(f"WO_B{index}", "P2::B", steel=index + 1, cut=200.0 + index, bevel=index % 2)
                for index in range(4)
            }
        )

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
                "PLS31": self._machine_dict("PLS31", "23"),
                "PLS32": self._machine_dict("PLS32", "23"),
            },
            phase1_assignments={"P1::A": "22", "P2::B": "23"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
        )

        self.assertEqual(len(candidate.transitions[0].policy_state.action_ids), 2)
        self.assertIn(candidate.transitions[0].selected_machine_id, {"PLS21", "PLS22"})
        self.assertEqual(sum(batch["wo_count"] for batch in candidate.batches), 8)

    def test_phase2_merged_candidate_bank_contains_heuristics_and_agent_samples(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }
        jobs["WO_A"].plate_length = 20000.0
        jobs["WO_B"].plate_length = 18000.0
        jobs["WO_C"].plate_length = 17000.0
        jobs["WO_C"].base_stage_minutes = {"cut": 30.0}
        model = Phase2SetPointerPolicy(hidden_dim=8)

        candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
            },
            phase1_assignments={"P1::A": "22"},
            model=model,
            heuristic_algorithms=("spt_batch", "lpt_batch", "best_fit_lth"),
            rollout_samples=2,
            seed=11,
        )

        sources = {candidate.source for candidate in candidates}
        self.assertEqual(
            sources,
            {"spt_batch", "lpt_batch", "best_fit_lth", "agent_greedy", "agent_sample_1", "agent_sample_2"},
        )
        self.assertTrue(all(candidate.score_tuple[0] == 0 for candidate in candidates))
        self.assertTrue(all(sum(batch["wo_count"] for batch in candidate.batches) == 3 for candidate in candidates))

    def test_phase2_candidate_bank_forks_one_pristine_post_phase1_environment(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
        }
        phase1_assignments = {"P1::A": "22"}
        profile = default_phase2_constraint_profile()
        environment = _build_phase2_common_environment(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            constraint_profile=profile,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        initial = environment.snapshot()

        candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            model=Phase2SetPointerPolicy(hidden_dim=8),
            heuristic_algorithms=("spt_batch", "lpt_batch"),
            rollout_samples=1,
            constraint_profile=profile,
            common_environment=environment,
            seed=17,
        )

        self.assertEqual(environment.state, initial.state)
        self.assertTrue(all(candidate.event_log for candidate in candidates))
        first_states = [candidate.transitions[0].policy_state for candidate in candidates]
        self.assertTrue(all(state.bay_context_features == first_states[0].bay_context_features for state in first_states))

    def test_phase2_heuristic_only_bank_uses_the_same_pristine_environment(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
        }
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
        }
        phase1_assignments = {"P1::A": "22"}
        profile = default_phase2_constraint_profile()
        environment = _build_phase2_common_environment(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            constraint_profile=profile,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        initial = environment.snapshot()

        candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            model=None,
            heuristic_algorithms=("spt_batch", "lpt_batch"),
            rollout_samples=0,
            constraint_profile=profile,
            common_environment=environment,
        )

        self.assertEqual(environment.state, initial.state)
        self.assertEqual([candidate.source for candidate in candidates], ["spt_batch", "lpt_batch"])

    def test_phase2_merged_action_pool_none_uses_linear_feasible_jobs(self) -> None:
        jobs = {
            f"WO_{index}": self._job(f"WO_{index}", "P1::A", steel=index + 1, cut=100.0 + index, bevel=index % 3)
            for index in range(9)
        }

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={"PLS21": self._machine_dict("PLS21", "22")},
            phase1_assignments={"P1::A": "22"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
            action_pool_limit=None,
        )

        self.assertEqual(candidate.transitions[0].action_type, "select_machine")
        self.assertEqual(len(candidate.transitions[0].policy_state.action_ids), 1)
        self.assertEqual(candidate.transitions[1].action_type, "select_wo")
        self.assertEqual(len(candidate.transitions[1].policy_state.action_ids), 9)
        self.assertEqual([batch["wo_count"] for batch in candidate.batches], [3, 3, 3])

    def test_phase2_merged_candidate_bank_accepts_lookahead_min_makespan(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }
        model = Phase2SetPointerPolicy(hidden_dim=8)

        candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines={
                "PLS21": self._machine_dict("PLS21", "22"),
                "PLS22": self._machine_dict("PLS22", "22"),
            },
            phase1_assignments={"P1::A": "22"},
            model=model,
            heuristic_algorithms=("lookahead_min_makespan",),
            rollout_samples=0,
            seed=11,
        )

        self.assertIn("lookahead_min_makespan", {candidate.source for candidate in candidates})

    def test_phase2_merged_workload_makespan_dispatch_starts_batch_with_long_tact_job(self) -> None:
        jobs = {
            "WO_LONG": self._job("WO_LONG", "P1::A", steel=8, cut=800.0, bevel=2),
            "WO_MID": self._job("WO_MID", "P1::A", steel=7, cut=700.0, bevel=2),
            "WO_SHORT": self._job("WO_SHORT", "P1::A", steel=2, cut=200.0, bevel=1),
        }
        jobs["WO_LONG"].base_stage_minutes = {"cut": 90.0}
        jobs["WO_MID"].base_stage_minutes = {"cut": 40.0}
        jobs["WO_SHORT"].base_stage_minutes = {"cut": 10.0}

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines={"PLS21": self._machine_dict("PLS21", "22")},
            phase1_assignments={"P1::A": "22"},
            source="workload_makespan_dispatch",
            model=None,
            max_wo_count=3,
            max_length_sum=55000.0,
            action_pool_limit=None,
        )

        first_wo_transition = next(
            transition for transition in candidate.transitions if transition.action_type == "select_wo"
        )
        self.assertEqual(first_wo_transition.selected_job_ids, ("WO_LONG",))
        self.assertEqual(candidate.score_tuple[0], 0)
        self.assertEqual([batch["wo_count"] for batch in candidate.batches], [3])

    def test_phase2_merged_dispatch_cache_preserves_fixed_candidate_results(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=9, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=4, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=5, cut=500.0, bevel=2),
            "WO_D": self._job("WO_D", "P1::A", steel=8, cut=700.0, bevel=4),
            "WO_E": self._job("WO_E", "P1::A", steel=3, cut=200.0, bevel=1),
            "WO_F": self._job("WO_F", "P1::A", steel=6, cut=600.0, bevel=0),
        }
        for job_id, minutes in {
            "WO_A": 90.0,
            "WO_B": 40.0,
            "WO_C": 10.0,
            "WO_D": 80.0,
            "WO_E": 30.0,
            "WO_F": 20.0,
        }.items():
            jobs[job_id].base_stage_minutes = {"cut": minutes}
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
        }
        expected = {
            "lpt_batch": [
                ("PLS21", ("WO_A", "WO_D", "WO_B"), 90.0),
                ("PLS22", ("WO_E", "WO_F", "WO_C"), 30.0),
            ],
            "workload_makespan_dispatch": [
                ("PLS21", ("WO_A", "WO_D", "WO_B"), 90.0),
                ("PLS22", ("WO_E", "WO_F", "WO_C"), 30.0),
            ],
            "lookahead_min_makespan": [
                ("PLS21", ("WO_C", "WO_F", "WO_E"), 30.0),
                ("PLS22", ("WO_B", "WO_D", "WO_A"), 90.0),
            ],
        }

        for source, expected_batches in expected.items():
            with self.subTest(source=source):
                candidate = run_phase2_batch_machine_candidate(
                    jobs=jobs,
                    machines=machines,
                    phase1_assignments={"P1::A": "22"},
                    source=source,
                    model=None,
                    max_wo_count=3,
                    max_length_sum=55000.0,
                    action_pool_limit=None,
                )
                actual_batches = [
                    (row["machine_id"], row["job_ids"], row["batch_duration"])
                    for row in candidate.batches
                ]
                self.assertEqual(candidate.score_tuple, (0, 90.0, 600.0, 0.0, 5.0, 60.0))
                self.assertEqual(actual_batches, expected_batches)

    def test_phase2_merged_training_writes_metrics_and_checkpoint(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                },
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch", "lpt_batch"),
                rollout_samples=1,
                seed=5,
            )
            metrics_exists = Path(summary["metrics_csv"]).exists()
            checkpoint_exists = Path(summary["checkpoint_path"]).exists()
            candidate_summary_exists = Path(summary["candidate_summary_csv"]).exists()
            best_timeline_exists = Path(summary["best_timeline_csv"]).exists()

        self.assertTrue(metrics_exists)
        self.assertTrue(checkpoint_exists)
        self.assertFalse(summary["write_candidate_summary"])
        self.assertFalse(candidate_summary_exists)
        self.assertTrue(best_timeline_exists)

    def test_phase2_merged_training_writes_periodic_checkpoints(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                },
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=2,
                hidden_dim=8,
                heuristic_algorithms=("lookahead_min_makespan", "spt_batch"),
                rollout_samples=1,
                action_pool_limit=None,
                checkpoint_every=1,
                seed=5,
            )
            checkpoint_dir = Path(summary["checkpoint_dir"])
            ep1_exists = (checkpoint_dir / "phase2_batch_machine_policy_ep00001.pt").exists()
            ep2_exists = (checkpoint_dir / "phase2_batch_machine_policy_ep00002.pt").exists()

        self.assertEqual(summary["checkpoint_every"], 1)
        self.assertIsNone(summary["action_pool_limit"])
        self.assertEqual(summary["heuristic_algorithms"], ["lookahead_min_makespan", "spt_batch"])
        self.assertTrue(ep1_exists)
        self.assertTrue(ep2_exists)

    def test_phase2_merged_training_generates_episode_jobs_on_demand(self) -> None:
        call_counts = {"train": 0, "validation": 0}

        def make_jobs(prefix: str) -> dict:
            return {
                f"{prefix}_A": self._job(f"{prefix}_A", f"{prefix}::A", steel=10, cut=900.0, bevel=3),
                f"{prefix}_B": self._job(f"{prefix}_B", f"{prefix}::A", steel=5, cut=300.0, bevel=1),
            }

        def episode_factory(episode: int) -> dict:
            call_counts["train"] += 1
            return make_jobs(f"EP{episode}")

        def validation_factory(validation_episode: int) -> dict:
            call_counts["validation"] += 1
            return make_jobs(f"VAL{validation_episode}")

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase2_batch_machine_self_labeling(
                jobs={},
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                },
                phase1_assignments={},
                output_dir=temp_dir,
                episodes=2,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=1,
                validation_every=1,
                validation_episodes=1,
                validation_rollout_samples=1,
                episode_job_factory=episode_factory,
                validation_episode_job_factory=validation_factory,
                phase1_heuristic="bevel_first_balanced",
                phase1_bay_ids=("22",),
                seed=5,
            )

        self.assertEqual(call_counts, {"train": 2, "validation": 2})

    def test_phase2_merged_training_writes_validation_with_separate_sampling(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                },
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=1,
                validation_every=1,
                validation_episodes=1,
                validation_rollout_samples=3,
                seed=5,
            )
            with Path(summary["validation_summary_csv"]).open(encoding="utf-8-sig") as file:
                validation_rows = list(csv.DictReader(file))
            with Path(summary["validation_candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                validation_candidate_rows = list(csv.DictReader(file))
            validation_makespan_png_exists = Path(summary["validation_makespan_png"]).exists()
            validation_best_counts_png_exists = Path(summary["validation_best_source_counts_png"]).exists()
            validation_policy_rank_png_exists = Path(summary["validation_policy_rank_png"]).exists()

        self.assertEqual(summary["rollout_samples"], 1)
        self.assertEqual(summary["validation_rollout_samples"], 3)
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(len(validation_rows), 1)
        self.assertEqual(validation_rows[0]["candidate_count"], "5")
        self.assertEqual(
            {row["source"] for row in validation_candidate_rows},
            {"spt_batch", "agent_greedy", "agent_sample_1", "agent_sample_2", "agent_sample_3"},
        )
        self.assertIn("proposed_best_rank", validation_rows[0])
        self.assertIn("greedy_rank", validation_rows[0])
        self.assertIn("proposed_makespan", validation_rows[0])
        self.assertIn("greedy_makespan", validation_rows[0])
        self.assertTrue(validation_makespan_png_exists)
        self.assertTrue(validation_best_counts_png_exists)
        self.assertTrue(validation_policy_rank_png_exists)

    def test_phase2_merged_training_overfits_fixed_small_problem(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::A", steel=8, cut=500.0, bevel=2),
            "WO_D": self._job("WO_D", "P1::A", steel=4, cut=250.0, bevel=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                },
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=12,
                lr=0.05,
                hidden_dim=16,
                heuristic_algorithms=("min_makespan",),
                rollout_samples=0,
                validation_episodes=0,
                seed=5,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))

        losses = [float(row["loss"]) for row in metrics]
        self.assertGreater(losses[0], losses[-1])

    def test_phase2_merged_training_builds_phase1_assignments_from_heuristic(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::B", steel=8, cut=500.0, bevel=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                    "PLS31": self._machine_dict("PLS31", "23"),
                },
                phase1_assignments={},
                output_dir=temp_dir,
                episodes=2,
                hidden_dim=8,
                heuristic_algorithms=("lpt_batch",),
                rollout_samples=1,
                seed=5,
                phase1_heuristic="bevel_first_balanced",
                phase1_bay_ids=("22", "23"),
            )
            best_timeline_exists = Path(summary["best_timeline_csv"]).exists()

        self.assertEqual(summary["episodes"], 2)
        self.assertTrue(best_timeline_exists)

    def test_phase2_merged_training_can_use_frozen_phase1_checkpoint_upstream(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::C", steel=8, cut=500.0, bevel=2),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "phase1_pair_pointer_best.pt"
            model = Phase1PairPointerPolicy(
                pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
                env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
                hidden_dim=8,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pair_feature_names": list(PHASE1_PAIR_FEATURE_NAMES),
                    "env_feature_names": list(PHASE1_PAIR_ENV_FEATURE_NAMES),
                    "hidden_dim": 8,
                },
                checkpoint_path,
            )
            phase1_builder = _phase1_agent_assignment_builder(
                checkpoint=str(checkpoint_path),
                bay_ids=("22", "23"),
                sample_count=2,
                temperature=1.0,
                seed=5,
                score_mode="steel_first",
                long_cut_hard_mask=True,
                bay_capacity_weights={"22": 1, "23": 1},
            )
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS31": self._machine_dict("PLS31", "23"),
                },
                phase1_assignments={},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=0,
                validation_episodes=0,
                write_candidate_summary=True,
                phase1_assignment_builder=phase1_builder,
                phase1_bay_ids=("22", "23"),
                seed=5,
            )
            with Path(summary["candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                candidate_rows = list(csv.DictReader(file))

        self.assertTrue(candidate_rows)
        self.assertTrue(all(row["bay_id"] for row in candidate_rows))

    def test_phase2_merged_training_self_labels_each_bay_subproblem(self) -> None:
        jobs = {
            "WO_A0": self._job("WO_A0", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_A1": self._job("WO_A1", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_B0": self._job("WO_B0", "P2::B", steel=8, cut=500.0, bevel=2),
            "WO_B1": self._job("WO_B1", "P2::B", steel=7, cut=400.0, bevel=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                    "PLS31": self._machine_dict("PLS31", "23"),
                    "PLS32": self._machine_dict("PLS32", "23"),
                },
                phase1_assignments={"P1::A": "22", "P2::B": "23"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch", "lpt_batch"),
                rollout_samples=0,
                validation_episodes=0,
                write_candidate_summary=True,
                seed=5,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))
            with Path(summary["candidate_summary_csv"]).open(encoding="utf-8-sig") as file:
                candidate_rows = list(csv.DictReader(file))

        self.assertEqual(metrics[0]["candidate_count"], "6")
        self.assertIn("22:", metrics[0]["best_source"])
        self.assertIn("23:", metrics[0]["best_source"])
        self.assertEqual({row["bay_id"] for row in candidate_rows}, {"22", "23"})

    def test_phase2_merged_training_can_select_normalized_score_and_writes_bay_metrics(self) -> None:
        jobs = {
            "WO_A0": self._job("WO_A0", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_A1": self._job("WO_A1", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_B0": self._job("WO_B0", "P2::B", steel=8, cut=500.0, bevel=2),
            "WO_B1": self._job("WO_B1", "P2::B", steel=7, cut=400.0, bevel=1),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines={
                    "PLS21": self._machine_dict("PLS21", "22"),
                    "PLS22": self._machine_dict("PLS22", "22"),
                    "PLS31": self._machine_dict("PLS31", "23"),
                    "PLS32": self._machine_dict("PLS32", "23"),
                },
                phase1_assignments={"P1::A": "22", "P2::B": "23"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch", "lpt_batch"),
                rollout_samples=0,
                validation_episodes=0,
                score_mode="normalized",
                seed=5,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))
            with Path(summary["subproblem_metrics_csv"]).open(encoding="utf-8-sig") as file:
                subproblem_rows = list(csv.DictReader(file))

        self.assertEqual(summary["score_mode"], "normalized")
        self.assertIn("bay_internal_normalized_wo_count_gap", summary["score_fields"])
        self.assertIn("raw_bay_internal_wo_count_gap", metrics[0])
        self.assertIn("normalized_bay_internal_wo_count_gap", metrics[0])
        self.assertEqual({row["bay_id"] for row in subproblem_rows}, {"22", "23"})
        self.assertTrue(all("normalized_bay_internal_cut_length_gap" in row for row in subproblem_rows))

    def test_phase2_merged_training_resumes_from_latest_checkpoint(self) -> None:
        jobs = {
            "WO_A0": self._job("WO_A0", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_A1": self._job("WO_A1", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_B0": self._job("WO_B0", "P2::B", steel=8, cut=500.0, bevel=2),
            "WO_B1": self._job("WO_B1", "P2::B", steel=7, cut=400.0, bevel=1),
        }
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
            "PLS31": self._machine_dict("PLS31", "23"),
            "PLS32": self._machine_dict("PLS32", "23"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines=machines,
                phase1_assignments={"P1::A": "22", "P2::B": "23"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=0,
                validation_episodes=0,
                checkpoint_every=1,
                seed=5,
            )
            summary = train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines=machines,
                phase1_assignments={"P1::A": "22", "P2::B": "23"},
                output_dir=temp_dir,
                episodes=3,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=0,
                validation_episodes=0,
                checkpoint_every=1,
                resume_checkpoint="latest",
                seed=5,
            )
            with Path(summary["metrics_csv"]).open(encoding="utf-8-sig") as file:
                metrics = list(csv.DictReader(file))

        self.assertEqual(summary["start_episode"], 2)
        self.assertEqual(summary["resumed_from_episode"], 1)
        self.assertEqual([row["episode"] for row in metrics], ["1", "2", "3"])

    def test_phase2_resume_rejects_run_spec_change(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
        }
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines=machines,
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=0,
                validation_episodes=0,
                checkpoint_every=1,
                action_pool_limit=None,
                seed=5,
            )

            with self.assertRaises(RuntimeError):
                train_phase2_batch_machine_self_labeling(
                    jobs=jobs,
                    machines=machines,
                    phase1_assignments={"P1::A": "22"},
                    output_dir=temp_dir,
                    episodes=2,
                    hidden_dim=8,
                    heuristic_algorithms=("spt_batch",),
                    rollout_samples=0,
                    validation_episodes=0,
                    checkpoint_every=1,
                    action_pool_limit=1,
                    resume_checkpoint="latest",
                    seed=5,
                )

    def test_phase2_resume_rejects_checkpoint_without_optimizer_state(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
        }
        machines = {
            "PLS21": self._machine_dict("PLS21", "22"),
            "PLS22": self._machine_dict("PLS22", "22"),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            train_phase2_batch_machine_self_labeling(
                jobs=jobs,
                machines=machines,
                phase1_assignments={"P1::A": "22"},
                output_dir=temp_dir,
                episodes=1,
                hidden_dim=8,
                heuristic_algorithms=("spt_batch",),
                rollout_samples=0,
                validation_episodes=0,
                checkpoint_every=1,
                seed=5,
            )
            checkpoint_path = Path(temp_dir) / "checkpoints" / "phase2_batch_machine_policy_ep00001.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            del checkpoint["optimizer_state_dict"]
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaises(RuntimeError):
                train_phase2_batch_machine_self_labeling(
                    jobs=jobs,
                    machines=machines,
                    phase1_assignments={"P1::A": "22"},
                    output_dir=temp_dir,
                    episodes=2,
                    hidden_dim=8,
                    heuristic_algorithms=("spt_batch",),
                    rollout_samples=0,
                    validation_episodes=0,
                    checkpoint_every=1,
                    resume_checkpoint="latest",
                    seed=5,
                )

    def test_phase2_full_workflow_connects_phase1_to_merged_batch_machine_timeline(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("steel_first_balanced",),
        )

        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [self._machine_dict("PLS21", "22"), self._machine_dict("PLS31", "23")],
            },
            phase1_plan=phase1["plan"],
        )

        self.assertEqual(result["assignment"]["summary"]["assigned_job_count"], 2)
        self.assertEqual(result["batches"]["summary"]["assigned_job_count"], 2)
        self.assertEqual(result["sequence"]["summary"]["batch_count"], result["batches"]["summary"]["batch_count"])
        self.assertTrue(
            {"score_mode", "score_tuple", "score_details", "per_bay", "diagnostics", "machine_load_score", "batch_score", "sequence_score"}
            <= set(result["evaluation"])
        )
        self.assertIn("makespan", result["evaluation"]["sequence_score"])
        self.assertEqual(
            result["evaluation"]["score_tuple"][1],
            result["evaluation"]["sequence_score"]["makespan"],
        )
        self.assertEqual(
            tuple(result["evaluation"]["score_tuple"]),
            tuple(result["batch_machine_candidate_score"]),
        )

    def test_phase2_full_workflow_accepts_fixed_merged_batch_machine_heuristic(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::B", steel=8, cut=500.0, bevel=9),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("bevel_first_balanced",),
        )

        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [
                    self._machine_dict("PLS21", "22"),
                    self._machine_dict("PLS22", "22"),
                    self._machine_dict("PLS31", "23"),
                ],
            },
            phase1_plan=phase1["plan"],
            batch_machine_heuristic="lpt_batch",
        )

        self.assertEqual(result["assignment"]["summary"]["assignment_source"], "merged_batch_machine")
        self.assertEqual(result["assignment"]["summary"]["batch_machine_source"], "lpt_batch")
        self.assertEqual(result["batches"]["summary"]["batch_source"], "lpt_batch")
        self.assertEqual(result["assignment"]["summary"]["assigned_job_count"], 3)
        self.assertEqual(result["batches"]["summary"]["assigned_job_count"], 3)
        self.assertLessEqual(result["evaluation"]["batch_score"]["max_batch_wo_count"], 3)

    def test_phase2_full_workflow_accepts_merged_checkpoint_model(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::B", steel=8, cut=500.0, bevel=9),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("bevel_first_balanced",),
        )
        model = Phase2SetPointerPolicy(hidden_dim=8)

        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [
                    self._machine_dict("PLS21", "22"),
                    self._machine_dict("PLS22", "22"),
                    self._machine_dict("PLS31", "23"),
                ],
            },
            phase1_plan=phase1["plan"],
            model=model,
            batch_machine_heuristic=None,
        )

        self.assertEqual(result["assignment"]["summary"]["assignment_source"], "merged_batch_machine")
        self.assertEqual(result["batches"]["summary"]["batch_source"], "agent_greedy")
        self.assertEqual(result["sequence"]["summary"]["sequence_source"], "agent_greedy")
        self.assertEqual(result["batches"]["summary"]["assigned_job_count"], 3)

    def test_phase2_full_workflow_uses_checkpoint_best_of_k_candidates(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::A", steel=5, cut=300.0, bevel=1),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22"],
            heuristic_algorithms=("bevel_first_balanced",),
        )

        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [self._machine_dict("PLS21", "22")],
            },
            phase1_plan=phase1["plan"],
            model=Phase2SetPointerPolicy(hidden_dim=8),
            batch_machine_heuristic=None,
            rollout_samples=2,
            seed=7,
        )

        self.assertEqual(result["batch_machine_candidate_count"], 3)
        self.assertIn(
            result["batches"]["summary"]["batch_source"],
            {"agent_greedy", "agent_sample_1", "agent_sample_2"},
        )

    def test_phase2_full_workflow_rejects_merged_heuristic_and_model_together(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22"],
            heuristic_algorithms=("steel_first_balanced",),
        )
        model = Phase2SetPointerPolicy(hidden_dim=8)

        with self.assertRaises(RuntimeError):
            run_phase2_full_graph_workflow(
                scenario={
                    "jobs": [self._job_dict(job) for job in jobs.values()],
                    "machines": [self._machine_dict("PLS21", "22")],
                },
                phase1_plan=phase1["plan"],
                batch_machine_heuristic="lpt_batch",
                model=model,
            )

    def test_phase2_checkpoint_loader_accepts_set_pointer_and_rejects_legacy_edge_schema(self) -> None:
        model = Phase2SetPointerPolicy(hidden_dim=8)
        run_spec = build_phase2_run_spec(
            score_mode="raw",
            score_fields=PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
            action_pool_limit=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            phase1_bay_capacity_weights={"22": 1.0},
            phase1_long_cut_hard_mask=True,
            constraint_profile=default_phase2_constraint_profile(),
            heuristic_algorithms=("spt_batch",),
            train_rollout_samples=1,
            validation_rollout_samples=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "phase2_set_pointer.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "policy_type": "phase2_set_pointer",
                    "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
                    "feature_schema": phase2_state_feature_schema(),
                    "hidden_dim": 8,
                    "run_spec": run_spec,
                },
                checkpoint_path,
            )

            loaded = load_phase2_set_pointer_checkpoint(
                checkpoint_path,
                context="phase2_batch_machine",
            )
            self.assertEqual(loaded.hidden_dim, 8)
            self.assertEqual(
                load_phase2_checkpoint_run_spec(checkpoint_path, context="phase2_batch_machine"),
                run_spec,
            )

            missing_run_spec_path = Path(temp_dir) / "missing_run_spec.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "policy_type": "phase2_set_pointer",
                    "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
                    "feature_schema": phase2_state_feature_schema(),
                    "hidden_dim": 8,
                },
                missing_run_spec_path,
            )
            with self.assertRaises(RuntimeError):
                load_phase2_set_pointer_checkpoint(
                    missing_run_spec_path,
                    context="phase2_missing_run_spec",
                )

            legacy_path = Path(temp_dir) / "legacy_edge.pt"
            torch.save(
                {
                    "model_state_dict": {},
                    "edge_feature_dim": 12,
                    "edge_feature_names": ["legacy"] * 12,
                    "hidden_dim": 8,
                },
                legacy_path,
            )
            with self.assertRaises(RuntimeError):
                load_phase2_set_pointer_checkpoint(
                    legacy_path,
                    context="phase2_batch_machine_mismatch",
                )

    def test_phase2_full_workflow_writes_debug_csv_outputs(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("steel_first_balanced",),
        )
        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [self._machine_dict("PLS21", "22"), self._machine_dict("PLS31", "23")],
            },
            phase1_plan=phase1["plan"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_phase2_workflow_outputs(result, temp_dir)
            generated = {name: Path(path).exists() for name, path in paths.items() if name.endswith("_csv")}
            event_log_exists = Path(paths["event_log_json"]).exists()
            with Path(paths["bay_metric_csv"]).open(encoding="utf-8-sig") as file:
                bay_metric_rows = list(csv.DictReader(file))
            report = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
            constraint_violation_count = paths["constraint_violation_count"]

        self.assertTrue(generated)
        self.assertTrue(all(generated.values()))
        self.assertTrue(event_log_exists)
        self.assertEqual({row["bay_id"] for row in bay_metric_rows}, {"22", "23"})
        self.assertEqual(constraint_violation_count, 0)
        self.assertEqual(report["batch_machine_candidate_count"], 1)
        self.assertEqual(
            report["batch_machine_candidate_score"],
            list(result["batch_machine_candidate_score"]),
        )

    def test_phase2_report_processing_time_does_not_fallback_to_alias(self) -> None:
        with self.assertRaises(RuntimeError):
            _job_processing_time({"job_id": "WO_A", "processing_time": 10.0})

    def test_phase2_full_workflow_writes_identity_checked_event_log(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
            "WO_C": self._job("WO_C", "P1::B", steel=8, cut=500.0, bevel=2),
        }
        phase1 = run_phase1_graph_workflow(
            jobs=jobs,
            bay_ids=["22", "23"],
            heuristic_algorithms=("steel_first_balanced",),
        )
        result = run_phase2_full_graph_workflow(
            scenario={
                "jobs": [self._job_dict(job) for job in jobs.values()],
                "machines": [self._machine_dict("PLS21", "22"), self._machine_dict("PLS31", "23")],
            },
            phase1_plan=phase1["plan"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_phase2_workflow_outputs(result, temp_dir)
            event_log = json.loads(Path(paths["event_log_json"]).read_text(encoding="utf-8"))

        self.assertIn("PROCESS_START", {row["event_type"] for row in event_log})
        self.assertIn("PROCESS_FINISH", {row["event_type"] for row in event_log})
        self.assertEqual(
            sorted(row["job_id"] for row in event_log if row["event_type"] == "PROCESS_START"),
            sorted(jobs),
        )
        self.assertEqual(
            sorted(row["job_id"] for row in event_log if row["event_type"] == "PROCESS_FINISH"),
            sorted(jobs),
        )

    @staticmethod
    def _job(job_id: str, block_set_id: str, steel: int, cut: float, bevel: int) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            steel_quantity=steel,
            cut_length=cut,
            bevel_quantity=bevel,
            plate_length=10000.0,
            thickness=13.0,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": 10.0},
            extra={"source_project_no": block_set_id.split("::")[0], "source_block_no": block_set_id.split("::")[1]},
        )

    @staticmethod
    def _job_dict(job: SimpleNamespace) -> dict:
        return {
            "job_id": job.job_id,
            "block_set_id": job.block_set_id,
            "family": job.family,
            "steel_quantity": job.steel_quantity,
            "cut_length": job.cut_length,
            "bevel_quantity": job.bevel_quantity,
            "plate_length": job.plate_length,
            "thickness": job.thickness,
            "base_stage_minutes": job.base_stage_minutes,
            "extra": job.extra,
        }

    @staticmethod
    def _machine_dict(machine_id: str, bay_id: str) -> dict:
        return {
            "machine_id": machine_id,
            "bay_id": bay_id,
            "machine_type": "PLS",
            "enabled": True,
            "eligible_families": ["NP"],
            "min_thickness": 0.0,
            "max_thickness": 100.0,
            "table_length_limit": 55000.0,
        }


if __name__ == "__main__":
    unittest.main()
