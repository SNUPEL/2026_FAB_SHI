"""MIXED 전용 Phase 2 synthetic planning 계약 회귀시험."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from Phase1.heuristics import run_phase1_heuristic_candidate
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
    build_mixed_phase2_training_machines,
    run_phase2_batch_machine_candidate,
)
from Phase2.run_spec import build_phase2_run_spec
from Environment.constraints.profiles import default_phase2_constraint_profile
from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs
from main import _build_mixed_full_flow_scenario


class Phase2MixedOnlyContractTests(unittest.TestCase):
    def test_run_spec_records_fixed_mixed_phase1_contract_without_legacy_toggle(self) -> None:
        spec = build_phase2_run_spec(
            score_mode="raw",
            score_fields=PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
            action_pool_limit=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            phase1_bay_capacity_weights=joint_phase1_bay_capacity_weights(),
            constraint_profile=default_phase2_constraint_profile(),
            heuristic_algorithms=("lpt_batch",),
            train_rollout_samples=1,
            validation_rollout_samples=1,
        )

        self.assertEqual(spec["phase1_rule_profile"], "multi_series_260711")
        self.assertEqual(spec["phase1_scope_version"], "joint_five_bay_v2_mapped_eqp")
        self.assertEqual(spec["phase1_score_mode"], "wo_first")
        self.assertNotIn("phase1_long_cut_hard_mask", spec)

    def test_mapped_machine_contract_matches_confirmed_five_bay_capacity(self) -> None:
        machines = build_mixed_phase2_training_machines()

        counts: dict[str, int] = {}
        for machine in machines.values():
            counts[str(machine.bay_id)] = counts.get(str(machine.bay_id), 0) + 1

        self.assertEqual(counts, {"22": 4, "23": 3, "24": 4, "25": 2, "trans": 2})
        self.assertEqual(
            set(machines),
            {
                "PLS21", "PLS22", "PLS23", "PLS24",
                "PLS31", "PLS32", "PLS33",
                "PLS41", "PLS42", "PLS43", "PLS44",
                "PLS51", "PLS52", "PLP01", "PLP02",
            },
        )
        self.assertNotIn("EQP_3", machines)
        self.assertFalse(any(machine_id.startswith("SYN_") for machine_id in machines))
        for machine in machines.values():
            expected = {"FN", "FL"} if str(machine.bay_id) in {"25", "trans"} else {"NP", "NC"}
            self.assertEqual(set(machine.eligible_families), expected)

    def test_mixed_episode_runs_phase1_then_phase2_without_hard_violation(self) -> None:
        episode = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=6,
            max_blocks=6,
            seed=20260715,
            verbose=False,
        )[0]
        jobs = episode["jobs"]
        weights = joint_phase1_bay_capacity_weights()
        phase1 = run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=tuple(weights),
            algorithm="wo_first_balanced",
            bay_capacity_weights=weights,
        )
        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=build_mixed_phase2_training_machines(),
            phase1_assignments=phase1.assignments,
            source="lpt_batch",
            model=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            seed=20260715,
        )

        self.assertEqual(len(candidate.machine_assignments), len(jobs))
        self.assertEqual(candidate.score_tuple[0], 0)

    def test_nc_actual_trans_exception_is_planned_only_on_22_23_24_pls_machines(self) -> None:
        job = SimpleNamespace(
            job_id="WO_NC_1",
            block_set_id="P1::NC::BLK_1",
            family="NC",
            steel_quantity=1,
            plate_length=10_000.0,
            plate_width=3_000.0,
            thickness=13.0,
            cut_length=500.0,
            bevel_quantity=2,
            processing_time=10.0,
            base_stage_minutes={"cut": 10.0},
            cut_bay=None,
            source_cut_bay="trans",
            allowed_bay_ids=("22", "23", "24"),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            extra={"source_eqp_nm": "EQP_3"},
        )

        candidate = run_phase2_batch_machine_candidate(
            jobs={job.job_id: job},
            machines=build_mixed_phase2_training_machines(),
            phase1_assignments={job.block_set_id: "22"},
            source="min_makespan",
            model=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            seed=20260716,
        )

        selected_machine = candidate.machine_assignments[job.job_id]
        self.assertIn(selected_machine, {"PLS21", "PLS22", "PLS23", "PLS24"})
        self.assertNotEqual(selected_machine, "EQP_3")
        self.assertEqual(candidate.score_tuple[0], 0)

    def test_full_flow_scenario_uses_same_mixed_episode_and_mapped_machine_contract(self) -> None:
        scenario = _build_mixed_full_flow_scenario(physical_block_count=6, seed=20260715)

        self.assertEqual(scenario["metadata"]["job_source"], "mixed_physical_block_joint_distribution")
        self.assertEqual(scenario["metadata"]["physical_block_count"], 6)
        self.assertEqual(scenario["metadata"]["machine_source"], "confirmed_eqp_pls_plp_mapping")
        self.assertTrue(scenario["metadata"]["actual_eqp_mapping_available"])
        self.assertEqual(len(scenario["machines"]), 15)
        self.assertEqual(
            {str(row["bay_id"]) for row in scenario["machines"]},
            {"22", "23", "24", "25", "trans"},
        )
        self.assertTrue(scenario["jobs"])


if __name__ == "__main__":
    unittest.main()
