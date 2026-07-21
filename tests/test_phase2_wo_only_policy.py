"""Phase 2 learned action이 W/O 선택 하나뿐인지 검증한다."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from Phase2.merged import _lookahead_makespan_lower_bound, run_phase2_batch_machine_candidate
from Phase2.state import PHASE2_BAY_CONTEXT_FEATURE_NAMES


class Phase2WoOnlyPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = {
            f"WO_{index}": SimpleNamespace(
                job_id=f"WO_{index}",
                block_set_id=f"P1::NP::BLK_{index}",
                family="NP",
                plate_length=10_000.0,
                thickness=13.0,
                cut_length=100.0 * index,
                bevel_quantity=index,
                cut_bay=None,
                allowed_bay_ids=(),
                allowed_machine_ids=(),
                prohibited_machine_ids=(),
                base_stage_minutes={"cut": 10.0 * index},
            )
            for index in range(1, 5)
        }
        self.machines = {
            machine_id: SimpleNamespace(
                machine_id=machine_id,
                bay_id="22",
                enabled=True,
                eligible_families=("NP",),
                min_thickness=0.0,
                max_thickness=100.0,
                table_length_limit=55_000.0,
            )
            for machine_id in ("PLS21", "PLS22", "PLS23")
        }
        self.phase1_assignments = {
            job.block_set_id: "22" for job in self.jobs.values()
        }

    def test_candidate_contains_one_learned_transition_per_wo(self) -> None:
        candidate = self._candidate()

        self.assertEqual(len(candidate.transitions), len(self.jobs))
        self.assertTrue(candidate.transitions)
        self.assertEqual({row.action_type for row in candidate.transitions}, {"select_wo"})
        self.assertTrue(all(not hasattr(row.policy_state, "stage") for row in candidate.transitions))

    def test_environment_dispatches_idle_machines_in_stable_order(self) -> None:
        candidate = self._candidate()

        self.assertEqual(
            [row["machine_id"] for row in candidate.batches],
            ["PLS21", "PLS22", "PLS23"],
        )
        self.assertEqual([row["wo_count"] for row in candidate.batches], [2, 1, 1])

    def test_state_schema_has_no_stage_selector_features(self) -> None:
        self.assertNotIn("stage_select_machine", PHASE2_BAY_CONTEXT_FEATURE_NAMES)
        self.assertNotIn("stage_select_wo", PHASE2_BAY_CONTEXT_FEATURE_NAMES)

    def test_multiple_bays_start_in_parallel_at_time_zero(self) -> None:
        jobs = {
            "WO_22": self._job("WO_22", "P1::NP::B22", 30.0),
            "WO_23": self._job("WO_23", "P1::NP::B23", 20.0),
        }
        machines = {
            "PLS21": self._machine("PLS21", "22"),
            "PLS31": self._machine("PLS31", "23"),
        }
        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::NP::B22": "22", "P1::NP::B23": "23"},
            source="lpt_batch",
            model=None,
            max_wo_count=1,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            seed=7,
        )

        self.assertEqual([row["start_time"] for row in candidate.timeline], [0.0, 0.0])

    def test_global_clock_jumps_to_earliest_completion_when_all_machines_are_busy(self) -> None:
        jobs = {
            f"WO_{tact}": self._job(f"WO_{tact}", f"P1::NP::B{tact}", float(tact))
            for tact in (40, 30, 20, 10)
        }
        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=self.machines,
            phase1_assignments={job.block_set_id: "22" for job in jobs.values()},
            source="lpt_batch",
            model=None,
            max_wo_count=1,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            seed=7,
        )

        self.assertEqual(
            [
                (row["machine_id"], row["start_time"], row["finish_time"])
                for row in candidate.timeline
            ],
            [
                ("PLS21", 0.0, 40.0),
                ("PLS22", 0.0, 30.0),
                ("PLS23", 0.0, 20.0),
                ("PLS23", 20.0, 30.0),
            ],
        )

    def test_lookahead_does_not_schedule_idle_machine_before_current_event_time(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::NP::BA", 5.0),
            "WO_B": self._job("WO_B", "P1::NP::BB", 30.0),
        }

        lower_bound = _lookahead_makespan_lower_bound(
            jobs=jobs,
            machine_clock={"PLS21": 10.0, "PLS22": 0.0},
            machine_bay_ids={"PLS21": "22", "PLS22": "22"},
            remaining_jobs_by_bay={"22": {"WO_A", "WO_B"}},
            bay_remaining_processing_sum={"22": 35.0},
            selected_machine_id="PLS21",
            selected_machine_start_time=20.0,
            selected_job_ids=("WO_A",),
            batch_duration=5.0,
            max_wo_count=1,
        )

        self.assertEqual(lower_bound, 37.5)

    def _candidate(self):
        return run_phase2_batch_machine_candidate(
            jobs=self.jobs,
            machines=self.machines,
            phase1_assignments=self.phase1_assignments,
            source="lpt_batch",
            model=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            seed=7,
        )

    @staticmethod
    def _job(job_id: str, block_set_id: str, tact: float):
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            plate_length=10_000.0,
            thickness=13.0,
            cut_length=100.0,
            bevel_quantity=1,
            cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": tact},
        )

    @staticmethod
    def _machine(machine_id: str, bay_id: str):
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id=bay_id,
            enabled=True,
            eligible_families=("NP",),
            min_thickness=0.0,
            max_thickness=100.0,
            table_length_limit=55_000.0,
        )


if __name__ == "__main__":
    unittest.main()
