"""Phase 2 action mask가 공통 제약 registry를 사용하는지 검증한다."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Environment.constraints.profiles import (
    audit_phase2_schedule_constraints,
    evaluate_phase2_action_constraints,
    load_phase_constraint_profile,
)
from Phase2.merged import run_phase2_batch_machine_candidate
from Utils.config import load_config


class Phase2ConstraintProfileTest(unittest.TestCase):
    def setUp(self) -> None:
        config = load_config("config_np_100.yaml")
        self.profile = load_phase_constraint_profile(config, "phase2")

    def test_profile_enables_only_confirmed_phase2_hard_rules(self) -> None:
        enabled = {name for name, value in self.profile.hard_enabled.items() if value}

        self.assertEqual(
            enabled,
            {
                "machine_enabled",
                "machine_bay_consistency",
                "block_set_same_bay",
                "machine_single_processing",
                "batch_wo_count_limit",
                "batch_length_sum_limit",
                "family_eligibility",
            },
        )
        self.assertTrue(self.profile.hard_enabled["family_eligibility"])
        self.assertFalse(self.profile.hard_enabled["thickness_range"])
        self.assertFalse(self.profile.hard_enabled["table_length_limit"])

    def test_family_eligibility_rejects_incompatible_job_in_mixed_family_batch(self) -> None:
        np_job = self._job("WO_NP", "P1::NP::A", plate_length=10_000.0, family="NP")
        fl_job = self._job("WO_FL", "P1::FL::A", plate_length=10_000.0, family="FL")
        np_machine = self._machine("PLS21", "22", eligible_families=("NP",))
        compatible_machine = self._machine("PLS22", "22", eligible_families=("NP", "FL"))
        jobs = {job.job_id: job for job in (np_job, fl_job)}
        assignments = {np_job.block_set_id: "22", fl_job.block_set_id: "22"}

        rejected = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=fl_job,
            machine=np_machine,
            jobs=jobs,
            machines={np_machine.machine_id: np_machine},
            phase1_assignments=assignments,
            machine_available_at={np_machine.machine_id: 0.0},
            current_time=0.0,
            candidate_batch_job_ids=(np_job.job_id, fl_job.job_id),
            candidate_batch_length_sum=20_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        accepted = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=fl_job,
            machine=compatible_machine,
            jobs=jobs,
            machines={compatible_machine.machine_id: compatible_machine},
            phase1_assignments=assignments,
            machine_available_at={compatible_machine.machine_id: 0.0},
            current_time=0.0,
            candidate_batch_job_ids=(np_job.job_id, fl_job.job_id),
            candidate_batch_length_sum=20_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )

        self.assertIn("family_eligibility", rejected.hard_failed_rule_names)
        self.assertTrue(accepted.hard_passed)

    def test_common_evaluator_rejects_disabled_wrong_bay_busy_and_batch_overflow(self) -> None:
        job = self._job("WO_A", "P1::A", plate_length=30_000.0)
        machine = self._machine("PLS21", "22")
        jobs = {job.job_id: job}
        machines = {machine.machine_id: machine}
        phase1_assignments = {job.block_set_id: "22"}

        passed = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=job,
            machine=machine,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at={"PLS21": 0.0},
            current_time=0.0,
            candidate_batch_job_ids=("WO_A",),
            candidate_batch_length_sum=30_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertTrue(passed.hard_passed)

        machine.enabled = False
        disabled = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=job,
            machine=machine,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at={"PLS21": 0.0},
            current_time=0.0,
            candidate_batch_job_ids=("WO_A",),
            candidate_batch_length_sum=30_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertIn("machine_enabled", disabled.hard_failed_rule_names)
        machine.enabled = True

        wrong_bay = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=job,
            machine=self._machine("PLS31", "23"),
            jobs=jobs,
            machines={"PLS31": self._machine("PLS31", "23")},
            phase1_assignments=phase1_assignments,
            machine_available_at={"PLS31": 0.0},
            current_time=0.0,
            candidate_batch_job_ids=("WO_A",),
            candidate_batch_length_sum=30_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertIn("machine_bay_consistency", wrong_bay.hard_failed_rule_names)
        self.assertIn("block_set_same_bay", wrong_bay.hard_failed_rule_names)

        busy = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=job,
            machine=machine,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at={"PLS21": 10.0},
            current_time=0.0,
            candidate_batch_job_ids=("WO_A",),
            candidate_batch_length_sum=30_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertIn("machine_single_processing", busy.hard_failed_rule_names)

        overflow = evaluate_phase2_action_constraints(
            profile=self.profile,
            job=job,
            machine=machine,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at={"PLS21": 0.0},
            current_time=0.0,
            candidate_batch_job_ids=("WO_A", "WO_B", "WO_C", "WO_D"),
            candidate_batch_length_sum=60_000.0,
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertIn("batch_wo_count_limit", overflow.hard_failed_rule_names)
        self.assertIn("batch_length_sum_limit", overflow.hard_failed_rule_names)

    def test_candidate_generation_masks_disabled_machine(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", plate_length=10_000.0),
            "WO_B": self._job("WO_B", "P1::A", plate_length=10_000.0),
        }
        machines = {
            "PLS21": self._machine("PLS21", "22", enabled=False),
            "PLS22": self._machine("PLS22", "22", enabled=True),
        }

        candidate = run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::A": "22"},
            source="min_makespan",
            model=None,
            constraint_profile=self.profile,
        )

        self.assertEqual(set(candidate.machine_assignments.values()), {"PLS22"})

    def test_generated_candidate_rejects_any_final_hard_violation(self) -> None:
        jobs = {"WO_A": self._job("WO_A", "P1::A", plate_length=10_000.0)}
        machines = {"PLS21": self._machine("PLS21", "22")}
        forced_audit = {
            "hard_violation_count": 1,
            "rows": [
                {
                    "batch_id": "B1",
                    "machine_id": "PLS21",
                    "job_id": "",
                    "rule_name": "batch_length_sum_limit",
                    "passed": False,
                    "reason": "forced regression violation",
                }
            ],
            "failed_rows": [],
        }

        with patch("Phase2.merged.audit_phase2_schedule_constraints", return_value=forced_audit):
            with self.assertRaises(RuntimeError):
                run_phase2_batch_machine_candidate(
                    jobs=jobs,
                    machines=machines,
                    phase1_assignments={"P1::A": "22"},
                    source="min_makespan",
                    model=None,
                    constraint_profile=self.profile,
                )

    def test_final_schedule_audit_replays_common_constraints(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", plate_length=30_000.0),
            "WO_B": self._job("WO_B", "P1::A", plate_length=30_000.0),
        }
        machines = {"PLS21": self._machine("PLS21", "22")}
        valid = audit_phase2_schedule_constraints(
            profile=self.profile,
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::A": "22"},
            batches=[
                {
                    "batch_id": "B1",
                    "machine_id": "PLS21",
                    "job_ids": ("WO_A",),
                    "start_time": 0.0,
                    "finish_time": 10.0,
                },
                {
                    "batch_id": "B2",
                    "machine_id": "PLS21",
                    "job_ids": ("WO_B",),
                    "start_time": 10.0,
                    "finish_time": 20.0,
                },
            ],
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertEqual(valid["hard_violation_count"], 0)

        invalid = audit_phase2_schedule_constraints(
            profile=self.profile,
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::A": "22"},
            batches=[
                {
                    "batch_id": "B1",
                    "machine_id": "PLS21",
                    "job_ids": ("WO_A", "WO_B"),
                    "start_time": 0.0,
                    "finish_time": 10.0,
                }
            ],
            max_wo_count=3,
            max_length_sum=55_000.0,
        )
        self.assertEqual(invalid["hard_violation_count"], 1)
        self.assertEqual(
            {row["rule_name"] for row in invalid["rows"] if not row["passed"]},
            {"batch_length_sum_limit"},
        )

    def test_final_schedule_audit_rejects_duration_not_equal_batch_max_tact(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", plate_length=10_000.0),
            "WO_B": self._job("WO_B", "P1::A", plate_length=10_000.0),
        }
        jobs["WO_B"].base_stage_minutes = {"cut": 30.0}
        machines = {"PLS21": self._machine("PLS21", "22")}

        with self.assertRaises(RuntimeError):
            audit_phase2_schedule_constraints(
                profile=self.profile,
                jobs=jobs,
                machines=machines,
                phase1_assignments={"P1::A": "22"},
                batches=[
                    {
                        "batch_id": "B1",
                        "machine_id": "PLS21",
                        "job_ids": ("WO_A", "WO_B"),
                        "start_time": 0.0,
                        "finish_time": 10.0,
                    }
                ],
                max_wo_count=3,
                max_length_sum=55_000.0,
            )

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        plate_length: float,
        family: str = "NP",
    ) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family=family,
            plate_length=plate_length,
            thickness=13.0,
            cut_length=100.0,
            bevel_quantity=1,
            cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": 10.0},
        )

    @staticmethod
    def _machine(
        machine_id: str,
        bay_id: str,
        enabled: bool = True,
        eligible_families: tuple[str, ...] = ("NP",),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id=bay_id,
            enabled=enabled,
            eligible_families=eligible_families,
            min_thickness=99.0,
            max_thickness=100.0,
            table_length_limit=1.0,
        )


if __name__ == "__main__":
    unittest.main()
