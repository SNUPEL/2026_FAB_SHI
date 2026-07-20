"""MIXED Phase 1 -> merged Phase 2 공개 workflow 회귀시험."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from Environment.constraints.profiles import default_phase2_constraint_profile
from Phase1.orchestrator import run_phase1_graph_workflow
from Phase2.merged import build_mixed_phase2_training_machines
from Phase2.orchestrator import (
    run_phase2_full_graph_workflow,
    run_phase2_graph_workflow,
    write_phase2_workflow_outputs,
)
from Utils.data.report_formula_data_generator import scenario_jobs_from_report_formula_jobs
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs


class PhaseOrchestratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episode = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=2,
            max_blocks=2,
            seed=20260715,
            verbose=False,
        )[0]
        cls.phase1 = run_phase1_graph_workflow(cls.episode["jobs"])
        cls.scenario = {
            "metadata": {
                "job_source": "mixed_physical_block_joint_distribution",
                "physical_block_count": cls.episode["physical_block_count"],
            },
            "jobs": scenario_jobs_from_report_formula_jobs(cls.episode["jobs"]),
            "machines": [
                asdict(machine)
                for machine in build_mixed_phase2_training_machines().values()
            ],
        }

    def test_phase1_workflow_returns_one_joint_five_bay_plan(self) -> None:
        plan = self.phase1["plan"]

        self.assertEqual(self.phase1["candidate_count"], 3)
        self.assertEqual(plan["rule_profile"], "multi_series_260711")
        self.assertEqual(plan["scope_version"], "joint_five_bay_v3_shared_pool")
        self.assertEqual(set(plan["bay_loads"]), {"22", "23", "24", "25", "trans"})
        self.assertEqual(plan["summary"]["job_count"], len(self.episode["jobs"]))
        self.assertEqual(plan["summary"]["assignment_count"], self.episode["block_count"])

    def test_phase2_graph_applies_every_phase1_block_series_assignment(self) -> None:
        result = run_phase2_graph_workflow(self.scenario, self.phase1["plan"])

        self.assertEqual(result["phase1_message_count"], self.episode["block_count"])
        self.assertEqual(result["applied_summary"]["assigned_job_count"], len(self.episode["jobs"]))
        self.assertEqual(result["graph"]["metadata"]["machine_count"], 15)
        self.assertGreater(result["graph"]["metadata"]["candidate_edge_count"], 0)

    def test_full_flow_schedules_every_wo_and_writes_transition_report(self) -> None:
        result = run_phase2_full_graph_workflow(
            scenario=self.scenario,
            phase1_plan=self.phase1["plan"],
            batch_machine_heuristic="lpt_batch",
            constraint_profile=default_phase2_constraint_profile(),
            max_wo_count=3,
            max_length_sum=55_000.0,
        )

        self.assertEqual(
            result["assignment"]["summary"]["assigned_job_count"],
            len(self.episode["jobs"]),
        )
        self.assertEqual(result["constraint_audit"]["hard_violation_count"], 0)
        self.assertLessEqual(result["evaluation"]["batch_score"]["max_batch_wo_count"], 3)
        self.assertLessEqual(result["evaluation"]["batch_score"]["max_batch_length_sum"], 55_000.0)
        self.assertGreater(len(result["sequence"]["event_log"]), 0)

        with tempfile.TemporaryDirectory() as temp_dir:
            written = write_phase2_workflow_outputs(result, temp_dir)
            self.assertTrue(Path(written["report_json"]).is_file())
            self.assertTrue(Path(written["event_log_json"]).is_file())
            self.assertEqual(written["constraint_violation_count"], 0)


if __name__ == "__main__":
    unittest.main()
