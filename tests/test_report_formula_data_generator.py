"""PDF report-formula synthetic W/O data tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from Phase1.orchestrator import candidate_to_phase1_plan
from Phase1.self_labeling import run_phase1_heuristic_candidate
from Phase2.orchestrator import run_phase2_full_graph_workflow
from Phase2.merged import train_phase2_batch_machine_self_labeling
from Utils.data.report_formula_data_generator import (
    BLOCK_COLUMNS,
    WO_COLUMNS,
    build_report_formula_episode_jobs,
    generate_report_formula_data,
    scenario_jobs_from_report_formula_jobs,
)


class ReportFormulaDataGeneratorTest(unittest.TestCase):
    """PDF 수식 생성기가 merged Phase 2용 W/O를 정확히 만드는지 검증한다."""

    def test_generates_wo_and_block_rows_with_aggregation_identity(self) -> None:
        generated = generate_report_formula_data(n_blocks=5, seed=77)

        self.assertEqual(tuple(generated.wo_df.columns), WO_COLUMNS)
        self.assertEqual(tuple(generated.block_df.columns), BLOCK_COLUMNS)
        self.assertEqual(len(generated.block_df), 5)
        self.assertEqual(len(generated.wo_df), int(generated.block_df["STL_QTY"].sum()))
        self.assertFalse(generated.wo_df.isna().any().any())
        self.assertFalse(generated.block_df.isna().any().any())
        self.assertGreater(generated.wo_df["TACT_TIME"].min(), 0)

    def test_episode_jobs_train_merged_phase2_smoke(self) -> None:
        episode = build_report_formula_episode_jobs(
            episode_count=1,
            min_blocks=3,
            max_blocks=3,
            seed=99,
        )[0]
        machines = {
            "PLS21": SimpleNamespace(machine_id="PLS21", bay_id="22", enabled=True),
            "PLS31": SimpleNamespace(machine_id="PLS31", bay_id="23", enabled=True),
            "PLS41": SimpleNamespace(machine_id="PLS41", bay_id="24", enabled=True),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            phase2 = train_phase2_batch_machine_self_labeling(
                jobs=episode["jobs"],
                machines=machines,
                phase1_assignments={},
                output_dir=Path(tmpdir) / "phase2",
                episodes=1,
                hidden_dim=8,
                rollout_samples=1,
                heuristic_algorithms=("min_makespan",),
                phase1_heuristic="bevel_first_balanced",
                phase1_bay_ids=("22", "23", "24"),
            )

        self.assertEqual(phase2["episodes"], 1)
        self.assertGreaterEqual(phase2["job_count"], 3)

    def test_report_formula_jobs_run_full_merged_phase2_flow(self) -> None:
        episode = build_report_formula_episode_jobs(
            episode_count=1,
            min_blocks=3,
            max_blocks=3,
            seed=101,
        )[0]
        bay_ids = ("22", "23", "24")
        scenario = {
            "jobs": scenario_jobs_from_report_formula_jobs(episode["jobs"]),
            "machines": [
                self._machine_dict("PLS21", "22"),
                self._machine_dict("PLS31", "23"),
                self._machine_dict("PLS41", "24"),
            ],
        }
        phase1_candidate = run_phase1_heuristic_candidate(
            jobs=episode["jobs"],
            bay_ids=bay_ids,
            algorithm="bevel_first_balanced",
        )
        phase1_plan = candidate_to_phase1_plan(
            jobs=episode["jobs"],
            bay_ids=bay_ids,
            candidate=phase1_candidate,
            score_mode="steel_first",
            long_cut_hard_mask=True,
        )

        result = run_phase2_full_graph_workflow(
            scenario=scenario,
            phase1_plan=phase1_plan,
            batch_machine_heuristic="spt_batch",
        )

        self.assertLessEqual(result["evaluation"]["batch_score"]["max_batch_wo_count"], 3)
        self.assertLessEqual(result["evaluation"]["batch_score"]["max_batch_length_sum"], 55000.0)
        self.assertEqual(
            result["assignment"]["summary"]["assigned_job_count"],
            len(episode["jobs"]),
        )

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
