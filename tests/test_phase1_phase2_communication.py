"""Phase 1 <-> Phase 2 limited communication contract tests."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from Utils.phase1_phase2_communication import (
    apply_phase1_messages_to_scenario,
    build_phase1_plan_messages,
    build_phase2_feedback_messages,
    build_phase2_feedback_score,
    load_communication_jsonl,
    score_phase1_assignments_with_phase2_feedback,
    write_phase1_phase2_communication_package,
)


class Phase1Phase2CommunicationTest(unittest.TestCase):
    """Phase agents should exchange only the minimum auditable planning message."""

    def test_phase1_message_contains_block_assignment_without_full_phase1_state(self) -> None:
        scenario = self._scenario()
        plan = self._plan()

        messages = build_phase1_plan_messages(
            scenario=scenario,
            plan=plan,
            wide_bth_threshold=4500.0,
        )
        wide_message = next(row for row in messages if row["block_set_id"] == "P1::WIDE")

        self.assertEqual(wide_message["direction"], "phase1_to_phase2")
        self.assertEqual(wide_message["assigned_bay"], "24")
        self.assertEqual(wide_message["job_ids"], ["WO_W1", "WO_W2"])
        self.assertEqual(wide_message["wide_plate_flag"], 1)
        self.assertEqual(wide_message["plate_width_max"], 4700.0)
        self.assertNotIn("phase1_policy_logits", wide_message)
        self.assertNotIn("all_bay_load_state", wide_message)
        self.assertNotIn("machine_timeline", wide_message)

    def test_phase2_feedback_recommends_bay22_23_for_wide_block_on_bay24(self) -> None:
        scenario = self._scenario()
        plan = self._plan()

        feedback = build_phase2_feedback_messages(
            scenario=scenario,
            plan=plan,
            wide_bth_threshold=4500.0,
            batch_max_wo_count=3,
            batch_max_length_sum=55000.0,
        )
        wide_feedback = next(row for row in feedback if row["block_set_id"] == "P1::WIDE")
        normal_feedback = next(row for row in feedback if row["block_set_id"] == "P1::NORMAL")

        self.assertEqual(wide_feedback["direction"], "phase2_to_phase1")
        self.assertEqual(wide_feedback["status"], "repair_required")
        self.assertEqual(wide_feedback["hard_violation_count"], 1)
        self.assertEqual(wide_feedback["wide_bay_violation_count"], 1)
        self.assertEqual(wide_feedback["recommended_bays"], ["22", "23"])
        self.assertIn("wide_plate_requires_bay22_23", wide_feedback["reason_codes"])
        self.assertEqual(normal_feedback["status"], "ok")
        self.assertEqual(normal_feedback["hard_violation_count"], 0)

    def test_package_writes_jsonl_csv_and_manifest(self) -> None:
        scenario = self._scenario()
        plan = self._plan()

        with tempfile.TemporaryDirectory() as temp_dir:
            package = write_phase1_phase2_communication_package(
                scenario=scenario,
                plan=plan,
                output_dir=temp_dir,
                wide_bth_threshold=4500.0,
                batch_max_wo_count=3,
                batch_max_length_sum=55000.0,
            )

            phase1_rows = [
                json.loads(line)
                for line in Path(package["phase1_to_phase2_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]
            feedback_rows = [
                json.loads(line)
                for line in Path(package["phase2_to_phase1_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]
            with Path(package["feedback_csv"]).open(encoding="utf-8-sig") as file:
                csv_rows = list(csv.DictReader(file))
            manifest = json.loads(Path(package["manifest_json"]).read_text(encoding="utf-8"))

        self.assertEqual(len(phase1_rows), 2)
        self.assertEqual(len(feedback_rows), 2)
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(manifest["summary"]["block_message_count"], 2)
        self.assertEqual(manifest["summary"]["repair_required_count"], 1)

    def test_phase2_can_apply_phase1_messages_as_allowed_bay_ids(self) -> None:
        scenario = self._scenario()
        plan = self._plan()
        messages = build_phase1_plan_messages(scenario=scenario, plan=plan)

        result = apply_phase1_messages_to_scenario(
            scenario=scenario,
            phase1_messages=messages,
            assignment_mode="allowed_bay_ids",
        )
        jobs = {job["job_id"]: job for job in result["scenario"]["jobs"]}

        self.assertEqual(jobs["WO_W1"]["allowed_bay_ids"], ["24"])
        self.assertEqual(jobs["WO_W2"]["allowed_bay_ids"], ["24"])
        self.assertEqual(jobs["WO_N1"]["allowed_bay_ids"], ["22"])
        self.assertEqual(result["summary"]["assigned_job_count"], 3)
        self.assertEqual(result["scenario"]["metadata"]["phase1_message_applied"], True)

    def test_jsonl_loader_reads_phase1_messages_for_phase2(self) -> None:
        scenario = self._scenario()
        plan = self._plan()

        with tempfile.TemporaryDirectory() as temp_dir:
            package = write_phase1_phase2_communication_package(
                scenario=scenario,
                plan=plan,
                output_dir=temp_dir,
            )
            messages = load_communication_jsonl(package["phase1_to_phase2_jsonl"])

        self.assertEqual(len(messages), 2)
        self.assertEqual({row["direction"] for row in messages}, {"phase1_to_phase2"})

    def test_phase2_feedback_score_prioritizes_hard_repairs_before_load_terms(self) -> None:
        feedback = build_phase2_feedback_messages(
            scenario=self._scenario(),
            plan=self._plan(),
            wide_bth_threshold=4500.0,
            batch_max_wo_count=3,
            batch_max_length_sum=55000.0,
        )

        score = build_phase2_feedback_score(feedback)

        self.assertEqual(score[:4], (1, 0, 1, 1))
        self.assertGreater(score[4], 0.0)

    def test_assignment_score_accepts_phase1_job_and_machine_objects(self) -> None:
        jobs = {
            "WO_W1": SimpleNamespace(
                job_id="WO_W1",
                block_set_id="P1::WIDE",
                family="NP",
                steel_quantity=6,
                cut_length=700.0,
                bevel_quantity=2,
                plate_length=12000.0,
                plate_width=4700.0,
                extra={"source_project_no": "P1", "source_block_no": "WIDE", "source_wk_ord_no": "WO_W1"},
            )
        }
        machines = [
            SimpleNamespace(machine_id="PLS21", bay_id="22", enabled=True),
            SimpleNamespace(machine_id="PLS31", bay_id="23", enabled=True),
            SimpleNamespace(machine_id="PLS41", bay_id="24", enabled=True),
        ]

        bad_score = score_phase1_assignments_with_phase2_feedback(
            assignments={"P1::WIDE": "24"},
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            machines=machines,
            wide_bth_threshold=4500.0,
        )
        good_score = score_phase1_assignments_with_phase2_feedback(
            assignments={"P1::WIDE": "22"},
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            machines=machines,
            wide_bth_threshold=4500.0,
        )

        self.assertEqual(bad_score[:4], (1, 0, 1, 1))
        self.assertEqual(good_score[:4], (0, 0, 0, 0))

    def test_assignment_score_fails_when_required_job_length_is_missing(self) -> None:
        jobs = {
            "WO_BAD": SimpleNamespace(
                job_id="WO_BAD",
                block_set_id="P1::BAD",
                steel_quantity=1,
                cut_length=10.0,
                bevel_quantity=0,
                extra={"source_project_no": "P1", "source_block_no": "BAD", "source_wk_ord_no": "WO_BAD"},
            )
        }
        machines = [SimpleNamespace(machine_id="PLS21", bay_id="22", enabled=True)]

        with self.assertRaisesRegex(RuntimeError, "missing_plate_length"):
            score_phase1_assignments_with_phase2_feedback(
                assignments={"P1::BAD": "22"},
                jobs=jobs,
                bay_ids=["22"],
                machines=machines,
            )

    @staticmethod
    def _scenario() -> dict:
        return {
            "metadata": {"scenario_name": "unit"},
            "machines": [
                {"machine_id": "PLS21", "bay_id": "22"},
                {"machine_id": "PLS31", "bay_id": "23"},
                {"machine_id": "PLS41", "bay_id": "24"},
            ],
            "jobs": [
                {
                    "job_id": "WO_W1",
                    "source_wk_ord_no": "WO_W1",
                    "block_set_id": "P1::WIDE",
                    "family": "NP",
                    "steel_quantity": 6,
                    "cut_length": 700.0,
                    "bevel_quantity": 2,
                    "plate_length": 12000.0,
                    "plate_width": 4700.0,
                },
                {
                    "job_id": "WO_W2",
                    "source_wk_ord_no": "WO_W2",
                    "block_set_id": "P1::WIDE",
                    "family": "NP",
                    "steel_quantity": 4,
                    "cut_length": 500.0,
                    "bevel_quantity": 3,
                    "plate_length": 15000.0,
                    "plate_width": 4600.0,
                },
                {
                    "job_id": "WO_N1",
                    "source_wk_ord_no": "WO_N1",
                    "block_set_id": "P1::NORMAL",
                    "family": "NP",
                    "steel_quantity": 5,
                    "cut_length": 300.0,
                    "bevel_quantity": 0,
                    "plate_length": 8000.0,
                    "plate_width": 3000.0,
                },
            ],
        }

    @staticmethod
    def _plan() -> dict:
        return {
            "algorithm": "multi_objective_balanced",
            "assignments": [
                {
                    "block_set_id": "P1::WIDE",
                    "project_no": "P1",
                    "block_no": "WIDE",
                    "assigned_bay": "24",
                    "candidate_bays": "22|23|24",
                },
                {
                    "block_set_id": "P1::NORMAL",
                    "project_no": "P1",
                    "block_no": "NORMAL",
                    "assigned_bay": "22",
                    "candidate_bays": "22|23|24",
                },
            ],
        }


if __name__ == "__main__":
    unittest.main()
