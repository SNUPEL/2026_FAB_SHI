"""Phase 1 <-> Phase 2 limited communication contract tests."""

import unittest

from Utils.learning.phase1_phase2_communication import (
    apply_phase1_messages_to_scenario,
    build_phase1_plan_messages,
    build_phase2_feedback_messages,
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
