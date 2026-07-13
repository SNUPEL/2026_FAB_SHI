"""Phase 2 학습/validation/full-flow가 공유할 metric contract 테스트."""

import unittest

from Environment.metrics import calculate_phase2_schedule_metrics
from Phase2.evaluation import evaluate_phase2_schedule


class Phase2MetricContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.machine_loads = {
            "M22A": {"wo_count": 10, "processing_time_sum": 120.0, "cut_length_sum": 100.0, "bevel_quantity_sum": 10},
            "M22B": {"wo_count": 8, "processing_time_sum": 90.0, "cut_length_sum": 80.0, "bevel_quantity_sum": 6},
            "M23A": {"wo_count": 100, "processing_time_sum": 900.0, "cut_length_sum": 1000.0, "bevel_quantity_sum": 100},
            "M23B": {"wo_count": 100, "processing_time_sum": 700.0, "cut_length_sum": 1000.0, "bevel_quantity_sum": 100},
        }
        self.machine_clock = {"M22A": 50.0, "M22B": 40.0, "M23A": 500.0, "M23B": 500.0}
        self.machine_bay_ids = {"M22A": "22", "M22B": "22", "M23A": "23", "M23B": "23"}
        self.batches = [
            {"batch_id": "B1", "machine_id": "M22A", "wo_count": 2, "length_sum": 20000.0},
            {"batch_id": "B2", "machine_id": "M23A", "wo_count": 3, "length_sum": 54000.0},
        ]

    def test_parent_score_sums_bay_internal_gaps(self) -> None:
        metrics = calculate_phase2_schedule_metrics(
            machine_loads=self.machine_loads,
            machine_clock=self.machine_clock,
            machine_bay_ids=self.machine_bay_ids,
            batches=self.batches,
            max_wo_count=3,
            max_length_sum=55000.0,
            hard_violation_count=0,
        )

        self.assertEqual(metrics["raw_score"], (0, 500.0, 20.0, 2.0, 4.0, 10.0))
        self.assertAlmostEqual(metrics["normalized_score"][2], 20.0 / 90.0)
        self.assertAlmostEqual(metrics["normalized_score"][3], 2.0 / 9.0)
        self.assertAlmostEqual(metrics["normalized_score"][4], 4.0 / 8.0)
        self.assertAlmostEqual(metrics["normalized_score"][5], 10.0 / 45.0)
        self.assertEqual(metrics["per_bay"]["22"]["raw_wo_count_gap"], 2.0)
        self.assertEqual(metrics["per_bay"]["23"]["raw_wo_count_gap"], 0.0)

    def test_global_gap_is_diagnostic_only(self) -> None:
        metrics = calculate_phase2_schedule_metrics(
            machine_loads=self.machine_loads,
            machine_clock=self.machine_clock,
            machine_bay_ids=self.machine_bay_ids,
            batches=self.batches,
            max_wo_count=3,
            max_length_sum=55000.0,
            hard_violation_count=0,
        )

        self.assertEqual(metrics["raw_score"][3], 2.0)
        self.assertEqual(metrics["diagnostics"]["global_wo_count_gap"], 92.0)
        self.assertNotEqual(metrics["raw_score"][3], metrics["diagnostics"]["global_wo_count_gap"])

    def test_individual_tact_sum_gap_is_not_batch_occupancy_gap(self) -> None:
        metrics = calculate_phase2_schedule_metrics(
            machine_loads=self.machine_loads,
            machine_clock=self.machine_clock,
            machine_bay_ids=self.machine_bay_ids,
            batches=self.batches,
            max_wo_count=3,
            max_length_sum=55000.0,
            hard_violation_count=0,
        )

        self.assertEqual(metrics["raw_score"][5], 10.0)
        self.assertEqual(metrics["diagnostics"]["global_individual_tact_sum_gap"], 810.0)
        self.assertEqual(metrics["diagnostics"]["global_occupancy_gap"], 460.0)

    def test_batch_capacity_violation_is_counted(self) -> None:
        violating_batches = [
            {"batch_id": "B1", "machine_id": "M22A", "wo_count": 4, "length_sum": 56000.0},
        ]
        metrics = calculate_phase2_schedule_metrics(
            machine_loads=self.machine_loads,
            machine_clock=self.machine_clock,
            machine_bay_ids=self.machine_bay_ids,
            batches=violating_batches,
            max_wo_count=3,
            max_length_sum=55000.0,
            hard_violation_count=1,
        )

        self.assertEqual(metrics["raw_score"][0], 1)
        self.assertEqual(metrics["hard_violation_count"], 1)

    def test_full_flow_evaluation_uses_shared_bay_internal_metrics(self) -> None:
        timeline = [
            {"batch_id": "B1", "machine_id": machine_id, "finish_time": finish_time}
            for machine_id, finish_time in self.machine_clock.items()
        ]
        result = evaluate_phase2_schedule(
            machine_loads=self.machine_loads,
            machine_bay_ids=self.machine_bay_ids,
            batches=self.batches,
            timeline=timeline,
            max_wo_count=3,
            max_length_sum=55000.0,
            score_mode="raw",
            hard_violation_count=0,
        )

        self.assertEqual(result["score_tuple"], (0, 500.0, 20.0, 2.0, 4.0, 10.0))
        self.assertEqual(result["machine_load_score"]["bay_internal_wo_count_gap"], 2.0)
        self.assertEqual(result["machine_load_score"]["diagnostic_global_wo_count_gap"], 92.0)
        self.assertEqual(result["machine_load_score"]["bay_internal_occupancy_gap"], 10.0)
        self.assertEqual(result["machine_load_score"]["diagnostic_global_individual_tact_sum_gap"], 810.0)
        self.assertEqual(result["per_bay"]["22"]["raw_occupancy_gap"], 10.0)


if __name__ == "__main__":
    unittest.main()
