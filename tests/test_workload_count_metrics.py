"""할당 W/O 개수 기준 workload balancing metric 검증."""

import unittest

from Utils.playback_builder import build_metrics_from_rows


class WorkloadCountMetricsTest(unittest.TestCase):
    """처리시간 부하와 작업 개수 부하가 동시에 계산되는지 검증한다."""

    def test_build_metrics_from_rows_includes_machine_and_bay_job_count_imbalance(self) -> None:
        """같은 처리시간이라도 작업 개수 편차를 별도 metric으로 보존해야 한다."""

        rows = [
            {
                "algorithm_machine_id": "PLS21",
                "algorithm_cut_bay": "22",
                "process_min": 10.0,
                "finish_min": 10.0,
                "machine_match": "True",
                "bay_match": "True",
            },
            {
                "algorithm_machine_id": "PLS21",
                "algorithm_cut_bay": "22",
                "process_min": 20.0,
                "finish_min": 20.0,
                "machine_match": "False",
                "bay_match": "True",
            },
            {
                "algorithm_machine_id": "PLS31",
                "algorithm_cut_bay": "23",
                "process_min": 15.0,
                "finish_min": 15.0,
                "machine_match": "True",
                "bay_match": "False",
            },
        ]
        validation = {
            "hard_passed": True,
            "hard_violation_count": 0,
        }

        metrics = build_metrics_from_rows(rows, validation)

        self.assertEqual(metrics["machine_job_counts"], {"PLS21": 2, "PLS31": 1})
        self.assertEqual(metrics["bay_job_counts"], {"22": 2, "23": 1})
        self.assertEqual(metrics["machine_job_count_imbalance"], 1)
        self.assertEqual(metrics["bay_job_count_imbalance"], 1)
        self.assertEqual(metrics["machine_load_imbalance"], 15.0)
        self.assertEqual(metrics["bay_process_load_imbalance"], 15.0)


if __name__ == "__main__":
    unittest.main()
