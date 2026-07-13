"""Pygame 공장 viewer 입력 변환 검증.

GUI를 띄우지 않고 CSV/JSON 입력 schema만 검증한다. generated schedule은 batch
컬럼이 반드시 있어야 하고, actual replay schedule은 `replay_mode=actual_historical`
일 때만 W/O 1건을 표시용 batch 1건으로 명시 변환한다.
"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Utils.reporting.pygame_factory_viewer import (
    _fit_text_to_width,
    build_completion_summary,
    build_run_summary_lines,
    build_workload_summary,
    build_active_machine_view,
    load_viewer_data,
    run_pygame_comparison_from_paths,
    validate_viewer_data,
)


class PygameFactoryViewerTest(unittest.TestCase):
    """Pygame viewer가 generated/actual 입력 schema를 구분하는지 확인한다."""

    def test_fit_text_to_width_truncates_long_labels(self) -> None:
        """긴 W/O명과 metric label은 박스 폭보다 넓게 그리지 않도록 줄인다."""

        font = _FakeFont(pixel_per_char=8)
        text = "PROJ_23NPWBLK_107WO31_LONG_LABEL"

        clipped = _fit_text_to_width(font, text, max_width=88)

        self.assertLessEqual(font.size(clipped)[0], 88)
        self.assertTrue(clipped.startswith("..."))
        self.assertNotEqual(clipped, text)

    def test_loads_generated_schedule_with_batch_fields(self) -> None:
        """generated schedule은 batch 컬럼을 그대로 읽어 batch 구조를 복원한다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_schedule(
                paths["schedule"],
                fieldnames=[
                    "job_id",
                    "work_order_no",
                    "algorithm_machine_id",
                    "algorithm_cut_bay",
                    "batch_id",
                    "batch_job_ids",
                    "batch_wo_count",
                    "batch_length_sum",
                    "start_min",
                    "finish_min",
                    "process_min",
                    "plate_length",
                    "thickness",
                    "source_machine_id",
                    "source_cut_bay",
                ],
                rows=[
                    {
                        "job_id": "job_001",
                        "work_order_no": "WO_001",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "14000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                    {
                        "job_id": "job_002",
                        "work_order_no": "WO_002",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "16000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                ],
            )

            data = load_viewer_data(
                event_log_path=paths["event_log"],
                layout_path=paths["layout"],
                schedule_path=paths["schedule"],
                metrics_path=paths["metrics"],
            )
            summary = validate_viewer_data(data)

            self.assertEqual(summary["operation_count"], 2)
            self.assertEqual(summary["batch_count"], 1)
            self.assertEqual(data.operations[0].batch_job_ids, ("job_001", "job_002"))
            self.assertEqual(data.operations[0].batch_wo_count, 2)
            self.assertEqual(data.operations[0].batch_length_sum, 30000.0)

            active = build_active_machine_view(data, current_time_min=1.0)
            self.assertEqual(len(active["PLS21"]), 2)

    def test_loads_actual_replay_schedule_as_display_batches(self) -> None:
        """actual replay는 실적 W/O 1건을 표시용 batch 1건으로 명시 변환한다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_schedule(
                paths["schedule"],
                fieldnames=[
                    "job_id",
                    "work_order_no",
                    "algorithm_machine_id",
                    "algorithm_cut_bay",
                    "start_min",
                    "finish_min",
                    "process_min",
                    "plate_length",
                    "thickness",
                    "source_machine_id",
                    "source_cut_bay",
                    "replay_mode",
                ],
                rows=[
                    {
                        "job_id": "job_actual_001",
                        "work_order_no": "WO_ACTUAL_001",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "start_min": "5",
                        "finish_min": "35",
                        "process_min": "30",
                        "plate_length": "12000",
                        "thickness": "11",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                        "replay_mode": "actual_historical",
                    }
                ],
            )

            data = load_viewer_data(
                event_log_path=paths["event_log"],
                layout_path=paths["layout"],
                schedule_path=paths["schedule"],
                metrics_path=paths["metrics"],
            )

            operation = data.operations[0]
            self.assertEqual(operation.batch_id, "actual:job_actual_001")
            self.assertEqual(operation.batch_job_ids, ("job_actual_001",))
            self.assertEqual(operation.batch_wo_count, 1)
            self.assertEqual(operation.batch_length_sum, 12000.0)

    def test_rejects_generated_schedule_without_batch_fields(self) -> None:
        """generated schedule의 batch 컬럼 누락은 actual replay로 몰래 처리하지 않는다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_schedule(
                paths["schedule"],
                fieldnames=[
                    "job_id",
                    "work_order_no",
                    "algorithm_machine_id",
                    "algorithm_cut_bay",
                    "start_min",
                    "finish_min",
                    "process_min",
                    "plate_length",
                    "thickness",
                    "source_machine_id",
                    "source_cut_bay",
                ],
                rows=[
                    {
                        "job_id": "job_generated_001",
                        "work_order_no": "WO_GENERATED_001",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "start_min": "0",
                        "finish_min": "10",
                        "process_min": "10",
                        "plate_length": "10000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "missing required field batch_"):
                load_viewer_data(
                    event_log_path=paths["event_log"],
                    layout_path=paths["layout"],
                    schedule_path=paths["schedule"],
                    metrics_path=paths["metrics"],
                )

    def test_comparison_viewer_dry_run_loads_two_runs(self) -> None:
        """actual/generated 비교 viewer는 두 입력 묶음을 동시에 검증한다."""

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            left = self._write_common_inputs(base_dir / "left")
            right = self._write_common_inputs(base_dir / "right")
            self._write_generated_two_job_schedule(left["schedule"])
            self._write_generated_two_job_schedule(right["schedule"])

            summary = run_pygame_comparison_from_paths(
                left_event_log_path=left["event_log"],
                left_layout_path=left["layout"],
                left_schedule_path=left["schedule"],
                left_metrics_path=left["metrics"],
                right_event_log_path=right["event_log"],
                right_layout_path=right["layout"],
                right_schedule_path=right["schedule"],
                right_metrics_path=right["metrics"],
                dry_run=True,
            )

            self.assertEqual(summary["left"]["operation_count"], 2)
            self.assertEqual(summary["right"]["operation_count"], 2)
            self.assertEqual(summary["left"]["batch_count"], 1)
            self.assertEqual(summary["right"]["batch_count"], 1)

    def test_comparison_viewer_exports_gif_without_opening_window(self) -> None:
        """비교 viewer는 GUI 창 없이 좌우 비교 GIF를 파일로 저장한다."""

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            left = self._write_common_inputs(base_dir / "left")
            right = self._write_common_inputs(base_dir / "right")
            gif_path = base_dir / "compare.gif"
            self._write_generated_two_job_schedule(left["schedule"])
            self._write_generated_two_job_schedule(right["schedule"])

            summary = run_pygame_comparison_from_paths(
                left_event_log_path=left["event_log"],
                left_layout_path=left["layout"],
                left_schedule_path=left["schedule"],
                left_metrics_path=left["metrics"],
                right_event_log_path=right["event_log"],
                right_layout_path=right["layout"],
                right_schedule_path=right["schedule"],
                right_metrics_path=right["metrics"],
                width=640,
                height=360,
                gif_path=gif_path,
                gif_frames=3,
                gif_duration_ms=80,
            )

            self.assertEqual(summary["left"]["operation_count"], 2)
            self.assertTrue(gif_path.exists())
            self.assertGreater(gif_path.stat().st_size, 0)

    def test_comparison_viewer_rejects_dry_run_with_gif(self) -> None:
        """dry-run은 렌더링하지 않으므로 GIF 저장 옵션과 같이 쓸 수 없다."""

        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            left = self._write_common_inputs(base_dir / "left")
            right = self._write_common_inputs(base_dir / "right")
            self._write_generated_two_job_schedule(left["schedule"])
            self._write_generated_two_job_schedule(right["schedule"])

            with self.assertRaisesRegex(ValueError, "--dry-run cannot be combined"):
                run_pygame_comparison_from_paths(
                    left_event_log_path=left["event_log"],
                    left_layout_path=left["layout"],
                    left_schedule_path=left["schedule"],
                    left_metrics_path=left["metrics"],
                    right_event_log_path=right["event_log"],
                    right_layout_path=right["layout"],
                    right_schedule_path=right["schedule"],
                    right_metrics_path=right["metrics"],
                    dry_run=True,
                    gif_path=base_dir / "compare.gif",
                )

    def test_workload_summary_counts_batch_time_once_and_includes_visible_machines(self) -> None:
        """부하평준화 비교는 batch 점유시간과 W/O 할당개수를 분리해서 계산한다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_two_bay_layout(paths["layout"])
            self._write_schedule(
                paths["schedule"],
                fieldnames=[
                    "job_id",
                    "work_order_no",
                    "algorithm_machine_id",
                    "algorithm_cut_bay",
                    "batch_id",
                    "batch_job_ids",
                    "batch_wo_count",
                    "batch_length_sum",
                    "start_min",
                    "finish_min",
                    "process_min",
                    "plate_length",
                    "thickness",
                    "source_machine_id",
                    "source_cut_bay",
                ],
                rows=[
                    {
                        "job_id": "job_001",
                        "work_order_no": "WO_001",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "14000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                    {
                        "job_id": "job_002",
                        "work_order_no": "WO_002",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "16000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                    {
                        "job_id": "job_003",
                        "work_order_no": "WO_003",
                        "algorithm_machine_id": "PLS31",
                        "algorithm_cut_bay": "23",
                        "batch_id": "B000002",
                        "batch_job_ids": "job_003",
                        "batch_wo_count": "1",
                        "batch_length_sum": "12000",
                        "start_min": "0",
                        "finish_min": "10",
                        "process_min": "10",
                        "plate_length": "12000",
                        "thickness": "13",
                        "source_machine_id": "PLS31",
                        "source_cut_bay": "23",
                    },
                ],
            )
            data = load_viewer_data(
                event_log_path=paths["event_log"],
                layout_path=paths["layout"],
                schedule_path=paths["schedule"],
                metrics_path=paths["metrics"],
            )

            summary = build_workload_summary(data)

            self.assertEqual(summary.machine_work_order_counts["PLS21"], 2)
            self.assertEqual(summary.machine_work_order_counts["PLS22"], 0)
            self.assertEqual(summary.machine_batch_counts["PLS21"], 1)
            self.assertEqual(summary.machine_batch_counts["PLS22"], 0)
            self.assertEqual(summary.machine_workload_minutes["PLS21"], 20.0)
            self.assertEqual(summary.machine_workload_minutes["PLS22"], 0.0)
            self.assertEqual(summary.bay_work_order_counts["22"], 2)
            self.assertEqual(summary.bay_batch_counts["22"], 1)
            self.assertEqual(summary.bay_workload_minutes["22"], 20.0)
            self.assertEqual(summary.machine_workload_imbalance_minutes, 20.0)
            self.assertEqual(summary.bay_work_order_imbalance, 1)
            self.assertEqual(summary.bay_batch_imbalance, 0)

    def test_completion_summary_counts_finished_work_orders_and_batches(self) -> None:
        """현재 playback 시각까지 완료된 W/O와 batch를 Bay/설비별로 계산한다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_two_bay_layout(paths["layout"])
            self._write_schedule(
                paths["schedule"],
                fieldnames=[
                    "job_id",
                    "work_order_no",
                    "algorithm_machine_id",
                    "algorithm_cut_bay",
                    "batch_id",
                    "batch_job_ids",
                    "batch_wo_count",
                    "batch_length_sum",
                    "start_min",
                    "finish_min",
                    "process_min",
                    "plate_length",
                    "thickness",
                    "source_machine_id",
                    "source_cut_bay",
                ],
                rows=[
                    {
                        "job_id": "job_001",
                        "work_order_no": "WO_001",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "14000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                    {
                        "job_id": "job_002",
                        "work_order_no": "WO_002",
                        "algorithm_machine_id": "PLS21",
                        "algorithm_cut_bay": "22",
                        "batch_id": "B000001",
                        "batch_job_ids": "job_001|job_002",
                        "batch_wo_count": "2",
                        "batch_length_sum": "30000",
                        "start_min": "0",
                        "finish_min": "20",
                        "process_min": "20",
                        "plate_length": "16000",
                        "thickness": "13",
                        "source_machine_id": "PLS21",
                        "source_cut_bay": "22",
                    },
                    {
                        "job_id": "job_003",
                        "work_order_no": "WO_003",
                        "algorithm_machine_id": "PLS31",
                        "algorithm_cut_bay": "23",
                        "batch_id": "B000002",
                        "batch_job_ids": "job_003",
                        "batch_wo_count": "1",
                        "batch_length_sum": "12000",
                        "start_min": "0",
                        "finish_min": "10",
                        "process_min": "10",
                        "plate_length": "12000",
                        "thickness": "13",
                        "source_machine_id": "PLS31",
                        "source_cut_bay": "23",
                    },
                ],
            )
            data = load_viewer_data(
                event_log_path=paths["event_log"],
                layout_path=paths["layout"],
                schedule_path=paths["schedule"],
                metrics_path=paths["metrics"],
            )

            summary = build_completion_summary(data, current_time_min=15)

            self.assertEqual(summary.machine_completed_work_orders["PLS21"], 0)
            self.assertEqual(summary.machine_completed_work_orders["PLS31"], 1)
            self.assertEqual(summary.machine_completed_batches["PLS21"], 0)
            self.assertEqual(summary.machine_completed_batches["PLS31"], 1)
            self.assertEqual(summary.bay_completed_work_orders["22"], 0)
            self.assertEqual(summary.bay_completed_work_orders["23"], 1)
            self.assertEqual(summary.bay_completed_batches["22"], 0)
            self.assertEqual(summary.bay_completed_batches["23"], 1)

    def test_run_summary_lines_use_reader_facing_labels(self) -> None:
        """화면 요약은 내부 metric key가 아니라 사람이 읽는 용어를 사용한다."""

        with TemporaryDirectory() as temp_dir:
            paths = self._write_common_inputs(Path(temp_dir))
            self._write_generated_two_job_schedule(paths["schedule"])
            data = load_viewer_data(
                event_log_path=paths["event_log"],
                layout_path=paths["layout"],
                schedule_path=paths["schedule"],
                metrics_path=paths["metrics"],
            )

            lines = build_run_summary_lines(data, build_workload_summary(data))

            joined = "\n".join(lines)
            self.assertIn("Work orders", joined)
            self.assertIn("Leveling rule: max-min", joined)
            self.assertIn("Bay W/O leveling", joined)
            self.assertIn("Machine W/O leveling", joined)
            self.assertIn("Bay Batch leveling", joined)
            self.assertIn("Machine Batch leveling", joined)
            self.assertNotIn("scheduled_jobs", joined)
            self.assertNotIn("machine_load", joined)

    @staticmethod
    def _write_common_inputs(base_dir: Path) -> dict[str, Path]:
        """viewer 입력 4종 중 schedule을 제외한 공통 JSON 파일을 만든다."""

        base_dir.mkdir(parents=True, exist_ok=True)
        event_log = base_dir / "event_log.json"
        layout = base_dir / "factory_layout.json"
        metrics = base_dir / "metrics.json"
        schedule = base_dir / "job_schedule.csv"

        event_log.write_text(json.dumps([{"event_type": "PROCESS_START"}]), encoding="utf-8")
        layout.write_text(
            json.dumps(
                {
                    "bays": [
                        {
                            "bay_id": "22",
                            "machines": [
                                {
                                    "machine_id": "PLS21",
                                    "machine_type": "plasma",
                                    "position": [0, 0],
                                    "parallel_capacity": 1,
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        metrics.write_text(json.dumps({"scheduled_jobs": 1, "makespan_minutes": 35.0}), encoding="utf-8")
        return {"event_log": event_log, "layout": layout, "metrics": metrics, "schedule": schedule}

    @staticmethod
    def _write_schedule(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        """테스트용 schedule CSV를 쓴다."""

        with path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_generated_two_job_schedule(path: Path) -> None:
        """테스트용 generated batch schedule을 만든다."""

        PygameFactoryViewerTest._write_schedule(
            path,
            fieldnames=[
                "job_id",
                "work_order_no",
                "algorithm_machine_id",
                "algorithm_cut_bay",
                "batch_id",
                "batch_job_ids",
                "batch_wo_count",
                "batch_length_sum",
                "start_min",
                "finish_min",
                "process_min",
                "plate_length",
                "thickness",
                "source_machine_id",
                "source_cut_bay",
            ],
            rows=[
                {
                    "job_id": "job_001",
                    "work_order_no": "WO_001_LONG_LABEL_SHOULD_CLIP",
                    "algorithm_machine_id": "PLS21",
                    "algorithm_cut_bay": "22",
                    "batch_id": "B000001",
                    "batch_job_ids": "job_001|job_002",
                    "batch_wo_count": "2",
                    "batch_length_sum": "30000",
                    "start_min": "0",
                    "finish_min": "20",
                    "process_min": "20",
                    "plate_length": "14000",
                    "thickness": "13",
                    "source_machine_id": "PLS21",
                    "source_cut_bay": "22",
                },
                {
                    "job_id": "job_002",
                    "work_order_no": "WO_002_LONG_LABEL_SHOULD_CLIP",
                    "algorithm_machine_id": "PLS21",
                    "algorithm_cut_bay": "22",
                    "batch_id": "B000001",
                    "batch_job_ids": "job_001|job_002",
                    "batch_wo_count": "2",
                    "batch_length_sum": "30000",
                    "start_min": "0",
                    "finish_min": "20",
                    "process_min": "20",
                    "plate_length": "16000",
                    "thickness": "13",
                    "source_machine_id": "PLS21",
                    "source_cut_bay": "22",
                },
            ],
        )

    @staticmethod
    def _write_two_bay_layout(path: Path) -> None:
        """부하평준화 계산 테스트용 2 Bay/3 설비 layout을 만든다."""

        path.write_text(
            json.dumps(
                {
                    "bays": [
                        {
                            "bay_id": "22",
                            "machines": [
                                {"machine_id": "PLS21", "machine_type": "plasma", "parallel_capacity": 1},
                                {"machine_id": "PLS22", "machine_type": "plasma", "parallel_capacity": 1},
                            ],
                        },
                        {
                            "bay_id": "23",
                            "machines": [
                                {"machine_id": "PLS31", "machine_type": "plasma", "parallel_capacity": 1}
                            ],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )


class _FakeFont:
    """Pygame 없이 text fitting을 검증하기 위한 최소 font stub."""

    def __init__(self, pixel_per_char: int) -> None:
        self.pixel_per_char = pixel_per_char

    def size(self, text: str) -> tuple[int, int]:
        return len(text) * self.pixel_per_char, 12


if __name__ == "__main__":
    unittest.main()
