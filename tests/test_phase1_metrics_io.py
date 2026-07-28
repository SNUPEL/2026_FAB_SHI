"""Phase 1 학습 지표 파일의 append 기록 계약을 검증한다."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from Phase1.pair_self_labeling import (
    _CANDIDATE_SUMMARY_FIELDS,
    _METRICS_FIELDS,
    _SUBPROBLEM_METRICS_FIELDS,
    _append_csv_rows,
    _append_jsonl_rows,
    _truncate_history_files,
)


class AppendCsvRowsTests(unittest.TestCase):
    def test_creates_header_once_and_appends_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            fields = ("episode", "loss")

            _append_csv_rows(path, fields, [{"episode": 1, "loss": 0.5}])
            _append_csv_rows(path, fields, [{"episode": 2, "loss": 0.4}])

            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual([row["episode"] for row in rows], ["1", "2"])
            self.assertEqual([row["loss"] for row in rows], ["0.5", "0.4"])
            self.assertEqual(path.read_text(encoding="utf-8-sig").count("episode,loss"), 1)

    def test_missing_key_becomes_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            _append_csv_rows(path, ("episode", "loss"), [{"episode": 1}])

            with path.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(rows[0]["loss"], "")

    def test_empty_rows_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            with self.assertRaises(RuntimeError):
                _append_csv_rows(path, ("episode",), [])


class AppendJsonlRowsTests(unittest.TestCase):
    def test_appends_one_line_per_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_action_table.jsonl"

            _append_jsonl_rows(path, [{"episode": 1, "step": 0}])
            _append_jsonl_rows(path, [{"episode": 2, "step": 0}, {"episode": 2, "step": 1}])

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual([json.loads(line)["episode"] for line in lines], [1, 2, 2])

    def test_empty_rows_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_action_table.jsonl"
            with self.assertRaises(RuntimeError):
                _append_jsonl_rows(path, [])


class FieldConstantTests(unittest.TestCase):
    def test_metrics_fields_match_documented_schema(self) -> None:
        self.assertEqual(_METRICS_FIELDS[0], "episode")
        self.assertIn("loss", _METRICS_FIELDS)
        self.assertIn("score_json", _METRICS_FIELDS)
        self.assertIn("subproblem_count", _METRICS_FIELDS)

    def test_subproblem_fields_carry_subproblem_id(self) -> None:
        self.assertIn("subproblem_id", _SUBPROBLEM_METRICS_FIELDS)

    def test_candidate_fields_carry_is_best(self) -> None:
        self.assertIn("is_best", _CANDIDATE_SUMMARY_FIELDS)


class TruncateHistoryFilesTests(unittest.TestCase):
    def _write_csv(self, path: Path, fields, rows) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_keeps_only_rows_before_start_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_csv(
                out / "metrics.csv",
                _METRICS_FIELDS,
                [{"episode": 1}, {"episode": 2}, {"episode": 3}],
            )
            self._write_csv(
                out / "subproblem_metrics.csv",
                _SUBPROBLEM_METRICS_FIELDS,
                [{"episode": 1}, {"episode": 2}],
            )

            metrics_rows, subproblem_rows = _truncate_history_files(
                out, start_episode=3, write_candidate_summary=False
            )

            self.assertEqual([row["episode"] for row in metrics_rows], ["1", "2"])
            self.assertEqual([row["episode"] for row in subproblem_rows], ["1", "2"])
            with (out / "metrics.csv").open(encoding="utf-8-sig", newline="") as file:
                on_disk = list(csv.DictReader(file))
            self.assertEqual([row["episode"] for row in on_disk], ["1", "2"])

    def test_fresh_run_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._write_csv(out / "metrics.csv", _METRICS_FIELDS, [{"episode": 1}])
            (out / "best_action_table.jsonl").write_text(
                json.dumps({"episode": 1}) + "\n", encoding="utf-8"
            )

            metrics_rows, subproblem_rows = _truncate_history_files(
                out, start_episode=1, write_candidate_summary=False
            )

            self.assertEqual(metrics_rows, [])
            self.assertEqual(subproblem_rows, [])
            self.assertFalse((out / "metrics.csv").exists())
            self.assertFalse((out / "best_action_table.jsonl").exists())

    def test_missing_files_are_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_rows, subproblem_rows = _truncate_history_files(
                Path(tmp), start_episode=5, write_candidate_summary=True
            )
            self.assertEqual(metrics_rows, [])
            self.assertEqual(subproblem_rows, [])


class Phase1CliFlagTests(unittest.TestCase):
    def test_write_candidate_summary_defaults_to_false(self) -> None:
        from main import build_parser

        args = build_parser().parse_args(
            [
                "phase1-train-pair-self-labeling",
                "--config", "config_mixed.yaml",
                "--output-dir", "output/tmp",
            ]
        )
        self.assertFalse(args.write_candidate_summary)

    def test_write_candidate_summary_can_be_enabled(self) -> None:
        from main import build_parser

        args = build_parser().parse_args(
            [
                "phase1-train-pair-self-labeling",
                "--config", "config_mixed.yaml",
                "--output-dir", "output/tmp",
                "--write-candidate-summary",
            ]
        )
        self.assertTrue(args.write_candidate_summary)


if __name__ == "__main__":
    unittest.main()
