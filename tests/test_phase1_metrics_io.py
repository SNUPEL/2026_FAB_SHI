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


if __name__ == "__main__":
    unittest.main()
