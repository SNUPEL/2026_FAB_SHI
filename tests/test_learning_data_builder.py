"""학습데이터 package 생성 검증."""

import json
import tempfile
import unittest
from pathlib import Path

from Utils.learning.learning_data_builder import build_learning_data_package


class LearningDataBuilderTest(unittest.TestCase):
    """계층형 trace들을 학습용 manifest로 묶는 helper를 검증한다."""

    def test_package_records_config_heuristic_and_algorithm_plan(self) -> None:
        """실적 기반 smoke dataset은 label 출처와 알고리즘 계획을 명시해야 한다."""

        built_envs = []

        def fake_env_builder(config_path: str):
            built_envs.append(config_path)
            return {"config_path": config_path}

        def fake_trace_exporter(env, heuristic_name: str, max_actions: int, output_dir: str):
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            trace_path = output_path / "hierarchical_trace.csv"
            table_path = output_path / "hierarchical_action_table.jsonl"
            summary_path = output_path / "summary.json"
            trace_path.write_text("hierarchical_step,phase\n0,SELECT_MACHINE\n", encoding="utf-8")
            table_path.write_text('{"action_id_table":[{"index":0}]}\n', encoding="utf-8")
            summary = {
                "scheduled_jobs": 5,
                "unscheduled_jobs": [],
                "hierarchical_step_count": 9,
                "flat_decision_count": 6,
                "phase_counts": {"SELECT_MACHINE": 3, "SELECT_WO": 6},
                "trace_csv": str(trace_path),
                "action_table_jsonl": str(table_path),
                "summary_json": str(summary_path),
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return summary

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config_np_100.yaml"
            config_path.write_text("paths: {}\n", encoding="utf-8")

            result = build_learning_data_package(
                config_paths=[str(config_path)],
                heuristics=["spt", "balanced_batch_count"],
                output_dir=str(Path(temp_dir) / "learning_data"),
                max_actions=64,
                data_role="actual_source_smoke",
                algorithm_plan=["self_labeling", "ppo", "reinforce"],
                env_builder=fake_env_builder,
                trace_exporter=fake_trace_exporter,
            )

            self.assertEqual(built_envs, [str(config_path), str(config_path)])
            self.assertEqual(result["dataset_count"], 2)
            self.assertEqual(result["data_role"], "actual_source_smoke")
            self.assertEqual(result["label_source"], "heuristic_pseudo_label")
            self.assertEqual(result["algorithm_plan"], ["self_labeling", "ppo", "reinforce"])

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["datasets"]), 2)
            self.assertTrue(all(row["scheduled_jobs"] == 5 for row in manifest["datasets"]))
            self.assertTrue(all(Path(row["trace_csv"]).exists() for row in manifest["datasets"]))

    def test_unknown_data_role_fails_without_fallback(self) -> None:
        """허용되지 않은 data_role은 조용히 기본값으로 대체하지 않는다."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("paths: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_learning_data_package(
                    config_paths=[str(config_path)],
                    heuristics=["spt"],
                    output_dir=str(Path(temp_dir) / "learning_data"),
                    max_actions=64,
                    data_role="unknown_role",
                    algorithm_plan=["self_labeling"],
                    env_builder=lambda _: object(),
                    trace_exporter=lambda **_: {},
                )


if __name__ == "__main__":
    unittest.main()
