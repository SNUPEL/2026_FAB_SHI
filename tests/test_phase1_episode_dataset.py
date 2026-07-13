"""Phase 1 variable-size episode dataset tests."""

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from Phase1.imitation import train_phase1_pointer_imitation
from Utils.phase1.phase1_episode_dataset import (
    build_phase1_actual_workday_jobs,
    build_phase1_candidate_workbook_jobs,
    build_phase1_episode_jobs,
    write_phase1_episode_dataset,
)


class Phase1EpisodeDatasetTest(unittest.TestCase):
    """Synthetic Phase 1 episodes should vary block count and train/eval cleanly."""

    def test_builds_variable_size_train_test_action_tables(self) -> None:
        """Each episode samples a block count and writes two action rows per block."""

        actual_blocks = self._actual_blocks()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = write_phase1_episode_dataset(
                actual_blocks=actual_blocks,
                output_dir=temp_dir,
                episode_count=4,
                min_blocks=2,
                max_blocks=5,
                train_ratio=0.5,
                bay_ids=["22", "23", "24"],
                algorithm="long_cut_preferred_balanced",
                seed=7,
                noise_ratio=0.0,
            )

            manifest = json.loads(Path(result["manifest_json"]).read_text(encoding="utf-8"))
            train_lines = Path(result["train_action_table_jsonl"]).read_text(encoding="utf-8").splitlines()
            test_lines = Path(result["test_action_table_jsonl"]).read_text(encoding="utf-8").splitlines()

        episode_block_counts = manifest["episode_block_counts"]
        self.assertEqual(manifest["episode_count"], 4)
        self.assertTrue(all(2 <= count <= 5 for count in episode_block_counts))
        self.assertGreater(len(set(episode_block_counts)), 1)
        self.assertEqual(manifest["total_action_count"], 2 * sum(episode_block_counts))
        self.assertGreater(len(train_lines), 0)
        self.assertGreater(len(test_lines), 0)
        self.assertEqual(len(train_lines) + len(test_lines), manifest["total_action_count"])

    def test_builds_variable_episode_jobs_for_self_labeling(self) -> None:
        """Self-labeling should consume changing jobs, not one fixed config."""

        episodes = build_phase1_episode_jobs(
            actual_blocks=self._actual_blocks(),
            episode_count=4,
            min_blocks=2,
            max_blocks=5,
            seed=17,
            noise_ratio=0.0,
        )

        block_counts = [episode["block_count"] for episode in episodes]
        self.assertEqual(len(episodes), 4)
        self.assertGreater(len(set(block_counts)), 1)
        self.assertTrue(all(2 <= count <= 5 for count in block_counts))
        self.assertTrue(all(episode["jobs"] for episode in episodes))
        self.assertEqual([episode["episode_id"] for episode in episodes], ["EP00001", "EP00002", "EP00003", "EP00004"])

    def test_builds_hard_case_episode_jobs_with_lower_steel_cut_correlation(self) -> None:
        """Hard-case sampling should expose episodes where steel and cut length diverge."""

        episodes = build_phase1_episode_jobs(
            actual_blocks=self._actual_blocks(),
            episode_count=3,
            min_blocks=6,
            max_blocks=6,
            seed=21,
            noise_ratio=0.0,
            hard_case_ratio=1.0,
            hard_case_mode="cut_shuffle",
            hard_case_target_corr=0.85,
            hard_case_max_attempts=20,
        )

        self.assertEqual(len(episodes), 3)
        for episode in episodes:
            self.assertEqual(episode["case_type"], "hard_cut_shuffle")
            self.assertEqual(episode["hard_case_mode"], "cut_shuffle")
            self.assertLessEqual(episode["hard_case_corr_steel_cut_after"], 0.85)
            if episode["hard_case_corr_steel_cut_before"] > 0.85:
                self.assertLess(episode["hard_case_corr_steel_cut_after"], episode["hard_case_corr_steel_cut_before"])

    def test_train_imitation_reports_eval_metrics_from_holdout_episodes(self) -> None:
        """The trainer should train on train episodes and report holdout accuracy."""

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = write_phase1_episode_dataset(
                actual_blocks=self._actual_blocks(),
                output_dir=Path(temp_dir) / "dataset",
                episode_count=4,
                min_blocks=2,
                max_blocks=4,
                train_ratio=0.5,
                bay_ids=["22", "23", "24"],
                algorithm="long_cut_preferred_balanced",
                seed=11,
                noise_ratio=0.0,
            )
            summary = train_phase1_pointer_imitation(
                action_table_path=dataset["train_action_table_jsonl"],
                output_dir=Path(temp_dir) / "train",
                eval_action_table_path=dataset["test_action_table_jsonl"],
                epochs=2,
                lr=0.001,
                hidden_dim=16,
                seed=0,
            )

        self.assertIn("eval_accuracy", summary)
        self.assertIn("eval_select_block_accuracy", summary)
        self.assertIn("eval_select_bay_accuracy", summary)
        self.assertGreaterEqual(summary["eval_accuracy"], 0.0)
        self.assertLessEqual(summary["eval_accuracy"], 1.0)
        self.assertGreater(summary["eval_example_count"], 0)

    def test_builds_actual_workday_jobs_with_0800_cutoff_and_bay_filter(self) -> None:
        """Actual validation should use 08:00 workday and remove blocks touching Bay 25."""

        rows = pd.DataFrame(
            [
                self._actual_wo_row("P1", "B1", "WO1", "202604070801", "22", 3, 100.0, 1),
                self._actual_wo_row("P1", "B1", "WO2", "202604080759", "23", 4, 200.0, 2),
                self._actual_wo_row("P2", "B2", "WO3", "202604080800", "24", 5, 300.0, 3),
                self._actual_wo_row("P3", "B3", "WO4", "202604070900", "25", 6, 400.0, 4),
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "actual.csv"
            rows.to_csv(source, index=False, encoding="utf-8-sig")
            payloads = build_phase1_actual_workday_jobs(
                source_path=source,
                workdays=["20260407", "20260408"],
                bay_ids=["22", "23", "24"],
                gyel="NP",
            )

        first_jobs = payloads[0]["jobs"]
        second_jobs = payloads[1]["jobs"]
        self.assertEqual(payloads[0]["metadata"]["validation_source"], "actual_8days")
        self.assertEqual(set(first_jobs), {"WO1", "WO2"})
        self.assertEqual(set(second_jobs), {"WO3"})
        self.assertNotIn("WO4", first_jobs)

    def test_builds_actual_8days_from_candidate_workbook(self) -> None:
        """Fixed Phase 1 actual_8days validation should use the candidate block workbook grain."""

        rows = pd.DataFrame(
            [
                {
                    "PROJ_NO": "P1",
                    "BLK_NO": "B1",
                    "LTH": 10000,
                    "THK": 13,
                    "CUT_LTH": 900.0,
                    "STL_QTY": 6,
                    "BV_QTY": 2,
                    "CUT_BAY": 22,
                },
                {
                    "PROJ_NO": "P2",
                    "BLK_NO": "B2",
                    "LTH": 11000,
                    "THK": 16,
                    "CUT_LTH": 1200.0,
                    "STL_QTY": 8,
                    "BV_QTY": 5,
                    "CUT_BAY": 24,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "candidate.xlsx"
            with pd.ExcelWriter(source) as writer:
                rows.to_excel(writer, sheet_name="20260407_BLK", index=False)
            payloads = build_phase1_candidate_workbook_jobs(
                candidate_path=source,
                workdays=["20260407"],
                bay_ids=["22", "23", "24"],
            )

        payload = payloads[0]
        self.assertEqual(payload["metadata"]["validation_source"], "actual_8days")
        self.assertEqual(payload["metadata"]["evaluation_input_type"], "candidate_workbook")
        self.assertEqual(payload["metadata"]["block_count"], 2)
        self.assertEqual({job.block_set_id for job in payload["jobs"].values()}, {"P1::B1", "P2::B2"})
        self.assertEqual({job.source_cut_bay for job in payload["jobs"].values()}, {"22", "24"})

    @staticmethod
    def _actual_blocks() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"LTH": 1000, "THK": 12, "CUT_LTH": 100, "MARK_LTH": 20, "STL_QTY": 3, "BV_QTY": 1},
                {"LTH": 1100, "THK": 13, "CUT_LTH": 1300, "MARK_LTH": 25, "STL_QTY": 8, "BV_QTY": 2},
                {"LTH": 1200, "THK": 14, "CUT_LTH": 800, "MARK_LTH": 30, "STL_QTY": 5, "BV_QTY": 9},
                {"LTH": 1300, "THK": 15, "CUT_LTH": 400, "MARK_LTH": 35, "STL_QTY": 6, "BV_QTY": 3},
                {"LTH": 1400, "THK": 16, "CUT_LTH": 200, "MARK_LTH": 40, "STL_QTY": 4, "BV_QTY": 7},
                {"LTH": 1500, "THK": 17, "CUT_LTH": 600, "MARK_LTH": 45, "STL_QTY": 7, "BV_QTY": 4},
            ]
        )

    @staticmethod
    def _actual_wo_row(
        project_no: str,
        block_no: str,
        wo_no: str,
        actual_start: str,
        cut_bay: str,
        steel_quantity: int,
        cut_length: float,
        bevel_quantity: int,
    ) -> dict:
        return {
            "PROJ_NO": project_no,
            "BLK_NO": block_no,
            "WK_ORD_NO": wo_no,
            "GYEL": "NP",
            "RT_CUT_ST_DTM": actual_start,
            "CUT_BAY": cut_bay,
            "LTH": 10000,
            "THK": 13,
            "STL_QTY": steel_quantity,
            "CUT_LTH": cut_length,
            "BV_QTY": bevel_quantity,
        }


if __name__ == "__main__":
    unittest.main()
