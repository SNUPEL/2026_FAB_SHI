"""MIXED Phase 1 on-the-fly episode 계약 회귀시험."""

from __future__ import annotations

import unittest

from pandas.testing import assert_frame_equal

from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs


class Phase1EpisodeDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.first = build_phase1_episode_jobs(
            episode_count=2,
            min_blocks=3,
            max_blocks=4,
            seed=20260715,
            verbose=False,
        )
        cls.second = build_phase1_episode_jobs(
            episode_count=2,
            min_blocks=3,
            max_blocks=4,
            seed=20260715,
            verbose=False,
        )

    def test_same_seed_reproduces_exact_mixed_episode(self) -> None:
        for first, second in zip(self.first, self.second):
            self.assertEqual(first["metadata"], second["metadata"])
            self.assertEqual(tuple(first["jobs"]), tuple(second["jobs"]))
            assert_frame_equal(first["blocks"], second["blocks"], check_exact=True)
            assert_frame_equal(first["wo_df"], second["wo_df"], check_exact=True)

    def test_physical_block_series_block_and_wo_identity_are_consistent(self) -> None:
        for episode in self.first:
            blocks = episode["blocks"]
            work_orders = episode["wo_df"]
            metadata = episode["metadata"]

            physical_count = blocks.groupby(["PROJ_NO", "BLK_NO"]).ngroups
            series_block_count = blocks.groupby(["PROJ_NO", "GYEL", "BLK_NO"]).ngroups
            self.assertEqual(physical_count, metadata["physical_block_count"])
            self.assertEqual(series_block_count, metadata["block_count"])
            self.assertEqual(len(work_orders), metadata["job_count"])
            self.assertEqual(len(episode["jobs"]), metadata["job_count"])
            self.assertEqual(int(blocks["WO_QTY"].sum()), metadata["job_count"])
            self.assertEqual(
                work_orders.groupby(["PROJ_NO", "GYEL", "BLK_NO"]).ngroups,
                series_block_count,
            )
            self.assertTrue(
                all(len(str(job.block_set_id).split("::")) == 3 for job in episode["jobs"].values())
            )

    def test_episode_exposes_only_joint_five_bay_contract(self) -> None:
        expected_weights = {
            "22": 4.0,
            "23": 3.0,
            "24": 4.0,
            "25": 2.0,
            "trans": 2.0,
        }
        for episode in self.first:
            self.assertEqual(
                episode["case_type"],
                "mixed_physical_block_joint_distribution",
            )
            self.assertEqual(episode["bay_ids"], list(expected_weights))
            self.assertEqual(episode["bay_capacity_weights"], expected_weights)
            self.assertEqual(
                len(episode["series_combinations"]),
                episode["physical_block_count"],
            )

    def test_invalid_sampling_range_fails_without_fallback(self) -> None:
        with self.assertRaises(RuntimeError):
            build_phase1_episode_jobs(0, 1, 1, verbose=False)
        with self.assertRaises(RuntimeError):
            build_phase1_episode_jobs(1, 3, 2, verbose=False)


if __name__ == "__main__":
    unittest.main()
