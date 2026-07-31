"""Phase 2 크기×분포 Type validation grid 계약을 검증한다."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import Phase2.merged as phase2_merged
from Phase2.validation_grid import (
    Phase2ValidationProblem,
    build_phase2_validation_grid,
    phase2_validation_grid_contract,
)


class Phase2ValidationGridTests(unittest.TestCase):
    def test_rank_summary_counts_tied_winners_without_forcing_one_source(self) -> None:
        method_rows = [
            self._method_row("P1", 10, 1, "proposed_best_of_k", (0, 10)),
            self._method_row("P1", 10, 1, "lpt_batch", (0, 10)),
            self._method_row("P1", 10, 1, "best_fit_lth", (0, 12)),
            self._method_row("P2", 20, 1, "proposed_best_of_k", (0, 11)),
            self._method_row("P2", 20, 1, "lpt_batch", (0, 9)),
            self._method_row("P2", 20, 1, "best_fit_lth", (0, 13)),
        ]
        parent_rows = [
            self._parent_row("P1", 10, 1, "tie", 1),
            self._parent_row("P2", 20, 1, "loss", 2),
        ]

        ranked = phase2_merged._rank_validation_method_rows(method_rows)
        p1 = {
            row["source"]: row
            for row in ranked
            if row["validation_problem_id"] == "P1"
        }
        self.assertEqual(p1["proposed_best_of_k"]["method_rank"], 1)
        self.assertEqual(p1["lpt_batch"]["method_rank"], 1)
        self.assertEqual(p1["best_fit_lth"]["method_rank"], 3)
        self.assertEqual(p1["proposed_best_of_k"]["rank1_tie_count"], 2)

        summaries = phase2_merged._validation_rank_summary_rows(
            parent_rows,
            ranked,
        )
        overall = next(row for row in summaries if row["scope"] == "overall")
        self.assertEqual(overall["problem_count"], 2)
        self.assertEqual(overall["proposed_win_count"], 0)
        self.assertEqual(overall["proposed_tie_count"], 1)
        self.assertEqual(overall["proposed_loss_count"], 1)
        self.assertEqual(overall["proposed_rank1_count"], 1)
        self.assertEqual(overall["proposed_mean_rank"], 1.5)
        self.assertEqual(
            overall["winner_counts_json"],
            '{"best_fit_lth": 0, "lpt_batch": 2, "proposed_best_of_k": 1}',
        )
        self.assertEqual(
            overall["sole_winner_counts_json"],
            '{"best_fit_lth": 0, "lpt_batch": 1, "proposed_best_of_k": 0}',
        )
        self.assertEqual(
            overall["tied_winner_counts_json"],
            '{"best_fit_lth": 0, "lpt_batch": 1, "proposed_best_of_k": 1}',
        )
        self.assertEqual(
            {(row["scope"], row["block_count"]) for row in summaries},
            {
                ("overall", ""),
                ("block_size", 10),
                ("block_size", 20),
                ("distribution_type", ""),
            },
        )

    def test_grid_crosses_inclusive_block_sizes_with_fixed_distribution_types(self) -> None:
        problems = build_phase2_validation_grid(
            problem_factory=self._problem_factory,
            min_blocks=10,
            max_blocks=20,
            block_gap=10,
            type_count=3,
            seed=17,
        )

        self.assertEqual(len(problems), 6)
        self.assertEqual(
            [(problem.block_count, problem.distribution_type) for problem in problems],
            [
                (10, 1),
                (10, 2),
                (10, 3),
                (20, 1),
                (20, 2),
                (20, 3),
            ],
        )
        for distribution_type in (1, 2, 3):
            targets = {
                tuple(problem.target_distribution.items())
                for problem in problems
                if problem.distribution_type == distribution_type
            }
            self.assertEqual(len(targets), 1)
        self.assertEqual(
            {problem.metadata["physical_block_count"] for problem in problems},
            {10, 20},
        )

    def test_grid_rejects_non_divisible_inclusive_range(self) -> None:
        with self.assertRaises(RuntimeError):
            build_phase2_validation_grid(
                problem_factory=self._problem_factory,
                min_blocks=31,
                max_blocks=60,
                block_gap=5,
                type_count=3,
                seed=17,
            )

    def test_contract_changes_when_fixed_job_payload_changes(self) -> None:
        first_jobs = {
            "WO_A": self._job("WO_A", "P1::NP::B1", 10.0),
        }
        second_jobs = {
            "WO_A": self._job("WO_A", "P1::NP::B1", 11.0),
        }
        first = self._validation_problem("B010_T01", 10, 1, 101, first_jobs)
        second = self._validation_problem("B010_T01", 10, 1, 101, second_jobs)

        self.assertNotEqual(
            phase2_validation_grid_contract((first,)),
            phase2_validation_grid_contract((second,)),
        )

    def test_training_writes_fixed_problem_and_checkpoint_evaluation_tree(self) -> None:
        jobs_10 = {
            "WO_10_A": self._job("WO_10_A", "P1::NP::B10A", 30.0),
            "WO_10_B": self._job("WO_10_B", "P1::NP::B10B", 10.0),
        }
        jobs_20 = {
            "WO_20_A": self._job("WO_20_A", "P1::NP::B20A", 20.0),
            "WO_20_B": self._job("WO_20_B", "P1::NP::B20B", 5.0),
        }
        validation_problems = (
            self._validation_problem("B010_T01", 10, 1, 101, jobs_10),
            self._validation_problem("B010_T02", 10, 2, 102, jobs_10),
            self._validation_problem("B020_T01", 20, 1, 201, jobs_20),
            self._validation_problem("B020_T02", 20, 2, 202, jobs_20),
        )
        assignments = {
            **{job.block_set_id: "22" for job in jobs_10.values()},
            **{job.block_set_id: "23" for job in jobs_20.values()},
        }

        def assignment_builder(jobs, _seed):
            return {
                job.block_set_id: assignments[job.block_set_id]
                for job in jobs.values()
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = phase2_merged.train_phase2_batch_machine_self_labeling(
                jobs=jobs_10,
                machines=phase2_merged.build_mixed_phase2_training_machines(),
                phase1_assignments={},
                output_dir=temp_dir,
                episodes=2,
                lr=1e-3,
                hidden_dim=16,
                seed=11,
                heuristic_algorithms=("lpt_batch",),
                rollout_samples=1,
                max_wo_count=1,
                max_length_sum=55_000.0,
                action_pool_limit=None,
                validation_every=1,
                validation_episodes=2,
                validation_rollout_samples=1,
                validation_problems=validation_problems,
                checkpoint_every=0,
                device="cpu",
                phase1_assignment_builder=assignment_builder,
                phase1_bay_ids=("22", "23", "24", "25", "trans"),
                phase1_bay_capacity_weights={
                    "22": 4.0,
                    "23": 3.0,
                    "24": 4.0,
                    "25": 2.0,
                    "trans": 2.0,
                },
            )

            validation_root = Path(summary["validation_root"])
            self.assertTrue(
                (validation_root / "problems" / "blocks_010" / "type_01" / "problem_metadata.json").is_file()
            )
            for train_episode in (1, 2):
                evaluation_root = validation_root / "evaluations" / f"ep_{train_episode:06d}"
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_010"
                        / "type_01"
                        / "bay_22"
                        / "candidate_summary.csv"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_020"
                        / "type_01"
                        / "bay_23"
                        / "proposed_solution.csv"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_010"
                        / "type_01"
                        / "bay_trans"
                        / "problem_metadata.json"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_010"
                        / "by_type"
                        / "parent"
                        / "makespan_by_type.png"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_010"
                        / "by_type"
                        / "bay_22"
                        / "cut_length_gap_by_type.png"
                    ).is_file()
                )
                self.assertFalse(
                    (
                        evaluation_root
                        / "blocks_010"
                        / "by_type"
                        / "bay_23"
                    ).exists()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "blocks_020"
                        / "by_type"
                        / "bay_23"
                        / "occupancy_gap_by_type.png"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        evaluation_root
                        / "aggregate"
                        / "makespan_by_block_size_boxplot.png"
                    ).is_file()
                )
                self.assertTrue(
                    (evaluation_root / "aggregate" / "method_summary_by_block_size.csv").is_file()
                )
                rank_summary_path = (
                    evaluation_root
                    / "aggregate"
                    / "validation_rank_summary.csv"
                )
                self.assertTrue(rank_summary_path.is_file())
                with rank_summary_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as file:
                    rank_summary_rows = list(csv.DictReader(file))
                self.assertEqual(
                    {
                        (row["scope"], row["block_count"], row["distribution_type"])
                        for row in rank_summary_rows
                    },
                    {
                        ("overall", "", ""),
                        ("block_size", "10", ""),
                        ("block_size", "20", ""),
                        ("distribution_type", "", "1"),
                        ("distribution_type", "", "2"),
                    },
                )
                with (
                    evaluation_root
                    / "aggregate"
                    / "method_problem_scores.csv"
                ).open("r", encoding="utf-8", newline="") as file:
                    method_score_rows = list(csv.DictReader(file))
                self.assertTrue(method_score_rows)
                self.assertTrue(
                    all(
                        row["method_rank"]
                        and row["is_rank1"]
                        and row["rank1_tie_count"]
                        for row in method_score_rows
                    )
                )
                with (
                    evaluation_root
                    / "blocks_010"
                    / "by_type"
                    / "parent"
                    / "method_summary_by_type.csv"
                ).open("r", encoding="utf-8", newline="") as file:
                    type_rows = list(csv.DictReader(file))
                self.assertEqual(len(type_rows), 4)
                self.assertEqual(
                    {row["distribution_type"] for row in type_rows},
                    {"1", "2"},
                )
                self.assertFalse(
                    any(
                        "hard_violation" in path.name
                        for path in evaluation_root.rglob("*.png")
                    )
                )

            with Path(summary["validation_parent_summary_csv"]).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                parent_rows = list(csv.DictReader(file))
            self.assertEqual(len(parent_rows), 8)
            self.assertEqual(
                {(row["block_count"], row["distribution_type"]) for row in parent_rows},
                {("10", "1"), ("10", "2"), ("20", "1"), ("20", "2")},
            )
            with Path(summary["validation_candidate_summary_csv"]).open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                latest_candidate_rows = list(csv.DictReader(file))
            self.assertTrue(latest_candidate_rows)
            self.assertEqual(
                {row["train_episode"] for row in latest_candidate_rows},
                {"2"},
            )
            self.assertTrue((validation_root / "best_checkpoint.json").is_file())
            self.assertTrue(Path(summary["best_checkpoint_path"]).is_file())
            rank_history_path = validation_root / "validation_rank_history.csv"
            self.assertTrue(rank_history_path.is_file())
            with rank_history_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as file:
                rank_history_rows = list(csv.DictReader(file))
            self.assertEqual(len(rank_history_rows), 10)
            self.assertEqual(
                {
                    row["train_episode"]
                    for row in rank_history_rows
                    if row["scope"] == "overall"
                },
                {"1", "2"},
            )
            self.assertTrue(
                (
                    validation_root
                    / "validation_overall_winner_count_history.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    validation_root
                    / "validation_proposed_mean_rank_history.png"
                ).is_file()
            )

            changed_validation_problems = (
                self._validation_problem("B010_T01", 10, 1, 999, jobs_10),
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "validation grid mismatch",
            ):
                phase2_merged.train_phase2_batch_machine_self_labeling(
                    jobs=jobs_10,
                    machines=phase2_merged.build_mixed_phase2_training_machines(),
                    phase1_assignments={},
                    output_dir=Path(temp_dir) / "mismatched_resume",
                    episodes=3,
                    lr=1e-3,
                    hidden_dim=16,
                    seed=11,
                    heuristic_algorithms=("lpt_batch",),
                    rollout_samples=1,
                    max_wo_count=1,
                    max_length_sum=55_000.0,
                    action_pool_limit=None,
                    validation_every=1,
                    validation_episodes=2,
                    validation_rollout_samples=1,
                    validation_problems=changed_validation_problems,
                    checkpoint_every=0,
                    device="cpu",
                    phase1_assignment_builder=assignment_builder,
                    phase1_bay_ids=("22", "23", "24", "25", "trans"),
                    phase1_bay_capacity_weights={
                        "22": 4.0,
                        "23": 3.0,
                        "24": 4.0,
                        "25": 2.0,
                        "trans": 2.0,
                    },
                    resume_checkpoint=summary["checkpoint_path"],
                )

    @staticmethod
    def _problem_factory(block_count: int, seed: int):
        families = ("NP", "NC", "FN", "FL")
        jobs = {}
        combinations = []
        for index in range(block_count):
            family = families[(seed + index) % len(families)]
            combinations.append((family,))
            job_id = f"WO_{block_count}_{seed}_{index}"
            jobs[job_id] = SimpleNamespace(
                job_id=job_id,
                block_set_id=f"P{index}::{family}::B{index}",
                family=family,
                plate_length=8_000.0 + ((seed + index) % 5) * 1_000.0,
                plate_width=2_000.0 + ((seed + index) % 4) * 1_000.0,
                thickness=12.0,
                cut_length=100.0 + ((seed + index) % 7) * 30.0,
                bevel_quantity=(seed + index) % 6,
                base_stage_minutes={"cut": 10.0 + ((seed + index) % 9)},
            )
        return {
            "jobs": jobs,
            "metadata": {
                "physical_block_count": block_count,
                "block_count": block_count,
                "job_count": len(jobs),
                "series_combinations": combinations,
                "seed": seed,
            },
        }

    @staticmethod
    def _validation_problem(
        problem_id: str,
        block_count: int,
        distribution_type: int,
        seed: int,
        jobs,
    ) -> Phase2ValidationProblem:
        return Phase2ValidationProblem(
            problem_id=problem_id,
            block_count=block_count,
            distribution_type=distribution_type,
            generation_seed=seed,
            target_distribution={"wo_per_block": 0.5},
            normalized_distribution={"wo_per_block": 0.5},
            actual_distribution={"wo_per_block": len(jobs) / block_count},
            jobs=jobs,
            metadata={
                "physical_block_count": block_count,
                "block_count": len(jobs),
                "job_count": len(jobs),
                "seed": seed,
            },
        )

    @staticmethod
    def _method_row(
        problem_id: str,
        block_count: int,
        distribution_type: int,
        source: str,
        score: tuple[int, int],
    ) -> dict:
        return {
            "train_episode": 100,
            "validation_episode": 1,
            "validation_problem_id": problem_id,
            "block_count": block_count,
            "distribution_type": distribution_type,
            "generation_seed": 1,
            "source": source,
            "score_json": str(list(score)),
        }

    @staticmethod
    def _parent_row(
        problem_id: str,
        block_count: int,
        distribution_type: int,
        relation: str,
        rank: int,
    ) -> dict:
        return {
            "train_episode": 100,
            "validation_problem_id": problem_id,
            "block_count": block_count,
            "distribution_type": distribution_type,
            "proposed_vs_best_heuristic": relation,
            "proposed_best_rank": rank,
        }

    @staticmethod
    def _job(job_id: str, block_set_id: str, tact: float):
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            plate_length=10_000.0,
            plate_width=2_000.0,
            thickness=13.0,
            cut_length=100.0,
            bevel_quantity=1,
            cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": tact},
        )


if __name__ == "__main__":
    unittest.main()
