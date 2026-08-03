"""Phase 2 `--eval-only` 재평가가 비교 근거로 쓸 수 있는지 검증한다.

여러 사람의 arm을 한 머신에서 다시 채점할 때, 채점 조건이 arm의 학습 설정(temperature
스케줄)에 딸려 흔들리면 같은 가중치도 다르게 평가된다. 여기서는 `validation_temperature`가
그 결합을 끊는지, 그리고 eval 산출물이 자기 계약을 스스로 증명하는지를 고정한다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import Phase2.merged as phase2_merged
from Phase2.validation_grid import Phase2ValidationProblem


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


def _validation_problem(problem_id: str, block_count: int, distribution_type: int, seed: int, jobs):
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
            "problem_id": problem_id,
            "distribution_type": distribution_type,
            "generation_seed": seed,
        },
    )


class Phase2EvalOnlyContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = {
            "WO_A": _job("WO_A", "P1::NP::BA", 30.0),
            "WO_B": _job("WO_B", "P1::NP::BB", 10.0),
            "WO_C": _job("WO_C", "P1::NP::BC", 20.0),
        }
        self.validation_problems = (
            _validation_problem("M010_T01", 10, 1, 101, self.jobs),
            _validation_problem("M010_T02", 10, 2, 102, self.jobs),
        )
        self.assignments = {job.block_set_id: "22" for job in self.jobs.values()}

    def _assignment_builder(self, jobs, _seed):
        return {job.block_set_id: self.assignments[job.block_set_id] for job in jobs.values()}

    def _run(self, output_dir, **overrides):
        kwargs = dict(
            jobs=self.jobs,
            machines=phase2_merged.build_mixed_phase2_training_machines(),
            phase1_assignments={},
            output_dir=output_dir,
            episodes=2,
            lr=1e-3,
            hidden_dim=16,
            seed=11,
            heuristic_algorithms=("lpt_batch",),
            rollout_samples=3,
            max_wo_count=2,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            validation_every=1,
            validation_episodes=2,
            validation_rollout_samples=3,
            validation_problems=self.validation_problems,
            checkpoint_every=1,
            device="cpu",
            phase1_assignment_builder=self._assignment_builder,
            phase1_bay_ids=("22", "23", "24", "25", "trans"),
            phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0, "25": 2.0, "trans": 2.0},
        )
        kwargs.update(overrides)
        return phase2_merged.train_phase2_batch_machine_self_labeling(**kwargs)

    def test_fixed_validation_temperature_makes_rescoring_arm_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            self._run(
                train_dir,
                temperature=1.5,
                temperature_min=0.3,
                temperature_anneal_episodes=2,
                validation_temperature=1.0,
            )
            checkpoint = train_dir / "checkpoints" / "phase2_batch_machine_policy_ep00002.pt"
            self.assertTrue(checkpoint.is_file())
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            # 같은 checkpoint를 서로 다른 temperature arm 플래그로 재평가한다.
            anneal_dir = root / "rescore_anneal"
            constant_dir = root / "rescore_constant"
            anneal_summary = self._run(
                anneal_dir,
                eval_only=True,
                resume_checkpoint=str(checkpoint),
                temperature=1.5,
                temperature_min=0.3,
                temperature_anneal_episodes=2,
                validation_temperature=1.0,
            )
            constant_summary = self._run(
                constant_dir,
                eval_only=True,
                resume_checkpoint=str(checkpoint),
                temperature=1.0,
                temperature_min=None,
                temperature_anneal_episodes=None,
                validation_temperature=1.0,
            )

            self.assertEqual(anneal_summary["eval_temperature"], 1.0)
            self.assertEqual(constant_summary["eval_temperature"], 1.0)
            anneal_rows = (anneal_dir / "validation" / "validation_parent_history.csv").read_text(
                encoding="utf-8"
            )
            constant_rows = (constant_dir / "validation" / "validation_parent_history.csv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(anneal_rows, constant_rows)

            # eval-only는 가중치를 바꾸지 않고 best checkpoint도 만들지 않는다.
            self.assertEqual(checkpoint_sha, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertFalse((anneal_dir / "phase2_best.pt").exists())
            self.assertFalse((anneal_dir / "validation" / "best_checkpoint.json").exists())

    def test_without_fixed_validation_temperature_arms_score_at_different_temperatures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            self._run(train_dir, temperature=1.0)
            checkpoint = train_dir / "checkpoints" / "phase2_batch_machine_policy_ep00002.pt"

            anneal_summary = self._run(
                root / "rescore_anneal",
                eval_only=True,
                resume_checkpoint=str(checkpoint),
                temperature=1.5,
                temperature_min=0.3,
                temperature_anneal_episodes=2,
            )
            constant_summary = self._run(
                root / "rescore_constant",
                eval_only=True,
                resume_checkpoint=str(checkpoint),
                temperature=1.0,
            )

        self.assertAlmostEqual(anneal_summary["eval_temperature"], 0.3)
        self.assertAlmostEqual(constant_summary["eval_temperature"], 1.0)

    def test_eval_summary_carries_run_spec_and_validation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            self._run(train_dir, validation_temperature=1.0)
            checkpoint = train_dir / "checkpoints" / "phase2_batch_machine_policy_ep00002.pt"
            eval_dir = root / "rescore"
            summary = self._run(
                eval_dir,
                eval_only=True,
                resume_checkpoint=str(checkpoint),
                validation_temperature=1.0,
                run_manifest_fields={
                    "phase": "phase2",
                    "command": "phase2-train-batch-machine-self-labeling",
                    "cli_argv": ["main.py", "phase2-train-batch-machine-self-labeling", "--eval-only"],
                    "resolved_args": {"eval_only": True},
                    "experiment_id": "exp_alpha",
                    "arm_group": "phase2_temperature",
                    "arm_label": "anneal",
                    "owner": "alice",
                    "run_note": "",
                },
            )
            written = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((eval_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(summary["eval_only"])
        for key in (
            "run_spec",
            "validation_contract",
            "validation_temperature",
            "eval_temperature",
            "heuristic_algorithms",
            "score_mode",
            "device",
        ):
            self.assertIn(key, written)
        self.assertEqual(
            written["run_spec"]["run_spec_schema_version"],
            summary["run_spec"]["run_spec_schema_version"],
        )
        self.assertEqual(written["validation_contract"]["schema"], "phase2_validation_grid_v1")
        self.assertEqual(manifest["identity"]["arm_label"], "anneal")
        self.assertTrue(manifest["extra"]["eval_only"])
        self.assertEqual(manifest["validation"]["main"]["problem_count"], 2)
        self.assertEqual(manifest["validation"]["validation_temperature"], 1.0)

    def test_invalid_validation_temperature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            phase2_merged._resolve_validation_temperature(0.0)
        with self.assertRaises(ValueError):
            phase2_merged._resolve_validation_temperature(-1.0)
        with self.assertRaises(ValueError):
            phase2_merged._resolve_validation_temperature("hot")
        self.assertIsNone(phase2_merged._resolve_validation_temperature(None))
        self.assertEqual(phase2_merged._resolve_validation_temperature(2), 2.0)


if __name__ == "__main__":
    unittest.main()
