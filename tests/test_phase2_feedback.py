"""Frozen merged Phase 2 schedule feedback contract."""

from types import SimpleNamespace
from pathlib import Path
import hashlib
import tempfile
import unittest

from Environment.constraints.profiles import default_phase2_constraint_profile
from Phase2.feedback import build_frozen_phase2_schedule_feedback_scorer, build_phase2_feedback_contract
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
    build_mixed_phase2_training_machines,
    run_phase2_batch_machine_candidate,
)
from Phase2.run_spec import build_phase2_run_spec
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase1.pair_self_labeling import PHASE1_PAIR_ENV_FEATURE_NAMES, PHASE1_PAIR_FEATURE_NAMES
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Utils.learning.phase_agent_checkpoints import load_phase1_feedback_contract
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    joint_phase1_bay_capacity_weights,
)
from main import _require_matching_phase1_feedback_contract


class Phase2FrozenFeedbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = {
            "WO_A": self._job("WO_A", "P1::A", 10.0, 900.0, 3),
            "WO_B": self._job("WO_B", "P1::A", 7.0, 500.0, 1),
            "WO_C": self._job("WO_C", "P1::B", 5.0, 300.0, 2),
        }
        self.machines = build_mixed_phase2_training_machines()
        self.profile = default_phase2_constraint_profile()
        self.model = Phase2SetPointerPolicy(hidden_dim=8)
        self.run_spec = build_phase2_run_spec(
            score_mode="raw",
            score_fields=PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
            action_pool_limit=None,
            max_wo_count=3,
            max_length_sum=55_000.0,
            phase1_bay_capacity_weights=joint_phase1_bay_capacity_weights(),
            constraint_profile=self.profile,
            heuristic_algorithms=("lpt_batch",),
            train_rollout_samples=1,
            validation_rollout_samples=1,
        )

    def test_frozen_feedback_equals_direct_greedy_phase2_schedule_score(self) -> None:
        assignments = {"P1::A": "22", "P1::B": "23"}
        scorer = build_frozen_phase2_schedule_feedback_scorer(
            model=self.model,
            machines=self.machines,
            run_spec=self.run_spec,
            constraint_profile=self.profile,
        )

        score = scorer(
            SimpleNamespace(assignments=assignments),
            self.jobs,
            tuple(joint_phase1_bay_capacity_weights()),
        )
        direct = run_phase2_batch_machine_candidate(
            jobs=self.jobs,
            machines=self.machines,
            phase1_assignments=assignments,
            source="agent_greedy",
            model=self.model,
            max_wo_count=3,
            max_length_sum=55_000.0,
            action_pool_limit=None,
            score_mode="raw",
            constraint_profile=self.profile,
        )

        self.assertEqual(score, direct.score_tuple)

    def test_frozen_feedback_rejects_machine_capacity_that_differs_from_run_spec(self) -> None:
        mismatched_machines = dict(self.machines)
        mismatched_machines.pop(next(iter(mismatched_machines)))

        with self.assertRaises(RuntimeError):
            build_frozen_phase2_schedule_feedback_scorer(
                model=self.model,
                machines=mismatched_machines,
                run_spec=self.run_spec,
                constraint_profile=self.profile,
            )

    def test_feedback_contract_fingerprints_exact_checkpoint_bytes(self) -> None:
        content = b"phase2-checkpoint-bytes"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "phase2.pt"
            path.write_bytes(content)

            contract = build_phase2_feedback_contract(path, self.run_spec)

        self.assertEqual(contract["checkpoint"], str(path.resolve()))
        self.assertEqual(contract["checkpoint_sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(contract["run_spec"], self.run_spec)

    def test_phase1_feedback_contract_loader_returns_saved_contract(self) -> None:
        contract = {
            "checkpoint": "/tmp/phase2.pt",
            "checkpoint_sha256": "a" * 64,
            "run_spec": self.run_spec,
        }
        model = Phase1PairPointerPolicy(
            pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
            env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
            hidden_dim=8,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "phase1.pt"
            import torch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pair_feature_names": PHASE1_PAIR_FEATURE_NAMES,
                    "env_feature_names": PHASE1_PAIR_ENV_FEATURE_NAMES,
                    "hidden_dim": 8,
                    "rule_profile": MULTI_SERIES_RULE_PROFILE,
                    "score_mode": "wo_first",
                    "episode_scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
                    "phase2_feedback_contract": contract,
                },
                path,
            )

            loaded = load_phase1_feedback_contract(path)

        self.assertEqual(loaded, contract)

    def test_full_flow_rejects_different_phase2_contract_for_feedback_trained_phase1(self) -> None:
        expected = {
            "checkpoint": "/tmp/phase2-a.pt",
            "checkpoint_sha256": "a" * 64,
            "run_spec": self.run_spec,
        }
        selected = {
            **expected,
            "checkpoint": "/tmp/phase2-b.pt",
            "checkpoint_sha256": "b" * 64,
        }

        with self.assertRaises(RuntimeError):
            _require_matching_phase1_feedback_contract(expected, selected)

    @staticmethod
    def _job(job_id: str, block_set_id: str, tact: float, cut: float, bevel: int) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            steel_quantity=1,
            plate_length=10_000.0,
            plate_width=3_000.0,
            thickness=13.0,
            cut_length=cut,
            bevel_quantity=bevel,
            processing_time=tact,
            base_stage_minutes={"cut": tact},
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            extra={"source_wk_ord_no": job_id},
        )

if __name__ == "__main__":
    unittest.main()
