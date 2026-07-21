"""MIXED-only Phase 2 checkpoint/evaluator RunSpec 계약 검증."""

from argparse import Namespace
from copy import deepcopy
import unittest

from Environment.constraints.profiles import default_phase2_constraint_profile
from Phase2.run_spec import (
    build_phase2_run_spec,
    require_matching_phase2_run_spec,
    validate_phase2_run_spec,
)
from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights
from main import _resolve_phase2_full_flow_run_spec


class Phase2RunSpecTest(unittest.TestCase):
    def _spec(self, action_pool_limit=None):
        return build_phase2_run_spec(
            score_mode="raw",
            score_fields=(
                "hard_violation_count",
                "makespan",
                "bay_internal_cut_length_gap",
                "bay_internal_wo_count_gap",
                "bay_internal_bevel_quantity_gap",
                "bay_internal_occupancy_gap",
            ),
            action_pool_limit=action_pool_limit,
            max_wo_count=3,
            max_length_sum=55_000.0,
            phase1_bay_capacity_weights=joint_phase1_bay_capacity_weights(),
            constraint_profile=default_phase2_constraint_profile(),
            heuristic_algorithms=("min_makespan", "lpt_batch"),
            train_rollout_samples=64,
            validation_rollout_samples=64,
        )

    def test_run_spec_contains_fixed_mixed_contract(self) -> None:
        spec = self._spec()

        self.assertEqual(spec["action_pool_limit"], None)
        self.assertEqual(
            spec["phase1_bay_capacity_weights"],
            joint_phase1_bay_capacity_weights(),
        )
        self.assertEqual(spec["phase1_rule_profile"], "multi_series_260711")
        self.assertEqual(spec["phase1_scope_version"], "joint_five_bay_v3_shared_pool")
        self.assertEqual(spec["phase1_score_mode"], "wo_first")
        self.assertEqual(spec["policy_action"], "select_wo_only")
        self.assertEqual(
            spec["machine_dispatch_rule"],
            "idle_at_current_time_else_next_completion_then_machine_id",
        )
        self.assertNotIn("phase1_long_cut_hard_mask", spec)

    def test_run_spec_mismatch_fails_instead_of_overriding_checkpoint(self) -> None:
        with self.assertRaises(RuntimeError):
            require_matching_phase2_run_spec(
                self._spec(action_pool_limit=None),
                self._spec(action_pool_limit=18),
                context="full_flow",
            )

    def test_run_spec_rejects_legacy_long_cut_toggle(self) -> None:
        spec = deepcopy(self._spec())
        spec["phase1_long_cut_hard_mask"] = True

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_run_spec_rejects_non_joint_capacity_weights(self) -> None:
        spec = deepcopy(self._spec())
        spec["phase1_bay_capacity_weights"] = {"22": 4.0, "23": 4.0, "24": 3.0}

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_run_spec_rejects_boolean_batch_limit(self) -> None:
        spec = deepcopy(self._spec())
        spec["max_wo_count"] = True

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_full_flow_inherits_checkpoint_run_spec_without_cli_overrides(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)

        resolved = _resolve_phase2_full_flow_run_spec(
            args=Namespace(),
            checkpoint_spec=checkpoint,
            constraint_profile=default_phase2_constraint_profile(),
            phase1_bay_capacity_weights=joint_phase1_bay_capacity_weights(),
            heuristic=None,
        )

        self.assertEqual(resolved, checkpoint)

    def test_full_flow_rejects_explicit_score_mode_mismatch(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)

        with self.assertRaises(RuntimeError):
            _resolve_phase2_full_flow_run_spec(
                args=Namespace(phase2_score_mode="normalized"),
                checkpoint_spec=checkpoint,
                constraint_profile=default_phase2_constraint_profile(),
                phase1_bay_capacity_weights=joint_phase1_bay_capacity_weights(),
                heuristic=None,
            )


if __name__ == "__main__":
    unittest.main()
