"""Phase 2 checkpoint/evaluator RunSpec 계약 검증."""

from argparse import Namespace
from copy import deepcopy
import unittest

from Environment.constraints.profiles import default_phase2_constraint_profile
from Phase2.run_spec import (
    build_phase2_run_spec,
    require_matching_phase2_run_spec,
    validate_phase2_run_spec,
)
from scripts.evaluate_phase2_candidate_workbook import _resolve_evaluation_run_spec
from main import _resolve_phase2_full_flow_run_spec


class Phase2RunSpecTest(unittest.TestCase):
    def _spec(self, action_pool_limit=None):
        return build_phase2_run_spec(
            score_mode="raw",
            score_fields=(
                "hard_violation_count",
                "makespan",
                "bay_internal_wo_count_gap",
                "bay_internal_cut_length_gap",
                "bay_internal_bevel_quantity_gap",
                "bay_internal_occupancy_gap",
            ),
            action_pool_limit=action_pool_limit,
            max_wo_count=3,
            max_length_sum=55_000.0,
            phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0},
            phase1_long_cut_hard_mask=True,
            constraint_profile=default_phase2_constraint_profile(),
            heuristic_algorithms=("min_makespan", "lpt_batch"),
            train_rollout_samples=64,
            validation_rollout_samples=64,
        )

    def test_run_spec_contains_all_reproducibility_fields(self) -> None:
        spec = self._spec()

        self.assertEqual(spec["action_pool_limit"], None)
        self.assertEqual(spec["phase1_bay_capacity_weights"], {"22": 4.0, "23": 3.0, "24": 4.0})
        self.assertEqual(spec["constraint_profile"]["name"], "phase2")
        self.assertEqual(spec["validation_rollout_samples"], 64)

    def test_run_spec_mismatch_fails_instead_of_overriding_checkpoint(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)
        requested = self._spec(action_pool_limit=18)

        with self.assertRaises(RuntimeError):
            require_matching_phase2_run_spec(checkpoint, requested, context="actual_8days")

    def test_run_spec_rejects_boolean_batch_limit(self) -> None:
        spec = deepcopy(self._spec())
        spec["max_wo_count"] = True

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_run_spec_builder_does_not_coerce_boolean_batch_limit(self) -> None:
        with self.assertRaises(RuntimeError):
            build_phase2_run_spec(
                score_mode="raw",
                score_fields=self._spec()["score_fields"],
                action_pool_limit=None,
                max_wo_count=True,
                max_length_sum=55_000.0,
                phase1_bay_capacity_weights={"22": 1.0},
                phase1_long_cut_hard_mask=True,
                constraint_profile=default_phase2_constraint_profile(),
                heuristic_algorithms=("min_makespan",),
                train_rollout_samples=1,
                validation_rollout_samples=1,
            )

    def test_run_spec_rejects_non_numeric_capacity_weight(self) -> None:
        spec = deepcopy(self._spec())
        spec["phase1_bay_capacity_weights"]["22"] = "invalid"

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_run_spec_rejects_score_fields_from_another_mode(self) -> None:
        spec = deepcopy(self._spec())
        spec["score_fields"] = [
            "hard_violation_count",
            "makespan",
            "bay_internal_normalized_wo_count_gap",
            "bay_internal_normalized_cut_length_gap",
            "bay_internal_normalized_bevel_quantity_gap",
            "bay_internal_normalized_occupancy_gap",
        ]

        with self.assertRaises(RuntimeError):
            validate_phase2_run_spec(spec)

    def test_actual_evaluator_inherits_checkpoint_run_spec_when_cli_does_not_override(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)
        args = Namespace(no_phase1_long_cut_hard_mask=False)

        resolved = _resolve_evaluation_run_spec(
            args=args,
            checkpoint_spec=checkpoint,
            constraint_profile=default_phase2_constraint_profile(),
            phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0},
        )

        self.assertEqual(resolved, checkpoint)

    def test_actual_evaluator_rejects_cli_value_that_differs_from_checkpoint(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)
        args = Namespace(
            no_phase1_long_cut_hard_mask=False,
            action_pool_limit=18,
        )

        with self.assertRaises(RuntimeError):
            _resolve_evaluation_run_spec(
                args=args,
                checkpoint_spec=checkpoint,
                constraint_profile=default_phase2_constraint_profile(),
                phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0},
            )

    def test_full_flow_inherits_checkpoint_run_spec_without_cli_overrides(self) -> None:
        checkpoint = self._spec(action_pool_limit=None)

        resolved = _resolve_phase2_full_flow_run_spec(
            args=Namespace(),
            checkpoint_spec=checkpoint,
            constraint_profile=default_phase2_constraint_profile(),
            phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0},
            phase1_long_cut_hard_mask=True,
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
                phase1_bay_capacity_weights={"22": 4.0, "23": 3.0, "24": 4.0},
                phase1_long_cut_hard_mask=True,
                heuristic=None,
            )


if __name__ == "__main__":
    unittest.main()
