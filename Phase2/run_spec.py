"""Phase 2 학습·validation·actual 평가가 공유하는 실행 계약."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from Environment.constraints.profiles import PhaseConstraintProfile
from Environment.metrics import PHASE2_METRIC_SCHEMA_VERSION, PHASE2_SCORE_FIELDS_BY_MODE
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    joint_phase1_bay_capacity_weights,
)
from Phase2.state import PHASE2_STATE_SCHEMA_VERSION


PHASE2_RUN_SPEC_SCHEMA_VERSION = "phase2_run_spec_v4_event_clock"
PHASE2_MACHINE_DISPATCH_RULE = "idle_at_current_time_else_next_completion_then_machine_id"


def build_phase2_run_spec(
    *,
    score_mode: str,
    score_fields: Sequence[str],
    action_pool_limit: int | None,
    max_wo_count: int,
    max_length_sum: float,
    phase1_bay_capacity_weights: Mapping[str, int | float],
    constraint_profile: PhaseConstraintProfile,
    heuristic_algorithms: Sequence[str],
    train_rollout_samples: int,
    validation_rollout_samples: int,
) -> Dict[str, Any]:
    """현재 실행 조건을 JSON 직렬화 가능한 단일 RunSpec으로 만든다."""

    spec = {
        "run_spec_schema_version": PHASE2_RUN_SPEC_SCHEMA_VERSION,
        "metric_schema_version": PHASE2_METRIC_SCHEMA_VERSION,
        "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
        "policy_action": "select_wo_only",
        "machine_dispatch_rule": PHASE2_MACHINE_DISPATCH_RULE,
        "score_mode": score_mode,
        "score_fields": list(score_fields),
        "action_pool_limit": action_pool_limit,
        "max_wo_count": max_wo_count,
        "max_length_sum": max_length_sum,
        "phase1_bay_capacity_weights": dict(phase1_bay_capacity_weights),
        "phase1_rule_profile": MULTI_SERIES_RULE_PROFILE,
        "phase1_scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
        "phase1_score_mode": "wo_first",
        "constraint_profile": phase_constraint_profile_to_dict(constraint_profile),
        "heuristic_algorithms": list(heuristic_algorithms),
        "train_rollout_samples": train_rollout_samples,
        "validation_rollout_samples": validation_rollout_samples,
    }
    validate_phase2_run_spec(spec)
    return spec


def phase_constraint_profile_to_dict(profile: PhaseConstraintProfile) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "hard_enabled": dict(sorted(profile.hard_enabled.items())),
        "soft_enabled": dict(sorted(profile.soft_enabled.items())),
        "soft_weights": dict(sorted(profile.soft_weights.items())),
        "category_flags": dict(sorted(profile.category_flags.items())),
    }


def validate_phase2_run_spec(spec: Mapping[str, Any]) -> None:
    """RunSpec 누락·버전·타입 오류를 추론 없이 거절한다."""

    required = {
        "run_spec_schema_version",
        "metric_schema_version",
        "feature_schema_version",
        "policy_action",
        "machine_dispatch_rule",
        "score_mode",
        "score_fields",
        "action_pool_limit",
        "max_wo_count",
        "max_length_sum",
        "phase1_bay_capacity_weights",
        "phase1_rule_profile",
        "phase1_scope_version",
        "phase1_score_mode",
        "constraint_profile",
        "heuristic_algorithms",
        "train_rollout_samples",
        "validation_rollout_samples",
    }
    missing = sorted(required - set(spec))
    extra = sorted(set(spec) - required)
    if missing or extra:
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=field_mismatch missing={missing} extra={extra}"
        )
        raise RuntimeError("Phase 2 RunSpec fields do not match the current contract")
    expected_versions = {
        "run_spec_schema_version": PHASE2_RUN_SPEC_SCHEMA_VERSION,
        "metric_schema_version": PHASE2_METRIC_SCHEMA_VERSION,
        "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
    }
    for key, expected in expected_versions.items():
        if spec[key] != expected:
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=version_mismatch field={key} actual={spec[key]} expected={expected}"
            )
            raise RuntimeError(f"Phase 2 RunSpec {key} mismatch")
    expected_action_contract = {
        "policy_action": "select_wo_only",
        "machine_dispatch_rule": PHASE2_MACHINE_DISPATCH_RULE,
    }
    for key, expected in expected_action_contract.items():
        if spec[key] != expected:
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=action_contract_mismatch field={key} actual={spec[key]} expected={expected}"
            )
            raise RuntimeError(f"Phase 2 RunSpec {key} mismatch")
    score_mode = spec["score_mode"]
    if score_mode not in PHASE2_SCORE_FIELDS_BY_MODE:
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=invalid_score_mode value={score_mode}"
        )
        raise RuntimeError("Phase 2 RunSpec score_mode is invalid")
    expected_score_fields = list(PHASE2_SCORE_FIELDS_BY_MODE[str(score_mode)])
    if spec["score_fields"] != expected_score_fields:
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=score_field_mismatch mode={score_mode} "
            f"actual={spec['score_fields']} expected={expected_score_fields}"
        )
        raise RuntimeError("Phase 2 RunSpec score fields do not match score mode")
    action_pool_limit = spec["action_pool_limit"]
    if action_pool_limit is not None and (
        isinstance(action_pool_limit, bool)
        or not isinstance(action_pool_limit, int)
        or action_pool_limit <= 0
    ):
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=invalid_action_pool_limit value={action_pool_limit}"
        )
        raise RuntimeError("Phase 2 RunSpec action_pool_limit is invalid")
    max_wo_count = spec["max_wo_count"]
    max_length_sum = spec["max_length_sum"]
    if (
        isinstance(max_wo_count, bool)
        or not isinstance(max_wo_count, int)
        or max_wo_count <= 0
        or isinstance(max_length_sum, bool)
        or not isinstance(max_length_sum, (int, float))
        or not math.isfinite(float(max_length_sum))
        or float(max_length_sum) <= 0
    ):
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=invalid_batch_limits max_wo_count={max_wo_count} max_length_sum={max_length_sum}"
        )
        raise RuntimeError("Phase 2 RunSpec batch limits are invalid")
    weights = spec["phase1_bay_capacity_weights"]
    if not isinstance(weights, Mapping) or not weights:
        print("[ERROR][Phase2.run_spec.validate_phase2_run_spec] cause=invalid_phase1_capacity_weights")
        raise RuntimeError("Phase 2 RunSpec Phase 1 capacity weights are invalid")
    for bay_id, value in weights.items():
        if (
            not isinstance(bay_id, str)
            or not bay_id.strip()
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=invalid_phase1_capacity_weight bay_id={bay_id} value={value}"
            )
            raise RuntimeError("Phase 2 RunSpec Phase 1 capacity weights are invalid")
    expected_weights = joint_phase1_bay_capacity_weights()
    normalized_weights = {str(key): float(value) for key, value in weights.items()}
    if normalized_weights != expected_weights:
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=phase1_capacity_contract_mismatch expected={expected_weights} "
            f"actual={normalized_weights}"
        )
        raise RuntimeError("Phase 2 RunSpec requires the MIXED five-Bay capacity contract")
    expected_phase1_contract = {
        "phase1_rule_profile": MULTI_SERIES_RULE_PROFILE,
        "phase1_scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
        "phase1_score_mode": "wo_first",
    }
    for field_name, expected_value in expected_phase1_contract.items():
        if spec[field_name] != expected_value:
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=phase1_contract_mismatch field={field_name} "
                f"expected={expected_value} actual={spec[field_name]}"
            )
            raise RuntimeError(f"Phase 2 RunSpec {field_name} mismatch")
    profile = spec["constraint_profile"]
    if not isinstance(profile, Mapping) or set(profile) != {
        "name", "hard_enabled", "soft_enabled", "soft_weights", "category_flags"
    }:
        print("[ERROR][Phase2.run_spec.validate_phase2_run_spec] cause=invalid_constraint_profile")
        raise RuntimeError("Phase 2 RunSpec constraint profile is invalid")
    if profile["name"] != "phase2":
        print(
            "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
            f"cause=invalid_constraint_profile_name value={profile['name']}"
        )
        raise RuntimeError("Phase 2 RunSpec constraint profile name is invalid")
    for field_name in ("hard_enabled", "soft_enabled", "category_flags"):
        values = profile[field_name]
        if not isinstance(values, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, bool)
            for key, value in values.items()
        ):
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=invalid_constraint_boolean_mapping field={field_name}"
            )
            raise RuntimeError(f"Phase 2 RunSpec {field_name} is invalid")
    soft_weights = profile["soft_weights"]
    if not isinstance(soft_weights, Mapping):
        print("[ERROR][Phase2.run_spec.validate_phase2_run_spec] cause=invalid_soft_weights")
        raise RuntimeError("Phase 2 RunSpec soft weights are invalid")
    for rule_name, value in soft_weights.items():
        if (
            not isinstance(rule_name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=invalid_soft_weight rule={rule_name} value={value}"
            )
            raise RuntimeError("Phase 2 RunSpec soft weights are invalid")
    algorithms = spec["heuristic_algorithms"]
    if (
        not isinstance(algorithms, list)
        or not algorithms
        or any(not isinstance(value, str) or not value.strip() for value in algorithms)
        or len(set(algorithms)) != len(algorithms)
    ):
        print("[ERROR][Phase2.run_spec.validate_phase2_run_spec] cause=invalid_heuristic_bank")
        raise RuntimeError("Phase 2 RunSpec heuristic bank is invalid")
    for key in ("train_rollout_samples", "validation_rollout_samples"):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            print(
                "[ERROR][Phase2.run_spec.validate_phase2_run_spec] "
                f"cause=invalid_sample_count field={key} value={value}"
            )
            raise RuntimeError(f"Phase 2 RunSpec {key} is invalid")


def require_matching_phase2_run_spec(
    checkpoint_spec: Mapping[str, Any],
    requested_spec: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """재개/평가 조건이 checkpoint와 정확히 같은지 검증한다."""

    validate_phase2_run_spec(checkpoint_spec)
    validate_phase2_run_spec(requested_spec)
    if dict(checkpoint_spec) != dict(requested_spec):
        differing = sorted(
            key for key in checkpoint_spec if checkpoint_spec.get(key) != requested_spec.get(key)
        )
        print(
            "[ERROR][Phase2.run_spec.require_matching_phase2_run_spec] "
            f"cause=run_spec_mismatch context={context} fields={differing}"
        )
        raise RuntimeError(f"Phase 2 RunSpec mismatch for {context}: {differing}")


__all__ = [
    "PHASE2_RUN_SPEC_SCHEMA_VERSION",
    "build_phase2_run_spec",
    "phase_constraint_profile_to_dict",
    "require_matching_phase2_run_spec",
    "validate_phase2_run_spec",
]
