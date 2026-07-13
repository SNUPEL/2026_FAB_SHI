"""Phase별로 확정된 제약만 공통 registry에 연결한다.

Global constraint 설정은 실제 DES의 전체 기능을 포함한다. 학습 Phase가 그
설정을 그대로 읽으면 아직 현업에서 확정되지 않은 family/두께/정반 제약까지
action mask에 섞인다. 이 모듈은 config의 명시적 Phase profile을 검증하고,
Phase 2 action 후보를 기존 ``ConstraintManager``로 평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

from .base import ConstraintContext
from .registry import CandidateConstraintBundle, ConstraintManager, RULES


PHASE2_REQUIRED_PROFILE_RULES = (
    "machine_enabled",
    "machine_bay_consistency",
    "block_set_same_bay",
    "machine_single_processing",
    "batch_wo_count_limit",
    "batch_length_sum_limit",
    "family_eligibility",
    "thickness_range",
    "table_length_limit",
)
PHASE2_DEFAULT_HARD_ENABLED = {
    "machine_enabled": True,
    "machine_bay_consistency": True,
    "block_set_same_bay": True,
    "machine_single_processing": True,
    "batch_wo_count_limit": True,
    "batch_length_sum_limit": True,
    "family_eligibility": False,
    "thickness_range": False,
    "table_length_limit": False,
}
PHASE2_DEFAULT_CATEGORIES = {
    "machine": True,
    "capacity": True,
    "downstream": False,
    "priority": False,
    "preference": False,
    "layout": True,
    "calendar": False,
}


@dataclass(frozen=True)
class PhaseConstraintProfile:
    """검증이 끝난 Phase별 hard/soft 제약 설정."""

    name: str
    hard_enabled: Dict[str, bool]
    soft_enabled: Dict[str, bool]
    soft_weights: Dict[str, float]
    category_flags: Dict[str, bool]

    def build_manager(self) -> ConstraintManager:
        return ConstraintManager(
            hard_enabled=dict(self.hard_enabled),
            soft_enabled=dict(self.soft_enabled),
            soft_weights=dict(self.soft_weights),
            category_flags=dict(self.category_flags),
        )


def default_phase2_constraint_profile() -> PhaseConstraintProfile:
    """직접 Python API 호출에도 동일한 확정 Phase 2 계약을 적용한다."""

    return PhaseConstraintProfile(
        name="phase2",
        hard_enabled=dict(PHASE2_DEFAULT_HARD_ENABLED),
        soft_enabled={},
        soft_weights={},
        category_flags=dict(PHASE2_DEFAULT_CATEGORIES),
    )


def load_phase_constraint_profile(config: Mapping[str, Any], phase_name: str) -> PhaseConstraintProfile:
    """config에서 Phase profile을 엄격하게 읽는다."""

    normalized_name = str(phase_name).strip().lower()
    constraints = _required_mapping(config, "constraints", "config")
    profiles = _required_mapping(constraints, "profiles", "constraints")
    profile = _required_mapping(profiles, normalized_name, "constraints.profiles")
    hard_enabled = _boolean_mapping(profile, "hard_enabled", normalized_name)
    soft_enabled = _boolean_mapping(profile, "soft_enabled", normalized_name)
    category_flags = _boolean_mapping(profile, "categories", normalized_name)
    soft_weights_raw = _required_mapping(profile, "soft_penalty_weights", normalized_name)
    soft_weights = {str(key): float(value) for key, value in soft_weights_raw.items()}

    unknown_rules = sorted((set(hard_enabled) | set(soft_enabled)) - set(RULES))
    if unknown_rules:
        print(
            "[ERROR][constraints.profiles.load_phase_constraint_profile] "
            f"cause=unknown_rules phase={normalized_name} rules={unknown_rules}"
        )
        raise RuntimeError(f"unknown rules in {normalized_name} constraint profile: {unknown_rules}")
    if normalized_name == "phase2":
        missing = sorted(set(PHASE2_REQUIRED_PROFILE_RULES) - set(hard_enabled))
        if missing:
            print(
                "[ERROR][constraints.profiles.load_phase_constraint_profile] "
                f"cause=missing_required_rules phase=phase2 rules={missing}"
            )
            raise RuntimeError(f"Phase 2 constraint profile is missing rules: {missing}")
        if hard_enabled != PHASE2_DEFAULT_HARD_ENABLED:
            print(
                "[ERROR][constraints.profiles.load_phase_constraint_profile] "
                "cause=phase2_contract_mismatch"
            )
            raise RuntimeError("Phase 2 hard constraint profile differs from the confirmed contract")

    return PhaseConstraintProfile(
        name=normalized_name,
        hard_enabled=hard_enabled,
        soft_enabled=soft_enabled,
        soft_weights=soft_weights,
        category_flags=category_flags,
    )


def evaluate_phase2_action_constraints(
    *,
    profile: PhaseConstraintProfile,
    job: object,
    machine: object,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    machine_available_at: Mapping[str, float],
    current_time: float,
    candidate_batch_job_ids: Sequence[str],
    candidate_batch_length_sum: float,
    max_wo_count: int,
    max_length_sum: float,
) -> CandidateConstraintBundle:
    """Phase 2의 W/O 추가 후보 하나를 공통 제약 registry로 평가한다."""

    if profile.name != "phase2":
        print(
            "[ERROR][constraints.profiles.evaluate_phase2_action_constraints] "
            f"cause=wrong_profile profile={profile.name}"
        )
        raise RuntimeError("Phase 2 action evaluation requires the phase2 profile")

    job_view = _attribute_view(job, "job")
    machine_view = _attribute_view(machine, "machine")
    job_id = str(_required_field(job_view, "job_id", "job"))
    machine_id = str(_required_field(machine_view, "machine_id", "machine"))
    block_set_id = str(_required_field(job_view, "block_set_id", job_id))
    machine_bay_id = str(_required_field(machine_view, "bay_id", machine_id))
    machine_enabled = bool(_required_field(machine_view, "enabled", machine_id))
    if machine_id not in machine_available_at:
        print(
            "[ERROR][constraints.profiles.evaluate_phase2_action_constraints] "
            f"cause=missing_machine_available_at machine_id={machine_id}"
        )
        raise RuntimeError(f"missing machine availability for {machine_id}")
    if block_set_id not in phase1_assignments:
        print(
            "[ERROR][constraints.profiles.evaluate_phase2_action_constraints] "
            f"cause=missing_phase1_assignment block_set_id={block_set_id} job_id={job_id}"
        )
        raise RuntimeError(f"missing Phase 1 Bay assignment for {block_set_id}")

    assigned_bay = str(phase1_assignments[block_set_id])
    state = SimpleNamespace(
        current_time=float(current_time),
        machine_available_at={str(key): float(value) for key, value in machine_available_at.items()},
    )
    context = ConstraintContext(
        job=job_view,
        machine=machine_view,
        state=state,
        jobs=dict(jobs),
        machines=dict(machines),
        bays={},
        layout=None,
        config={},
        machine_enabled_flag=machine_enabled,
        candidate_start_time=float(current_time),
        machine_bay_id=machine_bay_id,
        job_cut_bay=assigned_bay,
        block_set_id=block_set_id,
        assigned_block_set_bay=assigned_bay,
        candidate_batch_job_ids=tuple(str(value) for value in candidate_batch_job_ids),
        candidate_batch_wo_count=len(candidate_batch_job_ids),
        candidate_batch_length_sum=float(candidate_batch_length_sum),
        batch_wo_count_limit=int(max_wo_count),
        batch_length_sum_limit=float(max_length_sum),
    )
    return profile.build_manager().evaluate_candidate(context)


def audit_phase2_schedule_constraints(
    *,
    profile: PhaseConstraintProfile,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    batches: Sequence[Mapping[str, Any]],
    max_wo_count: int,
    max_length_sum: float,
) -> Dict[str, Any]:
    """완성된 Phase 2 schedule을 같은 공통 hard rule로 순서대로 재검증한다."""

    if not batches:
        print("[ERROR][constraints.profiles.audit_phase2_schedule_constraints] cause=no_batches")
        raise RuntimeError("Phase 2 final constraint audit requires batches")
    availability = {str(machine_id): 0.0 for machine_id in machines}
    scheduled_jobs: set[str] = set()
    rows: list[Dict[str, Any]] = []
    batch_scoped_rules = {
        "machine_enabled",
        "machine_single_processing",
        "batch_wo_count_limit",
        "batch_length_sum_limit",
    }

    ordered_batches = sorted(
        batches,
        key=lambda row: (
            float(_required_mapping_field(row, "start_time", "batch")),
            str(_required_mapping_field(row, "machine_id", "batch")),
            str(_required_mapping_field(row, "batch_id", "batch")),
        ),
    )
    for batch in ordered_batches:
        batch_id = str(_required_mapping_field(batch, "batch_id", "batch"))
        machine_id = str(_required_mapping_field(batch, "machine_id", batch_id))
        if machine_id not in machines:
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=unknown_machine batch_id={batch_id} machine_id={machine_id}"
            )
            raise RuntimeError(f"Phase 2 final audit found unknown machine: {machine_id}")
        raw_job_ids = _required_mapping_field(batch, "job_ids", batch_id)
        if not isinstance(raw_job_ids, (list, tuple)) or not raw_job_ids:
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=invalid_job_ids batch_id={batch_id} value={raw_job_ids}"
            )
            raise RuntimeError(f"Phase 2 final audit found invalid batch job_ids: {batch_id}")
        job_ids = tuple(str(value) for value in raw_job_ids)
        duplicate_jobs = sorted(set(job_ids) & scheduled_jobs)
        if duplicate_jobs or len(set(job_ids)) != len(job_ids):
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=duplicate_job_assignment batch_id={batch_id} job_ids={duplicate_jobs or list(job_ids)}"
            )
            raise RuntimeError("Phase 2 final audit found duplicate W/O assignment")
        missing_jobs = sorted(set(job_ids) - set(jobs))
        if missing_jobs:
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=unknown_jobs batch_id={batch_id} job_ids={missing_jobs}"
            )
            raise RuntimeError(f"Phase 2 final audit found unknown W/Os: {missing_jobs}")

        start_time = float(_required_mapping_field(batch, "start_time", batch_id))
        finish_time = float(_required_mapping_field(batch, "finish_time", batch_id))
        if start_time < 0 or finish_time <= start_time:
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=invalid_interval batch_id={batch_id} start={start_time} finish={finish_time}"
            )
            raise RuntimeError(f"Phase 2 final audit found invalid batch interval: {batch_id}")
        expected_duration = max(_processing_time(jobs[job_id], job_id) for job_id in job_ids)
        actual_duration = finish_time - start_time
        if not math.isclose(actual_duration, expected_duration, rel_tol=0.0, abs_tol=1e-5):
            print(
                "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
                f"cause=batch_duration_mismatch batch_id={batch_id} "
                f"actual={actual_duration} expected={expected_duration}"
            )
            raise RuntimeError(f"Phase 2 batch duration differs from max W/O TACT: {batch_id}")
        candidate_length = sum(
            float(_required_object_field(jobs[job_id], "plate_length", job_id))
            for job_id in job_ids
        )
        _validate_emitted_batch_aggregates(
            batch=batch,
            batch_id=batch_id,
            expected_wo_count=len(job_ids),
            expected_length_sum=candidate_length,
            expected_duration=expected_duration,
        )
        emitted: set[tuple[str, str]] = set()
        for job_id in job_ids:
            bundle = evaluate_phase2_action_constraints(
                profile=profile,
                job=jobs[job_id],
                machine=machines[machine_id],
                jobs=jobs,
                machines=machines,
                phase1_assignments=phase1_assignments,
                machine_available_at=availability,
                current_time=start_time,
                candidate_batch_job_ids=job_ids,
                candidate_batch_length_sum=candidate_length,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
            )
            for result in bundle.hard_results:
                scope_job_id = "" if result.rule_name in batch_scoped_rules else job_id
                dedup_key = (result.rule_name, scope_job_id)
                if dedup_key in emitted:
                    continue
                emitted.add(dedup_key)
                rows.append(
                    {
                        "batch_id": batch_id,
                        "machine_id": machine_id,
                        "job_id": scope_job_id,
                        "rule_name": result.rule_name,
                        "passed": bool(result.passed),
                        "reason": result.reason,
                    }
                )
        availability[machine_id] = finish_time
        scheduled_jobs.update(job_ids)

    missing_schedule_jobs = sorted(set(jobs) - scheduled_jobs)
    if missing_schedule_jobs:
        print(
            "[ERROR][constraints.profiles.audit_phase2_schedule_constraints] "
            f"cause=unscheduled_jobs count={len(missing_schedule_jobs)} examples={missing_schedule_jobs[:5]}"
        )
        raise RuntimeError("Phase 2 final audit found unscheduled W/Os")
    failed_rows = [row for row in rows if not row["passed"]]
    return {
        "hard_violation_count": len(failed_rows),
        "rows": rows,
        "failed_rows": failed_rows,
    }


def _required_mapping(source: Mapping[str, Any], key: str, location: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        print(
            "[ERROR][constraints.profiles._required_mapping] "
            f"cause=missing_or_invalid_mapping location={location} key={key}"
        )
        raise RuntimeError(f"{location}.{key} must be a mapping")
    return value


def _required_mapping_field(source: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in source or source[key] is None:
        print(
            "[ERROR][constraints.profiles._required_mapping_field] "
            f"cause=missing_field location={location} key={key}"
        )
        raise RuntimeError(f"{location}.{key} is required")
    return source[key]


def _required_object_field(source: object, key: str, location: str) -> Any:
    value = source.get(key) if isinstance(source, Mapping) else getattr(source, key, None)
    if value is None:
        print(
            "[ERROR][constraints.profiles._required_object_field] "
            f"cause=missing_field location={location} key={key}"
        )
        raise RuntimeError(f"{location}.{key} is required")
    return value


def _processing_time(job: object, job_id: str) -> float:
    """공통 DES와 같은 규칙으로 W/O TACT 합을 읽는다."""

    stages = _required_object_field(job, "base_stage_minutes", job_id)
    if not isinstance(stages, Mapping) or not stages:
        print(
            "[ERROR][constraints.profiles._processing_time] "
            f"cause=invalid_stage_minutes job_id={job_id}"
        )
        raise RuntimeError(f"invalid base_stage_minutes for {job_id}")
    try:
        duration = sum(float(value) for value in stages.values())
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][constraints.profiles._processing_time] "
            f"cause=non_numeric_stage_minutes job_id={job_id} values={stages}"
        )
        raise RuntimeError(f"non-numeric base_stage_minutes for {job_id}") from exc
    if not math.isfinite(duration) or duration <= 0:
        print(
            "[ERROR][constraints.profiles._processing_time] "
            f"cause=invalid_processing_time job_id={job_id} value={duration}"
        )
        raise RuntimeError(f"invalid processing time for {job_id}")
    return duration


def _validate_emitted_batch_aggregates(
    *,
    batch: Mapping[str, Any],
    batch_id: str,
    expected_wo_count: int,
    expected_length_sum: float,
    expected_duration: float,
) -> None:
    """출력 row에 aggregate가 있으면 원본 W/O 재계산값과 같아야 한다."""

    expected = {
        "wo_count": float(expected_wo_count),
        "length_sum": float(expected_length_sum),
        "batch_duration": float(expected_duration),
        "duration": float(expected_duration),
    }
    for field_name, expected_value in expected.items():
        if field_name not in batch:
            continue
        try:
            actual_value = float(batch[field_name])
        except (TypeError, ValueError) as exc:
            print(
                "[ERROR][constraints.profiles._validate_emitted_batch_aggregates] "
                f"cause=non_numeric_batch_field batch_id={batch_id} "
                f"field={field_name} value={batch[field_name]}"
            )
            raise RuntimeError(f"non-numeric Phase 2 batch field: {field_name}") from exc
        if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1e-5):
            print(
                "[ERROR][constraints.profiles._validate_emitted_batch_aggregates] "
                f"cause=batch_aggregate_mismatch batch_id={batch_id} field={field_name} "
                f"actual={actual_value} expected={expected_value}"
            )
            raise RuntimeError(f"Phase 2 batch aggregate mismatch: {batch_id}.{field_name}")


def _boolean_mapping(source: Mapping[str, Any], key: str, location: str) -> Dict[str, bool]:
    values = _required_mapping(source, key, location)
    invalid = [name for name, value in values.items() if not isinstance(value, bool)]
    if invalid:
        print(
            "[ERROR][constraints.profiles._boolean_mapping] "
            f"cause=non_boolean_flags location={location}.{key} keys={invalid}"
        )
        raise RuntimeError(f"{location}.{key} contains non-boolean flags: {invalid}")
    return {str(name): value for name, value in values.items()}


def _attribute_view(value: object, label: str) -> object:
    if isinstance(value, Mapping):
        return SimpleNamespace(**dict(value))
    if value is None:
        print(f"[ERROR][constraints.profiles._attribute_view] cause=missing_object label={label}")
        raise RuntimeError(f"missing {label} object")
    return value


def _required_field(value: object, field_name: str, key: str) -> Any:
    if not hasattr(value, field_name):
        print(
            "[ERROR][constraints.profiles._required_field] "
            f"cause=missing_field field={field_name} key={key}"
        )
        raise RuntimeError(f"missing {field_name} for {key}")
    result = getattr(value, field_name)
    if result is None or (isinstance(result, str) and not result.strip()):
        print(
            "[ERROR][constraints.profiles._required_field] "
            f"cause=empty_field field={field_name} key={key}"
        )
        raise RuntimeError(f"empty {field_name} for {key}")
    return result
