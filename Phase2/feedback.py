"""Frozen merged Phase 2 schedule score used by Phase 1 self-labeling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from Environment.constraints.profiles import PhaseConstraintProfile
from Phase2.merged import build_phase2_batch_machine_candidate_bank
from Phase2.run_spec import phase_constraint_profile_to_dict, validate_phase2_run_spec
from Phase2.set_pointer_policy import Phase2SetPointerPolicy


Phase2ScheduleFeedbackScorer = Callable[
    [object, Mapping[str, object], Sequence[str]],
    tuple[int | float, ...],
]


def build_phase2_feedback_contract(
    checkpoint_path: str | Path,
    run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Frozen Phase 2 가중치와 실행 계약을 Phase 1 checkpoint에 기록한다."""

    validate_phase2_run_spec(run_spec)
    path = Path(checkpoint_path)
    if not path.is_file():
        print(
            "[ERROR][Phase2.feedback.build_phase2_feedback_contract] "
            f"cause=missing_checkpoint path={path}"
        )
        raise RuntimeError(f"missing Phase 2 feedback checkpoint: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        print(
            "[ERROR][Phase2.feedback.build_phase2_feedback_contract] "
            f"cause=checkpoint_read_failed path={path} error={exc}"
        )
        raise RuntimeError(f"failed to fingerprint Phase 2 checkpoint: {path}") from exc
    normalized_run_spec = json.loads(json.dumps(dict(run_spec), ensure_ascii=False, sort_keys=True))
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "run_spec": normalized_run_spec,
    }


def build_frozen_phase2_schedule_feedback_scorer(
    *,
    model: Phase2SetPointerPolicy,
    machines: Mapping[str, object],
    run_spec: Mapping[str, Any],
    constraint_profile: PhaseConstraintProfile,
) -> Phase2ScheduleFeedbackScorer:
    """Phase 1 후보마다 frozen Phase 2 best-of-K schedule score를 계산한다."""

    if not isinstance(model, Phase2SetPointerPolicy):
        print(
            "[ERROR][Phase2.feedback.build_frozen_phase2_schedule_feedback_scorer] "
            f"cause=invalid_model_type type={type(model).__name__}"
        )
        raise RuntimeError("Phase 2 feedback requires a set-pointer policy")
    validate_phase2_run_spec(run_spec)
    if phase_constraint_profile_to_dict(constraint_profile) != run_spec["constraint_profile"]:
        print(
            "[ERROR][Phase2.feedback.build_frozen_phase2_schedule_feedback_scorer] "
            "cause=constraint_profile_mismatch"
        )
        raise RuntimeError("Phase 2 feedback constraint profile differs from the checkpoint RunSpec")
    expected_weights = {
        str(bay_id): float(weight)
        for bay_id, weight in run_spec["phase1_bay_capacity_weights"].items()
    }
    actual_weights = _enabled_machine_counts(machines)
    if actual_weights != expected_weights:
        print(
            "[ERROR][Phase2.feedback.build_frozen_phase2_schedule_feedback_scorer] "
            f"cause=machine_capacity_mismatch expected={expected_weights} actual={actual_weights}"
        )
        raise RuntimeError("Phase 2 feedback machine capacity differs from the checkpoint RunSpec")

    frozen_model = model.eval()
    score_mode = str(run_spec["score_mode"])
    max_wo_count = int(run_spec["max_wo_count"])
    max_length_sum = float(run_spec["max_length_sum"])
    action_pool_limit = run_spec["action_pool_limit"]
    rollout_samples = int(run_spec["validation_rollout_samples"])

    def score(candidate: object, jobs: Mapping[str, object], bay_ids: Sequence[str]) -> tuple[int | float, ...]:
        assignments = getattr(candidate, "assignments", None)
        if not isinstance(assignments, Mapping) or not assignments:
            print(
                "[ERROR][Phase2.feedback.score] "
                f"cause=missing_phase1_assignments candidate_type={type(candidate).__name__}"
            )
            raise RuntimeError("Phase 2 feedback requires a complete Phase 1 assignment candidate")
        normalized_bays = tuple(str(bay_id) for bay_id in bay_ids)
        if set(normalized_bays) != set(expected_weights):
            print(
                "[ERROR][Phase2.feedback.score] "
                f"cause=bay_contract_mismatch expected={sorted(expected_weights)} actual={sorted(normalized_bays)}"
            )
            raise RuntimeError("Phase 2 feedback Bay IDs differ from the checkpoint RunSpec")
        if not jobs:
            print("[ERROR][Phase2.feedback.score] cause=no_jobs")
            raise RuntimeError("Phase 2 feedback requires W/O-level jobs")
        seed = _feedback_seed(assignments, jobs)
        agent_candidates = build_phase2_batch_machine_candidate_bank(
            jobs=jobs,
            machines=machines,
            phase1_assignments={str(key): str(value) for key, value in assignments.items()},
            model=frozen_model,
            heuristic_algorithms=(),
            rollout_samples=rollout_samples,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            seed=seed,
            score_mode=score_mode,
            constraint_profile=constraint_profile,
        )
        best = min(agent_candidates, key=lambda item: (*item.score_tuple, item.source))
        return tuple(best.score_tuple)

    return score


def _enabled_machine_counts(machines: Mapping[str, object]) -> dict[str, float]:
    if not machines:
        print("[ERROR][Phase2.feedback._enabled_machine_counts] cause=no_machines")
        raise RuntimeError("Phase 2 feedback requires machines")
    counts: dict[str, float] = {}
    for machine_id, machine in machines.items():
        enabled = _required_field(machine, "enabled", str(machine_id))
        if not isinstance(enabled, bool):
            print(
                "[ERROR][Phase2.feedback._enabled_machine_counts] "
                f"cause=non_boolean_enabled machine_id={machine_id} value={enabled}"
            )
            raise RuntimeError(f"Phase 2 feedback machine enabled must be boolean: {machine_id}")
        if not enabled:
            continue
        bay_id = str(_required_field(machine, "bay_id", str(machine_id)))
        counts[bay_id] = counts.get(bay_id, 0.0) + 1.0
    if not counts:
        print("[ERROR][Phase2.feedback._enabled_machine_counts] cause=no_enabled_machines")
        raise RuntimeError("Phase 2 feedback requires enabled machines")
    return counts


def _feedback_seed(assignments: Mapping[object, object], jobs: Mapping[str, object]) -> int:
    payload = "|".join(
        [*(f"{key}={assignments[key]}" for key in sorted(assignments, key=str)), *sorted(str(job_id) for job_id in jobs)]
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _required_field(item: object, field_name: str, context: str) -> object:
    value = item.get(field_name) if isinstance(item, Mapping) else getattr(item, field_name, None)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        print(
            "[ERROR][Phase2.feedback._required_field] "
            f"cause=missing_or_invalid_field context={context} field={field_name} value={value}"
        )
        raise RuntimeError(f"Phase 2 feedback missing field {field_name}: {context}")
    return value


__all__ = [
    "Phase2ScheduleFeedbackScorer",
    "build_frozen_phase2_schedule_feedback_scorer",
    "build_phase2_feedback_contract",
]
