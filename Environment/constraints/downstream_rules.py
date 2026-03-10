"""후공정 베이와 적치 관련 제약 함수들."""

from .base import ConstraintContext, ConstraintResult


def check_downstream_capacity(context: ConstraintContext) -> ConstraintResult:
    """후공정 베이 적치 한계를 넘는지 확인합니다.

    이 제약은 현업에서 강하게 막아야 할 가능성이 높으므로
    기본값은 하드 제약으로 두었습니다.
    """

    bay = context.bays[context.job.downstream_bay]
    predicted_load = context.state.downstream_loads[bay.bay_id] + 1
    passed = predicted_load <= bay.capacity_limit
    reason = "" if passed else f"downstream capacity exceeded on {bay.bay_id}"
    return ConstraintResult("downstream_capacity", passed, reason)


def check_downstream_priority(context: ConstraintContext) -> ConstraintResult:
    """더 높은 우선순위의 후공정 베이가 남아있는지 확인합니다.

    현업에서 항상 강제해야 하는 규칙인지,
    아니면 가능한 한 우선 고려하는 규칙인지 아직 확정되지 않았으므로
    기본값은 소프트 제약으로 두는 것이 안전합니다.
    """

    current_rank = context.bays[context.job.downstream_bay].priority_rank
    remaining_ranks = [
        context.bays[other_job.downstream_bay].priority_rank
        for job_id, other_job in context.jobs.items()
        if job_id in context.state.unscheduled_jobs and job_id != context.job.job_id
    ]
    min_rank = min(remaining_ranks) if remaining_ranks else current_rank
    passed = current_rank <= min_rank
    reason = "" if passed else f"higher-priority downstream bay exists before {context.job.downstream_bay}"
    return ConstraintResult("downstream_priority", passed, reason)


def check_downstream_buffer_warning(context: ConstraintContext) -> ConstraintResult:
    """적치 초과는 아니지만, 적치량이 위험 구간에 들어오는지 확인합니다.

    예:
    - capacity_limit이 10일 때
    - 8~9 수준이면 아직 하드 위반은 아니지만 위험 구간으로 판단 가능
    """

    bay = context.bays[context.job.downstream_bay]
    predicted_load = context.state.downstream_loads[bay.bay_id] + 1
    usage_ratio = predicted_load / max(bay.capacity_limit, 1)
    passed = usage_ratio < 0.8
    reason = "" if passed else f"downstream buffer warning on {bay.bay_id}: usage_ratio={usage_ratio:.2f}"
    return ConstraintResult(
        "downstream_buffer_warning",
        passed,
        reason,
        penalty=max(0.0, usage_ratio - 0.8),
        metadata={"usage_ratio": usage_ratio},
    )
