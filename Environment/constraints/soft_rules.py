"""하드로 막지 않고 점수로만 반영하는 선호 제약."""

from .base import ConstraintContext, ConstraintResult


def check_due_date_urgency(context: ConstraintContext) -> ConstraintResult:
    """납기 여유가 거의 없는 작업을 너무 늦게 배정하는지 봅니다.

    데이터가 아직 없을 수 있으므로 due_date_minutes가 없는 경우는 통과시킵니다.
    """

    due_date = context.job.due_date_minutes
    if due_date is None:
        return ConstraintResult("due_date_urgency", True, "")

    estimated_finish = context.state.current_time + context.job.estimate_total_minutes(context.machine)
    slack = due_date - estimated_finish
    passed = slack >= 0
    penalty = abs(slack) / max(due_date, 1.0) if not passed else 0.0
    reason = "" if passed else f"due date may be missed by {-slack:.2f} minutes"
    return ConstraintResult("due_date_urgency", passed, reason, penalty=penalty, metadata={"slack": slack})


def check_preferred_machine_type(context: ConstraintContext) -> ConstraintResult:
    """작업이 선호하는 설비 타입에 배정되는지 확인합니다.

    예:
    - 얇은 강판은 laser 선호
    - 특정 계열은 plasma 우선
    """

    preferred_types = tuple(context.job.preferred_machine_types)
    if not preferred_types:
        return ConstraintResult("preferred_machine_type", True, "")

    passed = context.machine.machine_type in preferred_types
    reason = "" if passed else f"{context.machine.machine_type} is not a preferred type for {context.job.job_id}"
    return ConstraintResult("preferred_machine_type", passed, reason, penalty=0.5 if not passed else 0.0)


def check_load_balance_preference(context: ConstraintContext) -> ConstraintResult:
    """부하가 상대적으로 더 높은 설비에 배정하려는지 확인합니다.

    하드 제약은 아니고, 부하 평준화를 유도하기 위한 소프트 가이드입니다.
    """

    machine_loads = context.state.machine_loads
    average_load = sum(machine_loads.values()) / max(len(machine_loads), 1)
    current_load = machine_loads[context.machine.machine_id]
    overload = max(0.0, current_load - average_load)
    passed = overload <= 0.0
    reason = "" if passed else f"{context.machine.machine_id} is already above average load"
    penalty = overload / max(average_load + 1.0, 1.0)
    return ConstraintResult("load_balance_preference", passed, reason, penalty=penalty)
