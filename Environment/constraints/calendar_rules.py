"""캘린더 / 고장 관련 제약 함수들.

이 파일은 "현재 시각에 공장이 가동 가능한가?"와
"현재 시각에 특정 설비가 고장/정지 상태인가?"를 판단합니다.

중요:
- 현재 구현은 `작업 시작 시점` 기준으로만 판단합니다.
- 즉, 작업이 이미 시작된 뒤 점심시간이나 고장 시간이 중간에 끼어드는
  preemption / resume 로직은 아직 구현하지 않았습니다.
- 초반 연구용/스켈레톤 단계에서는 이 정도가 가장 단순하고 안전합니다.
"""

from .base import ConstraintContext, ConstraintResult


def check_calendar_open(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 공장 전체가 가동 가능한지 확인합니다."""

    passed = bool(context.global_calendar_open)
    reason = "" if passed else (context.global_calendar_reason or f"calendar closed on {context.current_day_key}")
    return ConstraintResult("calendar_open", passed, reason)


def check_machine_calendar_open(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 특정 설비가 계획된 운영시간 안에 있는지 확인합니다."""

    passed = bool(context.machine_calendar_open)
    reason = "" if passed else (
        context.machine_calendar_reason or f"{context.machine.machine_id} outside operating window on {context.current_day_key}"
    )
    return ConstraintResult("machine_calendar_open", passed, reason)


def check_machine_breakdown(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 특정 설비가 고장/정지 상태인지 확인합니다."""

    passed = not bool(context.machine_breakdown_active)
    reason = "" if passed else (
        context.machine_breakdown_reason or f"{context.machine.machine_id} breakdown on {context.current_day_key}"
    )
    return ConstraintResult("machine_breakdown", passed, reason)
