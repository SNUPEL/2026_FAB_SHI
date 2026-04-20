"""설비 자체와 직접 관련된 제약 함수들.

이 파일에는 "기계가 이 작업을 처리할 수 있는가?"에 가까운 규칙을 둡니다.
즉, 설비 enable 여부, 두께 범위, 길이 한계, 일일 용량 같은 규칙이 여기 들어갑니다.
"""

from .base import ConstraintContext, ConstraintResult


def check_machine_enabled(context: ConstraintContext) -> ConstraintResult:
    """설비가 현재 사용 가능 상태인지 확인합니다."""

    # base machine.enabled 뿐 아니라 날짜별 override까지 반영한 값입니다.
    passed = bool(context.machine_enabled_flag)
    reason = "" if passed else f"{context.machine.machine_id} is disabled on {context.current_day_key}"
    return ConstraintResult("machine_enabled", passed, reason)


def check_family_eligibility(context: ConstraintContext) -> ConstraintResult:
    """해당 작업 계열을 이 설비가 처리할 수 있는지 확인합니다."""

    passed = context.job.family in set(context.machine.eligible_families)
    reason = "" if passed else f"family {context.job.family} not supported by {context.machine.machine_id}"
    return ConstraintResult("family_eligibility", passed, reason)


def check_thickness_range(context: ConstraintContext) -> ConstraintResult:
    """작업 두께가 설비 허용 범위 안에 들어오는지 확인합니다."""

    passed = context.machine.min_thickness <= context.job.thickness <= context.machine.max_thickness
    reason = "" if passed else f"thickness {context.job.thickness} out of range"
    return ConstraintResult("thickness_range", passed, reason)


def check_table_length_limit(context: ConstraintContext) -> ConstraintResult:
    """정반 또는 장비 길이 한계를 넘는지 확인합니다.

    예:
    - 레이저 장비는 10m 초과 불가
    - 플라즈마 장비는 더 큰 길이를 허용할 수 있음
    """

    passed = context.job.plate_length <= context.machine.table_length_limit
    reason = "" if passed else f"plate length {context.job.plate_length} exceeds table limit {context.machine.table_length_limit}"
    return ConstraintResult("table_length_limit", passed, reason)


def check_machine_single_processing(context: ConstraintContext) -> ConstraintResult:
    """현재 시점에 설비의 처리 슬롯이 남아 있는지 확인합니다."""

    available_time = context.state.machine_available_at[context.machine.machine_id]
    passed = available_time <= context.state.current_time
    reason = "" if passed else f"all slots of {context.machine.machine_id} busy until {available_time:.2f}"
    return ConstraintResult("machine_single_processing", passed, reason)


def check_daily_capacity_limit(context: ConstraintContext) -> ConstraintResult:
    """당일 설비 가용 시간을 초과하는지 확인합니다."""

    current_load = float(context.machine_daily_load)
    next_load = current_load + float(context.candidate_processing_minutes)
    passed = next_load <= float(context.machine_daily_capacity_limit)
    reason = "" if passed else f"daily capacity exceeded on {context.machine.machine_id} ({context.current_day_key})"
    return ConstraintResult("daily_capacity_limit", passed, reason)


def check_daily_job_cap_limit(context: ConstraintContext) -> ConstraintResult:
    """오늘 전체 작업 수 제한을 넘는지 확인합니다.

    이 규칙은 날짜별 override가 설정된 경우에만 의미가 있습니다.
    설정이 없으면 자동 통과시킵니다.
    """

    if context.daily_job_cap is None:
        return ConstraintResult("daily_job_cap_limit", True, "")

    predicted_count = int(context.scheduled_job_count_today) + 1
    passed = predicted_count <= int(context.daily_job_cap)
    reason = "" if passed else f"daily job cap exceeded on {context.current_day_key}"
    return ConstraintResult("daily_job_cap_limit", passed, reason)


def check_auxiliary_resources_available(context: ConstraintContext) -> ConstraintResult:
    """후보 action이 요구하는 단순 슬롯 자원이 충분한지 확인합니다."""

    passed = not context.resource_shortages
    if passed:
        return ConstraintResult("auxiliary_resources_available", True, "")

    shortage_desc = ", ".join(
        f"{resource_id} short by {shortage}"
        for resource_id, shortage in sorted(context.resource_shortages.items())
    )
    return ConstraintResult("auxiliary_resources_available", False, shortage_desc)
