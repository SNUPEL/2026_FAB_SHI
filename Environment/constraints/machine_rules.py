"""설비 자체와 직접 관련된 제약 함수들.

이 파일에는 "기계가 이 작업을 처리할 수 있는가?"에 가까운 규칙을 둡니다.
즉, 설비 enable 여부, 두께 범위, 길이 한계, 일일 용량 같은 규칙이 여기 들어갑니다.
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_machine_enabled(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_enabled(context: ConstraintContext) -> ConstraintResult:
    """설비가 현재 사용 가능 상태인지 확인합니다."""

    # base machine.enabled 뿐 아니라 날짜별 override까지 반영한 값입니다.
    # LINE-BY-LINE: `passed`에 `bool(context.machine_enabled_flag)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = bool(context.machine_enabled_flag)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"{context.machine.machine_id} is disabled on {context.current_day_key}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"{context.machine.machine_id} is disabled on {context.current_day_key}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_enabled", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_enabled", passed, reason)


# LINE-BY-LINE: `check_family_eligibility(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_family_eligibility(context: ConstraintContext) -> ConstraintResult:
    """해당 작업 계열을 이 설비가 처리할 수 있는지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `context.job.family in set(context.machine.eligible_families)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = context.job.family in set(context.machine.eligible_families)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"family {context.job.family} not supported by {context.machine.machine_id}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"family {context.job.family} not supported by {context.machine.machine_id}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("family_eligibility", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("family_eligibility", passed, reason)


# LINE-BY-LINE: `check_thickness_range(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_thickness_range(context: ConstraintContext) -> ConstraintResult:
    """작업 두께가 설비 허용 범위 안에 들어오는지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `context.machine.min_thickness <= context.job.thickness <= context.machine.max_thickness` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = context.machine.min_thickness <= context.job.thickness <= context.machine.max_thickness
    # LINE-BY-LINE: `reason`에 `"" if passed else f"thickness {context.job.thickness} out of range"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"thickness {context.job.thickness} out of range"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("thickness_range", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("thickness_range", passed, reason)


# LINE-BY-LINE: `check_table_length_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_table_length_limit(context: ConstraintContext) -> ConstraintResult:
    """정반 또는 장비 길이 한계를 넘는지 확인합니다.

    예:
    - 레이저 장비는 10m 초과 불가
    - 플라즈마 장비는 더 큰 길이를 허용할 수 있음
    """

    # LINE-BY-LINE: `passed`에 `context.job.plate_length <= context.machine.table_length_limit` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = context.job.plate_length <= context.machine.table_length_limit
    # LINE-BY-LINE: `reason`에 `"" if passed else f"plate length {context.job.plate_length} exceeds table limit {context.machine....` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"plate length {context.job.plate_length} exceeds table limit {context.machine.table_length_limit}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("table_length_limit", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("table_length_limit", passed, reason)


# LINE-BY-LINE: `check_machine_single_processing(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_single_processing(context: ConstraintContext) -> ConstraintResult:
    """현재 시점에 설비의 처리 슬롯이 남아 있는지 확인합니다."""

    # LINE-BY-LINE: `available_time`에 `context.state.machine_available_at[context.machine.machine_id]` 결과를 저장합니다. 의미/사용: `available_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    available_time = context.state.machine_available_at[context.machine.machine_id]
    # LINE-BY-LINE: `passed`에 `available_time <= context.state.current_time` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = available_time <= context.state.current_time
    # LINE-BY-LINE: `reason`에 `"" if passed else f"all slots of {context.machine.machine_id} busy until {available_time:.2f}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"all slots of {context.machine.machine_id} busy until {available_time:.2f}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_single_processing", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_single_processing", passed, reason)


# LINE-BY-LINE: `check_daily_capacity_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_daily_capacity_limit(context: ConstraintContext) -> ConstraintResult:
    """당일 설비 가용 시간을 초과하는지 확인합니다."""

    # LINE-BY-LINE: `current_load`에 `float(context.machine_daily_load)` 결과를 저장합니다. 의미/사용: `current_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_load = float(context.machine_daily_load)
    # LINE-BY-LINE: `next_load`에 `current_load + float(context.candidate_processing_minutes)` 결과를 저장합니다. 의미/사용: `next_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    next_load = current_load + float(context.candidate_processing_minutes)
    # LINE-BY-LINE: `passed`에 `next_load <= float(context.machine_daily_capacity_limit)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = next_load <= float(context.machine_daily_capacity_limit)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"daily capacity exceeded on {context.machine.machine_id} ({context.current_day...` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"daily capacity exceeded on {context.machine.machine_id} ({context.current_day_key})"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("daily_capacity_limit", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("daily_capacity_limit", passed, reason)


# LINE-BY-LINE: `check_daily_job_cap_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_daily_job_cap_limit(context: ConstraintContext) -> ConstraintResult:
    """오늘 전체 작업 수 제한을 넘는지 확인합니다.

    이 규칙은 날짜별 override가 설정된 경우에만 의미가 있습니다.
    설정이 없으면 자동 통과시킵니다.
    """

    # LINE-BY-LINE: 조건 `context.daily_job_cap is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.daily_job_cap is None:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("daily_job_cap_limit", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("daily_job_cap_limit", True, "")

    # LINE-BY-LINE: `predicted_count`에 `int(context.scheduled_job_count_today) + 1` 결과를 저장합니다. 의미/사용: `predicted_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_count = int(context.scheduled_job_count_today) + 1
    # LINE-BY-LINE: `passed`에 `predicted_count <= int(context.daily_job_cap)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = predicted_count <= int(context.daily_job_cap)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"daily job cap exceeded on {context.current_day_key}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"daily job cap exceeded on {context.current_day_key}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("daily_job_cap_limit", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("daily_job_cap_limit", passed, reason)


# LINE-BY-LINE: `check_auxiliary_resources_available(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_auxiliary_resources_available(context: ConstraintContext) -> ConstraintResult:
    """후보 action이 요구하는 단순 슬롯 자원이 충분한지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `not context.resource_shortages` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = not context.resource_shortages
    # LINE-BY-LINE: 조건 `passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if passed:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("auxiliary_resources_available", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("auxiliary_resources_available", True, "")

    # LINE-BY-LINE: `shortage_desc`에 `", ".join(` 결과를 저장합니다. 의미/사용: `shortage_desc` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    shortage_desc = ", ".join(
        # LINE-BY-LINE: `f"{resource_id} short by {shortage}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"{resource_id} short by {shortage}"
        # LINE-BY-LINE: `resource_id, shortage in sorted(context.resource_shortages.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id, shortage in sorted(context.resource_shortages.items())
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("auxiliary_resources_available", False, shortage_desc)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("auxiliary_resources_available", False, shortage_desc)
