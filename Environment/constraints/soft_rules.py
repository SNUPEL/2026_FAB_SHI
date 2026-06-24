"""하드로 막지 않고 점수로만 반영하는 선호 제약."""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_due_date_urgency(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_due_date_urgency(context: ConstraintContext) -> ConstraintResult:
    """납기 여유가 거의 없는 작업을 너무 늦게 배정하는지 봅니다.

    데이터가 아직 없을 수 있으므로 due_date_minutes가 없는 경우는 통과시킵니다.
    """

    # LINE-BY-LINE: `due_date`에 `context.job.due_date_minutes` 결과를 저장합니다. 의미/사용: `due_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    due_date = context.job.due_date_minutes
    # LINE-BY-LINE: 조건 `due_date is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if due_date is None:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("due_date_urgency", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("due_date_urgency", True, "")

    # LINE-BY-LINE: `estimated_finish`에 `float(context.candidate_finish_time)` 결과를 저장합니다. 의미/사용: `estimated_finish` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    estimated_finish = float(context.candidate_finish_time)
    # LINE-BY-LINE: `slack`에 `due_date - estimated_finish` 결과를 저장합니다. 의미/사용: `slack` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    slack = due_date - estimated_finish
    # LINE-BY-LINE: `passed`에 `slack >= 0` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = slack >= 0
    # LINE-BY-LINE: `penalty`에 `abs(slack) / max(due_date, 1.0) if not passed else 0.0` 결과를 저장합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    penalty = abs(slack) / max(due_date, 1.0) if not passed else 0.0
    # LINE-BY-LINE: `reason`에 `"" if passed else f"due date may be missed by {-slack:.2f} minutes"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"due date may be missed by {-slack:.2f} minutes"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("due_date_urgency", passed, reason, penalty=penalty, metadata={"slack": slack})`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("due_date_urgency", passed, reason, penalty=penalty, metadata={"slack": slack})


# LINE-BY-LINE: `check_preferred_machine_type(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_preferred_machine_type(context: ConstraintContext) -> ConstraintResult:
    """작업이 선호하는 설비 타입에 배정되는지 확인합니다.

    예:
    - 얇은 강판은 laser 선호
    - 특정 계열은 plasma 우선
    """

    # LINE-BY-LINE: `preferred_types`에 `tuple(context.job.preferred_machine_types)` 결과를 저장합니다. 의미/사용: `preferred_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    preferred_types = tuple(context.job.preferred_machine_types)
    # LINE-BY-LINE: 조건 `not preferred_types`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not preferred_types:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("preferred_machine_type", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("preferred_machine_type", True, "")

    # LINE-BY-LINE: 조건 `context.machine.machine_type in preferred_types`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine.machine_type in preferred_types:
        # LINE-BY-LINE: `rank`에 `preferred_types.index(context.machine.machine_type)` 결과를 저장합니다. 의미/사용: `rank` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rank = preferred_types.index(context.machine.machine_type)
        # LINE-BY-LINE: `passed`에 `rank == 0` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
        passed = rank == 0
        # LINE-BY-LINE: `reason`에 `"" if passed else f"{context.machine.machine_type} is lower-ranked for {context.job.job_id}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
        reason = "" if passed else f"{context.machine.machine_type} is lower-ranked for {context.job.job_id}"
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("preferred_machine_type", passed, reason, penalty=0.25 * rank)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("preferred_machine_type", passed, reason, penalty=0.25 * rank)

    # LINE-BY-LINE: `reason`에 `f"{context.machine.machine_type} is not a preferred type for {context.job.job_id}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = f"{context.machine.machine_type} is not a preferred type for {context.job.job_id}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("preferred_machine_type", False, reason, penalty=1.0)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("preferred_machine_type", False, reason, penalty=1.0)


# LINE-BY-LINE: `check_load_balance_preference(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_load_balance_preference(context: ConstraintContext) -> ConstraintResult:
    """부하가 상대적으로 더 높은 설비에 배정하려는지 확인합니다.

    하드 제약은 아니고, 부하 평준화를 유도하기 위한 소프트 가이드입니다.
    """

    # LINE-BY-LINE: `machine_loads`에 `context.state.machine_loads` 결과를 저장합니다. 의미/사용: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_loads = context.state.machine_loads
    # LINE-BY-LINE: `average_load`에 `sum(machine_loads.values()) / max(len(machine_loads), 1)` 결과를 저장합니다. 의미/사용: `average_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    average_load = sum(machine_loads.values()) / max(len(machine_loads), 1)
    # LINE-BY-LINE: `current_load`에 `machine_loads[context.machine.machine_id]` 결과를 저장합니다. 의미/사용: `current_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_load = machine_loads[context.machine.machine_id]
    # LINE-BY-LINE: `overload`에 `max(0.0, current_load - average_load)` 결과를 저장합니다. 의미/사용: `overload` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    overload = max(0.0, current_load - average_load)
    # LINE-BY-LINE: `passed`에 `overload <= 0.0` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = overload <= 0.0
    # LINE-BY-LINE: `reason`에 `"" if passed else f"{context.machine.machine_id} is already above average load"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"{context.machine.machine_id} is already above average load"
    # LINE-BY-LINE: `penalty`에 `overload / max(average_load + 1.0, 1.0)` 결과를 저장합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    penalty = overload / max(average_load + 1.0, 1.0)
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("load_balance_preference", passed, reason, penalty=penalty)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("load_balance_preference", passed, reason, penalty=penalty)
