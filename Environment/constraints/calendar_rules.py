"""캘린더 / 고장 관련 제약 함수들.

이 파일은 "현재 시각에 공장이 가동 가능한가?"와
"현재 시각에 특정 설비가 고장/정지 상태인가?"를 판단합니다.

중요:
- 현재 구현은 `작업 시작 시점` 기준으로만 판단합니다.
- 즉, 작업이 이미 시작된 뒤 점심시간이나 고장 시간이 중간에 끼어드는
  preemption / resume 로직은 아직 구현하지 않았습니다.
- 초반 연구용/스켈레톤 단계에서는 이 정도가 가장 단순하고 안전합니다.
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_calendar_open(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_calendar_open(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 공장 전체가 가동 가능한지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `bool(context.global_calendar_open)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = bool(context.global_calendar_open)
    # LINE-BY-LINE: `reason`에 `"" if passed else (context.global_calendar_reason or f"calendar closed on {context.current_day_ke...` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (context.global_calendar_reason or f"calendar closed on {context.current_day_key}")
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("calendar_open", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("calendar_open", passed, reason)


# LINE-BY-LINE: `check_machine_calendar_open(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_calendar_open(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 특정 설비가 계획된 운영시간 안에 있는지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `bool(context.machine_calendar_open)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = bool(context.machine_calendar_open)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `context.machine_calendar_reason or f"{context.machine.machine_id} outside operating window on {co...` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        context.machine_calendar_reason or f"{context.machine.machine_id} outside operating window on {context.current_day_key}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_calendar_open", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_calendar_open", passed, reason)


# LINE-BY-LINE: `check_machine_breakdown(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_breakdown(context: ConstraintContext) -> ConstraintResult:
    """현재 시각에 특정 설비가 고장/정지 상태인지 확인합니다."""

    # LINE-BY-LINE: `passed`에 `not bool(context.machine_breakdown_active)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = not bool(context.machine_breakdown_active)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `context.machine_breakdown_reason or f"{context.machine.machine_id} breakdown on {context.current_...` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        context.machine_breakdown_reason or f"{context.machine.machine_id} breakdown on {context.current_day_key}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_breakdown", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_breakdown", passed, reason)
