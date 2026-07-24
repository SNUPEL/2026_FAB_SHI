"""설비 동시 작업 묶음 용량 관련 제약.

이 파일은 "현재 설비/정반 위에 얼마나 올라와 있는가"와
"하루 capacity를 넘는가"처럼 수량/길이/시간 capacity를 판단한다.

현재 중요한 해석:
- `machine_day_wo_count_limit`라는 이름은 과거 config 호환용이다.
- 과거에는 특정 시점 active W/O 수/길이합 rolling 제약으로 해석했다.
- 현업 추가 확인 후 기본 generated planning에서는 이 rule을 끄고, `batch_wo_count_limit`과
  `batch_length_sum_limit`을 사용한다.
- actual replay에서 동일 timestamp 다중 W/O는 동시작업 확정 근거가 아니라 데이터 오류 근거다.

legacy rolling capacity 기준:

```text
active_WO_count(machine, t) <= 3
sum(active_WO.length for same machine at t) <= 55000
```
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_machine_day_wo_count_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_day_wo_count_limit(context: ConstraintContext) -> ConstraintResult:
    """같은 설비에 동시에 올라가는 W/O 수 제한.

    rule 이름은 기존 config 호환을 위해 machine_day_*를 유지하지만,
    현재 기본 generated 환경에서는 batch rule을 사용한다.
    이 함수는 legacy rolling/debug 또는 별도 active interval audit에서만 쓴다.
    """

    # 후보 W/O가 지금 시작된다고 가정하고 기존 active W/O 수에 1을 더한다.
    # 이 값이 limit을 넘으면 이번 후보는 정반/라인 capacity를 초과한다.
    # LINE-BY-LINE: `limit`에 `context.machine_day_wo_count_limit` 결과를 저장합니다. 의미/사용: `limit`는 현재 rule의 허용 한계값입니다. 예: W/O 수 3 또는 길이합 55000.
    limit = context.machine_day_wo_count_limit
    # LINE-BY-LINE: 조건 `limit is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if limit is None:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_day_wo_count_limit", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("machine_day_wo_count_limit", True, "")

    # LINE-BY-LINE: `next_count`에 `int(context.machine_concurrent_job_count) + 1` 결과를 저장합니다. 의미/사용: `next_count`는 기존 active W/O 수에 후보 W/O 1개를 더한 예측 count입니다.
    next_count = int(context.machine_concurrent_job_count) + 1
    # LINE-BY-LINE: `passed`에 `next_count <= int(limit)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = next_count <= int(limit)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"{context.machine.machine_id} exceeds concurrent W/O count limit at "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"{context.machine.machine_id} exceeds concurrent W/O count limit at "
        # LINE-BY-LINE: `f"{context.candidate_start_time:.2f}: {next_count}>{limit}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"{context.candidate_start_time:.2f}: {next_count}>{limit}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_day_wo_count_limit", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_day_wo_count_limit", passed, reason)


# LINE-BY-LINE: `check_machine_day_length_sum_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_day_length_sum_limit(context: ConstraintContext) -> ConstraintResult:
    """같은 설비에 동시에 올라가는 W/O 길이 합 제한.

    legacy rolling capacity에서는 W/O 하나가 끝나면 active set에서 빠진다.
    현업 확인 결과 기본 모델은 batch이므로, generated planning 기본값에서는 이 rule을 끈다.
    """

    # 기존 active 길이합 + 이번 후보 W/O 길이가 55000을 넘는지 확인한다.
    # LINE-BY-LINE: `limit`에 `context.machine_day_length_sum_limit` 결과를 저장합니다. 의미/사용: `limit`는 현재 rule의 허용 한계값입니다. 예: W/O 수 3 또는 길이합 55000.
    limit = context.machine_day_length_sum_limit
    # LINE-BY-LINE: 조건 `limit is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if limit is None:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_day_length_sum_limit", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("machine_day_length_sum_limit", True, "")

    # LINE-BY-LINE: `next_length`에 `float(context.machine_concurrent_length_sum) + float(context.job.plate_length)` 결과를 저장합니다. 의미/사용: `next_length`는 기존 active 길이합에 후보 W/O 길이를 더한 예측 길이합입니다.
    next_length = float(context.machine_concurrent_length_sum) + float(context.job.plate_length)
    # LINE-BY-LINE: `passed`에 `next_length <= float(limit)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = next_length <= float(limit)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"{context.machine.machine_id} exceeds concurrent length sum limit at "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"{context.machine.machine_id} exceeds concurrent length sum limit at "
        # LINE-BY-LINE: `f"{context.candidate_start_time:.2f}: {next_length:.2f}>{float(limit):.2f}"`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        f"{context.candidate_start_time:.2f}: {next_length:.2f}>{float(limit):.2f}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_day_length_sum_limit", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_day_length_sum_limit", passed, reason)
