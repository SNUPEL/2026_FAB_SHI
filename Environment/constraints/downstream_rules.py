"""후공정 베이와 적치 관련 제약 함수들.

절단이 끝난 W/O는 바로 사라지는 것이 아니라 후공정/적치 Bay로 이동한다.
이 파일의 rule은 그 후공정 쪽 capacity와 우선순위를 검사한다.

현재 데이터 한계:
- 실제 Bay IN/OUT, 적치 면적, 크레인 이동시간이 아직 없다.
- 따라서 지금 rule은 scenario에 있는 단순 capacity와 priority만 사용한다.
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_downstream_capacity(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_downstream_capacity(context: ConstraintContext) -> ConstraintResult:
    """후공정 베이 적치 한계를 넘는지 확인합니다.

    이 제약은 현업에서 강하게 막아야 할 가능성이 높으므로
    기본값은 하드 제약으로 두었습니다.
    """

    # 후보 action이 완료되어 downstream bay에 도착했을 때의 예상 적치량이다.
    # LINE-BY-LINE: `bay`에 `context.bays[context.job.downstream_bay]` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
    bay = context.bays[context.job.downstream_bay]
    # LINE-BY-LINE: `predicted_load`에 `int(context.predicted_downstream_load_at_arrival)` 결과를 저장합니다. 의미/사용: `predicted_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_load = int(context.predicted_downstream_load_at_arrival)
    # LINE-BY-LINE: `passed`에 `predicted_load <= bay.capacity_limit` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = predicted_load <= bay.capacity_limit
    # LINE-BY-LINE: `reason`에 `"" if passed else f"downstream capacity exceeded on {bay.bay_id}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"downstream capacity exceeded on {bay.bay_id}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("downstream_capacity", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("downstream_capacity", passed, reason)


# LINE-BY-LINE: `check_downstream_priority(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_downstream_priority(context: ConstraintContext) -> ConstraintResult:
    """더 높은 우선순위의 후공정 베이가 남아있는지 확인합니다.

    현업에서 항상 강제해야 하는 규칙인지,
    아니면 가능한 한 우선 고려하는 규칙인지 아직 확정되지 않았으므로
    기본값은 소프트 제약으로 두는 것이 안전합니다.
    """

    # 낮은 rank가 더 높은 우선순위라고 본다.
    # LINE-BY-LINE: `current_rank`에 `context.bays[context.job.downstream_bay].priority_rank` 결과를 저장합니다. 의미/사용: `current_rank` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_rank = context.bays[context.job.downstream_bay].priority_rank
    # LINE-BY-LINE: `remaining_ranks`에 `[` 결과를 저장합니다. 의미/사용: `remaining_ranks` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    remaining_ranks = [
        # LINE-BY-LINE: `context.bays[other_job.downstream_bay].priority_rank` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        context.bays[other_job.downstream_bay].priority_rank
        # LINE-BY-LINE: `job_id, other_job in context.jobs.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job_id, other_job in context.jobs.items()
        # LINE-BY-LINE: 조건 `job_id in context.state.unscheduled_jobs and job_id != context.job.job_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if job_id in context.state.unscheduled_jobs and job_id != context.job.job_id
    ]
    # LINE-BY-LINE: `min_rank`에 `min(remaining_ranks) if remaining_ranks else current_rank` 결과를 저장합니다. 의미/사용: `min_rank` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    min_rank = min(remaining_ranks) if remaining_ranks else current_rank
    # LINE-BY-LINE: `passed`에 `current_rank <= min_rank` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = current_rank <= min_rank
    # LINE-BY-LINE: `reason`에 `"" if passed else f"higher-priority downstream bay exists before {context.job.downstream_bay}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"higher-priority downstream bay exists before {context.job.downstream_bay}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("downstream_priority", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("downstream_priority", passed, reason)


# LINE-BY-LINE: `check_downstream_buffer_warning(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_downstream_buffer_warning(context: ConstraintContext) -> ConstraintResult:
    """적치 초과는 아니지만, 적치량이 위험 구간에 들어오는지 확인합니다.

    예:
    - capacity_limit이 10일 때
    - 8~9 수준이면 아직 하드 위반은 아니지만 위험 구간으로 판단 가능
    """

    # hard fail은 아니지만 80% 이상이면 soft warning penalty를 준다.
    # LINE-BY-LINE: `bay`에 `context.bays[context.job.downstream_bay]` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
    bay = context.bays[context.job.downstream_bay]
    # LINE-BY-LINE: `predicted_load`에 `int(context.predicted_downstream_load_at_arrival)` 결과를 저장합니다. 의미/사용: `predicted_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_load = int(context.predicted_downstream_load_at_arrival)
    # LINE-BY-LINE: `usage_ratio`에 `predicted_load / max(bay.capacity_limit, 1)` 결과를 저장합니다. 의미/사용: `usage_ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    usage_ratio = predicted_load / max(bay.capacity_limit, 1)
    # LINE-BY-LINE: `passed`에 `usage_ratio < 0.8` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = usage_ratio < 0.8
    # LINE-BY-LINE: `reason`에 `"" if passed else f"downstream buffer warning on {bay.bay_id}: usage_ratio={usage_ratio:.2f}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"downstream buffer warning on {bay.bay_id}: usage_ratio={usage_ratio:.2f}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(
        # LINE-BY-LINE: 문자열 값 `"downstream_buffer_warning"`를 `ConstraintResult(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "downstream_buffer_warning",
        # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `passed` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        passed,
        # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `reason` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        reason,
        # LINE-BY-LINE: `penalty`에 `max(0.0, usage_ratio - 0.8)` 결과를 저장합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        penalty=max(0.0, usage_ratio - 0.8),
        # LINE-BY-LINE: `metadata`에 `{"usage_ratio": usage_ratio}` 결과를 저장합니다. 의미/사용: `metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        metadata={"usage_ratio": usage_ratio},
    )
