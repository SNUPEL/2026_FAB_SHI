"""Bay/layout 관련 hard constraint.

이 파일은 "설비가 어느 Bay에 속하는가"와 "블록 묶음이 같은 Bay에 있어야 하는가"를 본다.

중요 구분:
- `Machine.bay_id`: 설비가 물리적으로 놓인 절단 Bay
- `Job.cut_bay`: 이 W/O가 고정되어야 하는 절단 Bay가 있을 때 쓰는 값
- `Job.downstream_bay`: 절단 이후 후공정/적치 Bay

`CUT_BAY`와 downstream bay를 섞으면 검증이 틀어지므로 이 파일에서 분리해서 검사한다.
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `check_machine_bay_consistency(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_bay_consistency(context: ConstraintContext) -> ConstraintResult:
    """선택된 machine의 Bay 정보가 유효하고, job의 고정 cut_bay가 있으면 일치하는지 확인.

    예:
    - machine PLS21의 bay_id가 22
    - job.cut_bay가 23으로 고정되어 있으면 hard fail
    """

    # LINE-BY-LINE: `machine_bay`에 `context.machine_bay_id` 결과를 저장합니다. 의미/사용: `machine_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_bay = context.machine_bay_id
    # LINE-BY-LINE: 조건 `not machine_bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not machine_bay:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult(
            # LINE-BY-LINE: 문자열 값 `"machine_bay_consistency"`를 `ConstraintResult(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "machine_bay_consistency",
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `False` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            False,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `f"{context.machine.machine_id} has no bay_id"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            f"{context.machine.machine_id} has no bay_id",
        )

    # LINE-BY-LINE: `job_cut_bay`에 `context.job_cut_bay` 결과를 저장합니다. 의미/사용: `job_cut_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    job_cut_bay = context.job_cut_bay
    # LINE-BY-LINE: 조건 `job_cut_bay and str(job_cut_bay) != str(machine_bay)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if job_cut_bay and str(job_cut_bay) != str(machine_bay):
        # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult(
            # LINE-BY-LINE: 문자열 값 `"machine_bay_consistency"`를 `ConstraintResult(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "machine_bay_consistency",
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `False` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            False,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `f"job fixed cut_bay {job_cut_bay} does not match machine bay {machine_bay}"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            f"job fixed cut_bay {job_cut_bay} does not match machine bay {machine_bay}",
        )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("machine_bay_consistency", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("machine_bay_consistency", True, "")


# LINE-BY-LINE: `check_block_set_same_bay(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_block_set_same_bay(context: ConstraintContext) -> ConstraintResult:
    """동일 block_set_id는 이미 배정된 Bay와 같은 Bay에만 배정한다.

    현재 block_set_id는 보통 `project_no::block_no` 형태다.
    이미 같은 block_set_id가 Bay 22에 배정되었다면, 이후 같은 block은 Bay 23 후보가 hard fail된다.
    """

    # LINE-BY-LINE: `block_set_id`에 `context.block_set_id` 결과를 저장합니다. 의미/사용: `block_set_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    block_set_id = context.block_set_id
    # LINE-BY-LINE: 조건 `not block_set_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not block_set_id:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("block_set_same_bay", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("block_set_same_bay", True, "")

    # LINE-BY-LINE: `assigned_bay`에 `context.assigned_block_set_bay` 결과를 저장합니다. 의미/사용: `assigned_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    assigned_bay = context.assigned_block_set_bay
    # LINE-BY-LINE: 조건 `not assigned_bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not assigned_bay:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult("block_set_same_bay", True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult("block_set_same_bay", True, "")

    # LINE-BY-LINE: `machine_bay`에 `context.machine_bay_id` 결과를 저장합니다. 의미/사용: `machine_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_bay = context.machine_bay_id
    # LINE-BY-LINE: `passed`에 `str(machine_bay) == str(assigned_bay)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = str(machine_bay) == str(assigned_bay)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"block_set {block_set_id} already assigned to bay {assigned_bay}, "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"block_set {block_set_id} already assigned to bay {assigned_bay}, "
        # LINE-BY-LINE: `f"candidate bay is {machine_bay}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"candidate bay is {machine_bay}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult("block_set_same_bay", passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult("block_set_same_bay", passed, reason)
