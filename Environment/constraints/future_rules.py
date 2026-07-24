"""미래 확장 제약의 스켈레톤.

이 모듈은 "아직 데이터가 없는 제약"을 나중에 안전하게 추가하기 위한 자리다.

import 흐름:
- `Environment/constraints/registry.py`가 이 파일의 함수들을 import한다.
  예: `from .future_rules import check_batch_wo_count_limit`
- `registry.py`의 `RULES` dict에 rule 이름과 함수를 등록한다.
  예: `"batch_wo_count_limit": check_batch_wo_count_limit`
- `CuttingSimulation`은 `ConstraintManager.evaluate_candidate(context)`만 호출한다.
  즉 simulation 본문은 어떤 rule 파일에 함수가 있는지 몰라도 된다.

공통 입력:
- 모든 함수는 `ConstraintContext` 1개를 입력으로 받는다.
- `ConstraintContext`는 `Environment/constraints/base.py`에 정의되어 있다.
- context 안에는 `job`, `machine`, `state`, `config`, batch 정보, downstream 정보가 들어 있다.

공통 출력:
- 모든 함수는 `ConstraintResult` 1개를 반환한다.
- `passed=True`: 제약 만족
- `passed=False`: 제약 위반 또는 필요한 데이터 누락
- `reason`: 사람이 읽을 수 있는 실패 원인
- `metadata`: 후속 report/debug용 구조화 정보

입출력 예시:
```python
context.candidate_batch_wo_count = 4
context.batch_wo_count_limit = 3
result = check_batch_wo_count_limit(context)

assert result.passed is False
assert result.reason == "batch W/O count exceeded: 4>3"
```

중요 원칙:
- 이 파일의 규칙들은 기본 config에서 꺼져 있어야 한다.
- 나중에 현업 데이터가 들어와 rule을 켜면, 필요한 필드가 None인 경우
  조용히 통과하지 않고 `ConstraintResult(False, reason)`를 반환한다.
- 즉, "데이터가 없으니 일단 통과"는 금지한다.
"""

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult


# LINE-BY-LINE: `_missing_result(rule_name: str, field_name: str)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def _missing_result(rule_name: str, field_name: str) -> ConstraintResult:
    """필수 데이터가 없을 때 사용하는 표준 실패 결과.

    왜 helper로 뺐는가:
    - 모든 미래 제약에서 missing data 표현을 동일하게 하기 위해서다.
    - report에서 `metadata["missing_field"]`만 보면 어떤 값이 필요한지 알 수 있다.

    입력 예:
    - `rule_name="plate_width_range"`
    - `field_name="job.plate_width"`

    출력 예:
    - passed=False
    - reason="required field missing for plate_width_range: job.plate_width"
    """

    # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(
        # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `rule_name` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        rule_name,
        # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `False` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        False,
        # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `f"required field missing for {rule_name}: {field_name}"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        f"required field missing for {rule_name}: {field_name}",
        # LINE-BY-LINE: `metadata`에 `{"missing_field": field_name}` 결과를 저장합니다. 의미/사용: `metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        metadata={"missing_field": field_name},
    )


# LINE-BY-LINE: `check_batch_wo_count_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_batch_wo_count_limit(context: ConstraintContext) -> ConstraintResult:
    """batch/open-batch 후보의 W/O 수 제한."""

    # LINE-BY-LINE: `rule_name`에 `"batch_wo_count_limit"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "batch_wo_count_limit"
    # LINE-BY-LINE: 조건 `context.batch_wo_count_limit is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.batch_wo_count_limit is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "batch_wo_count_limit")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "batch_wo_count_limit")
    # LINE-BY-LINE: 조건 `context.candidate_batch_wo_count is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.candidate_batch_wo_count is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "candidate_batch_wo_count")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "candidate_batch_wo_count")

    # LINE-BY-LINE: `passed`에 `int(context.candidate_batch_wo_count) <= int(context.batch_wo_count_limit)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = int(context.candidate_batch_wo_count) <= int(context.batch_wo_count_limit)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"batch W/O count exceeded: "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"batch W/O count exceeded: "
        # LINE-BY-LINE: `f"{int(context.candidate_batch_wo_count)}>{int(context.batch_wo_count_limit)}"`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        f"{int(context.candidate_batch_wo_count)}>{int(context.batch_wo_count_limit)}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_batch_length_sum_limit(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_batch_length_sum_limit(context: ConstraintContext) -> ConstraintResult:
    """batch/open-batch 후보의 길이 합 제한."""

    # LINE-BY-LINE: `rule_name`에 `"batch_length_sum_limit"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "batch_length_sum_limit"
    # LINE-BY-LINE: 조건 `context.batch_length_sum_limit is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.batch_length_sum_limit is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "batch_length_sum_limit")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "batch_length_sum_limit")
    # LINE-BY-LINE: 조건 `context.candidate_batch_length_sum is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.candidate_batch_length_sum is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "candidate_batch_length_sum")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "candidate_batch_length_sum")

    # LINE-BY-LINE: `passed`에 `float(context.candidate_batch_length_sum) <= float(context.batch_length_sum_limit)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = float(context.candidate_batch_length_sum) <= float(context.batch_length_sum_limit)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"batch length sum exceeded: "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"batch length sum exceeded: "
        # LINE-BY-LINE: `f"{float(context.candidate_batch_length_sum):.2f}>{float(context.batch_length_sum_limit):.2f}"`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        f"{float(context.candidate_batch_length_sum):.2f}>{float(context.batch_length_sum_limit):.2f}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_allowed_machine_ids(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_allowed_machine_ids(context: ConstraintContext) -> ConstraintResult:
    """W/O별 허용 설비 whitelist."""

    # LINE-BY-LINE: `rule_name`에 `"allowed_machine_ids"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "allowed_machine_ids"
    # LINE-BY-LINE: `allowed`에 `tuple(context.job.allowed_machine_ids)` 결과를 저장합니다. 의미/사용: `allowed` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    allowed = tuple(context.job.allowed_machine_ids)
    # LINE-BY-LINE: 조건 `not allowed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not allowed:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.allowed_machine_ids")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.allowed_machine_ids")

    # LINE-BY-LINE: `passed`에 `context.machine.machine_id in set(allowed)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = context.machine.machine_id in set(allowed)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"{context.machine.machine_id} not in job allowed_machine_ids"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"{context.machine.machine_id} not in job allowed_machine_ids"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_prohibited_machine_ids(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_prohibited_machine_ids(context: ConstraintContext) -> ConstraintResult:
    """W/O별 금지 설비 blacklist."""

    # LINE-BY-LINE: `rule_name`에 `"prohibited_machine_ids"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "prohibited_machine_ids"
    # LINE-BY-LINE: `prohibited`에 `tuple(context.job.prohibited_machine_ids)` 결과를 저장합니다. 의미/사용: `prohibited` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    prohibited = tuple(context.job.prohibited_machine_ids)
    # LINE-BY-LINE: 조건 `not prohibited`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not prohibited:
        # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult(rule_name, True, "")

    # LINE-BY-LINE: `passed`에 `context.machine.machine_id not in set(prohibited)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = context.machine.machine_id not in set(prohibited)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"{context.machine.machine_id} is prohibited for {context.job.job_id}"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"{context.machine.machine_id} is prohibited for {context.job.job_id}"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_allowed_bay_ids(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_allowed_bay_ids(context: ConstraintContext) -> ConstraintResult:
    """W/O별 허용 절단 Bay whitelist."""

    # LINE-BY-LINE: `rule_name`에 `"allowed_bay_ids"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "allowed_bay_ids"
    # LINE-BY-LINE: `allowed`에 `tuple(context.job.allowed_bay_ids)` 결과를 저장합니다. 의미/사용: `allowed` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    allowed = tuple(context.job.allowed_bay_ids)
    # LINE-BY-LINE: 조건 `not allowed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not allowed:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.allowed_bay_ids")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.allowed_bay_ids")
    # LINE-BY-LINE: 조건 `context.machine_bay_id is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine_bay_id is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "machine.bay_id")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "machine.bay_id")

    # LINE-BY-LINE: `passed`에 `str(context.machine_bay_id) in set(str(bay_id) for bay_id in allowed)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = str(context.machine_bay_id) in set(str(bay_id) for bay_id in allowed)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"Bay {context.machine_bay_id} not in job allowed_bay_ids"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"Bay {context.machine_bay_id} not in job allowed_bay_ids"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_table_type_eligibility(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_table_type_eligibility(context: ConstraintContext) -> ConstraintResult:
    """fixed/conveyor/rotary 등 정반 타입 가능 여부."""

    # LINE-BY-LINE: `rule_name`에 `"table_type_eligibility"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "table_type_eligibility"
    # LINE-BY-LINE: 조건 `context.table_type is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.table_type is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "machine.table_type")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "machine.table_type")
    # LINE-BY-LINE: `allowed`에 `tuple(context.allowed_table_types)` 결과를 저장합니다. 의미/사용: `allowed` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    allowed = tuple(context.allowed_table_types)
    # LINE-BY-LINE: 조건 `not allowed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not allowed:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.allowed_table_types")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.allowed_table_types")

    # LINE-BY-LINE: `passed`에 `str(context.table_type) in set(str(table_type) for table_type in allowed)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = str(context.table_type) in set(str(table_type) for table_type in allowed)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"table_type {context.table_type} not allowed"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"table_type {context.table_type} not allowed"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_plate_width_range(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_plate_width_range(context: ConstraintContext) -> ConstraintResult:
    """광폭/폭 제약 후보."""

    # LINE-BY-LINE: `rule_name`에 `"plate_width_range"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "plate_width_range"
    # LINE-BY-LINE: 조건 `context.plate_width is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.plate_width is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.plate_width")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.plate_width")

    # LINE-BY-LINE: 조건 `context.machine_min_plate_width is not None and float(context.plate_width) < float(context.machin...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine_min_plate_width is not None and float(context.plate_width) < float(context.machine_min_plate_width):
        # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult(
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `rule_name` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            rule_name,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `False` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            False,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `f"plate width {float(context.plate_width):.2f} below min {float(context.machine_min_plate_width):...` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            f"plate width {float(context.plate_width):.2f} below min {float(context.machine_min_plate_width):.2f}",
        )
    # LINE-BY-LINE: 조건 `context.machine_max_plate_width is not None and float(context.plate_width) > float(context.machin...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine_max_plate_width is not None and float(context.plate_width) > float(context.machine_max_plate_width):
        # LINE-BY-LINE: 호출자에게 `ConstraintResult(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintResult(
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `rule_name` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            rule_name,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `False` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            False,
            # LINE-BY-LINE: `ConstraintResult(...)` 호출에 `f"plate width {float(context.plate_width):.2f} exceeds max {float(context.machine_max_plate_width...` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            f"plate width {float(context.plate_width):.2f} exceeds max {float(context.machine_max_plate_width):.2f}",
        )
    # LINE-BY-LINE: 조건 `context.machine_min_plate_width is None and context.machine_max_plate_width is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine_min_plate_width is None and context.machine_max_plate_width is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "machine.min_plate_width or machine.max_plate_width")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "machine.min_plate_width or machine.max_plate_width")

    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, True, "")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, True, "")


# LINE-BY-LINE: `check_machine_priority_tier_policy(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_machine_priority_tier_policy(context: ConstraintContext) -> ConstraintResult:
    """설비 우선순위 1/2가 있으면 3/4 후보를 막는 policy."""

    # LINE-BY-LINE: `rule_name`에 `"machine_priority_tier_policy"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "machine_priority_tier_policy"
    # LINE-BY-LINE: 조건 `context.machine_priority_tier is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.machine_priority_tier is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "machine_priority_tier")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "machine_priority_tier")
    # LINE-BY-LINE: 조건 `context.has_primary_machine_tier_candidate is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.has_primary_machine_tier_candidate is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "has_primary_machine_tier_candidate")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "has_primary_machine_tier_candidate")

    # LINE-BY-LINE: `tier`에 `int(context.machine_priority_tier)` 결과를 저장합니다. 의미/사용: `tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tier = int(context.machine_priority_tier)
    # LINE-BY-LINE: `passed`에 `not (bool(context.has_primary_machine_tier_candidate) and tier > 2)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = not (bool(context.has_primary_machine_tier_candidate) and tier > 2)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"machine tier {tier} blocked while tier 1/2 candidate exists"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"machine tier {tier} blocked while tier 1/2 candidate exists"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason, metadata={"tier": tier})`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason, metadata={"tier": tier})


# LINE-BY-LINE: `check_bay_priority_tier_policy(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_bay_priority_tier_policy(context: ConstraintContext) -> ConstraintResult:
    """Bay 우선순위 1/2가 있으면 3/4 후보를 막는 policy."""

    # LINE-BY-LINE: `rule_name`에 `"bay_priority_tier_policy"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "bay_priority_tier_policy"
    # LINE-BY-LINE: 조건 `context.bay_priority_tier is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.bay_priority_tier is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "bay_priority_tier")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "bay_priority_tier")
    # LINE-BY-LINE: 조건 `context.has_primary_bay_tier_candidate is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.has_primary_bay_tier_candidate is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "has_primary_bay_tier_candidate")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "has_primary_bay_tier_candidate")

    # LINE-BY-LINE: `tier`에 `int(context.bay_priority_tier)` 결과를 저장합니다. 의미/사용: `tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tier = int(context.bay_priority_tier)
    # LINE-BY-LINE: `passed`에 `not (bool(context.has_primary_bay_tier_candidate) and tier > 2)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = not (bool(context.has_primary_bay_tier_candidate) and tier > 2)
    # LINE-BY-LINE: `reason`에 `"" if passed else f"Bay tier {tier} blocked while tier 1/2 candidate exists"` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else f"Bay tier {tier} blocked while tier 1/2 candidate exists"
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason, metadata={"tier": tier})`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason, metadata={"tier": tier})


# LINE-BY-LINE: `check_downstream_due_date(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_downstream_due_date(context: ConstraintContext) -> ConstraintResult:
    """후공정 요구 시각 이전 도착 여부."""

    # LINE-BY-LINE: `rule_name`에 `"downstream_due_date"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "downstream_due_date"
    # LINE-BY-LINE: 조건 `context.downstream_due_minutes is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.downstream_due_minutes is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.downstream_due_minutes")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.downstream_due_minutes")
    # LINE-BY-LINE: 조건 `context.candidate_downstream_arrival_time is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.candidate_downstream_arrival_time is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "candidate_downstream_arrival_time")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "candidate_downstream_arrival_time")

    # LINE-BY-LINE: `passed`에 `float(context.candidate_downstream_arrival_time) <= float(context.downstream_due_minutes)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = float(context.candidate_downstream_arrival_time) <= float(context.downstream_due_minutes)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"downstream due exceeded: arrival`에 `{float(context.candidate_downstream_arrival_time):.2f}, "` 결과를 저장합니다. 의미/사용: `f"downstream due exceeded: arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"downstream due exceeded: arrival={float(context.candidate_downstream_arrival_time):.2f}, "
        # LINE-BY-LINE: `f"due`에 `{float(context.downstream_due_minutes):.2f}"` 결과를 저장합니다. 의미/사용: `f"due` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"due={float(context.downstream_due_minutes):.2f}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)


# LINE-BY-LINE: `check_downstream_required_start(context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `ConstraintResult`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
def check_downstream_required_start(context: ConstraintContext) -> ConstraintResult:
    """후공정 착수 가능 시각 이전에 절단 완료/도착해야 하는지 확인."""

    # LINE-BY-LINE: `rule_name`에 `"downstream_required_start"` 결과를 저장합니다. 의미/사용: `rule_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rule_name = "downstream_required_start"
    # LINE-BY-LINE: 조건 `context.downstream_required_start_minutes is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.downstream_required_start_minutes is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "job.downstream_required_start_minutes")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "job.downstream_required_start_minutes")
    # LINE-BY-LINE: 조건 `context.candidate_downstream_arrival_time is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if context.candidate_downstream_arrival_time is None:
        # LINE-BY-LINE: 호출자에게 `_missing_result(rule_name, "candidate_downstream_arrival_time")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _missing_result(rule_name, "candidate_downstream_arrival_time")

    # LINE-BY-LINE: `passed`에 `float(context.candidate_downstream_arrival_time) <= float(context.downstream_required_start_minutes)` 결과를 저장합니다. 의미/사용: `passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed = float(context.candidate_downstream_arrival_time) <= float(context.downstream_required_start_minutes)
    # LINE-BY-LINE: `reason`에 `"" if passed else (` 결과를 저장합니다. 의미/사용: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason = "" if passed else (
        # LINE-BY-LINE: `f"downstream required start missed: "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"downstream required start missed: "
        # LINE-BY-LINE: `f"arrival`에 `{float(context.candidate_downstream_arrival_time):.2f}, "` 결과를 저장합니다. 의미/사용: `f"arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"arrival={float(context.candidate_downstream_arrival_time):.2f}, "
        # LINE-BY-LINE: `f"required_start`에 `{float(context.downstream_required_start_minutes):.2f}"` 결과를 저장합니다. 의미/사용: `f"required_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"required_start={float(context.downstream_required_start_minutes):.2f}"
    )
    # LINE-BY-LINE: 호출자에게 `ConstraintResult(rule_name, passed, reason)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ConstraintResult(rule_name, passed, reason)
