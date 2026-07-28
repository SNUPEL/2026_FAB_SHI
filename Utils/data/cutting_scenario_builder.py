"""절단 실적 row를 현재 환경 scenario YAML로 변환.

이 파일의 책임:
- `Utils/data/cutting_data_loader.py`가 정리한 records를 환경 입력 scenario dict로 바꾼다.
- factory config를 machine/cut_bay 목록으로 확장한다.
- 처리시간 소스를 actual_duration / tact_time / estimated_actual_duration 중 하나로 선택한다.

중요:
- scenario의 `source_machine_id`, `source_cut_bay`는 실적 비교용 원본 정보다.
- scenario의 `cut_bay`는 알고리즘이 새로 결정할 절단 Bay 자리이므로 기본 None이다.
"""

# LINE-BY-LINE: `__future__` 모듈에서 `annotations`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from __future__ import annotations

# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Optional`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Optional

# LINE-BY-LINE: `.factory_builder` 모듈에서 `build_default_np_factory_config, build_factory_scenario_parts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .factory_builder import build_default_np_factory_config, build_factory_scenario_parts
# 실적시간 추정식은 reporting 분석 모듈의 동일 구현을 재사용한다.
from Utils.reporting.tact_time import (
    CORE_TACT_FEATURE_KEYS,
    fit_actual_duration_formula,
    predict_actual_duration_from_formula,
)


# LINE-BY-LINE: `_downstream_bay_id(record: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _downstream_bay_id(record: Dict[str, Any]) -> str:
    """현재는 후공정 상세 Bay가 없으므로 source cut bay별 placeholder를 만든다."""

    # LINE-BY-LINE: `source_cut_bay`에 `str(record.get("source_cut_bay", "unknown"))` 결과를 저장합니다. 의미/사용: `source_cut_bay`는 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
    source_cut_bay = str(record.get("source_cut_bay", "unknown"))
    # LINE-BY-LINE: 호출자에게 `f"downstream_{source_cut_bay}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return f"downstream_{source_cut_bay}"


# LINE-BY-LINE: `_build_downstream_bays(records: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _build_downstream_bays(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """records에 등장하는 source cut bay별 downstream placeholder를 만든다."""

    # LINE-BY-LINE: `bay_ids`에 `sorted({_downstream_bay_id(record) for record in records})` 결과를 저장합니다. 의미/사용: `bay_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_ids = sorted({_downstream_bay_id(record) for record in records})
    # LINE-BY-LINE: 호출자에게 list 반환을 시작합니다. 사용: 여러 row 또는 후보 값을 순서 있는 목록으로 전달합니다.
    return [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `bay_id` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            "bay_id": bay_id,
            # LINE-BY-LINE: `_build_downstream_bays`에서 반환/저장할 dict의 `priority_rank` 키에 `1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "priority_rank": 1,
            # LINE-BY-LINE: `_build_downstream_bays`에서 반환/저장할 dict의 `capacity_limit` 키에 `100000` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "capacity_limit": 100000,
            # LINE-BY-LINE: `_build_downstream_bays`에서 반환/저장할 dict의 `transfer_time_minutes` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "transfer_time_minutes": 0,
            # LINE-BY-LINE: `_build_downstream_bays`에서 반환/저장할 dict의 `release_delay_minutes` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "release_delay_minutes": 0,
        }
        # LINE-BY-LINE: `bay_id in bay_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for bay_id in bay_ids
    ]


def _factory_scope_text(machines: List[Dict[str, Any]]) -> str:
    """확장된 machine 목록으로 사람이 읽을 factory scope 문자열을 만든다."""

    if not machines:
        print("[ERROR][cutting_scenario_builder._factory_scope_text] cause=no_machines")
        raise ValueError("factory scope requires at least one machine")
    machines_by_bay: Dict[str, List[str]] = {}
    for machine in machines:
        bay_id = str(machine.get("bay_id") or "").strip()
        machine_id = str(machine.get("machine_id") or "").strip()
        if not bay_id or not machine_id:
            print(
                "[ERROR][cutting_scenario_builder._factory_scope_text] "
                f"cause=missing_machine_or_bay machine={machine}"
            )
            raise ValueError("machine_id and bay_id are required for factory scope")
        machines_by_bay.setdefault(bay_id, []).append(machine_id)
    parts = []
    for bay_id in sorted(machines_by_bay, key=lambda value: (not value.isdigit(), value)):
        parts.append(f"Bay {bay_id}: {','.join(sorted(machines_by_bay[bay_id]))}")
    return "; ".join(parts)


# LINE-BY-LINE: `_job_id(record: Dict[str, Any], index: int)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _job_id(record: Dict[str, Any], index: int) -> str:
    """원본 W/O 번호를 포함한 내부 job_id를 만든다."""

    # LINE-BY-LINE: `raw`에 `str(record.get("work_order_no") or f"WO_{index}")` 결과를 저장합니다. 의미/사용: `raw` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    raw = str(record.get("work_order_no") or f"WO_{index}")
    # LINE-BY-LINE: `safe`에 `raw.replace(" ", "_").replace("/", "_")` 결과를 저장합니다. 의미/사용: `safe` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    safe = raw.replace(" ", "_").replace("/", "_")
    # LINE-BY-LINE: 호출자에게 `f"job_{index:03d}_{safe}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return f"job_{index:03d}_{safe}"


# LINE-BY-LINE: `_tact_feature_row(record: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _tact_feature_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """TACT/elapsed 산식 학습에 필요한 feature dict를 만든다."""

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `record.get("cut_length", 0.0)` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
        "cut_length": record.get("cut_length", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `mark_length`에는 `record.get("mark_length", 0.0)` 값을 넣습니다. 의미: 마킹 길이입니다. 사용: tact time 분석 feature.
        "mark_length": record.get("mark_length", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `bevel_length`에는 `record.get("bevel_length", 0.0)` 값을 넣습니다. 의미: 베벨 길이입니다. 사용: tact time 분석 feature.
        "bevel_length": record.get("bevel_length", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `record.get("length", 0.0)` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 batch 55000 길이합 제약.
        "plate_length": record.get("length", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `record.get("thickness", 0.0)` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
        "thickness": record.get("thickness", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `record.get("part_qty", 0.0)` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
        "part_count": record.get("part_qty", 0.0),
        # LINE-BY-LINE: 딕셔너리 키 `steel_qty`에는 `record.get("steel_qty", 1.0)` 값을 넣습니다. 의미: 원본 강재 수량입니다. 예: `STL_QTY`, 사용: steel_quantity로 변환.
        "steel_qty": record.get("steel_qty", 1.0),
        # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `record.get("actual_duration_minutes", 0.0)` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
        "actual_duration_minutes": record.get("actual_duration_minutes", 0.0),
    }


def _require_positive_int_record(record: Dict[str, Any], key: str, context: str) -> int:
    """scenario 필수 양의 정수 값을 fallback 없이 읽는다."""

    value = record.get(key)
    if value in (None, ""):
        print(
            "[ERROR][cutting_scenario_builder._require_positive_int_record] "
            f"cause=missing_required_int key={key} context={context}"
        )
        raise ValueError(f"missing required integer field {key} for {context}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][cutting_scenario_builder._require_positive_int_record] "
            f"cause=invalid_required_int key={key} value={value} context={context}"
        )
        raise ValueError(f"invalid required integer field {key} for {context}") from exc
    if parsed <= 0:
        print(
            "[ERROR][cutting_scenario_builder._require_positive_int_record] "
            f"cause=non_positive_required_int key={key} value={value} context={context}"
        )
        raise ValueError(f"non-positive required integer field {key} for {context}")
    return parsed


def _optional_non_negative_int_record(record: Dict[str, Any], key: str, context: str) -> Optional[int]:
    """scenario 선택 정수 값을 fallback 없이 읽는다."""

    value = record.get(key)
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][cutting_scenario_builder._optional_non_negative_int_record] "
            f"cause=invalid_optional_int key={key} value={value} context={context}"
        )
        raise ValueError(f"invalid optional integer field {key} for {context}") from exc
    if parsed < 0:
        print(
            "[ERROR][cutting_scenario_builder._optional_non_negative_int_record] "
            f"cause=negative_optional_int key={key} value={value} context={context}"
        )
        raise ValueError(f"negative optional integer field {key} for {context}")
    return parsed


# LINE-BY-LINE: `_select_process_minutes` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _select_process_minutes(
    # LINE-BY-LINE: `record`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `record` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    record: Dict[str, Any],
    # LINE-BY-LINE: `process_time_source`를 `str,` 타입으로 선언합니다. 의미/사용: `process_time_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    process_time_source: str,
    # LINE-BY-LINE: `actual_duration_formula`를 `Optional[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `actual_duration_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_duration_formula: Optional[Dict[str, Any]],
# LINE-BY-LINE: `) -> float:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> float:
    """scenario에 들어갈 `base_stage_minutes.cut` 값을 선택한다.

    선택지:
    - actual_duration: 실적 착수-종료 차이 그대로 사용
    - estimated_actual_duration: 현재 데이터로 fit한 선형식 예측값 사용
    - tact_time: 현업 제공 TACT_TIME 사용

    실패 원칙:
    - 선택된 처리시간이 0 이하이면 scenario를 만들지 않는다.
    - 다른 값으로 조용히 대체하지 않는다.
    """

    # LINE-BY-LINE: `work_order_no`에 `record.get("work_order_no", "")` 결과를 저장합니다. 의미/사용: `work_order_no`는 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
    work_order_no = record.get("work_order_no", "")
    # LINE-BY-LINE: 조건 `process_time_source == "actual_duration"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if process_time_source == "actual_duration":
        # LINE-BY-LINE: `process_minutes`에 `float(record.get("actual_duration_minutes", 0.0) or 0.0)` 결과를 저장합니다. 의미/사용: `process_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_minutes = float(record.get("actual_duration_minutes", 0.0) or 0.0)
        # LINE-BY-LINE: 조건 `process_minutes <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if process_minutes <= 0:
            # LINE-BY-LINE: `ValueError(f"actual_duration_minutes must be positive for W/O {work_order_no}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"actual_duration_minutes must be positive for W/O {work_order_no}")
        # LINE-BY-LINE: 호출자에게 `process_minutes`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return process_minutes
    # LINE-BY-LINE: 조건 `process_time_source == "estimated_actual_duration"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if process_time_source == "estimated_actual_duration":
        # LINE-BY-LINE: 조건 `actual_duration_formula is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_duration_formula is None:
            # LINE-BY-LINE: `ValueError("estimated_actual_duration requires a fitted actual_duration_formula")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("estimated_actual_duration requires a fitted actual_duration_formula")
        # LINE-BY-LINE: `estimate`에 `predict_actual_duration_from_formula(_tact_feature_row(record), actual_duration_formula)` 결과를 저장합니다. 의미/사용: `estimate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        estimate = predict_actual_duration_from_formula(_tact_feature_row(record), actual_duration_formula)
        # LINE-BY-LINE: 조건 `estimate is None or estimate <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if estimate is None or estimate <= 0:
            # LINE-BY-LINE: `ValueError(f"estimated_actual_duration must be positive for W/O {work_order_no}: {estimate}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"estimated_actual_duration must be positive for W/O {work_order_no}: {estimate}")
        # LINE-BY-LINE: 호출자에게 `float(estimate)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return float(estimate)
    # LINE-BY-LINE: 조건 `process_time_source == "tact_time"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if process_time_source == "tact_time":
        # LINE-BY-LINE: `tact_minutes`에 `float(record.get("tact_time", 0.0) or 0.0)` 결과를 저장합니다. 의미/사용: `tact_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_minutes = float(record.get("tact_time", 0.0) or 0.0)
        # LINE-BY-LINE: 조건 `tact_minutes <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if tact_minutes <= 0:
            # LINE-BY-LINE: `ValueError(f"TACT_TIME must be positive for W/O {work_order_no}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"TACT_TIME must be positive for W/O {work_order_no}")
        # LINE-BY-LINE: 호출자에게 `tact_minutes`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return tact_minutes
    # LINE-BY-LINE: `ValueError(f"unknown process_time_source: {process_time_source}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise ValueError(f"unknown process_time_source: {process_time_source}")


# LINE-BY-LINE: `build_scenario_from_cutting_records` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def build_scenario_from_cutting_records(
    # LINE-BY-LINE: `records`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    records: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `factory_config` 변수에 `None` 결과를 저장합니다. 의미: `factory_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_config: Optional[Dict[str, Any]] = None,
    # LINE-BY-LINE: `source_file` 변수에 `""` 결과를 저장합니다. 의미: `source_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    source_file: str = "",
    # LINE-BY-LINE: `process_time_source` 변수에 `"tact_time"` 결과를 저장합니다. 의미: `process_time_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    process_time_source: str = "tact_time",
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """전처리된 절단 row를 scenario dict로 변환한다.

    출력 scenario 구조:
    - metadata: 원본 파일, job 수, 처리시간 출처
    - factory: factory config 원본
    - cut_bays: 절단 Bay 레이아웃용 목록
    - machines: 실제 machine instance 목록
    - bays: downstream/적치 Bay 목록
    - jobs: 작업 목록
    """

    # LINE-BY-LINE: `records`에 `list(records)` 결과를 저장합니다. 의미/사용: `records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    records = list(records)
    # factory_config가 명시되지 않으면 현재 1차 범위인 Bay 22/23 PLS 7대를 쓴다.
    # 추후에는 config에서 항상 넣는 방향이 더 안전하다.
    # LINE-BY-LINE: `factory_config`에 `factory_config or build_default_np_factory_config()` 결과를 저장합니다. 의미/사용: `factory_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_config = factory_config or build_default_np_factory_config()
    # LINE-BY-LINE: `machines, cut_bays` 여러 변수에 `build_factory_scenario_parts(factory_config)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    machines, cut_bays = build_factory_scenario_parts(factory_config)
    # LINE-BY-LINE: `downstream_bays`에 `_build_downstream_bays(records)` 결과를 저장합니다. 의미/사용: `downstream_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_bays = _build_downstream_bays(records)
    # LINE-BY-LINE: `actual_duration_formula`에 `None` 결과를 저장합니다. 의미/사용: `actual_duration_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_duration_formula = None
    # LINE-BY-LINE: 조건 `process_time_source == "estimated_actual_duration"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if process_time_source == "estimated_actual_duration":
        # LINE-BY-LINE: `actual_duration_formula`에 `fit_actual_duration_formula(` 결과를 저장합니다. 의미/사용: `actual_duration_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration_formula = fit_actual_duration_formula(
            # LINE-BY-LINE: `fit_actual_duration_formula(...)` 호출에 `(_tact_feature_row(record) for record in records)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            (_tact_feature_row(record) for record in records),
            # LINE-BY-LINE: `feature_keys`에 `CORE_TACT_FEATURE_KEYS` 결과를 저장합니다. 의미/사용: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            feature_keys=CORE_TACT_FEATURE_KEYS,
        )

    # LINE-BY-LINE: `jobs` 변수에 `[]` 결과를 저장합니다. 의미: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    jobs: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `index, record in enumerate(records, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for index, record in enumerate(records, start=1):
        # 처리시간은 scenario 생성 시점에 확정한다.
        # 환경 코어는 Excel 파싱을 모르고 scenario의 base_stage_minutes만 읽는다.
        # LINE-BY-LINE: `process_minutes`에 `_select_process_minutes(record, process_time_source, actual_duration_formula)` 결과를 저장합니다. 의미/사용: `process_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_minutes = _select_process_minutes(record, process_time_source, actual_duration_formula)
        work_order_no = str(record.get("work_order_no") or f"index_{index}")
        steel_quantity = _require_positive_int_record(record, "steel_qty", work_order_no)
        # LINE-BY-LINE: `jobs.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        jobs.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `_job_id(record, index)` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": _job_id(record, index),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `source_wk_ord_no` 키에 `record.get("work_order_no")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "source_wk_ord_no": record.get("work_order_no"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `source_project_no` 키에 `record.get("project_no")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "source_project_no": record.get("project_no"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `source_block_no` 키에 `record.get("block_no")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "source_block_no": record.get("block_no"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `source_row_index` 키에 `record.get("source_row_index")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "source_row_index": record.get("source_row_index"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `wk_seq` 키에 `record.get("wk_seq")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "wk_seq": record.get("wk_seq"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `block_set_id` 키에 `record.get("block_set_id")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "block_set_id": record.get("block_set_id"),
                # LINE-BY-LINE: 딕셔너리 키 `family`에는 `record.get("series", "NP")` 값을 넣습니다. 의미: 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
                "family": record.get("series", "NP"),
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `float(record.get("thickness", 0.0))` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": float(record.get("thickness", 0.0)),
                # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `float(record.get("length", 0.0))` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 batch 55000 길이합 제약.
                "plate_length": float(record.get("length", 0.0)),
                # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `float(record.get("cut_length", 0.0))` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
                "cut_length": float(record.get("cut_length", 0.0)),
                # LINE-BY-LINE: 딕셔너리 키 `mark_length`에는 `float(record.get("mark_length", 0.0))` 값을 넣습니다. 의미: 마킹 길이입니다. 사용: tact time 분석 feature.
                "mark_length": float(record.get("mark_length", 0.0)),
                # LINE-BY-LINE: 딕셔너리 키 `bevel_length`에는 `float(record.get("bevel_length", 0.0))` 값을 넣습니다. 의미: 베벨 길이입니다. 사용: tact time 분석 feature.
                "bevel_length": float(record.get("bevel_length", 0.0)),
                "bevel_quantity": _optional_non_negative_int_record(record, "bevel_qty", work_order_no),
                # LINE-BY-LINE: 딕셔너리 키 `steel_quantity`에는 `steel_quantity` 값을 넣습니다. 의미: 강재 수량입니다. 예: `STL_QTY`, 사용: Phase 1 Bay 부하평준화.
                "steel_quantity": steel_quantity,
                # LINE-BY-LINE: 딕셔너리 키 `steel_qty`에는 `steel_quantity` 값을 넣습니다. 의미: 과거 분석 함수 호환을 위한 원본 alias입니다.
                "steel_qty": steel_quantity,
                # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `int(record.get("part_qty", 0))` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
                "part_count": int(record.get("part_qty", 0)),
                # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `None` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                "cut_bay": None,
                # LINE-BY-LINE: 딕셔너리 키 `downstream_bay`에는 `_downstream_bay_id(record)` 값을 넣습니다. 의미: 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                "downstream_bay": _downstream_bay_id(record),
                # LINE-BY-LINE: 딕셔너리 키 `priority_weight`에는 `1` 값을 넣습니다. 의미: 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
                "priority_weight": 1,
                # LINE-BY-LINE: 딕셔너리 키 `due_date_minutes`에는 `record.get("due_date_minutes")` 값을 넣습니다. 의미: 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
                "due_date_minutes": record.get("due_date_minutes"),
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `preferred_machine_types` 키에 `["plasma"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "preferred_machine_types": ["plasma"],
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `required_resource_ids` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "required_resource_ids": [],
                # LINE-BY-LINE: 딕셔너리 키 `base_stage_minutes`에는 `{` 값을 넣습니다. 의미: setup/cut/finish 처리시간 dict입니다. 사용: Job.estimate_total_minutes().
                "base_stage_minutes": {
                    # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `setup` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "setup": 0.0,
                    # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `cut` 키에 `process_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "cut": process_minutes,
                    # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `finish` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "finish": 0.0,
                },
                # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `tact_time_minutes` 키에 `record.get("tact_time")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time_minutes": record.get("tact_time"),
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `record.get("actual_duration_minutes")` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": record.get("actual_duration_minutes"),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_minutes`에는 `record.get("actual_start_minutes")` 값을 넣습니다. 의미: 실적 착수시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
                "actual_start_minutes": record.get("actual_start_minutes"),
                # LINE-BY-LINE: 딕셔너리 키 `actual_finish_minutes`에는 `record.get("actual_finish_minutes")` 값을 넣습니다. 의미: 실적 종료시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
                "actual_finish_minutes": record.get("actual_finish_minutes"),
                # LINE-BY-LINE: 딕셔너리 키 `planned_start_minutes`에는 `record.get("planned_start_minutes")` 값을 넣습니다. 의미: 계획 착수일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
                "planned_start_minutes": record.get("planned_start_minutes"),
                # LINE-BY-LINE: 딕셔너리 키 `planned_finish_minutes`에는 `record.get("planned_finish_minutes")` 값을 넣습니다. 의미: 계획 종료일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
                "planned_finish_minutes": record.get("planned_finish_minutes"),
                # LINE-BY-LINE: 딕셔너리 키 `planned_start_date`에는 `str(record.get("planned_start_date"))` 값을 넣습니다. 의미: 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
                "planned_start_date": str(record.get("planned_start_date")),
                # LINE-BY-LINE: 딕셔너리 키 `planned_end_date`에는 `str(record.get("planned_end_date"))` 값을 넣습니다. 의미: 계획 종료일 원본입니다. 예: `GYEL_ACT_EDDT`, 사용: 계획일 기반 검증 후보.
                "planned_end_date": str(record.get("planned_end_date")),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `str(record.get("actual_start_datetime"))` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": str(record.get("actual_start_datetime")),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `str(record.get("actual_end_datetime"))` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": str(record.get("actual_end_datetime")),
                # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `record.get("source_machine_id")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                "source_machine_id": record.get("source_machine_id"),
                # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `record.get("source_cut_bay")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                "source_cut_bay": record.get("source_cut_bay"),
            }
        )

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `metadata` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "metadata": {
            # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `source_file` 키에 `source_file` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_file": source_file,
            # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `job_count` 키에 `len(jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "job_count": len(jobs),
            "factory_scope": _factory_scope_text(machines),
            # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `process_time_source` 키에 `process_time_source` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "process_time_source": process_time_source,
            # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `process_time_formula` 키에 `None if actual_duration_formula is None else actual_duration_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "process_time_formula": None if actual_duration_formula is None else actual_duration_formula,
            # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `note` 키에 `"cut_bay is decision output; source_cut_bay keeps actual historical bay."` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "note": "cut_bay is decision output; source_cut_bay keeps actual historical bay.",
        },
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `factory` 키에 `factory_config` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "factory": factory_config,
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `cut_bays` 키에 `cut_bays` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "cut_bays": cut_bays,
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `machines` 키에 `machines` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machines": machines,
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `resources` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "resources": [],
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `bays` 키에 `downstream_bays` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bays": downstream_bays,
        # LINE-BY-LINE: `build_scenario_from_cutting_records`에서 반환/저장할 dict의 `jobs` 키에 `jobs` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "jobs": jobs,
    }
