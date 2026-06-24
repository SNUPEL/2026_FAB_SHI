"""Factory/Bay 설비 정의를 machine 목록으로 확장하는 유틸.

이 파일의 책임:
- config의 Bay별 설비 수량 정의를 실제 machine instance 목록으로 바꾼다.
- 예: `Bay 6: PLS 3, LSR 2` -> PLS 3대 + LSR 2대 machine dict 생성

입력 예:
```yaml
factory:
  bays:
    - bay_id: "6"
      equipment:
        PLS: 3
        LSR:
          count: 2
          id_pattern: "LSR{bay_id}{index}"
```

출력 예:
```text
PLS61, PLS62, PLS63, LSR61, LSR62
```

중요:
- 현업 machine id가 있으면 `machine_ids`를 우선 사용한다.
- 없으면 `id_pattern`, 그것도 없으면 `default_machine_id()`를 사용한다.
"""

# LINE-BY-LINE: `__future__` 모듈에서 `annotations`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from __future__ import annotations

# LINE-BY-LINE: `copy` 모듈에서 `deepcopy`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from copy import deepcopy
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Tuple


# LINE-BY-LINE: `DEFAULT_EQUIPMENT_PROPERTIES` 변수에 `{` 결과를 저장합니다. 의미: `DEFAULT_EQUIPMENT_PROPERTIES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
DEFAULT_EQUIPMENT_PROPERTIES: Dict[str, Dict[str, Any]] = {
    # LINE-BY-LINE: 상수/설정 key `PLS`에 `{`를 연결합니다. 사용: registry, event type, 또는 컬럼 alias lookup에서 참조합니다.
    "PLS": {
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_type` 키에 `"plasma"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_type": "plasma",
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `enabled` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "enabled": True,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `eligible_families` 키에 `["NP"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "eligible_families": ["NP"],
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `min_thickness` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "min_thickness": 0,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `max_thickness` 키에 `100` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_thickness": 100,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_length_limit` 키에 `55000` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "table_length_limit": 55000,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `cut_speed_factor` 키에 `1.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "cut_speed_factor": 1.0,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_capacity_minutes` 키에 `1440` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "daily_capacity_minutes": 1440,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `parallel_capacity` 키에 `1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "parallel_capacity": 1,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `required_resource_ids` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "required_resource_ids": [],
    },
    # LINE-BY-LINE: 상수/설정 key `LSR`에 `{`를 연결합니다. 사용: registry, event type, 또는 컬럼 alias lookup에서 참조합니다.
    "LSR": {
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_type` 키에 `"laser"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_type": "laser",
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `enabled` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "enabled": True,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `eligible_families` 키에 `["NP"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "eligible_families": ["NP"],
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `min_thickness` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "min_thickness": 0,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `max_thickness` 키에 `20` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_thickness": 20,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_length_limit` 키에 `55000` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "table_length_limit": 55000,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `cut_speed_factor` 키에 `0.8` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "cut_speed_factor": 0.8,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_capacity_minutes` 키에 `1440` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "daily_capacity_minutes": 1440,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `parallel_capacity` 키에 `1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "parallel_capacity": 1,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `required_resource_ids` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "required_resource_ids": [],
    },
    # LINE-BY-LINE: 상수/설정 key `NCG`에 `{`를 연결합니다. 사용: registry, event type, 또는 컬럼 alias lookup에서 참조합니다.
    "NCG": {
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_type` 키에 `"gas"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_type": "gas",
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `enabled` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "enabled": True,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `eligible_families` 키에 `["NP"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "eligible_families": ["NP"],
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `min_thickness` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "min_thickness": 0,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `max_thickness` 키에 `100` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_thickness": 100,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_length_limit` 키에 `55000` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "table_length_limit": 55000,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `cut_speed_factor` 키에 `1.2` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "cut_speed_factor": 1.2,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_capacity_minutes` 키에 `1440` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "daily_capacity_minutes": 1440,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `parallel_capacity` 키에 `1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "parallel_capacity": 1,
        # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `required_resource_ids` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "required_resource_ids": [],
    },
}


# LINE-BY-LINE: `default_machine_id(equipment_type: str, bay_id: str, index: int)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def default_machine_id(equipment_type: str, bay_id: str, index: int) -> str:
    """Bay와 설비 타입으로 기본 machine_id를 만든다.

    예:
    - bay 22 + PLS + 1 -> PLS21
    - bay 23 + PLS + 1 -> PLS31
    - bay 6 + LSR + 2 -> LSR62
    """

    # LINE-BY-LINE: `bay_suffix`에 `str(bay_id)[-1]` 결과를 저장합니다. 의미/사용: `bay_suffix` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_suffix = str(bay_id)[-1]
    # LINE-BY-LINE: 호출자에게 `f"{equipment_type}{bay_suffix}{index}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return f"{equipment_type}{bay_suffix}{index}"


# LINE-BY-LINE: `_normalize_equipment_spec(raw_spec: Any)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _normalize_equipment_spec(raw_spec: Any) -> Dict[str, Any]:
    """축약형/상세형 설비 spec을 dict로 통일한다."""

    # LINE-BY-LINE: 조건 `isinstance(raw_spec, int)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(raw_spec, int):
        # LINE-BY-LINE: 호출자에게 `{"count": raw_spec}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return {"count": raw_spec}
    # LINE-BY-LINE: 조건 `isinstance(raw_spec, list)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(raw_spec, list):
        # LINE-BY-LINE: 호출자에게 `{"machine_ids": list(raw_spec), "count": len(raw_spec)}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return {"machine_ids": list(raw_spec), "count": len(raw_spec)}
    # LINE-BY-LINE: 조건 `isinstance(raw_spec, dict)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(raw_spec, dict):
        # LINE-BY-LINE: `spec`에 `dict(raw_spec)` 결과를 저장합니다. 의미/사용: `spec` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        spec = dict(raw_spec)
        # LINE-BY-LINE: 조건 `"machine_ids" in spec and "count" not in spec`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if "machine_ids" in spec and "count" not in spec:
            # LINE-BY-LINE: `spec["count"]`에 `len(spec["machine_ids"])` 결과를 저장합니다. 의미/사용: `spec["count"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            spec["count"] = len(spec["machine_ids"])
        # LINE-BY-LINE: 호출자에게 `spec`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return spec
    # LINE-BY-LINE: `TypeError(f"unsupported equipment spec: {raw_spec!r}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise TypeError(f"unsupported equipment spec: {raw_spec!r}")


# LINE-BY-LINE: `_machine_ids_for(equipment_type: str, bay_id: str, spec: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[str]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _machine_ids_for(equipment_type: str, bay_id: str, spec: Dict[str, Any]) -> List[str]:
    """설비 spec에서 machine_id 목록을 만든다.

    우선순위:
    1. spec.machine_ids: 현업 id를 그대로 사용
    2. spec.id_pattern: 포맷 문자열로 자동 생성
    3. default_machine_id(): 기본 규칙으로 자동 생성
    """

    # LINE-BY-LINE: 조건 `spec.get("machine_ids")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if spec.get("machine_ids"):
        # LINE-BY-LINE: 호출자에게 `[str(machine_id) for machine_id in spec["machine_ids"]]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return [str(machine_id) for machine_id in spec["machine_ids"]]

    # LINE-BY-LINE: `count`에 `int(spec.get("count", 0))` 결과를 저장합니다. 의미/사용: `count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    count = int(spec.get("count", 0))
    # LINE-BY-LINE: 조건 `count <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if count <= 0:
        # LINE-BY-LINE: 호출자에게 `[]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return []

    # LINE-BY-LINE: `id_pattern`에 `spec.get("id_pattern")` 결과를 저장합니다. 의미/사용: `id_pattern` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    id_pattern = spec.get("id_pattern")
    # LINE-BY-LINE: `start_index`에 `int(spec.get("start_index", 1))` 결과를 저장합니다. 의미/사용: `start_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_index = int(spec.get("start_index", 1))
    # LINE-BY-LINE: `machine_ids`에 `[]` 결과를 저장합니다. 의미/사용: `machine_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_ids = []
    # LINE-BY-LINE: `offset in range(count)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for offset in range(count):
        # LINE-BY-LINE: `index`에 `start_index + offset` 결과를 저장합니다. 의미/사용: `index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        index = start_index + offset
        # LINE-BY-LINE: 조건 `id_pattern`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if id_pattern:
            # LINE-BY-LINE: `machine_ids.append(str(id_pattern).format(index`에 `index, bay_id=bay_id, equipment_type=equipment_type))` 결과를 저장합니다. 의미/사용: `format(index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_ids.append(str(id_pattern).format(index=index, bay_id=bay_id, equipment_type=equipment_type))
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `machine_ids.append(default_machine_id(equipment_type, bay_id, index))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            machine_ids.append(default_machine_id(equipment_type, bay_id, index))
    # LINE-BY-LINE: 호출자에게 `machine_ids`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return machine_ids


# LINE-BY-LINE: `_machine_template(equipment_type: str, spec: Dict[str, Any], defaults: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _machine_template(equipment_type: str, spec: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """설비 타입 기본값과 spec override를 합친다."""

    # LINE-BY-LINE: `equipment_defaults`에 `deepcopy(DEFAULT_EQUIPMENT_PROPERTIES.get(equipment_type, {}))` 결과를 저장합니다. 의미/사용: `equipment_defaults` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    equipment_defaults = deepcopy(DEFAULT_EQUIPMENT_PROPERTIES.get(equipment_type, {}))
    # LINE-BY-LINE: `equipment_defaults.update(deepcopy(defaults.get(equipment_type, {})))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    equipment_defaults.update(deepcopy(defaults.get(equipment_type, {})))

    # LINE-BY-LINE: `ignored_keys`에 `{"count", "machine_ids", "id_pattern", "start_index"}` 결과를 저장합니다. 의미/사용: `ignored_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    ignored_keys = {"count", "machine_ids", "id_pattern", "start_index"}
    # LINE-BY-LINE: `overrides`에 `{key: value for key, value in spec.items() if key not in ignored_keys}` 결과를 저장합니다. 의미/사용: `overrides` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    overrides = {key: value for key, value in spec.items() if key not in ignored_keys}
    # LINE-BY-LINE: `equipment_defaults.update(overrides)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    equipment_defaults.update(overrides)
    # LINE-BY-LINE: 호출자에게 `equipment_defaults`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return equipment_defaults


# LINE-BY-LINE: `expand_bay_equipment(factory_config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def expand_bay_equipment(factory_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bay별 설비 수량 정의를 machine dict 목록으로 확장한다.

    입력:
    - config["factory"]

    출력:
    - scenario["machines"]에 들어갈 machine dict list

    동작:
    - Bay 하나씩 순회
    - Bay 안 equipment 타입별 수량/spec 해석
    - machine_id, bay_id, equipment_type, position 등을 채워 machine 생성
    """

    # LINE-BY-LINE: `machines` 변수에 `[]` 결과를 저장합니다. 의미: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machines: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `defaults`에 `factory_config.get("equipment_defaults", {}) or {}` 결과를 저장합니다. 의미/사용: `defaults` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    defaults = factory_config.get("equipment_defaults", {}) or {}

    # LINE-BY-LINE: `bay in factory_config.get("bays", []) or []` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bay in factory_config.get("bays", []) or []:
        # LINE-BY-LINE: `bay_id`에 `str(bay["bay_id"])` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(bay["bay_id"])
        # LINE-BY-LINE: `equipment_map`에 `bay.get("equipment", {}) or {}` 결과를 저장합니다. 의미/사용: `equipment_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        equipment_map = bay.get("equipment", {}) or {}
        # LINE-BY-LINE: `base_x`에 `float(bay.get("position", [0, 0])[0]) if isinstance(bay.get("position"), list) else 0.0` 결과를 저장합니다. 의미/사용: `base_x` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_x = float(bay.get("position", [0, 0])[0]) if isinstance(bay.get("position"), list) else 0.0
        # LINE-BY-LINE: `base_y`에 `float(bay.get("position", [0, 0])[1]) if isinstance(bay.get("position"), list) else float(bay_id)...` 결과를 저장합니다. 의미/사용: `base_y` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_y = float(bay.get("position", [0, 0])[1]) if isinstance(bay.get("position"), list) else float(bay_id) if bay_id.isdigit() else 0.0

        # LINE-BY-LINE: `machine_offset`에 `0` 결과를 저장합니다. 의미/사용: `machine_offset` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_offset = 0
        # 같은 Bay 안에 PLS, LSR, NCG가 섞여 있어도 타입별로 따로 확장한다.
        # LINE-BY-LINE: `equipment_type_raw, raw_spec in equipment_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for equipment_type_raw, raw_spec in equipment_map.items():
            # LINE-BY-LINE: `equipment_type`에 `str(equipment_type_raw).upper()` 결과를 저장합니다. 의미/사용: `equipment_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            equipment_type = str(equipment_type_raw).upper()
            # LINE-BY-LINE: `spec`에 `_normalize_equipment_spec(raw_spec)` 결과를 저장합니다. 의미/사용: `spec` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            spec = _normalize_equipment_spec(raw_spec)
            # LINE-BY-LINE: `template`에 `_machine_template(equipment_type, spec, defaults)` 결과를 저장합니다. 의미/사용: `template` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            template = _machine_template(equipment_type, spec, defaults)
            # LINE-BY-LINE: `machine_ids`에 `_machine_ids_for(equipment_type, bay_id, spec)` 결과를 저장합니다. 의미/사용: `machine_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_ids = _machine_ids_for(equipment_type, bay_id, spec)

            # machine_id 하나가 실제 설비 instance 1대다.
            # LINE-BY-LINE: `machine_id in machine_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for machine_id in machine_ids:
                # LINE-BY-LINE: `machine`에 `deepcopy(template)` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
                machine = deepcopy(template)
                # LINE-BY-LINE: `machine["machine_id"]`에 `machine_id` 결과를 저장합니다. 의미/사용: `machine["machine_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine["machine_id"] = machine_id
                # LINE-BY-LINE: `machine["bay_id"]`에 `bay_id` 결과를 저장합니다. 의미/사용: `machine["bay_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine["bay_id"] = bay_id
                # LINE-BY-LINE: `machine.setdefault("machine_type", equipment_type.lower())`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                machine.setdefault("machine_type", equipment_type.lower())
                # LINE-BY-LINE: `machine.setdefault("enabled", True)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                machine.setdefault("enabled", True)
                # LINE-BY-LINE: `machine.setdefault("eligible_families", ["NP"])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                machine.setdefault("eligible_families", ["NP"])
                # LINE-BY-LINE: `machine.setdefault("position", [base_x + machine_offset, base_y])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                machine.setdefault("position", [base_x + machine_offset, base_y])
                # LINE-BY-LINE: `machine["equipment_type"]`에 `equipment_type` 결과를 저장합니다. 의미/사용: `machine["equipment_type"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine["equipment_type"] = equipment_type
                # LINE-BY-LINE: `machines.append(machine)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                machines.append(machine)
                # LINE-BY-LINE: `machine_offset` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `machine_offset` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_offset += 1

    # LINE-BY-LINE: 호출자에게 `machines`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return machines


# LINE-BY-LINE: `build_machine_to_bay(machines: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `Dict[str, str]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def build_machine_to_bay(machines: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """machine_id -> bay_id 매핑을 만든다."""

    # LINE-BY-LINE: `mapping` 변수에 `{}` 결과를 저장합니다. 의미: `mapping` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mapping: Dict[str, str] = {}
    # LINE-BY-LINE: `machine in machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for machine in machines:
        # LINE-BY-LINE: `machine_id`에 `str(machine["machine_id"])` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id = str(machine["machine_id"])
        # LINE-BY-LINE: `bay_id`에 `machine.get("bay_id")` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = machine.get("bay_id")
        # LINE-BY-LINE: 조건 `bay_id is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if bay_id is not None:
            # LINE-BY-LINE: `mapping[machine_id]`에 `str(bay_id)` 결과를 저장합니다. 의미/사용: `mapping[machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            mapping[machine_id] = str(bay_id)
    # LINE-BY-LINE: 호출자에게 `mapping`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return mapping


# LINE-BY-LINE: `build_cut_bays_from_factory(factory_config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def build_cut_bays_from_factory(factory_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """factory config에서 시각화용 절단 Bay 목록을 만든다."""

    # LINE-BY-LINE: `cut_bays`에 `[]` 결과를 저장합니다. 의미/사용: `cut_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cut_bays = []
    # LINE-BY-LINE: `bay in factory_config.get("bays", []) or []` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bay in factory_config.get("bays", []) or []:
        # LINE-BY-LINE: `cut_bays.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        cut_bays.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `str(bay["bay_id"])` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": str(bay["bay_id"]),
                # LINE-BY-LINE: `build_cut_bays_from_factory`에서 반환/저장할 dict의 `name` 키에 `bay.get("name", f"Bay {bay['bay_id']}")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "name": bay.get("name", f"Bay {bay['bay_id']}"),
                # LINE-BY-LINE: `build_cut_bays_from_factory`에서 반환/저장할 dict의 `position` 키에 `bay.get("position", [0, 0])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "position": bay.get("position", [0, 0]),
                # LINE-BY-LINE: `build_cut_bays_from_factory`에서 반환/저장할 dict의 `equipment` 키에 `bay.get("equipment", {})` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "equipment": bay.get("equipment", {}),
            }
        )
    # LINE-BY-LINE: 호출자에게 `cut_bays`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return cut_bays


# LINE-BY-LINE: `build_default_np_factory_config()` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def build_default_np_factory_config() -> Dict[str, Any]:
    """현재 NP 1차 범위인 Bay 22/23 + PLS 7대 factory config."""

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `bays` 키에 `[` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bays": [
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `"22"` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": "22",
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `name` 키에 `"Bay 22"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "name": "Bay 22",
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `position` 키에 `[0, 0]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "position": [0, 0],
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `equipment` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "equipment": {
                    # LINE-BY-LINE: 상수/설정 key `PLS`에 `{`를 연결합니다. 사용: registry, event type, 또는 컬럼 alias lookup에서 참조합니다.
                    "PLS": {
                        # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `count` 키에 `4` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "count": 4,
                        # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `machine_ids` 키에 `["PLS21", "PLS22", "PLS23", "PLS24"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "machine_ids": ["PLS21", "PLS22", "PLS23", "PLS24"],
                    }
                },
            },
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `"23"` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": "23",
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `name` 키에 `"Bay 23"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "name": "Bay 23",
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `position` 키에 `[0, 1]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "position": [0, 1],
                # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `equipment` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "equipment": {
                    # LINE-BY-LINE: 상수/설정 key `PLS`에 `{`를 연결합니다. 사용: registry, event type, 또는 컬럼 alias lookup에서 참조합니다.
                    "PLS": {
                        # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `count` 키에 `3` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "count": 3,
                        # LINE-BY-LINE: `build_default_np_factory_config`에서 반환/저장할 dict의 `machine_ids` 키에 `["PLS31", "PLS32", "PLS33"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "machine_ids": ["PLS31", "PLS32", "PLS33"],
                    }
                },
            },
        ]
    }


# LINE-BY-LINE: `build_factory_scenario_parts(factory_config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def build_factory_scenario_parts(factory_config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """scenario에 넣을 machines/cut_bays를 함께 만든다."""

    # LINE-BY-LINE: `machines`에 `expand_bay_equipment(factory_config)` 결과를 저장합니다. 의미/사용: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machines = expand_bay_equipment(factory_config)
    # LINE-BY-LINE: `cut_bays`에 `build_cut_bays_from_factory(factory_config)` 결과를 저장합니다. 의미/사용: `cut_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cut_bays = build_cut_bays_from_factory(factory_config)
    # LINE-BY-LINE: 호출자에게 `machines, cut_bays`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return machines, cut_bays
