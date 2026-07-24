"""Factory event schema for the cutting DES core.

이 파일은 generated simulation과 actual replay가 공통으로 쓰는 event 형식을 정의한다.

왜 event log가 필요한가:
- 단순 schedule table은 "결과"만 보여준다.
- DES 검증과 playback은 "언제 어떤 상태 변화가 발생했는지"가 필요하다.
- 따라서 모든 generated/actual 흐름은 가능하면 같은 `FactoryEvent` schema로 변환한다.

대표 event 흐름:
```text
DECISION_EPOCH
MACHINE_ASSIGN
CUT_BAY_ASSIGN
PROCESS_START
PROCESS_FINISH
DOWNSTREAM_ARRIVAL
DOWNSTREAM_RELEASE
```

실패 원칙:
- event_type, source, required field가 틀리면 조용히 통과하지 않는다.
- `validate_event_required_fields()`에서 원인을 print하고 예외를 발생시킨다.
"""

# LINE-BY-LINE: `__future__` 모듈에서 `annotations`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from __future__ import annotations

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass, field`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass, field
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Optional`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Optional


# LINE-BY-LINE: `EVENT_DECISION_EPOCH`에 `"DECISION_EPOCH"` 결과를 저장합니다. 의미/사용: `EVENT_DECISION_EPOCH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_DECISION_EPOCH = "DECISION_EPOCH"
# LINE-BY-LINE: `EVENT_JOB_RELEASE`에 `"JOB_RELEASE"` 결과를 저장합니다. 의미/사용: `EVENT_JOB_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_JOB_RELEASE = "JOB_RELEASE"
# LINE-BY-LINE: `EVENT_MACHINE_ASSIGN`에 `"MACHINE_ASSIGN"` 결과를 저장합니다. 의미/사용: `EVENT_MACHINE_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_MACHINE_ASSIGN = "MACHINE_ASSIGN"
# LINE-BY-LINE: `EVENT_CUT_BAY_ASSIGN`에 `"CUT_BAY_ASSIGN"` 결과를 저장합니다. 의미/사용: `EVENT_CUT_BAY_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_CUT_BAY_ASSIGN = "CUT_BAY_ASSIGN"
# LINE-BY-LINE: `EVENT_PROCESS_START`에 `"PROCESS_START"` 결과를 저장합니다. 의미/사용: `EVENT_PROCESS_START` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_PROCESS_START = "PROCESS_START"
# LINE-BY-LINE: `EVENT_PROCESS_FINISH`에 `"PROCESS_FINISH"` 결과를 저장합니다. 의미/사용: `EVENT_PROCESS_FINISH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_PROCESS_FINISH = "PROCESS_FINISH"
# LINE-BY-LINE: `EVENT_DOWNSTREAM_ARRIVAL`에 `"DOWNSTREAM_ARRIVAL"` 결과를 저장합니다. 의미/사용: `EVENT_DOWNSTREAM_ARRIVAL` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_DOWNSTREAM_ARRIVAL = "DOWNSTREAM_ARRIVAL"
# LINE-BY-LINE: `EVENT_DOWNSTREAM_RELEASE`에 `"DOWNSTREAM_RELEASE"` 결과를 저장합니다. 의미/사용: `EVENT_DOWNSTREAM_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_DOWNSTREAM_RELEASE = "DOWNSTREAM_RELEASE"
# LINE-BY-LINE: `EVENT_RESOURCE_ACQUIRE`에 `"RESOURCE_ACQUIRE"` 결과를 저장합니다. 의미/사용: `EVENT_RESOURCE_ACQUIRE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_RESOURCE_ACQUIRE = "RESOURCE_ACQUIRE"
# LINE-BY-LINE: `EVENT_RESOURCE_RELEASE`에 `"RESOURCE_RELEASE"` 결과를 저장합니다. 의미/사용: `EVENT_RESOURCE_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_RESOURCE_RELEASE = "RESOURCE_RELEASE"
# LINE-BY-LINE: `EVENT_VALIDATION_FAILURE`에 `"VALIDATION_FAILURE"` 결과를 저장합니다. 의미/사용: `EVENT_VALIDATION_FAILURE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_VALIDATION_FAILURE = "VALIDATION_FAILURE"

# LINE-BY-LINE: `SOURCE_GENERATED`에 `"generated"` 결과를 저장합니다. 의미/사용: `SOURCE_GENERATED` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
SOURCE_GENERATED = "generated"
# LINE-BY-LINE: `SOURCE_ACTUAL`에 `"actual"` 결과를 저장합니다. 의미/사용: `SOURCE_ACTUAL` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
SOURCE_ACTUAL = "actual"
# LINE-BY-LINE: `SOURCE_DERIVED`에 `"derived"` 결과를 저장합니다. 의미/사용: `SOURCE_DERIVED` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
SOURCE_DERIVED = "derived"
# LINE-BY-LINE: `SOURCE_VALIDATION`에 `"validation"` 결과를 저장합니다. 의미/사용: `SOURCE_VALIDATION` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
SOURCE_VALIDATION = "validation"

# LINE-BY-LINE: `EVENT_TYPES`에 `{` 결과를 저장합니다. 의미/사용: `EVENT_TYPES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
EVENT_TYPES = {
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_DECISION_EPOCH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DECISION_EPOCH,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_JOB_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_JOB_RELEASE,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_MACHINE_ASSIGN,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_CUT_BAY_ASSIGN,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_START,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_FINISH,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_DOWNSTREAM_ARRIVAL` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DOWNSTREAM_ARRIVAL,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_DOWNSTREAM_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DOWNSTREAM_RELEASE,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_RESOURCE_ACQUIRE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_RESOURCE_ACQUIRE,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_RESOURCE_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_RESOURCE_RELEASE,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_VALIDATION_FAILURE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_VALIDATION_FAILURE,
}

# LINE-BY-LINE: `SOURCE_TYPES`에 `{` 결과를 저장합니다. 의미/사용: `SOURCE_TYPES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
SOURCE_TYPES = {
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `SOURCE_GENERATED` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_GENERATED,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `SOURCE_ACTUAL` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_ACTUAL,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `SOURCE_DERIVED` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_DERIVED,
    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `SOURCE_VALIDATION` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_VALIDATION,
}

# LINE-BY-LINE: `REQUIRED_FIELDS_BY_EVENT_TYPE`에 `{` 결과를 저장합니다. 의미/사용: `REQUIRED_FIELDS_BY_EVENT_TYPE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
REQUIRED_FIELDS_BY_EVENT_TYPE = {
    # LINE-BY-LINE: `EVENT_DECISION_EPOCH`를 `(),` 타입으로 선언합니다. 의미/사용: `EVENT_DECISION_EPOCH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_DECISION_EPOCH: (),
    # LINE-BY-LINE: `EVENT_JOB_RELEASE`를 `("job_id",),` 타입으로 선언합니다. 의미/사용: `EVENT_JOB_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_JOB_RELEASE: ("job_id",),
    # LINE-BY-LINE: `EVENT_MACHINE_ASSIGN`를 `("job_id", "machine_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_MACHINE_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_MACHINE_ASSIGN: ("job_id", "machine_id"),
    # LINE-BY-LINE: `EVENT_CUT_BAY_ASSIGN`를 `("job_id", "bay_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_CUT_BAY_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_CUT_BAY_ASSIGN: ("job_id", "bay_id"),
    # LINE-BY-LINE: `EVENT_PROCESS_START`를 `("job_id", "machine_id", "bay_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_START` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_PROCESS_START: ("job_id", "machine_id", "bay_id"),
    # LINE-BY-LINE: `EVENT_PROCESS_FINISH`를 `("job_id", "machine_id", "bay_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_FINISH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_PROCESS_FINISH: ("job_id", "machine_id", "bay_id"),
    # LINE-BY-LINE: `EVENT_DOWNSTREAM_ARRIVAL`를 `("job_id", "bay_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_DOWNSTREAM_ARRIVAL` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_DOWNSTREAM_ARRIVAL: ("job_id", "bay_id"),
    # LINE-BY-LINE: `EVENT_DOWNSTREAM_RELEASE`를 `("job_id", "bay_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_DOWNSTREAM_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_DOWNSTREAM_RELEASE: ("job_id", "bay_id"),
    # LINE-BY-LINE: `EVENT_RESOURCE_ACQUIRE`를 `("job_id", "resource_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_RESOURCE_ACQUIRE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_RESOURCE_ACQUIRE: ("job_id", "resource_id"),
    # LINE-BY-LINE: `EVENT_RESOURCE_RELEASE`를 `("job_id", "resource_id"),` 타입으로 선언합니다. 의미/사용: `EVENT_RESOURCE_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_RESOURCE_RELEASE: ("job_id", "resource_id"),
    # LINE-BY-LINE: `EVENT_VALIDATION_FAILURE`를 `("message",),` 타입으로 선언합니다. 의미/사용: `EVENT_VALIDATION_FAILURE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    EVENT_VALIDATION_FAILURE: ("message",),
}


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `FactoryEvent` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class FactoryEvent:
    """generated simulation 또는 actual replay에서 발생한 공장 event 1건.

    필드 의미:
    - event_id: event log 안에서 유일한 id
    - event_type: PROCESS_START 같은 상태 변화 종류
    - time_min: simulation 기준 분 단위 시각
    - job_id/machine_id/bay_id/resource_id: event가 영향을 주는 대상
    - operation_id: schedule row와 event를 연결하는 id
    - machine_slot_index: batch/single operation이 점유한 설비 slot index
    - source: generated / actual / derived / validation 중 하나
    - payload: event type별 추가 정보
    - message: validation failure처럼 사람이 읽어야 하는 설명
    """

    # LINE-BY-LINE: `event_id`를 `str` 타입으로 선언합니다. 의미/사용: `FactoryEvent.event_id`는 event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
    event_id: str
    # LINE-BY-LINE: `event_type`를 `str` 타입으로 선언합니다. 의미/사용: `FactoryEvent.event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
    event_type: str
    # LINE-BY-LINE: `time_min`를 `float` 타입으로 선언합니다. 의미/사용: `FactoryEvent.time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
    time_min: float
    # LINE-BY-LINE: `job_id` 변수에 `None` 결과를 저장합니다. 의미: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
    job_id: Optional[str] = None
    # LINE-BY-LINE: `machine_id` 변수에 `None` 결과를 저장합니다. 의미: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
    machine_id: Optional[str] = None
    # LINE-BY-LINE: `bay_id` 변수에 `None` 결과를 저장합니다. 의미: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: Optional[str] = None
    # LINE-BY-LINE: `resource_id` 변수에 `None` 결과를 저장합니다. 의미: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    resource_id: Optional[str] = None
    # LINE-BY-LINE: `operation_id` 변수에 `None` 결과를 저장합니다. 의미: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    operation_id: Optional[str] = None
    # LINE-BY-LINE: `machine_slot_index` 변수에 `None` 결과를 저장합니다. 의미: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_slot_index: Optional[int] = None
    # LINE-BY-LINE: `source` 변수에 `SOURCE_GENERATED` 결과를 저장합니다. 의미: `source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    source: str = SOURCE_GENERATED
    # LINE-BY-LINE: `payload` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
    payload: Dict[str, Any] = field(default_factory=dict)
    # LINE-BY-LINE: `message` 변수에 `""` 결과를 저장합니다. 의미: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
    message: str = ""

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `time_sec(self)` 함수를 정의합니다. 반환 타입: `int`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
    def time_sec(self) -> int:
        """playback HTML에서 쓰기 쉬운 초 단위 정수 시각."""

        # LINE-BY-LINE: 호출자에게 `int(round(float(self.time_min) * 60))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(round(float(self.time_min) * 60))

    # LINE-BY-LINE: `to_dict(self)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
    def to_dict(self) -> Dict[str, Any]:
        """JSON/CSV 저장용 dict로 변환한다.

        저장 직전에 schema validation을 수행한다.
        event가 잘못됐으면 저장 파일을 만들기 전에 실패해야 원인을 찾기 쉽다.
        """

        # LINE-BY-LINE: `validate_event_required_fields(self)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        validate_event_required_fields(self)
        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: 딕셔너리 키 `event_id`에는 `self.event_id` 값을 넣습니다. 의미: event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
            "event_id": self.event_id,
            # LINE-BY-LINE: 딕셔너리 키 `event_type`에는 `self.event_type` 값을 넣습니다. 의미: event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
            "event_type": self.event_type,
            # LINE-BY-LINE: 딕셔너리 키 `time_min`에는 `round(float(self.time_min), 6)` 값을 넣습니다. 의미: simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
            "time_min": round(float(self.time_min), 6),
            # LINE-BY-LINE: `to_dict`에서 반환/저장할 dict의 `time_sec` 키에 `self.time_sec` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "time_sec": self.time_sec,
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `self.job_id or ""` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": self.job_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `self.machine_id or ""` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": self.machine_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `self.bay_id or ""` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            "bay_id": self.bay_id or "",
            # LINE-BY-LINE: `to_dict`에서 반환/저장할 dict의 `resource_id` 키에 `self.resource_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "resource_id": self.resource_id or "",
            # LINE-BY-LINE: `to_dict`에서 반환/저장할 dict의 `operation_id` 키에 `self.operation_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "operation_id": self.operation_id or "",
            # LINE-BY-LINE: `to_dict`에서 반환/저장할 dict의 `machine_slot_index` 키에 `"" if self.machine_slot_index is None else int(self.machine_slot_index)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_slot_index": "" if self.machine_slot_index is None else int(self.machine_slot_index),
            # LINE-BY-LINE: `to_dict`에서 반환/저장할 dict의 `source` 키에 `self.source` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source": self.source,
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `self.message` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": self.message,
            # LINE-BY-LINE: 딕셔너리 키 `payload`에는 `self.payload` 값을 넣습니다. 의미: event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            "payload": self.payload,
        }


# LINE-BY-LINE: `validate_event_required_fields(event: FactoryEvent)` 함수를 정의합니다. 반환 타입: `None`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def validate_event_required_fields(event: FactoryEvent) -> None:
    """Validate event schema and required fields.

    This intentionally raises on invalid events. Event creation must not hide
    schema problems because downstream replay/validation depends on this log.
    """

    # LINE-BY-LINE: 조건 `event.event_type not in EVENT_TYPES`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if event.event_type not in EVENT_TYPES:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"[ERROR][events.validate_event_required_fields] "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"[ERROR][events.validate_event_required_fields] "
            # LINE-BY-LINE: `f"cause`에 `unknown_event_type key={event.event_id} event_type={event.event_type}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=unknown_event_type key={event.event_id} event_type={event.event_type}"
        )
        # LINE-BY-LINE: `ValueError(f"unknown event_type: {event.event_type}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"unknown event_type: {event.event_type}")
    # LINE-BY-LINE: 조건 `event.source not in SOURCE_TYPES`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if event.source not in SOURCE_TYPES:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"[ERROR][events.validate_event_required_fields] "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"[ERROR][events.validate_event_required_fields] "
            # LINE-BY-LINE: `f"cause`에 `unknown_source key={event.event_id} source={event.source}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=unknown_source key={event.event_id} source={event.source}"
        )
        # LINE-BY-LINE: `ValueError(f"unknown event source: {event.source}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"unknown event source: {event.source}")
    # LINE-BY-LINE: 조건 `event.time_min is None or float(event.time_min) < 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if event.time_min is None or float(event.time_min) < 0:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"[ERROR][events.validate_event_required_fields] "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"[ERROR][events.validate_event_required_fields] "
            # LINE-BY-LINE: `f"cause`에 `invalid_time key={event.event_id} time_min={event.time_min}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=invalid_time key={event.event_id} time_min={event.time_min}"
        )
        # LINE-BY-LINE: `ValueError(f"invalid event time_min: {event.time_min}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"invalid event time_min: {event.time_min}")
    # LINE-BY-LINE: 조건 `not isinstance(event.payload, dict)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not isinstance(event.payload, dict):
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"[ERROR][events.validate_event_required_fields] "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"[ERROR][events.validate_event_required_fields] "
            # LINE-BY-LINE: `f"cause`에 `payload_not_dict key={event.event_id} payload_type={type(event.payload).__name__}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=payload_not_dict key={event.event_id} payload_type={type(event.payload).__name__}"
        )
        # LINE-BY-LINE: `TypeError("event payload must be a dict")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise TypeError("event payload must be a dict")

    # LINE-BY-LINE: `field_name in REQUIRED_FIELDS_BY_EVENT_TYPE[event.event_type]` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for field_name in REQUIRED_FIELDS_BY_EVENT_TYPE[event.event_type]:
        # LINE-BY-LINE: `value`에 `getattr(event, field_name)` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value = getattr(event, field_name)
        # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if value in (None, ""):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: `f"[ERROR][events.validate_event_required_fields] "` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                f"[ERROR][events.validate_event_required_fields] "
                # LINE-BY-LINE: `f"cause`에 `missing_required_field key={event.event_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=missing_required_field key={event.event_id} "
                # LINE-BY-LINE: `f"event_type`에 `{event.event_type} field={field_name}"` 결과를 저장합니다. 의미/사용: `f"event_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"event_type={event.event_type} field={field_name}"
            )
            # LINE-BY-LINE: `ValueError(f"{event.event_type} requires {field_name}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"{event.event_type} requires {field_name}")


# LINE-BY-LINE: `event_sort_key(event: FactoryEvent | Dict[str, Any])` 함수를 정의합니다. 반환 타입: `tuple`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def event_sort_key(event: FactoryEvent | Dict[str, Any]) -> tuple:
    """playback/export용 안정 정렬 key.

    같은 시각에 여러 event가 있으면 아래 순서가 중요하다.
    예를 들어 PROCESS_FINISH보다 PROCESS_START를 먼저 보여주면
    동일 timestamp의 active count 계산이 달라질 수 있다.
    """

    # LINE-BY-LINE: 조건 `isinstance(event, FactoryEvent)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(event, FactoryEvent):
        # LINE-BY-LINE: 호출자에게 `(float(event.time_min), _event_type_order(event.event_type), event.event_id)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return (float(event.time_min), _event_type_order(event.event_type), event.event_id)
    # LINE-BY-LINE: 호출자에게 `(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return (
        # LINE-BY-LINE: `return(...)` 호출에 `float(event.get("time_min", 0.0) or 0.0)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        float(event.get("time_min", 0.0) or 0.0),
        # LINE-BY-LINE: `return(...)` 호출에 `_event_type_order(str(event.get("event_type", "")))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        _event_type_order(str(event.get("event_type", ""))),
        # LINE-BY-LINE: `return(...)` 호출에 `str(event.get("event_id", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        str(event.get("event_id", "")),
    )


# LINE-BY-LINE: `_event_type_order(event_type: str)` 함수를 정의합니다. 반환 타입: `int`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def _event_type_order(event_type: str) -> int:
    """같은 timestamp 안에서 event가 표시될 순서를 숫자로 반환한다."""

    # LINE-BY-LINE: `order`에 `{` 결과를 저장합니다. 의미/사용: `order` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    order = {
        # LINE-BY-LINE: `EVENT_JOB_RELEASE`를 `0,` 타입으로 선언합니다. 의미/사용: `EVENT_JOB_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_JOB_RELEASE: 0,
        # LINE-BY-LINE: `EVENT_DECISION_EPOCH`를 `1,` 타입으로 선언합니다. 의미/사용: `EVENT_DECISION_EPOCH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_DECISION_EPOCH: 1,
        # LINE-BY-LINE: `EVENT_MACHINE_ASSIGN`를 `2,` 타입으로 선언합니다. 의미/사용: `EVENT_MACHINE_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_MACHINE_ASSIGN: 2,
        # LINE-BY-LINE: `EVENT_CUT_BAY_ASSIGN`를 `3,` 타입으로 선언합니다. 의미/사용: `EVENT_CUT_BAY_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_CUT_BAY_ASSIGN: 3,
        # LINE-BY-LINE: `EVENT_RESOURCE_ACQUIRE`를 `4,` 타입으로 선언합니다. 의미/사용: `EVENT_RESOURCE_ACQUIRE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_RESOURCE_ACQUIRE: 4,
        # LINE-BY-LINE: `EVENT_PROCESS_START`를 `5,` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_START` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_PROCESS_START: 5,
        # LINE-BY-LINE: `EVENT_PROCESS_FINISH`를 `6,` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_FINISH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_PROCESS_FINISH: 6,
        # LINE-BY-LINE: `EVENT_RESOURCE_RELEASE`를 `7,` 타입으로 선언합니다. 의미/사용: `EVENT_RESOURCE_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_RESOURCE_RELEASE: 7,
        # LINE-BY-LINE: `EVENT_DOWNSTREAM_ARRIVAL`를 `8,` 타입으로 선언합니다. 의미/사용: `EVENT_DOWNSTREAM_ARRIVAL` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_DOWNSTREAM_ARRIVAL: 8,
        # LINE-BY-LINE: `EVENT_DOWNSTREAM_RELEASE`를 `9,` 타입으로 선언합니다. 의미/사용: `EVENT_DOWNSTREAM_RELEASE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_DOWNSTREAM_RELEASE: 9,
        # LINE-BY-LINE: `EVENT_VALIDATION_FAILURE`를 `99,` 타입으로 선언합니다. 의미/사용: `EVENT_VALIDATION_FAILURE` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        EVENT_VALIDATION_FAILURE: 99,
    }
    # LINE-BY-LINE: 호출자에게 `order.get(event_type, 50)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return order.get(event_type, 50)
