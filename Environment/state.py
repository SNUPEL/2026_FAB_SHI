"""시뮬레이션의 현재 상태를 저장하는 자료형.

이 파일은 "환경이 지금 어디까지 진행됐는지"를 저장합니다.

특히 초보자가 헷갈리기 쉬운 포인트는 아래 두 가지입니다.

1. `machine_loads`
   - 전체 horizon 기준 누적 부하

2. `machine_daily_loads`
   - 날짜별 누적 부하
   - 일일 capacity override를 넣으려면 이 값이 필요합니다.
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass, field`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass, field
# LINE-BY-LINE: `typing` 모듈에서 `Dict, List, Optional, Set`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List, Optional, Set

# LINE-BY-LINE: `.data` 모듈에서 `BatchCandidate, DownstreamEvent, ScheduledOperation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .data import BatchCandidate, DownstreamEvent, ScheduledOperation
# LINE-BY-LINE: `.events` 모듈에서 `FactoryEvent`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .events import FactoryEvent


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `SimulationState` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class SimulationState:
    """현재 의사결정 시점의 상태.

    current_time:
      현재 시뮬레이션 시간이 어디까지 왔는지
    unscheduled_jobs:
      아직 배정되지 않은 작업 ID 집합
    machine_available_at:
      각 설비에서 가장 빨리 비는 슬롯 시점
    machine_slot_available_at:
      각 설비 슬롯별 다시 비는 시점
    machine_loads:
      각 설비에 누적된 총 부하
    downstream_loads:
      각 후공정 베이에 현재 누적된 적치량
    machine_daily_loads:
      날짜별 설비 누적 부하
      예: {"2026-01-01": {"laser_01": 120.0, "plasma_01": 80.0}}
    machine_daily_job_counts:
      날짜별 설비별 W/O 수
    machine_daily_length_sums:
      날짜별 설비별 길이 합계
    scheduled_job_count_by_day:
      날짜별 배정된 총 작업 수
    block_set_bay_assignments:
      동일 호선+블록 묶음이 이미 배정된 절단 Bay
    machine_last_family:
      각 설비에서 직전에 처리한 작업 계열
    resource_active_until:
      보조자원별 현재 점유 종료 시각 목록
    downstream_events:
      후공정 버퍼 증감 예정 이벤트
    open_batches:
      아직 절단을 시작하지 않은 batch/open-batch 상태.

      import/사용 흐름:
      - `BatchCandidate`는 `Environment/data.py`에 정의되어 있다.
      - 이 state 파일은 `from .data import BatchCandidate`로 가져온다.
      - `CuttingSimulation.get_candidates()`가 action_space.mode=batch_open일 때
        이 dict를 보고 `add_to_batch` 또는 `close_batch` 후보를 만든다.

      예:
      - 비어 있음: 아직 batch를 구성 중인 설비가 없음
      - {"B000001": BatchCandidate(machine_id="PLS21", job_ids=("J1", "J2"))}
        PLS21에 J1/J2를 올릴 준비가 되었지만 아직 절단 start event는 발생하지 않음

      중요한 점:
      - open batch 안의 W/O는 `unscheduled_jobs`에서 제거한다.
      - 그래야 같은 W/O가 다른 설비 후보에 중복으로 뜨지 않는다.
      - 다만 아직 `schedule`에는 들어가지 않는다. `close_batch`가 선택될 때 schedule/event가 생성된다.
    batch_seq:
      batch_id 생성을 위한 증가 counter. B000001, B000002처럼 만든다.
    schedule:
      지금까지의 배정 결과
    event_log:
      generated simulation 또는 actual replay가 쓰는 표준 공장 event 로그
    event_seq:
      event_id 생성을 위한 증가 counter
    violation_log:
      나중에 soft violation 또는 post-check 결과를 쌓기 위한 자리
    """

    # LINE-BY-LINE: `current_time` 변수에 `0.0` 결과를 저장합니다. 의미: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_time: float = 0.0
    # LINE-BY-LINE: `unscheduled_jobs` 변수에 `field(default_factory=set)` 결과를 저장합니다. 의미: `unscheduled_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    unscheduled_jobs: Set[str] = field(default_factory=set)
    # LINE-BY-LINE: `machine_available_at` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_available_at` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_available_at: Dict[str, float] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_slot_available_at` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_slot_available_at` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_slot_available_at: Dict[str, List[float]] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_loads` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_loads: Dict[str, float] = field(default_factory=dict)
    # LINE-BY-LINE: `downstream_loads` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `downstream_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_loads: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_daily_loads` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_daily_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_loads: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_daily_job_counts` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_daily_job_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_job_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_daily_length_sums` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_daily_length_sums` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_length_sums: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # LINE-BY-LINE: `scheduled_job_count_by_day` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `scheduled_job_count_by_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scheduled_job_count_by_day: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `block_set_bay_assignments` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `block_set_bay_assignments` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    block_set_bay_assignments: Dict[str, str] = field(default_factory=dict)
    # LINE-BY-LINE: `machine_last_family` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_last_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_last_family: Dict[str, Optional[str]] = field(default_factory=dict)
    # LINE-BY-LINE: `resource_active_until` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `resource_active_until` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    resource_active_until: Dict[str, List[float]] = field(default_factory=dict)
    # LINE-BY-LINE: `downstream_events` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `downstream_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_events: List[DownstreamEvent] = field(default_factory=list)
    # LINE-BY-LINE: `open_batches` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `open_batches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    open_batches: Dict[str, BatchCandidate] = field(default_factory=dict)
    # LINE-BY-LINE: `batch_seq` 변수에 `0` 결과를 저장합니다. 의미: `batch_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_seq: int = 0
    # LINE-BY-LINE: `schedule` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `schedule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule: List[ScheduledOperation] = field(default_factory=list)
    # LINE-BY-LINE: `event_log` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `event_log`는 FactoryEvent list입니다. generated/actual DES 상태 변화를 시간순으로 기록합니다.
    event_log: List[FactoryEvent] = field(default_factory=list)
    # LINE-BY-LINE: `event_seq` 변수에 `0` 결과를 저장합니다. 의미: `event_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_seq: int = 0
    # LINE-BY-LINE: `violation_log` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `violation_log` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violation_log: List[str] = field(default_factory=list)
