"""시뮬레이션의 현재 상태를 저장하는 자료형.

이 파일은 "환경이 지금 어디까지 진행됐는지"를 저장합니다.

특히 초보자가 헷갈리기 쉬운 포인트는 아래 두 가지입니다.

1. `machine_loads`
   - 전체 horizon 기준 누적 부하

2. `machine_daily_loads`
   - 날짜별 누적 부하
   - 일일 capacity override를 넣으려면 이 값이 필요합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .data import DownstreamEvent, ScheduledOperation


@dataclass
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
    scheduled_job_count_by_day:
      날짜별 배정된 총 작업 수
    machine_last_family:
      각 설비에서 직전에 처리한 작업 계열
    resource_active_until:
      보조자원별 현재 점유 종료 시각 목록
    downstream_events:
      후공정 버퍼 증감 예정 이벤트
    schedule:
      지금까지의 배정 결과
    violation_log:
      나중에 soft violation 또는 post-check 결과를 쌓기 위한 자리
    """

    current_time: float = 0.0
    unscheduled_jobs: Set[str] = field(default_factory=set)
    machine_available_at: Dict[str, float] = field(default_factory=dict)
    machine_slot_available_at: Dict[str, List[float]] = field(default_factory=dict)
    machine_loads: Dict[str, float] = field(default_factory=dict)
    downstream_loads: Dict[str, int] = field(default_factory=dict)
    machine_daily_loads: Dict[str, Dict[str, float]] = field(default_factory=dict)
    scheduled_job_count_by_day: Dict[str, int] = field(default_factory=dict)
    machine_last_family: Dict[str, Optional[str]] = field(default_factory=dict)
    resource_active_until: Dict[str, List[float]] = field(default_factory=dict)
    downstream_events: List[DownstreamEvent] = field(default_factory=list)
    schedule: List[ScheduledOperation] = field(default_factory=list)
    violation_log: List[str] = field(default_factory=list)
