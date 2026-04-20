"""환경에서 사용하는 기본 데이터 구조."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class Job:
    """절단 대상 작업 1건.

    현재 스켈레톤에서는 실제 현업 데이터가 없으므로
    연구계획서에 나온 핵심 속성만 최소한으로 반영합니다.
    """

    job_id: str
    family: str
    thickness: float
    plate_length: float
    downstream_bay: str
    priority_weight: int
    base_stage_minutes: Dict[str, float]
    due_date_minutes: Optional[float] = None
    preferred_machine_types: Tuple[str, ...] = ()
    required_resource_ids: Tuple[str, ...] = ()

    def estimate_total_minutes(self, machine: "Machine") -> float:
        """설비별 총 예상 처리시간.

        현재는 3단계 공정을 단순 합산합니다.
        - setup
        - cut
        - finish

        나중에 실제 택트타임 모델이 들어오면
        이 함수 또는 별도 tact time estimator로 치환하면 됩니다.
        """

        setup = float(self.base_stage_minutes.get("setup", 0.0))
        cut = float(self.base_stage_minutes.get("cut", 0.0)) * machine.cut_speed_factor
        finish = float(self.base_stage_minutes.get("finish", 0.0))
        return setup + cut + finish


@dataclass
class Machine:
    """절단 설비 1대."""

    machine_id: str
    machine_type: str
    enabled: bool
    eligible_families: Iterable[str]
    min_thickness: float
    max_thickness: float
    table_length_limit: float
    cut_speed_factor: float
    daily_capacity_minutes: float
    parallel_capacity: int = 1
    required_resource_ids: Tuple[str, ...] = ()
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class DownstreamBay:
    """후공정 베이 또는 적치 대상 작업장."""

    bay_id: str
    priority_rank: int
    capacity_limit: int
    transfer_time_minutes: float = 0.0
    release_delay_minutes: Optional[float] = None


@dataclass
class AuxiliaryResource:
    """크레인, 정반 등 단순 슬롯 자원."""

    resource_id: str
    capacity: int


@dataclass(order=True)
class DownstreamEvent:
    """후공정 버퍼의 증감 이벤트."""

    event_time: float
    bay_id: str
    delta: int
    event_type: str = "arrival"


@dataclass
class ScheduledOperation:
    """스케줄 결과에 기록되는 배정 1건."""

    job_id: str
    machine_id: str
    start_time: float
    finish_time: float
    downstream_bay: str
    estimated_minutes: float
    processing_minutes: float = 0.0
    changeover_minutes: float = 0.0
    blocked_minutes: float = 0.0
    downstream_arrival_time: Optional[float] = None
    downstream_release_time: Optional[float] = None
    machine_slot_index: int = 0
    resource_ids: Tuple[str, ...] = ()
    stage_minutes: Dict[str, float] = field(default_factory=dict)
