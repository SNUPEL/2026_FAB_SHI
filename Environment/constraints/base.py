"""제약조건 공통 자료형.

초보자도 쉽게 새 제약을 추가할 수 있도록
모든 제약 함수는 동일한 입력과 동일한 반환 형식을 사용합니다.

핵심 아이디어:
- 제약 함수는 항상 `ConstraintContext`를 입력으로 받습니다.
- 제약 함수는 항상 `ConstraintResult`를 반환합니다.
- 그래서 제약을 추가할 때 simulation 내부를 수정하지 않고도
  규칙 함수를 파일에 하나 더 만드는 방식으로 확장할 수 있습니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ConstraintResult:
    """제약 평가 결과 1건.

    - passed:
      True면 제약을 만족했다는 뜻입니다.
      False면 제약을 만족하지 못했다는 뜻입니다.

    - penalty:
      소프트 제약에서만 주로 사용합니다.
      하드 제약은 보통 passed=False가 되면 액션 자체를 제거하므로
      penalty는 0으로 두는 경우가 많습니다.
    """

    rule_name: str
    passed: bool
    reason: str = ""
    penalty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintContext:
    """제약 함수가 판단할 때 필요한 전체 문맥.

    job:
      지금 배정하려는 작업
    machine:
      지금 배정하려는 설비
    state:
      현재 시뮬레이션 상태
    jobs / machines / bays:
      전체 작업 / 설비 / 후공정 베이 정보
    layout:
      현재는 위치 정보만 유지하는 단순 메타데이터
    config:
      config.yaml 전체 설정

    current_day_key / current_day_index:
      현재 시뮬레이션 시각이 어느 운영일에 해당하는지

    machine_enabled_flag:
      날짜별 override까지 반영한 "실제 오늘 사용 가능 여부"

    machine_daily_capacity_limit:
      날짜별 override까지 반영한 "오늘 이 설비의 사용 가능 시간"

    machine_daily_load:
      오늘 이 설비에 이미 누적된 사용 시간

    scheduled_job_count_today:
      오늘 이미 배정된 작업 수

    daily_job_cap:
      날짜별 override까지 반영한 오늘 전체 작업 수 제한
      없으면 None

    global_calendar_open / global_calendar_reason:
      현재 시각에 공장 전체가 가동 가능한지와, 불가능한 이유

    machine_calendar_open / machine_calendar_reason:
      현재 시각에 특정 설비가 계획된 운영시간 안에 있는지와, 그 이유

    machine_breakdown_active / machine_breakdown_reason:
      현재 시각에 특정 설비가 고장/비계획 정지 상태인지와, 그 이유

    candidate_processing_minutes / candidate_elapsed_minutes:
      현재 후보 action을 지금 시작했을 때의 순수 작업시간 / 실제 점유시간

    candidate_finish_time:
      작업구간 캘린더를 반영한 완료 시점

    candidate_downstream_arrival_time / candidate_downstream_release_time:
      후공정 버퍼 도착 및 해소 예정 시점

    predicted_downstream_load_at_arrival:
      후보 action까지 포함했을 때 도착 시점 예상 버퍼 적치량

    required_resource_ids / resource_shortages:
      후보 action이 점유해야 하는 단순 슬롯 자원과 부족 자원 목록
    """

    job: Any
    machine: Any
    state: Any
    jobs: Dict[str, Any]
    machines: Dict[str, Any]
    bays: Dict[str, Any]
    layout: Any
    config: Dict[str, Any]
    current_day_key: str = ""
    current_day_index: int = 0
    machine_enabled_flag: bool = True
    machine_daily_capacity_limit: float = 0.0
    machine_daily_load: float = 0.0
    scheduled_job_count_today: int = 0
    daily_job_cap: Any = None
    global_calendar_open: bool = True
    global_calendar_reason: str = ""
    machine_calendar_open: bool = True
    machine_calendar_reason: str = ""
    machine_breakdown_active: bool = False
    machine_breakdown_reason: str = ""
    candidate_processing_minutes: float = 0.0
    candidate_elapsed_minutes: float = 0.0
    candidate_finish_time: float = 0.0
    candidate_downstream_arrival_time: float = 0.0
    candidate_downstream_release_time: Any = None
    candidate_changeover_minutes: float = 0.0
    candidate_blocked_minutes: float = 0.0
    predicted_downstream_load_at_arrival: int = 0
    required_resource_ids: Any = field(default_factory=tuple)
    resource_shortages: Dict[str, int] = field(default_factory=dict)
