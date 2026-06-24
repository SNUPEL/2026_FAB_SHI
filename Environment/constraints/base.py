"""제약조건 공통 자료형.

초보자도 쉽게 새 제약을 추가할 수 있도록
모든 제약 함수는 동일한 입력과 동일한 반환 형식을 사용합니다.

핵심 아이디어:
- 제약 함수는 항상 `ConstraintContext`를 입력으로 받습니다.
- 제약 함수는 항상 `ConstraintResult`를 반환합니다.
- 그래서 제약을 추가할 때 simulation 내부를 수정하지 않고도
  규칙 함수를 파일에 하나 더 만드는 방식으로 확장할 수 있습니다.
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass, field`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass, field
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `ConstraintResult` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
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

    # LINE-BY-LINE: `rule_name`를 `str` 타입으로 선언합니다. 의미/사용: `ConstraintResult.rule_name` 필드/속성입니다. 사용: ConstraintResult 객체를 만들거나 이후 로직에서 참조합니다.
    rule_name: str
    # LINE-BY-LINE: `passed`를 `bool` 타입으로 선언합니다. 의미/사용: `ConstraintResult.passed`는 제약 통과 여부 boolean입니다. ConstraintResult.passed로 전달됩니다.
    passed: bool
    # LINE-BY-LINE: `reason` 변수에 `""` 결과를 저장합니다. 의미: `reason`는 제약 실패나 디버깅 설명 문자열입니다. report/로그에서 사람이 읽습니다.
    reason: str = ""
    # LINE-BY-LINE: `penalty` 변수에 `0.0` 결과를 저장합니다. 의미: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    penalty: float = 0.0
    # LINE-BY-LINE: `metadata` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metadata: Dict[str, Any] = field(default_factory=dict)


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `ConstraintContext` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
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

    machine_daily_job_count / machine_daily_length_sum:
      오늘 이 설비에 이미 배정된 W/O 수와 길이 합계. 일일 부하 리포트용이다.

    machine_day_wo_count_limit / machine_day_length_sum_limit:
      기존 config 호환을 위한 이름이다.
      현업 확인 후 generated planning 기본값에서는 이 legacy active/rolling rule을 끄고
      batch_* rule을 사용한다.

    candidate_start_time:
      후보 action의 시작 시각

    machine_concurrent_job_count / machine_concurrent_length_sum:
      후보 시작 시각에 같은 설비에서 이미 active인 W/O 수와 길이 합계.
      현재는 legacy rolling/debug 및 actual replay 데이터 품질 audit용이다.

    machine_bay_id / job_cut_bay:
      선택 설비가 속한 절단 Bay와, 작업에 고정 Bay가 있을 때의 Bay

    block_set_id / assigned_block_set_bay:
      같은 호선+블록 묶음을 같은 Bay로 유지하기 위한 정보

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

    batch_*:
      현업 확인된 3 W/O / 길이합 55000 batch action에 필요한 자리.
      batch는 같이 들어오고 같이 빠지는 묶음이며, generated planning 기본 제약이다.

    *_priority_tier:
      현업이 제공하는 설비/Bay/후공정 우선순위 matrix를 넣기 위한 자리.
      rule이 켜졌는데 값이 없으면 통과시키지 않고 원인을 반환해야 한다.
    """

    # LINE-BY-LINE: `job`를 `Any` 타입으로 선언합니다. 의미/사용: `ConstraintContext.job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
    job: Any
    # LINE-BY-LINE: `machine`를 `Any` 타입으로 선언합니다. 의미/사용: `ConstraintContext.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
    machine: Any
    # LINE-BY-LINE: `state`를 `Any` 타입으로 선언합니다. 의미/사용: `ConstraintContext.state` 필드/속성입니다. 사용: ConstraintContext 객체를 만들거나 이후 로직에서 참조합니다.
    state: Any
    # LINE-BY-LINE: `jobs`를 `Dict[str, Any]` 타입으로 선언합니다. 의미/사용: `ConstraintContext.jobs` 필드/속성입니다. 사용: ConstraintContext 객체를 만들거나 이후 로직에서 참조합니다.
    jobs: Dict[str, Any]
    # LINE-BY-LINE: `machines`를 `Dict[str, Any]` 타입으로 선언합니다. 의미/사용: `ConstraintContext.machines` 필드/속성입니다. 사용: ConstraintContext 객체를 만들거나 이후 로직에서 참조합니다.
    machines: Dict[str, Any]
    # LINE-BY-LINE: `bays`를 `Dict[str, Any]` 타입으로 선언합니다. 의미/사용: `ConstraintContext.bays` 필드/속성입니다. 사용: ConstraintContext 객체를 만들거나 이후 로직에서 참조합니다.
    bays: Dict[str, Any]
    # LINE-BY-LINE: `layout`를 `Any` 타입으로 선언합니다. 의미/사용: `ConstraintContext.layout` 필드/속성입니다. 사용: ConstraintContext 객체를 만들거나 이후 로직에서 참조합니다.
    layout: Any
    # LINE-BY-LINE: `config`를 `Dict[str, Any]` 타입으로 선언합니다. 의미/사용: `ConstraintContext.config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config: Dict[str, Any]
    # LINE-BY-LINE: `current_day_key` 변수에 `""` 결과를 저장합니다. 의미: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_day_key: str = ""
    # LINE-BY-LINE: `current_day_index` 변수에 `0` 결과를 저장합니다. 의미: `current_day_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current_day_index: int = 0
    # LINE-BY-LINE: `machine_enabled_flag` 변수에 `True` 결과를 저장합니다. 의미: `machine_enabled_flag` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_enabled_flag: bool = True
    # LINE-BY-LINE: `machine_daily_capacity_limit` 변수에 `0.0` 결과를 저장합니다. 의미: `machine_daily_capacity_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_capacity_limit: float = 0.0
    # LINE-BY-LINE: `machine_daily_load` 변수에 `0.0` 결과를 저장합니다. 의미: `machine_daily_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_load: float = 0.0
    # LINE-BY-LINE: `machine_daily_job_count` 변수에 `0` 결과를 저장합니다. 의미: `machine_daily_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_job_count: int = 0
    # LINE-BY-LINE: `machine_daily_length_sum` 변수에 `0.0` 결과를 저장합니다. 의미: `machine_daily_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_daily_length_sum: float = 0.0
    # LINE-BY-LINE: `machine_day_wo_count_limit` 변수에 `None` 결과를 저장합니다. 의미: `machine_day_wo_count_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_day_wo_count_limit: Any = None
    # LINE-BY-LINE: `machine_day_length_sum_limit` 변수에 `None` 결과를 저장합니다. 의미: `machine_day_length_sum_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_day_length_sum_limit: Any = None
    # LINE-BY-LINE: `candidate_start_time` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_start_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_start_time: float = 0.0
    # LINE-BY-LINE: `machine_concurrent_job_count` 변수에 `0` 결과를 저장합니다. 의미: `machine_concurrent_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_concurrent_job_count: int = 0
    # LINE-BY-LINE: `machine_concurrent_length_sum` 변수에 `0.0` 결과를 저장합니다. 의미: `machine_concurrent_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_concurrent_length_sum: float = 0.0
    # LINE-BY-LINE: `machine_bay_id` 변수에 `None` 결과를 저장합니다. 의미: `machine_bay_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_bay_id: Any = None
    # LINE-BY-LINE: `job_cut_bay` 변수에 `None` 결과를 저장합니다. 의미: `job_cut_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    job_cut_bay: Any = None
    # LINE-BY-LINE: `block_set_id` 변수에 `None` 결과를 저장합니다. 의미: `block_set_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    block_set_id: Any = None
    # LINE-BY-LINE: `assigned_block_set_bay` 변수에 `None` 결과를 저장합니다. 의미: `assigned_block_set_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    assigned_block_set_bay: Any = None
    # LINE-BY-LINE: `scheduled_job_count_today` 변수에 `0` 결과를 저장합니다. 의미: `scheduled_job_count_today` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scheduled_job_count_today: int = 0
    # LINE-BY-LINE: `daily_job_cap` 변수에 `None` 결과를 저장합니다. 의미: `daily_job_cap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    daily_job_cap: Any = None
    # LINE-BY-LINE: `global_calendar_open` 변수에 `True` 결과를 저장합니다. 의미: `global_calendar_open` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    global_calendar_open: bool = True
    # LINE-BY-LINE: `global_calendar_reason` 변수에 `""` 결과를 저장합니다. 의미: `global_calendar_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    global_calendar_reason: str = ""
    # LINE-BY-LINE: `machine_calendar_open` 변수에 `True` 결과를 저장합니다. 의미: `machine_calendar_open` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_calendar_open: bool = True
    # LINE-BY-LINE: `machine_calendar_reason` 변수에 `""` 결과를 저장합니다. 의미: `machine_calendar_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_calendar_reason: str = ""
    # LINE-BY-LINE: `machine_breakdown_active` 변수에 `False` 결과를 저장합니다. 의미: `machine_breakdown_active` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_breakdown_active: bool = False
    # LINE-BY-LINE: `machine_breakdown_reason` 변수에 `""` 결과를 저장합니다. 의미: `machine_breakdown_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_breakdown_reason: str = ""
    # LINE-BY-LINE: `candidate_processing_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_processing_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_processing_minutes: float = 0.0
    # LINE-BY-LINE: `candidate_elapsed_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_elapsed_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_elapsed_minutes: float = 0.0
    # LINE-BY-LINE: `candidate_finish_time` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_finish_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_finish_time: float = 0.0
    # LINE-BY-LINE: `candidate_downstream_arrival_time` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_downstream_arrival_time: float = 0.0
    # LINE-BY-LINE: `candidate_downstream_release_time` 변수에 `None` 결과를 저장합니다. 의미: `candidate_downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_downstream_release_time: Any = None
    # LINE-BY-LINE: `candidate_changeover_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_changeover_minutes: float = 0.0
    # LINE-BY-LINE: `candidate_blocked_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `candidate_blocked_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_blocked_minutes: float = 0.0
    # LINE-BY-LINE: `predicted_downstream_load_at_arrival` 변수에 `0` 결과를 저장합니다. 의미: `predicted_downstream_load_at_arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_downstream_load_at_arrival: int = 0
    # LINE-BY-LINE: `required_resource_ids` 변수에 `field(default_factory=tuple)` 결과를 저장합니다. 의미: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_resource_ids: Any = field(default_factory=tuple)
    # LINE-BY-LINE: `resource_shortages` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `resource_shortages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    resource_shortages: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `candidate_batch_id` 변수에 `None` 결과를 저장합니다. 의미: `candidate_batch_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_batch_id: Any = None
    # LINE-BY-LINE: `candidate_batch_job_ids` 변수에 `field(default_factory=tuple)` 결과를 저장합니다. 의미: `candidate_batch_job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_batch_job_ids: Any = field(default_factory=tuple)
    # LINE-BY-LINE: `candidate_batch_wo_count` 변수에 `None` 결과를 저장합니다. 의미: `candidate_batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_batch_wo_count: Any = None
    # LINE-BY-LINE: `candidate_batch_length_sum` 변수에 `None` 결과를 저장합니다. 의미: `candidate_batch_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_batch_length_sum: Any = None
    # LINE-BY-LINE: `candidate_batch_status` 변수에 `None` 결과를 저장합니다. 의미: `candidate_batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_batch_status: Any = None
    # LINE-BY-LINE: `batch_wo_count_limit` 변수에 `None` 결과를 저장합니다. 의미: `batch_wo_count_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_wo_count_limit: Any = None
    # LINE-BY-LINE: `batch_length_sum_limit` 변수에 `None` 결과를 저장합니다. 의미: `batch_length_sum_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_length_sum_limit: Any = None
    # LINE-BY-LINE: `machine_priority_tier` 변수에 `None` 결과를 저장합니다. 의미: `machine_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_priority_tier: Any = None
    # LINE-BY-LINE: `bay_priority_tier` 변수에 `None` 결과를 저장합니다. 의미: `bay_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_priority_tier: Any = None
    # LINE-BY-LINE: `downstream_priority_tier` 변수에 `None` 결과를 저장합니다. 의미: `downstream_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_priority_tier: Any = None
    # LINE-BY-LINE: `has_primary_machine_tier_candidate` 변수에 `None` 결과를 저장합니다. 의미: `has_primary_machine_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    has_primary_machine_tier_candidate: Any = None
    # LINE-BY-LINE: `has_primary_bay_tier_candidate` 변수에 `None` 결과를 저장합니다. 의미: `has_primary_bay_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    has_primary_bay_tier_candidate: Any = None
    # LINE-BY-LINE: `has_primary_downstream_tier_candidate` 변수에 `None` 결과를 저장합니다. 의미: `has_primary_downstream_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    has_primary_downstream_tier_candidate: Any = None
    # LINE-BY-LINE: `table_type` 변수에 `None` 결과를 저장합니다. 의미: `table_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    table_type: Any = None
    # LINE-BY-LINE: `allowed_table_types` 변수에 `field(default_factory=tuple)` 결과를 저장합니다. 의미: `allowed_table_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    allowed_table_types: Any = field(default_factory=tuple)
    # LINE-BY-LINE: `plate_width` 변수에 `None` 결과를 저장합니다. 의미: `plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    plate_width: Any = None
    # LINE-BY-LINE: `machine_min_plate_width` 변수에 `None` 결과를 저장합니다. 의미: `machine_min_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_min_plate_width: Any = None
    # LINE-BY-LINE: `machine_max_plate_width` 변수에 `None` 결과를 저장합니다. 의미: `machine_max_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_max_plate_width: Any = None
    # LINE-BY-LINE: `downstream_due_minutes` 변수에 `None` 결과를 저장합니다. 의미: `downstream_due_minutes`는 후공정 요구시각입니다. 사용: downstream_due_date hard rule 후보.
    downstream_due_minutes: Any = None
    # LINE-BY-LINE: `downstream_required_start_minutes` 변수에 `None` 결과를 저장합니다. 의미: `downstream_required_start_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_required_start_minutes: Any = None
    # LINE-BY-LINE: `candidate_downstream_slack_minutes` 변수에 `None` 결과를 저장합니다. 의미: `candidate_downstream_slack_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_downstream_slack_minutes: Any = None
    # LINE-BY-LINE: `predicted_storage_load_at_arrival` 변수에 `None` 결과를 저장합니다. 의미: `predicted_storage_load_at_arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_storage_load_at_arrival: Any = None
