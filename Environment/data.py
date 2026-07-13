"""환경에서 사용하는 기본 데이터 구조.

이 파일은 실제 알고리즘 로직을 수행하지 않는다.
대신 환경이 주고받는 핵심 객체의 필드 구조를 한 곳에 모아 둔다.

전체 흐름:
1. `Utils/data/cutting_scenario_builder.py`가 Excel/CSV row를 scenario dict로 만든다.
2. `Environment/environment.py`가 scenario dict를 이 파일의 dataclass로 변환한다.
3. `Environment/simulation.py`가 dataclass를 읽어 후보 생성, 제약 평가, event 생성을 수행한다.

주의:
- 여기에 필드를 추가한다고 제약이 자동으로 생기지는 않는다.
- 새 필드를 쓰는 제약은 `Environment/constraints/`와 registry에 별도로 추가해야 한다.
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass, field`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass, field
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, Optional, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, Optional, Tuple


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `Job` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class Job:
    """절단 대상 작업 1건.

    주요 필드:
    - job_id: 환경 내부 id. 원본 W/O 번호와 1:1일 수도 있고 prefix가 붙을 수도 있다.
    - family: NP/FN/FL 같은 계열
    - thickness: 두께
    - plate_length: 길이. rolling capacity의 55000 길이합 계산에 사용한다.
    - downstream_bay: 절단 이후 이동할 후공정/적치 Bay
    - base_stage_minutes: setup/cut/finish 시간 dict
    - cut_bay: 작업이 특정 절단 Bay에 고정될 때 사용한다.
    - source_machine_id/source_cut_bay: actual replay에서 실적 장비/Bay를 보존한다.
    - actual_*: 실적 착수/종료/elapsed 검증용 필드

    확장 필드:
    - nesting_id, batch_group_id, window_group_id는 현업 batch/window 데이터가 들어올 때 사용한다.
    - allowed/prohibited machine/bay는 W/O별 whitelist/blacklist 제약용이다.
    - extra는 아직 dataclass 필드로 정식 반영하지 않은 원본 정보를 보존하는 자리다.
    """

    # LINE-BY-LINE: `job_id`를 `str` 타입으로 선언합니다. 의미/사용: `Job.job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
    job_id: str
    # LINE-BY-LINE: `family`를 `str` 타입으로 선언합니다. 의미/사용: `Job.family`는 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
    family: str
    # LINE-BY-LINE: `thickness`를 `float` 타입으로 선언합니다. 의미/사용: `Job.thickness`는 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
    thickness: float
    # LINE-BY-LINE: `plate_length`를 `float` 타입으로 선언합니다. 의미/사용: `Job.plate_length`는 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
    plate_length: float
    # LINE-BY-LINE: `downstream_bay`를 `str` 타입으로 선언합니다. 의미/사용: `Job.downstream_bay`는 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
    downstream_bay: str
    # LINE-BY-LINE: `priority_weight`를 `int` 타입으로 선언합니다. 의미/사용: `Job.priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
    priority_weight: int
    # LINE-BY-LINE: `base_stage_minutes`를 `Dict[str, float]` 타입으로 선언합니다. 의미/사용: `Job.base_stage_minutes`는 setup/cut/finish 처리시간 dict입니다. 사용: Job.estimate_total_minutes().
    base_stage_minutes: Dict[str, float]
    # LINE-BY-LINE: `due_date_minutes` 변수에 `None` 결과를 저장합니다. 의미: `due_date_minutes`는 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
    due_date_minutes: Optional[float] = None
    # LINE-BY-LINE: `preferred_machine_types` 변수에 `()` 결과를 저장합니다. 의미: `preferred_machine_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    preferred_machine_types: Tuple[str, ...] = ()
    # LINE-BY-LINE: `required_resource_ids` 변수에 `()` 결과를 저장합니다. 의미: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_resource_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `cut_bay` 변수에 `None` 결과를 저장합니다. 의미: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
    cut_bay: Optional[str] = None
    # LINE-BY-LINE: `block_set_id` 변수에 `None` 결과를 저장합니다. 의미: `block_set_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    block_set_id: Optional[str] = None
    # LINE-BY-LINE: `source_machine_id` 변수에 `None` 결과를 저장합니다. 의미: `source_machine_id`는 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
    source_machine_id: Optional[str] = None
    # LINE-BY-LINE: `source_cut_bay` 변수에 `None` 결과를 저장합니다. 의미: `source_cut_bay`는 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
    source_cut_bay: Optional[str] = None
    # LINE-BY-LINE: `actual_duration_minutes` 변수에 `None` 결과를 저장합니다. 의미: `actual_duration_minutes`는 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
    actual_duration_minutes: Optional[float] = None
    # LINE-BY-LINE: `actual_start_minutes` 변수에 `None` 결과를 저장합니다. 의미: `actual_start_minutes`는 실적 착수시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
    actual_start_minutes: Optional[float] = None
    # LINE-BY-LINE: `actual_finish_minutes` 변수에 `None` 결과를 저장합니다. 의미: `actual_finish_minutes`는 실적 종료시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
    actual_finish_minutes: Optional[float] = None
    # LINE-BY-LINE: `planned_start_minutes` 변수에 `None` 결과를 저장합니다. 의미: `planned_start_minutes`는 계획 착수일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
    planned_start_minutes: Optional[float] = None
    # LINE-BY-LINE: `planned_finish_minutes` 변수에 `None` 결과를 저장합니다. 의미: `planned_finish_minutes`는 계획 종료일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
    planned_finish_minutes: Optional[float] = None
    # LINE-BY-LINE: `plate_width` 변수에 `None` 결과를 저장합니다. 의미: `plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    plate_width: Optional[float] = None
    # LINE-BY-LINE: `cut_length` 변수에 `None` 결과를 저장합니다. 의미: `cut_length`는 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
    cut_length: Optional[float] = None
    # LINE-BY-LINE: `marking_length` 변수에 `None` 결과를 저장합니다. 의미: `marking_length`는 마킹 길이입니다. 사용: tact time 분석 feature.
    marking_length: Optional[float] = None
    # LINE-BY-LINE: `bevel_length` 변수에 `None` 결과를 저장합니다. 의미: `bevel_length`는 베벨 길이입니다. 사용: tact time 분석 feature.
    bevel_length: Optional[float] = None
    # LINE-BY-LINE: `bevel_quantity` 변수에 `None` 결과를 저장합니다. 의미: `bevel_quantity`는 베벨/개선 수량입니다. 예: `BV_QTY`, 사용: Phase 1 Bay 다목적 평준화.
    bevel_quantity: Optional[int] = None
    # LINE-BY-LINE: `part_count` 변수에 `None` 결과를 저장합니다. 의미: `part_count`는 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
    part_count: Optional[int] = None
    # LINE-BY-LINE: `steel_quantity` 변수에 `None` 결과를 저장합니다. 의미: `steel_quantity`는 강재 수량입니다. 예: `STL_QTY`, 사용: 데이터 품질/처리시간 feature.
    steel_quantity: Optional[int] = None
    # LINE-BY-LINE: `downstream_due_minutes` 변수에 `None` 결과를 저장합니다. 의미: `downstream_due_minutes`는 후공정 요구시각입니다. 사용: downstream_due_date hard rule 후보.
    downstream_due_minutes: Optional[float] = None
    # LINE-BY-LINE: `downstream_required_start_minutes` 변수에 `None` 결과를 저장합니다. 의미: `downstream_required_start_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_required_start_minutes: Optional[float] = None
    # LINE-BY-LINE: `nesting_id` 변수에 `None` 결과를 저장합니다. 의미: `nesting_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    nesting_id: Optional[str] = None
    # LINE-BY-LINE: `batch_group_id` 변수에 `None` 결과를 저장합니다. 의미: `batch_group_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_group_id: Optional[str] = None
    # LINE-BY-LINE: `window_group_id` 변수에 `None` 결과를 저장합니다. 의미: `window_group_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    window_group_id: Optional[str] = None
    # LINE-BY-LINE: `allowed_machine_ids` 변수에 `()` 결과를 저장합니다. 의미: `allowed_machine_ids`는 W/O별 허용 설비 whitelist입니다. 예: `("PLS21",)`, 사용: allowed_machine_ids rule.
    allowed_machine_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `prohibited_machine_ids` 변수에 `()` 결과를 저장합니다. 의미: `prohibited_machine_ids`는 W/O별 금지 설비 blacklist입니다. 사용: prohibited_machine_ids rule.
    prohibited_machine_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `allowed_bay_ids` 변수에 `()` 결과를 저장합니다. 의미: `allowed_bay_ids`는 W/O별 허용 절단 Bay whitelist입니다. 예: `("22",)`, 사용: allowed_bay_ids rule.
    allowed_bay_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `allowed_table_types` 변수에 `()` 결과를 저장합니다. 의미: `allowed_table_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    allowed_table_types: Tuple[str, ...] = ()
    # LINE-BY-LINE: `machine_priority_tiers` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `machine_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_priority_tiers: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `bay_priority_tiers` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `bay_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_priority_tiers: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `downstream_priority_tiers` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `downstream_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_priority_tiers: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `is_wide_plate` 변수에 `None` 결과를 저장합니다. 의미: `is_wide_plate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    is_wide_plate: Optional[bool] = None
    # LINE-BY-LINE: `manual_bevel_required` 변수에 `None` 결과를 저장합니다. 의미: `manual_bevel_required` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    manual_bevel_required: Optional[bool] = None
    # LINE-BY-LINE: `forming_required` 변수에 `None` 결과를 저장합니다. 의미: `forming_required` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    forming_required: Optional[bool] = None
    # LINE-BY-LINE: `bom_status` 변수에 `None` 결과를 저장합니다. 의미: `bom_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bom_status: Optional[str] = None
    # LINE-BY-LINE: `extra` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `extra` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    extra: Dict[str, Any] = field(default_factory=dict)

    # LINE-BY-LINE: `estimate_total_minutes(self, machine: "Machine")` 함수를 정의합니다. 반환 타입: `float`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
    def estimate_total_minutes(self, machine: "Machine") -> float:
        """설비별 총 예상 처리시간.

        현재는 3단계 공정을 단순 합산합니다.
        - setup
        - cut
        - finish

        나중에 실제 택트타임 모델이 들어오면
        이 함수 또는 별도 tact time estimator로 치환하면 됩니다.
        """

        # LINE-BY-LINE: `setup`에 `float(self.base_stage_minutes.get("setup", 0.0))` 결과를 저장합니다. 의미/사용: `setup` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        setup = float(self.base_stage_minutes.get("setup", 0.0))
        # LINE-BY-LINE: `cut`에 `float(self.base_stage_minutes.get("cut", 0.0)) * machine.cut_speed_factor` 결과를 저장합니다. 의미/사용: `cut` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cut = float(self.base_stage_minutes.get("cut", 0.0)) * machine.cut_speed_factor
        # LINE-BY-LINE: `finish`에 `float(self.base_stage_minutes.get("finish", 0.0))` 결과를 저장합니다. 의미/사용: `finish` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish = float(self.base_stage_minutes.get("finish", 0.0))
        # LINE-BY-LINE: 호출자에게 `setup + cut + finish`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return setup + cut + finish


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `Machine` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class Machine:
    """절단 설비 1대.

    주요 필드:
    - machine_id: PLS21 같은 현업 설비 id
    - machine_type/equipment_type: plasma/laser/gas 또는 PLS/LSR/NCG
    - bay_id: 이 설비가 위치한 절단 Bay. action에서 machine을 고르면 Bay도 함께 확정된다.
    - parallel_capacity: 설비가 동시에 가질 수 있는 slot 수. rolling capacity 모드에서 사용한다.
    - max_batch_wo_count/max_batch_length_sum: 설비별 3 W/O/55000 같은 capacity override
    - table_length_limit: 단일 W/O 길이 한계

    주의:
    - `parallel_capacity`는 "하루에 몇 개 처리"가 아니다.
    - 특정 시점에 동시에 active일 수 있는 slot 수로 해석해야 한다.
    """

    # LINE-BY-LINE: `machine_id`를 `str` 타입으로 선언합니다. 의미/사용: `Machine.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
    machine_id: str
    # LINE-BY-LINE: `machine_type`를 `str` 타입으로 선언합니다. 의미/사용: `Machine.machine_type` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    machine_type: str
    # LINE-BY-LINE: `enabled`를 `bool` 타입으로 선언합니다. 의미/사용: `Machine.enabled` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    enabled: bool
    # LINE-BY-LINE: `eligible_families`를 `Iterable[str]` 타입으로 선언합니다. 의미/사용: `Machine.eligible_families` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    eligible_families: Iterable[str]
    # LINE-BY-LINE: `min_thickness`를 `float` 타입으로 선언합니다. 의미/사용: `Machine.min_thickness` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    min_thickness: float
    # LINE-BY-LINE: `max_thickness`를 `float` 타입으로 선언합니다. 의미/사용: `Machine.max_thickness` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    max_thickness: float
    # LINE-BY-LINE: `table_length_limit`를 `float` 타입으로 선언합니다. 의미/사용: `Machine.table_length_limit` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    table_length_limit: float
    # LINE-BY-LINE: `cut_speed_factor`를 `float` 타입으로 선언합니다. 의미/사용: `Machine.cut_speed_factor` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    cut_speed_factor: float
    # LINE-BY-LINE: `daily_capacity_minutes`를 `float` 타입으로 선언합니다. 의미/사용: `Machine.daily_capacity_minutes` 필드/속성입니다. 사용: Machine 객체를 만들거나 이후 로직에서 참조합니다.
    daily_capacity_minutes: float
    # LINE-BY-LINE: `parallel_capacity` 변수에 `1` 결과를 저장합니다. 의미: `parallel_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    parallel_capacity: int = 1
    # LINE-BY-LINE: `bay_id` 변수에 `None` 결과를 저장합니다. 의미: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: Optional[str] = None
    # LINE-BY-LINE: `required_resource_ids` 변수에 `()` 결과를 저장합니다. 의미: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_resource_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `position` 변수에 `(0.0, 0.0)` 결과를 저장합니다. 의미: `position` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    position: Tuple[float, float] = (0.0, 0.0)
    # LINE-BY-LINE: `equipment_type` 변수에 `None` 결과를 저장합니다. 의미: `equipment_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    equipment_type: Optional[str] = None
    # LINE-BY-LINE: `table_type` 변수에 `None` 결과를 저장합니다. 의미: `table_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    table_type: Optional[str] = None
    # LINE-BY-LINE: `max_batch_wo_count` 변수에 `None` 결과를 저장합니다. 의미: `max_batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_batch_wo_count: Optional[int] = None
    # LINE-BY-LINE: `max_batch_length_sum` 변수에 `None` 결과를 저장합니다. 의미: `max_batch_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_batch_length_sum: Optional[float] = None
    # LINE-BY-LINE: `max_plate_width` 변수에 `None` 결과를 저장합니다. 의미: `max_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_plate_width: Optional[float] = None
    # LINE-BY-LINE: `min_plate_width` 변수에 `None` 결과를 저장합니다. 의미: `min_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    min_plate_width: Optional[float] = None
    # LINE-BY-LINE: `priority_tiers_by_family` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `priority_tiers_by_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    priority_tiers_by_family: Dict[str, int] = field(default_factory=dict)
    # LINE-BY-LINE: `install_start_date` 변수에 `None` 결과를 저장합니다. 의미: `install_start_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    install_start_date: Optional[str] = None
    # LINE-BY-LINE: `install_end_date` 변수에 `None` 결과를 저장합니다. 의미: `install_end_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    install_end_date: Optional[str] = None
    # LINE-BY-LINE: `calendar_id` 변수에 `None` 결과를 저장합니다. 의미: `calendar_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    calendar_id: Optional[str] = None


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `DownstreamBay` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class DownstreamBay:
    """후공정 베이 또는 적치 대상 작업장.

    절단 Bay와 다르다.
    `CUT_BAY`는 절단이 일어나는 위치이고, `downstream_bay`는 절단 이후 흘러가는 위치다.
    """

    # LINE-BY-LINE: `bay_id`를 `str` 타입으로 선언합니다. 의미/사용: `DownstreamBay.bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: str
    # LINE-BY-LINE: `priority_rank`를 `int` 타입으로 선언합니다. 의미/사용: `DownstreamBay.priority_rank` 필드/속성입니다. 사용: DownstreamBay 객체를 만들거나 이후 로직에서 참조합니다.
    priority_rank: int
    # LINE-BY-LINE: `capacity_limit`를 `int` 타입으로 선언합니다. 의미/사용: `DownstreamBay.capacity_limit` 필드/속성입니다. 사용: DownstreamBay 객체를 만들거나 이후 로직에서 참조합니다.
    capacity_limit: int
    # LINE-BY-LINE: `transfer_time_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `transfer_time_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    transfer_time_minutes: float = 0.0
    # LINE-BY-LINE: `release_delay_minutes` 변수에 `None` 결과를 저장합니다. 의미: `release_delay_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    release_delay_minutes: Optional[float] = None
    # LINE-BY-LINE: `bay_type` 변수에 `None` 결과를 저장합니다. 의미: `bay_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_type: Optional[str] = None
    # LINE-BY-LINE: `storage_capacity_length` 변수에 `None` 결과를 저장합니다. 의미: `storage_capacity_length` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    storage_capacity_length: Optional[float] = None
    # LINE-BY-LINE: `storage_capacity_area` 변수에 `None` 결과를 저장합니다. 의미: `storage_capacity_area` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    storage_capacity_area: Optional[float] = None
    # LINE-BY-LINE: `storage_capacity_weight` 변수에 `None` 결과를 저장합니다. 의미: `storage_capacity_weight` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    storage_capacity_weight: Optional[float] = None
    # LINE-BY-LINE: `priority_tiers_by_family` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `priority_tiers_by_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    priority_tiers_by_family: Dict[str, int] = field(default_factory=dict)


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `AuxiliaryResource` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class AuxiliaryResource:
    """크레인, LM, 정반 같은 단순 슬롯 자원.

    현재 1차 범위에서는 대부분 비어 있지만, 향후 resource-flow DES에서 사용한다.
    """

    # LINE-BY-LINE: `resource_id`를 `str` 타입으로 선언합니다. 의미/사용: `AuxiliaryResource.resource_id` 필드/속성입니다. 사용: AuxiliaryResource 객체를 만들거나 이후 로직에서 참조합니다.
    resource_id: str
    # LINE-BY-LINE: `capacity`를 `int` 타입으로 선언합니다. 의미/사용: `AuxiliaryResource.capacity` 필드/속성입니다. 사용: AuxiliaryResource 객체를 만들거나 이후 로직에서 참조합니다.
    capacity: int
    # LINE-BY-LINE: `resource_type` 변수에 `None` 결과를 저장합니다. 의미: `resource_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    resource_type: Optional[str] = None
    # LINE-BY-LINE: `bay_id` 변수에 `None` 결과를 저장합니다. 의미: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: Optional[str] = None
    # LINE-BY-LINE: `transfer_speed_m_per_min` 변수에 `None` 결과를 저장합니다. 의미: `transfer_speed_m_per_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    transfer_speed_m_per_min: Optional[float] = None
    # LINE-BY-LINE: `position` 변수에 `None` 결과를 저장합니다. 의미: `position` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    position: Optional[Tuple[float, float]] = None


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `BatchCandidate` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class BatchCandidate:
    """DES state 안에 보관되는 open batch 1개.

    이 class는 `Environment.state.SimulationState.open_batches`에서 사용된다.

    import 흐름:
    - 이 파일에서 `BatchCandidate`를 정의한다.
    - `Environment/state.py`가 `from .data import BatchCandidate`로 import한다.
    - `Environment/simulation.py`의 `CuttingSimulation`이 state 안의 open batch를 읽고 쓴다.

    왜 필요한가:
    - 현업 제약에는 "각 설비에 W/O 3개까지, 길이합 55000 이하"가 있다.
    - 이것은 하루 총량이 아니라 한 설비/정반 위에 동시에 구성되는 작업 묶음으로 보는 것이 안전하다.
    - 따라서 action을 `job@machine` 한 번에 끝내면 이 제약을 정확히 표현하기 어렵다.
    - open batch는 "아직 절단을 시작하지 않았고, 같은 설비에 같이 올릴 W/O를 모으는 중"인 상태다.

    입출력 예시:
    - 입력 action 1: `open_batch:JOB001@PLS21`
      결과 state: `open_batches["B000001"].job_ids == ("JOB001",)`
    - 입력 action 2: `add_to_batch:B000001:JOB002`
      결과 state: `open_batches["B000001"].job_ids == ("JOB001", "JOB002")`
    - 입력 action 3: `close_batch:B000001@PLS21`
      결과 schedule: JOB001과 JOB002가 같은 machine/start/finish window로 기록된다.

    필드 의미:
    - `batch_id`: state 안에서 batch를 찾는 key. 예: B000001
    - `machine_id`: 이 batch가 배정될 설비. open 후에는 바뀌지 않는다.
    - `job_ids`: batch에 들어간 W/O id 목록.
    - `length_sum`: `job_ids`의 `Job.plate_length` 합. 55000 제한 검사에 쓴다.
    - `max_wo_count`, `max_length_sum`: 이 batch에 적용된 한계값을 기록한다.
    - `metadata`: 나중에 nesting id, 작업자, 정반 id 같은 추가 정보를 담는 확장 자리.
    """

    # LINE-BY-LINE: `batch_id`를 `str` 타입으로 선언합니다. 의미/사용: `BatchCandidate.batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
    batch_id: str
    # LINE-BY-LINE: `machine_id`를 `str` 타입으로 선언합니다. 의미/사용: `BatchCandidate.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
    machine_id: str
    # LINE-BY-LINE: `job_ids`를 `Tuple[str, ...]` 타입으로 선언합니다. 의미/사용: `BatchCandidate.job_ids` 필드/속성입니다. 사용: BatchCandidate 객체를 만들거나 이후 로직에서 참조합니다.
    job_ids: Tuple[str, ...]
    # LINE-BY-LINE: `bay_id` 변수에 `None` 결과를 저장합니다. 의미: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: Optional[str] = None
    # LINE-BY-LINE: `status` 변수에 `"candidate"` 결과를 저장합니다. 의미: `status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    status: str = "candidate"
    # LINE-BY-LINE: `start_time` 변수에 `None` 결과를 저장합니다. 의미: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
    start_time: Optional[float] = None
    # LINE-BY-LINE: `finish_time` 변수에 `None` 결과를 저장합니다. 의미: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
    finish_time: Optional[float] = None
    # LINE-BY-LINE: `length_sum` 변수에 `None` 결과를 저장합니다. 의미: `length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    length_sum: Optional[float] = None
    # LINE-BY-LINE: `max_wo_count` 변수에 `None` 결과를 저장합니다. 의미: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_wo_count: Optional[int] = None
    # LINE-BY-LINE: `max_length_sum` 변수에 `None` 결과를 저장합니다. 의미: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_length_sum: Optional[float] = None
    # LINE-BY-LINE: `metadata` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metadata: Dict[str, Any] = field(default_factory=dict)


# LINE-BY-LINE: `@dataclass(order=True)` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
@dataclass(order=True)
# LINE-BY-LINE: `DownstreamEvent` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class DownstreamEvent:
    """후공정 버퍼의 증감 이벤트.

    예:
    - 절단 완료 후 downstream arrival이면 delta=+1
    - 후공정 release 시점이면 delta=-1
    """

    # LINE-BY-LINE: `event_time`를 `float` 타입으로 선언합니다. 의미/사용: `DownstreamEvent.event_time` 필드/속성입니다. 사용: DownstreamEvent 객체를 만들거나 이후 로직에서 참조합니다.
    event_time: float
    # LINE-BY-LINE: `bay_id`를 `str` 타입으로 선언합니다. 의미/사용: `DownstreamEvent.bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
    bay_id: str
    # LINE-BY-LINE: `delta`를 `int` 타입으로 선언합니다. 의미/사용: `DownstreamEvent.delta` 필드/속성입니다. 사용: DownstreamEvent 객체를 만들거나 이후 로직에서 참조합니다.
    delta: int
    # LINE-BY-LINE: `event_type` 변수에 `"arrival"` 결과를 저장합니다. 의미: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
    event_type: str = "arrival"


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `ScheduledOperation` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class ScheduledOperation:
    """스케줄 결과에 기록되는 배정 1건.

    이 객체는 "작업 1건이 어느 설비에서 언제 시작/종료했는가"를 나타낸다.

    event log와의 관계:
    - ScheduledOperation은 결과 table에 가깝다.
    - FactoryEvent는 상태 변화 로그에 가깝다.
    - operation_id를 통해 둘을 연결한다.
    """

    # LINE-BY-LINE: `job_id`를 `str` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
    job_id: str
    # LINE-BY-LINE: `machine_id`를 `str` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
    machine_id: str
    # LINE-BY-LINE: `start_time`를 `float` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
    start_time: float
    # LINE-BY-LINE: `finish_time`를 `float` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
    finish_time: float
    # LINE-BY-LINE: `downstream_bay`를 `str` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.downstream_bay`는 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
    downstream_bay: str
    # LINE-BY-LINE: `estimated_minutes`를 `float` 타입으로 선언합니다. 의미/사용: `ScheduledOperation.estimated_minutes` 필드/속성입니다. 사용: ScheduledOperation 객체를 만들거나 이후 로직에서 참조합니다.
    estimated_minutes: float
    # LINE-BY-LINE: `cut_bay` 변수에 `None` 결과를 저장합니다. 의미: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
    cut_bay: Optional[str] = None
    # LINE-BY-LINE: `processing_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
    processing_minutes: float = 0.0
    # LINE-BY-LINE: `changeover_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    changeover_minutes: float = 0.0
    # LINE-BY-LINE: `blocked_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
    blocked_minutes: float = 0.0
    # LINE-BY-LINE: `downstream_arrival_time` 변수에 `None` 결과를 저장합니다. 의미: `downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_arrival_time: Optional[float] = None
    # LINE-BY-LINE: `downstream_release_time` 변수에 `None` 결과를 저장합니다. 의미: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    downstream_release_time: Optional[float] = None
    # LINE-BY-LINE: `machine_slot_index` 변수에 `0` 결과를 저장합니다. 의미: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_slot_index: int = 0
    # LINE-BY-LINE: `resource_ids` 변수에 `()` 결과를 저장합니다. 의미: `resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    resource_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `batch_id` 변수에 `None` 결과를 저장합니다. 의미: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
    batch_id: Optional[str] = None
    # LINE-BY-LINE: `batch_job_ids` 변수에 `()` 결과를 저장합니다. 의미: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
    batch_job_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `batch_length_sum` 변수에 `None` 결과를 저장합니다. 의미: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
    batch_length_sum: Optional[float] = None
    # LINE-BY-LINE: `stage_minutes` 변수에 `field(default_factory=dict)` 결과를 저장합니다. 의미: `stage_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    stage_minutes: Dict[str, float] = field(default_factory=dict)
