"""Gym 스타일과 유사한 reset/step 인터페이스를 제공하는 환경 래퍼.

이 파일은 "입력 시나리오"와 "실제 시뮬레이션 코어" 사이의 연결 계층입니다.

즉:
- YAML -> 내부 데이터 객체 변환
- 외부에 reset / step 제공
를 담당합니다.

이 파일이 직접 하지 않는 일:
- 후보 action 생성
- 제약 평가
- event log 생성
- reward 계산

위 작업은 모두 `Environment/simulation.py`가 담당한다.
"""

# LINE-BY-LINE: `typing` 모듈에서 `Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict

# LINE-BY-LINE: `.data` 모듈에서 `AuxiliaryResource, DownstreamBay, Job, Machine`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .data import AuxiliaryResource, DownstreamBay, Job, Machine
# LINE-BY-LINE: `.simulation` 모듈에서 `CuttingSimulation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .simulation import CuttingSimulation


# LINE-BY-LINE: `CuttingShopEnvironment` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
class CuttingShopEnvironment:
    """절단 공정 스케줄링 환경.

    역할:
    - 시나리오 파일을 읽어서 내부 데이터 구조로 변환
    - simulation 객체를 감싸서 외부에 reset/step 형태로 제공
    """

    # LINE-BY-LINE: `__init__(self, config: Dict, scenario: Dict)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def __init__(self, config: Dict, scenario: Dict):
        """config/scenario dict를 받아 simulation을 즉시 실행 가능한 상태로 만든다."""

        # LINE-BY-LINE: 현재 객체의 `config` 속성에 `config` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.config = config
        # 시나리오의 raw dict를 내부 dataclass 구조로 바꿉니다.
        # LINE-BY-LINE: 현재 객체의 `jobs` 속성에 `self._build_jobs(scenario["jobs"])` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.jobs = self._build_jobs(scenario["jobs"])
        # LINE-BY-LINE: 현재 객체의 `machines` 속성에 `self._build_machines(scenario["machines"])` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.machines = self._build_machines(scenario["machines"])
        # LINE-BY-LINE: 현재 객체의 `bays` 속성에 `self._build_bays(scenario["bays"])` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.bays = self._build_bays(scenario["bays"])
        # LINE-BY-LINE: 현재 객체의 `resources` 속성에 `self._build_resources(scenario.get("resources", []))` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.resources = self._build_resources(scenario.get("resources", []))

        # 현재는 위치 정보만 보관합니다.
        # 설비 크기, 이동 거리, 상세 레이아웃 모델은 이후 데이터가 들어오면 확장합니다.
        # LINE-BY-LINE: 현재 객체의 `layout` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.layout = {
            # LINE-BY-LINE: `machine["machine_id"]: tuple(machine.get("position", (0.0, 0.0))) for machine in scenario["machin...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            machine["machine_id"]: tuple(machine.get("position", (0.0, 0.0))) for machine in scenario["machines"]
        }

        # LINE-BY-LINE: 현재 객체의 `simulation` 속성에 `CuttingSimulation(` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.simulation = CuttingSimulation(
            # LINE-BY-LINE: `jobs`에 `self.jobs` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            jobs=self.jobs,
            # LINE-BY-LINE: `machines`에 `self.machines` 결과를 저장합니다. 의미/사용: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machines=self.machines,
            # LINE-BY-LINE: `bays`에 `self.bays` 결과를 저장합니다. 의미/사용: `bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bays=self.bays,
            # LINE-BY-LINE: `resources`에 `self.resources` 결과를 저장합니다. 의미/사용: `resources` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            resources=self.resources,
            # LINE-BY-LINE: `layout`에 `self.layout` 결과를 저장합니다. 의미/사용: `layout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            layout=self.layout,
            # LINE-BY-LINE: `config`에 `self.config` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
            config=self.config,
        )

    # LINE-BY-LINE: `_build_jobs(self, raw_jobs)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def _build_jobs(self, raw_jobs):
        """시나리오의 작업 정의를 내부 Job 객체로 변환합니다.

        입력:
        - scenario YAML의 `jobs` list

        출력:
        - `{job_id: Job}` dict

        extra 처리:
        - dataclass에 아직 없는 원본 필드는 버리지 않고 `Job.extra`에 보관한다.
        - 이렇게 해야 현업 데이터 컬럼이 늘어났을 때 원본 정보를 추적할 수 있다.
        """

        # LINE-BY-LINE: `dataclass_keys`에 `{` 결과를 저장합니다. 의미/사용: `dataclass_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        dataclass_keys = {
            # LINE-BY-LINE: 문자열 값 `"job_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "job_id",
            # LINE-BY-LINE: 문자열 값 `"family"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "family",
            # LINE-BY-LINE: 문자열 값 `"thickness"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "thickness",
            # LINE-BY-LINE: 문자열 값 `"plate_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "plate_length",
            # LINE-BY-LINE: 문자열 값 `"downstream_bay"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "downstream_bay",
            # LINE-BY-LINE: 문자열 값 `"priority_weight"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "priority_weight",
            # LINE-BY-LINE: 문자열 값 `"base_stage_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "base_stage_minutes",
            # LINE-BY-LINE: 문자열 값 `"due_date_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "due_date_minutes",
            # LINE-BY-LINE: 문자열 값 `"preferred_machine_types"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "preferred_machine_types",
            # LINE-BY-LINE: 문자열 값 `"required_resource_ids"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "required_resource_ids",
            # LINE-BY-LINE: 문자열 값 `"cut_bay"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "cut_bay",
            # LINE-BY-LINE: 문자열 값 `"block_set_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "block_set_id",
            # LINE-BY-LINE: 문자열 값 `"block_set_key"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "block_set_key",
            # LINE-BY-LINE: 문자열 값 `"source_machine_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "source_machine_id",
            # LINE-BY-LINE: 문자열 값 `"source_cut_bay"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "source_cut_bay",
            # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual_duration_minutes",
            # LINE-BY-LINE: 문자열 값 `"actual_start_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual_start_minutes",
            # LINE-BY-LINE: 문자열 값 `"actual_finish_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual_finish_minutes",
            # LINE-BY-LINE: 문자열 값 `"planned_start_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "planned_start_minutes",
            # LINE-BY-LINE: 문자열 값 `"planned_finish_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "planned_finish_minutes",
            # LINE-BY-LINE: 문자열 값 `"plate_width"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "plate_width",
            # LINE-BY-LINE: 문자열 값 `"cut_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "cut_length",
            # LINE-BY-LINE: 문자열 값 `"marking_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "marking_length",
            # LINE-BY-LINE: 문자열 값 `"bevel_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "bevel_length",
            "bevel_quantity",
            # LINE-BY-LINE: 문자열 값 `"part_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "part_count",
            # LINE-BY-LINE: 문자열 값 `"steel_quantity"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "steel_quantity",
            # LINE-BY-LINE: 문자열 값 `"downstream_due_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "downstream_due_minutes",
            # LINE-BY-LINE: 문자열 값 `"downstream_required_start_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "downstream_required_start_minutes",
            # LINE-BY-LINE: 문자열 값 `"nesting_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "nesting_id",
            # LINE-BY-LINE: 문자열 값 `"batch_group_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "batch_group_id",
            # LINE-BY-LINE: 문자열 값 `"window_group_id"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "window_group_id",
            # LINE-BY-LINE: 문자열 값 `"allowed_machine_ids"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "allowed_machine_ids",
            # LINE-BY-LINE: 문자열 값 `"prohibited_machine_ids"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "prohibited_machine_ids",
            # LINE-BY-LINE: 문자열 값 `"allowed_bay_ids"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "allowed_bay_ids",
            # LINE-BY-LINE: 문자열 값 `"allowed_table_types"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "allowed_table_types",
            # LINE-BY-LINE: 문자열 값 `"machine_priority_tiers"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "machine_priority_tiers",
            # LINE-BY-LINE: 문자열 값 `"bay_priority_tiers"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "bay_priority_tiers",
            # LINE-BY-LINE: 문자열 값 `"downstream_priority_tiers"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "downstream_priority_tiers",
            # LINE-BY-LINE: 문자열 값 `"is_wide_plate"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "is_wide_plate",
            # LINE-BY-LINE: 문자열 값 `"manual_bevel_required"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "manual_bevel_required",
            # LINE-BY-LINE: 문자열 값 `"forming_required"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "forming_required",
            # LINE-BY-LINE: 문자열 값 `"bom_status"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "bom_status",
        }

        # LINE-BY-LINE: `jobs`에 `{}` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs = {}
        # LINE-BY-LINE: `job in raw_jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job in raw_jobs:
            # dataclass가 아직 모르는 필드는 extra에 남긴다.
            # 이 값은 제약이 자동으로 쓰지는 않지만 debug와 후속 확장에 필요하다.
            # LINE-BY-LINE: `extra`에 `{key: value for key, value in job.items() if key not in dataclass_keys}` 결과를 저장합니다. 의미/사용: `extra` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            extra = {key: value for key, value in job.items() if key not in dataclass_keys}
            # LINE-BY-LINE: `jobs[job["job_id"]]`에 `Job(` 결과를 저장합니다. 의미/사용: `jobs[job["job_id"]]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            jobs[job["job_id"]] = Job(
                # LINE-BY-LINE: `job_id`에 `job["job_id"]` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=job["job_id"],
                # LINE-BY-LINE: `family`에 `job["family"]` 결과를 저장합니다. 의미/사용: `family`는 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
                family=job["family"],
                # LINE-BY-LINE: `thickness`에 `job["thickness"]` 결과를 저장합니다. 의미/사용: `thickness`는 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                thickness=job["thickness"],
                # LINE-BY-LINE: `plate_length`에 `job["plate_length"]` 결과를 저장합니다. 의미/사용: `plate_length`는 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 batch 55000 길이합 제약.
                plate_length=job["plate_length"],
                # LINE-BY-LINE: `downstream_bay`에 `job["downstream_bay"]` 결과를 저장합니다. 의미/사용: `downstream_bay`는 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                downstream_bay=job["downstream_bay"],
                # LINE-BY-LINE: `priority_weight`에 `job["priority_weight"]` 결과를 저장합니다. 의미/사용: `priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
                priority_weight=job["priority_weight"],
                # LINE-BY-LINE: `base_stage_minutes`에 `job["base_stage_minutes"]` 결과를 저장합니다. 의미/사용: `base_stage_minutes`는 setup/cut/finish 처리시간 dict입니다. 사용: Job.estimate_total_minutes().
                base_stage_minutes=job["base_stage_minutes"],
                # LINE-BY-LINE: `due_date_minutes`에 `job.get("due_date_minutes")` 결과를 저장합니다. 의미/사용: `due_date_minutes`는 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
                due_date_minutes=job.get("due_date_minutes"),
                # LINE-BY-LINE: `preferred_machine_types`에 `tuple(job.get("preferred_machine_types") or [])` 결과를 저장합니다. 의미/사용: `preferred_machine_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                preferred_machine_types=tuple(job.get("preferred_machine_types") or []),
                # LINE-BY-LINE: `required_resource_ids`에 `tuple(job.get("required_resource_ids") or [])` 결과를 저장합니다. 의미/사용: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                required_resource_ids=tuple(job.get("required_resource_ids") or []),
                # LINE-BY-LINE: `cut_bay`에 `None if job.get("cut_bay") in ("", None) else str(job.get("cut_bay"))` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                cut_bay=None if job.get("cut_bay") in ("", None) else str(job.get("cut_bay")),
                # LINE-BY-LINE: `block_set_id`에 `job.get("block_set_id") or job.get("block_set_key")` 결과를 저장합니다. 의미/사용: `block_set_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                block_set_id=job.get("block_set_id") or job.get("block_set_key"),
                # LINE-BY-LINE: `source_machine_id`에 `job.get("source_machine_id")` 결과를 저장합니다. 의미/사용: `source_machine_id`는 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                source_machine_id=job.get("source_machine_id"),
                # LINE-BY-LINE: `source_cut_bay`에 `None if job.get("source_cut_bay") in ("", None) else str(job.get("source_cut_bay"))` 결과를 저장합니다. 의미/사용: `source_cut_bay`는 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                source_cut_bay=None if job.get("source_cut_bay") in ("", None) else str(job.get("source_cut_bay")),
                # LINE-BY-LINE: `actual_duration_minutes`에 `job.get("actual_duration_minutes")` 결과를 저장합니다. 의미/사용: `actual_duration_minutes`는 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                actual_duration_minutes=job.get("actual_duration_minutes"),
                # LINE-BY-LINE: `actual_start_minutes`에 `job.get("actual_start_minutes")` 결과를 저장합니다. 의미/사용: `actual_start_minutes`는 실적 착수시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
                actual_start_minutes=job.get("actual_start_minutes"),
                # LINE-BY-LINE: `actual_finish_minutes`에 `job.get("actual_finish_minutes")` 결과를 저장합니다. 의미/사용: `actual_finish_minutes`는 실적 종료시각을 scenario 기준 minute로 바꾼 값입니다. 사용: actual replay schedule.
                actual_finish_minutes=job.get("actual_finish_minutes"),
                # LINE-BY-LINE: `planned_start_minutes`에 `job.get("planned_start_minutes")` 결과를 저장합니다. 의미/사용: `planned_start_minutes`는 계획 착수일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
                planned_start_minutes=job.get("planned_start_minutes"),
                # LINE-BY-LINE: `planned_finish_minutes`에 `job.get("planned_finish_minutes")` 결과를 저장합니다. 의미/사용: `planned_finish_minutes`는 계획 종료일을 scenario 기준 minute로 바꾼 값입니다. 사용: 계획일 기반 검증 후보.
                planned_finish_minutes=job.get("planned_finish_minutes"),
                # LINE-BY-LINE: `plate_width`에 `job.get("plate_width")` 결과를 저장합니다. 의미/사용: `plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                plate_width=job.get("plate_width"),
                # LINE-BY-LINE: `cut_length`에 `job.get("cut_length")` 결과를 저장합니다. 의미/사용: `cut_length`는 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
                cut_length=job.get("cut_length"),
                # LINE-BY-LINE: `marking_length`에 `job.get("marking_length")` 결과를 저장합니다. 의미/사용: `marking_length`는 마킹 길이입니다. 사용: tact time 분석 feature.
                marking_length=job.get("marking_length"),
                # LINE-BY-LINE: `bevel_length`에 `job.get("bevel_length")` 결과를 저장합니다. 의미/사용: `bevel_length`는 베벨 길이입니다. 사용: tact time 분석 feature.
                bevel_length=job.get("bevel_length"),
                bevel_quantity=job.get("bevel_quantity"),
                # LINE-BY-LINE: `part_count`에 `job.get("part_count")` 결과를 저장합니다. 의미/사용: `part_count`는 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
                part_count=job.get("part_count"),
                # LINE-BY-LINE: `steel_quantity`에 `job.get("steel_quantity")` 결과를 저장합니다. 의미/사용: `steel_quantity`는 강재 수량입니다. 예: `STL_QTY`, 사용: 데이터 품질/처리시간 feature.
                steel_quantity=job.get("steel_quantity"),
                # LINE-BY-LINE: `downstream_due_minutes`에 `job.get("downstream_due_minutes")` 결과를 저장합니다. 의미/사용: `downstream_due_minutes`는 후공정 요구시각입니다. 사용: downstream_due_date hard rule 후보.
                downstream_due_minutes=job.get("downstream_due_minutes"),
                # LINE-BY-LINE: `downstream_required_start_minutes`에 `job.get("downstream_required_start_minutes")` 결과를 저장합니다. 의미/사용: `downstream_required_start_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                downstream_required_start_minutes=job.get("downstream_required_start_minutes"),
                # LINE-BY-LINE: `nesting_id`에 `job.get("nesting_id")` 결과를 저장합니다. 의미/사용: `nesting_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                nesting_id=job.get("nesting_id"),
                # LINE-BY-LINE: `batch_group_id`에 `job.get("batch_group_id")` 결과를 저장합니다. 의미/사용: `batch_group_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                batch_group_id=job.get("batch_group_id"),
                # LINE-BY-LINE: `window_group_id`에 `job.get("window_group_id")` 결과를 저장합니다. 의미/사용: `window_group_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                window_group_id=job.get("window_group_id"),
                # LINE-BY-LINE: `allowed_machine_ids`에 `tuple(job.get("allowed_machine_ids") or [])` 결과를 저장합니다. 의미/사용: `allowed_machine_ids`는 W/O별 허용 설비 whitelist입니다. 예: `("PLS21",)`, 사용: allowed_machine_ids rule.
                allowed_machine_ids=tuple(job.get("allowed_machine_ids") or []),
                # LINE-BY-LINE: `prohibited_machine_ids`에 `tuple(job.get("prohibited_machine_ids") or [])` 결과를 저장합니다. 의미/사용: `prohibited_machine_ids`는 W/O별 금지 설비 blacklist입니다. 사용: prohibited_machine_ids rule.
                prohibited_machine_ids=tuple(job.get("prohibited_machine_ids") or []),
                # LINE-BY-LINE: `allowed_bay_ids`에 `tuple(str(bay_id) for bay_id in (job.get("allowed_bay_ids") or []))` 결과를 저장합니다. 의미/사용: `allowed_bay_ids`는 W/O별 허용 절단 Bay whitelist입니다. 예: `("22",)`, 사용: allowed_bay_ids rule.
                allowed_bay_ids=tuple(str(bay_id) for bay_id in (job.get("allowed_bay_ids") or [])),
                # LINE-BY-LINE: `allowed_table_types`에 `tuple(job.get("allowed_table_types") or [])` 결과를 저장합니다. 의미/사용: `allowed_table_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                allowed_table_types=tuple(job.get("allowed_table_types") or []),
                # LINE-BY-LINE: `machine_priority_tiers`에 `{` 결과를 저장합니다. 의미/사용: `machine_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_priority_tiers={
                    # LINE-BY-LINE: `str(machine_id): int(tier)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    str(machine_id): int(tier)
                    # LINE-BY-LINE: `machine_id, tier in (job.get("machine_priority_tiers", {}) or {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for machine_id, tier in (job.get("machine_priority_tiers", {}) or {}).items()
                },
                # LINE-BY-LINE: `bay_priority_tiers`에 `{` 결과를 저장합니다. 의미/사용: `bay_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bay_priority_tiers={
                    # LINE-BY-LINE: `str(bay_id): int(tier)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    str(bay_id): int(tier)
                    # LINE-BY-LINE: `bay_id, tier in (job.get("bay_priority_tiers", {}) or {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for bay_id, tier in (job.get("bay_priority_tiers", {}) or {}).items()
                },
                # LINE-BY-LINE: `downstream_priority_tiers`에 `{` 결과를 저장합니다. 의미/사용: `downstream_priority_tiers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                downstream_priority_tiers={
                    # LINE-BY-LINE: `str(bay_id): int(tier)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    str(bay_id): int(tier)
                    # LINE-BY-LINE: `bay_id, tier in (job.get("downstream_priority_tiers", {}) or {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for bay_id, tier in (job.get("downstream_priority_tiers", {}) or {}).items()
                },
                # LINE-BY-LINE: `is_wide_plate`에 `job.get("is_wide_plate")` 결과를 저장합니다. 의미/사용: `is_wide_plate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                is_wide_plate=job.get("is_wide_plate"),
                # LINE-BY-LINE: `manual_bevel_required`에 `job.get("manual_bevel_required")` 결과를 저장합니다. 의미/사용: `manual_bevel_required` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                manual_bevel_required=job.get("manual_bevel_required"),
                # LINE-BY-LINE: `forming_required`에 `job.get("forming_required")` 결과를 저장합니다. 의미/사용: `forming_required` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                forming_required=job.get("forming_required"),
                # LINE-BY-LINE: `bom_status`에 `job.get("bom_status")` 결과를 저장합니다. 의미/사용: `bom_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bom_status=job.get("bom_status"),
                # LINE-BY-LINE: `extra`에 `extra` 결과를 저장합니다. 의미/사용: `extra` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                extra=extra,
            )
        # LINE-BY-LINE: 호출자에게 `jobs`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return jobs

    # LINE-BY-LINE: `_build_machines(self, raw_machines)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def _build_machines(self, raw_machines):
        """시나리오의 설비 정의를 내부 Machine 객체로 변환합니다.

        입력:
        - scenario YAML의 `machines` list

        출력:
        - `{machine_id: Machine}` dict

        핵심:
        - `bay_id`는 필수에 가깝다. 설비를 고르면 Bay가 같이 정해져야 하기 때문이다.
        - batch_open 모드에서는 batch가 설비 slot 1개를 점유한다.
        - rolling_capacity는 legacy/debug opt-in일 때만 max W/O count를 slot 수로 본다.
        """

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `machine["machine_id"]: Machine(`를 실행합니다. 의미/사용: scenario machine dict를 환경 내부 Machine dataclass로 변환합니다.
            machine["machine_id"]: Machine(
                # LINE-BY-LINE: `machine_id`에 `machine["machine_id"]` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id=machine["machine_id"],
                # LINE-BY-LINE: `machine_type`에 `machine["machine_type"]` 결과를 저장합니다. 의미/사용: `machine_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_type=machine["machine_type"],
                # LINE-BY-LINE: `enabled`에 `machine["enabled"]` 결과를 저장합니다. 의미/사용: `enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                enabled=machine["enabled"],
                # LINE-BY-LINE: `eligible_families`에 `machine["eligible_families"]` 결과를 저장합니다. 의미/사용: `eligible_families` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                eligible_families=machine["eligible_families"],
                # LINE-BY-LINE: `min_thickness`에 `machine["min_thickness"]` 결과를 저장합니다. 의미/사용: `min_thickness` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                min_thickness=machine["min_thickness"],
                # LINE-BY-LINE: `max_thickness`에 `machine["max_thickness"]` 결과를 저장합니다. 의미/사용: `max_thickness` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                max_thickness=machine["max_thickness"],
                # LINE-BY-LINE: `table_length_limit`에 `machine["table_length_limit"]` 결과를 저장합니다. 의미/사용: `table_length_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                table_length_limit=machine["table_length_limit"],
                # LINE-BY-LINE: `cut_speed_factor`에 `machine["cut_speed_factor"]` 결과를 저장합니다. 의미/사용: `cut_speed_factor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                cut_speed_factor=machine["cut_speed_factor"],
                # LINE-BY-LINE: `daily_capacity_minutes`에 `machine["daily_capacity_minutes"]` 결과를 저장합니다. 의미/사용: `daily_capacity_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                daily_capacity_minutes=machine["daily_capacity_minutes"],
                # LINE-BY-LINE: `parallel_capacity`에 `int(machine.get("parallel_capacity", 1))` 결과를 저장합니다. 의미/사용: `parallel_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                parallel_capacity=int(machine.get("parallel_capacity", 1)),
                # LINE-BY-LINE: `bay_id`에 `None if machine.get("bay_id") in ("", None) else str(machine.get("bay_id"))` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=None if machine.get("bay_id") in ("", None) else str(machine.get("bay_id")),
                # LINE-BY-LINE: `required_resource_ids`에 `tuple(machine.get("required_resource_ids") or [])` 결과를 저장합니다. 의미/사용: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                required_resource_ids=tuple(machine.get("required_resource_ids") or []),
                # LINE-BY-LINE: `position`에 `tuple(machine.get("position", (0.0, 0.0)))` 결과를 저장합니다. 의미/사용: `position` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                position=tuple(machine.get("position", (0.0, 0.0))),
                # LINE-BY-LINE: `equipment_type`에 `machine.get("equipment_type")` 결과를 저장합니다. 의미/사용: `equipment_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                equipment_type=machine.get("equipment_type"),
                # LINE-BY-LINE: `table_type`에 `machine.get("table_type")` 결과를 저장합니다. 의미/사용: `table_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                table_type=machine.get("table_type"),
                # LINE-BY-LINE: `max_batch_wo_count`에 `machine.get("max_batch_wo_count")` 결과를 저장합니다. 의미/사용: `max_batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                max_batch_wo_count=machine.get("max_batch_wo_count"),
                # LINE-BY-LINE: `max_batch_length_sum`에 `machine.get("max_batch_length_sum")` 결과를 저장합니다. 의미/사용: `max_batch_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                max_batch_length_sum=machine.get("max_batch_length_sum"),
                # LINE-BY-LINE: `max_plate_width`에 `machine.get("max_plate_width")` 결과를 저장합니다. 의미/사용: `max_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                max_plate_width=machine.get("max_plate_width"),
                # LINE-BY-LINE: `min_plate_width`에 `machine.get("min_plate_width")` 결과를 저장합니다. 의미/사용: `min_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                min_plate_width=machine.get("min_plate_width"),
                # LINE-BY-LINE: `priority_tiers_by_family`에 `{` 결과를 저장합니다. 의미/사용: `priority_tiers_by_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                priority_tiers_by_family={
                    # LINE-BY-LINE: `str(family): int(tier)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    str(family): int(tier)
                    # LINE-BY-LINE: `family, tier in (machine.get("priority_tiers_by_family", {}) or {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for family, tier in (machine.get("priority_tiers_by_family", {}) or {}).items()
                },
                # LINE-BY-LINE: `install_start_date`에 `machine.get("install_start_date")` 결과를 저장합니다. 의미/사용: `install_start_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                install_start_date=machine.get("install_start_date"),
                # LINE-BY-LINE: `install_end_date`에 `machine.get("install_end_date")` 결과를 저장합니다. 의미/사용: `install_end_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                install_end_date=machine.get("install_end_date"),
                # LINE-BY-LINE: `calendar_id`에 `machine.get("calendar_id")` 결과를 저장합니다. 의미/사용: `calendar_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                calendar_id=machine.get("calendar_id"),
            )
            # LINE-BY-LINE: `machine in raw_machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for machine in raw_machines
        }

    # LINE-BY-LINE: `_build_bays(self, raw_bays)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def _build_bays(self, raw_bays):
        """시나리오의 후공정 베이 정의를 내부 객체로 변환합니다.

        이 bay는 절단 Bay가 아니라 downstream/적치 Bay다.
        절단 Bay 정보는 `Machine.bay_id`와 `cut_bays` 쪽에 있다.
        """

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `bay["bay_id"]: DownstreamBay(`를 실행합니다. 의미/사용: scenario bay dict를 후공정/적치 Bay dataclass로 변환합니다.
            bay["bay_id"]: DownstreamBay(
                # LINE-BY-LINE: `bay_id`에 `bay["bay_id"]` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=bay["bay_id"],
                # LINE-BY-LINE: `priority_rank`에 `bay["priority_rank"]` 결과를 저장합니다. 의미/사용: `priority_rank` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                priority_rank=bay["priority_rank"],
                # LINE-BY-LINE: `capacity_limit`에 `bay["capacity_limit"]` 결과를 저장합니다. 의미/사용: `capacity_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                capacity_limit=bay["capacity_limit"],
                # LINE-BY-LINE: `transfer_time_minutes`에 `float(bay.get("transfer_time_minutes", 0.0))` 결과를 저장합니다. 의미/사용: `transfer_time_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                transfer_time_minutes=float(bay.get("transfer_time_minutes", 0.0)),
                # LINE-BY-LINE: `release_delay_minutes`에 `(` 결과를 저장합니다. 의미/사용: `release_delay_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                release_delay_minutes=(
                    # LINE-BY-LINE: `None if bay.get("release_delay_minutes") is None else float(bay.get("release_delay_minutes"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    None if bay.get("release_delay_minutes") is None else float(bay.get("release_delay_minutes"))
                ),
                # LINE-BY-LINE: `bay_type`에 `bay.get("bay_type")` 결과를 저장합니다. 의미/사용: `bay_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bay_type=bay.get("bay_type"),
                # LINE-BY-LINE: `storage_capacity_length`에 `bay.get("storage_capacity_length")` 결과를 저장합니다. 의미/사용: `storage_capacity_length` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                storage_capacity_length=bay.get("storage_capacity_length"),
                # LINE-BY-LINE: `storage_capacity_area`에 `bay.get("storage_capacity_area")` 결과를 저장합니다. 의미/사용: `storage_capacity_area` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                storage_capacity_area=bay.get("storage_capacity_area"),
                # LINE-BY-LINE: `storage_capacity_weight`에 `bay.get("storage_capacity_weight")` 결과를 저장합니다. 의미/사용: `storage_capacity_weight` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                storage_capacity_weight=bay.get("storage_capacity_weight"),
                # LINE-BY-LINE: `priority_tiers_by_family`에 `{` 결과를 저장합니다. 의미/사용: `priority_tiers_by_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                priority_tiers_by_family={
                    # LINE-BY-LINE: `str(family): int(tier)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    str(family): int(tier)
                    # LINE-BY-LINE: `family, tier in (bay.get("priority_tiers_by_family", {}) or {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for family, tier in (bay.get("priority_tiers_by_family", {}) or {}).items()
                },
            )
            # LINE-BY-LINE: `bay in raw_bays` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for bay in raw_bays
        }

    # LINE-BY-LINE: `_build_resources(self, raw_resources)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def _build_resources(self, raw_resources):
        """시나리오의 보조자원 정의를 내부 객체로 변환합니다.

        현재 1차 NP 검증에는 보조자원이 거의 없지만,
        크레인/LM/정반 같은 resource-flow DES로 확장할 때 이 dict가 사용된다.
        """

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `resource["resource_id"]: AuxiliaryResource(`를 실행합니다. 의미/사용: resource dict를 크레인/LM 같은 보조자원 dataclass로 변환합니다.
            resource["resource_id"]: AuxiliaryResource(
                # LINE-BY-LINE: `resource_id`에 `resource["resource_id"]` 결과를 저장합니다. 의미/사용: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                resource_id=resource["resource_id"],
                # LINE-BY-LINE: `capacity`에 `int(resource["capacity"])` 결과를 저장합니다. 의미/사용: `capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                capacity=int(resource["capacity"]),
                # LINE-BY-LINE: `resource_type`에 `resource.get("resource_type")` 결과를 저장합니다. 의미/사용: `resource_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                resource_type=resource.get("resource_type"),
                # LINE-BY-LINE: `bay_id`에 `None if resource.get("bay_id") in ("", None) else str(resource.get("bay_id"))` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=None if resource.get("bay_id") in ("", None) else str(resource.get("bay_id")),
                # LINE-BY-LINE: `transfer_speed_m_per_min`에 `resource.get("transfer_speed_m_per_min")` 결과를 저장합니다. 의미/사용: `transfer_speed_m_per_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                transfer_speed_m_per_min=resource.get("transfer_speed_m_per_min"),
                # LINE-BY-LINE: `position`에 `(` 결과를 저장합니다. 의미/사용: `position` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                position=(
                    # LINE-BY-LINE: `None` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    None
                    # LINE-BY-LINE: 조건 `resource.get("position") in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if resource.get("position") in (None, "")
                    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                    else tuple(resource.get("position"))
                ),
            )
            # LINE-BY-LINE: `resource in raw_resources` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for resource in raw_resources
        }

    # LINE-BY-LINE: `reset(self)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def reset(self):
        """환경 초기화.

        반환 형식은 Gym 스타일을 따라
        `(observation, info)` 형태로 둡니다.
        """

        # LINE-BY-LINE: `self.simulation.reset()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.simulation.reset()
        # LINE-BY-LINE: 호출자에게 `self.simulation.build_observation(), {"message": "reset complete"}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.simulation.build_observation(), {"message": "reset complete"}

    # LINE-BY-LINE: `get_action_candidates(self)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def get_action_candidates(self):
        """현재 가능한 action 후보 목록을 반환합니다."""

        # LINE-BY-LINE: 호출자에게 `self.simulation.get_candidates()`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.simulation.get_candidates()

    # LINE-BY-LINE: `step(self, action_id: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def step(self, action_id: str):
        """선택한 action을 환경에 적용합니다."""

        # LINE-BY-LINE: 호출자에게 `self.simulation.step_action(action_id)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.simulation.step_action(action_id)

    # LINE-BY-LINE: `run_with_policy(self, policy_fn)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: scenario YAML을 dataclass로 바꿔 simulation에 넘기는 단계에서 사용됩니다.
    def run_with_policy(self, policy_fn):
        """휴리스틱 또는 간단 정책 함수로 한 에피소드를 끝까지 실행합니다."""

        # LINE-BY-LINE: 호출자에게 `self.simulation.run(policy_fn)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.simulation.run(policy_fn)
