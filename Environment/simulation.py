"""절단 공정 스케줄링 시뮬레이션 코어.

이 파일이 사실상 환경의 중심입니다.

역할:
- 현재 시뮬레이션 시간이 어디인지 관리
- 어떤 job-machine pair가 가능한지 계산
- 하드/소프트 제약을 평가
- action을 적용해서 상태를 전진
- observation을 RL 네트워크가 읽을 수 있는 dict로 변환

초보자 입장에서 가장 중요한 포인트는 두 가지입니다.

1. `get_candidates()`
   - 지금 선택 가능한 action 후보를 만드는 함수

2. `step_action(action_id)`
   - 선택한 action 하나를 실제로 적용하는 함수
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass, field`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass, field
# LINE-BY-LINE: `datetime` 모듈에서 `datetime, timedelta`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from datetime import datetime, timedelta
# LINE-BY-LINE: `typing` 모듈에서 `Callable, Dict, List, Optional, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Callable, Dict, List, Optional, Tuple

# LINE-BY-LINE: `.constraints` 모듈에서 `ConstraintContext, ConstraintManager`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .constraints import ConstraintContext, ConstraintManager
# LINE-BY-LINE: `.constraints.registry` 모듈에서 `CandidateConstraintBundle`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .constraints.registry import CandidateConstraintBundle
# LINE-BY-LINE: `.data` 모듈에서 `AuxiliaryResource, BatchCandidate, DownstreamBay, DownstreamEvent, Job, Machine, ScheduledOperation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .data import AuxiliaryResource, BatchCandidate, DownstreamBay, DownstreamEvent, Job, Machine, ScheduledOperation
# LINE-BY-LINE: `.events` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .events import (
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_CUT_BAY_ASSIGN,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_DECISION_EPOCH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DECISION_EPOCH,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_DOWNSTREAM_ARRIVAL` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DOWNSTREAM_ARRIVAL,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_DOWNSTREAM_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_DOWNSTREAM_RELEASE,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_MACHINE_ASSIGN,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_FINISH,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_START,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_RESOURCE_ACQUIRE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_RESOURCE_ACQUIRE,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_RESOURCE_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_RESOURCE_RELEASE,
    # LINE-BY-LINE: `import(...)` 호출에 `FactoryEvent` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    FactoryEvent,
    # LINE-BY-LINE: `import(...)` 호출에 `SOURCE_GENERATED` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_GENERATED,
    # LINE-BY-LINE: `import(...)` 호출에 `event_sort_key` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    event_sort_key,
    # LINE-BY-LINE: `import(...)` 호출에 `validate_event_required_fields` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    validate_event_required_fields,
)
# LINE-BY-LINE: `.reward` 모듈에서 `compute_load_imbalance, compute_step_reward`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .reward import compute_load_imbalance, compute_step_reward
# LINE-BY-LINE: `.state` 모듈에서 `SimulationState`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .state import SimulationState


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `ActionCandidate` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
class ActionCandidate:
    """현재 step에서 선택 가능한 action 1개.

    이 class는 `CuttingSimulation.get_candidates()`의 출력이다.

    import 흐름:
    - `ActionCandidate`는 이 파일(`Environment/simulation.py`) 안에 정의되어 있다.
    - `Environment/environment.py`의 `CuttingShopEnvironment.get_action_candidates()`가 그대로 반환한다.
    - `Agent/heuristics.py`의 `select_action_by_rule()`가 후보 리스트를 받아 하나를 고른다.
    - 선택된 후보는 `CuttingSimulation.step_candidate()` 또는 `step_action(action_id)`로 들어온다.

    action_id 예시:
    - 단건 dispatch: `J0001@PLS21`
    - open batch 시작: `open_batch:J0001@PLS21`
    - open batch에 W/O 추가: `add_to_batch:B000001:J0002`
    - open batch 절단 시작: `close_batch:B000001@PLS21`

    입력/출력 구조:
    - 입력: 현재 state, Job, Machine, config 제약을 조합해서 만든 후보 action
    - 출력: step에서 실제 state 변경에 필요한 최소 정보

    batch 관련 필드:
    - `action_kind`: 단건인지, open/add/close batch인지 구분한다.
    - `batch_id`: 이미 열린 batch를 참조할 때 사용한다.
    - `batch_job_ids`: 이 action이 최종적으로 포함하는 W/O 목록이다.
    - `batch_wo_count`, `batch_length_sum`: 3 W/O / 55000 제약 확인에 사용한다.

    하위 호환:
    - 기존 휴리스틱은 `candidate.job`, `candidate.machine`, `candidate.estimated_minutes`만 봐도 동작한다.
    - batch 후보에서도 `job`은 대표 W/O 또는 이번 action으로 추가되는 W/O로 채운다.
    """

    # LINE-BY-LINE: `action_id`를 `str` 타입으로 선언합니다. 의미/사용: `ActionCandidate.action_id` 필드/속성입니다. 사용: ActionCandidate 객체를 만들거나 이후 로직에서 참조합니다.
    action_id: str
    # LINE-BY-LINE: `job`를 `Job` 타입으로 선언합니다. 의미/사용: `ActionCandidate.job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
    job: Job
    # LINE-BY-LINE: `machine`를 `Machine` 타입으로 선언합니다. 의미/사용: `ActionCandidate.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
    machine: Machine
    # LINE-BY-LINE: `estimated_minutes`를 `float` 타입으로 선언합니다. 의미/사용: `ActionCandidate.estimated_minutes` 필드/속성입니다. 사용: ActionCandidate 객체를 만들거나 이후 로직에서 참조합니다.
    estimated_minutes: float
    # LINE-BY-LINE: `processing_minutes`를 `float` 타입으로 선언합니다. 의미/사용: `ActionCandidate.processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
    processing_minutes: float
    # LINE-BY-LINE: `finish_time`를 `float` 타입으로 선언합니다. 의미/사용: `ActionCandidate.finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
    finish_time: float
    # LINE-BY-LINE: `downstream_arrival_time`를 `float` 타입으로 선언합니다. 의미/사용: `ActionCandidate.downstream_arrival_time` 필드/속성입니다. 사용: ActionCandidate 객체를 만들거나 이후 로직에서 참조합니다.
    downstream_arrival_time: float
    # LINE-BY-LINE: `downstream_release_time`를 `Optional[float]` 타입으로 선언합니다. 의미/사용: `ActionCandidate.downstream_release_time` 필드/속성입니다. 사용: ActionCandidate 객체를 만들거나 이후 로직에서 참조합니다.
    downstream_release_time: Optional[float]
    # LINE-BY-LINE: `changeover_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    changeover_minutes: float = 0.0
    # LINE-BY-LINE: `blocked_minutes` 변수에 `0.0` 결과를 저장합니다. 의미: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
    blocked_minutes: float = 0.0
    # LINE-BY-LINE: `required_resource_ids` 변수에 `()` 결과를 저장합니다. 의미: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_resource_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `hard_reasons` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `hard_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hard_reasons: List[str] = field(default_factory=list)
    # LINE-BY-LINE: `soft_reasons` 변수에 `field(default_factory=list)` 결과를 저장합니다. 의미: `soft_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    soft_reasons: List[str] = field(default_factory=list)
    # LINE-BY-LINE: `soft_penalty` 변수에 `0.0` 결과를 저장합니다. 의미: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    soft_penalty: float = 0.0
    # LINE-BY-LINE: `action_kind` 변수에 `"single_dispatch"` 결과를 저장합니다. 의미: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    action_kind: str = "single_dispatch"
    # LINE-BY-LINE: `batch_id` 변수에 `None` 결과를 저장합니다. 의미: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
    batch_id: Optional[str] = None
    # LINE-BY-LINE: `batch_job_ids` 변수에 `()` 결과를 저장합니다. 의미: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
    batch_job_ids: Tuple[str, ...] = ()
    # LINE-BY-LINE: `batch_wo_count` 변수에 `1` 결과를 저장합니다. 의미: `batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_wo_count: int = 1
    # LINE-BY-LINE: `batch_length_sum` 변수에 `0.0` 결과를 저장합니다. 의미: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
    batch_length_sum: float = 0.0
    # LINE-BY-LINE: `batch_processing_rule` 변수에 `"single_job"` 결과를 저장합니다. 의미: `batch_processing_rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_processing_rule: str = "single_job"


# LINE-BY-LINE: `CuttingSimulation` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
class CuttingSimulation:
    """PMSP형 절단 스케줄링 시뮬레이션.

    현재 버전의 핵심 설계:
    - action = 작업-설비 쌍 1개
    - 하드 제약은 마스킹
    - 소프트 제약은 penalty
    - 가능한 액션이 없으면 다음 완료 시점까지 자동 전진
    """

    # LINE-BY-LINE: `__init__` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def __init__(
        # LINE-BY-LINE: `__init__(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `jobs`를 `Dict[str, Job],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.jobs` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        jobs: Dict[str, Job],
        # LINE-BY-LINE: `machines`를 `Dict[str, Machine],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machines` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        machines: Dict[str, Machine],
        # LINE-BY-LINE: `bays`를 `Dict[str, DownstreamBay],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.bays` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        bays: Dict[str, DownstreamBay],
        # LINE-BY-LINE: `resources`를 `Dict[str, AuxiliaryResource],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.resources` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        resources: Dict[str, AuxiliaryResource],
        # LINE-BY-LINE: `__init__(...)` 호출에 `layout` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        layout,
        # LINE-BY-LINE: `config`를 `Dict,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
        config: Dict,
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        """시뮬레이션 코어를 초기화한다.

        입력:
        - jobs/machines/bays/resources: `Environment/environment.py`가 만든 dataclass dict
        - layout: 현재는 machine 위치 dict
        - config: constraints/action_space/calendar/reward 설정 전체

        출력:
        - 반환값은 없고, 내부 `self.state`를 episode 시작 상태로 만든다.
        """

        # LINE-BY-LINE: 현재 객체의 `jobs` 속성에 `jobs` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.jobs = jobs
        # LINE-BY-LINE: 현재 객체의 `machines` 속성에 `machines` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.machines = machines
        # LINE-BY-LINE: 현재 객체의 `bays` 속성에 `bays` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.bays = bays
        # LINE-BY-LINE: 현재 객체의 `resources` 속성에 `resources` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.resources = resources
        # LINE-BY-LINE: 현재 객체의 `layout` 속성에 `layout` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.layout = layout
        # LINE-BY-LINE: 현재 객체의 `config` 속성에 `config` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.config = config
        # LINE-BY-LINE: 현재 객체의 `minutes_per_day` 속성에 `int(config["simulation"].get("minutes_per_day", 1440))` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.minutes_per_day = int(config["simulation"].get("minutes_per_day", 1440))
        # LINE-BY-LINE: 현재 객체의 `start_date` 속성에 `datetime.strptime(config["simulation"].get("start_date", "2026-01-01"), "%Y-%m-%d").date()` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.start_date = datetime.strptime(config["simulation"].get("start_date", "2026-01-01"), "%Y-%m-%d").date()
        # LINE-BY-LINE: 현재 객체의 `category_flags` 속성에 `config["constraints"].get("categories", {})` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.category_flags = config["constraints"].get("categories", {})
        # LINE-BY-LINE: 현재 객체의 `override_config` 속성에 `config["constraints"].get("overrides", {})` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.override_config = config["constraints"].get("overrides", {})
        # LINE-BY-LINE: 현재 객체의 `family_to_index` 속성에 `{name: idx for idx, name in enumerate(sorted({job.family for job in jobs.values()}))}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.family_to_index = {name: idx for idx, name in enumerate(sorted({job.family for job in jobs.values()}))}
        # LINE-BY-LINE: 현재 객체의 `machine_type_to_index` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.machine_type_to_index = {
            # LINE-BY-LINE: `name`를 `idx for idx, name in enumerate(sorted({machine.machine_type for machine in machines.values()}))` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.name` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
            name: idx for idx, name in enumerate(sorted({machine.machine_type for machine in machines.values()}))
        }
        # LINE-BY-LINE: 현재 객체의 `bay_to_index` 속성에 `{name: idx for idx, name in enumerate(sorted(bays.keys()))}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.bay_to_index = {name: idx for idx, name in enumerate(sorted(bays.keys()))}
        # LINE-BY-LINE: 현재 객체의 `constraint_manager` 속성에 `ConstraintManager(` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.constraint_manager = ConstraintManager(
            # LINE-BY-LINE: `hard_enabled`에 `config["constraints"]["hard_enabled"]` 결과를 저장합니다. 의미/사용: `hard_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            hard_enabled=config["constraints"]["hard_enabled"],
            # LINE-BY-LINE: `soft_enabled`에 `config["constraints"]["soft_enabled"]` 결과를 저장합니다. 의미/사용: `soft_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_enabled=config["constraints"]["soft_enabled"],
            # LINE-BY-LINE: `soft_weights`에 `config["constraints"]["soft_penalty_weights"]` 결과를 저장합니다. 의미/사용: `soft_weights` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_weights=config["constraints"]["soft_penalty_weights"],
            # LINE-BY-LINE: `category_flags`에 `self.category_flags` 결과를 저장합니다. 의미/사용: `category_flags` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            category_flags=self.category_flags,
        )
        # LINE-BY-LINE: 현재 객체의 `_spt_job_order_by_machine: Dict[str, List[str]]` 속성에 `{}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self._spt_job_order_by_machine: Dict[str, List[str]] = {}
        # LINE-BY-LINE: 현재 객체의 `state` 속성에 `self._build_initial_state()` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state = self._build_initial_state()

    def _validate_action_space_consistency(self) -> None:
        """현재 config의 action mode가 절단 설비 capacity 해석과 충돌하지 않는지 확인한다.

        현업 추가 확인 후 현재 과제의 기본 해석은 rolling이 아니라 batch다.
        즉 W/O 1~3개가 batch로 같이 들어오고 같이 빠진다.

        따라서 기본 실행은 `batch_open`이어야 한다.
        `job_machine_pair`는 단건 baseline/debug일 때만 허용한다.
        `rolling_capacity`는 과거 가정 검토용 legacy mode이므로 명시 opt-in 없이는 막는다.
        """

        mode = self._action_mode()
        action_config = self.config.get("action_space", {}) or {}
        if mode == "rolling_capacity" and not bool(action_config.get("allow_legacy_rolling_capacity", False)):
            print(
                "[ERROR][CuttingSimulation._validate_action_space_consistency] "
                "cause=rolling_capacity_conflicts_with_confirmed_batch_mode "
                f"mode={mode} allow_legacy_rolling_capacity=false "
                "fix=set action_space.mode=batch_open or explicitly set "
                "action_space.allow_legacy_rolling_capacity=true for legacy/debug"
            )
            raise RuntimeError(
                "rolling_capacity is blocked because 현업 confirmed batch mode; "
                "use batch_open or set allow_legacy_rolling_capacity=true explicitly"
            )

        if mode != "job_machine_pair":
            return

        allow_single_slot_baseline = bool(action_config.get("allow_single_slot_baseline", False))
        if allow_single_slot_baseline:
            return

        hard_enabled = self.config.get("constraints", {}).get("hard_enabled", {}) or {}
        capacity_rules_enabled = bool(
            hard_enabled.get("machine_day_wo_count_limit", False)
            or hard_enabled.get("machine_day_length_sum_limit", False)
            or hard_enabled.get("batch_wo_count_limit", False)
            or hard_enabled.get("batch_length_sum_limit", False)
        )
        if not capacity_rules_enabled:
            return

        max_wo_limits: List[int] = []
        for machine in self.machines.values():
            machine_limit = machine.max_batch_wo_count
            if machine_limit is None:
                machine_limit = self._get_machine_day_wo_count_limit(machine.machine_id)
            if machine_limit is not None:
                max_wo_limits.append(int(machine_limit))

        max_wo_count = max(max_wo_limits or [1])
        if max_wo_count <= 1:
            return

        print(
            "[ERROR][CuttingSimulation._validate_action_space_consistency] "
            "cause=job_machine_pair_conflicts_with_batch_capacity "
            f"mode={mode} max_wo_count={max_wo_count} "
            "allow_single_slot_baseline=false "
            "fix=set action_space.mode=batch_open or explicitly set "
            "action_space.allow_single_slot_baseline=true for baseline/debug"
        )
        raise RuntimeError(
            "job_machine_pair is blocked because batch capacity limits are active; "
            "use batch_open or set allow_single_slot_baseline=true explicitly"
        )

    # LINE-BY-LINE: `_build_initial_state(self)` 함수를 정의합니다. 반환 타입: `SimulationState`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _build_initial_state(self) -> SimulationState:
        """에피소드 시작 상태를 만듭니다."""

        self._validate_action_space_consistency()
        # LINE-BY-LINE: 호출자에게 `SimulationState(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return SimulationState(
            # LINE-BY-LINE: `current_time`에 `0.0` 결과를 저장합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            current_time=0.0,
            # LINE-BY-LINE: `unscheduled_jobs`에 `set(self.jobs.keys())` 결과를 저장합니다. 의미/사용: `unscheduled_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            unscheduled_jobs=set(self.jobs.keys()),
            # LINE-BY-LINE: `machine_available_at`에 `{machine_id: 0.0 for machine_id in self.machines}` 결과를 저장합니다. 의미/사용: `machine_available_at` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_available_at={machine_id: 0.0 for machine_id in self.machines},
            # LINE-BY-LINE: `machine_slot_available_at`에 `{` 결과를 저장합니다. 의미/사용: `machine_slot_available_at` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_available_at={
                # LINE-BY-LINE: `machine_id`를 `[0.0 for _ in range(self._effective_machine_slot_count(machine))]` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id: [0.0 for _ in range(self._effective_machine_slot_count(machine))]
                # LINE-BY-LINE: `machine_id, machine in self.machines.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for machine_id, machine in self.machines.items()
            },
            # LINE-BY-LINE: `machine_loads`에 `{machine_id: 0.0 for machine_id in self.machines}` 결과를 저장합니다. 의미/사용: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_loads={machine_id: 0.0 for machine_id in self.machines},
            # LINE-BY-LINE: `downstream_loads`에 `{bay_id: 0 for bay_id in self.bays}` 결과를 저장합니다. 의미/사용: `downstream_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_loads={bay_id: 0 for bay_id in self.bays},
            # LINE-BY-LINE: `machine_daily_loads`에 `{}` 결과를 저장합니다. 의미/사용: `machine_daily_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_loads={},
            # LINE-BY-LINE: `machine_daily_job_counts`에 `{}` 결과를 저장합니다. 의미/사용: `machine_daily_job_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_job_counts={},
            # LINE-BY-LINE: `machine_daily_length_sums`에 `{}` 결과를 저장합니다. 의미/사용: `machine_daily_length_sums` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_length_sums={},
            # LINE-BY-LINE: `scheduled_job_count_by_day`에 `{}` 결과를 저장합니다. 의미/사용: `scheduled_job_count_by_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            scheduled_job_count_by_day={},
            # LINE-BY-LINE: `block_set_bay_assignments`에 `{}` 결과를 저장합니다. 의미/사용: `block_set_bay_assignments` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            block_set_bay_assignments={},
            # LINE-BY-LINE: `machine_last_family`에 `{machine_id: None for machine_id in self.machines}` 결과를 저장합니다. 의미/사용: `machine_last_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_last_family={machine_id: None for machine_id in self.machines},
            # LINE-BY-LINE: `resource_active_until`에 `{resource_id: [] for resource_id in self.resources}` 결과를 저장합니다. 의미/사용: `resource_active_until` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            resource_active_until={resource_id: [] for resource_id in self.resources},
        )

    # LINE-BY-LINE: `@staticmethod` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @staticmethod
    # LINE-BY-LINE: `_normalize_date_key(date_key: str)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _normalize_date_key(date_key: str) -> str:
        """override key를 YYYY-MM-DD 형태로 통일합니다."""

        # LINE-BY-LINE: 조건 `len(date_key) == 8 and date_key.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(date_key) == 8 and date_key.isdigit():
            # LINE-BY-LINE: 호출자에게 `f"{date_key[0:4]}-{date_key[4:6]}-{date_key[6:8]}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return f"{date_key[0:4]}-{date_key[4:6]}-{date_key[6:8]}"
        # LINE-BY-LINE: 호출자에게 `date_key`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return date_key

    # LINE-BY-LINE: `_current_day_index(self, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `int`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _current_day_index(self, at_time: Optional[float] = None) -> int:
        """현재 시각이 시작일로부터 며칠째인지 계산합니다."""

        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: 호출자에게 `int(target_time // self.minutes_per_day)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(target_time // self.minutes_per_day)

    # LINE-BY-LINE: `_day_key_from_index(self, day_index: int)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _day_key_from_index(self, day_index: int) -> str:
        """day index를 실제 날짜 문자열로 바꿉니다."""

        # LINE-BY-LINE: 호출자에게 `(self.start_date + timedelta(days=day_index)).isoformat()`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return (self.start_date + timedelta(days=day_index)).isoformat()

    # LINE-BY-LINE: `_current_day_key(self, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _current_day_key(self, at_time: Optional[float] = None) -> str:
        """현재 시각 기준 날짜 key를 반환합니다."""

        # LINE-BY-LINE: 호출자에게 `self._day_key_from_index(self._current_day_index(at_time))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._day_key_from_index(self._current_day_index(at_time))

    # LINE-BY-LINE: `@staticmethod` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @staticmethod
    # LINE-BY-LINE: `_minute_of_day(at_time: float, minutes_per_day: int)` 함수를 정의합니다. 반환 타입: `int`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _minute_of_day(at_time: float, minutes_per_day: int) -> int:
        """현재 시각을 하루 기준 분(minute-of-day)로 변환합니다."""

        # LINE-BY-LINE: 호출자에게 `int(at_time % minutes_per_day)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(at_time % minutes_per_day)

    # LINE-BY-LINE: `@staticmethod` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @staticmethod
    # LINE-BY-LINE: `_parse_time_string(time_string: str)` 함수를 정의합니다. 반환 타입: `int`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _parse_time_string(time_string: str) -> int:
        """HH:MM 문자열을 분 단위 정수로 바꿉니다."""

        # LINE-BY-LINE: `hour, minute` 여러 변수에 `time_string.split(":")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        hour, minute = time_string.split(":")
        # LINE-BY-LINE: 호출자에게 `int(hour) * 60 + int(minute)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(hour) * 60 + int(minute)

    # LINE-BY-LINE: `_is_in_time_window(self, minute_of_day: int, window: str)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_in_time_window(self, minute_of_day: int, window: str) -> bool:
        """현재 분이 특정 시간 구간 안에 들어오는지 확인합니다.

        지원 형식:
        - 12:00-13:00
        - 22:00-02:00  (자정 넘김)
        """

        # LINE-BY-LINE: `start_raw, end_raw` 여러 변수에 `window.split("-")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        start_raw, end_raw = window.split("-")
        # LINE-BY-LINE: `start_minute`에 `self._parse_time_string(start_raw)` 결과를 저장합니다. 의미/사용: `start_minute` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_minute = self._parse_time_string(start_raw)
        # LINE-BY-LINE: `end_minute`에 `self._parse_time_string(end_raw)` 결과를 저장합니다. 의미/사용: `end_minute` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        end_minute = self._parse_time_string(end_raw)

        # LINE-BY-LINE: 조건 `start_minute <= end_minute`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_minute <= end_minute:
            # LINE-BY-LINE: 호출자에게 `start_minute <= minute_of_day < end_minute`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return start_minute <= minute_of_day < end_minute
        # LINE-BY-LINE: 호출자에게 `minute_of_day >= start_minute or minute_of_day < end_minute`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return minute_of_day >= start_minute or minute_of_day < end_minute

    # LINE-BY-LINE: `_matches_date_spec(self, day_key: str, date_spec: str)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _matches_date_spec(self, day_key: str, date_spec: str) -> bool:
        """날짜가 단일 날짜 또는 범위 지정과 일치하는지 확인합니다.

        지원 형식:
        - 2026-01-01
        - 20260101
        - 2026-01-01~2026-01-03
        - 20260101..20260103
        """

        # LINE-BY-LINE: `normalized_spec`에 `self._normalize_date_key(date_spec)` 결과를 저장합니다. 의미/사용: `normalized_spec` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized_spec = self._normalize_date_key(date_spec)
        # LINE-BY-LINE: 조건 `"~" in normalized_spec`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if "~" in normalized_spec:
            # LINE-BY-LINE: `start_key, end_key` 여러 변수에 `normalized_spec.split("~", 1)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            start_key, end_key = normalized_spec.split("~", 1)
            # LINE-BY-LINE: 호출자에게 `start_key <= day_key <= self._normalize_date_key(end_key)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return start_key <= day_key <= self._normalize_date_key(end_key)
        # LINE-BY-LINE: 조건 `".." in normalized_spec`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if ".." in normalized_spec:
            # LINE-BY-LINE: `start_key, end_key` 여러 변수에 `normalized_spec.split("..", 1)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            start_key, end_key = normalized_spec.split("..", 1)
            # LINE-BY-LINE: 호출자에게 `start_key <= day_key <= self._normalize_date_key(end_key)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return start_key <= day_key <= self._normalize_date_key(end_key)
        # LINE-BY-LINE: 호출자에게 `day_key == normalized_spec`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return day_key == normalized_spec

    # LINE-BY-LINE: `_day_matches_any_spec(self, day_key: str, date_specs: List[str])` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _day_matches_any_spec(self, day_key: str, date_specs: List[str]) -> bool:
        """현재 날짜가 date spec 목록 중 하나와 일치하는지 확인합니다."""

        # LINE-BY-LINE: 호출자에게 `any(self._matches_date_spec(day_key, str(spec)) for spec in date_specs)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return any(self._matches_date_spec(day_key, str(spec)) for spec in date_specs)

    # LINE-BY-LINE: `_ensure_day_state(self, day_key: str)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _ensure_day_state(self, day_key: str) -> None:
        """해당 날짜에 대한 상태 dict가 없으면 기본값으로 만듭니다."""

        # LINE-BY-LINE: 조건 `day_key not in self.state.machine_daily_loads`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if day_key not in self.state.machine_daily_loads:
            # LINE-BY-LINE: 현재 객체의 `state.machine_daily_loads[day_key]` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.state.machine_daily_loads[day_key] = {
                # LINE-BY-LINE: `machine_id`를 `0.0 for machine_id in self.machines` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id: 0.0 for machine_id in self.machines
            }
        # LINE-BY-LINE: 조건 `day_key not in self.state.machine_daily_job_counts`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if day_key not in self.state.machine_daily_job_counts:
            # LINE-BY-LINE: 현재 객체의 `state.machine_daily_job_counts[day_key]` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.state.machine_daily_job_counts[day_key] = {
                # LINE-BY-LINE: `machine_id`를 `0 for machine_id in self.machines` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id: 0 for machine_id in self.machines
            }
        # LINE-BY-LINE: 조건 `day_key not in self.state.machine_daily_length_sums`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if day_key not in self.state.machine_daily_length_sums:
            # LINE-BY-LINE: 현재 객체의 `state.machine_daily_length_sums[day_key]` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.state.machine_daily_length_sums[day_key] = {
                # LINE-BY-LINE: `machine_id`를 `0.0 for machine_id in self.machines` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id: 0.0 for machine_id in self.machines
            }
        # LINE-BY-LINE: 조건 `day_key not in self.state.scheduled_job_count_by_day`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if day_key not in self.state.scheduled_job_count_by_day:
            # LINE-BY-LINE: 현재 객체의 `state.scheduled_job_count_by_day[day_key]` 속성에 `0` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.state.scheduled_job_count_by_day[day_key] = 0

    # LINE-BY-LINE: `_sync_machine_earliest_availability(self, machine_id: str)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _sync_machine_earliest_availability(self, machine_id: str) -> None:
        """설비 슬롯 목록을 기반으로 가장 이른 가용 시각을 갱신합니다."""

        # LINE-BY-LINE: `slot_times`에 `self.state.machine_slot_available_at[machine_id]` 결과를 저장합니다. 의미/사용: `slot_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times = self.state.machine_slot_available_at[machine_id]
        # LINE-BY-LINE: 현재 객체의 `state.machine_available_at[machine_id]` 속성에 `min(slot_times) if slot_times else self.state.current_time` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.machine_available_at[machine_id] = min(slot_times) if slot_times else self.state.current_time

    # LINE-BY-LINE: `_sync_all_machine_availability(self)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _sync_all_machine_availability(self) -> None:
        """모든 설비의 가장 이른 가용 시각을 다시 계산합니다."""

        # LINE-BY-LINE: `machine_id in self.machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine_id in self.machines:
            # LINE-BY-LINE: `self._sync_machine_earliest_availability(machine_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._sync_machine_earliest_availability(machine_id)

    # LINE-BY-LINE: `_prune_resource_allocations(self, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _prune_resource_allocations(self, at_time: Optional[float] = None) -> None:
        """현재 시점 이전에 종료된 보조자원 점유를 제거합니다."""

        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: `resource_id, finish_times in self.state.resource_active_until.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id, finish_times in self.state.resource_active_until.items():
            # LINE-BY-LINE: 현재 객체의 `state.resource_active_until[resource_id]` 속성에 `[` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.state.resource_active_until[resource_id] = [
                # LINE-BY-LINE: `finish_time` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                finish_time
                # LINE-BY-LINE: `finish_time in finish_times` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for finish_time in finish_times
                # LINE-BY-LINE: 조건 `finish_time > target_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if finish_time > target_time
            ]

    # LINE-BY-LINE: `_apply_due_downstream_events(self, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
    def _apply_due_downstream_events(self, at_time: Optional[float] = None) -> None:
        """현재 시점까지 도달한 후공정 버퍼 이벤트를 적용합니다."""

        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: 조건 `not self.state.downstream_events`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self.state.downstream_events:
            # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
            return

        # LINE-BY-LINE: `pending_events` 변수에 `[]` 결과를 저장합니다. 의미: `pending_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        pending_events: List[DownstreamEvent] = []
        # LINE-BY-LINE: `event in sorted(self.state.downstream_events, key=lambda item: item.event_time)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for event in sorted(self.state.downstream_events, key=lambda item: item.event_time):
            # LINE-BY-LINE: 조건 `event.event_time <= target_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if event.event_time <= target_time:
                # LINE-BY-LINE: `next_load`에 `self.state.downstream_loads[event.bay_id] + int(event.delta)` 결과를 저장합니다. 의미/사용: `next_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                next_load = self.state.downstream_loads[event.bay_id] + int(event.delta)
                # LINE-BY-LINE: 현재 객체의 `state.downstream_loads[event.bay_id]` 속성에 `max(0, next_load)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
                self.state.downstream_loads[event.bay_id] = max(0, next_load)
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else:
                # LINE-BY-LINE: `pending_events.append(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                pending_events.append(event)
        # LINE-BY-LINE: 현재 객체의 `state.downstream_events` 속성에 `pending_events` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.downstream_events = pending_events

    # LINE-BY-LINE: `_move_current_time(self, new_time: float)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _move_current_time(self, new_time: float) -> None:
        """시각 이동 시 상태 동기화를 함께 수행합니다."""

        # LINE-BY-LINE: 현재 객체의 `state.current_time` 속성에 `new_time` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.current_time = new_time
        # LINE-BY-LINE: `self._ensure_day_state(self._current_day_key())`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(self._current_day_key())
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()

    # LINE-BY-LINE: `_resource_shortages_for(self, resource_ids: Tuple[str, ...])` 함수를 정의합니다. 반환 타입: `Dict[str, int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _resource_shortages_for(self, resource_ids: Tuple[str, ...]) -> Dict[str, int]:
        """현재 시점 기준으로 부족한 보조자원 슬롯을 계산합니다."""

        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `shortages` 변수에 `{}` 결과를 저장합니다. 의미: `shortages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        shortages: Dict[str, int] = {}
        # LINE-BY-LINE: `resource_id in resource_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id in resource_ids:
            # LINE-BY-LINE: `resource`에 `self.resources.get(resource_id)` 결과를 저장합니다. 의미/사용: `resource` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            resource = self.resources.get(resource_id)
            # LINE-BY-LINE: 조건 `resource is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if resource is None:
                # LINE-BY-LINE: `shortages[resource_id]`에 `1` 결과를 저장합니다. 의미/사용: `shortages[resource_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                shortages[resource_id] = 1
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `active_count`에 `len(self.state.resource_active_until.get(resource_id, []))` 결과를 저장합니다. 의미/사용: `active_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_count = len(self.state.resource_active_until.get(resource_id, []))
            # LINE-BY-LINE: `shortage`에 `active_count + 1 - int(resource.capacity)` 결과를 저장합니다. 의미/사용: `shortage` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            shortage = active_count + 1 - int(resource.capacity)
            # LINE-BY-LINE: 조건 `shortage > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if shortage > 0:
                # LINE-BY-LINE: `shortages[resource_id]`에 `shortage` 결과를 저장합니다. 의미/사용: `shortages[resource_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                shortages[resource_id] = shortage
        # LINE-BY-LINE: 호출자에게 `shortages`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return shortages

    # LINE-BY-LINE: `_required_resource_ids(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `Tuple[str, ...]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _required_resource_ids(self, job: Job, machine: Machine) -> Tuple[str, ...]:
        """설비/작업이 동시에 요구하는 자원 목록을 정규화합니다."""

        # LINE-BY-LINE: `resource_ids`에 `dict.fromkeys(` 결과를 저장합니다. 의미/사용: `resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        resource_ids = dict.fromkeys(
            # LINE-BY-LINE: `[*tuple(machine.required_resource_ids), *tuple(job.required_resource_ids)]`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            [*tuple(machine.required_resource_ids), *tuple(job.required_resource_ids)]
        )
        # LINE-BY-LINE: 호출자에게 `tuple(resource_ids.keys())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return tuple(resource_ids.keys())

    # LINE-BY-LINE: `_get_daily_override_map(self, flag_name: str, map_name: str)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_daily_override_map(self, flag_name: str, map_name: str) -> Dict:
        """override 활성화 여부와 실제 override map을 함께 읽습니다."""

        # LINE-BY-LINE: 조건 `not self.override_config.get(flag_name, False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self.override_config.get(flag_name, False):
            # LINE-BY-LINE: 호출자에게 `{}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return {}
        # LINE-BY-LINE: 호출자에게 `self.override_config.get(map_name, {}) or {}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.override_config.get(map_name, {}) or {}

    # LINE-BY-LINE: `_get_machine_enabled_flag(self, machine_id: str, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_machine_enabled_flag(self, machine_id: str, at_time: Optional[float] = None) -> bool:
        """기본 enabled + 날짜별 machine enable override를 반영합니다."""

        # LINE-BY-LINE: `base_enabled`에 `bool(self.machines[machine_id].enabled)` 결과를 저장합니다. 의미/사용: `base_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_enabled = bool(self.machines[machine_id].enabled)
        # LINE-BY-LINE: `override_map`에 `self._get_daily_override_map(` 결과를 저장합니다. 의미/사용: `override_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        override_map = self._get_daily_override_map(
            # LINE-BY-LINE: 문자열 값 `"enable_daily_machine_enable_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "enable_daily_machine_enable_overrides",
            # LINE-BY-LINE: 문자열 값 `"daily_machine_enable_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "daily_machine_enable_overrides",
        )
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key(at_time)` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key(at_time)
        # LINE-BY-LINE: `normalized`에 `{` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized = {
            # LINE-BY-LINE: `self._normalize_date_key(day_key): value`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._normalize_date_key(day_key): value
            # LINE-BY-LINE: `day_key, value in override_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for day_key, value in override_map.items()
        }
        # LINE-BY-LINE: `day_override`에 `normalized.get(current_day_key, {})` 결과를 저장합니다. 의미/사용: `day_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        day_override = normalized.get(current_day_key, {})
        # LINE-BY-LINE: 조건 `machine_id in day_override`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine_id in day_override:
            # LINE-BY-LINE: 호출자에게 `bool(day_override[machine_id])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return bool(day_override[machine_id])
        # LINE-BY-LINE: 호출자에게 `base_enabled`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return base_enabled

    # LINE-BY-LINE: `_is_global_calendar_open(self, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `tuple[bool, str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_global_calendar_open(self, at_time: Optional[float] = None) -> tuple[bool, str]:
        """현재 시각에 공장 전체가 가동 가능한지 판단합니다."""

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key(target_time)` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key(target_time)
        # LINE-BY-LINE: `minute_of_day`에 `self._minute_of_day(target_time, self.minutes_per_day)` 결과를 저장합니다. 의미/사용: `minute_of_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        minute_of_day = self._minute_of_day(target_time, self.minutes_per_day)

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_holidays_off", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_holidays_off", False):
            # LINE-BY-LINE: `holiday_specs`에 `calendar_cfg.get("holidays_off", []) or []` 결과를 저장합니다. 의미/사용: `holiday_specs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            holiday_specs = calendar_cfg.get("holidays_off", []) or []
            # LINE-BY-LINE: 조건 `self._day_matches_any_spec(current_day_key, holiday_specs)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._day_matches_any_spec(current_day_key, holiday_specs):
                # LINE-BY-LINE: 호출자에게 `False, f"holiday off on {current_day_key}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False, f"holiday off on {current_day_key}"

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_half_day_off", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_half_day_off", False):
            # LINE-BY-LINE: `half_day_specs`에 `calendar_cfg.get("half_day_off", []) or []` 결과를 저장합니다. 의미/사용: `half_day_specs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            half_day_specs = calendar_cfg.get("half_day_off", []) or []
            # LINE-BY-LINE: `half_day_window`에 `calendar_cfg.get("half_day_off_window", "12:00-24:00")` 결과를 저장합니다. 의미/사용: `half_day_window` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            half_day_window = calendar_cfg.get("half_day_off_window", "12:00-24:00")
            # LINE-BY-LINE: 조건 `self._day_matches_any_spec(current_day_key, half_day_specs)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._day_matches_any_spec(current_day_key, half_day_specs):
                # LINE-BY-LINE: 조건 `self._is_in_time_window(minute_of_day, half_day_window)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if self._is_in_time_window(minute_of_day, half_day_window):
                    # LINE-BY-LINE: 호출자에게 `False, f"half day off active ({half_day_window})"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                    return False, f"half day off active ({half_day_window})"

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_lunch_break", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_lunch_break", False):
            # LINE-BY-LINE: `lunch_window`에 `calendar_cfg.get("lunch_break")` 결과를 저장합니다. 의미/사용: `lunch_window` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            lunch_window = calendar_cfg.get("lunch_break")
            # LINE-BY-LINE: `lunch_dates`에 `calendar_cfg.get("lunch_break_dates", []) or []` 결과를 저장합니다. 의미/사용: `lunch_dates` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            lunch_dates = calendar_cfg.get("lunch_break_dates", []) or []
            # LINE-BY-LINE: `lunch_applies`에 `(not lunch_dates) or self._day_matches_any_spec(current_day_key, lunch_dates)` 결과를 저장합니다. 의미/사용: `lunch_applies` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            lunch_applies = (not lunch_dates) or self._day_matches_any_spec(current_day_key, lunch_dates)
            # LINE-BY-LINE: 조건 `lunch_window and lunch_applies and self._is_in_time_window(minute_of_day, lunch_window)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if lunch_window and lunch_applies and self._is_in_time_window(minute_of_day, lunch_window):
                # LINE-BY-LINE: 호출자에게 `False, f"lunch break active ({lunch_window})"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False, f"lunch break active ({lunch_window})"

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_global_shutdown_windows", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_global_shutdown_windows", False):
            # LINE-BY-LINE: `shutdown_map`에 `calendar_cfg.get("global_shutdown_windows", {}) or {}` 결과를 저장합니다. 의미/사용: `shutdown_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            shutdown_map = calendar_cfg.get("global_shutdown_windows", {}) or {}
            # LINE-BY-LINE: `normalized`에 `{` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            normalized = {
                # LINE-BY-LINE: `self._normalize_date_key(day_key): windows`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self._normalize_date_key(day_key): windows
                # LINE-BY-LINE: `day_key, windows in shutdown_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for day_key, windows in shutdown_map.items()
            }
            # LINE-BY-LINE: `day_windows`에 `normalized.get(current_day_key, [])` 결과를 저장합니다. 의미/사용: `day_windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            day_windows = normalized.get(current_day_key, [])
            # LINE-BY-LINE: `window in day_windows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for window in day_windows:
                # LINE-BY-LINE: 조건 `self._is_in_time_window(minute_of_day, str(window))`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if self._is_in_time_window(minute_of_day, str(window)):
                    # LINE-BY-LINE: 호출자에게 `False, f"global shutdown active ({window})"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                    return False, f"global shutdown active ({window})"

        # LINE-BY-LINE: 호출자에게 `True, ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return True, ""

    # LINE-BY-LINE: `_machine_windows_for_day(self, config_key: str, machine_id: str, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `List[str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _machine_windows_for_day(self, config_key: str, machine_id: str, at_time: Optional[float] = None) -> List[str]:
        """설비별 운영/정지 시간 구간을 읽습니다.

        지원 형태:
        - machine_operating_windows:
            default:
              laser_01: ["08:00-18:00"]
            "2026-01-01":
              laser_01: ["10:00-16:00"]

        우선순위:
        1. 해당 날짜 key
        2. default
        """

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: `raw_map`에 `calendar_cfg.get(config_key, {}) or {}` 결과를 저장합니다. 의미/사용: `raw_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        raw_map = calendar_cfg.get(config_key, {}) or {}
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key(at_time)` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key(at_time)

        # LINE-BY-LINE: `normalized`에 `{}` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized = {}
        # LINE-BY-LINE: `day_key, machine_map in raw_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for day_key, machine_map in raw_map.items():
            # LINE-BY-LINE: 조건 `day_key == "default"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if day_key == "default":
                # LINE-BY-LINE: `normalized["default"]`에 `machine_map` 결과를 저장합니다. 의미/사용: `normalized["default"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                normalized["default"] = machine_map
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else:
                # LINE-BY-LINE: `normalized[self._normalize_date_key(day_key)]`에 `machine_map` 결과를 저장합니다. 의미/사용: `_normalize_date_key(day_key)]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                normalized[self._normalize_date_key(day_key)] = machine_map

        # LINE-BY-LINE: 조건 `current_day_key in normalized and machine_id in normalized[current_day_key]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if current_day_key in normalized and machine_id in normalized[current_day_key]:
            # LINE-BY-LINE: `windows`에 `normalized[current_day_key][machine_id]` 결과를 저장합니다. 의미/사용: `windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            windows = normalized[current_day_key][machine_id]
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `windows`에 `normalized.get("default", {}).get(machine_id, [])` 결과를 저장합니다. 의미/사용: `windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            windows = normalized.get("default", {}).get(machine_id, [])

        # LINE-BY-LINE: 조건 `isinstance(windows, str)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if isinstance(windows, str):
            # LINE-BY-LINE: 호출자에게 `[windows]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return [windows]
        # LINE-BY-LINE: 호출자에게 `[str(window) for window in windows]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return [str(window) for window in windows]

    # LINE-BY-LINE: `_machine_calendar_status(self, machine_id: str, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `tuple[bool, str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _machine_calendar_status(self, machine_id: str, at_time: Optional[float] = None) -> tuple[bool, str]:
        """현재 시각에 특정 설비가 계획된 운영시간 안에 있는지 판단합니다.

        이 함수는 "고장"이 아니라 "계획된 운영 캘린더"를 다룹니다.
        """

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key(target_time)` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key(target_time)
        # LINE-BY-LINE: `minute_of_day`에 `self._minute_of_day(target_time, self.minutes_per_day)` 결과를 저장합니다. 의미/사용: `minute_of_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        minute_of_day = self._minute_of_day(target_time, self.minutes_per_day)

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_machine_operating_windows", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_machine_operating_windows", False):
            # LINE-BY-LINE: `windows`에 `self._machine_windows_for_day("machine_operating_windows", machine_id, at_time=target_time)` 결과를 저장합니다. 의미/사용: `windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            windows = self._machine_windows_for_day("machine_operating_windows", machine_id, at_time=target_time)
            # LINE-BY-LINE: 조건 `windows and not any(self._is_in_time_window(minute_of_day, window) for window in windows)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if windows and not any(self._is_in_time_window(minute_of_day, window) for window in windows):
                # LINE-BY-LINE: 호출자에게 `False, f"{machine_id} outside operating window on {current_day_key}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False, f"{machine_id} outside operating window on {current_day_key}"

        # LINE-BY-LINE: 조건 `calendar_cfg.get("enable_machine_shutdown_windows", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if calendar_cfg.get("enable_machine_shutdown_windows", False):
            # LINE-BY-LINE: `windows`에 `self._machine_windows_for_day("machine_shutdown_windows", machine_id, at_time=target_time)` 결과를 저장합니다. 의미/사용: `windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            windows = self._machine_windows_for_day("machine_shutdown_windows", machine_id, at_time=target_time)
            # LINE-BY-LINE: `window in windows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for window in windows:
                # LINE-BY-LINE: 조건 `self._is_in_time_window(minute_of_day, window)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if self._is_in_time_window(minute_of_day, window):
                    # LINE-BY-LINE: 호출자에게 `False, f"{machine_id} planned shutdown active ({window})"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                    return False, f"{machine_id} planned shutdown active ({window})"

        # LINE-BY-LINE: 호출자에게 `True, ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return True, ""

    # LINE-BY-LINE: `_machine_breakdown_status(self, machine_id: str, at_time: Optional[float] = None)` 함수를 정의합니다. 반환 타입: `tuple[bool, str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _machine_breakdown_status(self, machine_id: str, at_time: Optional[float] = None) -> tuple[bool, str]:
        """현재 시각에 특정 설비가 고장/정지 상태인지 판단합니다."""

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: 조건 `not calendar_cfg.get("enable_machine_breakdowns", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not calendar_cfg.get("enable_machine_breakdowns", False):
            # LINE-BY-LINE: 호출자에게 `False, ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False, ""

        # LINE-BY-LINE: `target_time`에 `self.state.current_time if at_time is None else at_time` 결과를 저장합니다. 의미/사용: `target_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_time = self.state.current_time if at_time is None else at_time
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key(target_time)` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key(target_time)
        # LINE-BY-LINE: `minute_of_day`에 `self._minute_of_day(target_time, self.minutes_per_day)` 결과를 저장합니다. 의미/사용: `minute_of_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        minute_of_day = self._minute_of_day(target_time, self.minutes_per_day)
        # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
        breakdown_map = calendar_cfg.get("machine_breakdowns", {}) or {}
        # LINE-BY-LINE: `normalized`에 `{` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized = {
            # LINE-BY-LINE: `self._normalize_date_key(day_key): machine_map`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._normalize_date_key(day_key): machine_map
            # LINE-BY-LINE: `day_key, machine_map in breakdown_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for day_key, machine_map in breakdown_map.items()
        }
        # LINE-BY-LINE: `day_breakdowns`에 `normalized.get(current_day_key, {})` 결과를 저장합니다. 의미/사용: `day_breakdowns` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        day_breakdowns = normalized.get(current_day_key, {})
        # LINE-BY-LINE: `machine_windows`에 `day_breakdowns.get(machine_id)` 결과를 저장합니다. 의미/사용: `machine_windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_windows = day_breakdowns.get(machine_id)

        # LINE-BY-LINE: 조건 `not machine_windows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not machine_windows:
            # LINE-BY-LINE: 호출자에게 `False, ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False, ""

        # full_day / true / 00:00-24:00 같은 간단 표기도 허용합니다.
        # LINE-BY-LINE: 조건 `isinstance(machine_windows, str)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if isinstance(machine_windows, str):
            # LINE-BY-LINE: 조건 `machine_windows in {"full_day", "FULL_DAY", "true", "True"}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if machine_windows in {"full_day", "FULL_DAY", "true", "True"}:
                # LINE-BY-LINE: 호출자에게 `True, f"{machine_id} breakdown full day"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True, f"{machine_id} breakdown full day"
            # LINE-BY-LINE: `machine_windows`에 `[machine_windows]` 결과를 저장합니다. 의미/사용: `machine_windows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_windows = [machine_windows]
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `machine_windows is True`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif machine_windows is True:
            # LINE-BY-LINE: 호출자에게 `True, f"{machine_id} breakdown full day"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True, f"{machine_id} breakdown full day"

        # LINE-BY-LINE: `window in machine_windows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for window in machine_windows:
            # LINE-BY-LINE: 조건 `self._is_in_time_window(minute_of_day, str(window))`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._is_in_time_window(minute_of_day, str(window)):
                # LINE-BY-LINE: 호출자에게 `True, f"{machine_id} breakdown active ({window})"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True, f"{machine_id} breakdown active ({window})"

        # LINE-BY-LINE: 호출자에게 `False, ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False, ""

    # LINE-BY-LINE: `_is_machine_working(self, machine_id: str, at_time: float)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_machine_working(self, machine_id: str, at_time: float) -> bool:
        """주어진 시각에 작업이 실제로 진행될 수 있는지 판단합니다."""

        # LINE-BY-LINE: `global_open, _` 여러 변수에 `self._is_global_calendar_open(at_time=at_time)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        global_open, _ = self._is_global_calendar_open(at_time=at_time)
        # LINE-BY-LINE: `machine_open, _` 여러 변수에 `self._machine_calendar_status(machine_id, at_time=at_time)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        machine_open, _ = self._machine_calendar_status(machine_id, at_time=at_time)
        # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
        breakdown_active, _ = self._machine_breakdown_status(machine_id, at_time=at_time)
        # LINE-BY-LINE: `machine_enabled`에 `self._get_machine_enabled_flag(machine_id, at_time=at_time)` 결과를 저장합니다. 의미/사용: `machine_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_enabled = self._get_machine_enabled_flag(machine_id, at_time=at_time)
        # LINE-BY-LINE: 호출자에게 `global_open and machine_open and (not breakdown_active) and machine_enabled`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return global_open and machine_open and (not breakdown_active) and machine_enabled

    # LINE-BY-LINE: `_changeover_minutes_for(self, machine_id: str, job: Job)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _changeover_minutes_for(self, machine_id: str, job: Job) -> float:
        """직전 계열과 다를 때 추가 셋업 시간을 계산합니다."""

        # LINE-BY-LINE: `setup_cfg`에 `self.config.get("setup", {})` 결과를 저장합니다. 의미/사용: `setup_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        setup_cfg = self.config.get("setup", {})
        # LINE-BY-LINE: 조건 `not setup_cfg.get("enable_family_changeover", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not setup_cfg.get("enable_family_changeover", False):
            # LINE-BY-LINE: 호출자에게 `0.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return 0.0

        # LINE-BY-LINE: `previous_family`에 `self.state.machine_last_family.get(machine_id)` 결과를 저장합니다. 의미/사용: `previous_family` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        previous_family = self.state.machine_last_family.get(machine_id)
        # LINE-BY-LINE: 조건 `previous_family is None or previous_family == job.family`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if previous_family is None or previous_family == job.family:
            # LINE-BY-LINE: 호출자에게 `0.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return 0.0

        # LINE-BY-LINE: `machine_type`에 `self.machines[machine_id].machine_type` 결과를 저장합니다. 의미/사용: `machine_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_type = self.machines[machine_id].machine_type
        # LINE-BY-LINE: `by_type`에 `setup_cfg.get("machine_type_changeover_minutes", {}) or {}` 결과를 저장합니다. 의미/사용: `by_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        by_type = setup_cfg.get("machine_type_changeover_minutes", {}) or {}
        # LINE-BY-LINE: 조건 `machine_type in by_type`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine_type in by_type:
            # LINE-BY-LINE: 호출자에게 `float(by_type[machine_type])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return float(by_type[machine_type])
        # LINE-BY-LINE: 호출자에게 `float(setup_cfg.get("default_family_changeover_minutes", 0.0))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return float(setup_cfg.get("default_family_changeover_minutes", 0.0))

    # LINE-BY-LINE: `_effective_finish_time(self, machine_id: str, start_time: float, processing_minutes: float)` 함수를 정의합니다. 반환 타입: `tuple[float, float]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _effective_finish_time(self, machine_id: str, start_time: float, processing_minutes: float) -> tuple[float, float]:
        """작업 구간 중 비가동 시간을 반영해 실제 완료 시각을 계산합니다."""

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: 조건 `not calendar_cfg.get("enable_operation_time_adjustment", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not calendar_cfg.get("enable_operation_time_adjustment", False):
            # LINE-BY-LINE: 호출자에게 `start_time + processing_minutes, 0.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return start_time + processing_minutes, 0.0
        # LINE-BY-LINE: 조건 `not self._has_calendar_blocking_source()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self._has_calendar_blocking_source():
            # LINE-BY-LINE: 호출자에게 `start_time + processing_minutes, 0.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return start_time + processing_minutes, 0.0

        # LINE-BY-LINE: `remaining`에 `float(processing_minutes)` 결과를 저장합니다. 의미/사용: `remaining` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        remaining = float(processing_minutes)
        # LINE-BY-LINE: `current_time`에 `float(start_time)` 결과를 저장합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_time = float(start_time)
        # LINE-BY-LINE: `blocked_minutes`에 `0.0` 결과를 저장합니다. 의미/사용: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
        blocked_minutes = 0.0
        # LINE-BY-LINE: `max_extension`에 `float(calendar_cfg.get("operation_time_adjustment_limit_minutes", 10080))` 결과를 저장합니다. 의미/사용: `max_extension` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_extension = float(calendar_cfg.get("operation_time_adjustment_limit_minutes", 10080))
        # LINE-BY-LINE: `guard_end`에 `start_time + processing_minutes + max_extension` 결과를 저장합니다. 의미/사용: `guard_end` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        guard_end = start_time + processing_minutes + max_extension

        # LINE-BY-LINE: 조건 `remaining > 1e-9 and current_time <= guard_end`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while remaining > 1e-9 and current_time <= guard_end:
            # LINE-BY-LINE: 조건 `self._is_machine_working(machine_id, current_time)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._is_machine_working(machine_id, current_time):
                # LINE-BY-LINE: `work_chunk`에 `min(1.0, remaining)` 결과를 저장합니다. 의미/사용: `work_chunk` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                work_chunk = min(1.0, remaining)
                # LINE-BY-LINE: `current_time` 값을 `work_chunk` 기준으로 누적/증가합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                current_time += work_chunk
                # LINE-BY-LINE: `remaining` 값을 `work_chunk` 기준으로 차감합니다. 의미/사용: `remaining` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                remaining -= work_chunk
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else:
                # LINE-BY-LINE: `current_time` 값을 `1.0` 기준으로 누적/증가합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                current_time += 1.0
                # LINE-BY-LINE: `blocked_minutes` 값을 `1.0` 기준으로 누적/증가합니다. 의미/사용: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
                blocked_minutes += 1.0

        # LINE-BY-LINE: 호출자에게 `current_time, blocked_minutes`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return current_time, blocked_minutes

    # LINE-BY-LINE: `_has_calendar_blocking_source(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_calendar_blocking_source(self) -> bool:
        """작업 중단을 만들 수 있는 calendar 설정이 하나라도 켜져 있는지 확인합니다."""

        # LINE-BY-LINE: `calendar_cfg`에 `self.config.get("calendar", {})` 결과를 저장합니다. 의미/사용: `calendar_cfg` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        calendar_cfg = self.config.get("calendar", {})
        # LINE-BY-LINE: 호출자에게 `any(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return any(
            # LINE-BY-LINE: `bool(calendar_cfg.get(key, False))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            bool(calendar_cfg.get(key, False))
            # LINE-BY-LINE: `key in (` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key in (
                # LINE-BY-LINE: 문자열 값 `"enable_holidays_off"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_holidays_off",
                # LINE-BY-LINE: 문자열 값 `"enable_half_day_off"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_half_day_off",
                # LINE-BY-LINE: 문자열 값 `"enable_lunch_break"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_lunch_break",
                # LINE-BY-LINE: 문자열 값 `"enable_machine_operating_windows"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_machine_operating_windows",
                # LINE-BY-LINE: 문자열 값 `"enable_machine_shutdown_windows"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_machine_shutdown_windows",
                # LINE-BY-LINE: 문자열 값 `"enable_machine_breakdowns"`를 `in(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "enable_machine_breakdowns",
            )
        )

    # LINE-BY-LINE: `_candidate_timing(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `Dict[str, float | Optional[float]]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _candidate_timing(self, job: Job, machine: Machine) -> Dict[str, float | Optional[float]]:
        """현재 시점에 후보 action을 시작했을 때의 시간 정보를 계산합니다."""

        # LINE-BY-LINE: `start_time`에 `self.state.current_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
        start_time = self.state.current_time
        # LINE-BY-LINE: `changeover_minutes`에 `self._changeover_minutes_for(machine.machine_id, job)` 결과를 저장합니다. 의미/사용: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        changeover_minutes = self._changeover_minutes_for(machine.machine_id, job)
        # LINE-BY-LINE: `processing_minutes`에 `job.estimate_total_minutes(machine) + changeover_minutes` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
        processing_minutes = job.estimate_total_minutes(machine) + changeover_minutes
        # LINE-BY-LINE: `finish_time, blocked_minutes` 여러 변수에 `self._effective_finish_time(` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        finish_time, blocked_minutes = self._effective_finish_time(
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `machine.machine_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.machine_id,
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            start_time,
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `processing_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            processing_minutes,
        )
        # LINE-BY-LINE: `bay`에 `self.bays[job.downstream_bay]` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
        bay = self.bays[job.downstream_bay]
        # LINE-BY-LINE: `arrival_time`에 `finish_time + float(bay.transfer_time_minutes)` 결과를 저장합니다. 의미/사용: `arrival_time`는 절단 완료 후 downstream bay 도착 예정 minute입니다.
        arrival_time = finish_time + float(bay.transfer_time_minutes)
        # LINE-BY-LINE: `release_time`에 `None` 결과를 저장합니다. 의미/사용: `release_time`는 downstream buffer에서 빠지는 예정 minute입니다.
        release_time = None
        # LINE-BY-LINE: 조건 `bay.release_delay_minutes is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if bay.release_delay_minutes is not None:
            # LINE-BY-LINE: `release_time`에 `arrival_time + float(bay.release_delay_minutes)` 결과를 저장합니다. 의미/사용: `release_time`는 downstream buffer에서 빠지는 예정 minute입니다.
            release_time = arrival_time + float(bay.release_delay_minutes)

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `start_time` 키에 `start_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "start_time": start_time,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `processing_minutes` 키에 `processing_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "processing_minutes": processing_minutes,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `elapsed_minutes` 키에 `finish_time - start_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "elapsed_minutes": finish_time - start_time,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `finish_time` 키에 `finish_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "finish_time": finish_time,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `changeover_minutes` 키에 `changeover_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "changeover_minutes": changeover_minutes,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `blocked_minutes` 키에 `blocked_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "blocked_minutes": blocked_minutes,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `downstream_arrival_time` 키에 `arrival_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_arrival_time": arrival_time,
            # LINE-BY-LINE: `_candidate_timing`에서 반환/저장할 dict의 `downstream_release_time` 키에 `release_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_release_time": release_time,
        }

    # LINE-BY-LINE: `_machine_concurrent_stats(self, machine_id: str, at_time: float)` 함수를 정의합니다. 반환 타입: `Dict[str, float | int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _machine_concurrent_stats(self, machine_id: str, at_time: float) -> Dict[str, float | int]:
        """특정 시각에 같은 설비에서 이미 active인 W/O 수와 길이 합."""

        # LINE-BY-LINE: `active_job_count`에 `0` 결과를 저장합니다. 의미/사용: `active_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_job_count = 0
        # LINE-BY-LINE: `active_length_sum`에 `0.0` 결과를 저장합니다. 의미/사용: `active_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_length_sum = 0.0
        # LINE-BY-LINE: `operation in self.state.schedule` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for operation in self.state.schedule:
            # LINE-BY-LINE: 조건 `operation.machine_id != machine_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if operation.machine_id != machine_id:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: 조건 `operation.start_time <= at_time < operation.finish_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if operation.start_time <= at_time < operation.finish_time:
                # LINE-BY-LINE: `active_job_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `active_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                active_job_count += 1
                # LINE-BY-LINE: `active_length_sum` 값을 `float(self.jobs[operation.job_id].plate_length)` 기준으로 누적/증가합니다. 의미/사용: `active_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                active_length_sum += float(self.jobs[operation.job_id].plate_length)
        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_machine_concurrent_stats`에서 반환/저장할 dict의 `job_count` 키에 `active_job_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "job_count": active_job_count,
            # LINE-BY-LINE: `_machine_concurrent_stats`에서 반환/저장할 dict의 `length_sum` 키에 `active_length_sum` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "length_sum": active_length_sum,
        }

    # LINE-BY-LINE: `_project_downstream_load(self, bay_id: str, at_time: float)` 함수를 정의합니다. 반환 타입: `int`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _project_downstream_load(self, bay_id: str, at_time: float) -> int:
        """현재 시점 이후 이벤트를 반영해 특정 시점 예상 적치량을 계산합니다."""

        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `predicted_load`에 `int(self.state.downstream_loads[bay_id])` 결과를 저장합니다. 의미/사용: `predicted_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        predicted_load = int(self.state.downstream_loads[bay_id])
        # LINE-BY-LINE: `event in self.state.downstream_events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for event in self.state.downstream_events:
            # LINE-BY-LINE: 조건 `self.state.current_time < event.event_time <= at_time and event.bay_id == bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self.state.current_time < event.event_time <= at_time and event.bay_id == bay_id:
                # LINE-BY-LINE: `predicted_load` 값을 `int(event.delta)` 기준으로 누적/증가합니다. 의미/사용: `predicted_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                predicted_load += int(event.delta)
        # LINE-BY-LINE: 호출자에게 `max(0, predicted_load)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return max(0, predicted_load)

    # LINE-BY-LINE: `_has_temporary_blocking_condition(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_temporary_blocking_condition(self) -> bool:
        """현재 후보가 없는 이유가 "시간이 지나면 풀릴 가능성이 있는가"를 판단합니다.

        왜 필요한가:
        - 단순 PMSP라면 후보가 없을 때 다음 설비 완료 시점으로만 가도 됩니다.
        - 하지만 현재 프로젝트에는 점심시간, 휴무일, 설비 고장, 일일 용량 제한처럼
          "같은 설비라도 나중에는 다시 가능해질 수 있는" 제약이 있습니다.
        - 그래서 후보가 없는 순간을 무조건 dead-end로 처리하면 안 됩니다.

        현재는 아래 상황을 "임시 차단"으로 봅니다.
        - 공장 전체 휴무 / 점심 / shutdown window
        - 오늘 일일 전체 작업 수 제한 도달
        - 특정 설비의 현재 고장/정지
        - 특정 설비의 오늘 일일 용량 초과

        반대로, 아래는 임시 차단으로 보지 않습니다.
        - 계열 미일치
        - 두께 미일치
        - 정반 길이 미일치
        즉, 시간이 지나도 풀리지 않는 정적 제약은 여기서 제외합니다.
        """

        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `global_calendar_open, _` 여러 변수에 `self._is_global_calendar_open()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        global_calendar_open, _ = self._is_global_calendar_open()
        # LINE-BY-LINE: 조건 `not global_calendar_open`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not global_calendar_open:
            # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True

        # LINE-BY-LINE: `daily_job_cap`에 `self._get_daily_job_cap()` 결과를 저장합니다. 의미/사용: `daily_job_cap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        daily_job_cap = self._get_daily_job_cap()
        # LINE-BY-LINE: 조건 `daily_job_cap is not None and self.state.scheduled_job_count_by_day[current_day_key] >= daily_job...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if daily_job_cap is not None and self.state.scheduled_job_count_by_day[current_day_key] >= daily_job_cap:
            # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True

        # LINE-BY-LINE: `override_enabled`에 `self.override_config.get("enable_daily_machine_enable_overrides", False)` 결과를 저장합니다. 의미/사용: `override_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        override_enabled = self.override_config.get("enable_daily_machine_enable_overrides", False)

        # LINE-BY-LINE: `machine_id in self.machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine_id in self.machines:
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `machine_enabled_flag`에 `self._get_machine_enabled_flag(machine_id)` 결과를 저장합니다. 의미/사용: `machine_enabled_flag` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_enabled_flag = self._get_machine_enabled_flag(machine_id)
            # LINE-BY-LINE: 조건 `not machine_enabled_flag and override_enabled`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not machine_enabled_flag and override_enabled:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

            # LINE-BY-LINE: `machine_calendar_open, _` 여러 변수에 `self._machine_calendar_status(machine_id)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            machine_calendar_open, _ = self._machine_calendar_status(machine_id)
            # LINE-BY-LINE: 조건 `not machine_calendar_open`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not machine_calendar_open:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            breakdown_active, _ = self._machine_breakdown_status(machine_id)
            # LINE-BY-LINE: 조건 `breakdown_active`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if breakdown_active:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

            # LINE-BY-LINE: `daily_load`에 `self.state.machine_daily_loads[current_day_key][machine_id]` 결과를 저장합니다. 의미/사용: `daily_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            daily_load = self.state.machine_daily_loads[current_day_key][machine_id]
            # LINE-BY-LINE: `daily_capacity`에 `self._get_machine_daily_capacity_limit(machine_id)` 결과를 저장합니다. 의미/사용: `daily_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            daily_capacity = self._get_machine_daily_capacity_limit(machine_id)
            # LINE-BY-LINE: 조건 `daily_load >= daily_capacity`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if daily_load >= daily_capacity:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `finish_times in self.state.resource_active_until.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for finish_times in self.state.resource_active_until.values():
            # LINE-BY-LINE: 조건 `finish_times`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if finish_times:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

        # LINE-BY-LINE: 조건 `any(event.event_time > self.state.current_time for event in self.state.downstream_events)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if any(event.event_time > self.state.current_time for event in self.state.downstream_events):
            # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True

        # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False

    # LINE-BY-LINE: `_future_decision_times(self)` 함수를 정의합니다. 반환 타입: `List[float]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _future_decision_times(self) -> List[float]:
        """설비/자원/후공정 이벤트 때문에 다시 판단해볼 만한 미래 시각."""

        # LINE-BY-LINE: 호출자에게 `sorted(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return sorted(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `slot_time` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                slot_time
                # LINE-BY-LINE: `slot_list in self.state.machine_slot_available_at.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for slot_list in self.state.machine_slot_available_at.values()
                # LINE-BY-LINE: `slot_time in slot_list` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for slot_time in slot_list
                # LINE-BY-LINE: 조건 `slot_time > self.state.current_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if slot_time > self.state.current_time
            }
            # LINE-BY-LINE: `| {` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            | {
                # LINE-BY-LINE: `finish_time` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                finish_time
                # LINE-BY-LINE: `finish_times in self.state.resource_active_until.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for finish_times in self.state.resource_active_until.values()
                # LINE-BY-LINE: `finish_time in finish_times` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for finish_time in finish_times
                # LINE-BY-LINE: 조건 `finish_time > self.state.current_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if finish_time > self.state.current_time
            }
            # LINE-BY-LINE: `| {` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            | {
                # LINE-BY-LINE: `event.event_time` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                event.event_time
                # LINE-BY-LINE: `event in self.state.downstream_events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for event in self.state.downstream_events
                # LINE-BY-LINE: 조건 `event.event_time > self.state.current_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if event.event_time > self.state.current_time
            }
        )

    # LINE-BY-LINE: `_current_day_bucket_limits_blocking(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _current_day_bucket_limits_blocking(self) -> bool:
        """현재 날짜 bucket이 꽉 차서 다음 날이 되어야 풀리는 상태인지 판단."""

        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `daily_job_cap`에 `self._get_daily_job_cap()` 결과를 저장합니다. 의미/사용: `daily_job_cap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        daily_job_cap = self._get_daily_job_cap()
        # LINE-BY-LINE: 조건 `daily_job_cap is not None and self.state.scheduled_job_count_by_day[current_day_key] >= daily_job...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if daily_job_cap is not None and self.state.scheduled_job_count_by_day[current_day_key] >= daily_job_cap:
            # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True

        # LINE-BY-LINE: 조건 `not self.state.unscheduled_jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self.state.unscheduled_jobs:
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False

        # LINE-BY-LINE: `machine_id, machine in self.machines.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine_id, machine in self.machines.items():
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `daily_load`에 `self.state.machine_daily_loads[current_day_key][machine_id]` 결과를 저장합니다. 의미/사용: `daily_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            daily_load = self.state.machine_daily_loads[current_day_key][machine_id]
            # LINE-BY-LINE: `daily_capacity`에 `self._get_machine_daily_capacity_limit(machine_id)` 결과를 저장합니다. 의미/사용: `daily_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            daily_capacity = self._get_machine_daily_capacity_limit(machine_id)
            # LINE-BY-LINE: `min_processing`에 `min(` 결과를 저장합니다. 의미/사용: `min_processing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            min_processing = min(
                # LINE-BY-LINE: `job.estimate_total_minutes(machine)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                job.estimate_total_minutes(machine)
                # LINE-BY-LINE: `job in (self.jobs[job_id] for job_id in self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for job in (self.jobs[job_id] for job_id in self.state.unscheduled_jobs)
            )
            # LINE-BY-LINE: 조건 `daily_load + min_processing > daily_capacity`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if daily_load + min_processing > daily_capacity:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True

        # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False

    # LINE-BY-LINE: `_next_temporary_unblock_time(self, horizon_minutes: float)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _next_temporary_unblock_time(self, horizon_minutes: float) -> float:
        """임시 차단 시 다음 확인 시각을 계산한다.

        일자 bucket 제약은 1분씩 기다려도 풀리지 않으므로 다음 날짜 시작으로 점프한다.
        다만 그 전에 설비 완료 이벤트가 있으면 그 시점에서 먼저 다시 판단한다.
        """

        # LINE-BY-LINE: `next_time`에 `min(self.state.current_time + 1.0, horizon_minutes)` 결과를 저장합니다. 의미/사용: `next_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        next_time = min(self.state.current_time + 1.0, horizon_minutes)
        # LINE-BY-LINE: 조건 `not self._current_day_bucket_limits_blocking()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self._current_day_bucket_limits_blocking():
            # LINE-BY-LINE: 호출자에게 `next_time`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return next_time

        # LINE-BY-LINE: `next_day_start`에 `float((self._current_day_index() + 1) * self.minutes_per_day)` 결과를 저장합니다. 의미/사용: `next_day_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        next_day_start = float((self._current_day_index() + 1) * self.minutes_per_day)
        # LINE-BY-LINE: `future_times`에 `self._future_decision_times()` 결과를 저장합니다. 의미/사용: `future_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        future_times = self._future_decision_times()
        # LINE-BY-LINE: 조건 `future_times and future_times[0] < next_day_start`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if future_times and future_times[0] < next_day_start:
            # LINE-BY-LINE: 호출자에게 `min(future_times[0], horizon_minutes)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return min(future_times[0], horizon_minutes)
        # LINE-BY-LINE: 호출자에게 `min(next_day_start, horizon_minutes)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return min(next_day_start, horizon_minutes)

    # LINE-BY-LINE: `_get_machine_daily_capacity_limit(self, machine_id: str)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_machine_daily_capacity_limit(self, machine_id: str) -> float:
        """기본 일일 용량 + 날짜별 machine capacity override를 반영합니다."""

        # LINE-BY-LINE: `base_capacity`에 `float(self.machines[machine_id].daily_capacity_minutes)` 결과를 저장합니다. 의미/사용: `base_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_capacity = float(self.machines[machine_id].daily_capacity_minutes)
        # LINE-BY-LINE: `override_map`에 `self._get_daily_override_map(` 결과를 저장합니다. 의미/사용: `override_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        override_map = self._get_daily_override_map(
            # LINE-BY-LINE: 문자열 값 `"enable_daily_machine_capacity_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "enable_daily_machine_capacity_overrides",
            # LINE-BY-LINE: 문자열 값 `"daily_machine_capacity_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "daily_machine_capacity_overrides",
        )
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `normalized`에 `{` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized = {
            # LINE-BY-LINE: `self._normalize_date_key(day_key): value`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._normalize_date_key(day_key): value
            # LINE-BY-LINE: `day_key, value in override_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for day_key, value in override_map.items()
        }
        # LINE-BY-LINE: `day_override`에 `normalized.get(current_day_key, {})` 결과를 저장합니다. 의미/사용: `day_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        day_override = normalized.get(current_day_key, {})
        # LINE-BY-LINE: 조건 `machine_id in day_override`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine_id in day_override:
            # LINE-BY-LINE: 호출자에게 `float(day_override[machine_id])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return float(day_override[machine_id])
        # LINE-BY-LINE: 호출자에게 `base_capacity`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return base_capacity

    # LINE-BY-LINE: `_get_machine_day_limit_value(self, machine_id: str, limit_key: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_machine_day_limit_value(self, machine_id: str, limit_key: str):
        """설비 동시 작업 묶음 제약값을 읽습니다.

        지원 형태:
        constraints:
          machine_day_limits:
            max_wo_count: 3
            max_length_sum: 55000
            by_machine:
              PLS21:
                max_wo_count: 2
        """

        # LINE-BY-LINE: `limits`에 `self.config.get("constraints", {}).get("machine_day_limits", {}) or {}` 결과를 저장합니다. 의미/사용: `limits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        limits = self.config.get("constraints", {}).get("machine_day_limits", {}) or {}
        # LINE-BY-LINE: `by_machine`에 `limits.get("by_machine", {}) or {}` 결과를 저장합니다. 의미/사용: `by_machine` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        by_machine = limits.get("by_machine", {}) or {}
        # LINE-BY-LINE: `machine_override`에 `by_machine.get(machine_id, {}) or {}` 결과를 저장합니다. 의미/사용: `machine_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_override = by_machine.get(machine_id, {}) or {}
        # LINE-BY-LINE: `value`에 `machine_override.get(limit_key, limits.get(limit_key))` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value = machine_override.get(limit_key, limits.get(limit_key))
        # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if value in (None, ""):
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None
        # LINE-BY-LINE: 호출자에게 `value`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return value

    # LINE-BY-LINE: `_get_machine_day_wo_count_limit(self, machine_id: str)` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_machine_day_wo_count_limit(self, machine_id: str) -> Optional[int]:
        """설비별 동시 W/O 수 제한."""

        # LINE-BY-LINE: `value`에 `self._get_machine_day_limit_value(machine_id, "max_wo_count")` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value = self._get_machine_day_limit_value(machine_id, "max_wo_count")
        # LINE-BY-LINE: 호출자에게 `None if value is None else int(value)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None if value is None else int(value)

    # LINE-BY-LINE: `_get_machine_day_length_sum_limit(self, machine_id: str)` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_machine_day_length_sum_limit(self, machine_id: str) -> Optional[float]:
        """설비별 동시 길이 합계 제한."""

        # LINE-BY-LINE: `value`에 `self._get_machine_day_limit_value(machine_id, "max_length_sum")` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value = self._get_machine_day_limit_value(machine_id, "max_length_sum")
        # LINE-BY-LINE: 호출자에게 `None if value is None else float(value)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None if value is None else float(value)

    # LINE-BY-LINE: `_get_daily_job_cap(self)` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_daily_job_cap(self) -> Optional[int]:
        """오늘 전체 작업 수 제한이 있으면 반환하고, 없으면 None을 반환합니다."""

        # LINE-BY-LINE: `override_map`에 `self._get_daily_override_map(` 결과를 저장합니다. 의미/사용: `override_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        override_map = self._get_daily_override_map(
            # LINE-BY-LINE: 문자열 값 `"enable_daily_job_cap_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "enable_daily_job_cap_overrides",
            # LINE-BY-LINE: 문자열 값 `"daily_job_cap_overrides"`를 `_get_daily_override_map(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "daily_job_cap_overrides",
        )
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `normalized`에 `{` 결과를 저장합니다. 의미/사용: `normalized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized = {
            # LINE-BY-LINE: `self._normalize_date_key(day_key): value`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._normalize_date_key(day_key): value
            # LINE-BY-LINE: `day_key, value in override_map.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for day_key, value in override_map.items()
        }
        # LINE-BY-LINE: 조건 `current_day_key in normalized`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if current_day_key in normalized:
            # LINE-BY-LINE: 호출자에게 `int(normalized[current_day_key])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return int(normalized[current_day_key])
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `machine_loads(self)` 함수를 정의합니다. 반환 타입: `Dict[str, float]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def machine_loads(self) -> Dict[str, float]:
        """휴리스틱에서 현재 설비 부하를 쉽게 참조할 수 있게 합니다."""

        # LINE-BY-LINE: 호출자에게 `self.state.machine_loads`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.state.machine_loads

    # LINE-BY-LINE: `_action_mode(self)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def _action_mode(self) -> str:
        """현재 action space 모드를 반환한다.

        설정 위치:
        - config YAML의 `action_space.mode`

        지원 모드:
        - `job_machine_pair`
          기존 방식. 후보 하나가 W/O 1개와 machine 1개를 바로 schedule한다.
          action_id 예: `JOB001@PLS21`

        - `rolling_capacity`
          과거 가정 검토용 legacy 방식이다.
          현업 확인 결과 기본 모델이 아니며, 명시 opt-in 없이는 초기화에서 차단한다.
          action은 여전히 `JOB001@PLS21` 단건이지만, machine 내부에 여러 active slot이 있다.

        - `batch_open`
          open-batch 방식. 먼저 batch를 열고, W/O를 추가한 뒤, close 시점에 절단을 시작한다.
          action_id 예: `open_batch:JOB001@PLS21`, `add_to_batch:B000001:JOB002`, `close_batch:B000001@PLS21`
          현업 확인 결과 현재 기본 모델이다. Batch는 같이 들어오고 같이 빠진다.

        왜 문자열 검증을 여기서 하는가:
        - 잘못된 mode를 조용히 `job_machine_pair`로 돌리면 fallback이 된다.
        - AGENTS.md 원칙상 잘못된 config는 원인을 출력하고 중단해야 한다.
        """

        # LINE-BY-LINE: `mode`에 `str(self.config.get("action_space", {}).get("mode", "job_machine_pair"))` 결과를 저장합니다. 의미/사용: `mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        mode = str(self.config.get("action_space", {}).get("mode", "job_machine_pair"))
        # LINE-BY-LINE: `allowed_modes`에 `{"job_machine_pair", "rolling_capacity", "batch_open"}` 결과를 저장합니다. 의미/사용: `allowed_modes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        allowed_modes = {"job_machine_pair", "rolling_capacity", "batch_open"}
        # LINE-BY-LINE: 조건 `mode not in allowed_modes`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if mode not in allowed_modes:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._action_mode] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._action_mode] "
                # LINE-BY-LINE: `f"cause`에 `unsupported_action_mode mode={mode} allowed={sorted(allowed_modes)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unsupported_action_mode mode={mode} allowed={sorted(allowed_modes)}"
            )
            # LINE-BY-LINE: `ValueError(f"unsupported action_space.mode: {mode}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"unsupported action_space.mode: {mode}")
        # LINE-BY-LINE: 호출자에게 `mode`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return mode

    # LINE-BY-LINE: `_is_batch_open_mode(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_batch_open_mode(self) -> bool:
        """batch/open-batch 후보를 생성해야 하는지 확인한다."""

        # LINE-BY-LINE: 호출자에게 `self._action_mode() == "batch_open"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._action_mode() == "batch_open"

    # LINE-BY-LINE: `_is_rolling_capacity_mode(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_rolling_capacity_mode(self) -> bool:
        """legacy rolling concurrent capacity 모드인지 확인한다.

        rolling_capacity 모드 의미:
        - 별도 `open_batch`/`close_batch` action을 만들지 않는다.
        - action은 `job_id@machine_id` 하나다.
        - 다만 machine이 "완전히 비어야" 다음 작업을 시작하는 것이 아니다.
        - 현재 active W/O 수와 길이합이 한계 안이면, machine에 남은 slot으로 바로 투입한다.
        - 현업 확인 결과 기본 모델이 아니며, `_validate_action_space_consistency()`가 명시 opt-in 없이는 막는다.

        예:
        - PLS21 active W/O: A, B
        - max_wo_count=3
        - active length sum=45,000
        - 새 W/O C length=9,000이면 54,000 <= 55,000이므로 후보 가능
        - 새 W/O D length=12,000이면 57,000 > 55,000이므로 후보 불가능
        """

        # LINE-BY-LINE: 호출자에게 `self._action_mode() == "rolling_capacity"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._action_mode() == "rolling_capacity"

    # LINE-BY-LINE: `_effective_machine_slot_count(self, machine: Machine)` 함수를 정의합니다. 반환 타입: `int`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _effective_machine_slot_count(self, machine: Machine) -> int:
        """현재 action mode에서 machine이 동시에 열 수 있는 slot 수를 계산한다.

        입력:
        - `machine`: slot 수를 계산할 설비

        출력:
        - 정수 slot 수

        모드별 의미:
        - `job_machine_pair`: 기존 `machine.parallel_capacity`만 사용한다.
        - `rolling_capacity`: legacy/debug 모드에서만 `max_wo_count`를 slot 수로 사용한다.
        - `batch_open`: open/close batch 구조를 사용한다. Batch는 machine slot 1개를 점유한다.

        fallback 금지:
        - rolling_capacity에서 max_wo_count가 없으면 slot 수를 알 수 없으므로 실패한다.
        """

        # LINE-BY-LINE: 조건 `self._is_rolling_capacity_mode()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._is_rolling_capacity_mode():
            # LINE-BY-LINE: `max_wo_count, _` 여러 변수에 `self._batch_limits_for_machine(machine)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            max_wo_count, _ = self._batch_limits_for_machine(machine)
            # LINE-BY-LINE: 호출자에게 `max(1, int(max_wo_count))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return max(1, int(max_wo_count))
        # LINE-BY-LINE: 호출자에게 `max(1, int(machine.parallel_capacity))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return max(1, int(machine.parallel_capacity))

    # LINE-BY-LINE: `_has_pending_work(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_pending_work(self) -> bool:
        """아직 끝나지 않은 일이 있는지 확인한다.

        단건 dispatch에서는 `unscheduled_jobs`만 보면 충분하다.
        batch_open에서는 open batch 안으로 들어간 W/O가 `unscheduled_jobs`에서 빠진다.
        따라서 open batch가 남아 있으면 반드시 close action을 선택해야 종료된다.
        """

        # LINE-BY-LINE: 호출자에게 `bool(self.state.unscheduled_jobs or self.state.open_batches)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return bool(self.state.unscheduled_jobs or self.state.open_batches)

    # LINE-BY-LINE: `_next_batch_id(self)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _next_batch_id(self) -> str:
        """새 open batch id를 만든다.

        출력 예:
        - 첫 번째 batch: B000001
        - 두 번째 batch: B000002
        """

        # LINE-BY-LINE: `self.state.batch_seq` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `batch_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.batch_seq += 1
        # LINE-BY-LINE: 호출자에게 `f"B{self.state.batch_seq:06d}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return f"B{self.state.batch_seq:06d}"

    # LINE-BY-LINE: `_batch_processing_rule(self)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_processing_rule(self) -> str:
        """batch 처리시간 계산 규칙을 읽고 검증한다.

        설정 위치:
        - config YAML의 `batch.processing_time_rule`

        지원 값:
        - `max_individual`
          같은 정반에서 동시에 처리되는 batch window를 가정하고,
          batch 처리시간을 batch 안 W/O별 예상시간 중 최댓값으로 둔다.

        - `sum_individual`
          batch를 묶되 실제 처리는 순차라고 가정하고,
          batch 처리시간을 W/O별 예상시간 합으로 둔다.

        주의:
        - 현재 현업 데이터에는 nesting/batch id가 없다.
        - 그래서 이 규칙은 반드시 config에 명시되어야 한다.
        - 모르는 값을 조용히 max나 sum으로 바꾸지 않는다.
        """

        # LINE-BY-LINE: `batch_config`에 `self.config.get("batch", {}) or {}` 결과를 저장합니다. 의미/사용: `batch_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_config = self.config.get("batch", {}) or {}
        # LINE-BY-LINE: `rule`에 `batch_config.get("processing_time_rule")` 결과를 저장합니다. 의미/사용: `rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rule = batch_config.get("processing_time_rule")
        # LINE-BY-LINE: 조건 `rule in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if rule in (None, ""):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._batch_processing_rule] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._batch_processing_rule] "
                # LINE-BY-LINE: `"cause`에 `missing_batch_processing_time_rule config_key=batch.processing_time_rule"` 결과를 저장합니다. 의미/사용: `"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                "cause=missing_batch_processing_time_rule config_key=batch.processing_time_rule"
            )
            # LINE-BY-LINE: `RuntimeError("batch_open mode requires batch.processing_time_rule")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("batch_open mode requires batch.processing_time_rule")
        # LINE-BY-LINE: `rule`에 `str(rule)` 결과를 저장합니다. 의미/사용: `rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rule = str(rule)
        # LINE-BY-LINE: `allowed_rules`에 `{"max_individual", "sum_individual"}` 결과를 저장합니다. 의미/사용: `allowed_rules` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        allowed_rules = {"max_individual", "sum_individual"}
        # LINE-BY-LINE: 조건 `rule not in allowed_rules`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if rule not in allowed_rules:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._batch_processing_rule] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._batch_processing_rule] "
                # LINE-BY-LINE: `f"cause`에 `unsupported_batch_processing_time_rule rule={rule} allowed={sorted(allowed_rules)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unsupported_batch_processing_time_rule rule={rule} allowed={sorted(allowed_rules)}"
            )
            # LINE-BY-LINE: `ValueError(f"unsupported batch.processing_time_rule: {rule}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"unsupported batch.processing_time_rule: {rule}")
        # LINE-BY-LINE: 호출자에게 `rule`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return rule

    # LINE-BY-LINE: `_batch_limits_for_machine(self, machine: Machine)` 함수를 정의합니다. 반환 타입: `tuple[int, float]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_limits_for_machine(self, machine: Machine) -> tuple[int, float]:
        """설비별 batch W/O 수/길이 한계를 반환한다.

        입력:
        - `machine`: batch를 올릴 설비

        출력:
        - `(max_wo_count, max_length_sum)`
          예: `(3, 55000.0)`

        값 출처 우선순위:
        1. `Machine.max_batch_wo_count`, `Machine.max_batch_length_sum`
        2. config의 `constraints.machine_day_limits`

        왜 실패 처리하는가:
        - batch_open mode에서 한계값이 없으면 3 W/O/55000 제약을 검증할 수 없다.
        - 한계값 없이 진행하면 실제보다 느슨한 환경이 되므로 fallback으로 보면 된다.
        """

        # LINE-BY-LINE: `max_wo_count`에 `machine.max_batch_wo_count` 결과를 저장합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_wo_count = machine.max_batch_wo_count
        # LINE-BY-LINE: `max_length_sum`에 `machine.max_batch_length_sum` 결과를 저장합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_length_sum = machine.max_batch_length_sum
        # LINE-BY-LINE: 조건 `max_wo_count is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if max_wo_count is None:
            # LINE-BY-LINE: `max_wo_count`에 `self._get_machine_day_wo_count_limit(machine.machine_id)` 결과를 저장합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            max_wo_count = self._get_machine_day_wo_count_limit(machine.machine_id)
        # LINE-BY-LINE: 조건 `max_length_sum is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if max_length_sum is None:
            # LINE-BY-LINE: `max_length_sum`에 `self._get_machine_day_length_sum_limit(machine.machine_id)` 결과를 저장합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            max_length_sum = self._get_machine_day_length_sum_limit(machine.machine_id)
        # LINE-BY-LINE: 조건 `max_wo_count in (None, "") or max_length_sum in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if max_wo_count in (None, "") or max_length_sum in (None, ""):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._batch_limits_for_machine] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._batch_limits_for_machine] "
                # LINE-BY-LINE: `f"cause`에 `missing_batch_limits machine_id={machine.machine_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=missing_batch_limits machine_id={machine.machine_id} "
                # LINE-BY-LINE: `f"max_wo_count`에 `{max_wo_count} max_length_sum={max_length_sum}"` 결과를 저장합니다. 의미/사용: `f"max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"max_wo_count={max_wo_count} max_length_sum={max_length_sum}"
            )
            # LINE-BY-LINE: `RuntimeError(f"batch limits missing for machine {machine.machine_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(f"batch limits missing for machine {machine.machine_id}")
        # LINE-BY-LINE: 호출자에게 `int(max_wo_count), float(max_length_sum)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(max_wo_count), float(max_length_sum)

    # LINE-BY-LINE: `_jobs_from_ids(self, job_ids: Tuple[str, ...])` 함수를 정의합니다. 반환 타입: `Tuple[Job, ...]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _jobs_from_ids(self, job_ids: Tuple[str, ...]) -> Tuple[Job, ...]:
        """job id tuple을 Job tuple로 변환한다. 알 수 없는 id는 즉시 실패한다."""

        # LINE-BY-LINE: `jobs`에 `[]` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs = []
        # LINE-BY-LINE: `job_id in job_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job_id in job_ids:
            # LINE-BY-LINE: 조건 `job_id not in self.jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if job_id not in self.jobs:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._jobs_from_ids] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][CuttingSimulation._jobs_from_ids] "
                    # LINE-BY-LINE: `f"cause`에 `unknown_job_id job_id={job_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=unknown_job_id job_id={job_id}"
                )
                # LINE-BY-LINE: `KeyError(f"unknown job_id in batch: {job_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise KeyError(f"unknown job_id in batch: {job_id}")
            # LINE-BY-LINE: `jobs.append(self.jobs[job_id])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            jobs.append(self.jobs[job_id])
        # LINE-BY-LINE: 호출자에게 `tuple(jobs)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return tuple(jobs)

    # LINE-BY-LINE: `_batch_length_sum(self, jobs: Tuple[Job, ...])` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_length_sum(self, jobs: Tuple[Job, ...]) -> float:
        """batch 안 W/O의 `plate_length` 합을 계산한다."""

        # LINE-BY-LINE: 호출자에게 `sum(float(job.plate_length) for job in jobs)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return sum(float(job.plate_length) for job in jobs)

    # LINE-BY-LINE: `_batch_processing_minutes_for(self, jobs: Tuple[Job, ...], machine: Machine)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_processing_minutes_for(self, jobs: Tuple[Job, ...], machine: Machine) -> float:
        """batch가 machine을 점유하는 시간을 계산한다.

        입력:
        - `jobs`: 같은 batch에 들어갈 W/O 목록
        - `machine`: batch를 처리할 설비

        출력:
        - batch가 설비 슬롯을 점유하는 시간(minute)

        계산 예:
        - processing_time_rule=max_individual
        - J1 예상시간 30분, J2 예상시간 40분
        - 출력 40분

        현재 한계:
        - 실제 nesting id가 없으므로 "동시 절단"을 확정하는 모델은 아니다.
        - 그래도 DES state에서는 한 batch window가 machine slot 하나를 점유하도록 연결한다.
        """

        # LINE-BY-LINE: 조건 `not jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not jobs:
            # LINE-BY-LINE: 콘솔에 `print("[ERROR][CuttingSimulation._batch_processing_minutes_for] cause=empty_jobs")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[ERROR][CuttingSimulation._batch_processing_minutes_for] cause=empty_jobs")
            # LINE-BY-LINE: `ValueError("batch must contain at least one job")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("batch must contain at least one job")

        # LINE-BY-LINE: `individual_minutes`에 `[float(job.estimate_total_minutes(machine)) for job in jobs]` 결과를 저장합니다. 의미/사용: `individual_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        individual_minutes = [float(job.estimate_total_minutes(machine)) for job in jobs]
        # LINE-BY-LINE: `rule`에 `self._batch_processing_rule()` 결과를 저장합니다. 의미/사용: `rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rule = self._batch_processing_rule()
        # LINE-BY-LINE: 조건 `rule == "max_individual"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if rule == "max_individual":
            # LINE-BY-LINE: 호출자에게 `max(individual_minutes)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return max(individual_minutes)
        # LINE-BY-LINE: 조건 `rule == "sum_individual"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if rule == "sum_individual":
            # LINE-BY-LINE: 호출자에게 `sum(individual_minutes)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return sum(individual_minutes)

        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._batch_processing_minutes_for] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][CuttingSimulation._batch_processing_minutes_for] "
            # LINE-BY-LINE: `f"cause`에 `unsupported_rule rule={rule}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=unsupported_rule rule={rule}"
        )
        # LINE-BY-LINE: `ValueError(f"unsupported batch processing rule: {rule}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"unsupported batch processing rule: {rule}")

    # LINE-BY-LINE: `_batch_candidate_timing(self, jobs: Tuple[Job, ...], machine: Machine)` 함수를 정의합니다. 반환 타입: `Dict[str, float | Optional[float]]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_candidate_timing(self, jobs: Tuple[Job, ...], machine: Machine) -> Dict[str, float | Optional[float]]:
        """batch 후보를 지금 close하면 발생할 시간 정보를 계산한다.

        단건 `_candidate_timing()`과 같은 shape의 dict를 반환한다.
        그래서 기존 `ConstraintContext` 생성 흐름을 재사용할 수 있다.

        반환 예:
        ```python
        {
            "start_time": 0.0,
            "processing_minutes": 42.0,
            "elapsed_minutes": 42.0,
            "finish_time": 42.0,
            "changeover_minutes": 0.0,
            "blocked_minutes": 0.0,
            "downstream_arrival_time": 42.0,
            "downstream_release_time": None,
        }
        ```
        """

        # LINE-BY-LINE: `start_time`에 `self.state.current_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
        start_time = self.state.current_time
        # LINE-BY-LINE: `processing_minutes`에 `self._batch_processing_minutes_for(jobs, machine)` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
        processing_minutes = self._batch_processing_minutes_for(jobs, machine)
        # LINE-BY-LINE: `finish_time, blocked_minutes` 여러 변수에 `self._effective_finish_time(` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        finish_time, blocked_minutes = self._effective_finish_time(
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `machine.machine_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.machine_id,
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            start_time,
            # LINE-BY-LINE: `_effective_finish_time(...)` 호출에 `processing_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            processing_minutes,
        )
        # LINE-BY-LINE: `representative_job`에 `jobs[0]` 결과를 저장합니다. 의미/사용: `representative_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        representative_job = jobs[0]
        # LINE-BY-LINE: `bay`에 `self.bays[representative_job.downstream_bay]` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
        bay = self.bays[representative_job.downstream_bay]
        # LINE-BY-LINE: `arrival_time`에 `finish_time + float(bay.transfer_time_minutes)` 결과를 저장합니다. 의미/사용: `arrival_time`는 절단 완료 후 downstream bay 도착 예정 minute입니다.
        arrival_time = finish_time + float(bay.transfer_time_minutes)
        # LINE-BY-LINE: `release_time`에 `None` 결과를 저장합니다. 의미/사용: `release_time`는 downstream buffer에서 빠지는 예정 minute입니다.
        release_time = None
        # LINE-BY-LINE: 조건 `bay.release_delay_minutes is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if bay.release_delay_minutes is not None:
            # LINE-BY-LINE: `release_time`에 `arrival_time + float(bay.release_delay_minutes)` 결과를 저장합니다. 의미/사용: `release_time`는 downstream buffer에서 빠지는 예정 minute입니다.
            release_time = arrival_time + float(bay.release_delay_minutes)

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `start_time` 키에 `start_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "start_time": start_time,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `processing_minutes` 키에 `processing_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "processing_minutes": processing_minutes,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `elapsed_minutes` 키에 `finish_time - start_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "elapsed_minutes": finish_time - start_time,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `finish_time` 키에 `finish_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "finish_time": finish_time,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `changeover_minutes` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "changeover_minutes": 0.0,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `blocked_minutes` 키에 `blocked_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "blocked_minutes": blocked_minutes,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `downstream_arrival_time` 키에 `arrival_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_arrival_time": arrival_time,
            # LINE-BY-LINE: `_batch_candidate_timing`에서 반환/저장할 dict의 `downstream_release_time` 키에 `release_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_release_time": release_time,
        }

    # LINE-BY-LINE: `reset(self)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def reset(self) -> None:
        """새 에피소드 시작."""

        # LINE-BY-LINE: 현재 객체의 `state` 속성에 `self._build_initial_state()` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state = self._build_initial_state()
        # LINE-BY-LINE: `self._sync_all_machine_availability()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._sync_all_machine_availability()
        # LINE-BY-LINE: 현재 객체의 `_ensure_day_state(self._current_day_key(at_time` 속성에 `0.0))` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self._ensure_day_state(self._current_day_key(at_time=0.0))
        # LINE-BY-LINE: 현재 객체의 `_prune_resource_allocations(at_time` 속성에 `0.0)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self._prune_resource_allocations(at_time=0.0)
        # LINE-BY-LINE: 현재 객체의 `_apply_due_downstream_events(at_time` 속성에 `0.0)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self._apply_due_downstream_events(at_time=0.0)
        # LINE-BY-LINE: `self._advance_to_decision_epoch()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._advance_to_decision_epoch()

    # LINE-BY-LINE: `_make_operation_id(self, operation: ScheduledOperation)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _make_operation_id(self, operation: ScheduledOperation) -> str:
        """operation과 event를 연결하기 위한 안정적인 ID."""

        # LINE-BY-LINE: 호출자에게 `(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return (
            # LINE-BY-LINE: `f"{operation.job_id}@{operation.machine_id}@"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"{operation.job_id}@{operation.machine_id}@"
            # LINE-BY-LINE: `f"{operation.start_time:.6f}-{operation.finish_time:.6f}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            f"{operation.start_time:.6f}-{operation.finish_time:.6f}"
        )

    # LINE-BY-LINE: `emit_event` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def emit_event(
        # LINE-BY-LINE: `emit_event(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `event_type`를 `str,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
        event_type: str,
        # LINE-BY-LINE: `time_min`를 `float,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
        time_min: float,
        # LINE-BY-LINE: `emit_event(...)` 호출에 `*` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        *,
        # LINE-BY-LINE: `job_id` 변수에 `None` 결과를 저장합니다. 의미: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
        job_id: Optional[str] = None,
        # LINE-BY-LINE: `machine_id` 변수에 `None` 결과를 저장합니다. 의미: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id: Optional[str] = None,
        # LINE-BY-LINE: `bay_id` 변수에 `None` 결과를 저장합니다. 의미: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id: Optional[str] = None,
        # LINE-BY-LINE: `resource_id` 변수에 `None` 결과를 저장합니다. 의미: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        resource_id: Optional[str] = None,
        # LINE-BY-LINE: `operation_id` 변수에 `None` 결과를 저장합니다. 의미: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_id: Optional[str] = None,
        # LINE-BY-LINE: `machine_slot_index` 변수에 `None` 결과를 저장합니다. 의미: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_slot_index: Optional[int] = None,
        # LINE-BY-LINE: `source` 변수에 `SOURCE_GENERATED` 결과를 저장합니다. 의미: `source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        source: str = SOURCE_GENERATED,
        # LINE-BY-LINE: `payload` 변수에 `None` 결과를 저장합니다. 의미: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
        payload: Optional[Dict] = None,
        # LINE-BY-LINE: `message` 변수에 `""` 결과를 저장합니다. 의미: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
        message: str = "",
    # LINE-BY-LINE: `) -> str:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ) -> str:
        """공장 event를 생성해 event_log에 추가한다.

        실패 시 조용히 넘기지 않는다. event log는 DES 검증의 기준이므로
        schema나 id 문제가 있으면 즉시 원인을 출력하고 중단한다.
        """

        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: 조건 `job_id is not None and str(job_id) not in self.jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if job_id is not None and str(job_id) not in self.jobs:
                # LINE-BY-LINE: `KeyError(f"unknown job_id: {job_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise KeyError(f"unknown job_id: {job_id}")
            # LINE-BY-LINE: 조건 `machine_id is not None and str(machine_id) not in self.machines`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if machine_id is not None and str(machine_id) not in self.machines:
                # LINE-BY-LINE: `KeyError(f"unknown machine_id: {machine_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise KeyError(f"unknown machine_id: {machine_id}")

            # LINE-BY-LINE: `normalized_machine_id`에 `None if machine_id in (None, "") else str(machine_id)` 결과를 저장합니다. 의미/사용: `normalized_machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            normalized_machine_id = None if machine_id in (None, "") else str(machine_id)
            # LINE-BY-LINE: `normalized_bay_id`에 `None if bay_id in (None, "") else str(bay_id)` 결과를 저장합니다. 의미/사용: `normalized_bay_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            normalized_bay_id = None if bay_id in (None, "") else str(bay_id)
            # LINE-BY-LINE: 조건 `normalized_machine_id and normalized_bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if normalized_machine_id and normalized_bay_id:
                # LINE-BY-LINE: `machine_bay_id`에 `self.machines[normalized_machine_id].bay_id` 결과를 저장합니다. 의미/사용: `machine_bay_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_bay_id = self.machines[normalized_machine_id].bay_id
                # LINE-BY-LINE: `machine_bound_events`에 `{` 결과를 저장합니다. 의미/사용: `machine_bound_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_bound_events = {
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_MACHINE_ASSIGN,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_CUT_BAY_ASSIGN,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_PROCESS_START,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_PROCESS_FINISH,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_RESOURCE_ACQUIRE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_RESOURCE_ACQUIRE,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_RESOURCE_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    EVENT_RESOURCE_RELEASE,
                }
                # LINE-BY-LINE: 조건 `(`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if (
                    # LINE-BY-LINE: `event_type in machine_bound_events` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    event_type in machine_bound_events
                    # LINE-BY-LINE: `and machine_bay_id not in (None, "")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    and machine_bay_id not in (None, "")
                    # LINE-BY-LINE: `and str(machine_bay_id) != normalized_bay_id`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    and str(machine_bay_id) != normalized_bay_id
                # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                ):
                    # LINE-BY-LINE: `ValueError(` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                    raise ValueError(
                        # LINE-BY-LINE: `f"machine/bay mismatch: machine_id`에 `{normalized_machine_id}, "` 결과를 저장합니다. 의미/사용: `f"machine/bay mismatch: machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        f"machine/bay mismatch: machine_id={normalized_machine_id}, "
                        # LINE-BY-LINE: `f"machine_bay_id`에 `{machine_bay_id}, event_bay_id={normalized_bay_id}"` 결과를 저장합니다. 의미/사용: `f"machine_bay_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        f"machine_bay_id={machine_bay_id}, event_bay_id={normalized_bay_id}"
                    )

            # LINE-BY-LINE: `self.state.event_seq` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `event_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            self.state.event_seq += 1
            # LINE-BY-LINE: `event`에 `FactoryEvent(` 결과를 저장합니다. 의미/사용: `event`는 FactoryEvent 객체 또는 event row입니다. playback과 validation에 사용됩니다.
            event = FactoryEvent(
                # LINE-BY-LINE: `event_id`에 `f"E{self.state.event_seq:08d}"` 결과를 저장합니다. 의미/사용: `event_id`는 event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
                event_id=f"E{self.state.event_seq:08d}",
                # LINE-BY-LINE: `event_type`에 `event_type` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                event_type=event_type,
                # LINE-BY-LINE: `time_min`에 `float(time_min)` 결과를 저장합니다. 의미/사용: `time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
                time_min=float(time_min),
                # LINE-BY-LINE: `job_id`에 `None if job_id in (None, "") else str(job_id)` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=None if job_id in (None, "") else str(job_id),
                # LINE-BY-LINE: `machine_id`에 `normalized_machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id=normalized_machine_id,
                # LINE-BY-LINE: `bay_id`에 `normalized_bay_id` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=normalized_bay_id,
                # LINE-BY-LINE: `resource_id`에 `None if resource_id in (None, "") else str(resource_id)` 결과를 저장합니다. 의미/사용: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                resource_id=None if resource_id in (None, "") else str(resource_id),
                # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                operation_id=operation_id,
                # LINE-BY-LINE: `machine_slot_index`에 `machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_slot_index=machine_slot_index,
                # LINE-BY-LINE: `source`에 `source` 결과를 저장합니다. 의미/사용: `source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                source=source,
                # LINE-BY-LINE: `payload`에 `payload or {}` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                payload=payload or {},
                # LINE-BY-LINE: `message`에 `message` 결과를 저장합니다. 의미/사용: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                message=message,
            )
            # LINE-BY-LINE: `validate_event_required_fields(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            validate_event_required_fields(event)
            # LINE-BY-LINE: `self.state.event_log.append(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.event_log.append(event)
            # LINE-BY-LINE: 호출자에게 `event.event_id`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return event.event_id
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation.emit_event] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation.emit_event] "
                # LINE-BY-LINE: `f"cause`에 `{exc} event_type={event_type} job_id={job_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} event_type={event_type} job_id={job_id} "
                # LINE-BY-LINE: `f"machine_id`에 `{machine_id} bay_id={bay_id} time_min={time_min}"` 결과를 저장합니다. 의미/사용: `f"machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"machine_id={machine_id} bay_id={bay_id} time_min={time_min}"
            )
            # LINE-BY-LINE: `RuntimeError(` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(
                # LINE-BY-LINE: `f"failed to emit factory event: event_type`에 `{event_type}, job_id={job_id}"` 결과를 저장합니다. 의미/사용: `f"failed to emit factory event: event_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"failed to emit factory event: event_type={event_type}, job_id={job_id}"
            # LINE-BY-LINE: `) from exc` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            ) from exc

    # LINE-BY-LINE: `export_event_log(self)` 함수를 정의합니다. 반환 타입: `List[Dict]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
    def export_event_log(self) -> List[Dict]:
        """현재 event_log를 JSON/CSV 저장 가능한 dict list로 반환."""

        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: 호출자에게 `[event.to_dict() for event in sorted(self.state.event_log, key=event_sort_key)]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return [event.to_dict() for event in sorted(self.state.event_log, key=event_sort_key)]
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation.export_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation.export_event_log] "
                # LINE-BY-LINE: `f"cause`에 `{exc} event_count={len(self.state.event_log)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} event_count={len(self.state.event_log)}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to export event log") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to export event log") from exc

    # LINE-BY-LINE: `_emit_operation_events(self, operation: ScheduledOperation, selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
    def _emit_operation_events(self, operation: ScheduledOperation, selected: ActionCandidate) -> None:
        """선택된 operation에서 generated DES event를 생성한다."""

        # LINE-BY-LINE: `job`에 `self.jobs[operation.job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job = self.jobs[operation.job_id]
        # LINE-BY-LINE: `machine`에 `self.machines[operation.machine_id]` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine = self.machines[operation.machine_id]
        # LINE-BY-LINE: `cut_bay`에 `operation.cut_bay or machine.bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
        cut_bay = operation.cut_bay or machine.bay_id
        # LINE-BY-LINE: `operation_id`에 `self._make_operation_id(operation)` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_id = self._make_operation_id(operation)
        # LINE-BY-LINE: `common_payload`에 `{` 결과를 저장합니다. 의미/사용: `common_payload` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        common_payload = {
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `action_id` 키에 `selected.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_id": selected.action_id,
            # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.extra.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
            "work_order_no": job.extra.get("source_wk_ord_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `job.extra.get("source_project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
            "project_no": job.extra.get("source_project_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `job.extra.get("source_block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
            "block_no": job.extra.get("source_block_no", ""),
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `block_set_id` 키에 `job.block_set_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "block_set_id": job.block_set_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `job.plate_length` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
            "plate_length": job.plate_length,
            # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `job.thickness` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
            "thickness": job.thickness,
            # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `job.extra.get("part_count", "")` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
            "part_count": job.extra.get("part_count", ""),
            # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `job.actual_duration_minutes or ""` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
            "actual_duration_minutes": job.actual_duration_minutes or "",
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `estimated_minutes` 키에 `selected.estimated_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "estimated_minutes": selected.estimated_minutes,
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `processing_minutes` 키에 `operation.processing_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "processing_minutes": operation.processing_minutes,
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `changeover_minutes` 키에 `operation.changeover_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "changeover_minutes": operation.changeover_minutes,
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `blocked_minutes` 키에 `operation.blocked_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "blocked_minutes": operation.blocked_minutes,
            # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `job.source_machine_id or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
            "source_machine_id": job.source_machine_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `job.source_cut_bay or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
            "source_cut_bay": job.source_cut_bay or "",
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `action_kind` 키에 `selected.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_kind": selected.action_kind,
            # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `operation.batch_id or selected.batch_id or ""` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            "batch_id": operation.batch_id or selected.batch_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `"|".join(operation.batch_job_ids or selected.batch_job_ids)` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
            "batch_job_ids": "|".join(operation.batch_job_ids or selected.batch_job_ids),
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `batch_wo_count` 키에 `len(operation.batch_job_ids or selected.batch_job_ids or (operation.job_id,))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "batch_wo_count": len(operation.batch_job_ids or selected.batch_job_ids or (operation.job_id,)),
            # LINE-BY-LINE: 딕셔너리 키 `batch_length_sum`에는 `operation.batch_length_sum if operation.batch_length_sum is not None else ""` 값을 넣습니다. 의미: batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
            "batch_length_sum": operation.batch_length_sum if operation.batch_length_sum is not None else "",
            # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `batch_processing_rule` 키에 `selected.batch_processing_rule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "batch_processing_rule": selected.batch_processing_rule,
        }

        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_DECISION_EPOCH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_DECISION_EPOCH,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            operation.start_time,
            # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=operation.job_id,
            # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=operation.machine_id,
            # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=cut_bay,
            # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id=operation_id,
            # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_index=operation.machine_slot_index,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                **common_payload,
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `remaining_jobs_before` 키에 `len(self.state.unscheduled_jobs) + 1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "remaining_jobs_before": len(self.state.unscheduled_jobs) + 1,
            },
        )
        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_MACHINE_ASSIGN,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            operation.start_time,
            # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=operation.job_id,
            # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=operation.machine_id,
            # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=cut_bay,
            # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id=operation_id,
            # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_index=operation.machine_slot_index,
            # LINE-BY-LINE: `payload`에 `common_payload` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload=common_payload,
        )
        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_CUT_BAY_ASSIGN,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            operation.start_time,
            # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=operation.job_id,
            # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=operation.machine_id,
            # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=cut_bay,
            # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id=operation_id,
            # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_index=operation.machine_slot_index,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                **common_payload,
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `job_cut_bay` 키에 `job.cut_bay or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "job_cut_bay": job.cut_bay or "",
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `machine_bay_id` 키에 `machine.bay_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_bay_id": machine.bay_id or "",
            },
        )

        # LINE-BY-LINE: `resource_id in operation.resource_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id in operation.resource_ids:
            # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.emit_event(
                # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_RESOURCE_ACQUIRE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                EVENT_RESOURCE_ACQUIRE,
                # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                operation.start_time,
                # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=operation.job_id,
                # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id=operation.machine_id,
                # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=cut_bay,
                # LINE-BY-LINE: `resource_id`에 `resource_id` 결과를 저장합니다. 의미/사용: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                resource_id=resource_id,
                # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                operation_id=operation_id,
                # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_slot_index=operation.machine_slot_index,
                # LINE-BY-LINE: `payload`에 `common_payload` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                payload=common_payload,
            )

        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_PROCESS_START,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.start_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            operation.start_time,
            # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=operation.job_id,
            # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=operation.machine_id,
            # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=cut_bay,
            # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id=operation_id,
            # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_index=operation.machine_slot_index,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                **common_payload,
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `stage_minutes` 키에 `operation.stage_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "stage_minutes": operation.stage_minutes,
            },
        )
        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_PROCESS_FINISH,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.finish_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            operation.finish_time,
            # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=operation.job_id,
            # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=operation.machine_id,
            # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=cut_bay,
            # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id=operation_id,
            # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_slot_index=operation.machine_slot_index,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                **common_payload,
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `elapsed_minutes` 키에 `operation.finish_time - operation.start_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "elapsed_minutes": operation.finish_time - operation.start_time,
                # LINE-BY-LINE: `_emit_operation_events`에서 반환/저장할 dict의 `stage_minutes` 키에 `operation.stage_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "stage_minutes": operation.stage_minutes,
            },
        )

        # LINE-BY-LINE: `resource_id in operation.resource_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id in operation.resource_ids:
            # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.emit_event(
                # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_RESOURCE_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                EVENT_RESOURCE_RELEASE,
                # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.finish_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                operation.finish_time,
                # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=operation.job_id,
                # LINE-BY-LINE: `machine_id`에 `operation.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                machine_id=operation.machine_id,
                # LINE-BY-LINE: `bay_id`에 `cut_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=cut_bay,
                # LINE-BY-LINE: `resource_id`에 `resource_id` 결과를 저장합니다. 의미/사용: `resource_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                resource_id=resource_id,
                # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                operation_id=operation_id,
                # LINE-BY-LINE: `machine_slot_index`에 `operation.machine_slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                machine_slot_index=operation.machine_slot_index,
                # LINE-BY-LINE: `payload`에 `common_payload` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                payload=common_payload,
            )

        # LINE-BY-LINE: 조건 `operation.downstream_arrival_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if operation.downstream_arrival_time is not None:
            # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.emit_event(
                # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_DOWNSTREAM_ARRIVAL` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                EVENT_DOWNSTREAM_ARRIVAL,
                # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.downstream_arrival_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                operation.downstream_arrival_time,
                # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=operation.job_id,
                # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=operation.downstream_bay,
                # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                operation_id=operation_id,
                # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                payload={
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    **common_payload,
                    # LINE-BY-LINE: 딕셔너리 키 `downstream_bay`에는 `operation.downstream_bay` 값을 넣습니다. 의미: 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                    "downstream_bay": operation.downstream_bay,
                },
            )
        # LINE-BY-LINE: 조건 `operation.downstream_release_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if operation.downstream_release_time is not None:
            # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.emit_event(
                # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_DOWNSTREAM_RELEASE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                EVENT_DOWNSTREAM_RELEASE,
                # LINE-BY-LINE: `emit_event(...)` 호출에 `operation.downstream_release_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                operation.downstream_release_time,
                # LINE-BY-LINE: `job_id`에 `operation.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                job_id=operation.job_id,
                # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                bay_id=operation.downstream_bay,
                # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                operation_id=operation_id,
                # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                payload={
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**common_payload` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    **common_payload,
                    # LINE-BY-LINE: 딕셔너리 키 `downstream_bay`에는 `operation.downstream_bay` 값을 넣습니다. 의미: 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                    "downstream_bay": operation.downstream_bay,
                },
            )

    # LINE-BY-LINE: `_is_machine_available(self, machine_id: str)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _is_machine_available(self, machine_id: str) -> bool:
        """현재 시각에 해당 설비가 비어 있는지 확인합니다."""

        # LINE-BY-LINE: 호출자에게 `self.state.machine_available_at[machine_id] <= self.state.current_time`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.state.machine_available_at[machine_id] <= self.state.current_time

    # LINE-BY-LINE: `_build_constraint_context` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _build_constraint_context(
        # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `job`를 `Job,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job: Job,
        # LINE-BY-LINE: `machine`를 `Machine,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine: Machine,
        # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `*` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        *,
        # LINE-BY-LINE: `batch_jobs` 변수에 `None` 결과를 저장합니다. 의미: `batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_jobs: Optional[Tuple[Job, ...]] = None,
        # LINE-BY-LINE: `batch_id` 변수에 `None` 결과를 저장합니다. 의미: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id: Optional[str] = None,
        # LINE-BY-LINE: `batch_status` 변수에 `None` 결과를 저장합니다. 의미: `batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_status: Optional[str] = None,
        # LINE-BY-LINE: `timing_override` 변수에 `None` 결과를 저장합니다. 의미: `timing_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        timing_override: Optional[Dict[str, float | Optional[float]]] = None,
    # LINE-BY-LINE: `) -> ConstraintContext:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ) -> ConstraintContext:
        """제약 평가에 필요한 문맥 객체를 만든다.

        import/호출 흐름:
        - `Environment/constraints/base.py`의 `ConstraintContext`를 import해서 반환한다.
        - 반환된 context는 `ConstraintManager.evaluate_candidate()`로 들어간다.
        - 실제 rule 함수는 `Environment/constraints/*.py`에서 이 context만 보고 판단한다.

        단건 입력 예:
        ```python
        context = self._build_constraint_context(job=J1, machine=PLS21)
        ```

        batch 입력 예:
        ```python
        context = self._build_constraint_context(
            job=J1,
            machine=PLS21,
            batch_jobs=(J1, J2, J3),
            batch_id="B000001",
            batch_status="closing",
            timing_override=batch_timing,
        )
        ```

        batch에서 `job`과 `batch_jobs`를 둘 다 받는 이유:
        - 기존 rule 함수들은 `context.job` 하나를 기준으로 작성되어 있다.
        - 따라서 batch도 각 W/O별로 기존 rule을 반복 평가한다.
        - 동시에 batch 전체 수/길이 제약은 `candidate_batch_*` 필드로 전달한다.
        """

        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)
        # LINE-BY-LINE: `normalized_batch_jobs`에 `tuple(batch_jobs or (job,))` 결과를 저장합니다. 의미/사용: `normalized_batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized_batch_jobs = tuple(batch_jobs or (job,))
        # LINE-BY-LINE: `timing`에 `timing_override or self._candidate_timing(job, machine)` 결과를 저장합니다. 의미/사용: `timing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        timing = timing_override or self._candidate_timing(job, machine)
        # LINE-BY-LINE: `required_resource_ids`에 `self._required_resource_ids(job, machine)` 결과를 저장합니다. 의미/사용: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        required_resource_ids = self._required_resource_ids(job, machine)
        # LINE-BY-LINE: `resource_shortages`에 `self._resource_shortages_for(required_resource_ids)` 결과를 저장합니다. 의미/사용: `resource_shortages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        resource_shortages = self._resource_shortages_for(required_resource_ids)
        # LINE-BY-LINE: `global_calendar_open, global_calendar_reason` 여러 변수에 `self._is_global_calendar_open()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        global_calendar_open, global_calendar_reason = self._is_global_calendar_open()
        # LINE-BY-LINE: `machine_calendar_open, machine_calendar_reason` 여러 변수에 `self._machine_calendar_status(machine.machine_id)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        machine_calendar_open, machine_calendar_reason = self._machine_calendar_status(machine.machine_id)
        # LINE-BY-LINE: `machine_breakdown_active, machine_breakdown_reason` 여러 변수에 `self._machine_breakdown_status(machine.machine_id)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        machine_breakdown_active, machine_breakdown_reason = self._machine_breakdown_status(machine.machine_id)
        # LINE-BY-LINE: `concurrent_stats`에 `self._machine_concurrent_stats(` 결과를 저장합니다. 의미/사용: `concurrent_stats` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        concurrent_stats = self._machine_concurrent_stats(
            # LINE-BY-LINE: `_machine_concurrent_stats(...)` 호출에 `machine.machine_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.machine_id,
            # LINE-BY-LINE: `_machine_concurrent_stats(...)` 호출에 `float(timing["start_time"])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            float(timing["start_time"]),
        )
        # LINE-BY-LINE: `batch_length_sum`에 `self._batch_length_sum(normalized_batch_jobs)` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
        batch_length_sum = self._batch_length_sum(normalized_batch_jobs)
        # LINE-BY-LINE: `batch_wo_count`에 `len(normalized_batch_jobs)` 결과를 저장합니다. 의미/사용: `batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_wo_count = len(normalized_batch_jobs)
        # LINE-BY-LINE: `machine_wo_limit`에 `self._get_machine_day_wo_count_limit(machine.machine_id)` 결과를 저장합니다. 의미/사용: `machine_wo_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_wo_limit = self._get_machine_day_wo_count_limit(machine.machine_id)
        # LINE-BY-LINE: `machine_length_limit`에 `self._get_machine_day_length_sum_limit(machine.machine_id)` 결과를 저장합니다. 의미/사용: `machine_length_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_length_limit = self._get_machine_day_length_sum_limit(machine.machine_id)
        # LINE-BY-LINE: `batch_wo_limit`에 `machine.max_batch_wo_count if machine.max_batch_wo_count is not None else machine_wo_limit` 결과를 저장합니다. 의미/사용: `batch_wo_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_wo_limit = machine.max_batch_wo_count if machine.max_batch_wo_count is not None else machine_wo_limit
        # LINE-BY-LINE: `batch_length_limit`에 `(` 결과를 저장합니다. 의미/사용: `batch_length_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_length_limit = (
            # LINE-BY-LINE: `machine.max_batch_length_sum` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            machine.max_batch_length_sum
            # LINE-BY-LINE: 조건 `machine.max_batch_length_sum is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if machine.max_batch_length_sum is not None
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else machine_length_limit
        )
        # LINE-BY-LINE: `machine_priority_tier`에 `self._machine_priority_tier(job, machine)` 결과를 저장합니다. 의미/사용: `machine_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_priority_tier = self._machine_priority_tier(job, machine)
        # LINE-BY-LINE: `bay_priority_tier`에 `self._bay_priority_tier(job, machine)` 결과를 저장합니다. 의미/사용: `bay_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bay_priority_tier = self._bay_priority_tier(job, machine)
        # LINE-BY-LINE: `downstream_priority_tier`에 `self._downstream_priority_tier(job)` 결과를 저장합니다. 의미/사용: `downstream_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        downstream_priority_tier = self._downstream_priority_tier(job)
        # LINE-BY-LINE: `downstream_due_minutes`에 `job.downstream_due_minutes` 결과를 저장합니다. 의미/사용: `downstream_due_minutes`는 후공정 요구시각입니다. 사용: downstream_due_date hard rule 후보.
        downstream_due_minutes = job.downstream_due_minutes
        # LINE-BY-LINE: `downstream_slack`에 `(` 결과를 저장합니다. 의미/사용: `downstream_slack` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        downstream_slack = (
            # LINE-BY-LINE: `None` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            None
            # LINE-BY-LINE: 조건 `downstream_due_minutes is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if downstream_due_minutes is None
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else float(downstream_due_minutes) - float(timing["downstream_arrival_time"])
        )
        # LINE-BY-LINE: 호출자에게 `ConstraintContext(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ConstraintContext(
            # LINE-BY-LINE: `job`에 `job` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
            job=job,
            # LINE-BY-LINE: `machine`에 `machine` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
            machine=machine,
            # LINE-BY-LINE: `state`에 `self.state` 결과를 저장합니다. 의미/사용: `state` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            state=self.state,
            # LINE-BY-LINE: `jobs`에 `self.jobs` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            jobs=self.jobs,
            # LINE-BY-LINE: `machines`에 `self.machines` 결과를 저장합니다. 의미/사용: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machines=self.machines,
            # LINE-BY-LINE: `bays`에 `self.bays` 결과를 저장합니다. 의미/사용: `bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bays=self.bays,
            # LINE-BY-LINE: `layout`에 `self.layout` 결과를 저장합니다. 의미/사용: `layout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            layout=self.layout,
            # LINE-BY-LINE: `config`에 `self.config` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
            config=self.config,
            # LINE-BY-LINE: `current_day_key`에 `current_day_key` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            current_day_key=current_day_key,
            # LINE-BY-LINE: `current_day_index`에 `self._current_day_index()` 결과를 저장합니다. 의미/사용: `current_day_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            current_day_index=self._current_day_index(),
            # LINE-BY-LINE: `machine_enabled_flag`에 `self._get_machine_enabled_flag(machine.machine_id)` 결과를 저장합니다. 의미/사용: `machine_enabled_flag` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_enabled_flag=self._get_machine_enabled_flag(machine.machine_id),
            # LINE-BY-LINE: `machine_daily_capacity_limit`에 `self._get_machine_daily_capacity_limit(machine.machine_id)` 결과를 저장합니다. 의미/사용: `machine_daily_capacity_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_capacity_limit=self._get_machine_daily_capacity_limit(machine.machine_id),
            # LINE-BY-LINE: `machine_daily_load`에 `self.state.machine_daily_loads[current_day_key][machine.machine_id]` 결과를 저장합니다. 의미/사용: `machine_daily_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_load=self.state.machine_daily_loads[current_day_key][machine.machine_id],
            # LINE-BY-LINE: `machine_daily_job_count`에 `self.state.machine_daily_job_counts[current_day_key][machine.machine_id]` 결과를 저장합니다. 의미/사용: `machine_daily_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_job_count=self.state.machine_daily_job_counts[current_day_key][machine.machine_id],
            # LINE-BY-LINE: `machine_daily_length_sum`에 `self.state.machine_daily_length_sums[current_day_key][machine.machine_id]` 결과를 저장합니다. 의미/사용: `machine_daily_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_daily_length_sum=self.state.machine_daily_length_sums[current_day_key][machine.machine_id],
            # LINE-BY-LINE: `machine_day_wo_count_limit`에 `machine_wo_limit` 결과를 저장합니다. 의미/사용: `machine_day_wo_count_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_day_wo_count_limit=machine_wo_limit,
            # LINE-BY-LINE: `machine_day_length_sum_limit`에 `machine_length_limit` 결과를 저장합니다. 의미/사용: `machine_day_length_sum_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_day_length_sum_limit=machine_length_limit,
            # LINE-BY-LINE: `candidate_start_time`에 `float(timing["start_time"])` 결과를 저장합니다. 의미/사용: `candidate_start_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_start_time=float(timing["start_time"]),
            # LINE-BY-LINE: `machine_concurrent_job_count`에 `int(concurrent_stats["job_count"]) + max(batch_wo_count - 1, 0)` 결과를 저장합니다. 의미/사용: `machine_concurrent_job_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_concurrent_job_count=int(concurrent_stats["job_count"]) + max(batch_wo_count - 1, 0),
            # LINE-BY-LINE: `machine_concurrent_length_sum`에 `(` 결과를 저장합니다. 의미/사용: `machine_concurrent_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_concurrent_length_sum=(
                # LINE-BY-LINE: `float(concurrent_stats["length_sum"]) + max(batch_length_sum - float(job.plate_length), 0.0)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                float(concurrent_stats["length_sum"]) + max(batch_length_sum - float(job.plate_length), 0.0)
            ),
            # LINE-BY-LINE: `machine_bay_id`에 `machine.bay_id` 결과를 저장합니다. 의미/사용: `machine_bay_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_bay_id=machine.bay_id,
            # LINE-BY-LINE: `job_cut_bay`에 `job.cut_bay` 결과를 저장합니다. 의미/사용: `job_cut_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            job_cut_bay=job.cut_bay,
            # LINE-BY-LINE: `block_set_id`에 `job.block_set_id` 결과를 저장합니다. 의미/사용: `block_set_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            block_set_id=job.block_set_id,
            # LINE-BY-LINE: `assigned_block_set_bay`에 `(` 결과를 저장합니다. 의미/사용: `assigned_block_set_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            assigned_block_set_bay=(
                # LINE-BY-LINE: `None` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                None
                # LINE-BY-LINE: 조건 `not job.block_set_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not job.block_set_id
                # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                else self.state.block_set_bay_assignments.get(job.block_set_id)
            ),
            # LINE-BY-LINE: `scheduled_job_count_today`에 `self.state.scheduled_job_count_by_day[current_day_key]` 결과를 저장합니다. 의미/사용: `scheduled_job_count_today` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            scheduled_job_count_today=self.state.scheduled_job_count_by_day[current_day_key],
            # LINE-BY-LINE: `daily_job_cap`에 `self._get_daily_job_cap()` 결과를 저장합니다. 의미/사용: `daily_job_cap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            daily_job_cap=self._get_daily_job_cap(),
            # LINE-BY-LINE: `global_calendar_open`에 `global_calendar_open` 결과를 저장합니다. 의미/사용: `global_calendar_open` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            global_calendar_open=global_calendar_open,
            # LINE-BY-LINE: `global_calendar_reason`에 `global_calendar_reason` 결과를 저장합니다. 의미/사용: `global_calendar_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            global_calendar_reason=global_calendar_reason,
            # LINE-BY-LINE: `machine_calendar_open`에 `machine_calendar_open` 결과를 저장합니다. 의미/사용: `machine_calendar_open` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_calendar_open=machine_calendar_open,
            # LINE-BY-LINE: `machine_calendar_reason`에 `machine_calendar_reason` 결과를 저장합니다. 의미/사용: `machine_calendar_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_calendar_reason=machine_calendar_reason,
            # LINE-BY-LINE: `machine_breakdown_active`에 `machine_breakdown_active` 결과를 저장합니다. 의미/사용: `machine_breakdown_active` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_breakdown_active=machine_breakdown_active,
            # LINE-BY-LINE: `machine_breakdown_reason`에 `machine_breakdown_reason` 결과를 저장합니다. 의미/사용: `machine_breakdown_reason` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_breakdown_reason=machine_breakdown_reason,
            # LINE-BY-LINE: `candidate_processing_minutes`에 `float(timing["processing_minutes"])` 결과를 저장합니다. 의미/사용: `candidate_processing_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_processing_minutes=float(timing["processing_minutes"]),
            # LINE-BY-LINE: `candidate_elapsed_minutes`에 `float(timing["elapsed_minutes"])` 결과를 저장합니다. 의미/사용: `candidate_elapsed_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_elapsed_minutes=float(timing["elapsed_minutes"]),
            # LINE-BY-LINE: `candidate_finish_time`에 `float(timing["finish_time"])` 결과를 저장합니다. 의미/사용: `candidate_finish_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_finish_time=float(timing["finish_time"]),
            # LINE-BY-LINE: `candidate_downstream_arrival_time`에 `float(timing["downstream_arrival_time"])` 결과를 저장합니다. 의미/사용: `candidate_downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_downstream_arrival_time=float(timing["downstream_arrival_time"]),
            # LINE-BY-LINE: `candidate_downstream_release_time`에 `timing["downstream_release_time"]` 결과를 저장합니다. 의미/사용: `candidate_downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_downstream_release_time=timing["downstream_release_time"],
            # LINE-BY-LINE: `candidate_changeover_minutes`에 `float(timing["changeover_minutes"])` 결과를 저장합니다. 의미/사용: `candidate_changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_changeover_minutes=float(timing["changeover_minutes"]),
            # LINE-BY-LINE: `candidate_blocked_minutes`에 `float(timing["blocked_minutes"])` 결과를 저장합니다. 의미/사용: `candidate_blocked_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_blocked_minutes=float(timing["blocked_minutes"]),
            # LINE-BY-LINE: `predicted_downstream_load_at_arrival`에 `(` 결과를 저장합니다. 의미/사용: `predicted_downstream_load_at_arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            predicted_downstream_load_at_arrival=(
                # LINE-BY-LINE: `self._project_downstream_load(job.downstream_bay, float(timing["downstream_arrival_time"])) + 1`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self._project_downstream_load(job.downstream_bay, float(timing["downstream_arrival_time"])) + 1
            ),
            # LINE-BY-LINE: `required_resource_ids`에 `required_resource_ids` 결과를 저장합니다. 의미/사용: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            required_resource_ids=required_resource_ids,
            # LINE-BY-LINE: `resource_shortages`에 `resource_shortages` 결과를 저장합니다. 의미/사용: `resource_shortages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            resource_shortages=resource_shortages,
            # LINE-BY-LINE: `candidate_batch_id`에 `batch_id or job.batch_group_id or job.nesting_id or job.window_group_id` 결과를 저장합니다. 의미/사용: `candidate_batch_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_batch_id=batch_id or job.batch_group_id or job.nesting_id or job.window_group_id,
            # LINE-BY-LINE: `candidate_batch_job_ids`에 `tuple(batch_job.job_id for batch_job in normalized_batch_jobs)` 결과를 저장합니다. 의미/사용: `candidate_batch_job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_batch_job_ids=tuple(batch_job.job_id for batch_job in normalized_batch_jobs),
            # LINE-BY-LINE: `candidate_batch_wo_count`에 `int(concurrent_stats["job_count"]) + batch_wo_count` 결과를 저장합니다. 의미/사용: `candidate_batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_batch_wo_count=int(concurrent_stats["job_count"]) + batch_wo_count,
            # LINE-BY-LINE: `candidate_batch_length_sum`에 `float(concurrent_stats["length_sum"]) + batch_length_sum` 결과를 저장합니다. 의미/사용: `candidate_batch_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_batch_length_sum=float(concurrent_stats["length_sum"]) + batch_length_sum,
            # LINE-BY-LINE: `candidate_batch_status`에 `batch_status or "single_dispatch"` 결과를 저장합니다. 의미/사용: `candidate_batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_batch_status=batch_status or "single_dispatch",
            # LINE-BY-LINE: `batch_wo_count_limit`에 `batch_wo_limit` 결과를 저장합니다. 의미/사용: `batch_wo_count_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_wo_count_limit=batch_wo_limit,
            # LINE-BY-LINE: `batch_length_sum_limit`에 `batch_length_limit` 결과를 저장합니다. 의미/사용: `batch_length_sum_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_length_sum_limit=batch_length_limit,
            # LINE-BY-LINE: `machine_priority_tier`에 `machine_priority_tier` 결과를 저장합니다. 의미/사용: `machine_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_priority_tier=machine_priority_tier,
            # LINE-BY-LINE: `bay_priority_tier`에 `bay_priority_tier` 결과를 저장합니다. 의미/사용: `bay_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bay_priority_tier=bay_priority_tier,
            # LINE-BY-LINE: `downstream_priority_tier`에 `downstream_priority_tier` 결과를 저장합니다. 의미/사용: `downstream_priority_tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_priority_tier=downstream_priority_tier,
            # LINE-BY-LINE: `has_primary_machine_tier_candidate`에 `self._has_primary_machine_tier_candidate(job)` 결과를 저장합니다. 의미/사용: `has_primary_machine_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            has_primary_machine_tier_candidate=self._has_primary_machine_tier_candidate(job),
            # LINE-BY-LINE: `has_primary_bay_tier_candidate`에 `self._has_primary_bay_tier_candidate(job)` 결과를 저장합니다. 의미/사용: `has_primary_bay_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            has_primary_bay_tier_candidate=self._has_primary_bay_tier_candidate(job),
            # LINE-BY-LINE: `has_primary_downstream_tier_candidate`에 `self._has_primary_downstream_tier_candidate()` 결과를 저장합니다. 의미/사용: `has_primary_downstream_tier_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            has_primary_downstream_tier_candidate=self._has_primary_downstream_tier_candidate(),
            # LINE-BY-LINE: `table_type`에 `machine.table_type` 결과를 저장합니다. 의미/사용: `table_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            table_type=machine.table_type,
            # LINE-BY-LINE: `allowed_table_types`에 `job.allowed_table_types` 결과를 저장합니다. 의미/사용: `allowed_table_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            allowed_table_types=job.allowed_table_types,
            # LINE-BY-LINE: `plate_width`에 `job.plate_width` 결과를 저장합니다. 의미/사용: `plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            plate_width=job.plate_width,
            # LINE-BY-LINE: `machine_min_plate_width`에 `machine.min_plate_width` 결과를 저장합니다. 의미/사용: `machine_min_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_min_plate_width=machine.min_plate_width,
            # LINE-BY-LINE: `machine_max_plate_width`에 `machine.max_plate_width` 결과를 저장합니다. 의미/사용: `machine_max_plate_width` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_max_plate_width=machine.max_plate_width,
            # LINE-BY-LINE: `downstream_due_minutes`에 `downstream_due_minutes` 결과를 저장합니다. 의미/사용: `downstream_due_minutes`는 후공정 요구시각입니다. 사용: downstream_due_date hard rule 후보.
            downstream_due_minutes=downstream_due_minutes,
            # LINE-BY-LINE: `downstream_required_start_minutes`에 `job.downstream_required_start_minutes` 결과를 저장합니다. 의미/사용: `downstream_required_start_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_required_start_minutes=job.downstream_required_start_minutes,
            # LINE-BY-LINE: `candidate_downstream_slack_minutes`에 `downstream_slack` 결과를 저장합니다. 의미/사용: `candidate_downstream_slack_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_downstream_slack_minutes=downstream_slack,
            # LINE-BY-LINE: `predicted_storage_load_at_arrival`에 `(` 결과를 저장합니다. 의미/사용: `predicted_storage_load_at_arrival` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            predicted_storage_load_at_arrival=(
                # LINE-BY-LINE: `self._project_downstream_load(job.downstream_bay, float(timing["downstream_arrival_time"])) + 1`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self._project_downstream_load(job.downstream_bay, float(timing["downstream_arrival_time"])) + 1
            ),
        )

    # LINE-BY-LINE: `@staticmethod` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @staticmethod
    # LINE-BY-LINE: `_tier_from_mapping(mapping: Dict[str, int], *keys: Optional[str])` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _tier_from_mapping(mapping: Dict[str, int], *keys: Optional[str]) -> Optional[int]:
        """우선순위 mapping에서 후보 key 순서대로 tier를 읽는다."""

        # LINE-BY-LINE: `key in keys` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for key in keys:
            # LINE-BY-LINE: 조건 `key is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key is None:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `normalized_key`에 `str(key)` 결과를 저장합니다. 의미/사용: `normalized_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            normalized_key = str(key)
            # LINE-BY-LINE: 조건 `normalized_key in mapping`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if normalized_key in mapping:
                # LINE-BY-LINE: 호출자에게 `int(mapping[normalized_key])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return int(mapping[normalized_key])
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None

    # LINE-BY-LINE: `_machine_priority_tier(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _machine_priority_tier(self, job: Job, machine: Machine) -> Optional[int]:
        """작업-설비 우선순위 tier를 읽는다.

        현업 matrix가 아직 없으면 None으로 둔다. 이 값이 필요한 rule을 켜면
        future rule이 None을 통과시키지 않고 원인을 반환한다.
        """

        # LINE-BY-LINE: 호출자에게 `self._tier_from_mapping(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._tier_from_mapping(
            # LINE-BY-LINE: `_tier_from_mapping(...)` 호출에 `job.machine_priority_tiers` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            job.machine_priority_tiers,
            # LINE-BY-LINE: `_tier_from_mapping(...)` 호출에 `machine.machine_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.machine_id,
            # LINE-BY-LINE: `_tier_from_mapping(...)` 호출에 `machine.equipment_type` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.equipment_type,
            # LINE-BY-LINE: `_tier_from_mapping(...)` 호출에 `machine.machine_type` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine.machine_type,
        # LINE-BY-LINE: `) or self._tier_from_mapping(machine.priority_tiers_by_family, job.family)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        ) or self._tier_from_mapping(machine.priority_tiers_by_family, job.family)

    # LINE-BY-LINE: `_bay_priority_tier(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _bay_priority_tier(self, job: Job, machine: Machine) -> Optional[int]:
        """작업-Bay 우선순위 tier를 읽는다."""

        # LINE-BY-LINE: 호출자에게 `self._tier_from_mapping(job.bay_priority_tiers, machine.bay_id)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._tier_from_mapping(job.bay_priority_tiers, machine.bay_id)

    # LINE-BY-LINE: `_downstream_priority_tier(self, job: Job)` 함수를 정의합니다. 반환 타입: `Optional[int]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _downstream_priority_tier(self, job: Job) -> Optional[int]:
        """작업-후공정 Bay 우선순위 tier를 읽는다."""

        # LINE-BY-LINE: 호출자에게 `self._tier_from_mapping(job.downstream_priority_tiers, job.downstream_bay)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._tier_from_mapping(job.downstream_priority_tiers, job.downstream_bay)

    # LINE-BY-LINE: `_has_primary_machine_tier_candidate(self, job: Job)` 함수를 정의합니다. 반환 타입: `Optional[bool]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_primary_machine_tier_candidate(self, job: Job) -> Optional[bool]:
        """tier 1/2 설비 후보가 남아 있는지 확인한다."""

        # LINE-BY-LINE: 조건 `not job.machine_priority_tiers`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not job.machine_priority_tiers:
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None
        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._passes_static_candidate_prefilter(job, machine):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `tier`에 `self._machine_priority_tier(job, machine)` 결과를 저장합니다. 의미/사용: `tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tier = self._machine_priority_tier(job, machine)
            # LINE-BY-LINE: 조건 `tier is not None and int(tier) <= 2`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if tier is not None and int(tier) <= 2:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True
        # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False

    # LINE-BY-LINE: `_has_primary_bay_tier_candidate(self, job: Job)` 함수를 정의합니다. 반환 타입: `Optional[bool]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_primary_bay_tier_candidate(self, job: Job) -> Optional[bool]:
        """tier 1/2 Bay 후보가 남아 있는지 확인한다."""

        # LINE-BY-LINE: 조건 `not job.bay_priority_tiers`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not job.bay_priority_tiers:
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None
        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._passes_static_candidate_prefilter(job, machine):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `tier`에 `self._bay_priority_tier(job, machine)` 결과를 저장합니다. 의미/사용: `tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tier = self._bay_priority_tier(job, machine)
            # LINE-BY-LINE: 조건 `tier is not None and int(tier) <= 2`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if tier is not None and int(tier) <= 2:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True
        # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False

    # LINE-BY-LINE: `_has_primary_downstream_tier_candidate(self)` 함수를 정의합니다. 반환 타입: `Optional[bool]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_primary_downstream_tier_candidate(self) -> Optional[bool]:
        """미배정 작업 중 후공정 tier 1/2가 남아 있는지 확인한다."""

        # LINE-BY-LINE: `seen_tier_data`에 `False` 결과를 저장합니다. 의미/사용: `seen_tier_data` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        seen_tier_data = False
        # LINE-BY-LINE: `job_id in self.state.unscheduled_jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job_id in self.state.unscheduled_jobs:
            # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
            job = self.jobs[job_id]
            # LINE-BY-LINE: `tier`에 `self._downstream_priority_tier(job)` 결과를 저장합니다. 의미/사용: `tier` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tier = self._downstream_priority_tier(job)
            # LINE-BY-LINE: 조건 `tier is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if tier is None:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `seen_tier_data`에 `True` 결과를 저장합니다. 의미/사용: `seen_tier_data` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            seen_tier_data = True
            # LINE-BY-LINE: 조건 `int(tier) <= 2`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if int(tier) <= 2:
                # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return True
        # LINE-BY-LINE: 호출자에게 `False if seen_tier_data else None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False if seen_tier_data else None

    # LINE-BY-LINE: `_evaluate_candidate(self, context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `CandidateConstraintBundle`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _evaluate_candidate(self, context: ConstraintContext) -> CandidateConstraintBundle:
        """후보 action 1개에 대해 하드/소프트 제약을 함께 평가합니다."""

        # LINE-BY-LINE: 호출자에게 `self.constraint_manager.evaluate_candidate(context)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.constraint_manager.evaluate_candidate(context)

    # LINE-BY-LINE: `_build_action_candidate_from_context` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _build_action_candidate_from_context(
        # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `job`를 `Job,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job: Job,
        # LINE-BY-LINE: `machine`를 `Machine,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine: Machine,
        # LINE-BY-LINE: `context`를 `ConstraintContext,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.context` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        context: ConstraintContext,
        # LINE-BY-LINE: `bundle`를 `CandidateConstraintBundle,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.bundle` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        bundle: CandidateConstraintBundle,
        # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `*` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        *,
        # LINE-BY-LINE: `action_kind` 변수에 `"single_dispatch"` 결과를 저장합니다. 의미: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_kind: str = "single_dispatch",
        # LINE-BY-LINE: `action_id` 변수에 `None` 결과를 저장합니다. 의미: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_id: Optional[str] = None,
        # LINE-BY-LINE: `batch_id` 변수에 `None` 결과를 저장합니다. 의미: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id: Optional[str] = None,
        # LINE-BY-LINE: `batch_jobs` 변수에 `None` 결과를 저장합니다. 의미: `batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_jobs: Optional[Tuple[Job, ...]] = None,
    # LINE-BY-LINE: `) -> ActionCandidate:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ) -> ActionCandidate:
        """제약 평가가 끝난 context를 실행 후보 객체로 변환한다.

        입력:
        - `job`: 대표 W/O 또는 이번 action으로 새로 추가되는 W/O
        - `machine`: action 대상 설비
        - `context`: 제약 평가용 입력 문맥
        - `bundle`: hard/soft 제약 평가 결과 묶음
        - `action_kind`: `single_dispatch`, `open_batch`, `add_to_batch`, `close_batch`
        - `batch_jobs`: batch 후보에 포함될 전체 W/O 목록

        출력:
        - `ActionCandidate`

        예:
        - 단건: `action_id="J1@PLS21"`, `batch_job_ids=("J1",)`
        - batch close: `action_id="close_batch:B000001@PLS21"`, `batch_job_ids=("J1", "J2")`
        """

        # LINE-BY-LINE: `normalized_batch_jobs`에 `tuple(batch_jobs or (job,))` 결과를 저장합니다. 의미/사용: `normalized_batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        normalized_batch_jobs = tuple(batch_jobs or (job,))
        # LINE-BY-LINE: `batch_job_ids`에 `tuple(batch_job.job_id for batch_job in normalized_batch_jobs)` 결과를 저장합니다. 의미/사용: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
        batch_job_ids = tuple(batch_job.job_id for batch_job in normalized_batch_jobs)
        # LINE-BY-LINE: `batch_length_sum`에 `self._batch_length_sum(normalized_batch_jobs)` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
        batch_length_sum = self._batch_length_sum(normalized_batch_jobs)

        # LINE-BY-LINE: 호출자에게 `ActionCandidate(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ActionCandidate(
            # LINE-BY-LINE: `action_id`에 `action_id or f"{job.job_id}@{machine.machine_id}"` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            action_id=action_id or f"{job.job_id}@{machine.machine_id}",
            # LINE-BY-LINE: `job`에 `job` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
            job=job,
            # LINE-BY-LINE: `machine`에 `machine` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
            machine=machine,
            # LINE-BY-LINE: `estimated_minutes`에 `float(context.candidate_elapsed_minutes)` 결과를 저장합니다. 의미/사용: `estimated_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            estimated_minutes=float(context.candidate_elapsed_minutes),
            # LINE-BY-LINE: `processing_minutes`에 `float(context.candidate_processing_minutes)` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
            processing_minutes=float(context.candidate_processing_minutes),
            # LINE-BY-LINE: `finish_time`에 `float(context.candidate_finish_time)` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
            finish_time=float(context.candidate_finish_time),
            # LINE-BY-LINE: `downstream_arrival_time`에 `float(context.candidate_downstream_arrival_time)` 결과를 저장합니다. 의미/사용: `downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_arrival_time=float(context.candidate_downstream_arrival_time),
            # LINE-BY-LINE: `downstream_release_time`에 `context.candidate_downstream_release_time` 결과를 저장합니다. 의미/사용: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_release_time=context.candidate_downstream_release_time,
            # LINE-BY-LINE: `changeover_minutes`에 `float(context.candidate_changeover_minutes)` 결과를 저장합니다. 의미/사용: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            changeover_minutes=float(context.candidate_changeover_minutes),
            # LINE-BY-LINE: `blocked_minutes`에 `float(context.candidate_blocked_minutes)` 결과를 저장합니다. 의미/사용: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
            blocked_minutes=float(context.candidate_blocked_minutes),
            # LINE-BY-LINE: `required_resource_ids`에 `tuple(context.required_resource_ids)` 결과를 저장합니다. 의미/사용: `required_resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            required_resource_ids=tuple(context.required_resource_ids),
            # LINE-BY-LINE: `hard_reasons`에 `bundle.hard_reasons` 결과를 저장합니다. 의미/사용: `hard_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            hard_reasons=bundle.hard_reasons,
            # LINE-BY-LINE: `soft_reasons`에 `bundle.soft_reasons` 결과를 저장합니다. 의미/사용: `soft_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_reasons=bundle.soft_reasons,
            # LINE-BY-LINE: `soft_penalty`에 `bundle.soft_penalty` 결과를 저장합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_penalty=bundle.soft_penalty,
            # LINE-BY-LINE: `action_kind`에 `action_kind` 결과를 저장합니다. 의미/사용: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            action_kind=action_kind,
            # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=batch_id,
            # LINE-BY-LINE: `batch_job_ids`에 `batch_job_ids` 결과를 저장합니다. 의미/사용: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
            batch_job_ids=batch_job_ids,
            # LINE-BY-LINE: `batch_wo_count`에 `len(batch_job_ids)` 결과를 저장합니다. 의미/사용: `batch_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_wo_count=len(batch_job_ids),
            # LINE-BY-LINE: `batch_length_sum`에 `batch_length_sum` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
            batch_length_sum=batch_length_sum,
            # LINE-BY-LINE: `batch_processing_rule`에 `(` 결과를 저장합니다. 의미/사용: `batch_processing_rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_processing_rule=(
                # LINE-BY-LINE: `"single_job" if action_kind == "single_dispatch" else self._batch_processing_rule()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                "single_job" if action_kind == "single_dispatch" else self._batch_processing_rule()
            ),
        )

    # LINE-BY-LINE: `_hard_rule_active(self, rule_name: str)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _hard_rule_active(self, rule_name: str) -> bool:
        """config와 category를 함께 고려해 hard rule이 켜져 있는지 확인합니다."""

        # LINE-BY-LINE: `hard_enabled`에 `self.config.get("constraints", {}).get("hard_enabled", {})` 결과를 저장합니다. 의미/사용: `hard_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hard_enabled = self.config.get("constraints", {}).get("hard_enabled", {})
        # LINE-BY-LINE: 호출자에게 `bool(hard_enabled.get(rule_name, False)) and self.constraint_manager._is_category_enabled(rule_name)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return bool(hard_enabled.get(rule_name, False)) and self.constraint_manager._is_category_enabled(rule_name)

    # LINE-BY-LINE: `_passes_static_candidate_prefilter(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _passes_static_candidate_prefilter(self, job: Job, machine: Machine) -> bool:
        """시간 계산 없이 판정 가능한 hard rule을 먼저 걸러 후보 폭발을 줄입니다."""

        # LINE-BY-LINE: 조건 `self._hard_rule_active("machine_enabled") and not self._get_machine_enabled_flag(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("machine_enabled") and not self._get_machine_enabled_flag(machine.machine_id):
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False
        # LINE-BY-LINE: 조건 `self._hard_rule_active("family_eligibility") and job.family not in set(machine.eligible_families)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("family_eligibility") and job.family not in set(machine.eligible_families):
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False
        # LINE-BY-LINE: 조건 `self._hard_rule_active("thickness_range") and not (`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("thickness_range") and not (
            # LINE-BY-LINE: `machine.min_thickness <= job.thickness <= machine.max_thickness` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            machine.min_thickness <= job.thickness <= machine.max_thickness
        # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        ):
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False
        # LINE-BY-LINE: 조건 `self._hard_rule_active("table_length_limit") and job.plate_length > machine.table_length_limit`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("table_length_limit") and job.plate_length > machine.table_length_limit:
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False
        # LINE-BY-LINE: 조건 `self._hard_rule_active("machine_bay_consistency")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("machine_bay_consistency"):
            # LINE-BY-LINE: 조건 `not machine.bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not machine.bay_id:
                # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False
            # LINE-BY-LINE: 조건 `job.cut_bay and str(job.cut_bay) != str(machine.bay_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if job.cut_bay and str(job.cut_bay) != str(machine.bay_id):
                # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False
        # LINE-BY-LINE: 조건 `self._hard_rule_active("block_set_same_bay") and job.block_set_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._hard_rule_active("block_set_same_bay") and job.block_set_id:
            # LINE-BY-LINE: `assigned_bay`에 `self.state.block_set_bay_assignments.get(job.block_set_id)` 결과를 저장합니다. 의미/사용: `assigned_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            assigned_bay = self.state.block_set_bay_assignments.get(job.block_set_id)
            # LINE-BY-LINE: 조건 `assigned_bay and str(assigned_bay) != str(machine.bay_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if assigned_bay and str(assigned_bay) != str(machine.bay_id):
                # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                return False
        # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return True

    # LINE-BY-LINE: `_spt_sort_key_for(self, job: Job, machine: Machine)` 함수를 정의합니다. 반환 타입: `tuple[float, int, str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _spt_sort_key_for(self, job: Job, machine: Machine) -> tuple[float, int, str]:
        """SPT 휴리스틱의 1차 정렬에 쓰는 정적 key."""

        # LINE-BY-LINE: 호출자에게 `(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return (
            # LINE-BY-LINE: `return(...)` 호출에 `float(job.estimate_total_minutes(machine))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            float(job.estimate_total_minutes(machine)),
            # LINE-BY-LINE: `return(...)` 호출에 `-int(job.priority_weight)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            -int(job.priority_weight),
            # LINE-BY-LINE: `return(...)` 호출에 `job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            job.job_id,
        )

    # LINE-BY-LINE: `_get_spt_job_order(self, machine: Machine)` 함수를 정의합니다. 반환 타입: `List[str]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_spt_job_order(self, machine: Machine) -> List[str]:
        """설비별 SPT 정렬 job id 목록을 캐시합니다."""

        # LINE-BY-LINE: 조건 `machine.machine_id not in self._spt_job_order_by_machine`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine.machine_id not in self._spt_job_order_by_machine:
            # LINE-BY-LINE: 현재 객체의 `_spt_job_order_by_machine[machine.machine_id]` 속성에 `sorted(` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self._spt_job_order_by_machine[machine.machine_id] = sorted(
                # LINE-BY-LINE: `sorted(...)` 호출에 `self.jobs` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                self.jobs,
                # LINE-BY-LINE: `key`에 `lambda job_id: self._spt_sort_key_for(self.jobs[job_id], machine)` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                key=lambda job_id: self._spt_sort_key_for(self.jobs[job_id], machine),
            )
        # LINE-BY-LINE: 호출자에게 `self._spt_job_order_by_machine[machine.machine_id]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._spt_job_order_by_machine[machine.machine_id]

    # LINE-BY-LINE: `select_spt_candidate_fast(self)` 함수를 정의합니다. 반환 타입: `Optional[ActionCandidate]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def select_spt_candidate_fast(self) -> Optional[ActionCandidate]:
        """전체 후보 리스트를 만들지 않고 SPT 최선 후보 1개만 찾습니다."""

        # LINE-BY-LINE: `self._ensure_fast_heuristic_safe("select_spt_candidate_fast")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_fast_heuristic_safe("select_spt_candidate_fast")

        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `best_candidate` 변수에 `None` 결과를 저장합니다. 의미: `best_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        best_candidate: Optional[ActionCandidate] = None
        # LINE-BY-LINE: `best_key` 변수에 `None` 결과를 저장합니다. 의미: `best_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        best_key: Optional[tuple[float, float, int, str]] = None

        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `job_id in self._get_spt_job_order(machine)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in self._get_spt_job_order(machine):
                # LINE-BY-LINE: 조건 `job_id not in self.state.unscheduled_jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if job_id not in self.state.unscheduled_jobs:
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: `static_estimate`에 `float(job.estimate_total_minutes(machine))` 결과를 저장합니다. 의미/사용: `static_estimate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                static_estimate = float(job.estimate_total_minutes(machine))
                # LINE-BY-LINE: 조건 `best_key is not None and static_estimate > best_key[0] + 1e-9`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if best_key is not None and static_estimate > best_key[0] + 1e-9:
                    # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                    break
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue

                # LINE-BY-LINE: `context`에 `self._build_constraint_context(job, machine)` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                context = self._build_constraint_context(job, machine)
                # LINE-BY-LINE: `bundle`에 `self._evaluate_candidate(context)` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bundle = self._evaluate_candidate(context)
                # LINE-BY-LINE: 조건 `not bundle.hard_passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not bundle.hard_passed:
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue

                # LINE-BY-LINE: `candidate`에 `self._build_action_candidate_from_context(job, machine, context, bundle)` 결과를 저장합니다. 의미/사용: `candidate`는 현재 선택 가능 action 후보 1개입니다. 예: `job_001@PLS21`.
                candidate = self._build_action_candidate_from_context(job, machine, context, bundle)
                # LINE-BY-LINE: `candidate_key`에 `(` 결과를 저장합니다. 의미/사용: `candidate_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                candidate_key = (
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `float(candidate.estimated_minutes)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    float(candidate.estimated_minutes),
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `float(candidate.soft_penalty)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    float(candidate.soft_penalty),
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `-int(candidate.job.priority_weight)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    -int(candidate.job.priority_weight),
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `candidate.job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    candidate.job.job_id,
                )
                # LINE-BY-LINE: 조건 `best_key is None or candidate_key < best_key`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if best_key is None or candidate_key < best_key:
                    # LINE-BY-LINE: `best_candidate`에 `candidate` 결과를 저장합니다. 의미/사용: `best_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    best_candidate = candidate
                    # LINE-BY-LINE: `best_key`에 `candidate_key` 결과를 저장합니다. 의미/사용: `best_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    best_key = candidate_key

        # LINE-BY-LINE: 호출자에게 `best_candidate`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return best_candidate

    # LINE-BY-LINE: `_ensure_fast_heuristic_safe(self, function_name: str)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _ensure_fast_heuristic_safe(self, function_name: str) -> None:
        """fast selector가 정확도를 유지할 수 있는 전제조건을 확인합니다."""

        # LINE-BY-LINE: 조건 `self.config.get("setup", {}).get("enable_family_changeover", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self.config.get("setup", {}).get("enable_family_changeover", False):
            # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][CuttingSimulation.{function_name}] cause=family_changeover_enabled")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"[ERROR][CuttingSimulation.{function_name}] cause=family_changeover_enabled")
            # LINE-BY-LINE: `RuntimeError("fast heuristic requires family changeover disabled")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast heuristic requires family changeover disabled")
        # LINE-BY-LINE: 조건 `self._has_calendar_blocking_source()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._has_calendar_blocking_source():
            # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][CuttingSimulation.{function_name}] cause=calendar_blocking_source_enabled")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"[ERROR][CuttingSimulation.{function_name}] cause=calendar_blocking_source_enabled")
            # LINE-BY-LINE: `RuntimeError("fast heuristic requires blocking calendar sources disabled")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast heuristic requires blocking calendar sources disabled")

    # LINE-BY-LINE: `select_load_balance_candidate_fast(self)` 함수를 정의합니다. 반환 타입: `Optional[ActionCandidate]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def select_load_balance_candidate_fast(self) -> Optional[ActionCandidate]:
        """전체 후보 리스트를 만들지 않고 load_balance 최선 후보 1개만 찾습니다."""

        # LINE-BY-LINE: `self._ensure_fast_heuristic_safe("select_load_balance_candidate_fast")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_fast_heuristic_safe("select_load_balance_candidate_fast")
        # LINE-BY-LINE: `self._ensure_fast_load_balance_safe()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_fast_load_balance_safe()
        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `machine_order`에 `{machine_id: index for index, machine_id in enumerate(self.machines)}` 결과를 저장합니다. 의미/사용: `machine_order` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_order = {machine_id: index for index, machine_id in enumerate(self.machines)}
        # LINE-BY-LINE: `ordered_machines`에 `sorted(` 결과를 저장합니다. 의미/사용: `ordered_machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        ordered_machines = sorted(
            # LINE-BY-LINE: `sorted(...)` 호출에 `self.machines.values()` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            self.machines.values(),
            # LINE-BY-LINE: `key`에 `lambda machine: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda machine: (
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `float(self.state.machine_loads[machine.machine_id])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                float(self.state.machine_loads[machine.machine_id]),
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `machine_order[machine.machine_id]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                machine_order[machine.machine_id],
            ),
        )

        # LINE-BY-LINE: `best_candidate` 변수에 `None` 결과를 저장합니다. 의미: `best_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        best_candidate: Optional[ActionCandidate] = None
        # LINE-BY-LINE: `best_key` 변수에 `None` 결과를 저장합니다. 의미: `best_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        best_key: Optional[tuple[float, float, float, int, int, str]] = None

        # LINE-BY-LINE: `machine in ordered_machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in ordered_machines:
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `machine_load`에 `float(self.state.machine_loads[machine.machine_id])` 결과를 저장합니다. 의미/사용: `machine_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_load = float(self.state.machine_loads[machine.machine_id])
            # LINE-BY-LINE: 조건 `best_key is not None and machine_load > best_key[0] + 1e-9`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if best_key is not None and machine_load > best_key[0] + 1e-9:
                # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                break

            # LINE-BY-LINE: `job_id in sorted(self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in sorted(self.state.unscheduled_jobs):
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `processing_minutes`에 `float(job.estimate_total_minutes(machine))` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
                processing_minutes = float(job.estimate_total_minutes(machine))
                # LINE-BY-LINE: `finish_time`에 `float(self.state.current_time) + processing_minutes` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
                finish_time = float(self.state.current_time) + processing_minutes
                # LINE-BY-LINE: `soft_penalty`에 `self._fast_load_balance_soft_penalty(machine, job, finish_time)` 결과를 저장합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                soft_penalty = self._fast_load_balance_soft_penalty(machine, job, finish_time)
                # LINE-BY-LINE: `candidate_key`에 `(` 결과를 저장합니다. 의미/사용: `candidate_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                candidate_key = (
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `machine_load` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    machine_load,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `soft_penalty` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    soft_penalty,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `processing_minutes` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    processing_minutes,
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `-int(job.priority_weight)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    -int(job.priority_weight),
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `machine_order[machine.machine_id]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    machine_order[machine.machine_id],
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `job.job_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    job.job_id,
                )
                # LINE-BY-LINE: 조건 `best_key is None or candidate_key < best_key`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if best_key is None or candidate_key < best_key:
                    # LINE-BY-LINE: `context`에 `self._build_constraint_context(job, machine)` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    context = self._build_constraint_context(job, machine)
                    # LINE-BY-LINE: `bundle`에 `self._evaluate_candidate(context)` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    bundle = self._evaluate_candidate(context)
                    # LINE-BY-LINE: 조건 `not bundle.hard_passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if not bundle.hard_passed:
                        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                        print(
                            # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation.select_load_balance_candidate_fast] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                            "[ERROR][CuttingSimulation.select_load_balance_candidate_fast] "
                            # LINE-BY-LINE: `f"cause`에 `unexpected_dynamic_hard_failure job_id={job.job_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                            f"cause=unexpected_dynamic_hard_failure job_id={job.job_id} "
                            # LINE-BY-LINE: `f"machine_id`에 `{machine.machine_id} reasons={bundle.hard_reasons}"` 결과를 저장합니다. 의미/사용: `f"machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                            f"machine_id={machine.machine_id} reasons={bundle.hard_reasons}"
                        )
                        # LINE-BY-LINE: `RuntimeError("fast load_balance encountered unexpected dynamic hard failure")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                        raise RuntimeError("fast load_balance encountered unexpected dynamic hard failure")
                    # LINE-BY-LINE: `best_candidate`에 `self._build_action_candidate_from_context(job, machine, context, bundle)` 결과를 저장합니다. 의미/사용: `best_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    best_candidate = self._build_action_candidate_from_context(job, machine, context, bundle)
                    # LINE-BY-LINE: `best_key`에 `candidate_key` 결과를 저장합니다. 의미/사용: `best_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    best_key = candidate_key

        # LINE-BY-LINE: 호출자에게 `best_candidate`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return best_candidate

    # LINE-BY-LINE: `_ensure_fast_load_balance_safe(self)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _ensure_fast_load_balance_safe(self) -> None:
        """현재 fast load_balance가 generic load_balance와 같은 key를 내는 조건인지 확인."""

        # LINE-BY-LINE: `hard_enabled`에 `self.config.get("constraints", {}).get("hard_enabled", {})` 결과를 저장합니다. 의미/사용: `hard_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hard_enabled = self.config.get("constraints", {}).get("hard_enabled", {})
        # LINE-BY-LINE: 조건 `hard_enabled.get("daily_capacity_limit", False) or hard_enabled.get("daily_job_cap_limit", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if hard_enabled.get("daily_capacity_limit", False) or hard_enabled.get("daily_job_cap_limit", False):
            # LINE-BY-LINE: 콘솔에 `print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=daily_dynamic_rule_enabled")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=daily_dynamic_rule_enabled")
            # LINE-BY-LINE: `RuntimeError("fast load_balance requires daily dynamic hard rules disabled")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast load_balance requires daily dynamic hard rules disabled")
        # LINE-BY-LINE: 조건 `self.config.get("constraints", {}).get("hard_enabled", {}).get("downstream_capacity", False)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self.config.get("constraints", {}).get("hard_enabled", {}).get("downstream_capacity", False):
            # LINE-BY-LINE: 콘솔에 `print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=downstream_capacity_enabled")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=downstream_capacity_enabled")
            # LINE-BY-LINE: `RuntimeError("fast load_balance requires downstream capacity disabled")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast load_balance requires downstream capacity disabled")
        # LINE-BY-LINE: 조건 `any(job.required_resource_ids for job in self.jobs.values()) or any(`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if any(job.required_resource_ids for job in self.jobs.values()) or any(
            # LINE-BY-LINE: `machine.required_resource_ids for machine in self.machines.values()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            machine.required_resource_ids for machine in self.machines.values()
        # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        ):
            # LINE-BY-LINE: 콘솔에 `print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=required_resources_present")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] cause=required_resources_present")
            # LINE-BY-LINE: `RuntimeError("fast load_balance requires no auxiliary resource requirements")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast load_balance requires no auxiliary resource requirements")
        # LINE-BY-LINE: `active_soft`에 `{` 결과를 저장합니다. 의미/사용: `active_soft` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_soft = {
            # LINE-BY-LINE: `rule_name` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            rule_name
            # LINE-BY-LINE: `rule_name, enabled in self.config.get("constraints", {}).get("soft_enabled", {}).items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for rule_name, enabled in self.config.get("constraints", {}).get("soft_enabled", {}).items()
            # LINE-BY-LINE: 조건 `enabled and self.constraint_manager._is_category_enabled(rule_name)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if enabled and self.constraint_manager._is_category_enabled(rule_name)
        }
        # LINE-BY-LINE: `supported_soft`에 `{"due_date_urgency", "load_balance_preference"}` 결과를 저장합니다. 의미/사용: `supported_soft` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        supported_soft = {"due_date_urgency", "load_balance_preference"}
        # LINE-BY-LINE: `unsupported`에 `sorted(active_soft - supported_soft)` 결과를 저장합니다. 의미/사용: `unsupported` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        unsupported = sorted(active_soft - supported_soft)
        # LINE-BY-LINE: 조건 `unsupported`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if unsupported:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._ensure_fast_load_balance_safe] "
                # LINE-BY-LINE: `f"cause`에 `unsupported_soft_rules rules={unsupported}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unsupported_soft_rules rules={unsupported}"
            )
            # LINE-BY-LINE: `RuntimeError("fast load_balance supports due_date_urgency and load_balance_preference only")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast load_balance supports due_date_urgency and load_balance_preference only")

    # LINE-BY-LINE: `_fast_load_balance_soft_penalty(self, machine: Machine, job: Job, finish_time: float)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _fast_load_balance_soft_penalty(self, machine: Machine, job: Job, finish_time: float) -> float:
        """현재 config의 load_balance soft penalty를 context 생성 없이 계산합니다."""

        # LINE-BY-LINE: `soft_enabled`에 `self.config.get("constraints", {}).get("soft_enabled", {})` 결과를 저장합니다. 의미/사용: `soft_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        soft_enabled = self.config.get("constraints", {}).get("soft_enabled", {})
        # LINE-BY-LINE: `weights`에 `self.config.get("constraints", {}).get("soft_penalty_weights", {})` 결과를 저장합니다. 의미/사용: `weights` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        weights = self.config.get("constraints", {}).get("soft_penalty_weights", {})
        # LINE-BY-LINE: `penalty`에 `0.0` 결과를 저장합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        penalty = 0.0
        # LINE-BY-LINE: 조건 `soft_enabled.get("due_date_urgency", False) and self.constraint_manager._is_category_enabled("due...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if soft_enabled.get("due_date_urgency", False) and self.constraint_manager._is_category_enabled("due_date_urgency"):
            # LINE-BY-LINE: `due_date`에 `job.due_date_minutes` 결과를 저장합니다. 의미/사용: `due_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            due_date = job.due_date_minutes
            # LINE-BY-LINE: 조건 `due_date is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if due_date is not None:
                # LINE-BY-LINE: `slack`에 `float(due_date) - float(finish_time)` 결과를 저장합니다. 의미/사용: `slack` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                slack = float(due_date) - float(finish_time)
                # LINE-BY-LINE: 조건 `slack < 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if slack < 0:
                    # LINE-BY-LINE: `penalty` 값을 `(abs(slack) / max(float(due_date), 1.0)) * float(weights.get("due_date_urgency", 1.0))` 기준으로 누적/증가합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    penalty += (abs(slack) / max(float(due_date), 1.0)) * float(weights.get("due_date_urgency", 1.0))

        # LINE-BY-LINE: 조건 `soft_enabled.get("load_balance_preference", False) and self.constraint_manager._is_category_enabl...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if soft_enabled.get("load_balance_preference", False) and self.constraint_manager._is_category_enabled("load_balance_preference"):
            # LINE-BY-LINE: `machine_loads`에 `self.state.machine_loads` 결과를 저장합니다. 의미/사용: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            machine_loads = self.state.machine_loads
            # LINE-BY-LINE: `average_load`에 `sum(machine_loads.values()) / max(len(machine_loads), 1)` 결과를 저장합니다. 의미/사용: `average_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            average_load = sum(machine_loads.values()) / max(len(machine_loads), 1)
            # LINE-BY-LINE: `current_load`에 `machine_loads[machine.machine_id]` 결과를 저장합니다. 의미/사용: `current_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            current_load = machine_loads[machine.machine_id]
            # LINE-BY-LINE: `overload`에 `max(0.0, current_load - average_load)` 결과를 저장합니다. 의미/사용: `overload` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            overload = max(0.0, current_load - average_load)
            # LINE-BY-LINE: 조건 `overload > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if overload > 0:
                # LINE-BY-LINE: `penalty` 값을 `(overload / max(average_load + 1.0, 1.0)) * float(` 기준으로 누적/증가합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                penalty += (overload / max(average_load + 1.0, 1.0)) * float(
                    # LINE-BY-LINE: `weights.get("load_balance_preference", 1.0)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    weights.get("load_balance_preference", 1.0)
                )
        # LINE-BY-LINE: 호출자에게 `penalty`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return penalty

    # LINE-BY-LINE: `_evaluate_batch_jobs` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _evaluate_batch_jobs(
        # LINE-BY-LINE: `_evaluate_batch_jobs(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `jobs`를 `Tuple[Job, ...],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.jobs` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        jobs: Tuple[Job, ...],
        # LINE-BY-LINE: `machine`를 `Machine,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine: Machine,
        # LINE-BY-LINE: `_evaluate_batch_jobs(...)` 호출에 `*` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        *,
        # LINE-BY-LINE: `batch_id`를 `Optional[str],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id: Optional[str],
        # LINE-BY-LINE: `batch_status`를 `str,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.batch_status` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        batch_status: str,
        # LINE-BY-LINE: `timing`를 `Dict[str, float | Optional[float]],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.timing` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        timing: Dict[str, float | Optional[float]],
    # LINE-BY-LINE: `) -> tuple[bool, List[str], List[str], float]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ) -> tuple[bool, List[str], List[str], float]:
        """batch 후보에 포함된 모든 W/O의 제약을 평가한다.

        입력:
        - `jobs`: batch에 들어갈 W/O 전체
        - `machine`: batch가 올라갈 설비
        - `batch_id`: 이미 열린 batch면 B000001 같은 id, 새로 여는 후보면 None
        - `batch_status`: `opening`, `adding`, `closing` 중 하나
        - `timing`: `_batch_candidate_timing()`이 계산한 batch window 시간

        출력:
        - `hard_passed`: batch 안 모든 W/O가 hard 제약을 통과했는지
        - `hard_reasons`: 실패 원인 목록
        - `soft_reasons`: soft penalty 원인 목록
        - `soft_penalty`: batch 전체 soft penalty 합

        왜 모든 W/O를 반복 평가하는가:
        - 예를 들어 J1은 PLS21 가능, J2는 PLS21 불가능일 수 있다.
        - batch는 하나라도 불가능한 W/O가 있으면 후보가 되면 안 된다.
        - 따라서 대표 W/O 하나만 보고 통과시키면 안 된다.
        """

        # LINE-BY-LINE: `hard_reasons` 변수에 `[]` 결과를 저장합니다. 의미: `hard_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hard_reasons: List[str] = []
        # LINE-BY-LINE: `soft_reasons` 변수에 `[]` 결과를 저장합니다. 의미: `soft_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        soft_reasons: List[str] = []
        # LINE-BY-LINE: `soft_penalty`에 `0.0` 결과를 저장합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        soft_penalty = 0.0

        # LINE-BY-LINE: `job in jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job in jobs:
            # LINE-BY-LINE: `context`에 `self._build_constraint_context(` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            context = self._build_constraint_context(
                # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `job` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                job,
                # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `machine` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                machine,
                # LINE-BY-LINE: `batch_jobs`에 `jobs` 결과를 저장합니다. 의미/사용: `batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                batch_jobs=jobs,
                # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                batch_id=batch_id,
                # LINE-BY-LINE: `batch_status`에 `batch_status` 결과를 저장합니다. 의미/사용: `batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                batch_status=batch_status,
                # LINE-BY-LINE: `timing_override`에 `timing` 결과를 저장합니다. 의미/사용: `timing_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                timing_override=timing,
            )
            # LINE-BY-LINE: `bundle`에 `self._evaluate_candidate(context)` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bundle = self._evaluate_candidate(context)
            # LINE-BY-LINE: `hard_reasons.extend(bundle.hard_reasons)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            hard_reasons.extend(bundle.hard_reasons)
            # LINE-BY-LINE: `soft_reasons.extend(bundle.soft_reasons)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            soft_reasons.extend(bundle.soft_reasons)
            # LINE-BY-LINE: `soft_penalty` 값을 `float(bundle.soft_penalty)` 기준으로 누적/증가합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_penalty += float(bundle.soft_penalty)

        # LINE-BY-LINE: 호출자에게 `not hard_reasons, hard_reasons, soft_reasons, soft_penalty`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return not hard_reasons, hard_reasons, soft_reasons, soft_penalty

    # LINE-BY-LINE: `_batch_jobs_fit_limits(self, jobs: Tuple[Job, ...], machine: Machine)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _batch_jobs_fit_limits(self, jobs: Tuple[Job, ...], machine: Machine) -> bool:
        """W/O 수와 길이 합이 설비 batch 한계 안에 들어오는지 확인한다."""

        # LINE-BY-LINE: `max_wo_count, max_length_sum` 여러 변수에 `self._batch_limits_for_machine(machine)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        max_wo_count, max_length_sum = self._batch_limits_for_machine(machine)
        # LINE-BY-LINE: 호출자에게 `len(jobs) <= max_wo_count and self._batch_length_sum(jobs) <= max_length_sum`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return len(jobs) <= max_wo_count and self._batch_length_sum(jobs) <= max_length_sum

    # LINE-BY-LINE: `_make_batch_action_candidate` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _make_batch_action_candidate(
        # LINE-BY-LINE: `_make_batch_action_candidate(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `_make_batch_action_candidate(...)` 호출에 `*` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        *,
        # LINE-BY-LINE: `action_kind`를 `str,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.action_kind` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        action_kind: str,
        # LINE-BY-LINE: `machine`를 `Machine,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine: Machine,
        # LINE-BY-LINE: `jobs`를 `Tuple[Job, ...],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.jobs` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        jobs: Tuple[Job, ...],
        # LINE-BY-LINE: `action_id`를 `str,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.action_id` 필드/속성입니다. 사용: CuttingSimulation 객체를 만들거나 이후 로직에서 참조합니다.
        action_id: str,
        # LINE-BY-LINE: `batch_id`를 `Optional[str],` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id: Optional[str],
        # LINE-BY-LINE: `selected_job` 변수에 `None` 결과를 저장합니다. 의미: `selected_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        selected_job: Optional[Job] = None,
    # LINE-BY-LINE: `) -> Optional[ActionCandidate]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ) -> Optional[ActionCandidate]:
        """open/add/close batch action 후보 1개를 만든다.

        입력 예:
        ```python
        self._make_batch_action_candidate(
            action_kind="add_to_batch",
            machine=PLS21,
            jobs=(J1, J2),
            action_id="add_to_batch:B000001:J2",
            batch_id="B000001",
            selected_job=J2,
        )
        ```

        출력:
        - hard 제약을 모두 통과하면 `ActionCandidate`
        - 하나라도 hard 제약을 실패하면 `None`

        주의:
        - `None` 반환은 오류 은폐가 아니다.
        - action 후보 생성에서 hard 제약 실패는 "선택 불가능한 후보"라는 정상 의미다.
        - 실제 데이터/스키마 오류는 각 rule 또는 helper에서 print 후 raise 한다.
        """

        # LINE-BY-LINE: 조건 `not jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not jobs:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._make_batch_action_candidate] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._make_batch_action_candidate] "
                # LINE-BY-LINE: `f"cause`에 `empty_jobs action_kind={action_kind} machine_id={machine.machine_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=empty_jobs action_kind={action_kind} machine_id={machine.machine_id}"
            )
            # LINE-BY-LINE: `ValueError("batch action candidate requires at least one job")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("batch action candidate requires at least one job")

        # LINE-BY-LINE: 조건 `not self._batch_jobs_fit_limits(jobs, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self._batch_jobs_fit_limits(jobs, machine):
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None

        # LINE-BY-LINE: `timing`에 `self._batch_candidate_timing(jobs, machine)` 결과를 저장합니다. 의미/사용: `timing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        timing = self._batch_candidate_timing(jobs, machine)
        # LINE-BY-LINE: `hard_passed, hard_reasons, soft_reasons, soft_penalty` 여러 변수에 `self._evaluate_batch_jobs(` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        hard_passed, hard_reasons, soft_reasons, soft_penalty = self._evaluate_batch_jobs(
            # LINE-BY-LINE: `_evaluate_batch_jobs(...)` 호출에 `jobs` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            jobs,
            # LINE-BY-LINE: `_evaluate_batch_jobs(...)` 호출에 `machine` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine,
            # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=batch_id,
            # LINE-BY-LINE: `batch_status`에 `action_kind` 결과를 저장합니다. 의미/사용: `batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_status=action_kind,
            # LINE-BY-LINE: `timing`에 `timing` 결과를 저장합니다. 의미/사용: `timing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            timing=timing,
        )
        # LINE-BY-LINE: 조건 `not hard_passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not hard_passed:
            # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return None

        # LINE-BY-LINE: `representative_job`에 `selected_job or jobs[0]` 결과를 저장합니다. 의미/사용: `representative_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        representative_job = selected_job or jobs[0]
        # LINE-BY-LINE: `context`에 `self._build_constraint_context(` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        context = self._build_constraint_context(
            # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `representative_job` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            representative_job,
            # LINE-BY-LINE: `_build_constraint_context(...)` 호출에 `machine` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine,
            # LINE-BY-LINE: `batch_jobs`에 `jobs` 결과를 저장합니다. 의미/사용: `batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_jobs=jobs,
            # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=batch_id,
            # LINE-BY-LINE: `batch_status`에 `action_kind` 결과를 저장합니다. 의미/사용: `batch_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_status=action_kind,
            # LINE-BY-LINE: `timing_override`에 `timing` 결과를 저장합니다. 의미/사용: `timing_override` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            timing_override=timing,
        )
        # LINE-BY-LINE: `bundle`에 `CandidateConstraintBundle(` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bundle = CandidateConstraintBundle(
            # LINE-BY-LINE: `hard_results`에 `[]` 결과를 저장합니다. 의미/사용: `hard_results` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            hard_results=[],
            # LINE-BY-LINE: `soft_results`에 `[]` 결과를 저장합니다. 의미/사용: `soft_results` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_results=[],
        )
        # LINE-BY-LINE: `candidate`에 `self._build_action_candidate_from_context(` 결과를 저장합니다. 의미/사용: `candidate`는 현재 선택 가능 action 후보 1개입니다. 예: `job_001@PLS21`.
        candidate = self._build_action_candidate_from_context(
            # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `representative_job` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            representative_job,
            # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `machine` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            machine,
            # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `context` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            context,
            # LINE-BY-LINE: `_build_action_candidate_from_context(...)` 호출에 `bundle` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            bundle,
            # LINE-BY-LINE: `action_kind`에 `action_kind` 결과를 저장합니다. 의미/사용: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            action_kind=action_kind,
            # LINE-BY-LINE: `action_id`에 `action_id` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            action_id=action_id,
            # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=batch_id,
            # LINE-BY-LINE: `batch_jobs`에 `jobs` 결과를 저장합니다. 의미/사용: `batch_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            batch_jobs=jobs,
        )
        # LINE-BY-LINE: `candidate.soft_reasons`에 `soft_reasons` 결과를 저장합니다. 의미/사용: `soft_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate.soft_reasons = soft_reasons
        # LINE-BY-LINE: `candidate.soft_penalty`에 `soft_penalty` 결과를 저장합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate.soft_penalty = soft_penalty
        # LINE-BY-LINE: `candidate.hard_reasons`에 `hard_reasons` 결과를 저장합니다. 의미/사용: `hard_reasons` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate.hard_reasons = hard_reasons
        # LINE-BY-LINE: 호출자에게 `candidate`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return candidate

    # LINE-BY-LINE: `_get_batch_open_candidates(self)` 함수를 정의합니다. 반환 타입: `List[ActionCandidate]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _get_batch_open_candidates(self) -> List[ActionCandidate]:
        """현재 DES state에서 가능한 batch/open-batch action 후보를 생성한다.

        출력 action 종류:
        - `open_batch:JOB@MACHINE`
          idle machine에 새 batch를 열고 첫 W/O를 넣는다.

        - `add_to_batch:BATCH:JOB`
          이미 열린 batch에 W/O 1개를 더 넣는다.
          이때 W/O 수 <= 3, 길이합 <= 55000 같은 한계를 검사한다.

        - `close_batch:BATCH@MACHINE`
          open batch를 닫고, batch 안 모든 W/O를 같은 start/finish window로 schedule한다.

        DES state 연결:
        - 후보 생성은 `self.state.open_batches`를 읽는다.
        - open/add action은 `self.state.open_batches`를 갱신한다.
        - close action은 `self.state.schedule`, `machine_slot_available_at`, `event_log`를 갱신한다.

        왜 조합을 한 번에 만들지 않는가:
        - 100개 W/O에서 3개 조합을 모두 만들면 후보 수가 급증한다.
        - open-batch는 exact feasible interaction을 유지하면서도 action space를 작게 쪼갠다.
        """

        # LINE-BY-LINE: `candidates` 변수에 `[]` 결과를 저장합니다. 의미: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates: List[ActionCandidate] = []
        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `open_machine_ids`에 `{` 결과를 저장합니다. 의미/사용: `open_machine_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        open_machine_ids = {
            # LINE-BY-LINE: `batch.machine_id` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            batch.machine_id
            # LINE-BY-LINE: `batch in self.state.open_batches.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for batch in self.state.open_batches.values()
        }

        # LINE-BY-LINE: `batch_id, batch in sorted(self.state.open_batches.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for batch_id, batch in sorted(self.state.open_batches.items()):
            # LINE-BY-LINE: 조건 `batch.machine_id not in self.machines`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if batch.machine_id not in self.machines:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._get_batch_open_candidates] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][CuttingSimulation._get_batch_open_candidates] "
                    # LINE-BY-LINE: `f"cause`에 `unknown_batch_machine batch_id={batch_id} machine_id={batch.machine_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=unknown_batch_machine batch_id={batch_id} machine_id={batch.machine_id}"
                )
                # LINE-BY-LINE: `KeyError(f"unknown machine_id in open batch: {batch.machine_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise KeyError(f"unknown machine_id in open batch: {batch.machine_id}")
            # LINE-BY-LINE: `machine`에 `self.machines[batch.machine_id]` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
            machine = self.machines[batch.machine_id]
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._get_batch_open_candidates] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][CuttingSimulation._get_batch_open_candidates] "
                    # LINE-BY-LINE: `f"cause`에 `open_batch_machine_busy batch_id={batch_id} machine_id={machine.machine_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=open_batch_machine_busy batch_id={batch_id} machine_id={machine.machine_id}"
                )
                # LINE-BY-LINE: `RuntimeError(f"open batch machine became busy: {machine.machine_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise RuntimeError(f"open batch machine became busy: {machine.machine_id}")

            # LINE-BY-LINE: `existing_jobs`에 `self._jobs_from_ids(tuple(batch.job_ids))` 결과를 저장합니다. 의미/사용: `existing_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            existing_jobs = self._jobs_from_ids(tuple(batch.job_ids))
            # LINE-BY-LINE: `close_candidate`에 `self._make_batch_action_candidate(` 결과를 저장합니다. 의미/사용: `close_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            close_candidate = self._make_batch_action_candidate(
                # LINE-BY-LINE: `action_kind`에 `"close_batch"` 결과를 저장합니다. 의미/사용: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                action_kind="close_batch",
                # LINE-BY-LINE: `machine`에 `machine` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
                machine=machine,
                # LINE-BY-LINE: `jobs`에 `existing_jobs` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                jobs=existing_jobs,
                # LINE-BY-LINE: `action_id`에 `f"close_batch:{batch_id}@{machine.machine_id}"` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                action_id=f"close_batch:{batch_id}@{machine.machine_id}",
                # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                batch_id=batch_id,
            )
            # LINE-BY-LINE: 조건 `close_candidate is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if close_candidate is not None:
                # LINE-BY-LINE: `candidates.append(close_candidate)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                candidates.append(close_candidate)

            # LINE-BY-LINE: `job_id in sorted(self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in sorted(self.state.unscheduled_jobs):
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `next_jobs`에 `(*existing_jobs, job)` 결과를 저장합니다. 의미/사용: `next_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                next_jobs = (*existing_jobs, job)
                # LINE-BY-LINE: `add_candidate`에 `self._make_batch_action_candidate(` 결과를 저장합니다. 의미/사용: `add_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                add_candidate = self._make_batch_action_candidate(
                    # LINE-BY-LINE: `action_kind`에 `"add_to_batch"` 결과를 저장합니다. 의미/사용: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    action_kind="add_to_batch",
                    # LINE-BY-LINE: `machine`에 `machine` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
                    machine=machine,
                    # LINE-BY-LINE: `jobs`에 `next_jobs` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    jobs=next_jobs,
                    # LINE-BY-LINE: `action_id`에 `f"add_to_batch:{batch_id}:{job.job_id}"` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    action_id=f"add_to_batch:{batch_id}:{job.job_id}",
                    # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                    batch_id=batch_id,
                    # LINE-BY-LINE: `selected_job`에 `job` 결과를 저장합니다. 의미/사용: `selected_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    selected_job=job,
                )
                # LINE-BY-LINE: 조건 `add_candidate is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if add_candidate is not None:
                    # LINE-BY-LINE: `candidates.append(add_candidate)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    candidates.append(add_candidate)

        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `machine.machine_id in open_machine_ids`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if machine.machine_id in open_machine_ids:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `job_id in sorted(self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in sorted(self.state.unscheduled_jobs):
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `open_candidate`에 `self._make_batch_action_candidate(` 결과를 저장합니다. 의미/사용: `open_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                open_candidate = self._make_batch_action_candidate(
                    # LINE-BY-LINE: `action_kind`에 `"open_batch"` 결과를 저장합니다. 의미/사용: `action_kind` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    action_kind="open_batch",
                    # LINE-BY-LINE: `machine`에 `machine` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
                    machine=machine,
                    # LINE-BY-LINE: `jobs`에 `(job,)` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    jobs=(job,),
                    # LINE-BY-LINE: `action_id`에 `f"open_batch:{job.job_id}@{machine.machine_id}"` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    action_id=f"open_batch:{job.job_id}@{machine.machine_id}",
                    # LINE-BY-LINE: `batch_id`에 `None` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                    batch_id=None,
                    # LINE-BY-LINE: `selected_job`에 `job` 결과를 저장합니다. 의미/사용: `selected_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    selected_job=job,
                )
                # LINE-BY-LINE: 조건 `open_candidate is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if open_candidate is not None:
                    # LINE-BY-LINE: `candidates.append(open_candidate)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    candidates.append(open_candidate)

        # LINE-BY-LINE: 호출자에게 `candidates`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return candidates

    # LINE-BY-LINE: `get_candidates(self)` 함수를 정의합니다. 반환 타입: `List[ActionCandidate]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def get_candidates(self) -> List[ActionCandidate]:
        """현재 의사결정 시점에서 가능한 action 후보 목록을 만듭니다."""

        # LINE-BY-LINE: 조건 `self._is_batch_open_mode()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._is_batch_open_mode():
            # LINE-BY-LINE: 호출자에게 `self._get_batch_open_candidates()`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return self._get_batch_open_candidates()

        # LINE-BY-LINE: `candidates` 변수에 `[]` 결과를 저장합니다. 의미: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates: List[ActionCandidate] = []
        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # 현재 비어 있는 설비들에 대해서만 후보를 만듭니다.
        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `job_id in sorted(self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in sorted(self.state.unscheduled_jobs):
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `context`에 `self._build_constraint_context(job, machine)` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                context = self._build_constraint_context(job, machine)
                # LINE-BY-LINE: `bundle`에 `self._evaluate_candidate(context)` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bundle = self._evaluate_candidate(context)

                # 하드 제약 하나라도 실패하면 이 action은 후보에서 제거합니다.
                # LINE-BY-LINE: 조건 `not bundle.hard_passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not bundle.hard_passed:
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue

                # LINE-BY-LINE: `candidates.append(self._build_action_candidate_from_context(job, machine, context, bundle))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                candidates.append(self._build_action_candidate_from_context(job, machine, context, bundle))

        # LINE-BY-LINE: 호출자에게 `candidates`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return candidates

    # LINE-BY-LINE: `_has_any_candidate(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _has_any_candidate(self) -> bool:
        """후보 존재 여부만 확인합니다. 전체 후보 목록을 만들지 않습니다."""

        # LINE-BY-LINE: 조건 `self._is_batch_open_mode()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._is_batch_open_mode():
            # LINE-BY-LINE: 호출자에게 `bool(self._get_batch_open_candidates())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return bool(self._get_batch_open_candidates())

        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)

        # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in self.machines.values():
            # LINE-BY-LINE: 조건 `not self._is_machine_available(machine.machine_id)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_machine_available(machine.machine_id):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `job_id in sorted(self.state.unscheduled_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job_id in sorted(self.state.unscheduled_jobs):
                # LINE-BY-LINE: `job`에 `self.jobs[job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
                job = self.jobs[job_id]
                # LINE-BY-LINE: 조건 `not self._passes_static_candidate_prefilter(job, machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if not self._passes_static_candidate_prefilter(job, machine):
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue
                # LINE-BY-LINE: `context`에 `self._build_constraint_context(job, machine)` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                context = self._build_constraint_context(job, machine)
                # LINE-BY-LINE: `bundle`에 `self._evaluate_candidate(context)` 결과를 저장합니다. 의미/사용: `bundle` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                bundle = self._evaluate_candidate(context)
                # LINE-BY-LINE: 조건 `bundle.hard_passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if bundle.hard_passed:
                    # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
                    return True
        # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return False

    # LINE-BY-LINE: `get_action_map(self)` 함수를 정의합니다. 반환 타입: `Dict[str, ActionCandidate]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def get_action_map(self) -> Dict[str, ActionCandidate]:
        """action_id -> candidate 매핑."""

        # LINE-BY-LINE: 호출자에게 `{candidate.action_id: candidate for candidate in self.get_candidates()}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return {candidate.action_id: candidate for candidate in self.get_candidates()}

    # LINE-BY-LINE: `_current_completion_time(self)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _current_completion_time(self) -> float:
        """현재까지의 예상 전체 완료 시점을 계산합니다."""

        # LINE-BY-LINE: `slot_times`에 `[` 결과를 저장합니다. 의미/사용: `slot_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times = [
            # LINE-BY-LINE: `slot_time` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            slot_time
            # LINE-BY-LINE: `slot_list in self.state.machine_slot_available_at.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for slot_list in self.state.machine_slot_available_at.values()
            # LINE-BY-LINE: `slot_time in slot_list` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for slot_time in slot_list
        ]
        # LINE-BY-LINE: 조건 `slot_times`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if slot_times:
            # LINE-BY-LINE: 호출자에게 `max(slot_times)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return max(slot_times)
        # LINE-BY-LINE: 호출자에게 `self.state.current_time`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.state.current_time

    # LINE-BY-LINE: `get_makespan(self)` 함수를 정의합니다. 반환 타입: `float`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def get_makespan(self) -> float:
        """현재 스케줄의 makespan을 반환합니다.

        주의:
        - current_time은 "현재 의사결정 시점"입니다.
        - makespan은 "모든 설비 중 가장 늦은 완료 시각"입니다.
        최종 성능 평가는 makespan으로 보는 것이 맞습니다.
        """

        # LINE-BY-LINE: 호출자에게 `self._current_completion_time()`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._current_completion_time()

    # LINE-BY-LINE: `_build_scheduled_operation(self, candidate: ActionCandidate)` 함수를 정의합니다. 반환 타입: `ScheduledOperation`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _build_scheduled_operation(self, candidate: ActionCandidate) -> ScheduledOperation:
        """선택한 action으로부터 schedule 레코드를 만듭니다."""

        # LINE-BY-LINE: `start_time`에 `self.state.current_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
        start_time = self.state.current_time
        # LINE-BY-LINE: `finish_time`에 `candidate.finish_time` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
        finish_time = candidate.finish_time

        # LINE-BY-LINE: 호출자에게 `ScheduledOperation(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return ScheduledOperation(
            # LINE-BY-LINE: `job_id`에 `candidate.job.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=candidate.job.job_id,
            # LINE-BY-LINE: `machine_id`에 `candidate.machine.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=candidate.machine.machine_id,
            # LINE-BY-LINE: `start_time`에 `start_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
            start_time=start_time,
            # LINE-BY-LINE: `finish_time`에 `finish_time` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
            finish_time=finish_time,
            # LINE-BY-LINE: `downstream_bay`에 `candidate.job.downstream_bay` 결과를 저장합니다. 의미/사용: `downstream_bay`는 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
            downstream_bay=candidate.job.downstream_bay,
            # LINE-BY-LINE: `estimated_minutes`에 `candidate.estimated_minutes` 결과를 저장합니다. 의미/사용: `estimated_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            estimated_minutes=candidate.estimated_minutes,
            # LINE-BY-LINE: `cut_bay`에 `candidate.machine.bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
            cut_bay=candidate.machine.bay_id,
            # LINE-BY-LINE: `processing_minutes`에 `candidate.processing_minutes` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
            processing_minutes=candidate.processing_minutes,
            # LINE-BY-LINE: `changeover_minutes`에 `candidate.changeover_minutes` 결과를 저장합니다. 의미/사용: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            changeover_minutes=candidate.changeover_minutes,
            # LINE-BY-LINE: `blocked_minutes`에 `candidate.blocked_minutes` 결과를 저장합니다. 의미/사용: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
            blocked_minutes=candidate.blocked_minutes,
            # LINE-BY-LINE: `downstream_arrival_time`에 `candidate.downstream_arrival_time` 결과를 저장합니다. 의미/사용: `downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_arrival_time=candidate.downstream_arrival_time,
            # LINE-BY-LINE: `downstream_release_time`에 `candidate.downstream_release_time` 결과를 저장합니다. 의미/사용: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_release_time=candidate.downstream_release_time,
            # LINE-BY-LINE: `resource_ids`에 `tuple(candidate.required_resource_ids)` 결과를 저장합니다. 의미/사용: `resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            resource_ids=tuple(candidate.required_resource_ids),
            # LINE-BY-LINE: `batch_id`에 `candidate.batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=candidate.batch_id,
            # LINE-BY-LINE: `batch_job_ids`에 `tuple(candidate.batch_job_ids or (candidate.job.job_id,))` 결과를 저장합니다. 의미/사용: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
            batch_job_ids=tuple(candidate.batch_job_ids or (candidate.job.job_id,)),
            # LINE-BY-LINE: `batch_length_sum`에 `float(candidate.batch_length_sum or candidate.job.plate_length)` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
            batch_length_sum=float(candidate.batch_length_sum or candidate.job.plate_length),
            # LINE-BY-LINE: `stage_minutes`에 `{` 결과를 저장합니다. 의미/사용: `stage_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            stage_minutes={
                # LINE-BY-LINE: `_build_scheduled_operation`에서 반환/저장할 dict의 `setup` 키에 `candidate.job.base_stage_minutes.get("setup", 0.0) + candidate.changeover_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "setup": candidate.job.base_stage_minutes.get("setup", 0.0) + candidate.changeover_minutes,
                # LINE-BY-LINE: `_build_scheduled_operation`에서 반환/저장할 dict의 `cut` 키에 `candidate.job.base_stage_minutes.get("cut", 0.0) * candidate.machine.cut_speed_factor` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "cut": candidate.job.base_stage_minutes.get("cut", 0.0) * candidate.machine.cut_speed_factor,
                # LINE-BY-LINE: `_build_scheduled_operation`에서 반환/저장할 dict의 `finish` 키에 `candidate.job.base_stage_minutes.get("finish", 0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish": candidate.job.base_stage_minutes.get("finish", 0.0),
            },
        )

    # LINE-BY-LINE: `_apply_operation(self, operation: ScheduledOperation)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _apply_operation(self, operation: ScheduledOperation) -> None:
        """선택 결과를 상태에 반영합니다."""

        # LINE-BY-LINE: `operation_day_key`에 `self._current_day_key(at_time=operation.start_time)` 결과를 저장합니다. 의미/사용: `operation_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_day_key = self._current_day_key(at_time=operation.start_time)
        # LINE-BY-LINE: `self._ensure_day_state(operation_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(operation_day_key)
        # LINE-BY-LINE: `slot_times`에 `self.state.machine_slot_available_at[operation.machine_id]` 결과를 저장합니다. 의미/사용: `slot_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times = self.state.machine_slot_available_at[operation.machine_id]
        # LINE-BY-LINE: `slot_index`에 `min(` 결과를 저장합니다. 의미/사용: `slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_index = min(
            # LINE-BY-LINE: `min(...)` 호출에 `range(len(slot_times))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            range(len(slot_times)),
            # LINE-BY-LINE: `key`에 `lambda index: (slot_times[index], index)` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda index: (slot_times[index], index),
        )
        # LINE-BY-LINE: `slot_times[slot_index]`에 `operation.finish_time` 결과를 저장합니다. 의미/사용: `slot_times[slot_index]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times[slot_index] = operation.finish_time
        # LINE-BY-LINE: `operation.machine_slot_index`에 `slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation.machine_slot_index = slot_index
        # LINE-BY-LINE: `self._sync_machine_earliest_availability(operation.machine_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._sync_machine_earliest_availability(operation.machine_id)

        # LINE-BY-LINE: `self.state.schedule.append(operation)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.state.schedule.append(operation)
        # LINE-BY-LINE: `self.state.unscheduled_jobs.remove(operation.job_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.state.unscheduled_jobs.remove(operation.job_id)
        # LINE-BY-LINE: `self.state.machine_loads[operation.machine_id]` 값을 `operation.processing_minutes` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_loads[operation.machine_id] += operation.processing_minutes
        # LINE-BY-LINE: `self.state.machine_daily_loads[operation_day_key][operation.machine_id]` 값을 `operation.processing_minutes` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_loads[operation_day_key][operation.machine_id] += operation.processing_minutes
        # LINE-BY-LINE: `self.state.machine_daily_job_counts[operation_day_key][operation.machine_id]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_job_counts[operation_day_key][operation.machine_id] += 1
        # LINE-BY-LINE: `self.state.machine_daily_length_sums[operation_day_key][operation.machine_id]` 값을 `float(` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_length_sums[operation_day_key][operation.machine_id] += float(
            # LINE-BY-LINE: `self.jobs[operation.job_id].plate_length` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            self.jobs[operation.job_id].plate_length
        )
        # LINE-BY-LINE: `self.state.scheduled_job_count_by_day[operation_day_key]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `scheduled_job_count_by_day[operation_day_key]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.scheduled_job_count_by_day[operation_day_key] += 1
        # LINE-BY-LINE: 현재 객체의 `state.machine_last_family[operation.machine_id]` 속성에 `self.jobs[operation.job_id].family` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.machine_last_family[operation.machine_id] = self.jobs[operation.job_id].family

        # LINE-BY-LINE: `job`에 `self.jobs[operation.job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job = self.jobs[operation.job_id]
        # LINE-BY-LINE: `cut_bay`에 `operation.cut_bay or self.machines[operation.machine_id].bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
        cut_bay = operation.cut_bay or self.machines[operation.machine_id].bay_id
        # LINE-BY-LINE: 조건 `job.block_set_id and cut_bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if job.block_set_id and cut_bay:
            # LINE-BY-LINE: `self.state.block_set_bay_assignments.setdefault(job.block_set_id, str(cut_bay))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.block_set_bay_assignments.setdefault(job.block_set_id, str(cut_bay))

        # LINE-BY-LINE: `resource_id in operation.resource_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id in operation.resource_ids:
            # LINE-BY-LINE: `self.state.resource_active_until.setdefault(resource_id, []).append(operation.finish_time)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.resource_active_until.setdefault(resource_id, []).append(operation.finish_time)

        # LINE-BY-LINE: 조건 `operation.downstream_arrival_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if operation.downstream_arrival_time is not None:
            # LINE-BY-LINE: `self.state.downstream_events.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.downstream_events.append(
                # LINE-BY-LINE: `DownstreamEvent(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                DownstreamEvent(
                    # LINE-BY-LINE: `event_time`에 `float(operation.downstream_arrival_time)` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    event_time=float(operation.downstream_arrival_time),
                    # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    bay_id=operation.downstream_bay,
                    # LINE-BY-LINE: `delta`에 `1` 결과를 저장합니다. 의미/사용: `delta` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    delta=1,
                    # LINE-BY-LINE: `event_type`에 `"arrival"` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                    event_type="arrival",
                )
            )
        # LINE-BY-LINE: 조건 `operation.downstream_release_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if operation.downstream_release_time is not None:
            # LINE-BY-LINE: `self.state.downstream_events.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.downstream_events.append(
                # LINE-BY-LINE: `DownstreamEvent(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                DownstreamEvent(
                    # LINE-BY-LINE: `event_time`에 `float(operation.downstream_release_time)` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    event_time=float(operation.downstream_release_time),
                    # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    bay_id=operation.downstream_bay,
                    # LINE-BY-LINE: `delta`에 `-1` 결과를 저장합니다. 의미/사용: `delta` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    delta=-1,
                    # LINE-BY-LINE: `event_type`에 `"release"` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                    event_type="release",
                )
            )
        # LINE-BY-LINE: 현재 객체의 `state.downstream_events.sort(key` 속성에 `lambda item: item.event_time)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.downstream_events.sort(key=lambda item: item.event_time)

    # LINE-BY-LINE: `advance_time(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def advance_time(self) -> bool:
        """다음 설비 완료 시점으로 시간을 넘깁니다."""

        # LINE-BY-LINE: `future_times`에 `self._future_decision_times()` 결과를 저장합니다. 의미/사용: `future_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        future_times = self._future_decision_times()
        # LINE-BY-LINE: 조건 `not future_times`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not future_times:
            # LINE-BY-LINE: 호출자에게 `False`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return False
        # LINE-BY-LINE: `self._move_current_time(future_times[0])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._move_current_time(future_times[0])
        # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return True

    # LINE-BY-LINE: `_advance_to_decision_epoch(self)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _advance_to_decision_epoch(self) -> None:
        """가능한 action이 나올 때까지 다음 의사결정 시점으로 전진합니다.

        전진 규칙:
        1. 후보가 있으면 즉시 멈춤
        2. 후보가 없고, 원인이 "바쁜 설비"면 다음 완료 시점으로 점프
        3. 후보가 없고, 원인이 "캘린더/고장/일일 용량"이면 1분씩 전진
           - 이 방식은 느릴 수 있지만 초보자에게 가장 직관적입니다.
           - 또한 현재 구현은 preemption이 없기 때문에,
             시작 가능 시점만 찾으면 충분합니다.
        4. 후보가 없고, 시간 지나도 안 풀리는 정적 제약뿐이면 중단
        """

        # LINE-BY-LINE: `horizon_minutes`에 `float(self.config["simulation"]["horizon_minutes"])` 결과를 저장합니다. 의미/사용: `horizon_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        horizon_minutes = float(self.config["simulation"]["horizon_minutes"])

        # LINE-BY-LINE: 조건 `self._has_pending_work()`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while self._has_pending_work():
            # LINE-BY-LINE: 조건 `self._has_any_candidate()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._has_any_candidate():
                # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
                return

            # LINE-BY-LINE: 조건 `not self.config["simulation"]["auto_advance_time"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self.config["simulation"]["auto_advance_time"]:
                # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
                return

            # LINE-BY-LINE: 조건 `self.state.current_time >= horizon_minutes`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self.state.current_time >= horizon_minutes:
                # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
                return

            # LINE-BY-LINE: 조건 `self._has_temporary_blocking_condition()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if self._has_temporary_blocking_condition():
                # LINE-BY-LINE: `self._move_current_time(self._next_temporary_unblock_time(horizon_minutes))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self._move_current_time(self._next_temporary_unblock_time(horizon_minutes))
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `moved`에 `self.advance_time()` 결과를 저장합니다. 의미/사용: `moved` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            moved = self.advance_time()
            # LINE-BY-LINE: 조건 `not moved`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not moved:
                # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
                return

    # LINE-BY-LINE: `build_observation(self)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def build_observation(self) -> Dict:
        """현재 상태를 외부 정책이 읽기 쉬운 dict 형태로 반환합니다."""

        # LINE-BY-LINE: `self._apply_due_downstream_events()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_due_downstream_events()
        # LINE-BY-LINE: `self._prune_resource_allocations()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._prune_resource_allocations()
        # LINE-BY-LINE: `candidates`에 `self.get_candidates()` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates = self.get_candidates()
        # LINE-BY-LINE: `current_day_key`에 `self._current_day_key()` 결과를 저장합니다. 의미/사용: `current_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_day_key = self._current_day_key()
        # LINE-BY-LINE: `self._ensure_day_state(current_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(current_day_key)
        # LINE-BY-LINE: `total_jobs`에 `len(self.jobs)` 결과를 저장합니다. 의미/사용: `total_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        total_jobs = len(self.jobs)
        # LINE-BY-LINE: `machine_count`에 `max(len(self.machines), 1)` 결과를 저장합니다. 의미/사용: `machine_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_count = max(len(self.machines), 1)
        # LINE-BY-LINE: `average_machine_load`에 `sum(self.state.machine_loads.values()) / machine_count` 결과를 저장합니다. 의미/사용: `average_machine_load` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        average_machine_load = sum(self.state.machine_loads.values()) / machine_count
        # LINE-BY-LINE: `average_machine_capacity`에 `sum(` 결과를 저장합니다. 의미/사용: `average_machine_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        average_machine_capacity = sum(
            # LINE-BY-LINE: `self._get_machine_daily_capacity_limit(machine.machine_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._get_machine_daily_capacity_limit(machine.machine_id)
            # LINE-BY-LINE: `machine in self.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for machine in self.machines.values()
        # LINE-BY-LINE: `) / machine_count` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        ) / machine_count
        # LINE-BY-LINE: `max_downstream_ratio`에 `0.0` 결과를 저장합니다. 의미/사용: `max_downstream_ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_downstream_ratio = 0.0
        # LINE-BY-LINE: `bay_id, bay in self.bays.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for bay_id, bay in self.bays.items():
            # LINE-BY-LINE: `ratio`에 `self.state.downstream_loads[bay_id] / max(bay.capacity_limit, 1)` 결과를 저장합니다. 의미/사용: `ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            ratio = self.state.downstream_loads[bay_id] / max(bay.capacity_limit, 1)
            # LINE-BY-LINE: `max_downstream_ratio`에 `max(max_downstream_ratio, ratio)` 결과를 저장합니다. 의미/사용: `max_downstream_ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            max_downstream_ratio = max(max_downstream_ratio, ratio)

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `current_time` 키에 `self.state.current_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_time": self.state.current_time,
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `current_day_key` 키에 `current_day_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_day_key": current_day_key,
            # LINE-BY-LINE: 실행 summary의 `makespan` 항목에 `self.get_makespan()`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "makespan": self.get_makespan(),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `remaining_job_count` 키에 `len(self.state.unscheduled_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "remaining_job_count": len(self.state.unscheduled_jobs),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `total_job_count` 키에 `total_jobs` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "total_job_count": total_jobs,
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `remaining_jobs` 키에 `sorted(self.state.unscheduled_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "remaining_jobs": sorted(self.state.unscheduled_jobs),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_available_at` 키에 `dict(self.state.machine_available_at)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_available_at": dict(self.state.machine_available_at),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_loads` 키에 `dict(self.state.machine_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_loads": dict(self.state.machine_loads),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_daily_job_counts` 키에 `dict(self.state.machine_daily_job_counts.get(current_day_key, {}))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_daily_job_counts": dict(self.state.machine_daily_job_counts.get(current_day_key, {})),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_daily_length_sums` 키에 `dict(self.state.machine_daily_length_sums.get(current_day_key, {}))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_daily_length_sums": dict(self.state.machine_daily_length_sums.get(current_day_key, {})),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `block_set_bay_assignments` 키에 `dict(self.state.block_set_bay_assignments)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "block_set_bay_assignments": dict(self.state.block_set_bay_assignments),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `open_batches` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "open_batches": {
                # LINE-BY-LINE: `batch_id`를 `{` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                batch_id: {
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `batch.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": batch.machine_id,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `job_ids` 키에 `list(batch.job_ids)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "job_ids": list(batch.job_ids),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `length_sum` 키에 `batch.length_sum` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "length_sum": batch.length_sum,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `max_wo_count` 키에 `batch.max_wo_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "max_wo_count": batch.max_wo_count,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `max_length_sum` 키에 `batch.max_length_sum` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "max_length_sum": batch.max_length_sum,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `status` 키에 `batch.status` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "status": batch.status,
                }
                # LINE-BY-LINE: `batch_id, batch in sorted(self.state.open_batches.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for batch_id, batch in sorted(self.state.open_batches.items())
            },
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `downstream_loads` 키에 `dict(self.state.downstream_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_loads": dict(self.state.downstream_loads),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `event_count` 키에 `len(self.state.event_log)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_count": len(self.state.event_log),
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `env_features` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "env_features": {
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `current_time_norm` 키에 `self.state.current_time / max(self.config["simulation"]["horizon_minutes"], 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "current_time_norm": self.state.current_time / max(self.config["simulation"]["horizon_minutes"], 1),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `makespan_norm` 키에 `self.get_makespan() / max(self.config["simulation"]["horizon_minutes"], 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "makespan_norm": self.get_makespan() / max(self.config["simulation"]["horizon_minutes"], 1),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `remaining_job_ratio` 키에 `len(self.state.unscheduled_jobs) / max(total_jobs, 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "remaining_job_ratio": len(self.state.unscheduled_jobs) / max(total_jobs, 1),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `available_machine_ratio` 키에 `sum(self._is_machine_available(machine_id) for machine_id in self.machines) / machine_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "available_machine_ratio": sum(self._is_machine_available(machine_id) for machine_id in self.machines) / machine_count,
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `calendar_open_flag` 키에 `1.0 if self._is_global_calendar_open()[0] else 0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "calendar_open_flag": 1.0 if self._is_global_calendar_open()[0] else 0.0,
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_calendar_open_ratio` 키에 `sum(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_calendar_open_ratio": sum(
                    # LINE-BY-LINE: `1` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    1
                    # LINE-BY-LINE: `machine_id in self.machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for machine_id in self.machines
                    # LINE-BY-LINE: 조건 `self._machine_calendar_status(machine_id)[0]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if self._machine_calendar_status(machine_id)[0]
                # LINE-BY-LINE: `sum(...)` 호출에 `) / machine_count` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                ) / machine_count,
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `broken_machine_ratio` 키에 `sum(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "broken_machine_ratio": sum(
                    # LINE-BY-LINE: `1` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    1
                    # LINE-BY-LINE: `machine_id in self.machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for machine_id in self.machines
                    # LINE-BY-LINE: 조건 `self._machine_breakdown_status(machine_id)[0]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if self._machine_breakdown_status(machine_id)[0]
                # LINE-BY-LINE: `sum(...)` 호출에 `) / machine_count` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                ) / machine_count,
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `average_machine_load_ratio` 키에 `average_machine_load / max(average_machine_capacity, 1.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "average_machine_load_ratio": average_machine_load / max(average_machine_capacity, 1.0),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `max_downstream_ratio` 키에 `max_downstream_ratio` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "max_downstream_ratio": max_downstream_ratio,
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `available_action_ratio` 키에 `len(candidates) / max(len(self.state.unscheduled_jobs) * machine_count, 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "available_action_ratio": len(candidates) / max(len(self.state.unscheduled_jobs) * machine_count, 1),
            },
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `vocab_sizes` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "vocab_sizes": {
                # LINE-BY-LINE: 딕셔너리 키 `family`에는 `len(self.family_to_index)` 값을 넣습니다. 의미: 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
                "family": len(self.family_to_index),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_type` 키에 `len(self.machine_type_to_index)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_type": len(self.machine_type_to_index),
                # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `bay` 키에 `len(self.bay_to_index)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "bay": len(self.bay_to_index),
            },
            # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `available_actions` 키에 `[` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "available_actions": [
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `action_id` 키에 `candidate.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "action_id": candidate.action_id,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `action_kind` 키에 `candidate.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "action_kind": candidate.action_kind,
                    # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `candidate.batch_id or ""` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                    "batch_id": candidate.batch_id or "",
                    # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `list(candidate.batch_job_ids)` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
                    "batch_job_ids": list(candidate.batch_job_ids),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `batch_wo_count` 키에 `candidate.batch_wo_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "batch_wo_count": candidate.batch_wo_count,
                    # LINE-BY-LINE: 딕셔너리 키 `batch_length_sum`에는 `candidate.batch_length_sum` 값을 넣습니다. 의미: batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
                    "batch_length_sum": candidate.batch_length_sum,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `batch_processing_rule` 키에 `candidate.batch_processing_rule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "batch_processing_rule": candidate.batch_processing_rule,
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `candidate.job.job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": candidate.job.job_id,
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `candidate.machine.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": candidate.machine.machine_id,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_type` 키에 `candidate.machine.machine_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_type": candidate.machine.machine_type,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_type_index` 키에 `self.machine_type_to_index[candidate.machine.machine_type]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_type_index": self.machine_type_to_index[candidate.machine.machine_type],
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_bay_id` 키에 `candidate.machine.bay_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_bay_id": candidate.machine.bay_id,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `job_cut_bay` 키에 `candidate.job.cut_bay` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "job_cut_bay": candidate.job.cut_bay,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `block_set_id` 키에 `candidate.job.block_set_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "block_set_id": candidate.job.block_set_id,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `estimated_minutes` 키에 `candidate.estimated_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "estimated_minutes": candidate.estimated_minutes,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `processing_minutes` 키에 `candidate.processing_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "processing_minutes": candidate.processing_minutes,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `blocked_minutes` 키에 `candidate.blocked_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "blocked_minutes": candidate.blocked_minutes,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `changeover_minutes` 키에 `candidate.changeover_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "changeover_minutes": candidate.changeover_minutes,
                    # LINE-BY-LINE: 딕셔너리 키 `priority_weight`에는 `candidate.job.priority_weight` 값을 넣습니다. 의미: 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
                    "priority_weight": candidate.job.priority_weight,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `soft_penalty` 키에 `candidate.soft_penalty` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "soft_penalty": candidate.soft_penalty,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `soft_reasons` 키에 `candidate.soft_reasons` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "soft_reasons": candidate.soft_reasons,
                    # LINE-BY-LINE: 딕셔너리 키 `family`에는 `candidate.job.family` 값을 넣습니다. 의미: 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
                    "family": candidate.job.family,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `family_index` 키에 `self.family_to_index[candidate.job.family]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "family_index": self.family_to_index[candidate.job.family],
                    # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `candidate.job.thickness` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                    "thickness": candidate.job.thickness,
                    # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `candidate.job.plate_length` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                    "plate_length": candidate.job.plate_length,
                    # LINE-BY-LINE: 딕셔너리 키 `downstream_bay`에는 `candidate.job.downstream_bay` 값을 넣습니다. 의미: 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                    "downstream_bay": candidate.job.downstream_bay,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `downstream_bay_index` 키에 `self.bay_to_index[candidate.job.downstream_bay]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "downstream_bay_index": self.bay_to_index[candidate.job.downstream_bay],
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `downstream_priority_rank` 키에 `self.bays[candidate.job.downstream_bay].priority_rank` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "downstream_priority_rank": self.bays[candidate.job.downstream_bay].priority_rank,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `downstream_load_ratio` 키에 `self.state.downstream_loads[candidate.job.downstream_bay]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "downstream_load_ratio": self.state.downstream_loads[candidate.job.downstream_bay]
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `/ max(self.bays[candidate.job.downstream_bay].capacity_limit, 1)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    / max(self.bays[candidate.job.downstream_bay].capacity_limit, 1),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `downstream_arrival_time_norm` 키에 `candidate.downstream_arrival_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "downstream_arrival_time_norm": candidate.downstream_arrival_time
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `/ max(self.config["simulation"]["horizon_minutes"], 1)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    / max(self.config["simulation"]["horizon_minutes"], 1),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_speed_factor` 키에 `candidate.machine.cut_speed_factor` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_speed_factor": candidate.machine.cut_speed_factor,
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_load_ratio` 키에 `self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_load_ratio": self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id]
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `/ max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    / max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_daily_job_count` 키에 `self.state.machine_daily_job_counts[current_day_key][candidate.machine.machine_id]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_daily_job_count": self.state.machine_daily_job_counts[current_day_key][candidate.machine.machine_id],
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_daily_length_sum` 키에 `self.state.machine_daily_length_sums[current_day_key][candidate.machine.machine_id]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_daily_length_sum": self.state.machine_daily_length_sums[current_day_key][candidate.machine.machine_id],
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_day_wo_count_limit` 키에 `self._get_machine_day_wo_count_limit(candidate.machine.machine_id)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_day_wo_count_limit": self._get_machine_day_wo_count_limit(candidate.machine.machine_id),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `machine_day_length_sum_limit` 키에 `self._get_machine_day_length_sum_limit(candidate.machine.machine_id)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_day_length_sum_limit": self._get_machine_day_length_sum_limit(candidate.machine.machine_id),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `remaining_machine_capacity_ratio` 키에 `max(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "remaining_machine_capacity_ratio": max(
                        # LINE-BY-LINE: `max(...)` 호출에 `0.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        0.0,
                        # LINE-BY-LINE: `self._get_machine_daily_capacity_limit(candidate.machine.machine_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                        self._get_machine_daily_capacity_limit(candidate.machine.machine_id)
                        # LINE-BY-LINE: `max(...)` 호출에 `- self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        - self.state.machine_daily_loads[current_day_key][candidate.machine.machine_id],
                    )
                    # LINE-BY-LINE: `현재 표현식(...)` 호출에 `/ max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    / max(self._get_machine_daily_capacity_limit(candidate.machine.machine_id), 1.0),
                    # LINE-BY-LINE: `build_observation`에서 반환/저장할 dict의 `due_date_slack_norm` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "due_date_slack_norm": (
                        # LINE-BY-LINE: `0.0` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                        0.0
                        # LINE-BY-LINE: 조건 `candidate.job.due_date_minutes is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                        if candidate.job.due_date_minutes is None
                        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                        else (candidate.job.due_date_minutes - (self.state.current_time + candidate.estimated_minutes))
                        # LINE-BY-LINE: `/ max(self.config["simulation"]["horizon_minutes"], 1)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                        / max(self.config["simulation"]["horizon_minutes"], 1)
                    ),
                }
                # LINE-BY-LINE: `candidate in candidates` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for candidate in candidates
            ],
        }

    # LINE-BY-LINE: `_apply_open_batch_action(self, selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `str`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def _apply_open_batch_action(self, selected: ActionCandidate) -> str:
        """`open_batch:JOB@MACHINE` action을 DES state에 반영한다.

        입력:
        - `selected.action_kind == "open_batch"`
        - `selected.job`: 새 batch에 처음 넣을 W/O
        - `selected.machine`: batch를 열 설비

        출력:
        - 생성된 batch_id. 예: B000001

        state 변경:
        - `unscheduled_jobs`에서 W/O 제거
        - `open_batches[batch_id]` 생성
        - 아직 `schedule`과 `event_log`에는 절단 start/finish를 만들지 않는다.

        왜 schedule을 만들지 않는가:
        - open은 "정반에 같이 올릴 후보를 구성하는 의사결정"이다.
        - 실제 machine 점유는 `close_batch`가 선택되어야 시작된다.
        """

        # LINE-BY-LINE: 조건 `selected.job.job_id not in self.state.unscheduled_jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.job.job_id not in self.state.unscheduled_jobs:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_open_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_open_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `job_not_unscheduled job_id={selected.job.job_id} action_id={selected.action_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=job_not_unscheduled job_id={selected.job.job_id} action_id={selected.action_id}"
            )
            # LINE-BY-LINE: `RuntimeError(f"cannot open batch with non-unscheduled job: {selected.job.job_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(f"cannot open batch with non-unscheduled job: {selected.job.job_id}")
        # LINE-BY-LINE: 조건 `selected.machine.machine_id in {batch.machine_id for batch in self.state.open_batches.values()}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.machine.machine_id in {batch.machine_id for batch in self.state.open_batches.values()}:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_open_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_open_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `machine_already_has_open_batch machine_id={selected.machine.machine_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=machine_already_has_open_batch machine_id={selected.machine.machine_id}"
            )
            # LINE-BY-LINE: `RuntimeError(f"machine already has open batch: {selected.machine.machine_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(f"machine already has open batch: {selected.machine.machine_id}")

        # LINE-BY-LINE: `max_wo_count, max_length_sum` 여러 변수에 `self._batch_limits_for_machine(selected.machine)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        max_wo_count, max_length_sum = self._batch_limits_for_machine(selected.machine)
        # LINE-BY-LINE: `batch_id`에 `self._next_batch_id()` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id = self._next_batch_id()
        # LINE-BY-LINE: 현재 객체의 `state.open_batches[batch_id]` 속성에 `BatchCandidate(` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.open_batches[batch_id] = BatchCandidate(
            # LINE-BY-LINE: `batch_id`에 `batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            batch_id=batch_id,
            # LINE-BY-LINE: `machine_id`에 `selected.machine.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=selected.machine.machine_id,
            # LINE-BY-LINE: `job_ids`에 `(selected.job.job_id,)` 결과를 저장합니다. 의미/사용: `job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            job_ids=(selected.job.job_id,),
            # LINE-BY-LINE: `bay_id`에 `selected.machine.bay_id` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=selected.machine.bay_id,
            # LINE-BY-LINE: `status`에 `"open"` 결과를 저장합니다. 의미/사용: `status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            status="open",
            # LINE-BY-LINE: `start_time`에 `self.state.current_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
            start_time=self.state.current_time,
            # LINE-BY-LINE: `length_sum`에 `float(selected.job.plate_length)` 결과를 저장합니다. 의미/사용: `length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            length_sum=float(selected.job.plate_length),
            # LINE-BY-LINE: `max_wo_count`에 `max_wo_count` 결과를 저장합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            max_wo_count=max_wo_count,
            # LINE-BY-LINE: `max_length_sum`에 `max_length_sum` 결과를 저장합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            max_length_sum=max_length_sum,
            # LINE-BY-LINE: `metadata`에 `{` 결과를 저장합니다. 의미/사용: `metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            metadata={
                # LINE-BY-LINE: `_apply_open_batch_action`에서 반환/저장할 dict의 `opened_by_action_id` 키에 `selected.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "opened_by_action_id": selected.action_id,
                # LINE-BY-LINE: `_apply_open_batch_action`에서 반환/저장할 dict의 `processing_time_rule` 키에 `self._batch_processing_rule()` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "processing_time_rule": self._batch_processing_rule(),
            },
        )
        # LINE-BY-LINE: `self.state.unscheduled_jobs.remove(selected.job.job_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.state.unscheduled_jobs.remove(selected.job.job_id)
        # LINE-BY-LINE: 조건 `selected.job.block_set_id and selected.machine.bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.job.block_set_id and selected.machine.bay_id:
            # LINE-BY-LINE: `self.state.block_set_bay_assignments.setdefault(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.block_set_bay_assignments.setdefault(
                # LINE-BY-LINE: `setdefault(...)` 호출에 `selected.job.block_set_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                selected.job.block_set_id,
                # LINE-BY-LINE: `setdefault(...)` 호출에 `str(selected.machine.bay_id)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                str(selected.machine.bay_id),
            )
        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_DECISION_EPOCH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_DECISION_EPOCH,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `self.state.current_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            self.state.current_time,
            # LINE-BY-LINE: `job_id`에 `selected.job.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=selected.job.job_id,
            # LINE-BY-LINE: `machine_id`에 `selected.machine.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=selected.machine.machine_id,
            # LINE-BY-LINE: `bay_id`에 `selected.machine.bay_id` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=selected.machine.bay_id,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `_apply_open_batch_action`에서 반환/저장할 dict의 `action_kind` 키에 `selected.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_kind": selected.action_kind,
                # LINE-BY-LINE: `_apply_open_batch_action`에서 반환/저장할 dict의 `action_id` 키에 `selected.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_id": selected.action_id,
                # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `batch_id` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                "batch_id": batch_id,
                # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `selected.job.job_id` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
                "batch_job_ids": selected.job.job_id,
                # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"open batch; no machine processing started yet"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                "message": "open batch; no machine processing started yet",
            },
        )
        # LINE-BY-LINE: 호출자에게 `batch_id`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return batch_id

    # LINE-BY-LINE: `_apply_add_to_batch_action(self, selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def _apply_add_to_batch_action(self, selected: ActionCandidate) -> None:
        """`add_to_batch:BATCH:JOB` action을 DES state에 반영한다."""

        # LINE-BY-LINE: `batch_id`에 `selected.batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
        batch_id = selected.batch_id
        # LINE-BY-LINE: 조건 `batch_id not in self.state.open_batches`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if batch_id not in self.state.open_batches:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_add_to_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_add_to_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `unknown_open_batch batch_id={batch_id} action_id={selected.action_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unknown_open_batch batch_id={batch_id} action_id={selected.action_id}"
            )
            # LINE-BY-LINE: `KeyError(f"unknown open batch: {batch_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise KeyError(f"unknown open batch: {batch_id}")
        # LINE-BY-LINE: 조건 `selected.job.job_id not in self.state.unscheduled_jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.job.job_id not in self.state.unscheduled_jobs:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_add_to_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_add_to_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `job_not_unscheduled job_id={selected.job.job_id} batch_id={batch_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=job_not_unscheduled job_id={selected.job.job_id} batch_id={batch_id}"
            )
            # LINE-BY-LINE: `RuntimeError(f"cannot add non-unscheduled job to batch: {selected.job.job_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(f"cannot add non-unscheduled job to batch: {selected.job.job_id}")

        # LINE-BY-LINE: `batch`에 `self.state.open_batches[batch_id]` 결과를 저장합니다. 의미/사용: `batch` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch = self.state.open_batches[batch_id]
        # LINE-BY-LINE: `next_job_ids`에 `(*tuple(batch.job_ids), selected.job.job_id)` 결과를 저장합니다. 의미/사용: `next_job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        next_job_ids = (*tuple(batch.job_ids), selected.job.job_id)
        # LINE-BY-LINE: `next_jobs`에 `self._jobs_from_ids(next_job_ids)` 결과를 저장합니다. 의미/사용: `next_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        next_jobs = self._jobs_from_ids(next_job_ids)
        # LINE-BY-LINE: 조건 `not self._batch_jobs_fit_limits(next_jobs, selected.machine)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not self._batch_jobs_fit_limits(next_jobs, selected.machine):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_add_to_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_add_to_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `batch_limit_exceeded batch_id={batch_id} job_ids={next_job_ids}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=batch_limit_exceeded batch_id={batch_id} job_ids={next_job_ids}"
            )
            # LINE-BY-LINE: `RuntimeError(f"batch limit exceeded while applying add_to_batch: {batch_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError(f"batch limit exceeded while applying add_to_batch: {batch_id}")

        # LINE-BY-LINE: `batch.job_ids`에 `next_job_ids` 결과를 저장합니다. 의미/사용: `job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch.job_ids = next_job_ids
        # LINE-BY-LINE: `batch.length_sum`에 `self._batch_length_sum(next_jobs)` 결과를 저장합니다. 의미/사용: `length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch.length_sum = self._batch_length_sum(next_jobs)
        # LINE-BY-LINE: `batch.metadata["last_added_action_id"]`에 `selected.action_id` 결과를 저장합니다. 의미/사용: `metadata["last_added_action_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch.metadata["last_added_action_id"] = selected.action_id
        # LINE-BY-LINE: `self.state.unscheduled_jobs.remove(selected.job.job_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.state.unscheduled_jobs.remove(selected.job.job_id)
        # LINE-BY-LINE: 조건 `selected.job.block_set_id and selected.machine.bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.job.block_set_id and selected.machine.bay_id:
            # LINE-BY-LINE: `self.state.block_set_bay_assignments.setdefault(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.block_set_bay_assignments.setdefault(
                # LINE-BY-LINE: `setdefault(...)` 호출에 `selected.job.block_set_id` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                selected.job.block_set_id,
                # LINE-BY-LINE: `setdefault(...)` 호출에 `str(selected.machine.bay_id)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                str(selected.machine.bay_id),
            )
        # LINE-BY-LINE: `self.emit_event(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.emit_event(
            # LINE-BY-LINE: `emit_event(...)` 호출에 `EVENT_DECISION_EPOCH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            EVENT_DECISION_EPOCH,
            # LINE-BY-LINE: `emit_event(...)` 호출에 `self.state.current_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            self.state.current_time,
            # LINE-BY-LINE: `job_id`에 `selected.job.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=selected.job.job_id,
            # LINE-BY-LINE: `machine_id`에 `selected.machine.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=selected.machine.machine_id,
            # LINE-BY-LINE: `bay_id`에 `selected.machine.bay_id` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=selected.machine.bay_id,
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload={
                # LINE-BY-LINE: `_apply_add_to_batch_action`에서 반환/저장할 dict의 `action_kind` 키에 `selected.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_kind": selected.action_kind,
                # LINE-BY-LINE: `_apply_add_to_batch_action`에서 반환/저장할 dict의 `action_id` 키에 `selected.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_id": selected.action_id,
                # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `batch_id` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                "batch_id": batch_id,
                # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `"|".join(next_job_ids)` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
                "batch_job_ids": "|".join(next_job_ids),
                # LINE-BY-LINE: `_apply_add_to_batch_action`에서 반환/저장할 dict의 `batch_wo_count` 키에 `len(next_job_ids)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "batch_wo_count": len(next_job_ids),
                # LINE-BY-LINE: 딕셔너리 키 `batch_length_sum`에는 `batch.length_sum` 값을 넣습니다. 의미: batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
                "batch_length_sum": batch.length_sum,
                # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"add W/O to open batch; no machine processing started yet"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                "message": "add W/O to open batch; no machine processing started yet",
            },
        )

    # LINE-BY-LINE: `_build_scheduled_operations_for_batch(self, selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `List[ScheduledOperation]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _build_scheduled_operations_for_batch(self, selected: ActionCandidate) -> List[ScheduledOperation]:
        """close된 batch 후보를 job-level `ScheduledOperation` 목록으로 변환한다.

        출력 구조:
        - batch W/O가 3개면 `ScheduledOperation`도 3개를 만든다.
        - 세 row는 같은 `batch_id`, `machine_id`, `start_time`, `finish_time`을 가진다.
        - machine slot은 `_apply_batch_operations()`에서 한 번만 점유한다.
        """

        # LINE-BY-LINE: `batch_job_ids`에 `tuple(selected.batch_job_ids)` 결과를 저장합니다. 의미/사용: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
        batch_job_ids = tuple(selected.batch_job_ids)
        # LINE-BY-LINE: `jobs`에 `self._jobs_from_ids(batch_job_ids)` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs = self._jobs_from_ids(batch_job_ids)
        # LINE-BY-LINE: 조건 `not jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not jobs:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._build_scheduled_operations_for_batch] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._build_scheduled_operations_for_batch] "
                # LINE-BY-LINE: `f"cause`에 `empty_batch action_id={selected.action_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=empty_batch action_id={selected.action_id}"
            )
            # LINE-BY-LINE: `RuntimeError("cannot schedule empty batch")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("cannot schedule empty batch")

        # LINE-BY-LINE: `timing`에 `self._batch_candidate_timing(jobs, selected.machine)` 결과를 저장합니다. 의미/사용: `timing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        timing = self._batch_candidate_timing(jobs, selected.machine)
        # LINE-BY-LINE: `start_time`에 `float(timing["start_time"])` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
        start_time = float(timing["start_time"])
        # LINE-BY-LINE: `finish_time`에 `float(timing["finish_time"])` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
        finish_time = float(timing["finish_time"])
        # LINE-BY-LINE: `processing_minutes`에 `float(timing["processing_minutes"])` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
        processing_minutes = float(timing["processing_minutes"])
        # LINE-BY-LINE: `batch_length_sum`에 `self._batch_length_sum(jobs)` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
        batch_length_sum = self._batch_length_sum(jobs)
        # LINE-BY-LINE: `operations` 변수에 `[]` 결과를 저장합니다. 의미: `operations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operations: List[ScheduledOperation] = []

        # LINE-BY-LINE: `job in jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job in jobs:
            # LINE-BY-LINE: `bay`에 `self.bays[job.downstream_bay]` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
            bay = self.bays[job.downstream_bay]
            # LINE-BY-LINE: `downstream_arrival_time`에 `finish_time + float(bay.transfer_time_minutes)` 결과를 저장합니다. 의미/사용: `downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_arrival_time = finish_time + float(bay.transfer_time_minutes)
            # LINE-BY-LINE: `downstream_release_time`에 `None` 결과를 저장합니다. 의미/사용: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            downstream_release_time = None
            # LINE-BY-LINE: 조건 `bay.release_delay_minutes is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if bay.release_delay_minutes is not None:
                # LINE-BY-LINE: `downstream_release_time`에 `downstream_arrival_time + float(bay.release_delay_minutes)` 결과를 저장합니다. 의미/사용: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                downstream_release_time = downstream_arrival_time + float(bay.release_delay_minutes)
            # LINE-BY-LINE: `operations.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            operations.append(
                # LINE-BY-LINE: `ScheduledOperation(`를 실행합니다. 의미/사용: 확정된 schedule row 1건을 생성합니다.
                ScheduledOperation(
                    # LINE-BY-LINE: `job_id`에 `job.job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    job_id=job.job_id,
                    # LINE-BY-LINE: `machine_id`에 `selected.machine.machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    machine_id=selected.machine.machine_id,
                    # LINE-BY-LINE: `start_time`에 `start_time` 결과를 저장합니다. 의미/사용: `start_time`는 후보 작업이 시작되는 simulation minute입니다. event PROCESS_START에도 사용됩니다.
                    start_time=start_time,
                    # LINE-BY-LINE: `finish_time`에 `finish_time` 결과를 저장합니다. 의미/사용: `finish_time`는 후보 작업이 끝나는 simulation minute입니다. event PROCESS_FINISH에도 사용됩니다.
                    finish_time=finish_time,
                    # LINE-BY-LINE: `downstream_bay`에 `job.downstream_bay` 결과를 저장합니다. 의미/사용: `downstream_bay`는 절단 이후 흘러가는 후공정/적치 Bay입니다. CUT_BAY와 다릅니다.
                    downstream_bay=job.downstream_bay,
                    # LINE-BY-LINE: `estimated_minutes`에 `finish_time - start_time` 결과를 저장합니다. 의미/사용: `estimated_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    estimated_minutes=finish_time - start_time,
                    # LINE-BY-LINE: `cut_bay`에 `selected.machine.bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                    cut_bay=selected.machine.bay_id,
                    # LINE-BY-LINE: `processing_minutes`에 `processing_minutes` 결과를 저장합니다. 의미/사용: `processing_minutes`는 설비가 실제 처리에 쓰는 minute입니다. machine load와 tact 비교에 사용됩니다.
                    processing_minutes=processing_minutes,
                    # LINE-BY-LINE: `changeover_minutes`에 `0.0` 결과를 저장합니다. 의미/사용: `changeover_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    changeover_minutes=0.0,
                    # LINE-BY-LINE: `blocked_minutes`에 `float(timing["blocked_minutes"])` 결과를 저장합니다. 의미/사용: `blocked_minutes`는 calendar/운영시간 때문에 밀린 minute입니다. 현재 1차 범위에서는 보통 0입니다.
                    blocked_minutes=float(timing["blocked_minutes"]),
                    # LINE-BY-LINE: `downstream_arrival_time`에 `downstream_arrival_time` 결과를 저장합니다. 의미/사용: `downstream_arrival_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    downstream_arrival_time=downstream_arrival_time,
                    # LINE-BY-LINE: `downstream_release_time`에 `downstream_release_time` 결과를 저장합니다. 의미/사용: `downstream_release_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    downstream_release_time=downstream_release_time,
                    # LINE-BY-LINE: `resource_ids`에 `tuple(selected.required_resource_ids)` 결과를 저장합니다. 의미/사용: `resource_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    resource_ids=tuple(selected.required_resource_ids),
                    # LINE-BY-LINE: `batch_id`에 `selected.batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                    batch_id=selected.batch_id,
                    # LINE-BY-LINE: `batch_job_ids`에 `batch_job_ids` 결과를 저장합니다. 의미/사용: `batch_job_ids`는 batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
                    batch_job_ids=batch_job_ids,
                    # LINE-BY-LINE: `batch_length_sum`에 `batch_length_sum` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
                    batch_length_sum=batch_length_sum,
                    # LINE-BY-LINE: `stage_minutes`에 `{` 결과를 저장합니다. 의미/사용: `stage_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    stage_minutes={
                        # LINE-BY-LINE: `_build_scheduled_operations_for_batch`에서 반환/저장할 dict의 `setup` 키에 `max(float(job.base_stage_minutes.get("setup", 0.0)) for job in jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "setup": max(float(job.base_stage_minutes.get("setup", 0.0)) for job in jobs),
                        # LINE-BY-LINE: `_build_scheduled_operations_for_batch`에서 반환/저장할 dict의 `cut` 키에 `processing_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "cut": processing_minutes,
                        # LINE-BY-LINE: `_build_scheduled_operations_for_batch`에서 반환/저장할 dict의 `finish` 키에 `max(float(job.base_stage_minutes.get("finish", 0.0)) for job in jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "finish": max(float(job.base_stage_minutes.get("finish", 0.0)) for job in jobs),
                    },
                )
            )
        # LINE-BY-LINE: 호출자에게 `operations`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return operations

    # LINE-BY-LINE: `_apply_batch_operations(self, operations: List[ScheduledOperation], selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `None`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _apply_batch_operations(self, operations: List[ScheduledOperation], selected: ActionCandidate) -> None:
        """batch close 결과를 state에 반영한다.

        핵심 차이:
        - `schedule`에는 W/O별 row를 모두 남긴다.
        - 하지만 machine slot과 machine load는 batch window 1개로 한 번만 반영한다.
        - 그래야 "같은 설비에서 W/O 3개까지 동시 작업"을 표현할 수 있다.
        """

        # LINE-BY-LINE: 조건 `not operations`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not operations:
            # LINE-BY-LINE: 콘솔에 `print("[ERROR][CuttingSimulation._apply_batch_operations] cause=empty_operations")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[ERROR][CuttingSimulation._apply_batch_operations] cause=empty_operations")
            # LINE-BY-LINE: `RuntimeError("cannot apply empty batch operations")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("cannot apply empty batch operations")

        # LINE-BY-LINE: `first_operation`에 `operations[0]` 결과를 저장합니다. 의미/사용: `first_operation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        first_operation = operations[0]
        # LINE-BY-LINE: `operation_day_key`에 `self._current_day_key(at_time=first_operation.start_time)` 결과를 저장합니다. 의미/사용: `operation_day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_day_key = self._current_day_key(at_time=first_operation.start_time)
        # LINE-BY-LINE: `self._ensure_day_state(operation_day_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._ensure_day_state(operation_day_key)
        # LINE-BY-LINE: `slot_times`에 `self.state.machine_slot_available_at[first_operation.machine_id]` 결과를 저장합니다. 의미/사용: `slot_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times = self.state.machine_slot_available_at[first_operation.machine_id]
        # LINE-BY-LINE: `slot_index`에 `min(` 결과를 저장합니다. 의미/사용: `slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_index = min(
            # LINE-BY-LINE: `min(...)` 호출에 `range(len(slot_times))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            range(len(slot_times)),
            # LINE-BY-LINE: `key`에 `lambda index: (slot_times[index], index)` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key=lambda index: (slot_times[index], index),
        )
        # LINE-BY-LINE: `slot_times[slot_index]`에 `first_operation.finish_time` 결과를 저장합니다. 의미/사용: `slot_times[slot_index]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slot_times[slot_index] = first_operation.finish_time
        # LINE-BY-LINE: `self._sync_machine_earliest_availability(first_operation.machine_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._sync_machine_earliest_availability(first_operation.machine_id)

        # LINE-BY-LINE: `batch_processing_minutes`에 `float(first_operation.processing_minutes)` 결과를 저장합니다. 의미/사용: `batch_processing_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        batch_processing_minutes = float(first_operation.processing_minutes)
        # LINE-BY-LINE: `batch_length_sum`에 `float(first_operation.batch_length_sum or 0.0)` 결과를 저장합니다. 의미/사용: `batch_length_sum`는 batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
        batch_length_sum = float(first_operation.batch_length_sum or 0.0)
        # LINE-BY-LINE: `self.state.machine_loads[first_operation.machine_id]` 값을 `batch_processing_minutes` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_loads[first_operation.machine_id] += batch_processing_minutes
        # LINE-BY-LINE: `self.state.machine_daily_loads[operation_day_key][first_operation.machine_id]` 값을 `batch_processing_minutes` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_loads[operation_day_key][first_operation.machine_id] += batch_processing_minutes
        # LINE-BY-LINE: `self.state.machine_daily_job_counts[operation_day_key][first_operation.machine_id]` 값을 `len(operations)` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_job_counts[operation_day_key][first_operation.machine_id] += len(operations)
        # LINE-BY-LINE: `self.state.machine_daily_length_sums[operation_day_key][first_operation.machine_id]` 값을 `batch_length_sum` 기준으로 누적/증가합니다. 의미/사용: `machine_id]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.machine_daily_length_sums[operation_day_key][first_operation.machine_id] += batch_length_sum
        # LINE-BY-LINE: `self.state.scheduled_job_count_by_day[operation_day_key]` 값을 `len(operations)` 기준으로 누적/증가합니다. 의미/사용: `scheduled_job_count_by_day[operation_day_key]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        self.state.scheduled_job_count_by_day[operation_day_key] += len(operations)

        # LINE-BY-LINE: `families`에 `{self.jobs[operation.job_id].family for operation in operations}` 결과를 저장합니다. 의미/사용: `families` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        families = {self.jobs[operation.job_id].family for operation in operations}
        # LINE-BY-LINE: 현재 객체의 `state.machine_last_family[first_operation.machine_id]` 속성에 `next(iter(families)) if len(families) == 1 else "MIXED"` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.machine_last_family[first_operation.machine_id] = next(iter(families)) if len(families) == 1 else "MIXED"

        # LINE-BY-LINE: `operation in operations` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for operation in operations:
            # LINE-BY-LINE: `operation.machine_slot_index`에 `slot_index` 결과를 저장합니다. 의미/사용: `machine_slot_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation.machine_slot_index = slot_index
            # LINE-BY-LINE: `self.state.schedule.append(operation)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.schedule.append(operation)
            # LINE-BY-LINE: `self.state.unscheduled_jobs.discard(operation.job_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.unscheduled_jobs.discard(operation.job_id)

            # LINE-BY-LINE: `job`에 `self.jobs[operation.job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
            job = self.jobs[operation.job_id]
            # LINE-BY-LINE: `cut_bay`에 `operation.cut_bay or self.machines[operation.machine_id].bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
            cut_bay = operation.cut_bay or self.machines[operation.machine_id].bay_id
            # LINE-BY-LINE: 조건 `job.block_set_id and cut_bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if job.block_set_id and cut_bay:
                # LINE-BY-LINE: `self.state.block_set_bay_assignments.setdefault(job.block_set_id, str(cut_bay))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self.state.block_set_bay_assignments.setdefault(job.block_set_id, str(cut_bay))

            # LINE-BY-LINE: 조건 `operation.downstream_arrival_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if operation.downstream_arrival_time is not None:
                # LINE-BY-LINE: `self.state.downstream_events.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self.state.downstream_events.append(
                    # LINE-BY-LINE: `DownstreamEvent(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    DownstreamEvent(
                        # LINE-BY-LINE: `event_time`에 `float(operation.downstream_arrival_time)` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        event_time=float(operation.downstream_arrival_time),
                        # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                        bay_id=operation.downstream_bay,
                        # LINE-BY-LINE: `delta`에 `1` 결과를 저장합니다. 의미/사용: `delta` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        delta=1,
                        # LINE-BY-LINE: `event_type`에 `"arrival"` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                        event_type="arrival",
                    )
                )
            # LINE-BY-LINE: 조건 `operation.downstream_release_time is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if operation.downstream_release_time is not None:
                # LINE-BY-LINE: `self.state.downstream_events.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self.state.downstream_events.append(
                    # LINE-BY-LINE: `DownstreamEvent(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    DownstreamEvent(
                        # LINE-BY-LINE: `event_time`에 `float(operation.downstream_release_time)` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        event_time=float(operation.downstream_release_time),
                        # LINE-BY-LINE: `bay_id`에 `operation.downstream_bay` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                        bay_id=operation.downstream_bay,
                        # LINE-BY-LINE: `delta`에 `-1` 결과를 저장합니다. 의미/사용: `delta` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                        delta=-1,
                        # LINE-BY-LINE: `event_type`에 `"release"` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                        event_type="release",
                    )
                )
            # LINE-BY-LINE: `self._emit_operation_events(operation, selected)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._emit_operation_events(operation, selected)

        # LINE-BY-LINE: `resource_id in first_operation.resource_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for resource_id in first_operation.resource_ids:
            # LINE-BY-LINE: `self.state.resource_active_until.setdefault(resource_id, []).append(first_operation.finish_time)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.state.resource_active_until.setdefault(resource_id, []).append(first_operation.finish_time)

        # LINE-BY-LINE: 현재 객체의 `state.downstream_events.sort(key` 속성에 `lambda item: item.event_time)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.state.downstream_events.sort(key=lambda item: item.event_time)

        # LINE-BY-LINE: 조건 `selected.batch_id in self.state.open_batches`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.batch_id in self.state.open_batches:
            # LINE-BY-LINE: `del self.state.open_batches[selected.batch_id]` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            del self.state.open_batches[selected.batch_id]

    # LINE-BY-LINE: `_apply_close_batch_action(self, selected: ActionCandidate)` 함수를 정의합니다. 반환 타입: `List[ScheduledOperation]`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def _apply_close_batch_action(self, selected: ActionCandidate) -> List[ScheduledOperation]:
        """`close_batch:BATCH@MACHINE` action을 실제 schedule/event로 연결한다."""

        # LINE-BY-LINE: 조건 `selected.batch_id not in self.state.open_batches`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.batch_id not in self.state.open_batches:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._apply_close_batch_action] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._apply_close_batch_action] "
                # LINE-BY-LINE: `f"cause`에 `unknown_open_batch batch_id={selected.batch_id} action_id={selected.action_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unknown_open_batch batch_id={selected.batch_id} action_id={selected.action_id}"
            )
            # LINE-BY-LINE: `KeyError(f"unknown open batch: {selected.batch_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise KeyError(f"unknown open batch: {selected.batch_id}")

        # LINE-BY-LINE: `operations`에 `self._build_scheduled_operations_for_batch(selected)` 결과를 저장합니다. 의미/사용: `operations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operations = self._build_scheduled_operations_for_batch(selected)
        # LINE-BY-LINE: `self._apply_batch_operations(operations, selected)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self._apply_batch_operations(operations, selected)
        # LINE-BY-LINE: 호출자에게 `operations`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return operations

    # LINE-BY-LINE: `step_action(self, action_id: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
    def step_action(self, action_id: str):
        """외부에서 action_id를 받아 1 step 진행합니다."""

        # LINE-BY-LINE: `action_map`에 `self.get_action_map()` 결과를 저장합니다. 의미/사용: `action_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_map = self.get_action_map()
        # LINE-BY-LINE: 조건 `action_id not in action_map`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if action_id not in action_map:
            # LINE-BY-LINE: `ValueError(f"invalid action_id: {action_id}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"invalid action_id: {action_id}")

        # LINE-BY-LINE: 호출자에게 `self.step_candidate(action_map[action_id])`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.step_candidate(action_map[action_id])

    # LINE-BY-LINE: `step_candidate` 함수를 정의합니다. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def step_candidate(
        # LINE-BY-LINE: `step_candidate(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `selected`를 `ActionCandidate,` 타입으로 선언합니다. 의미/사용: `CuttingSimulation.selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
        selected: ActionCandidate,
        # LINE-BY-LINE: `build_observation_result` 변수에 `True` 결과를 저장합니다. 의미: `build_observation_result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        build_observation_result: bool = True,
        # LINE-BY-LINE: `advance_decision_epoch` 변수에 `True` 결과를 저장합니다. 의미: `advance_decision_epoch` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        advance_decision_epoch: bool = True,
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        """이미 평가된 후보를 받아 1 step 진행합니다.

        휴리스틱/정책 실행 중에는 선택된 후보 객체가 이미 있으므로,
        action_id 검증을 위해 전체 후보를 다시 생성하지 않는다.
        """

        # LINE-BY-LINE: `action_id`에 `selected.action_id` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_id = selected.action_id

        # 보상 계산을 위해 step 전 상태를 저장합니다.
        # LINE-BY-LINE: `before_completion`에 `self._current_completion_time()` 결과를 저장합니다. 의미/사용: `before_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        before_completion = self._current_completion_time()
        # LINE-BY-LINE: `before_imbalance`에 `compute_load_imbalance(self.state.machine_loads)` 결과를 저장합니다. 의미/사용: `before_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        before_imbalance = compute_load_imbalance(self.state.machine_loads)

        # LINE-BY-LINE: `operation`에 `None` 결과를 저장합니다. 의미/사용: `operation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation = None
        # LINE-BY-LINE: `operations` 변수에 `[]` 결과를 저장합니다. 의미: `operations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operations: List[ScheduledOperation] = []
        # LINE-BY-LINE: 조건 `selected.action_kind == "open_batch"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.action_kind == "open_batch":
            # LINE-BY-LINE: `new_batch_id`에 `self._apply_open_batch_action(selected)` 결과를 저장합니다. 의미/사용: `new_batch_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            new_batch_id = self._apply_open_batch_action(selected)
            # LINE-BY-LINE: `selected.batch_id`에 `new_batch_id` 결과를 저장합니다. 의미/사용: `batch_id`는 open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            selected.batch_id = new_batch_id
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `selected.action_kind == "add_to_batch"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif selected.action_kind == "add_to_batch":
            # LINE-BY-LINE: `self._apply_add_to_batch_action(selected)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._apply_add_to_batch_action(selected)
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `selected.action_kind == "close_batch"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif selected.action_kind == "close_batch":
            # LINE-BY-LINE: `operations`에 `self._apply_close_batch_action(selected)` 결과를 저장합니다. 의미/사용: `operations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operations = self._apply_close_batch_action(selected)
            # LINE-BY-LINE: `operation`에 `operations[0] if operations else None` 결과를 저장합니다. 의미/사용: `operation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation = operations[0] if operations else None
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `selected.action_kind == "single_dispatch"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif selected.action_kind == "single_dispatch":
            # LINE-BY-LINE: `operation`에 `self._build_scheduled_operation(selected)` 결과를 저장합니다. 의미/사용: `operation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation = self._build_scheduled_operation(selected)
            # LINE-BY-LINE: `self._apply_operation(operation)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._apply_operation(operation)
            # LINE-BY-LINE: `self._emit_operation_events(operation, selected)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._emit_operation_events(operation, selected)
            # LINE-BY-LINE: `operations`에 `[operation]` 결과를 저장합니다. 의미/사용: `operations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operations = [operation]
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation.step_candidate] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation.step_candidate] "
                # LINE-BY-LINE: `f"cause`에 `unsupported_action_kind action_kind={selected.action_kind} action_id={selected.action_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=unsupported_action_kind action_kind={selected.action_kind} action_id={selected.action_id}"
            )
            # LINE-BY-LINE: `ValueError(f"unsupported action_kind: {selected.action_kind}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"unsupported action_kind: {selected.action_kind}")

        # step 후 상태 기준 지표를 다시 계산합니다.
        # LINE-BY-LINE: `after_completion`에 `self._current_completion_time()` 결과를 저장합니다. 의미/사용: `after_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        after_completion = self._current_completion_time()
        # LINE-BY-LINE: `after_imbalance`에 `compute_load_imbalance(self.state.machine_loads)` 결과를 저장합니다. 의미/사용: `after_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        after_imbalance = compute_load_imbalance(self.state.machine_loads)
        # LINE-BY-LINE: 조건 `selected.action_kind in {"single_dispatch", "close_batch"}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.action_kind in {"single_dispatch", "close_batch"}:
            # LINE-BY-LINE: `priority_weight`에 `(` 결과를 저장합니다. 의미/사용: `priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
            priority_weight = (
                # LINE-BY-LINE: `sum(int(self.jobs[job_id].priority_weight) for job_id in selected.batch_job_ids)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                sum(int(self.jobs[job_id].priority_weight) for job_id in selected.batch_job_ids)
                # LINE-BY-LINE: 조건 `selected.batch_job_ids`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if selected.batch_job_ids
                # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                else int(selected.job.priority_weight)
            )
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `priority_weight`에 `0` 결과를 저장합니다. 의미/사용: `priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
            priority_weight = 0

        # LINE-BY-LINE: `reward`에 `compute_step_reward(` 결과를 저장합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        reward = compute_step_reward(
            # LINE-BY-LINE: `before_completion`에 `before_completion` 결과를 저장합니다. 의미/사용: `before_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            before_completion=before_completion,
            # LINE-BY-LINE: `after_completion`에 `after_completion` 결과를 저장합니다. 의미/사용: `after_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            after_completion=after_completion,
            # LINE-BY-LINE: `before_imbalance`에 `before_imbalance` 결과를 저장합니다. 의미/사용: `before_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            before_imbalance=before_imbalance,
            # LINE-BY-LINE: `after_imbalance`에 `after_imbalance` 결과를 저장합니다. 의미/사용: `after_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            after_imbalance=after_imbalance,
            # LINE-BY-LINE: `priority_weight`에 `priority_weight` 결과를 저장합니다. 의미/사용: `priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
            priority_weight=priority_weight,
            # LINE-BY-LINE: `soft_penalty`에 `selected.soft_penalty` 결과를 저장합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_penalty=selected.soft_penalty,
            # LINE-BY-LINE: `reward_config`에 `self.config["reward"]` 결과를 저장합니다. 의미/사용: `reward_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            reward_config=self.config["reward"],
        )

        # LINE-BY-LINE: 조건 `advance_decision_epoch`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if advance_decision_epoch:
            # LINE-BY-LINE: `self._advance_to_decision_epoch()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self._advance_to_decision_epoch()
        # LINE-BY-LINE: `terminated`에 `not self._has_pending_work()` 결과를 저장합니다. 의미/사용: `terminated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        terminated = not self._has_pending_work()
        # LINE-BY-LINE: `truncated`에 `self.state.current_time > self.config["simulation"]["horizon_minutes"]` 결과를 저장합니다. 의미/사용: `truncated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        truncated = self.state.current_time > self.config["simulation"]["horizon_minutes"]
        # LINE-BY-LINE: 조건 `build_observation_result`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if build_observation_result:
            # LINE-BY-LINE: `observation`에 `self.build_observation()` 결과를 저장합니다. 의미/사용: `observation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            observation = self.build_observation()
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `observation`에 `{` 결과를 저장합니다. 의미/사용: `observation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            observation = {
                # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `current_time` 키에 `self.state.current_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "current_time": self.state.current_time,
                # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `remaining_job_count` 키에 `len(self.state.unscheduled_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "remaining_job_count": len(self.state.unscheduled_jobs),
            }
        # LINE-BY-LINE: `info`에 `{` 결과를 저장합니다. 의미/사용: `info` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        info = {
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `selected_action` 키에 `action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "selected_action": action_id,
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `action_kind` 키에 `selected.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_kind": selected.action_kind,
            # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `selected.batch_id or ""` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            "batch_id": selected.batch_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `tuple(selected.batch_job_ids)` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
            "batch_job_ids": tuple(selected.batch_job_ids),
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `selected.job.job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": selected.job.job_id,
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `selected.machine.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": selected.machine.machine_id,
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `operation` 키에 `operation` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "operation": operation,
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `operations` 키에 `operations` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "operations": operations,
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `soft_penalty` 키에 `selected.soft_penalty` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "soft_penalty": selected.soft_penalty,
            # LINE-BY-LINE: `step_candidate`에서 반환/저장할 dict의 `soft_reasons` 키에 `selected.soft_reasons` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "soft_reasons": selected.soft_reasons,
        }
        # LINE-BY-LINE: 호출자에게 `observation, reward, terminated, truncated, info`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return observation, reward, terminated, truncated, info

    # LINE-BY-LINE: `run_spt_fast(self)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def run_spt_fast(self) -> Dict:
        """SPT 전용 빠른 실행 경로. 후보 전체 materialization을 피합니다."""

        # LINE-BY-LINE: 호출자에게 `self._run_fast_selector(self.select_spt_candidate_fast)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._run_fast_selector(self.select_spt_candidate_fast)

    # LINE-BY-LINE: `run_load_balance_fast(self)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def run_load_balance_fast(self) -> Dict:
        """load_balance 전용 빠른 실행 경로. 후보 전체 materialization을 피합니다."""

        # LINE-BY-LINE: 호출자에게 `self._run_fast_selector(self.select_load_balance_candidate_fast)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self._run_fast_selector(self.select_load_balance_candidate_fast)

    # LINE-BY-LINE: `_run_fast_selector(self, selector_fn: Callable[[], Optional[ActionCandidate]])` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def _run_fast_selector(self, selector_fn: Callable[[], Optional[ActionCandidate]]) -> Dict:
        """fast selector 공통 실행 루프."""

        # LINE-BY-LINE: 조건 `self._is_batch_open_mode()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if self._is_batch_open_mode():
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][CuttingSimulation._run_fast_selector] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][CuttingSimulation._run_fast_selector] "
                # LINE-BY-LINE: `"cause`에 `batch_open_mode_not_supported_by_fast_selector"` 결과를 저장합니다. 의미/사용: `"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                "cause=batch_open_mode_not_supported_by_fast_selector"
            )
            # LINE-BY-LINE: `RuntimeError("fast selectors are only valid for job_machine_pair mode")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("fast selectors are only valid for job_machine_pair mode")

        # LINE-BY-LINE: `self.reset()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.reset()
        # LINE-BY-LINE: `horizon_minutes`에 `float(self.config["simulation"]["horizon_minutes"])` 결과를 저장합니다. 의미/사용: `horizon_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        horizon_minutes = float(self.config["simulation"]["horizon_minutes"])
        # LINE-BY-LINE: 조건 `self._has_pending_work() and self.state.current_time <= horizon_minutes`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while self._has_pending_work() and self.state.current_time <= horizon_minutes:
            # LINE-BY-LINE: `selected`에 `selector_fn()` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
            selected = selector_fn()
            # LINE-BY-LINE: 조건 `selected is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if selected is None:
                # LINE-BY-LINE: `before_time`에 `self.state.current_time` 결과를 저장합니다. 의미/사용: `before_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                before_time = self.state.current_time
                # LINE-BY-LINE: `self._advance_to_decision_epoch()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                self._advance_to_decision_epoch()
                # LINE-BY-LINE: 조건 `self.state.current_time <= before_time + 1e-9`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if self.state.current_time <= before_time + 1e-9:
                    # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                    break
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `self.step_candidate(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            self.step_candidate(
                # LINE-BY-LINE: `step_candidate(...)` 호출에 `selected` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                selected,
                # LINE-BY-LINE: `build_observation_result`에 `False` 결과를 저장합니다. 의미/사용: `build_observation_result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                build_observation_result=False,
                # LINE-BY-LINE: `advance_decision_epoch`에 `False` 결과를 저장합니다. 의미/사용: `advance_decision_epoch` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                advance_decision_epoch=False,
            )

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_run_fast_selector`에서 반환/저장할 dict의 `current_time` 키에 `self.state.current_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_time": self.state.current_time,
            # LINE-BY-LINE: 실행 summary의 `makespan` 항목에 `self.get_makespan()`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "makespan": self.get_makespan(),
            # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `len(self.state.schedule)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "scheduled_jobs": len(self.state.schedule),
            # LINE-BY-LINE: 실행 summary의 `unscheduled_jobs` 항목에 `sorted(self.state.unscheduled_jobs)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "unscheduled_jobs": sorted(self.state.unscheduled_jobs),
            # LINE-BY-LINE: `_run_fast_selector`에서 반환/저장할 dict의 `downstream_loads` 키에 `dict(self.state.downstream_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_loads": dict(self.state.downstream_loads),
            # LINE-BY-LINE: `_run_fast_selector`에서 반환/저장할 dict의 `machine_loads` 키에 `dict(self.state.machine_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_loads": dict(self.state.machine_loads),
            # LINE-BY-LINE: `_run_fast_selector`에서 반환/저장할 dict의 `event_log` 키에 `self.export_event_log()` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_log": self.export_event_log(),
            # LINE-BY-LINE: `_run_fast_selector`에서 반환/저장할 dict의 `schedule` 키에 `self.state.schedule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "schedule": self.state.schedule,
        }

    # LINE-BY-LINE: `run(self, policy_fn: Callable[[List[ActionCandidate], "CuttingSimulation"], Optio...)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: CuttingSimulation의 후보 생성, 상태 전진, event 생성 흐름에서 사용됩니다.
    def run(self, policy_fn: Callable[[List[ActionCandidate], "CuttingSimulation"], Optional[ActionCandidate]]) -> Dict:
        """휴리스틱 또는 간단 정책으로 에피소드를 끝까지 실행합니다."""

        # LINE-BY-LINE: `self.reset()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        self.reset()
        # LINE-BY-LINE: 조건 `self._has_pending_work() and self.state.current_time <= self.config["simulation"]["horizon_minutes"]`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while self._has_pending_work() and self.state.current_time <= self.config["simulation"]["horizon_minutes"]:
            # LINE-BY-LINE: `candidates`에 `self.get_candidates()` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
            candidates = self.get_candidates()
            # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not candidates:
                # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                break

            # LINE-BY-LINE: `selected`에 `policy_fn(candidates, self)` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
            selected = policy_fn(candidates, self)
            # LINE-BY-LINE: 조건 `selected is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if selected is None:
                # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                break

            # LINE-BY-LINE: 현재 객체의 `step_candidate(selected, build_observation_result` 속성에 `False)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.step_candidate(selected, build_observation_result=False)

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `run`에서 반환/저장할 dict의 `current_time` 키에 `self.state.current_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_time": self.state.current_time,
            # LINE-BY-LINE: 실행 summary의 `makespan` 항목에 `self.get_makespan()`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "makespan": self.get_makespan(),
            # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `len(self.state.schedule)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "scheduled_jobs": len(self.state.schedule),
            # LINE-BY-LINE: 실행 summary의 `unscheduled_jobs` 항목에 `sorted(self.state.unscheduled_jobs)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
            "unscheduled_jobs": sorted(self.state.unscheduled_jobs),
            # LINE-BY-LINE: `run`에서 반환/저장할 dict의 `downstream_loads` 키에 `dict(self.state.downstream_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "downstream_loads": dict(self.state.downstream_loads),
            # LINE-BY-LINE: `run`에서 반환/저장할 dict의 `machine_loads` 키에 `dict(self.state.machine_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_loads": dict(self.state.machine_loads),
            # LINE-BY-LINE: `run`에서 반환/저장할 dict의 `event_log` 키에 `self.export_event_log()` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_log": self.export_event_log(),
            # LINE-BY-LINE: `run`에서 반환/저장할 dict의 `schedule` 키에 `self.state.schedule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "schedule": self.state.schedule,
        }
