"""보상 계산 함수.

현재 reward는 "step 1개를 적용했을 때 시스템이 좋아졌는가?"를 보는 구조입니다.

조합 항목:
- makespan 증가량
- 설비 부하 불균형 증가량
- 우선 작업 보너스
- soft constraint penalty

즉, 절대 makespan 값 그 자체보다
"이번 선택 때문에 전체 완료시점이 얼마나 늦어졌는가"를 보는 방식입니다.
"""

# LINE-BY-LINE: `typing` 모듈에서 `Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict


# LINE-BY-LINE: `compute_load_imbalance(machine_loads: Dict[str, float])` 함수를 정의합니다. 반환 타입: `float`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def compute_load_imbalance(machine_loads: Dict[str, float]) -> float:
    """설비 부하 편차를 분산 형태로 계산합니다.

    값이 클수록 특정 설비에 workload가 더 몰려 있다는 뜻입니다.
    """

    # LINE-BY-LINE: 조건 `not machine_loads`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not machine_loads:
        # LINE-BY-LINE: 호출자에게 `0.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return 0.0
    # LINE-BY-LINE: `average`에 `sum(machine_loads.values()) / len(machine_loads)` 결과를 저장합니다. 의미/사용: `average` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    average = sum(machine_loads.values()) / len(machine_loads)
    # LINE-BY-LINE: 호출자에게 `sum((load - average) ** 2 for load in machine_loads.values())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sum((load - average) ** 2 for load in machine_loads.values())


# LINE-BY-LINE: `compute_step_reward` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def compute_step_reward(
    # LINE-BY-LINE: `before_completion`를 `float,` 타입으로 선언합니다. 의미/사용: `before_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    before_completion: float,
    # LINE-BY-LINE: `after_completion`를 `float,` 타입으로 선언합니다. 의미/사용: `after_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    after_completion: float,
    # LINE-BY-LINE: `before_imbalance`를 `float,` 타입으로 선언합니다. 의미/사용: `before_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    before_imbalance: float,
    # LINE-BY-LINE: `after_imbalance`를 `float,` 타입으로 선언합니다. 의미/사용: `after_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    after_imbalance: float,
    # LINE-BY-LINE: `priority_weight`를 `float,` 타입으로 선언합니다. 의미/사용: `priority_weight`는 작업 우선순위 가중치입니다. 사용: priority 휴리스틱과 reward 계산.
    priority_weight: float,
    # LINE-BY-LINE: `soft_penalty`를 `float,` 타입으로 선언합니다. 의미/사용: `soft_penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    soft_penalty: float,
    # LINE-BY-LINE: `reward_config`를 `Dict,` 타입으로 선언합니다. 의미/사용: `reward_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward_config: Dict,
# LINE-BY-LINE: `) -> float:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> float:
    """step 보상 계산.

    before_completion / after_completion:
      액션 전후의 예상 전체 완료 시점
    before_imbalance / after_imbalance:
      액션 전후의 설비 부하 편차
    priority_weight:
      긴급 작업에 대한 보너스
    soft_penalty:
      소프트 제약 위반 penalty 합
    """

    # LINE-BY-LINE: `delta_completion`에 `after_completion - before_completion` 결과를 저장합니다. 의미/사용: `delta_completion` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    delta_completion = after_completion - before_completion
    # LINE-BY-LINE: `delta_imbalance`에 `after_imbalance - before_imbalance` 결과를 저장합니다. 의미/사용: `delta_imbalance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    delta_imbalance = after_imbalance - before_imbalance

    # 기본 원칙:
    # - 완료시점이 늦어질수록 reward 감소
    # - 부하 불균형이 커질수록 reward 감소
    # - priority가 높은 작업은 약간 보너스
    # - soft 위반은 별도 penalty
    # LINE-BY-LINE: `reward`에 `0.0` 결과를 저장합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward = 0.0
    # LINE-BY-LINE: `reward` 값을 `reward_config["makespan_weight"] * delta_completion` 기준으로 차감합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward -= reward_config["makespan_weight"] * delta_completion
    # LINE-BY-LINE: `reward` 값을 `reward_config["load_balance_weight"] * delta_imbalance` 기준으로 차감합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward -= reward_config["load_balance_weight"] * delta_imbalance
    # LINE-BY-LINE: `reward` 값을 `reward_config["priority_bonus_weight"] * priority_weight` 기준으로 누적/증가합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward += reward_config["priority_bonus_weight"] * priority_weight
    # LINE-BY-LINE: `reward` 값을 `reward_config["soft_penalty_weight"] * soft_penalty` 기준으로 차감합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reward -= reward_config["soft_penalty_weight"] * soft_penalty
    # LINE-BY-LINE: 호출자에게 `reward`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return reward
