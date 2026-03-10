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

from typing import Dict


def compute_load_imbalance(machine_loads: Dict[str, float]) -> float:
    """설비 부하 편차를 분산 형태로 계산합니다.

    값이 클수록 특정 설비에 workload가 더 몰려 있다는 뜻입니다.
    """

    if not machine_loads:
        return 0.0
    average = sum(machine_loads.values()) / len(machine_loads)
    return sum((load - average) ** 2 for load in machine_loads.values())


def compute_step_reward(
    before_completion: float,
    after_completion: float,
    before_imbalance: float,
    after_imbalance: float,
    priority_weight: float,
    soft_penalty: float,
    reward_config: Dict,
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

    delta_completion = after_completion - before_completion
    delta_imbalance = after_imbalance - before_imbalance

    # 기본 원칙:
    # - 완료시점이 늦어질수록 reward 감소
    # - 부하 불균형이 커질수록 reward 감소
    # - priority가 높은 작업은 약간 보너스
    # - soft 위반은 별도 penalty
    reward = 0.0
    reward -= reward_config["makespan_weight"] * delta_completion
    reward -= reward_config["load_balance_weight"] * delta_imbalance
    reward += reward_config["priority_bonus_weight"] * priority_weight
    reward -= reward_config["soft_penalty_weight"] * soft_penalty
    return reward
