"""비교/검증용 기본 휴리스틱.

이 파일은 강화학습이 없더라도 환경을 바로 실행해볼 수 있게 해줍니다.

현재 제공하는 휴리스틱:
- spt: 처리시간이 짧은 작업 우선
- priority: 우선순위가 높은 작업 우선
- load_balance: 현재 부하가 낮은 설비를 선호
"""

from typing import Sequence


def select_action_by_rule(candidates: Sequence, simulation, rule_name: str):
    """현재 후보 액션 중 하나를 고릅니다.

    candidates는 이미 하드 제약을 통과한 후보들입니다.
    따라서 이 함수는 주로
    - 처리시간
    - 우선순위
    - 부하 편차
    - 소프트 penalty
    를 기준으로 선택합니다.
    """

    if not candidates:
        return None

    if rule_name == "spt":
        # 가장 짧은 처리시간 우선.
        # 다만 soft penalty가 낮은 후보를 약하게 선호합니다.
        return min(
            candidates,
            key=lambda c: (
                c.estimated_minutes,
                c.soft_penalty,
                -c.job.priority_weight,
                c.job.job_id,
            ),
        )

    if rule_name == "priority":
        # 숫자가 클수록 더 중요한 작업이라고 가정합니다.
        # 따라서 정렬 key에서는 -priority_weight를 사용합니다.
        return min(
            candidates,
            key=lambda c: (
                -c.job.priority_weight,
                c.soft_penalty,
                c.estimated_minutes,
                c.job.job_id,
            ),
        )

    if rule_name == "load_balance":
        # 현재 누적 부하가 낮은 설비를 우선 사용해서
        # 설비 간 workload가 한쪽으로 치우치지 않게 유도합니다.
        return min(
            candidates,
            key=lambda c: (
                simulation.machine_loads[c.machine.machine_id],
                c.soft_penalty,
                c.estimated_minutes,
                -c.job.priority_weight,
            ),
        )

    raise ValueError(f"unknown heuristic: {rule_name}")
