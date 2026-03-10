"""제약 등록부.

초보자용 규칙:
- 새 제약 함수는 파일에 만들고
- 여기 RULES에 이름만 등록하면
- config.yaml에서 켜고 끌 수 있습니다.

이 파일에는 두 가지 표가 있습니다.

1. `RULES`
   - 규칙 이름 -> 실제 함수

2. `RULE_CATEGORIES`
   - 규칙 이름 -> 사람이 이해하기 쉬운 분류
   - 예: machine / capacity / downstream / priority / preference

즉, 현재 프로젝트는
"제약 종류(category)"와 "하드/소프트"를 동시에 관리할 수 있게 설계합니다.
"""

from dataclasses import dataclass
from typing import Dict, List

from .base import ConstraintContext, ConstraintResult
from .calendar_rules import check_calendar_open, check_machine_breakdown, check_machine_calendar_open
from .downstream_rules import (
    check_downstream_buffer_warning,
    check_downstream_capacity,
    check_downstream_priority,
)
from .machine_rules import (
    check_daily_job_cap_limit,
    check_daily_capacity_limit,
    check_family_eligibility,
    check_machine_enabled,
    check_machine_single_processing,
    check_table_length_limit,
    check_thickness_range,
)
from .soft_rules import (
    check_due_date_urgency,
    check_load_balance_preference,
    check_preferred_machine_type,
)


RULES = {
    "calendar_open": check_calendar_open,
    "machine_calendar_open": check_machine_calendar_open,
    "machine_breakdown": check_machine_breakdown,
    "machine_enabled": check_machine_enabled,
    "family_eligibility": check_family_eligibility,
    "thickness_range": check_thickness_range,
    "table_length_limit": check_table_length_limit,
    "machine_single_processing": check_machine_single_processing,
    "daily_capacity_limit": check_daily_capacity_limit,
    "daily_job_cap_limit": check_daily_job_cap_limit,
    "downstream_capacity": check_downstream_capacity,
    "downstream_priority": check_downstream_priority,
    "downstream_buffer_warning": check_downstream_buffer_warning,
    "due_date_urgency": check_due_date_urgency,
    "preferred_machine_type": check_preferred_machine_type,
    "load_balance_preference": check_load_balance_preference,
}


RULE_CATEGORIES = {
    "calendar_open": "calendar",
    "machine_calendar_open": "calendar",
    "machine_breakdown": "calendar",
    "machine_enabled": "machine",
    "family_eligibility": "machine",
    "thickness_range": "machine",
    "table_length_limit": "machine",
    "machine_single_processing": "machine",
    "daily_capacity_limit": "capacity",
    "daily_job_cap_limit": "capacity",
    "downstream_capacity": "downstream",
    "downstream_priority": "priority",
    "downstream_buffer_warning": "downstream",
    "due_date_urgency": "priority",
    "preferred_machine_type": "preference",
    "load_balance_preference": "preference",
}


@dataclass
class CandidateConstraintBundle:
    """후보 액션 1개에 대한 제약 평가 묶음."""

    hard_results: List[ConstraintResult]
    soft_results: List[ConstraintResult]

    @property
    def hard_passed(self) -> bool:
        return all(result.passed for result in self.hard_results)

    @property
    def hard_reasons(self) -> List[str]:
        return [result.reason for result in self.hard_results if not result.passed]

    @property
    def soft_penalty(self) -> float:
        return sum(result.penalty for result in self.soft_results if not result.passed)

    @property
    def soft_reasons(self) -> List[str]:
        return [result.reason for result in self.soft_results if not result.passed]


class ConstraintManager:
    """config 기준으로 제약을 나눠 평가합니다.

    현재는 아래 순서로 판단합니다.

    1. category가 꺼져 있으면 해당 규칙 전체 skip
    2. hard_enabled / soft_enabled에서 꺼져 있으면 skip
    3. 실행된 soft rule은 penalty weight를 곱해서 반영
    """

    def __init__(
        self,
        hard_enabled: Dict[str, bool],
        soft_enabled: Dict[str, bool],
        soft_weights: Dict[str, float],
        category_flags: Dict[str, bool] | None = None,
    ):
        self.hard_enabled = hard_enabled
        self.soft_enabled = soft_enabled
        self.soft_weights = soft_weights
        self.category_flags = category_flags or {}

    def _is_category_enabled(self, rule_name: str) -> bool:
        """규칙이 속한 상위 category가 켜져 있는지 확인합니다."""

        category = RULE_CATEGORIES.get(rule_name)
        if category is None:
            return True
        return bool(self.category_flags.get(category, True))

    def _evaluate_rule_group(self, context: ConstraintContext, enabled_flags: Dict[str, bool], apply_weight: bool) -> List[ConstraintResult]:
        results: List[ConstraintResult] = []
        for rule_name, enabled in enabled_flags.items():
            if not enabled:
                continue
            if not self._is_category_enabled(rule_name):
                continue

            result = RULES[rule_name](context)
            if apply_weight and not result.passed:
                result.penalty *= self.soft_weights.get(rule_name, 1.0)
            results.append(result)
        return results

    def evaluate_candidate(self, context: ConstraintContext) -> CandidateConstraintBundle:
        return CandidateConstraintBundle(
            hard_results=self._evaluate_rule_group(context, self.hard_enabled, apply_weight=False),
            soft_results=self._evaluate_rule_group(context, self.soft_enabled, apply_weight=True),
        )
