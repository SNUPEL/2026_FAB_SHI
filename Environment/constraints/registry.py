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

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass
# LINE-BY-LINE: `typing` 모듈에서 `Dict, List`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List

# LINE-BY-LINE: `.base` 모듈에서 `ConstraintContext, ConstraintResult`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .base import ConstraintContext, ConstraintResult
# LINE-BY-LINE: `.calendar_rules` 모듈에서 `check_calendar_open, check_machine_breakdown, check_machine_calendar_open`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .calendar_rules import check_calendar_open, check_machine_breakdown, check_machine_calendar_open
# LINE-BY-LINE: `.capacity_rules` 모듈에서 `check_machine_day_length_sum_limit, check_machine_day_wo_count_limit`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .capacity_rules import check_machine_day_length_sum_limit, check_machine_day_wo_count_limit
# LINE-BY-LINE: `.downstream_rules` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .downstream_rules import (
    # LINE-BY-LINE: `import(...)` 호출에 `check_downstream_buffer_warning` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_downstream_buffer_warning,
    # LINE-BY-LINE: `import(...)` 호출에 `check_downstream_capacity` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_downstream_capacity,
    # LINE-BY-LINE: `import(...)` 호출에 `check_downstream_priority` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_downstream_priority,
)
# LINE-BY-LINE: `.future_rules` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .future_rules import (
    # LINE-BY-LINE: `import(...)` 호출에 `check_allowed_bay_ids` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_allowed_bay_ids,
    # LINE-BY-LINE: `import(...)` 호출에 `check_allowed_machine_ids` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_allowed_machine_ids,
    # LINE-BY-LINE: `import(...)` 호출에 `check_batch_length_sum_limit` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_batch_length_sum_limit,
    # LINE-BY-LINE: `import(...)` 호출에 `check_batch_wo_count_limit` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_batch_wo_count_limit,
    # LINE-BY-LINE: `import(...)` 호출에 `check_bay_priority_tier_policy` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_bay_priority_tier_policy,
    # LINE-BY-LINE: `import(...)` 호출에 `check_downstream_due_date` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_downstream_due_date,
    # LINE-BY-LINE: `import(...)` 호출에 `check_downstream_required_start` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_downstream_required_start,
    # LINE-BY-LINE: `import(...)` 호출에 `check_machine_priority_tier_policy` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_machine_priority_tier_policy,
    # LINE-BY-LINE: `import(...)` 호출에 `check_plate_width_range` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_plate_width_range,
    # LINE-BY-LINE: `import(...)` 호출에 `check_prohibited_machine_ids` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_prohibited_machine_ids,
    # LINE-BY-LINE: `import(...)` 호출에 `check_table_type_eligibility` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_table_type_eligibility,
)
# LINE-BY-LINE: `.layout_rules` 모듈에서 `check_block_set_same_bay, check_machine_bay_consistency`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .layout_rules import check_block_set_same_bay, check_machine_bay_consistency
# LINE-BY-LINE: `.machine_rules` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .machine_rules import (
    # LINE-BY-LINE: `import(...)` 호출에 `check_auxiliary_resources_available` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_auxiliary_resources_available,
    # LINE-BY-LINE: `import(...)` 호출에 `check_daily_job_cap_limit` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_daily_job_cap_limit,
    # LINE-BY-LINE: `import(...)` 호출에 `check_daily_capacity_limit` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_daily_capacity_limit,
    # LINE-BY-LINE: `import(...)` 호출에 `check_family_eligibility` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_family_eligibility,
    # LINE-BY-LINE: `import(...)` 호출에 `check_machine_enabled` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_machine_enabled,
    # LINE-BY-LINE: `import(...)` 호출에 `check_machine_single_processing` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_machine_single_processing,
    # LINE-BY-LINE: `import(...)` 호출에 `check_table_length_limit` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_table_length_limit,
    # LINE-BY-LINE: `import(...)` 호출에 `check_thickness_range` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_thickness_range,
)
# LINE-BY-LINE: `.soft_rules` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .soft_rules import (
    # LINE-BY-LINE: `import(...)` 호출에 `check_due_date_urgency` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_due_date_urgency,
    # LINE-BY-LINE: `import(...)` 호출에 `check_load_balance_preference` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_load_balance_preference,
    # LINE-BY-LINE: `import(...)` 호출에 `check_preferred_machine_type` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    check_preferred_machine_type,
)


# LINE-BY-LINE: `RULES`에 `{` 결과를 저장합니다. 의미/사용: `RULES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
RULES = {
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `calendar_open` 키에 `check_calendar_open` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "calendar_open": check_calendar_open,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_calendar_open` 키에 `check_machine_calendar_open` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_calendar_open": check_machine_calendar_open,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_breakdown` 키에 `check_machine_breakdown` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_breakdown": check_machine_breakdown,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_enabled` 키에 `check_machine_enabled` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_enabled": check_machine_enabled,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `family_eligibility` 키에 `check_family_eligibility` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "family_eligibility": check_family_eligibility,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `thickness_range` 키에 `check_thickness_range` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "thickness_range": check_thickness_range,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_length_limit` 키에 `check_table_length_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "table_length_limit": check_table_length_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_single_processing` 키에 `check_machine_single_processing` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_single_processing": check_machine_single_processing,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_bay_consistency` 키에 `check_machine_bay_consistency` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_bay_consistency": check_machine_bay_consistency,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `block_set_same_bay` 키에 `check_block_set_same_bay` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "block_set_same_bay": check_block_set_same_bay,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `auxiliary_resources_available` 키에 `check_auxiliary_resources_available` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "auxiliary_resources_available": check_auxiliary_resources_available,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_capacity_limit` 키에 `check_daily_capacity_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "daily_capacity_limit": check_daily_capacity_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_job_cap_limit` 키에 `check_daily_job_cap_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "daily_job_cap_limit": check_daily_job_cap_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_day_wo_count_limit` 키에 `check_machine_day_wo_count_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_day_wo_count_limit": check_machine_day_wo_count_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_day_length_sum_limit` 키에 `check_machine_day_length_sum_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_day_length_sum_limit": check_machine_day_length_sum_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_capacity` 키에 `check_downstream_capacity` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_capacity": check_downstream_capacity,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_priority` 키에 `check_downstream_priority` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_priority": check_downstream_priority,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_buffer_warning` 키에 `check_downstream_buffer_warning` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_buffer_warning": check_downstream_buffer_warning,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `due_date_urgency` 키에 `check_due_date_urgency` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "due_date_urgency": check_due_date_urgency,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `preferred_machine_type` 키에 `check_preferred_machine_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "preferred_machine_type": check_preferred_machine_type,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `load_balance_preference` 키에 `check_load_balance_preference` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "load_balance_preference": check_load_balance_preference,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `batch_wo_count_limit` 키에 `check_batch_wo_count_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "batch_wo_count_limit": check_batch_wo_count_limit,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `batch_length_sum_limit` 키에 `check_batch_length_sum_limit` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "batch_length_sum_limit": check_batch_length_sum_limit,
    # LINE-BY-LINE: 딕셔너리 키 `allowed_machine_ids`에는 `check_allowed_machine_ids` 값을 넣습니다. 의미: W/O별 허용 설비 whitelist입니다. 예: `("PLS21",)`, 사용: allowed_machine_ids rule.
    "allowed_machine_ids": check_allowed_machine_ids,
    # LINE-BY-LINE: 딕셔너리 키 `prohibited_machine_ids`에는 `check_prohibited_machine_ids` 값을 넣습니다. 의미: W/O별 금지 설비 blacklist입니다. 사용: prohibited_machine_ids rule.
    "prohibited_machine_ids": check_prohibited_machine_ids,
    # LINE-BY-LINE: 딕셔너리 키 `allowed_bay_ids`에는 `check_allowed_bay_ids` 값을 넣습니다. 의미: W/O별 허용 절단 Bay whitelist입니다. 예: `("22",)`, 사용: allowed_bay_ids rule.
    "allowed_bay_ids": check_allowed_bay_ids,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_type_eligibility` 키에 `check_table_type_eligibility` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "table_type_eligibility": check_table_type_eligibility,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `plate_width_range` 키에 `check_plate_width_range` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "plate_width_range": check_plate_width_range,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_priority_tier_policy` 키에 `check_machine_priority_tier_policy` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_priority_tier_policy": check_machine_priority_tier_policy,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `bay_priority_tier_policy` 키에 `check_bay_priority_tier_policy` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "bay_priority_tier_policy": check_bay_priority_tier_policy,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_due_date` 키에 `check_downstream_due_date` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_due_date": check_downstream_due_date,
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_required_start` 키에 `check_downstream_required_start` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_required_start": check_downstream_required_start,
}


# LINE-BY-LINE: `RULE_CATEGORIES`에 `{` 결과를 저장합니다. 의미/사용: `RULE_CATEGORIES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
RULE_CATEGORIES = {
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `calendar_open` 키에 `"calendar"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "calendar_open": "calendar",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_calendar_open` 키에 `"calendar"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_calendar_open": "calendar",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_breakdown` 키에 `"calendar"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_breakdown": "calendar",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_enabled` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_enabled": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `family_eligibility` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "family_eligibility": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `thickness_range` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "thickness_range": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_length_limit` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "table_length_limit": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_single_processing` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_single_processing": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_bay_consistency` 키에 `"layout"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_bay_consistency": "layout",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `block_set_same_bay` 키에 `"layout"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "block_set_same_bay": "layout",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `auxiliary_resources_available` 키에 `"layout"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "auxiliary_resources_available": "layout",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_capacity_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "daily_capacity_limit": "capacity",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `daily_job_cap_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "daily_job_cap_limit": "capacity",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_day_wo_count_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_day_wo_count_limit": "capacity",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_day_length_sum_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_day_length_sum_limit": "capacity",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_capacity` 키에 `"downstream"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_capacity": "downstream",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_priority` 키에 `"priority"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_priority": "priority",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_buffer_warning` 키에 `"downstream"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_buffer_warning": "downstream",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `due_date_urgency` 키에 `"priority"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "due_date_urgency": "priority",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `preferred_machine_type` 키에 `"preference"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "preferred_machine_type": "preference",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `load_balance_preference` 키에 `"preference"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "load_balance_preference": "preference",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `batch_wo_count_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "batch_wo_count_limit": "capacity",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `batch_length_sum_limit` 키에 `"capacity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "batch_length_sum_limit": "capacity",
    # LINE-BY-LINE: 딕셔너리 키 `allowed_machine_ids`에는 `"machine"` 값을 넣습니다. 의미: W/O별 허용 설비 whitelist입니다. 예: `("PLS21",)`, 사용: allowed_machine_ids rule.
    "allowed_machine_ids": "machine",
    # LINE-BY-LINE: 딕셔너리 키 `prohibited_machine_ids`에는 `"machine"` 값을 넣습니다. 의미: W/O별 금지 설비 blacklist입니다. 사용: prohibited_machine_ids rule.
    "prohibited_machine_ids": "machine",
    # LINE-BY-LINE: 딕셔너리 키 `allowed_bay_ids`에는 `"layout"` 값을 넣습니다. 의미: W/O별 허용 절단 Bay whitelist입니다. 예: `("22",)`, 사용: allowed_bay_ids rule.
    "allowed_bay_ids": "layout",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `table_type_eligibility` 키에 `"layout"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "table_type_eligibility": "layout",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `plate_width_range` 키에 `"machine"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "plate_width_range": "machine",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `machine_priority_tier_policy` 키에 `"priority"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "machine_priority_tier_policy": "priority",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `bay_priority_tier_policy` 키에 `"priority"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "bay_priority_tier_policy": "priority",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_due_date` 키에 `"downstream"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_due_date": "downstream",
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_required_start` 키에 `"downstream"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_required_start": "downstream",
}


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `CandidateConstraintBundle` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
class CandidateConstraintBundle:
    """후보 액션 1개에 대한 제약 평가 묶음.

    입력:
    - `ConstraintManager.evaluate_candidate()`가 hard rule 결과와 soft rule 결과를 따로 넣는다.

    출력/사용:
    - `hard_passed`: 후보를 action list에 남길지 결정한다.
    - `hard_reasons`: 왜 후보가 탈락했는지 debug/report에 쓴다.
    - `soft_penalty`: 후보는 유지하되 reward나 휴리스틱 정렬에서 불리하게 만든다.
    - `soft_reasons`: 사람이 읽는 설명용이다.
    """

    # LINE-BY-LINE: `hard_results`를 `List[ConstraintResult]` 타입으로 선언합니다. 의미/사용: `CandidateConstraintBundle.hard_results` 필드/속성입니다. 사용: CandidateConstraintBundle 객체를 만들거나 이후 로직에서 참조합니다.
    hard_results: List[ConstraintResult]
    # LINE-BY-LINE: `soft_results`를 `List[ConstraintResult]` 타입으로 선언합니다. 의미/사용: `CandidateConstraintBundle.soft_results` 필드/속성입니다. 사용: CandidateConstraintBundle 객체를 만들거나 이후 로직에서 참조합니다.
    soft_results: List[ConstraintResult]

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `hard_passed(self)` 함수를 정의합니다. 반환 타입: `bool`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def hard_passed(self) -> bool:
        """모든 hard rule을 통과했는지 반환한다."""

        # LINE-BY-LINE: 호출자에게 `all(result.passed for result in self.hard_results)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return all(result.passed for result in self.hard_results)

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `hard_reasons(self)` 함수를 정의합니다. 반환 타입: `List[str]`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def hard_reasons(self) -> List[str]:
        """통과하지 못한 hard rule의 reason만 모아 반환한다."""

        # LINE-BY-LINE: 호출자에게 `[result.reason for result in self.hard_results if not result.passed]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return [result.reason for result in self.hard_results if not result.passed]

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `soft_penalty(self)` 함수를 정의합니다. 반환 타입: `float`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def soft_penalty(self) -> float:
        """통과하지 못한 soft rule의 penalty 합을 반환한다."""

        # LINE-BY-LINE: 호출자에게 `sum(result.penalty for result in self.soft_results if not result.passed)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return sum(result.penalty for result in self.soft_results if not result.passed)

    # LINE-BY-LINE: `@property` 데코레이터를 바로 다음 class/function에 적용합니다. 사용: 생성/검증/테스트 동작을 보강합니다.
    @property
    # LINE-BY-LINE: `soft_reasons(self)` 함수를 정의합니다. 반환 타입: `List[str]`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def soft_reasons(self) -> List[str]:
        """통과하지 못한 soft rule의 reason만 모아 반환한다."""

        # LINE-BY-LINE: 호출자에게 `[result.reason for result in self.soft_results if not result.passed]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return [result.reason for result in self.soft_results if not result.passed]


# LINE-BY-LINE: `ConstraintManager` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
class ConstraintManager:
    """config 기준으로 제약을 나눠 평가합니다.

    현재는 아래 순서로 판단합니다.

    1. category가 꺼져 있으면 해당 규칙 전체 skip
    2. hard_enabled / soft_enabled에서 꺼져 있으면 skip
    3. 실행된 soft rule은 penalty weight를 곱해서 반영
    """

    # LINE-BY-LINE: `__init__` 함수를 정의합니다. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def __init__(
        # LINE-BY-LINE: `__init__(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `hard_enabled`를 `Dict[str, bool],` 타입으로 선언합니다. 의미/사용: `ConstraintManager.hard_enabled` 필드/속성입니다. 사용: ConstraintManager 객체를 만들거나 이후 로직에서 참조합니다.
        hard_enabled: Dict[str, bool],
        # LINE-BY-LINE: `soft_enabled`를 `Dict[str, bool],` 타입으로 선언합니다. 의미/사용: `ConstraintManager.soft_enabled` 필드/속성입니다. 사용: ConstraintManager 객체를 만들거나 이후 로직에서 참조합니다.
        soft_enabled: Dict[str, bool],
        # LINE-BY-LINE: `soft_weights`를 `Dict[str, float],` 타입으로 선언합니다. 의미/사용: `ConstraintManager.soft_weights` 필드/속성입니다. 사용: ConstraintManager 객체를 만들거나 이후 로직에서 참조합니다.
        soft_weights: Dict[str, float],
        # LINE-BY-LINE: `category_flags` 변수에 `None` 결과를 저장합니다. 의미: `category_flags` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        category_flags: Dict[str, bool] | None = None,
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        """config에서 읽은 제약 활성화 정보를 보관한다."""

        # LINE-BY-LINE: 현재 객체의 `hard_enabled` 속성에 `hard_enabled` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.hard_enabled = hard_enabled
        # LINE-BY-LINE: 현재 객체의 `soft_enabled` 속성에 `soft_enabled` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.soft_enabled = soft_enabled
        # LINE-BY-LINE: 현재 객체의 `soft_weights` 속성에 `soft_weights` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.soft_weights = soft_weights
        # LINE-BY-LINE: 현재 객체의 `category_flags` 속성에 `category_flags or {}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.category_flags = category_flags or {}

    # LINE-BY-LINE: `_is_category_enabled(self, rule_name: str)` 함수를 정의합니다. 반환 타입: `bool`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def _is_category_enabled(self, rule_name: str) -> bool:
        """규칙이 속한 상위 category가 켜져 있는지 확인합니다."""

        # LINE-BY-LINE: `category`에 `RULE_CATEGORIES.get(rule_name)` 결과를 저장합니다. 의미/사용: `category` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        category = RULE_CATEGORIES.get(rule_name)
        # LINE-BY-LINE: 조건 `category is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if category is None:
            # LINE-BY-LINE: 호출자에게 `True`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return True
        # LINE-BY-LINE: 호출자에게 `bool(self.category_flags.get(category, True))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return bool(self.category_flags.get(category, True))

    # LINE-BY-LINE: `_evaluate_rule_group(self, context: ConstraintContext, enabled_flags: Dict[str, bool], apply_weigh...)` 함수를 정의합니다. 반환 타입: `List[ConstraintResult]`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def _evaluate_rule_group(self, context: ConstraintContext, enabled_flags: Dict[str, bool], apply_weight: bool) -> List[ConstraintResult]:
        """hard 또는 soft rule 묶음을 순서대로 실행한다.

        입력:
        - context: 지금 평가하려는 job-machine 후보의 전체 문맥
        - enabled_flags: config에서 켜진 rule 목록
        - apply_weight: soft rule이면 penalty weight를 곱한다

        실패 처리:
        - registry에 없는 rule 이름은 조용히 무시하지 않는다.
        - 원인을 print한 뒤 KeyError를 발생시켜 config 오류를 바로 드러낸다.
        """

        # LINE-BY-LINE: `results` 변수에 `[]` 결과를 저장합니다. 의미: `results` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        results: List[ConstraintResult] = []
        # LINE-BY-LINE: `rule_name, enabled in enabled_flags.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for rule_name, enabled in enabled_flags.items():
            # LINE-BY-LINE: 조건 `not enabled`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not enabled:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: 조건 `not self._is_category_enabled(rule_name)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not self._is_category_enabled(rule_name):
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: 조건 `rule_name not in RULES`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if rule_name not in RULES:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][ConstraintManager._evaluate_rule_group] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][ConstraintManager._evaluate_rule_group] "
                    # LINE-BY-LINE: `f"cause`에 `unknown_rule rule_name={rule_name}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=unknown_rule rule_name={rule_name}"
                )
                # LINE-BY-LINE: `KeyError(f"unknown constraint rule: {rule_name}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise KeyError(f"unknown constraint rule: {rule_name}")

            # LINE-BY-LINE: `result`에 `RULES[rule_name](context)` 결과를 저장합니다. 의미/사용: `result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            result = RULES[rule_name](context)
            # LINE-BY-LINE: 조건 `apply_weight and not result.passed`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if apply_weight and not result.passed:
                # LINE-BY-LINE: `result.penalty` 값을 `self.soft_weights.get(rule_name, 1.0)` 기준으로 곱해서 갱신합니다. 의미/사용: `penalty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                result.penalty *= self.soft_weights.get(rule_name, 1.0)
            # LINE-BY-LINE: `results.append(result)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            results.append(result)
        # LINE-BY-LINE: 호출자에게 `results`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return results

    # LINE-BY-LINE: `evaluate_candidate(self, context: ConstraintContext)` 함수를 정의합니다. 반환 타입: `CandidateConstraintBundle`. 사용: 제약 registry가 이 함수를 호출해 후보 action의 통과/탈락을 판단합니다.
    def evaluate_candidate(self, context: ConstraintContext) -> CandidateConstraintBundle:
        """후보 action 1개에 대해 hard/soft 제약을 모두 평가한다."""

        # LINE-BY-LINE: 호출자에게 `CandidateConstraintBundle(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return CandidateConstraintBundle(
            # LINE-BY-LINE: `hard_results`에 `self._evaluate_rule_group(context, self.hard_enabled, apply_weight=False)` 결과를 저장합니다. 의미/사용: `hard_results` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            hard_results=self._evaluate_rule_group(context, self.hard_enabled, apply_weight=False),
            # LINE-BY-LINE: `soft_results`에 `self._evaluate_rule_group(context, self.soft_enabled, apply_weight=True)` 결과를 저장합니다. 의미/사용: `soft_results` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            soft_results=self._evaluate_rule_group(context, self.soft_enabled, apply_weight=True),
        )
