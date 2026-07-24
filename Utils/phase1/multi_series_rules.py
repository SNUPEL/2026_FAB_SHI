"""2026-07-11 Q&A로 확정된 Phase 1 다계열 Bay 규칙."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from Utils.data.multi_series_cutting_data import MIXED_PLANNING_MACHINE_IDS_BY_BAY


MULTI_SERIES_RULE_PROFILE = "multi_series_260711"
PHASE1_MULTI_SERIES_SCOPE_VERSION = "resource_pool_subproblems_v1"

SERIES_BALANCING_GROUP = {
    "NP": "NP",
    "FN": "FN",
    "FL": "FL",
    "NC": "NC",
}

PHASE1_RESOURCE_POOL_ORDER = ("NP_NC", "FN_FL")
PHASE1_RESOURCE_POOL_SERIES = {
    "NP_NC": ("NP", "NC"),
    "FN_FL": ("FN", "FL"),
}
PHASE1_RESOURCE_POOL_BAYS = {
    "NP_NC": ("22", "23", "24"),
    "FN_FL": ("25", "trans"),
}
PHASE1_RESOURCE_POOL_BY_SERIES = {
    series: pool_id
    for pool_id, series_values in PHASE1_RESOURCE_POOL_SERIES.items()
    for series in series_values
}

SERIES_ALLOWED_BAYS = {
    "NP": ("22", "23", "24"),
    "FN": ("25", "trans"),
    "FL": ("25", "trans"),
    "NC": ("22", "23", "24"),
}

# 확정 PLS/PLP identity 수를 Phase 1 capacity 분모로 사용한다.
GROUP_BAY_CAPACITY_WEIGHTS = {
    "NP": {
        bay_id: float(len(MIXED_PLANNING_MACHINE_IDS_BY_BAY[bay_id]))
        for bay_id in ("22", "23", "24")
    },
    "NC": {
        bay_id: float(len(MIXED_PLANNING_MACHINE_IDS_BY_BAY[bay_id]))
        for bay_id in ("22", "23", "24")
    },
    "FN": {
        bay_id: float(len(MIXED_PLANNING_MACHINE_IDS_BY_BAY[bay_id]))
        for bay_id in ("25", "trans")
    },
    "FL": {
        bay_id: float(len(MIXED_PLANNING_MACHINE_IDS_BY_BAY[bay_id]))
        for bay_id in ("25", "trans")
    },
}

PHASE1_BALANCING_GROUP_ORDER = ("NP", "NC", "FN", "FL")
JOINT_PHASE1_BAY_CAPACITY_WEIGHTS = {
    bay_id: float(len(machine_ids))
    for bay_id, machine_ids in MIXED_PLANNING_MACHINE_IDS_BY_BAY.items()
}
MULTI_SERIES_LOAD_METRICS = (
    "wo_count",
    "cut_length_sum",
    "bevel_quantity_sum",
)

PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES = "shared_and_series"
PHASE1_OBJECTIVE_SCOPE_SERIES_ONLY = "series_only"
PHASE1_OBJECTIVE_SCOPES = (
    PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
    PHASE1_OBJECTIVE_SCOPE_SERIES_ONLY,
)
PHASE1_OBJECTIVE_FIELDS_BY_SCOPE = {
    PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES: (
        "shared_wo_gap",
        "series_wo_gap",
        "shared_cut_gap",
        "series_cut_gap",
        "shared_bevel_gap",
        "series_bevel_gap",
    ),
    PHASE1_OBJECTIVE_SCOPE_SERIES_ONLY: (
        "series_wo_gap",
        "series_cut_gap",
        "series_bevel_gap",
    ),
}


@dataclass(frozen=True)
class Phase1BayMaskResult:
    """한 block에 적용된 계열 eligibility와 hard mask 결과."""

    series: str
    balancing_group: str
    allowed_bay_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def normalize_phase1_objective_scope(value: object) -> str:
    """Phase 1 목적함수 범위를 검증하고 canonical 문자열로 반환한다."""

    normalized = str(value).strip().lower()
    if normalized not in PHASE1_OBJECTIVE_SCOPES:
        print(
            "[ERROR][multi_series_rules.normalize_phase1_objective_scope] "
            f"cause=invalid_objective_scope value={value} "
            f"allowed={list(PHASE1_OBJECTIVE_SCOPES)}"
        )
        raise RuntimeError(f"invalid Phase 1 objective scope: {value}")
    return normalized


def phase1_objective_field_names(objective_scope: object) -> tuple[str, ...]:
    """선택한 scope의 사전식 score 의미를 순서대로 반환한다."""

    normalized = normalize_phase1_objective_scope(objective_scope)
    return PHASE1_OBJECTIVE_FIELDS_BY_SCOPE[normalized]


def balancing_group_for_series(series: object) -> str:
    """NP, NC, FN, FL 계열별 평준화 그룹을 반환한다."""

    normalized = _normalize_series(series)
    return SERIES_BALANCING_GROUP[normalized]


def phase1_resource_pool_for_series(series: object) -> str:
    """Phase 1 계열이 공유하는 절단 Bay 자원군을 반환한다."""

    normalized = _normalize_series(series)
    return PHASE1_RESOURCE_POOL_BY_SERIES[normalized]


def split_phase1_jobs_by_resource_pool(
    jobs: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """W/O를 설비를 공유하는 `NP_NC`와 `FN_FL` 문제로 엄격히 분리한다."""

    if not jobs:
        print("[ERROR][multi_series_rules.split_phase1_jobs_by_resource_pool] cause=no_jobs")
        raise RuntimeError("Phase 1 resource-pool split requires at least one W/O")
    partitioned: dict[str, dict[str, object]] = {
        pool_id: {} for pool_id in PHASE1_RESOURCE_POOL_ORDER
    }
    for job_key, job in jobs.items():
        family = job.get("family") if isinstance(job, Mapping) else getattr(job, "family", None)
        try:
            pool_id = phase1_resource_pool_for_series(family)
        except RuntimeError as exc:
            print(
                "[ERROR][multi_series_rules.split_phase1_jobs_by_resource_pool] "
                f"cause=unsupported_series job_key={job_key} family={family}"
            )
            raise RuntimeError(f"unsupported Phase 1 series: {family}") from exc
        partitioned[pool_id][str(job_key)] = job
    return {
        pool_id: partitioned[pool_id]
        for pool_id in PHASE1_RESOURCE_POOL_ORDER
        if partitioned[pool_id]
    }


def bay_capacity_weights_for_group(group: object) -> dict[str, float]:
    """평준화 그룹의 Bay별 설비 수 분모를 복사해 반환한다."""

    normalized = str(group).strip().upper()
    if normalized not in GROUP_BAY_CAPACITY_WEIGHTS:
        print(
            "[ERROR][multi_series_rules.bay_capacity_weights_for_group] "
            f"cause=unknown_balancing_group group={group}"
        )
        raise RuntimeError(f"unknown Phase 1 balancing group: {group}")
    return dict(GROUP_BAY_CAPACITY_WEIGHTS[normalized])


def joint_phase1_bay_capacity_weights() -> dict[str, float]:
    """다계열 joint episode의 다섯 Bay와 설비 수 계약을 반환한다."""

    return dict(JOINT_PHASE1_BAY_CAPACITY_WEIGHTS)


def multi_series_group_load_field(group: object, metric: object) -> str:
    """Bay load row에 저장할 그룹별 부하 field 이름을 엄격히 만든다."""

    normalized_group = str(group).strip().upper()
    normalized_metric = str(metric).strip()
    if normalized_group not in GROUP_BAY_CAPACITY_WEIGHTS:
        print(
            "[ERROR][multi_series_rules.multi_series_group_load_field] "
            f"cause=unknown_balancing_group group={group}"
        )
        raise RuntimeError(f"unknown Phase 1 balancing group: {group}")
    if normalized_metric not in MULTI_SERIES_LOAD_METRICS:
        print(
            "[ERROR][multi_series_rules.multi_series_group_load_field] "
            f"cause=unknown_load_metric metric={metric}"
        )
        raise RuntimeError(f"unknown Phase 1 multi-series load metric: {metric}")
    return f"group_{normalized_group.lower()}_{normalized_metric}"


def initialize_multi_series_group_loads(
    bay_loads: dict[str, dict[str, int | float]],
) -> None:
    """다섯 Bay row에 모든 평준화 그룹의 누적 부하 field를 0으로 추가한다."""

    expected_bays = set(JOINT_PHASE1_BAY_CAPACITY_WEIGHTS)
    if set(bay_loads) != expected_bays:
        print(
            "[ERROR][multi_series_rules.initialize_multi_series_group_loads] "
            f"cause=joint_bay_scope_mismatch expected={sorted(expected_bays)} "
            f"actual={sorted(bay_loads)}"
        )
        raise RuntimeError("multi-series Phase 1 group loads require the five-Bay joint scope")
    for bay_id, row in bay_loads.items():
        expected_weight = JOINT_PHASE1_BAY_CAPACITY_WEIGHTS[bay_id]
        try:
            actual_weight = float(row["capacity_weight"])
        except (KeyError, TypeError, ValueError) as exc:
            print(
                "[ERROR][multi_series_rules.initialize_multi_series_group_loads] "
                f"cause=invalid_capacity_weight bay_id={bay_id} row={row}"
            )
            raise RuntimeError(f"invalid Phase 1 capacity weight: {bay_id}") from exc
        if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-9):
            print(
                "[ERROR][multi_series_rules.initialize_multi_series_group_loads] "
                f"cause=capacity_weight_mismatch bay_id={bay_id} "
                f"expected={expected_weight} actual={actual_weight}"
            )
            raise RuntimeError(f"Phase 1 capacity weight mismatch: {bay_id}")
        for group in PHASE1_BALANCING_GROUP_ORDER:
            row[multi_series_group_load_field(group, "wo_count")] = 0
            row[multi_series_group_load_field(group, "cut_length_sum")] = 0.0
            row[multi_series_group_load_field(group, "bevel_quantity_sum")] = 0


def add_multi_series_group_load(
    row: dict[str, int | float],
    *,
    group: object,
    wo_count: int,
    cut_length_sum: float,
    bevel_quantity_sum: int,
) -> None:
    """한 block의 부하를 해당 평준화 그룹 field에만 누적한다."""

    normalized_group = str(group).strip().upper()
    values = {
        "wo_count": wo_count,
        "cut_length_sum": cut_length_sum,
        "bevel_quantity_sum": bevel_quantity_sum,
    }
    for metric, value in values.items():
        field = multi_series_group_load_field(normalized_group, metric)
        if field not in row:
            print(
                "[ERROR][multi_series_rules.add_multi_series_group_load] "
                f"cause=uninitialized_group_load group={normalized_group} metric={metric}"
            )
            raise RuntimeError(
                f"multi-series group load is not initialized: {normalized_group}/{metric}"
            )
        row[field] += value


def multi_series_group_load_value(
    row: Mapping[str, int | float],
    group: object,
    metric: object,
) -> float:
    """초기화된 그룹별 Bay 부하를 숫자로 읽고 누락 시 실패한다."""

    field = multi_series_group_load_field(group, metric)
    if field not in row:
        print(
            "[ERROR][multi_series_rules.multi_series_group_load_value] "
            f"cause=missing_group_load field={field}"
        )
        raise RuntimeError(f"missing multi-series group load: {field}")
    try:
        value = float(row[field])
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][multi_series_rules.multi_series_group_load_value] "
            f"cause=non_numeric_group_load field={field} value={row[field]}"
        )
        raise RuntimeError(f"non-numeric multi-series group load: {field}") from exc
    if not math.isfinite(value) or value < 0:
        print(
            "[ERROR][multi_series_rules.multi_series_group_load_value] "
            f"cause=invalid_group_load field={field} value={value}"
        )
        raise RuntimeError(f"invalid multi-series group load: {field}")
    return value


def apply_phase1_series_bay_mask(
    *,
    series: object,
    block_no: object,
    width_max: object,
    cut_length_sum: object,
    requested_bays: Iterable[object],
) -> Phase1BayMaskResult:
    """계열 eligibility와 NP 광폭/CNT/장척 hard mask를 적용한다."""

    normalized_series = _normalize_series(series)
    normalized_block = _required_text(block_no, "BLK_NO").upper()
    normalized_requested = _normalize_requested_bays(requested_bays)
    width = _required_non_negative_number(width_max, "BTH")
    cut_length = _required_non_negative_number(cut_length_sum, "CUT_LTH")

    eligible = set(SERIES_ALLOWED_BAYS[normalized_series])
    reasons = [f"series_{normalized_series.lower()}_eligibility"]
    if normalized_series == "NP":
        if width > 4500.0:
            eligible &= {"22", "23"}
            reasons.append("np_wide_bth_gt_4500")
        if normalized_block.startswith("CNT_BLK"):
            eligible &= {"22", "23"}
            reasons.append("np_cnt_block")
        if cut_length >= 1000.0:
            eligible &= {"22", "23"}
            reasons.append("np_cut_lth_ge_1000")

    allowed = tuple(bay_id for bay_id in normalized_requested if bay_id in eligible)
    if not allowed:
        print(
            "[ERROR][multi_series_rules.apply_phase1_series_bay_mask] "
            f"cause=no_feasible_bay series={normalized_series} block_no={normalized_block} "
            f"requested_bays={normalized_requested} eligible_bays={sorted(eligible)}"
        )
        raise RuntimeError(
            f"no feasible Phase 1 Bay: series={normalized_series} block_no={normalized_block}"
        )
    return Phase1BayMaskResult(
        series=normalized_series,
        balancing_group=SERIES_BALANCING_GROUP[normalized_series],
        allowed_bay_ids=allowed,
        reason_codes=tuple(reasons),
    )


def _normalize_series(series: object) -> str:
    normalized = _required_text(series, "GYEL").upper()
    if normalized not in SERIES_BALANCING_GROUP:
        print(
            "[ERROR][multi_series_rules._normalize_series] "
            f"cause=unsupported_series series={normalized}"
        )
        raise RuntimeError(f"unsupported Phase 1 series: {normalized}")
    return normalized


def _normalize_requested_bays(values: Iterable[object]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    if not normalized:
        print("[ERROR][multi_series_rules._normalize_requested_bays] cause=no_requested_bays")
        raise RuntimeError("Phase 1 requires requested Bays")
    if len(set(normalized)) != len(normalized):
        print(
            "[ERROR][multi_series_rules._normalize_requested_bays] "
            f"cause=duplicate_requested_bays values={normalized}"
        )
        raise RuntimeError(f"duplicate Phase 1 requested Bays: {normalized}")
    return normalized


def _required_text(value: object, field: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][multi_series_rules._required_text] "
            f"cause=missing_required_text field={field}"
        )
        raise RuntimeError(f"missing Phase 1 rule field: {field}")
    return text


def _required_non_negative_number(value: object, field: str) -> float:
    if value in (None, ""):
        print(
            "[ERROR][multi_series_rules._required_non_negative_number] "
            f"cause=missing_required_number field={field}"
        )
        raise RuntimeError(f"missing Phase 1 rule field: {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][multi_series_rules._required_non_negative_number] "
            f"cause=invalid_required_number field={field} value={value}"
        )
        raise RuntimeError(f"invalid Phase 1 rule field: {field}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        print(
            "[ERROR][multi_series_rules._required_non_negative_number] "
            f"cause=invalid_required_number field={field} value={value}"
        )
        raise RuntimeError(f"invalid Phase 1 rule field: {field}")
    return parsed
