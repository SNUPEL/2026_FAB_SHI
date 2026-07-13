"""2026-07-11 Q&A로 확정된 Phase 1 다계열 Bay 규칙."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


MULTI_SERIES_RULE_PROFILE = "multi_series_260711"
LEGACY_NP_RULE_PROFILE = "legacy_np"

SERIES_BALANCING_GROUP = {
    "NP": "NP",
    "FN": "FN_FL",
    "FL": "FN_FL",
    "NC": "NC",
}

SERIES_ALLOWED_BAYS = {
    "NP": ("22", "23", "24"),
    "FN": ("25", "trans"),
    "FL": ("25", "trans"),
    "NC": ("22", "23", "24"),
}

# 최신 Q&A의 설비 수를 Phase 1 capacity 분모로 사용한다. EQP_NM의 실제
# identity 매핑과 별개인 planning capacity 계약이며, 실제 매핑 수령 후 대조한다.
GROUP_BAY_CAPACITY_WEIGHTS = {
    "NP": {"22": 4.0, "23": 4.0, "24": 3.0},
    "NC": {"22": 4.0, "23": 4.0, "24": 3.0},
    "FN_FL": {"25": 2.0, "trans": 2.0},
}


@dataclass(frozen=True)
class Phase1BayMaskResult:
    """한 block에 적용된 계열 eligibility와 hard mask 결과."""

    series: str
    balancing_group: str
    allowed_bay_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def balancing_group_for_series(series: object) -> str:
    """NP, FN+FL, NC 평준화 그룹을 반환한다."""

    normalized = _normalize_series(series)
    return SERIES_BALANCING_GROUP[normalized]


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
