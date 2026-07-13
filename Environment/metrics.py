"""Phase 2 학습, 검증, full-flow가 공유하는 일정 metric 계산."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


LOAD_KEYS = ("wo_count", "processing_time_sum", "cut_length_sum", "bevel_quantity_sum")
PHASE2_METRIC_SCHEMA_VERSION = "phase2_metric_v1"
PHASE2_RAW_SCORE_FIELD_NAMES = (
    "hard_violation_count",
    "makespan",
    "bay_internal_wo_count_gap",
    "bay_internal_cut_length_gap",
    "bay_internal_bevel_quantity_gap",
    "bay_internal_occupancy_gap",
)
PHASE2_NORMALIZED_SCORE_FIELD_NAMES = (
    "hard_violation_count",
    "makespan",
    "bay_internal_normalized_wo_count_gap",
    "bay_internal_normalized_cut_length_gap",
    "bay_internal_normalized_bevel_quantity_gap",
    "bay_internal_normalized_occupancy_gap",
)
PHASE2_SCORE_FIELDS_BY_MODE = {
    "raw": PHASE2_RAW_SCORE_FIELD_NAMES,
    "normalized": PHASE2_NORMALIZED_SCORE_FIELD_NAMES,
}


def calculate_phase2_schedule_metrics(
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_clock: Mapping[str, int | float],
    machine_bay_ids: Mapping[str, str],
    batches: Sequence[Mapping],
    max_wo_count: int,
    max_length_sum: float,
    hard_violation_count: int,
) -> Dict:
    """Bay 내부 설비 gap의 합과 전체 공장 진단 metric을 함께 계산한다."""

    _validate_inputs(
        machine_loads,
        machine_clock,
        machine_bay_ids,
        batches,
        max_wo_count,
        max_length_sum,
        hard_violation_count,
    )
    machine_ids_by_bay: Dict[str, list[str]] = {}
    for machine_id in sorted(machine_loads):
        machine_ids_by_bay.setdefault(str(machine_bay_ids[machine_id]), []).append(machine_id)

    per_bay: Dict[str, Dict[str, int | float]] = {}
    for bay_id, machine_ids in sorted(machine_ids_by_bay.items()):
        wo_values = [float(machine_loads[machine_id]["wo_count"]) for machine_id in machine_ids]
        tact_values = [float(machine_loads[machine_id]["processing_time_sum"]) for machine_id in machine_ids]
        cut_values = [float(machine_loads[machine_id]["cut_length_sum"]) for machine_id in machine_ids]
        bevel_values = [float(machine_loads[machine_id]["bevel_quantity_sum"]) for machine_id in machine_ids]
        clock_values = [float(machine_clock[machine_id]) for machine_id in machine_ids]
        per_bay[bay_id] = {
            "machine_count": len(machine_ids),
            "makespan": round(max(clock_values), 9),
            "raw_wo_count_gap": _raw_gap(wo_values),
            "raw_individual_tact_sum_gap": _raw_gap(tact_values),
            "raw_cut_length_gap": _raw_gap(cut_values),
            "raw_bevel_quantity_gap": _raw_gap(bevel_values),
            "raw_occupancy_gap": _raw_gap(clock_values),
            "normalized_wo_count_gap": _normalized_gap(wo_values, f"wo_count:{bay_id}"),
            "normalized_individual_tact_sum_gap": _normalized_gap(tact_values, f"processing_time_sum:{bay_id}"),
            "normalized_cut_length_gap": _normalized_gap(cut_values, f"cut_length_sum:{bay_id}"),
            "normalized_bevel_quantity_gap": _normalized_gap(bevel_values, f"bevel_quantity_sum:{bay_id}"),
            "normalized_occupancy_gap": _normalized_gap(clock_values, f"occupancy:{bay_id}"),
        }

    makespan = round(max(float(value) for value in machine_clock.values()), 9)
    raw_score = (
        hard_violation_count,
        makespan,
        _sum_bay_metric(per_bay, "raw_wo_count_gap"),
        _sum_bay_metric(per_bay, "raw_cut_length_gap"),
        _sum_bay_metric(per_bay, "raw_bevel_quantity_gap"),
        _sum_bay_metric(per_bay, "raw_occupancy_gap"),
    )
    normalized_score = (
        hard_violation_count,
        makespan,
        _sum_bay_metric(per_bay, "normalized_wo_count_gap"),
        _sum_bay_metric(per_bay, "normalized_cut_length_gap"),
        _sum_bay_metric(per_bay, "normalized_bevel_quantity_gap"),
        _sum_bay_metric(per_bay, "normalized_occupancy_gap"),
    )
    all_loads = list(machine_loads.values())
    diagnostics = {
        "global_wo_count_gap": _raw_gap([float(row["wo_count"]) for row in all_loads]),
        "global_individual_tact_sum_gap": _raw_gap([float(row["processing_time_sum"]) for row in all_loads]),
        "global_cut_length_gap": _raw_gap([float(row["cut_length_sum"]) for row in all_loads]),
        "global_bevel_quantity_gap": _raw_gap([float(row["bevel_quantity_sum"]) for row in all_loads]),
        "global_occupancy_gap": _raw_gap([float(value) for value in machine_clock.values()]),
    }
    score_details = {
        "raw_hard_violation_count": raw_score[0],
        "raw_makespan": raw_score[1],
        "raw_bay_internal_wo_count_gap": raw_score[2],
        "raw_bay_internal_cut_length_gap": raw_score[3],
        "raw_bay_internal_bevel_quantity_gap": raw_score[4],
        "raw_bay_internal_occupancy_gap": raw_score[5],
        "normalized_bay_internal_wo_count_gap": normalized_score[2],
        "normalized_bay_internal_cut_length_gap": normalized_score[3],
        "normalized_bay_internal_bevel_quantity_gap": normalized_score[4],
        "normalized_bay_internal_occupancy_gap": normalized_score[5],
    }
    return {
        "hard_violation_count": hard_violation_count,
        "makespan": makespan,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "score_details": score_details,
        "per_bay": per_bay,
        "diagnostics": diagnostics,
    }


def _validate_inputs(
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_clock: Mapping[str, int | float],
    machine_bay_ids: Mapping[str, str],
    batches: Sequence[Mapping],
    max_wo_count: int,
    max_length_sum: float,
    hard_violation_count: int,
) -> None:
    if not machine_loads:
        print("[ERROR][Environment.metrics._validate_inputs] cause=no_machine_loads")
        raise RuntimeError("Phase 2 metrics require machine loads")
    if not batches:
        print("[ERROR][Environment.metrics._validate_inputs] cause=no_batches")
        raise RuntimeError("Phase 2 metrics require batches")
    if max_wo_count <= 0 or max_length_sum <= 0:
        print(
            "[ERROR][Environment.metrics._validate_inputs] "
            f"cause=invalid_batch_limits max_wo_count={max_wo_count} max_length_sum={max_length_sum}"
        )
        raise ValueError("Phase 2 batch limits must be positive")
    if isinstance(hard_violation_count, bool) or not isinstance(hard_violation_count, int) or hard_violation_count < 0:
        print(
            "[ERROR][Environment.metrics._validate_inputs] "
            f"cause=invalid_hard_violation_count value={hard_violation_count}"
        )
        raise RuntimeError("Phase 2 audited hard violation count must be a non-negative integer")
    machine_ids = set(machine_loads)
    if set(machine_clock) != machine_ids or set(machine_bay_ids) != machine_ids:
        print(
            "[ERROR][Environment.metrics._validate_inputs] cause=machine_key_mismatch "
            f"loads={sorted(machine_ids)} clocks={sorted(machine_clock)} bays={sorted(machine_bay_ids)}"
        )
        raise RuntimeError("Phase 2 metric machine keys must match")
    for machine_id, loads in machine_loads.items():
        if not str(machine_bay_ids[machine_id]).strip():
            print(f"[ERROR][Environment.metrics._validate_inputs] cause=missing_bay machine_id={machine_id}")
            raise RuntimeError(f"Phase 2 metric machine has no Bay: {machine_id}")
        for key in LOAD_KEYS:
            if key not in loads:
                print(
                    "[ERROR][Environment.metrics._validate_inputs] "
                    f"cause=missing_load_field machine_id={machine_id} field={key}"
                )
                raise RuntimeError(f"Phase 2 machine load missing field: {key}")
            _require_non_negative(loads[key], f"machine_loads[{machine_id}].{key}")
        _require_non_negative(machine_clock[machine_id], f"machine_clock[{machine_id}]")
    for batch in batches:
        for key in ("wo_count", "length_sum"):
            if key not in batch:
                print(f"[ERROR][Environment.metrics._validate_inputs] cause=missing_batch_field field={key}")
                raise RuntimeError(f"Phase 2 batch missing field: {key}")


def _require_non_negative(value: int | float, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        print(f"[ERROR][Environment.metrics._require_non_negative] cause=not_numeric key={key} value={value}")
        raise RuntimeError(f"Phase 2 metric value is not numeric: {key}") from exc
    if not math.isfinite(number) or number < 0:
        print(f"[ERROR][Environment.metrics._require_non_negative] cause=invalid_value key={key} value={value}")
        raise RuntimeError(f"Phase 2 metric value must be finite and non-negative: {key}")
    return number


def _raw_gap(values: Sequence[float]) -> float:
    return round(max(values) - min(values), 9)


def _normalized_gap(values: Sequence[float], key: str) -> float:
    gap = max(values) - min(values)
    mean = sum(values) / float(len(values))
    if math.isclose(mean, 0.0, rel_tol=0.0, abs_tol=1e-12):
        if math.isclose(gap, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return 0.0
        print(f"[ERROR][Environment.metrics._normalized_gap] cause=zero_mean_nonzero_gap key={key} values={values}")
        raise RuntimeError(f"Phase 2 normalized gap has zero mean: {key}")
    return round(gap / mean, 9)


def _sum_bay_metric(per_bay: Mapping[str, Mapping[str, int | float]], key: str) -> float:
    return round(sum(float(metrics[key]) for metrics in per_bay.values()), 9)


__all__ = ["PHASE2_METRIC_SCHEMA_VERSION", "calculate_phase2_schedule_metrics"]
