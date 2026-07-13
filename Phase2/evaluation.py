"""Merged Phase 2 batch-machine schedule의 공통 평가 함수."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from Environment.metrics import calculate_phase2_schedule_metrics


def evaluate_phase2_schedule(
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_bay_ids: Mapping[str, str],
    batches: Sequence[Mapping],
    timeline: Sequence[Mapping],
    max_wo_count: int,
    max_length_sum: float,
    score_mode: str,
    hard_violation_count: int,
) -> Dict:
    """Batch-machine 결과를 공통 metric 계약으로 재계산해 반환한다."""

    if not machine_loads:
        print("[ERROR][Phase2.evaluation.evaluate_phase2_schedule] cause=no_machine_loads")
        raise RuntimeError("machine_loads are required for Phase 2 evaluation")
    if not batches:
        print("[ERROR][Phase2.evaluation.evaluate_phase2_schedule] cause=no_batches")
        raise RuntimeError("batches are required for Phase 2 evaluation")
    if not timeline:
        print("[ERROR][Phase2.evaluation.evaluate_phase2_schedule] cause=no_timeline")
        raise RuntimeError("timeline is required for Phase 2 evaluation")
    normalized_mode = str(score_mode).strip().lower()
    if normalized_mode not in {"raw", "normalized"}:
        print(f"[ERROR][Phase2.evaluation.evaluate_phase2_schedule] cause=unknown_score_mode value={score_mode}")
        raise RuntimeError(f"unknown Phase 2 score mode: {score_mode}")

    machine_clock = {str(machine_id): 0.0 for machine_id in machine_loads}
    for row in timeline:
        machine_id = str(row.get("machine_id") or "").strip()
        if machine_id not in machine_clock:
            print(
                "[ERROR][Phase2.evaluation.evaluate_phase2_schedule] "
                f"cause=timeline_unknown_machine machine_id={machine_id}"
            )
            raise RuntimeError(f"Phase 2 timeline references unknown machine: {machine_id}")
        if "finish_time" not in row:
            print(
                "[ERROR][Phase2.evaluation.evaluate_phase2_schedule] "
                f"cause=timeline_missing_finish_time machine_id={machine_id}"
            )
            raise RuntimeError("Phase 2 timeline row missing finish_time")
        machine_clock[machine_id] = max(machine_clock[machine_id], float(row["finish_time"]))

    metrics = calculate_phase2_schedule_metrics(
        machine_loads=machine_loads,
        machine_clock=machine_clock,
        machine_bay_ids=machine_bay_ids,
        batches=batches,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        hard_violation_count=hard_violation_count,
    )
    selected_score = metrics[f"{normalized_mode}_score"]
    per_bay = metrics["per_bay"]
    diagnostics = metrics["diagnostics"]

    return {
        "score_mode": normalized_mode,
        "score_tuple": selected_score,
        "score_details": metrics["score_details"],
        "per_bay": per_bay,
        "diagnostics": diagnostics,
        "machine_load_score": {
            "bay_internal_wo_count_gap": selected_score[2],
            "bay_internal_cut_length_gap": selected_score[3],
            "bay_internal_bevel_quantity_gap": selected_score[4],
            "bay_internal_occupancy_gap": selected_score[5],
            "bay_internal_individual_tact_sum_gap": round(
                sum(float(row[f"{normalized_mode}_individual_tact_sum_gap"]) for row in per_bay.values()),
                9,
            ),
            "diagnostic_global_wo_count_gap": diagnostics["global_wo_count_gap"],
            "diagnostic_global_individual_tact_sum_gap": diagnostics["global_individual_tact_sum_gap"],
            "diagnostic_global_cut_length_gap": diagnostics["global_cut_length_gap"],
            "diagnostic_global_bevel_quantity_gap": diagnostics["global_bevel_quantity_gap"],
            "diagnostic_global_occupancy_gap": diagnostics["global_occupancy_gap"],
        },
        "batch_score": {
            "batch_count": len(batches),
            "max_batch_wo_count": max(int(batch["wo_count"]) for batch in batches),
            "max_batch_length_sum": round(max(float(batch["length_sum"]) for batch in batches), 6),
        },
        "sequence_score": {
            "makespan": metrics["makespan"],
            "timeline_count": len(timeline),
        },
    }
