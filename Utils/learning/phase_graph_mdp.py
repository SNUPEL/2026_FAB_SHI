"""Graph MDP state builders for Phase 1 and Phase 2.

This module does not train a model. It creates the variable-size node/edge
state that a later pointer/GNN policy can consume without hard-coding Bay
22/23/24 or a fixed number of machines.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping, Sequence

from Utils.data.multi_series_cutting_data import SUPPORTED_SERIES
from Utils.phase1.phase1_bay_balancer import (
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _empty_phase1_bay_loads,
    _job_attr,
    _multi_objective_load_score,
    _normalize_bay_capacity_weights,
    _normalize_bay_ids,
    _require_capacity_weight,
    _require_non_negative_float,
    _require_text,
    _validate_multi_series_plan_scope,
)
from Utils.phase1.multi_series_rules import (
    GROUP_BAY_CAPACITY_WEIGHTS,
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_BALANCING_GROUP_ORDER,
    multi_series_group_load_value,
)


PHASE1_MULTI_SERIES_BLOCK_NODE_FEATURES = [
    "wo_count_ratio",
    "cut_length_ratio",
    "bevel_quantity_ratio",
    "balancing_group_np",
    "balancing_group_fn_fl",
    "balancing_group_nc",
    "wide_plate_over_4500",
    "cnt_block",
    "long_cut_over_1000",
]

PHASE1_MULTI_SERIES_BAY_NODE_FEATURES = [
    "current_wo_per_capacity_ratio",
    "current_cut_per_capacity_ratio",
    "current_bevel_per_capacity_ratio",
    "capacity_weight_ratio",
]

PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES = [
    "block_wo_count_ratio",
    "block_cut_length_ratio",
    "block_bevel_quantity_ratio",
    "balancing_group_np",
    "balancing_group_fn_fl",
    "balancing_group_nc",
    "wide_plate_over_4500",
    "cnt_block",
    "long_cut_over_1000",
    "bay_current_wo_per_capacity_ratio",
    "bay_current_cut_per_capacity_ratio",
    "bay_current_bevel_per_capacity_ratio",
    "bay_capacity_weight_ratio",
    "projected_shared_wo_gap_ratio",
    "projected_series_wo_gap_ratio",
    "projected_shared_cut_gap_ratio",
    "projected_series_cut_gap_ratio",
    "projected_shared_bevel_gap_ratio",
    "projected_series_bevel_gap_ratio",
]

PHASE2_WO_NODE_FEATURES = [
    "processing_time_ratio",
    "cut_length_ratio",
    "bevel_quantity_ratio",
    "plate_length_ratio",
    "thickness_ratio",
    "family_np",
    "family_fn",
    "family_fl",
    "family_nc",
]

PHASE2_MACHINE_NODE_FEATURES = [
    "current_wo_count_ratio",
    "current_processing_time_ratio",
    "current_cut_length_ratio",
    "current_bevel_quantity_ratio",
    "eligible_family_np",
    "eligible_family_fn",
    "eligible_family_fl",
    "eligible_family_nc",
]

PHASE2_WO_MACHINE_EDGE_FEATURES = [
    "job_processing_time_ratio",
    "job_cut_length_ratio",
    "job_bevel_quantity_ratio",
    "machine_current_wo_count_ratio",
    "machine_current_processing_time_ratio",
    "same_phase1_bay",
    "family_eligible",
]


def build_phase1_block_bay_graph(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_loads: Mapping[str, Mapping[str, int | float]] | None = None,
    assigned_block_ids: Iterable[str] | None = None,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
) -> Dict:
    """확정된 MIXED joint 5-Bay Phase 1 graph state를 만든다.

    Nodes:
    - block nodes: one node per unassigned block.
    - Bay nodes: one node per available Bay.

    Edges:
    - block -> Bay candidate edges after confirmed hard masks.
    """

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    if bay_capacity_weights is None:
        print(
            "[ERROR][phase_graph_mdp.build_phase1_block_bay_graph] "
            "cause=missing_capacity_weights"
        )
        raise RuntimeError("MIXED Phase 1 graph requires Bay capacity weights")
    normalized_capacity_weights = _normalize_bay_capacity_weights(
        normalized_bay_ids,
        bay_capacity_weights,
    )
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids)
    _validate_multi_series_plan_scope(blocks, normalized_bay_ids, normalized_capacity_weights)
    assigned = {str(block_id) for block_id in (assigned_block_ids or [])}
    active_blocks = [block for block in blocks if block.block_set_id not in assigned]
    current_loads = _copy_or_empty_bay_loads(
        normalized_bay_ids,
        bay_loads,
        normalized_capacity_weights,
    )
    totals = _phase1_totals(blocks, current_loads)
    feature_names = {
        "block": PHASE1_MULTI_SERIES_BLOCK_NODE_FEATURES,
        "bay": PHASE1_MULTI_SERIES_BAY_NODE_FEATURES,
        "edge": PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES,
    }
    block_nodes = [_phase1_multi_series_block_node(block, totals) for block in active_blocks]
    bay_nodes = [
        _phase1_multi_series_bay_node(bay_id, current_loads[bay_id], totals)
        for bay_id in normalized_bay_ids
    ]
    candidate_edges = []
    for block in active_blocks:
        for bay_id in block.allowed_bay_ids:
            candidate_edges.append(
                _phase1_multi_series_block_bay_edge(block, bay_id, current_loads, totals)
            )

    return {
        "phase": "phase1_block_bay",
        "feature_names": feature_names,
        "block_nodes": block_nodes,
        "bay_nodes": bay_nodes,
        "candidate_edges": candidate_edges,
        "metadata": {
            "bay_ids": list(normalized_bay_ids),
            "block_count": len(blocks),
            "unassigned_block_count": len(active_blocks),
            "candidate_edge_count": len(candidate_edges),
            "score_mode": "wo_first",
            "rule_profile": MULTI_SERIES_RULE_PROFILE,
        },
    }


def build_phase2_wo_machine_graph(
    jobs: Mapping[str, object],
    machines: Mapping[str, object] | Sequence[object],
    phase1_assignments: Mapping[str, str] | None = None,
    machine_loads: Mapping[str, Mapping[str, int | float]] | None = None,
) -> Dict:
    """Build a variable-size Phase 2 W/O-to-machine graph state.

    `phase1_assignments` maps `block_set_id -> bay_id`. When supplied, each W/O
    can connect only to machines in the assigned Bay.
    """

    if not jobs:
        print("[ERROR][phase_graph_mdp.build_phase2_wo_machine_graph] cause=no_jobs")
        raise RuntimeError("Phase 2 graph requires at least one job")
    machine_map = _normalize_machines(machines)
    current_loads = _copy_or_empty_machine_loads(machine_map, machine_loads)
    totals = _phase2_totals(jobs, current_loads)

    wo_nodes = [_phase2_wo_node(job, totals) for job in jobs.values()]
    machine_nodes = [
        _phase2_machine_node(machine_id, machine, current_loads[machine_id], totals)
        for machine_id, machine in machine_map.items()
    ]
    candidate_edges = []
    disabled_edge_mask_count = 0
    family_edge_mask_count = 0
    for job in jobs.values():
        allowed_bays = _phase2_allowed_bays(job, phase1_assignments)
        allowed_machines = _phase2_allowed_machines(job)
        prohibited_machines = _phase2_prohibited_machines(job)
        job_family = _phase2_required_family(job)
        for machine_id, machine in machine_map.items():
            if not _phase2_machine_enabled(machine, machine_id):
                disabled_edge_mask_count += 1
                continue
            if job_family not in _phase2_machine_families(machine, machine_id):
                family_edge_mask_count += 1
                continue
            if machine_id in prohibited_machines:
                continue
            if allowed_machines and machine_id not in allowed_machines:
                continue
            machine_bay = _require_text(_job_attr(machine, "bay_id"), "machine.bay_id", machine_id)
            if allowed_bays and machine_bay not in allowed_bays:
                continue
            candidate_edges.append(
                _phase2_wo_machine_edge(job, machine, current_loads[machine_id], totals, allowed_bays)
            )

    return {
        "phase": "phase2_wo_machine",
        "feature_names": {
            "wo": PHASE2_WO_NODE_FEATURES,
            "machine": PHASE2_MACHINE_NODE_FEATURES,
            "edge": PHASE2_WO_MACHINE_EDGE_FEATURES,
        },
        "wo_nodes": wo_nodes,
        "machine_nodes": machine_nodes,
        "candidate_edges": candidate_edges,
        "metadata": {
            "job_count": len(jobs),
            "machine_count": len(machine_map),
            "candidate_edge_count": len(candidate_edges),
            "phase1_assignment_count": len(phase1_assignments or {}),
            "disabled_edge_mask_count": disabled_edge_mask_count,
            "family_edge_mask_count": family_edge_mask_count,
        },
    }


def _phase1_multi_series_block_node(block: Phase1Block, totals: Mapping[str, float]) -> Dict:
    """신규 다계열 block 상태를 확정 목적함수와 hard mask 기준으로 표현한다."""

    group_flags = _phase1_group_flags(block)
    return {
        "node_id": f"block:{block.block_set_id}",
        "block_set_id": block.block_set_id,
        "project_no": block.project_no,
        "block_no": block.block_no,
        "family": block.family,
        "balancing_group": block.balancing_group,
        "features": [
            _ratio(block.wo_count, totals["wo_count"]),
            _ratio(block.cut_length_sum, totals["cut_length_sum"]),
            _ratio(block.bevel_quantity_sum, totals["bevel_quantity_sum"]),
            *group_flags,
            float(block.wide_plate_over_4500),
            float(block.cnt_block),
            float(block.long_cut_over_1000),
        ],
    }


def _phase1_multi_series_bay_node(
    bay_id: str,
    loads: Mapping[str, int | float],
    totals: Mapping[str, float],
) -> Dict:
    """Bay 누적 부하를 설비 수로 나눈 뒤 문제 평균 대비 비율로 표현한다."""

    return {
        "node_id": f"bay:{bay_id}",
        "bay_id": bay_id,
        "features": [
            _capacity_load_ratio(loads, "wo_count", totals["wo_per_capacity_average"]),
            _capacity_load_ratio(loads, "cut_length_sum", totals["cut_per_capacity_average"]),
            _capacity_load_ratio(loads, "bevel_quantity_sum", totals["bevel_per_capacity_average"]),
            _ratio(_require_capacity_weight(loads, f"multi_graph_bay:{bay_id}"), totals["capacity_weight_sum"]),
        ],
    }


def _phase1_multi_series_block_bay_edge(
    block: Phase1Block,
    bay_id: str,
    current_loads: Mapping[str, Mapping[str, int | float]],
    totals: Mapping[str, float],
) -> Dict:
    """mask를 통과한 `(block, Bay)`의 투입 후 공유/계열별 W/O-first gap을 표현한다."""

    projected_loads = copy.deepcopy(dict(current_loads))
    _add_block_load(projected_loads[bay_id], bay_id, block)
    score = _multi_objective_load_score(projected_loads)
    group_flags = _phase1_group_flags(block)
    group_wo_average = totals[_phase1_group_average_key(block.balancing_group, "wo_count")]
    group_cut_average = totals[_phase1_group_average_key(block.balancing_group, "cut_length_sum")]
    group_bevel_average = totals[
        _phase1_group_average_key(block.balancing_group, "bevel_quantity_sum")
    ]
    return {
        "edge_id": f"block:{block.block_set_id}->bay:{bay_id}",
        "source_block_id": block.block_set_id,
        "target_bay_id": bay_id,
        "score": list(score),
        "mask_reason_codes": list(block.bay_mask_reason_codes),
        "features": [
            _ratio(block.wo_count, totals["wo_count"]),
            _ratio(block.cut_length_sum, totals["cut_length_sum"]),
            _ratio(block.bevel_quantity_sum, totals["bevel_quantity_sum"]),
            *group_flags,
            float(block.wide_plate_over_4500),
            float(block.cnt_block),
            float(block.long_cut_over_1000),
            _group_capacity_load_ratio(
                current_loads[bay_id], block.balancing_group, "wo_count", group_wo_average
            ),
            _group_capacity_load_ratio(
                current_loads[bay_id],
                block.balancing_group,
                "cut_length_sum",
                group_cut_average,
            ),
            _group_capacity_load_ratio(
                current_loads[bay_id],
                block.balancing_group,
                "bevel_quantity_sum",
                group_bevel_average,
            ),
            _ratio(
                _require_capacity_weight(current_loads[bay_id], f"multi_graph_edge:{block.block_set_id}@{bay_id}"),
                totals["capacity_weight_sum"],
            ),
            _ratio(score[0], totals["wo_per_capacity_average"]),
            _ratio(score[1], totals["wo_per_capacity_average"]),
            _ratio(score[2], totals["cut_per_capacity_average"]),
            _ratio(score[3], totals["cut_per_capacity_average"]),
            _ratio(score[4], totals["bevel_per_capacity_average"]),
            _ratio(score[5], totals["bevel_per_capacity_average"]),
        ],
    }


def _phase1_group_flags(block: Phase1Block) -> list[float]:
    groups = ("NP", "FN_FL", "NC")
    if block.balancing_group not in groups:
        print(
            "[ERROR][phase_graph_mdp._phase1_group_flags] "
            f"cause=unknown_balancing_group block_set_id={block.block_set_id} group={block.balancing_group}"
        )
        raise RuntimeError(f"unknown Phase 1 balancing group: {block.balancing_group}")
    return [1.0 if block.balancing_group == group else 0.0 for group in groups]


def _capacity_load_ratio(
    loads: Mapping[str, int | float],
    field: str,
    average_per_capacity: float,
) -> float:
    capacity = _require_capacity_weight(loads, f"multi_graph_load:{field}")
    return _ratio(float(loads[field]) / capacity, average_per_capacity)


def _group_capacity_load_ratio(
    loads: Mapping[str, int | float],
    group: str,
    metric: str,
    average_per_capacity: float,
) -> float:
    """후보 block 그룹의 현재 Bay 부하만 설비 수 기준으로 정규화한다."""

    capacity = _require_capacity_weight(loads, f"multi_graph_group_load:{group}:{metric}")
    return _ratio(
        multi_series_group_load_value(loads, group, metric) / capacity,
        average_per_capacity,
    )


def _phase1_group_average_key(group: str, metric: str) -> str:
    return f"group_{group.lower()}_{metric}_per_capacity_average"


def _phase2_wo_node(job: object, totals: Mapping[str, float]) -> Dict:
    job_id = _require_text(_job_attr(job, "job_id"), "job_id", "phase2")
    family = _phase2_required_family(job)
    return {
        "node_id": f"wo:{job_id}",
        "job_id": job_id,
        "block_set_id": _require_text(_job_attr(job, "block_set_id"), "block_set_id", job_id),
        "features": [
            _ratio(_processing_time(job), totals["processing_time_sum"]),
            _ratio(_required_non_negative(job, "cut_length"), totals["cut_length_sum"]),
            _ratio(_required_non_negative(job, "bevel_quantity"), totals["bevel_quantity_sum"]),
            _ratio(_required_non_negative(job, "plate_length"), totals["plate_length_sum"]),
            _ratio(_required_non_negative(job, "thickness"), totals["thickness_max"]),
            *[1.0 if family == value else 0.0 for value in SUPPORTED_SERIES],
        ],
    }


def _phase2_machine_node(
    machine_id: str,
    machine: object,
    loads: Mapping[str, int | float],
    totals: Mapping[str, float],
) -> Dict:
    eligible_families = _phase2_machine_families(machine, machine_id)
    return {
        "node_id": f"machine:{machine_id}",
        "machine_id": machine_id,
        "bay_id": _require_text(_job_attr(machine, "bay_id"), "machine.bay_id", machine_id),
        "features": [
            _ratio(loads["wo_count"], totals["job_count"]),
            _ratio(loads["processing_time_sum"], totals["processing_time_sum"]),
            _ratio(loads["cut_length_sum"], totals["cut_length_sum"]),
            _ratio(loads["bevel_quantity_sum"], totals["bevel_quantity_sum"]),
            *[1.0 if family in eligible_families else 0.0 for family in SUPPORTED_SERIES],
        ],
    }


def _phase2_wo_machine_edge(
    job: object,
    machine: object,
    machine_loads: Mapping[str, int | float],
    totals: Mapping[str, float],
    allowed_bays: Sequence[str],
) -> Dict:
    job_id = _require_text(_job_attr(job, "job_id"), "job_id", "phase2")
    machine_id = _require_text(_job_attr(machine, "machine_id"), "machine_id", job_id)
    block_set_id = _require_text(_job_attr(job, "block_set_id"), "block_set_id", job_id)
    machine_bay = _require_text(_job_attr(machine, "bay_id"), "machine.bay_id", machine_id)
    family_eligible = _phase2_required_family(job) in _phase2_machine_families(machine, machine_id)
    if not family_eligible:
        print(
            "[ERROR][phase_graph_mdp._phase2_wo_machine_edge] "
            f"cause=ineligible_edge job_id={job_id} machine_id={machine_id}"
        )
        raise RuntimeError("Phase 2 graph cannot emit a family-ineligible edge")
    return {
        "edge_id": f"wo:{job_id}->machine:{machine_id}",
        "source_job_id": job_id,
        "source_block_id": block_set_id,
        "target_machine_id": machine_id,
        "target_bay_id": machine_bay,
        "family_eligible": True,
        "features": [
            _ratio(_processing_time(job), totals["processing_time_sum"]),
            _ratio(_required_non_negative(job, "cut_length"), totals["cut_length_sum"]),
            _ratio(_required_non_negative(job, "bevel_quantity"), totals["bevel_quantity_sum"]),
            _ratio(machine_loads["wo_count"], totals["job_count"]),
            _ratio(machine_loads["processing_time_sum"], totals["processing_time_sum"]),
            1.0 if allowed_bays and machine_bay in allowed_bays else 0.0,
            1.0,
        ],
    }


def _phase2_required_family(job: object) -> str:
    job_id = _require_text(_job_attr(job, "job_id"), "job_id", "phase2_family")
    family = _require_text(_job_attr(job, "family"), "family", job_id).upper()
    if family not in SUPPORTED_SERIES:
        print(
            "[ERROR][phase_graph_mdp._phase2_required_family] "
            f"cause=unsupported_family job_id={job_id} family={family}"
        )
        raise RuntimeError(f"unsupported Phase 2 family: {family}")
    return family


def _phase2_machine_families(machine: object, machine_id: str) -> frozenset[str]:
    raw_values = _job_attr(machine, "eligible_families")
    if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
        print(
            "[ERROR][phase_graph_mdp._phase2_machine_families] "
            f"cause=invalid_type machine_id={machine_id} value={raw_values}"
        )
        raise RuntimeError(f"invalid eligible_families for {machine_id}")
    values = frozenset(str(value).strip().upper() for value in raw_values if str(value).strip())
    unknown = sorted(values - set(SUPPORTED_SERIES))
    if not values or unknown:
        print(
            "[ERROR][phase_graph_mdp._phase2_machine_families] "
            f"cause=invalid_values machine_id={machine_id} values={sorted(values)} unknown={unknown}"
        )
        raise RuntimeError(f"invalid eligible_families for {machine_id}")
    return values


def _phase2_machine_enabled(machine: object, machine_id: str) -> bool:
    value = _job_attr(machine, "enabled")
    if not isinstance(value, bool):
        print(
            "[ERROR][phase_graph_mdp._phase2_machine_enabled] "
            f"cause=not_boolean machine_id={machine_id} value={value}"
        )
        raise RuntimeError(f"invalid enabled flag for {machine_id}")
    return value


def _phase1_totals(
    blocks: Sequence[Phase1Block],
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> Dict[str, float]:
    """전체 MIXED 문제를 기준으로 node/edge 정규화 분모를 계산한다."""

    capacity_weight_sum = max(
        1.0,
        sum(
            _require_capacity_weight(row, f"graph_totals:{bay_id}")
            for bay_id, row in bay_loads.items()
        ),
    )
    wo_count = _positive_total(
        sum(block.wo_count for block in blocks),
        "phase1 wo_count",
    )
    cut_length_sum = _positive_total(
        sum(block.cut_length_sum for block in blocks),
        "phase1 cut_length_sum",
    )
    bevel_quantity_sum = max(
        1.0,
        sum(block.bevel_quantity_sum for block in blocks),
    )
    result = {
        "wo_count": wo_count,
        "cut_length_sum": cut_length_sum,
        "bevel_quantity_sum": bevel_quantity_sum,
        "capacity_weight_sum": capacity_weight_sum,
        "wo_per_capacity_average": wo_count / capacity_weight_sum,
        "cut_per_capacity_average": cut_length_sum / capacity_weight_sum,
        "bevel_per_capacity_average": max(1.0, bevel_quantity_sum / capacity_weight_sum),
    }
    present_groups = {block.balancing_group for block in blocks}
    group_averages: Dict[str, Dict[str, float]] = {}
    for group in PHASE1_BALANCING_GROUP_ORDER:
        group_blocks = [block for block in blocks if block.balancing_group == group]
        capacity_sum = sum(GROUP_BAY_CAPACITY_WEIGHTS[group].values())
        raw_averages = {
            "wo_count": sum(block.wo_count for block in group_blocks) / capacity_sum,
            "cut_length_sum": sum(block.cut_length_sum for block in group_blocks) / capacity_sum,
            "bevel_quantity_sum": sum(block.bevel_quantity_sum for block in group_blocks) / capacity_sum,
        }
        group_averages[group] = raw_averages
        for metric, raw_average in raw_averages.items():
            result[_phase1_group_average_key(group, metric)] = max(1.0, raw_average)
    result["wo_per_capacity_average"] = max(
        1.0,
        sum(group_averages[group]["wo_count"] for group in present_groups),
    )
    result["cut_per_capacity_average"] = max(
        1.0,
        sum(group_averages[group]["cut_length_sum"] for group in present_groups),
    )
    result["bevel_per_capacity_average"] = max(
        1.0,
        sum(group_averages[group]["bevel_quantity_sum"] for group in present_groups),
    )
    return result


def _phase2_totals(
    jobs: Mapping[str, object],
    machine_loads: Mapping[str, Mapping[str, int | float]],
) -> Dict[str, float]:
    job_values = list(jobs.values())
    return {
        "job_count": max(1.0, float(len(jobs))),
        "processing_time_sum": _positive_total(
            sum(_processing_time(job) for job in job_values)
            + sum(float(row["processing_time_sum"]) for row in machine_loads.values()),
            "phase2 processing_time_sum",
        ),
        "cut_length_sum": _positive_total(
            sum(_required_non_negative(job, "cut_length") for job in job_values)
            + sum(float(row["cut_length_sum"]) for row in machine_loads.values()),
            "phase2 cut_length_sum",
        ),
        "bevel_quantity_sum": max(
            1.0,
            sum(_required_non_negative(job, "bevel_quantity") for job in job_values)
            + sum(float(row["bevel_quantity_sum"]) for row in machine_loads.values()),
        ),
        "plate_length_sum": _positive_total(
            sum(_required_non_negative(job, "plate_length") for job in job_values),
            "phase2 plate_length_sum",
        ),
        "thickness_max": max(1.0, *(_required_non_negative(job, "thickness") for job in job_values)),
    }


def _copy_or_empty_bay_loads(
    bay_ids: Sequence[str],
    bay_loads: Mapping[str, Mapping[str, int | float]] | None,
    bay_capacity_weights: Mapping[str, int | float],
) -> Dict[str, Dict[str, int | float]]:
    if bay_loads is None:
        return _empty_phase1_bay_loads(tuple(bay_ids), bay_capacity_weights)
    missing = [bay_id for bay_id in bay_ids if bay_id not in bay_loads]
    if missing:
        print(f"[ERROR][phase_graph_mdp._copy_or_empty_bay_loads] cause=missing_bay_loads bay_ids={missing}")
        raise RuntimeError(f"missing Bay loads: {missing}")
    copied = {bay_id: dict(bay_loads[bay_id]) for bay_id in bay_ids}
    for bay_id, row in copied.items():
        actual = _require_capacity_weight(row, f"graph_input:{bay_id}")
        expected = float(bay_capacity_weights[bay_id])
        if actual != expected:
            print(
                "[ERROR][phase_graph_mdp._copy_or_empty_bay_loads] "
                f"cause=capacity_weight_mismatch bay_id={bay_id} expected={expected} actual={actual}"
            )
            raise RuntimeError(f"Phase 1 graph capacity mismatch: {bay_id}")
    for group in PHASE1_BALANCING_GROUP_ORDER:
        for metric in ("wo_count", "cut_length_sum", "bevel_quantity_sum"):
            for bay_id in bay_ids:
                multi_series_group_load_value(copied[bay_id], group, metric)
    return copied


def _copy_or_empty_machine_loads(
    machines: Mapping[str, object],
    machine_loads: Mapping[str, Mapping[str, int | float]] | None,
) -> Dict[str, Dict[str, int | float]]:
    if machine_loads is None:
        return {
            machine_id: {
                "wo_count": 0,
                "processing_time_sum": 0.0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
            }
            for machine_id in machines
        }
    missing = [machine_id for machine_id in machines if machine_id not in machine_loads]
    if missing:
        print(
            "[ERROR][phase_graph_mdp._copy_or_empty_machine_loads] "
            f"cause=missing_machine_loads machine_ids={missing}"
        )
        raise RuntimeError(f"missing machine loads: {missing}")
    return {machine_id: dict(machine_loads[machine_id]) for machine_id in machines}


def _normalize_machines(machines: Mapping[str, object] | Sequence[object]) -> Dict[str, object]:
    if isinstance(machines, Mapping):
        items = list(machines.items())
    else:
        items = [(_require_text(_job_attr(machine, "machine_id"), "machine_id", "machine"), machine) for machine in machines]
    if not items:
        print("[ERROR][phase_graph_mdp._normalize_machines] cause=no_machines")
        raise RuntimeError("Phase 2 graph requires at least one machine")
    return {
        _require_text(machine_id, "machine_id", "machine"): machine
        for machine_id, machine in sorted(items, key=lambda item: str(item[0]))
    }


def _phase2_allowed_bays(job: object, phase1_assignments: Mapping[str, str] | None) -> tuple[str, ...]:
    job_id = _require_text(_job_attr(job, "job_id"), "job_id", "phase2")
    block_set_id = _require_text(_job_attr(job, "block_set_id"), "block_set_id", job_id)
    if phase1_assignments and block_set_id in phase1_assignments:
        return (_require_text(phase1_assignments[block_set_id], "phase1_bay", block_set_id),)
    allowed = tuple(str(bay).strip() for bay in (_job_attr(job, "allowed_bay_ids") or ()) if str(bay).strip())
    if allowed:
        return allowed
    cut_bay = _job_attr(job, "cut_bay")
    return () if cut_bay in (None, "") else (_require_text(cut_bay, "cut_bay", job_id),)


def _phase2_allowed_machines(job: object) -> tuple[str, ...]:
    return tuple(str(machine).strip() for machine in (_job_attr(job, "allowed_machine_ids") or ()) if str(machine).strip())


def _phase2_prohibited_machines(job: object) -> tuple[str, ...]:
    return tuple(str(machine).strip() for machine in (_job_attr(job, "prohibited_machine_ids") or ()) if str(machine).strip())


def _required_non_negative(job: object, field_name: str) -> float:
    job_id = str(_job_attr(job, "job_id") or "unknown")
    return _require_non_negative_float(_job_attr(job, field_name), field_name, job_id)


def _processing_time(job: object) -> float:
    job_id = str(_job_attr(job, "job_id") or "unknown")
    base_stage_minutes = _job_attr(job, "base_stage_minutes")
    if isinstance(base_stage_minutes, Mapping):
        value = sum(_require_non_negative_float(stage, "base_stage_minutes", job_id) for stage in base_stage_minutes.values())
        return _positive_total(value, f"processing_time:{job_id}")
    value = _job_attr(job, "processing_time")
    return _positive_total(_require_non_negative_float(value, "processing_time", job_id), f"processing_time:{job_id}")


def _positive_total(value: int | float, label: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        print(f"[ERROR][phase_graph_mdp._positive_total] cause=non_positive_total label={label} value={value}")
        raise RuntimeError(f"non_positive_total: {label}")
    return parsed


def _ratio(value: int | float, denominator: int | float) -> float:
    return round(float(value) / float(denominator), 9)
