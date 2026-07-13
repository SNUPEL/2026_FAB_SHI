"""Graph MDP state builders for Phase 1 and Phase 2.

This module does not train a model. It creates the variable-size node/edge
state that a later pointer/GNN policy can consume without hard-coding Bay
22/23/24 or a fixed number of machines.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, Mapping, Sequence

from Utils.phase1.phase1_bay_balancer import (
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _job_attr,
    _multi_objective_load_score,
    _normalize_bay_ids,
    _require_capacity_weight,
    _require_non_negative_float,
    _require_text,
)


PHASE1_BLOCK_NODE_FEATURES = [
    "steel_quantity_ratio",
    "cut_length_ratio",
    "bevel_quantity_ratio",
    "long_cut_over_1000",
    "plate_length_avg_ratio",
    "thickness_avg_ratio",
]

PHASE1_BAY_NODE_FEATURES = [
    "current_steel_quantity_ratio",
    "current_cut_length_ratio",
    "current_bevel_quantity_ratio",
    "current_block_count_ratio",
    "capacity_weight_ratio",
]

PHASE1_BLOCK_BAY_EDGE_FEATURES = [
    "block_steel_quantity_ratio",
    "block_cut_length_ratio",
    "block_bevel_quantity_ratio",
    "bay_current_steel_quantity_ratio",
    "bay_current_cut_length_ratio",
    "bay_current_bevel_quantity_ratio",
    "bay_capacity_weight_ratio",
    "projected_steel_gap_ratio",
    "projected_cut_gap_ratio",
    "projected_bevel_gap_ratio",
]

PHASE2_WO_NODE_FEATURES = [
    "processing_time_ratio",
    "cut_length_ratio",
    "bevel_quantity_ratio",
    "plate_length_ratio",
    "thickness_ratio",
]

PHASE2_MACHINE_NODE_FEATURES = [
    "current_wo_count_ratio",
    "current_processing_time_ratio",
    "current_cut_length_ratio",
    "current_bevel_quantity_ratio",
]

PHASE2_WO_MACHINE_EDGE_FEATURES = [
    "job_processing_time_ratio",
    "job_cut_length_ratio",
    "job_bevel_quantity_ratio",
    "machine_current_wo_count_ratio",
    "machine_current_processing_time_ratio",
    "same_phase1_bay",
]


def build_phase1_block_bay_graph(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_loads: Mapping[str, Mapping[str, int | float]] | None = None,
    assigned_block_ids: Iterable[str] | None = None,
    long_cut_hard_mask: bool = True,
) -> Dict:
    """Build a variable-size Phase 1 graph state.

    Nodes:
    - block nodes: one node per unassigned block.
    - Bay nodes: one node per available Bay.

    Edges:
    - block -> Bay candidate edges after confirmed hard masks.
    """

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        require_multi_objective=True,
        long_cut_hard_mask=long_cut_hard_mask,
    )
    assigned = {str(block_id) for block_id in (assigned_block_ids or [])}
    active_blocks = [block for block in blocks if block.block_set_id not in assigned]
    current_loads = _copy_or_empty_bay_loads(normalized_bay_ids, bay_loads)
    totals = _phase1_totals(blocks, current_loads)

    block_nodes = [_phase1_block_node(block, totals) for block in active_blocks]
    bay_nodes = [_phase1_bay_node(bay_id, current_loads[bay_id], totals) for bay_id in normalized_bay_ids]
    candidate_edges = []
    for block in active_blocks:
        for bay_id in block.allowed_bay_ids:
            candidate_edges.append(_phase1_block_bay_edge(block, bay_id, current_loads, totals))

    return {
        "phase": "phase1_block_bay",
        "feature_names": {
            "block": PHASE1_BLOCK_NODE_FEATURES,
            "bay": PHASE1_BAY_NODE_FEATURES,
            "edge": PHASE1_BLOCK_BAY_EDGE_FEATURES,
        },
        "block_nodes": block_nodes,
        "bay_nodes": bay_nodes,
        "candidate_edges": candidate_edges,
        "metadata": {
            "bay_ids": list(normalized_bay_ids),
            "block_count": len(blocks),
            "unassigned_block_count": len(active_blocks),
            "candidate_edge_count": len(candidate_edges),
            "long_cut_hard_mask": bool(long_cut_hard_mask),
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
    for job in jobs.values():
        allowed_bays = _phase2_allowed_bays(job, phase1_assignments)
        allowed_machines = _phase2_allowed_machines(job)
        prohibited_machines = _phase2_prohibited_machines(job)
        for machine_id, machine in machine_map.items():
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
        },
    }


def _phase1_block_node(block: Phase1Block, totals: Mapping[str, float]) -> Dict:
    return {
        "node_id": f"block:{block.block_set_id}",
        "block_set_id": block.block_set_id,
        "project_no": block.project_no,
        "block_no": block.block_no,
        "features": [
            _ratio(block.steel_quantity_sum, totals["steel_quantity_sum"]),
            _ratio(block.cut_length_sum, totals["cut_length_sum"]),
            _ratio(block.bevel_quantity_sum, totals["bevel_quantity_sum"]),
            float(block.long_cut_over_1000),
            _ratio(block.length_avg or 0.0, totals["plate_length_avg_max"]),
            _ratio(block.thickness_avg or 0.0, totals["thickness_avg_max"]),
        ],
    }


def _phase1_bay_node(bay_id: str, loads: Mapping[str, int | float], totals: Mapping[str, float]) -> Dict:
    return {
        "node_id": f"bay:{bay_id}",
        "bay_id": bay_id,
        "features": [
            _ratio(loads["steel_quantity_sum"], totals["steel_quantity_sum"]),
            _ratio(loads["cut_length_sum"], totals["cut_length_sum"]),
            _ratio(loads["bevel_quantity_sum"], totals["bevel_quantity_sum"]),
            _ratio(loads["block_count"], totals["block_count"]),
            _ratio(_require_capacity_weight(loads, f"graph_bay_node:{bay_id}"), totals["capacity_weight_sum"]),
        ],
    }


def _phase1_block_bay_edge(
    block: Phase1Block,
    bay_id: str,
    current_loads: Mapping[str, Mapping[str, int | float]],
    totals: Mapping[str, float],
) -> Dict:
    projected_loads = copy.deepcopy(dict(current_loads))
    _add_block_load(projected_loads[bay_id], bay_id, block)
    score = _multi_objective_load_score(projected_loads)
    return {
        "edge_id": f"block:{block.block_set_id}->bay:{bay_id}",
        "source_block_id": block.block_set_id,
        "target_bay_id": bay_id,
        "score": list(score),
        "features": [
            _ratio(block.steel_quantity_sum, totals["steel_quantity_sum"]),
            _ratio(block.cut_length_sum, totals["cut_length_sum"]),
            _ratio(block.bevel_quantity_sum, totals["bevel_quantity_sum"]),
            _ratio(current_loads[bay_id]["steel_quantity_sum"], totals["steel_quantity_sum"]),
            _ratio(current_loads[bay_id]["cut_length_sum"], totals["cut_length_sum"]),
            _ratio(current_loads[bay_id]["bevel_quantity_sum"], totals["bevel_quantity_sum"]),
            _ratio(
                _require_capacity_weight(current_loads[bay_id], f"graph_edge:{block.block_set_id}@{bay_id}"),
                totals["capacity_weight_sum"],
            ),
            _ratio(score[0], totals["steel_quantity_sum"]),
            _ratio(score[1], totals["cut_length_sum"]),
            _ratio(score[2], totals["bevel_quantity_sum"]),
        ],
    }


def _phase2_wo_node(job: object, totals: Mapping[str, float]) -> Dict:
    job_id = _require_text(_job_attr(job, "job_id"), "job_id", "phase2")
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
        ],
    }


def _phase2_machine_node(
    machine_id: str,
    machine: object,
    loads: Mapping[str, int | float],
    totals: Mapping[str, float],
) -> Dict:
    return {
        "node_id": f"machine:{machine_id}",
        "machine_id": machine_id,
        "bay_id": _require_text(_job_attr(machine, "bay_id"), "machine.bay_id", machine_id),
        "features": [
            _ratio(loads["wo_count"], totals["job_count"]),
            _ratio(loads["processing_time_sum"], totals["processing_time_sum"]),
            _ratio(loads["cut_length_sum"], totals["cut_length_sum"]),
            _ratio(loads["bevel_quantity_sum"], totals["bevel_quantity_sum"]),
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
    return {
        "edge_id": f"wo:{job_id}->machine:{machine_id}",
        "source_job_id": job_id,
        "source_block_id": block_set_id,
        "target_machine_id": machine_id,
        "target_bay_id": machine_bay,
        "features": [
            _ratio(_processing_time(job), totals["processing_time_sum"]),
            _ratio(_required_non_negative(job, "cut_length"), totals["cut_length_sum"]),
            _ratio(_required_non_negative(job, "bevel_quantity"), totals["bevel_quantity_sum"]),
            _ratio(machine_loads["wo_count"], totals["job_count"]),
            _ratio(machine_loads["processing_time_sum"], totals["processing_time_sum"]),
            1.0 if allowed_bays and machine_bay in allowed_bays else 0.0,
        ],
    }


def _phase1_totals(
    blocks: Sequence[Phase1Block],
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> Dict[str, float]:
    return {
        "steel_quantity_sum": _positive_total(
            sum(block.steel_quantity_sum for block in blocks)
            + sum(float(row["steel_quantity_sum"]) for row in bay_loads.values()),
            "phase1 steel_quantity_sum",
        ),
        "cut_length_sum": _positive_total(
            sum(block.cut_length_sum for block in blocks)
            + sum(float(row["cut_length_sum"]) for row in bay_loads.values()),
            "phase1 cut_length_sum",
        ),
        "bevel_quantity_sum": max(
            1.0,
            sum(block.bevel_quantity_sum for block in blocks)
            + sum(float(row["bevel_quantity_sum"]) for row in bay_loads.values()),
        ),
        "block_count": max(
            1.0,
            len(blocks) + sum(float(row["block_count"]) for row in bay_loads.values()),
        ),
        "capacity_weight_sum": max(
            1.0,
            sum(
                _require_capacity_weight(row, f"graph_totals:{bay_id}")
                for bay_id, row in bay_loads.items()
            ),
        ),
        "plate_length_avg_max": max(1.0, *(float(block.length_avg or 0.0) for block in blocks)),
        "thickness_avg_max": max(1.0, *(float(block.thickness_avg or 0.0) for block in blocks)),
    }


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
) -> Dict[str, Dict[str, int | float]]:
    if bay_loads is None:
        return {
            bay_id: {
                "steel_quantity_sum": 0,
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
                "long_cut_bay24_count": 0,
                "wo_count": 0,
                "block_count": 0,
                "capacity_weight": 1.0,
            }
            for bay_id in bay_ids
        }
    missing = [bay_id for bay_id in bay_ids if bay_id not in bay_loads]
    if missing:
        print(f"[ERROR][phase_graph_mdp._copy_or_empty_bay_loads] cause=missing_bay_loads bay_ids={missing}")
        raise RuntimeError(f"missing Bay loads: {missing}")
    return {bay_id: dict(bay_loads[bay_id]) for bay_id in bay_ids}


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
