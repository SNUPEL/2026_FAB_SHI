"""MIXED Phase 1 pair-policy와 Phase 2가 공유하는 dispatching 휴리스틱."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights
from Utils.phase1.phase1_bay_balancer import (
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _empty_phase1_bay_loads,
    _multi_objective_load_score,
    _normalize_bay_capacity_weights,
    _normalize_bay_ids,
    _validate_multi_series_plan_scope,
)


PHASE1_HEURISTIC_BANK = (
    "wo_first_balanced",
    "cut_first_balanced",
    "bevel_first_balanced",
)


@dataclass(frozen=True)
class Phase1HeuristicCandidate:
    """한 휴리스틱이 만든 완전한 block-series -> Bay 배정."""

    source: str
    assignments: Dict[str, str]
    bay_loads: Dict[str, Dict[str, int | float]]
    transitions: List[object]


def run_phase1_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
) -> Phase1HeuristicCandidate:
    """확정된 MIXED hard mask와 W/O-first score로 한 후보를 만든다."""

    if algorithm not in PHASE1_HEURISTIC_BANK:
        print(
            "[ERROR][phase1_heuristics.run_phase1_heuristic_candidate] "
            f"cause=unsupported_algorithm algorithm={algorithm}"
        )
        raise RuntimeError(f"unsupported MIXED Phase 1 heuristic: {algorithm}")

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    required_weights = joint_phase1_bay_capacity_weights()
    if tuple(normalized_bay_ids) != tuple(required_weights):
        print(
            "[ERROR][phase1_heuristics.run_phase1_heuristic_candidate] "
            f"cause=joint_bay_scope_mismatch expected={list(required_weights)} "
            f"actual={list(normalized_bay_ids)}"
        )
        raise RuntimeError("MIXED Phase 1 requires the joint five-Bay scope")
    weights = _normalize_bay_capacity_weights(
        normalized_bay_ids,
        required_weights if bay_capacity_weights is None else bay_capacity_weights,
    )
    if weights != required_weights:
        print(
            "[ERROR][phase1_heuristics.run_phase1_heuristic_candidate] "
            f"cause=capacity_weight_mismatch expected={required_weights} actual={weights}"
        )
        raise RuntimeError("MIXED Phase 1 Bay capacity weights do not match the confirmed contract")

    blocks = _sort_blocks(
        _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids),
        algorithm.removesuffix("_balanced"),
    )
    _validate_multi_series_plan_scope(blocks, normalized_bay_ids, weights)
    bay_loads = _empty_phase1_bay_loads(normalized_bay_ids, weights)
    assignments: Dict[str, str] = {}
    for block in blocks:
        selected_bay = min(
            block.allowed_bay_ids,
            key=lambda bay_id: _projected_score(bay_loads, bay_id, block),
        )
        assignments[block.block_set_id] = selected_bay
        _add_block_load(bay_loads[selected_bay], selected_bay, block)

    return Phase1HeuristicCandidate(
        source=algorithm,
        assignments=assignments,
        bay_loads={bay_id: dict(loads) for bay_id, loads in bay_loads.items()},
        transitions=[],
    )


def score_phase1_bay_loads(
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> tuple:
    """MIXED Phase 1의 W/O -> CUT -> BV 사전식 score를 반환한다."""

    return _multi_objective_load_score(bay_loads)


def _sort_blocks(blocks: Sequence[Phase1Block], order_name: str) -> List[Phase1Block]:
    if order_name == "wo_first":
        key = lambda block: (
            -block.wo_count,
            -block.cut_length_sum,
            -block.bevel_quantity_sum,
            block.block_set_id,
        )
    elif order_name == "cut_first":
        key = lambda block: (
            -block.cut_length_sum,
            -block.wo_count,
            -block.bevel_quantity_sum,
            block.block_set_id,
        )
    elif order_name == "bevel_first":
        key = lambda block: (
            -block.bevel_quantity_sum,
            -block.wo_count,
            -block.cut_length_sum,
            block.block_set_id,
        )
    else:
        print(
            "[ERROR][phase1_heuristics._sort_blocks] "
            f"cause=unsupported_order order_name={order_name}"
        )
        raise RuntimeError(f"unsupported MIXED Phase 1 block order: {order_name}")
    return sorted(blocks, key=key)


def _projected_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    bay_id: str,
    block: Phase1Block,
) -> tuple:
    projected = {
        current_bay_id: dict(loads)
        for current_bay_id, loads in bay_loads.items()
    }
    _add_block_load(projected[bay_id], bay_id, block)
    return score_phase1_bay_loads(projected) + (str(bay_id),)
