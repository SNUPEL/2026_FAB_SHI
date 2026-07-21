"""MIXED Phase 1 pair-policy와 Phase 2가 공유하는 dispatching 휴리스틱."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from Utils.phase1.multi_series_rules import (
    PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
    PHASE1_RESOURCE_POOL_ORDER,
    joint_phase1_bay_capacity_weights,
    normalize_phase1_objective_scope,
    phase1_resource_pool_for_series,
    split_phase1_jobs_by_resource_pool,
)
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


def run_phase1_resource_pool_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Phase1HeuristicCandidate:
    """동일 휴리스틱을 두 자원군에 독립 적용한 뒤 완전 배정으로 병합한다."""

    subproblems = split_phase1_jobs_by_resource_pool(jobs)
    selected = {
        pool_id: run_phase1_heuristic_candidate(
            jobs=pool_jobs,
            bay_ids=bay_ids,
            algorithm=algorithm,
            bay_capacity_weights=bay_capacity_weights,
            objective_scope=objective_scope,
        )
        for pool_id, pool_jobs in subproblems.items()
    }
    return merge_phase1_resource_pool_heuristic_candidates(
        jobs=jobs,
        bay_ids=bay_ids,
        bay_capacity_weights=bay_capacity_weights,
        selected_candidates=selected,
    )


def merge_phase1_resource_pool_heuristic_candidates(
    *,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float] | None,
    selected_candidates: Mapping[str, Phase1HeuristicCandidate],
) -> Phase1HeuristicCandidate:
    """자원군별 휴리스틱 해를 누락·중복·mask 위반 없이 병합한다."""

    subproblems = split_phase1_jobs_by_resource_pool(jobs)
    if set(selected_candidates) != set(subproblems):
        print(
            "[ERROR][phase1_heuristics.merge_phase1_resource_pool_heuristic_candidates] "
            f"cause=subproblem_scope_mismatch expected={sorted(subproblems)} "
            f"actual={sorted(selected_candidates)}"
        )
        raise RuntimeError("Phase 1 heuristic subproblem selections are incomplete")
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    required_weights = joint_phase1_bay_capacity_weights()
    weights = _normalize_bay_capacity_weights(
        normalized_bay_ids,
        required_weights if bay_capacity_weights is None else bay_capacity_weights,
    )
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids)
    expected_blocks_by_pool = {
        pool_id: {
            block.block_set_id
            for block in blocks
            if phase1_resource_pool_for_series(block.family) == pool_id
        }
        for pool_id in subproblems
    }
    for pool_id, candidate in selected_candidates.items():
        if set(candidate.assignments) != expected_blocks_by_pool[pool_id]:
            print(
                "[ERROR][phase1_heuristics.merge_phase1_resource_pool_heuristic_candidates] "
                f"cause=assignment_scope_mismatch subproblem_id={pool_id} "
                f"expected={sorted(expected_blocks_by_pool[pool_id])} "
                f"actual={sorted(candidate.assignments)}"
            )
            raise RuntimeError(f"invalid Phase 1 heuristic assignments: {pool_id}")
    bay_loads = _empty_phase1_bay_loads(normalized_bay_ids, weights)
    assignments: Dict[str, str] = {}
    for block in blocks:
        pool_id = phase1_resource_pool_for_series(block.family)
        selected_bay = selected_candidates[pool_id].assignments[block.block_set_id]
        if selected_bay not in block.allowed_bay_ids:
            print(
                "[ERROR][phase1_heuristics.merge_phase1_resource_pool_heuristic_candidates] "
                f"cause=masked_assignment block_set_id={block.block_set_id} "
                f"bay_id={selected_bay} allowed={list(block.allowed_bay_ids)}"
            )
            raise RuntimeError(f"masked Phase 1 heuristic assignment: {block.block_set_id}")
        assignments[block.block_set_id] = selected_bay
        _add_block_load(bay_loads[selected_bay], selected_bay, block)
    source = "|".join(
        f"{pool_id}:{selected_candidates[pool_id].source}"
        for pool_id in PHASE1_RESOURCE_POOL_ORDER
        if pool_id in selected_candidates
    )
    return Phase1HeuristicCandidate(
        source=source,
        assignments=assignments,
        bay_loads={bay_id: dict(loads) for bay_id, loads in bay_loads.items()},
        transitions=[],
    )


def run_phase1_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Phase1HeuristicCandidate:
    """한 자원군에 확정 mask와 선택한 사전식 score를 적용한다."""

    normalized_objective_scope = normalize_phase1_objective_scope(objective_scope)
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
            key=lambda bay_id: _projected_score(
                bay_loads,
                bay_id,
                block,
                normalized_objective_scope,
            ),
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
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> tuple:
    """선택한 범위의 W/O -> CUT -> BV 사전식 score를 반환한다."""

    return _multi_objective_load_score(bay_loads, objective_scope)


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
    objective_scope: str,
) -> tuple:
    projected = {
        current_bay_id: dict(loads)
        for current_bay_id, loads in bay_loads.items()
    }
    _add_block_load(projected[bay_id], bay_id, block)
    return score_phase1_bay_loads(projected, objective_scope) + (str(bay_id),)
