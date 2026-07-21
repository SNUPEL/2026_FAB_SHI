"""MIXED Phase 1 graph, 휴리스틱, plan을 연결하는 실행 계층."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    Phase1HeuristicCandidate,
    merge_phase1_resource_pool_heuristic_candidates,
    run_phase1_heuristic_candidate,
    score_phase1_bay_loads,
)
from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
    PHASE1_RESOURCE_POOL_ORDER,
    joint_phase1_bay_capacity_weights,
    normalize_phase1_objective_scope,
    split_phase1_jobs_by_resource_pool,
)
from Utils.phase1.phase1_bay_balancer import _collect_blocks, _normalize_bay_ids


def run_phase1_graph_workflow(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str] | None = None,
    heuristic_algorithms: Sequence[str] = PHASE1_HEURISTIC_BANK,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Dict:
    """두 자원군에서 휴리스틱을 독립 비교한 뒤 하나의 plan으로 병합한다."""

    weights = joint_phase1_bay_capacity_weights()
    normalized_bay_ids = _joint_bay_ids(bay_ids, weights)
    normalized_objective_scope = normalize_phase1_objective_scope(objective_scope)
    graph = build_phase1_block_bay_graph(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        bay_capacity_weights=weights,
    )
    algorithms = tuple(heuristic_algorithms)
    if not algorithms:
        print("[ERROR][Phase1.orchestrator.run_phase1_graph_workflow] cause=no_candidates")
        raise RuntimeError("Phase 1 workflow requires at least one candidate")
    subproblems = split_phase1_jobs_by_resource_pool(jobs)
    candidate_rows = []
    selected_candidates: Dict[str, Phase1HeuristicCandidate] = {}
    for pool_id in PHASE1_RESOURCE_POOL_ORDER:
        pool_jobs = subproblems.get(pool_id)
        if not pool_jobs:
            continue
        candidates = [
            run_phase1_heuristic_candidate(
                jobs=pool_jobs,
                bay_ids=normalized_bay_ids,
                algorithm=algorithm,
                bay_capacity_weights=weights,
                objective_scope=normalized_objective_scope,
            )
            for algorithm in algorithms
        ]
        best = min(
            candidates,
            key=lambda candidate: score_phase1_bay_loads(
                candidate.bay_loads,
                objective_scope=normalized_objective_scope,
            ),
        )
        selected_candidates[pool_id] = best
        candidate_rows.extend(
            _candidate_summary(
                candidate,
                objective_scope=normalized_objective_scope,
                subproblem_id=pool_id,
            )
            for candidate in candidates
        )
    merged_best = merge_phase1_resource_pool_heuristic_candidates(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        bay_capacity_weights=weights,
        selected_candidates=selected_candidates,
    )
    return {
        "phase": "phase1",
        "graph": graph,
        "subproblem_count": len(subproblems),
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "best_source": merged_best.source,
        "best_candidate": _candidate_summary(
            merged_best,
            objective_scope=normalized_objective_scope,
            subproblem_id="MERGED",
        ),
        "selected_sources_by_subproblem": {
            pool_id: candidate.source
            for pool_id, candidate in selected_candidates.items()
        },
        "plan": _candidate_to_plan(
            jobs,
            normalized_bay_ids,
            merged_best,
            objective_scope=normalized_objective_scope,
        ),
    }


def _candidate_summary(
    candidate: Phase1HeuristicCandidate,
    *,
    objective_scope: str,
    subproblem_id: str,
) -> Dict:
    return {
        "subproblem_id": subproblem_id,
        "source": candidate.source,
        "score": list(
            score_phase1_bay_loads(
                candidate.bay_loads,
                objective_scope=objective_scope,
            )
        ),
        "assignment_count": len(candidate.assignments),
        "assignments": dict(candidate.assignments),
        "bay_loads": candidate.bay_loads,
    }


def _candidate_to_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    candidate: Any,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Dict:
    normalized_objective_scope = normalize_phase1_objective_scope(objective_scope)
    blocks = _collect_blocks(jobs=jobs, bay_ids=tuple(bay_ids))
    assignments = []
    for block in blocks:
        assigned_bay = candidate.assignments.get(block.block_set_id)
        if assigned_bay is None:
            print(
                "[ERROR][Phase1.orchestrator._candidate_to_plan] "
                f"cause=missing_assignment block_set_id={block.block_set_id} "
                f"source={candidate.source}"
            )
            raise RuntimeError(f"missing Phase 1 assignment for {block.block_set_id}")
        assignments.append(
            {
                "block_set_id": block.block_set_id,
                "project_no": block.project_no,
                "series": block.family,
                "block_no": block.block_no,
                "assigned_bay": assigned_bay,
                "candidate_bays": "|".join(block.allowed_bay_ids),
                "job_ids": "|".join(block.job_ids),
                "wo_count": block.wo_count,
                "cut_length_sum": round(block.cut_length_sum, 6),
                "bevel_quantity_sum": block.bevel_quantity_sum,
                "steel_quantity_sum": block.steel_quantity_sum,
                "width_max": round(block.width_max, 6),
                "long_cut_over_1000": block.long_cut_over_1000,
                "wide_plate_over_4500": block.wide_plate_over_4500,
                "cnt_block": block.cnt_block,
                "bay_mask_reason_codes": "|".join(block.bay_mask_reason_codes),
            }
        )
    score = list(
        score_phase1_bay_loads(
            candidate.bay_loads,
            objective_scope=normalized_objective_scope,
        )
    )
    return {
        "phase": "phase1_block_series_to_bay",
        "algorithm": candidate.source,
        "score_mode": "wo_first",
        "objective_scope": normalized_objective_scope,
        "rule_profile": MULTI_SERIES_RULE_PROFILE,
        "scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
        "score": score,
        "bay_capacity_weights": {
            str(bay_id): float(candidate.bay_loads[str(bay_id)]["capacity_weight"])
            for bay_id in bay_ids
        },
        "summary": {
            "job_count": len(jobs),
            "block_count": len(blocks),
            "assignment_count": len(assignments),
            "score": score,
        },
        "bay_loads": candidate.bay_loads,
        "assignments": assignments,
    }


def candidate_to_phase1_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    candidate: Any,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Dict:
    """완성된 MIXED 후보를 Phase 2가 읽는 plan 계약으로 변환한다."""

    weights = joint_phase1_bay_capacity_weights()
    return _candidate_to_plan(
        jobs,
        _joint_bay_ids(bay_ids, weights),
        candidate,
        objective_scope=objective_scope,
    )


def _joint_bay_ids(
    bay_ids: Sequence[str] | None,
    weights: Mapping[str, float],
) -> tuple[str, ...]:
    expected = tuple(weights)
    normalized = expected if bay_ids is None else _normalize_bay_ids(bay_ids)
    if normalized != expected:
        print(
            "[ERROR][Phase1.orchestrator._joint_bay_ids] "
            f"cause=joint_bay_scope_mismatch expected={list(expected)} actual={list(normalized)}"
        )
        raise RuntimeError("MIXED Phase 1 requires the joint five-Bay scope")
    return normalized
