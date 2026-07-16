"""MIXED Phase 1 graph, 휴리스틱, plan을 연결하는 실행 계층."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    Phase1HeuristicCandidate,
    run_phase1_heuristic_candidate,
    score_phase1_bay_loads,
)
from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    joint_phase1_bay_capacity_weights,
)
from Utils.phase1.phase1_bay_balancer import _collect_blocks, _normalize_bay_ids


def run_phase1_graph_workflow(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str] | None = None,
    heuristic_algorithms: Sequence[str] = PHASE1_HEURISTIC_BANK,
) -> Dict:
    """MIXED joint 5-Bay graph에서 휴리스틱 후보 중 최선 plan을 반환한다."""

    weights = joint_phase1_bay_capacity_weights()
    normalized_bay_ids = _joint_bay_ids(bay_ids, weights)
    graph = build_phase1_block_bay_graph(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        bay_capacity_weights=weights,
    )
    candidates = [
        run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=normalized_bay_ids,
            algorithm=algorithm,
            bay_capacity_weights=weights,
        )
        for algorithm in heuristic_algorithms
    ]
    if not candidates:
        print("[ERROR][Phase1.orchestrator.run_phase1_graph_workflow] cause=no_candidates")
        raise RuntimeError("Phase 1 workflow requires at least one candidate")

    best = min(candidates, key=lambda candidate: score_phase1_bay_loads(candidate.bay_loads))
    return {
        "phase": "phase1",
        "graph": graph,
        "candidate_count": len(candidates),
        "candidates": [_candidate_summary(candidate) for candidate in candidates],
        "best_source": best.source,
        "best_candidate": _candidate_summary(best),
        "plan": _candidate_to_plan(jobs, normalized_bay_ids, best),
    }


def _candidate_summary(candidate: Phase1HeuristicCandidate) -> Dict:
    return {
        "source": candidate.source,
        "score": list(score_phase1_bay_loads(candidate.bay_loads)),
        "assignment_count": len(candidate.assignments),
        "assignments": dict(candidate.assignments),
        "bay_loads": candidate.bay_loads,
    }


def _candidate_to_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    candidate: Any,
) -> Dict:
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
    score = list(score_phase1_bay_loads(candidate.bay_loads))
    return {
        "phase": "phase1_block_series_to_bay",
        "algorithm": candidate.source,
        "score_mode": "wo_first",
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
) -> Dict:
    """완성된 MIXED 후보를 Phase 2가 읽는 plan 계약으로 변환한다."""

    weights = joint_phase1_bay_capacity_weights()
    return _candidate_to_plan(jobs, _joint_bay_ids(bay_ids, weights), candidate)


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
