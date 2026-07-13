"""Phase 1 graph state, 휴리스틱, score를 연결하는 얇은 실행 계층."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from Phase1.self_labeling import (
    PHASE1_SELF_LABEL_HEURISTIC_BANK,
    Phase1SelfLabelCandidate,
    _score_bay_loads,
    run_phase1_heuristic_candidate,
)
from Utils.phase1.phase1_bay_balancer import _collect_blocks, _normalize_bay_ids
from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph


def run_phase1_graph_workflow(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    heuristic_algorithms: Sequence[str] = PHASE1_SELF_LABEL_HEURISTIC_BANK,
    score_mode: str = "steel_first",
    long_cut_hard_mask: bool = True,
) -> Dict:
    """기존 Phase 1 graph와 휴리스틱으로 후보를 평가해 공통 plan을 만든다."""

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    graph = build_phase1_block_bay_graph(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        long_cut_hard_mask=long_cut_hard_mask,
    )
    candidates = [
        run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=normalized_bay_ids,
            algorithm=algorithm,
            long_cut_hard_mask=long_cut_hard_mask,
        )
        for algorithm in heuristic_algorithms
    ]
    if not candidates:
        print("[ERROR][Phase1.orchestrator.run_phase1_graph_workflow] cause=no_candidates")
        raise RuntimeError("Phase 1 workflow requires at least one candidate")

    best = min(candidates, key=lambda candidate: _score_bay_loads(candidate.bay_loads, score_mode))
    plan = _candidate_to_plan(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        candidate=best,
        score_mode=score_mode,
        long_cut_hard_mask=long_cut_hard_mask,
    )
    return {
        "phase": "phase1",
        "graph": graph,
        "candidate_count": len(candidates),
        "candidates": [_candidate_summary(candidate, score_mode) for candidate in candidates],
        "best_source": best.source,
        "best_candidate": _candidate_summary(best, score_mode),
        "plan": plan,
    }


def _candidate_summary(candidate: Phase1SelfLabelCandidate, score_mode: str) -> Dict:
    score = _score_bay_loads(candidate.bay_loads, score_mode)
    return {
        "source": candidate.source,
        "score": list(score),
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "assignments": dict(candidate.assignments),
        "bay_loads": candidate.bay_loads,
    }


def _candidate_to_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    candidate: Phase1SelfLabelCandidate | Any,
    score_mode: str,
    long_cut_hard_mask: bool,
) -> Dict:
    blocks = _collect_blocks(
        jobs=jobs,
        bay_ids=tuple(bay_ids),
        require_multi_objective=True,
        long_cut_hard_mask=long_cut_hard_mask,
    )
    assignments = []
    for block in blocks:
        assigned_bay = candidate.assignments.get(block.block_set_id)
        if assigned_bay is None:
            print(
                "[ERROR][Phase1.orchestrator._candidate_to_plan] "
                f"cause=missing_assignment block_set_id={block.block_set_id} source={candidate.source}"
            )
            raise RuntimeError(f"missing Phase 1 assignment for {block.block_set_id}")
        assignments.append(
            {
                "block_set_id": block.block_set_id,
                "project_no": block.project_no,
                "block_no": block.block_no,
                "assigned_bay": assigned_bay,
                "candidate_bays": "|".join(block.allowed_bay_ids),
                "job_ids": "|".join(block.job_ids),
                "wo_count": block.wo_count,
                "steel_quantity_sum": block.steel_quantity_sum,
                "cut_length_sum": round(block.cut_length_sum, 6),
                "bevel_quantity_sum": block.bevel_quantity_sum,
                "long_cut_over_1000": block.long_cut_over_1000,
            }
        )
    return {
        "phase": "phase1_block_to_bay",
        "algorithm": candidate.source,
        "score_mode": score_mode,
        "score": list(_score_bay_loads(candidate.bay_loads, score_mode)),
        "bay_capacity_weights": {
            str(bay_id): float(candidate.bay_loads[str(bay_id)]["capacity_weight"])
            for bay_id in bay_ids
        },
        "long_cut_hard_mask": bool(long_cut_hard_mask),
        "summary": {
            "job_count": len(jobs),
            "block_count": len(blocks),
            "assignment_count": len(assignments),
            "score": list(_score_bay_loads(candidate.bay_loads, score_mode)),
        },
        "bay_loads": candidate.bay_loads,
        "assignments": assignments,
    }


def candidate_to_phase1_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    candidate: Any,
    score_mode: str = "steel_first",
    long_cut_hard_mask: bool = True,
) -> Dict:
    """완성된 Phase 1 후보를 Phase 2가 읽는 공통 plan 계약으로 변환한다."""

    return _candidate_to_plan(
        jobs=jobs,
        bay_ids=bay_ids,
        candidate=candidate,
        score_mode=score_mode,
        long_cut_hard_mask=long_cut_hard_mask,
    )
