"""Phase 1 block-to-Bay workload balancing.

Phase 1 does not schedule machines and does not create W/O batches.
It solves only this first-stage planning question:

    "Which cutting Bay should each block set go to?"

The result is intentionally shaped so Phase 2 can consume it later:
every W/O in a block set can receive the selected Bay as `allowed_bay_ids`
or fixed `cut_bay` before machine-level batch scheduling starts.
"""

from __future__ import annotations

import csv
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


CANONICAL_PHASE1_HEURISTIC = "steel_lpt_greedy_insertion"
MULTI_OBJECTIVE_PHASE1_HEURISTIC = "multi_objective_balanced"
PRIORITY_SWEEP_PHASE1_HEURISTIC = "priority_sweep_balanced"
PRIORITY_GREEDY_PHASE1_HEURISTIC = "priority_greedy_insertion"
LONG_CUT_PREFERRED_PHASE1_HEURISTIC = "long_cut_preferred_balanced"
MULTI_OBJECTIVE_BEAM_WIDTH = 512
SUPPORTED_PHASE1_ALGORITHMS = {
    CANONICAL_PHASE1_HEURISTIC,
    MULTI_OBJECTIVE_PHASE1_HEURISTIC,
    PRIORITY_SWEEP_PHASE1_HEURISTIC,
    PRIORITY_GREEDY_PHASE1_HEURISTIC,
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    "lpt_steel_quantity",
    "heuristic",
}


@dataclass(frozen=True)
class Phase1Block:
    """Aggregated input unit for Phase 1.

    A block is identified by `block_set_id`, usually `호선번호::블록명`.
    `steel_quantity_sum` is the primary workload used for Bay balancing.
    `wo_count` is reported as a secondary workload metric.
    """

    block_set_id: str
    project_no: str
    block_no: str
    job_ids: Tuple[str, ...]
    wo_count: int
    steel_quantity_sum: int
    cut_length_sum: float
    bevel_quantity_sum: int
    long_cut_over_1000: int
    allowed_bay_ids: Tuple[str, ...]
    length_avg: float | None = None
    thickness_avg: float | None = None


def build_phase1_bay_plan(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str = CANONICAL_PHASE1_HEURISTIC,
) -> Dict:
    """Assign block sets to cutting Bays using a deterministic LPT heuristic.

    Inputs:
    - `jobs`: environment `Job` mapping or Job-like objects.
    - `bay_ids`: available cutting Bay IDs. Example: `["22", "23", "24", "25"]`.
    - `algorithm`: canonical `steel_lpt_greedy_insertion` or legacy aliases.

    Output:
    - JSON-serializable dict containing assignment rows and Bay load summary.
    """

    normalized_algorithm = _normalize_algorithm(algorithm)
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        require_multi_objective=normalized_algorithm
        in {
            MULTI_OBJECTIVE_PHASE1_HEURISTIC,
            PRIORITY_SWEEP_PHASE1_HEURISTIC,
            PRIORITY_GREEDY_PHASE1_HEURISTIC,
            LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
        },
    )
    if normalized_algorithm == MULTI_OBJECTIVE_PHASE1_HEURISTIC:
        assignments, bay_loads = _assign_blocks_multi_objective(
            blocks=blocks,
            bay_ids=normalized_bay_ids,
            score_mode="steel_first",
        )
    elif normalized_algorithm == PRIORITY_SWEEP_PHASE1_HEURISTIC:
        assignments, bay_loads = _assign_blocks_multi_objective(
            blocks=blocks,
            bay_ids=normalized_bay_ids,
            score_mode="steel_first",
        )
    elif normalized_algorithm == LONG_CUT_PREFERRED_PHASE1_HEURISTIC:
        assignments, bay_loads = _assign_blocks_multi_objective(
            blocks=blocks,
            bay_ids=normalized_bay_ids,
            score_mode="steel_first",
        )
    elif normalized_algorithm == PRIORITY_GREEDY_PHASE1_HEURISTIC:
        assignments, bay_loads = _assign_blocks_priority_greedy(blocks=blocks, bay_ids=normalized_bay_ids)
    else:
        assignments, bay_loads = _assign_blocks_lpt(blocks=blocks, bay_ids=normalized_bay_ids)
    summary = _build_summary(
        assignments=assignments,
        bay_loads=bay_loads,
        job_count=len(jobs),
        block_count=len(blocks),
        algorithm=normalized_algorithm,
    )
    return {
        "phase": "phase1_block_to_bay",
        "algorithm": normalized_algorithm,
        "summary": summary,
        "bay_loads": bay_loads,
        "assignments": assignments,
    }


def apply_phase1_plan_to_scenario(
    scenario: Mapping,
    plan: Mapping,
    assignment_mode: str = "cut_bay",
) -> Dict:
    """Apply Phase 1 block-to-Bay assignments to a Phase 2 scenario.

    `assignment_mode="cut_bay"` writes the selected Bay into every W/O's
    `cut_bay`, so existing machine-Bay consistency constraints enforce it.
    `assignment_mode="allowed_bay_ids"` writes `[assigned_bay]` and is useful
    when Phase 2 wants to keep the original `cut_bay` as reference data.
    """

    if assignment_mode not in {"cut_bay", "allowed_bay_ids"}:
        print(
            "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
            f"cause=unknown_assignment_mode assignment_mode={assignment_mode}"
        )
        raise ValueError(f"unknown Phase 1 assignment mode: {assignment_mode}")

    assignment_by_block = _phase1_assignment_map(plan)
    phase2_scenario = copy.deepcopy(dict(scenario))
    jobs = phase2_scenario.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        print(
            "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
            f"cause=no_scenario_jobs jobs_type={type(jobs).__name__}"
        )
        raise RuntimeError("Phase 1 apply requires a scenario with non-empty jobs")

    assigned_job_count = 0
    assigned_block_ids: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            print(
                "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
                f"cause=invalid_job_row index={index} row_type={type(job).__name__}"
            )
            raise RuntimeError(f"invalid job row at index={index}")
        job_id = str(job.get("job_id") or job.get("id") or f"index_{index}")
        block_set_id = _require_text(
            job.get("block_set_id") or job.get("block_set_key"),
            "block_set_id",
            job_id,
        )
        assigned_bay = assignment_by_block.get(block_set_id)
        if assigned_bay is None:
            print(
                "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
                f"cause=missing_phase1_assignment job_id={job_id} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"missing_phase1_assignment: {block_set_id}")

        if assignment_mode == "cut_bay":
            job["cut_bay"] = assigned_bay
        else:
            job["allowed_bay_ids"] = [assigned_bay]
        assigned_job_count += 1
        assigned_block_ids.add(block_set_id)

    metadata = phase2_scenario.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        print(
            "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
            f"cause=invalid_metadata metadata_type={type(metadata).__name__}"
        )
        raise RuntimeError("scenario metadata must be a mapping")
    metadata["phase1_applied"] = True
    metadata["phase1_assignment_mode"] = assignment_mode
    metadata["phase1_plan_algorithm"] = str(plan.get("algorithm") or "")
    metadata["phase1_assigned_block_count"] = len(assigned_block_ids)
    metadata["phase1_assigned_job_count"] = assigned_job_count

    return {
        "scenario": phase2_scenario,
        "summary": {
            "assignment_mode": assignment_mode,
            "plan_algorithm": str(plan.get("algorithm") or ""),
            "assigned_job_count": assigned_job_count,
            "assigned_block_count": len(assigned_block_ids),
            "plan_assignment_count": len(assignment_by_block),
        },
    }


def write_phase1_bay_plan(plan: Mapping, output_dir: str | Path) -> Dict[str, str]:
    """Write Phase 1 result JSON and CSV files.

    Files:
    - `phase1_block_bay_plan.json`: full machine-readable result.
    - `phase1_block_assignments.csv`: one row per block set.
    - `phase1_bay_loads.csv`: one row per Bay.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "phase1_block_bay_plan.json"
    assignments_path = output_path / "phase1_block_assignments.csv"
    bay_loads_path = output_path / "phase1_bay_loads.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(plan, file, ensure_ascii=False, indent=2)

    assignment_rows = list(plan.get("assignments", []))
    assignment_fields = [
        "block_set_id",
        "project_no",
        "block_no",
        "assigned_bay",
        "wo_count",
        "steel_quantity_sum",
        "cut_length_sum",
        "bevel_quantity_sum",
        "long_cut_over_1000",
        "candidate_bays",
        "job_ids",
        "bay_steel_quantity_before",
        "bay_steel_quantity_after",
    ]
    _write_csv(assignments_path, assignment_fields, assignment_rows)

    bay_rows = [
        {
            "bay_id": bay_id,
            **loads,
        }
        for bay_id, loads in sorted(plan.get("bay_loads", {}).items())
    ]
    bay_fields = [
        "bay_id",
        "steel_quantity_sum",
        "cut_length_sum",
        "bevel_quantity_sum",
        "long_cut_bay24_count",
        "wo_count",
        "block_count",
    ]
    _write_csv(bay_loads_path, bay_fields, bay_rows)

    return {
        "json": str(json_path),
        "assignments_csv": str(assignments_path),
        "bay_loads_csv": str(bay_loads_path),
    }


def _normalize_algorithm(algorithm: str) -> str:
    """Validate the Phase 1 algorithm name."""

    if algorithm not in SUPPORTED_PHASE1_ALGORITHMS:
        print(
            "[ERROR][phase1_bay_balancer._normalize_algorithm] "
            f"cause=unknown_phase1_algorithm algorithm={algorithm}"
        )
        raise ValueError(f"unknown Phase 1 algorithm: {algorithm}")
    if algorithm in {"heuristic", "lpt_steel_quantity"}:
        return CANONICAL_PHASE1_HEURISTIC
    return algorithm


def _phase1_assignment_map(plan: Mapping) -> Dict[str, str]:
    """Return `block_set_id -> assigned_bay` from a Phase 1 plan."""

    assignment_rows = plan.get("assignments")
    if not isinstance(assignment_rows, list) or not assignment_rows:
        print(
            "[ERROR][phase1_bay_balancer._phase1_assignment_map] "
            f"cause=no_phase1_assignments rows_type={type(assignment_rows).__name__}"
        )
        raise RuntimeError("Phase 1 plan requires non-empty assignments")

    assignment_by_block: Dict[str, str] = {}
    for index, row in enumerate(assignment_rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_bay_balancer._phase1_assignment_map] "
                f"cause=invalid_assignment_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid Phase 1 assignment row at index={index}")
        block_set_id = _require_text(row.get("block_set_id"), "block_set_id", f"assignment_{index}")
        assigned_bay = _require_text(row.get("assigned_bay"), "assigned_bay", block_set_id)
        if block_set_id in assignment_by_block:
            print(
                "[ERROR][phase1_bay_balancer._phase1_assignment_map] "
                f"cause=duplicate_block_assignment block_set_id={block_set_id}"
            )
            raise RuntimeError(f"duplicate Phase 1 assignment: {block_set_id}")
        assignment_by_block[block_set_id] = assigned_bay
    return assignment_by_block


def _normalize_bay_ids(bay_ids: Sequence[str]) -> Tuple[str, ...]:
    """Return deterministic non-empty Bay ID tuple."""

    normalized = tuple(sorted({str(bay_id).strip() for bay_id in bay_ids if str(bay_id).strip()}))
    if not normalized:
        print("[ERROR][phase1_bay_balancer._normalize_bay_ids] cause=no_bay_ids")
        raise RuntimeError("Phase 1 requires at least one Bay ID")
    return normalized


def _collect_blocks(
    jobs: Mapping[str, object],
    bay_ids: Tuple[str, ...],
    require_multi_objective: bool,
    long_cut_hard_mask: bool = True,
) -> List[Phase1Block]:
    """Group W/O jobs by block set and validate Phase 1 required fields."""

    if not jobs:
        print("[ERROR][phase1_bay_balancer._collect_blocks] cause=no_jobs")
        raise RuntimeError("Phase 1 requires at least one job")

    grouped: Dict[str, List[object]] = {}
    for job_key, job in jobs.items():
        block_set_id = _require_text(_job_attr(job, "block_set_id"), "block_set_id", str(job_key))
        grouped.setdefault(block_set_id, []).append(job)

    blocks: List[Phase1Block] = []
    for block_set_id, block_jobs in sorted(grouped.items()):
        job_ids = tuple(_require_text(_job_attr(job, "job_id"), "job_id", block_set_id) for job in block_jobs)
        steel_quantity_sum = 0
        cut_length_sum = 0.0
        bevel_quantity_sum = 0
        length_values: List[float] = []
        thickness_values: List[float] = []
        for job in block_jobs:
            job_id = _require_text(_job_attr(job, "job_id"), "job_id", block_set_id)
            steel_quantity_sum += _require_positive_int(
                _job_attr(job, "steel_quantity"),
                "steel_quantity",
                job_id,
            )
            cut_length_value = _job_attr(job, "cut_length")
            if require_multi_objective or cut_length_value not in (None, ""):
                cut_length_sum += _require_non_negative_float(
                    cut_length_value,
                    "cut_length",
                    job_id,
                )
            bevel_quantity_value = _job_attr(job, "bevel_quantity")
            if require_multi_objective or bevel_quantity_value not in (None, ""):
                bevel_quantity_sum += _require_non_negative_int(
                    bevel_quantity_value,
                    "bevel_quantity",
                    job_id,
                )
            length_value = _job_attr(job, "plate_length")
            if length_value not in (None, ""):
                length_values.append(_require_non_negative_float(length_value, "plate_length", job_id))
            thickness_value = _job_attr(job, "thickness")
            if thickness_value not in (None, ""):
                thickness_values.append(_require_non_negative_float(thickness_value, "thickness", job_id))
        long_cut_over_1000 = 1 if cut_length_sum > 1000 else 0
        allowed_bay_ids = _allowed_bays_for_block(block_jobs, bay_ids, block_set_id)
        if long_cut_hard_mask:
            allowed_bay_ids = _apply_long_cut_hard_mask(
                allowed_bay_ids=allowed_bay_ids,
                long_cut_over_1000=long_cut_over_1000,
                block_set_id=block_set_id,
            )
        project_no, block_no = _project_block_labels(block_set_id, block_jobs)
        blocks.append(
            Phase1Block(
                block_set_id=block_set_id,
                project_no=project_no,
                block_no=block_no,
                job_ids=job_ids,
                wo_count=len(block_jobs),
                steel_quantity_sum=steel_quantity_sum,
                cut_length_sum=cut_length_sum,
                bevel_quantity_sum=bevel_quantity_sum,
                long_cut_over_1000=long_cut_over_1000,
                allowed_bay_ids=allowed_bay_ids,
                length_avg=None if not length_values else sum(length_values) / len(length_values),
                thickness_avg=None if not thickness_values else sum(thickness_values) / len(thickness_values),
            )
        )

    return blocks


def _assign_blocks_lpt(blocks: Iterable[Phase1Block], bay_ids: Tuple[str, ...]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Largest-processing-time assignment using steel quantity as load."""

    bay_loads = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    assignments: List[Dict] = []
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (-block.steel_quantity_sum, -block.wo_count, block.block_set_id),
    )

    for block in sorted_blocks:
        best_bay = min(
            block.allowed_bay_ids,
            key=lambda bay_id: _projected_score(bay_loads, bay_id, block),
        )
        load_before = int(bay_loads[best_bay]["steel_quantity_sum"])
        _add_block_load(bay_loads[best_bay], best_bay, block)
        assignments.append(
            {
                "block_set_id": block.block_set_id,
                "project_no": block.project_no,
                "block_no": block.block_no,
                "assigned_bay": best_bay,
                "wo_count": block.wo_count,
                "steel_quantity_sum": block.steel_quantity_sum,
                "cut_length_sum": round(block.cut_length_sum, 6),
                "bevel_quantity_sum": block.bevel_quantity_sum,
                "long_cut_over_1000": block.long_cut_over_1000,
                "candidate_bays": "|".join(block.allowed_bay_ids),
                "job_ids": "|".join(block.job_ids),
                "bay_steel_quantity_before": load_before,
                "bay_steel_quantity_after": int(bay_loads[best_bay]["steel_quantity_sum"]),
            }
        )

    return assignments, bay_loads


def _assignment_rows_from_mapping(
    blocks: Sequence[Phase1Block],
    assignment_by_block: Mapping[str, str],
    bay_ids: Tuple[str, ...],
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Build report rows and final Bay loads from fixed block assignments."""

    bay_loads = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    assignments: List[Dict] = []
    for block in blocks:
        best_bay = assignment_by_block[block.block_set_id]
        load_before = int(bay_loads[best_bay]["steel_quantity_sum"])
        _add_block_load(bay_loads[best_bay], best_bay, block)
        assignments.append(
            {
                "block_set_id": block.block_set_id,
                "project_no": block.project_no,
                "block_no": block.block_no,
                "assigned_bay": best_bay,
                "wo_count": block.wo_count,
                "steel_quantity_sum": block.steel_quantity_sum,
                "cut_length_sum": round(block.cut_length_sum, 6),
                "bevel_quantity_sum": block.bevel_quantity_sum,
                "long_cut_over_1000": block.long_cut_over_1000,
                "candidate_bays": "|".join(block.allowed_bay_ids),
                "job_ids": "|".join(block.job_ids),
                "bay_steel_quantity_before": load_before,
                "bay_steel_quantity_after": int(bay_loads[best_bay]["steel_quantity_sum"]),
            }
        )

    return assignments, bay_loads


def _assign_blocks_priority_greedy(
    blocks: Iterable[Phase1Block],
    bay_ids: Tuple[str, ...],
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Assign each block immediately using the presentation-friendly rule."""

    bay_loads = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    assignments: List[Dict] = []
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            -block.long_cut_over_1000,
            -block.steel_quantity_sum,
            -block.cut_length_sum,
            -block.bevel_quantity_sum,
            block.block_set_id,
        ),
    )

    for block in sorted_blocks:
        best_bay = min(
            block.allowed_bay_ids,
            key=lambda bay_id: _priority_greedy_projected_score(bay_loads, bay_id, block),
        )
        load_before = int(bay_loads[best_bay]["steel_quantity_sum"])
        _add_block_load(bay_loads[best_bay], best_bay, block)
        assignments.append(
            {
                "block_set_id": block.block_set_id,
                "project_no": block.project_no,
                "block_no": block.block_no,
                "assigned_bay": best_bay,
                "wo_count": block.wo_count,
                "steel_quantity_sum": block.steel_quantity_sum,
                "cut_length_sum": round(block.cut_length_sum, 6),
                "bevel_quantity_sum": block.bevel_quantity_sum,
                "long_cut_over_1000": block.long_cut_over_1000,
                "candidate_bays": "|".join(block.allowed_bay_ids),
                "job_ids": "|".join(block.job_ids),
                "bay_steel_quantity_before": load_before,
                "bay_steel_quantity_after": int(bay_loads[best_bay]["steel_quantity_sum"]),
            }
        )

    return assignments, bay_loads


def _improve_multi_objective_assignment(
    blocks: Sequence[Phase1Block],
    assignment_by_block: Mapping[str, str],
    bay_ids: Tuple[str, ...],
    score_mode: str = "steel_first",
) -> Dict[str, str]:
    """Improve final score by deterministic move and swap local search.

    # ponytail: capped three passes; use full metaheuristic only if Phase 1 block
    counts grow enough that move/swap local search is not sufficient.
    """

    current_assignment = dict(assignment_by_block)
    best_score = _multi_objective_assignment_score(blocks, current_assignment, bay_ids, score_mode=score_mode)
    for _ in range(3):
        improved = False
        for block in blocks:
            current_bay = current_assignment[block.block_set_id]
            for candidate_bay in block.allowed_bay_ids:
                if candidate_bay == current_bay:
                    continue
                trial_assignment = dict(current_assignment)
                trial_assignment[block.block_set_id] = candidate_bay
                trial_score = _multi_objective_assignment_score(
                    blocks,
                    trial_assignment,
                    bay_ids,
                    score_mode=score_mode,
                )
                if trial_score < best_score:
                    current_assignment = trial_assignment
                    best_score = trial_score
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        for left_index, left_block in enumerate(blocks):
            left_current_bay = current_assignment[left_block.block_set_id]
            for right_block in blocks[left_index + 1 :]:
                right_current_bay = current_assignment[right_block.block_set_id]
                if left_current_bay == right_current_bay:
                    continue
                if right_current_bay not in left_block.allowed_bay_ids:
                    continue
                if left_current_bay not in right_block.allowed_bay_ids:
                    continue
                trial_assignment = dict(current_assignment)
                trial_assignment[left_block.block_set_id] = right_current_bay
                trial_assignment[right_block.block_set_id] = left_current_bay
                trial_score = _multi_objective_assignment_score(
                    blocks,
                    trial_assignment,
                    bay_ids,
                    score_mode=score_mode,
                )
                if trial_score < best_score:
                    current_assignment = trial_assignment
                    best_score = trial_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current_assignment


def _multi_objective_assignment_score(
    blocks: Sequence[Phase1Block],
    assignment_by_block: Mapping[str, str],
    bay_ids: Tuple[str, ...],
    score_mode: str = "steel_first",
) -> Tuple:
    """Score a complete Phase 1 assignment with the confirmed objective order."""

    bay_loads = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    for block in blocks:
        assigned_bay = assignment_by_block[block.block_set_id]
        _add_block_load(bay_loads[assigned_bay], assigned_bay, block)
    return _multi_objective_load_score(bay_loads, score_mode=score_mode)


def _multi_objective_load_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    score_mode: str = "steel_first",
) -> Tuple:
    """Score current Bay loads with the single confirmed Phase 1 objective order.

    The tuple contract is intentionally fixed:
    `(steel_gap, cut_gap, bevel_gap, long_cut_bay24_count)`.

    Long-cut Bay 24 avoidance is enforced earlier by candidate hard masking.
    It remains in the score only as an audit/count field, not as the first
    lexicographic objective.
    """

    steel_values = [row["steel_quantity_sum"] for row in bay_loads.values()]
    cut_values = [row["cut_length_sum"] for row in bay_loads.values()]
    bevel_values = [row["bevel_quantity_sum"] for row in bay_loads.values()]
    long_cut_bay24_count = sum(int(row["long_cut_bay24_count"]) for row in bay_loads.values())
    if score_mode != "steel_first":
        print(
            "[ERROR][phase1_bay_balancer._multi_objective_load_score] "
            f"cause=unknown_score_mode score_mode={score_mode}"
        )
        raise RuntimeError(f"unknown_score_mode: {score_mode}")
    return (
        _round_score(_gap(steel_values)),
        _round_score(_gap(cut_values)),
        _round_score(_gap(bevel_values)),
        long_cut_bay24_count,
    )


def _priority_sweep_load_score(bay_loads: Mapping[str, Mapping[str, int | float]]) -> Tuple:
    """Backward-compatible wrapper for the fixed Phase 1 score tuple."""

    return _multi_objective_load_score(bay_loads, score_mode="steel_first")


def _assign_blocks_multi_objective(
    blocks: Iterable[Phase1Block],
    bay_ids: Tuple[str, ...],
    score_mode: str = "steel_first",
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Multi-start beam search with the confirmed Phase 1 objective priority.

    Priority is lexicographic, not weighted:
    1. steel quantity balance;
    2. cutting length balance;
    3. BV_QTY balance;
    4. fewer long-cut blocks assigned to Bay 24.
    """

    block_list = list(blocks)
    best_assignment: Dict[str, str] | None = None
    best_score: Tuple | None = None
    for ordered_blocks in _multi_objective_block_orders(block_list):
        assignment = _beam_search_multi_objective_assignment(ordered_blocks, bay_ids, score_mode=score_mode)
        improved_assignment = _improve_multi_objective_assignment(
            blocks=block_list,
            assignment_by_block=assignment,
            bay_ids=bay_ids,
            score_mode=score_mode,
        )
        score = _multi_objective_assignment_score(
            block_list,
            improved_assignment,
            bay_ids,
            score_mode=score_mode,
        )
        tie_breaker = tuple((block.block_set_id, improved_assignment[block.block_set_id]) for block in block_list)
        comparable_score = score + (tie_breaker,)
        if best_score is None or comparable_score < best_score:
            best_assignment = improved_assignment
            best_score = comparable_score
    if best_assignment is None:
        print("[ERROR][phase1_bay_balancer._assign_blocks_multi_objective] cause=no_assignment")
        raise RuntimeError("multi-objective Phase 1 failed to build an assignment")
    return _assignment_rows_from_mapping(
        blocks=sorted(block_list, key=lambda block: block.block_set_id),
        assignment_by_block=best_assignment,
        bay_ids=bay_ids,
    )


def _multi_objective_block_orders(blocks: Sequence[Phase1Block]) -> List[Tuple[Phase1Block, ...]]:
    """Return deterministic block orders for multi-start search."""

    if not blocks:
        return []
    max_steel = max(block.steel_quantity_sum for block in blocks) or 1
    max_cut = max(block.cut_length_sum for block in blocks) or 1.0
    max_bevel = max(block.bevel_quantity_sum for block in blocks) or 1

    def combined_load(block: Phase1Block) -> float:
        return (
            block.steel_quantity_sum / max_steel
            + block.cut_length_sum / max_cut
            + block.bevel_quantity_sum / max_bevel
            + block.long_cut_over_1000
        )

    order_specs = [
        lambda block: (
            -block.steel_quantity_sum,
            -block.cut_length_sum,
            -block.bevel_quantity_sum,
            -block.long_cut_over_1000,
            block.block_set_id,
        ),
        lambda block: (
            -block.cut_length_sum,
            -block.steel_quantity_sum,
            -block.bevel_quantity_sum,
            -block.long_cut_over_1000,
            block.block_set_id,
        ),
        lambda block: (
            -block.bevel_quantity_sum,
            -block.steel_quantity_sum,
            -block.cut_length_sum,
            -block.long_cut_over_1000,
            block.block_set_id,
        ),
        lambda block: (
            -block.long_cut_over_1000,
            -block.cut_length_sum,
            -block.steel_quantity_sum,
            -block.bevel_quantity_sum,
            block.block_set_id,
        ),
        lambda block: (
            -combined_load(block),
            -block.steel_quantity_sum,
            -block.cut_length_sum,
            -block.bevel_quantity_sum,
            block.block_set_id,
        ),
    ]
    seen: set[Tuple[str, ...]] = set()
    orders: List[Tuple[Phase1Block, ...]] = []
    for order_key in order_specs:
        ordered = tuple(sorted(blocks, key=order_key))
        signature = tuple(block.block_set_id for block in ordered)
        if signature in seen:
            continue
        seen.add(signature)
        orders.append(ordered)
    return orders


def _beam_search_multi_objective_assignment(
    blocks: Sequence[Phase1Block],
    bay_ids: Tuple[str, ...],
    score_mode: str = "steel_first",
) -> Dict[str, str]:
    """Search block-to-Bay assignments with bounded multi-objective beam search."""

    initial_loads = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "wo_count": 0,
            "block_count": 0,
        }
        for bay_id in bay_ids
    }
    states: List[Tuple[Tuple, Tuple[str, ...], Dict[str, Dict]]] = [
        (_multi_objective_load_score(initial_loads, score_mode=score_mode), tuple(), initial_loads)
    ]
    for block in blocks:
        expanded: List[Tuple[Tuple, Tuple[str, ...], Dict[str, Dict]]] = []
        for _, assigned_bays, loads in states:
            for candidate_bay in block.allowed_bay_ids:
                next_loads = {bay_id: dict(row) for bay_id, row in loads.items()}
                _add_block_load(next_loads[candidate_bay], candidate_bay, block)
                next_assigned_bays = assigned_bays + (candidate_bay,)
                expanded.append(
                    (
                        _multi_objective_load_score(next_loads, score_mode=score_mode),
                        next_assigned_bays,
                        next_loads,
                    )
                )
        expanded.sort(key=lambda state: (state[0], state[1]))
        states = expanded[:MULTI_OBJECTIVE_BEAM_WIDTH]

    best_assigned_bays = states[0][1]
    return {
        block.block_set_id: assigned_bay
        for block, assigned_bay in zip(blocks, best_assigned_bays)
    }


def _add_block_load(loads: Dict, bay_id: str, block: Phase1Block) -> None:
    """Mutate one Bay load row by adding one block."""

    loads["steel_quantity_sum"] += block.steel_quantity_sum
    loads["cut_length_sum"] += block.cut_length_sum
    loads["bevel_quantity_sum"] += block.bevel_quantity_sum
    loads["long_cut_bay24_count"] += 1 if block.long_cut_over_1000 and str(bay_id) == "24" else 0
    loads["wo_count"] += block.wo_count
    loads["block_count"] += 1


def _projected_score(bay_loads: Mapping[str, Mapping[str, int]], bay_id: str, block: Phase1Block) -> Tuple:
    """Score a candidate Bay after assigning one block.

    Lower is better. The primary target is Bay steel quantity balance.
    W/O count and Bay ID are deterministic tie-breakers.
    """

    projected_steel = {
        current_bay_id: int(loads["steel_quantity_sum"])
        + (block.steel_quantity_sum if current_bay_id == bay_id else 0)
        for current_bay_id, loads in bay_loads.items()
    }
    projected_wo = {
        current_bay_id: int(loads["wo_count"]) + (block.wo_count if current_bay_id == bay_id else 0)
        for current_bay_id, loads in bay_loads.items()
    }
    steel_values = list(projected_steel.values())
    wo_values = list(projected_wo.values())
    steel_mean = sum(steel_values) / max(len(steel_values), 1)
    wo_mean = sum(wo_values) / max(len(wo_values), 1)
    return (
        max(steel_values) - min(steel_values),
        _round_score(sum(abs(value - steel_mean) for value in steel_values)),
        max(wo_values) - min(wo_values),
        _round_score(sum(abs(value - wo_mean) for value in wo_values)),
        bay_id,
    )


def _priority_greedy_projected_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    bay_id: str,
    block: Phase1Block,
) -> Tuple:
    """Score one immediate greedy insertion candidate."""

    projected = {current_bay_id: dict(loads) for current_bay_id, loads in bay_loads.items()}
    _add_block_load(projected[bay_id], bay_id, block)
    steel_values = [row["steel_quantity_sum"] for row in projected.values()]
    cut_values = [row["cut_length_sum"] for row in projected.values()]
    bevel_values = [row["bevel_quantity_sum"] for row in projected.values()]
    return (
        _round_score(_gap(steel_values)),
        _round_score(_gap(cut_values)),
        _round_score(_gap(bevel_values)),
        sum(int(row["long_cut_bay24_count"]) for row in projected.values()),
        bay_id,
    )


def _build_summary(
    assignments: Sequence[Mapping],
    bay_loads: Mapping[str, Mapping[str, int]],
    job_count: int,
    block_count: int,
    algorithm: str,
) -> Dict:
    """Build top-level numeric summary for CLI/report output."""

    steel_values = [int(row["steel_quantity_sum"]) for row in bay_loads.values()]
    cut_values = [float(row.get("cut_length_sum", 0.0)) for row in bay_loads.values()]
    bevel_values = [int(row.get("bevel_quantity_sum", 0)) for row in bay_loads.values()]
    wo_values = [int(row["wo_count"]) for row in bay_loads.values()]
    block_values = [int(row["block_count"]) for row in bay_loads.values()]
    if algorithm == MULTI_OBJECTIVE_PHASE1_HEURISTIC:
        load_metric = "multi_objective_lexicographic"
        objective_priority = [
            "steel_quantity_sum",
            "cut_length_sum",
            "bevel_quantity_sum",
            "long_cut_over_1000_prefer_bay22_23",
        ]
    elif algorithm == PRIORITY_SWEEP_PHASE1_HEURISTIC:
        load_metric = "multi_objective_lexicographic"
        objective_priority = [
            "steel_quantity_sum",
            "cut_length_sum",
            "bevel_quantity_sum",
            "long_cut_over_1000_prefer_bay22_23",
        ]
    elif algorithm == LONG_CUT_PREFERRED_PHASE1_HEURISTIC:
        load_metric = "multi_objective_lexicographic"
        objective_priority = [
            "steel_quantity_sum",
            "cut_length_sum",
            "bevel_quantity_sum",
            "long_cut_over_1000_prefer_bay22_23",
        ]
    elif algorithm == PRIORITY_GREEDY_PHASE1_HEURISTIC:
        load_metric = "priority_greedy_lexicographic"
        objective_priority = [
            "steel_quantity_sum",
            "cut_length_sum",
            "bevel_quantity_sum",
            "long_cut_over_1000_prefer_bay22_23",
        ]
    else:
        load_metric = "steel_quantity_sum"
        objective_priority = ["steel_quantity_sum"]
    return {
        "algorithm": algorithm,
        "load_metric": load_metric,
        "objective_priority": objective_priority,
        "block_selection_priority": [
            "long_cut_over_1000_first",
            "steel_quantity_sum_desc",
            "cut_length_sum_desc",
            "bevel_quantity_sum_desc",
        ]
        if algorithm == PRIORITY_GREEDY_PHASE1_HEURISTIC
        else [],
        "secondary_load_metric": "wo_count",
        "job_count": int(job_count),
        "block_count": int(block_count),
        "assigned_block_count": int(len(assignments)),
        "bay_count": int(len(bay_loads)),
        "steel_quantity_total": int(sum(steel_values)),
        "steel_quantity_gap": int(max(steel_values) - min(steel_values)) if steel_values else 0,
        "cut_length_total": round(sum(cut_values), 6),
        "cut_length_gap": round(max(cut_values) - min(cut_values), 6) if cut_values else 0.0,
        "bevel_quantity_total": int(sum(bevel_values)),
        "bevel_quantity_gap": int(max(bevel_values) - min(bevel_values)) if bevel_values else 0,
        "long_cut_bay24_count": int(
            sum(int(row.get("long_cut_bay24_count", 0)) for row in bay_loads.values())
        ),
        "wo_count_total": int(sum(wo_values)),
        "wo_count_gap": int(max(wo_values) - min(wo_values)) if wo_values else 0,
        "block_count_gap": int(max(block_values) - min(block_values)) if block_values else 0,
    }


def _allowed_bays_for_block(
    block_jobs: Sequence[object],
    bay_ids: Tuple[str, ...],
    block_set_id: str,
) -> Tuple[str, ...]:
    """Intersect per-job Bay restrictions for one block."""

    allowed = set(bay_ids)
    saw_restriction = False
    for job in block_jobs:
        cut_bay = _job_attr(job, "cut_bay")
        if cut_bay not in (None, ""):
            saw_restriction = True
            allowed &= {str(cut_bay)}
        job_allowed = tuple(str(bay_id) for bay_id in (_job_attr(job, "allowed_bay_ids") or ()))
        if job_allowed:
            saw_restriction = True
            allowed &= set(job_allowed)

    if not allowed:
        print(
            "[ERROR][phase1_bay_balancer._allowed_bays_for_block] "
            "cause=no_feasible_bay_for_block "
            f"block_set_id={block_set_id} saw_restriction={saw_restriction}"
        )
        raise RuntimeError(f"no feasible Bay for block_set_id={block_set_id}")
    return tuple(bay_id for bay_id in bay_ids if bay_id in allowed)


def _apply_long_cut_hard_mask(
    allowed_bay_ids: Tuple[str, ...],
    long_cut_over_1000: int,
    block_set_id: str,
) -> Tuple[str, ...]:
    """Remove Bay 24 from long-cut block candidates.

    Phase 1 현업 룰은 `CUT_LTH > 1000` block을 Bay 22/23에 우선 배정하는
    선호에서, 현재 Proposed 실험 기준 하드마스크로 승격한다. 따라서 generated
    planning 경로에서는 장척 block의 Bay 24 action을 만들지 않는다.
    """

    if not long_cut_over_1000:
        return allowed_bay_ids
    masked = tuple(bay_id for bay_id in allowed_bay_ids if str(bay_id) != "24")
    if not masked:
        print(
            "[ERROR][phase1_bay_balancer._apply_long_cut_hard_mask] "
            f"cause=no_feasible_bay_after_long_cut_mask block_set_id={block_set_id} "
            f"allowed_bay_ids={allowed_bay_ids}"
        )
        raise RuntimeError(f"no feasible Bay after long-cut hard mask: {block_set_id}")
    return masked


def _project_block_labels(block_set_id: str, block_jobs: Sequence[object]) -> Tuple[str, str]:
    """Return source project/block labels for reports."""

    first_job = block_jobs[0]
    extra = _job_attr(first_job, "extra") or {}
    project_no = str(extra.get("source_project_no") or "")
    block_no = str(extra.get("source_block_no") or "")
    if not project_no or not block_no:
        if "::" in block_set_id:
            project_no, block_no = block_set_id.split("::", 1)
    return project_no, block_no


def _job_attr(job: object, name: str):
    """Read a field from a dataclass-like object or dict."""

    if isinstance(job, Mapping):
        return job.get(name)
    return getattr(job, name, None)


def _require_text(value, field_name: str, row_key: str) -> str:
    """Require a non-empty text value."""

    text = "" if value is None else str(value).strip()
    if not text:
        print(
            "[ERROR][phase1_bay_balancer._require_text] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    return text


def _require_positive_int(value, field_name: str, row_key: str) -> int:
    """Require a positive integer numeric field without fallback."""

    if value in (None, ""):
        print(
            "[ERROR][phase1_bay_balancer._require_positive_int] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_positive_int] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid_{field_name}: {row_key}") from exc
    if parsed <= 0:
        print(
            "[ERROR][phase1_bay_balancer._require_positive_int] "
            f"cause=non_positive_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_positive_{field_name}: {row_key}")
    return parsed


def _require_non_negative_int(value, field_name: str, row_key: str) -> int:
    """Require a zero-or-positive integer numeric field without fallback."""

    if value in (None, ""):
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_int] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_int] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid_{field_name}: {row_key}") from exc
    if parsed < 0:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_int] "
            f"cause=negative_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative_{field_name}: {row_key}")
    return parsed


def _require_non_negative_float(value, field_name: str, row_key: str) -> float:
    """Require a zero-or-positive float numeric field without fallback."""

    if value in (None, ""):
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_float] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing_{field_name}: {row_key}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_float] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid_{field_name}: {row_key}") from exc
    if parsed < 0:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_float] "
            f"cause=negative_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative_{field_name}: {row_key}")
    return parsed


def _gap(values: Sequence[int | float]) -> float:
    """Return max-min for a non-empty numeric sequence."""

    return max(values) - min(values) if values else 0


def _round_score(value: int | float) -> float:
    """Stabilize lexicographic ties that differ only by floating-point noise."""

    return round(float(value), 9)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write rows to CSV with a fixed header."""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
