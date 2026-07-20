"""MIXED Phase 1 block-series 집계, Bay 부하, plan 입출력 공통 로직."""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from Utils.phase1.multi_series_rules import (
    GROUP_BAY_CAPACITY_WEIGHTS,
    PHASE1_BALANCING_GROUP_ORDER,
    add_multi_series_group_load,
    apply_phase1_series_bay_mask,
    initialize_multi_series_group_loads,
    joint_phase1_bay_capacity_weights,
    multi_series_group_load_value,
)


@dataclass(frozen=True)
class Phase1Block:
    """`PROJ_NO+GYEL+BLK_NO`로 묶은 Phase 1 의사결정 단위."""

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
    length_avg: float
    thickness_avg: float
    family: str
    balancing_group: str
    width_max: float
    wide_plate_over_4500: int
    cnt_block: int
    bay_mask_reason_codes: Tuple[str, ...]


def apply_phase1_plan_to_scenario(
    scenario: Mapping,
    plan: Mapping,
    assignment_mode: str = "allowed_bay_ids",
) -> Dict:
    """Phase 1 block-series 배정을 Phase 2 시나리오의 모든 W/O에 적용한다."""

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

    assigned_blocks: set[str] = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            print(
                "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
                f"cause=invalid_job_row index={index} row_type={type(job).__name__}"
            )
            raise RuntimeError(f"invalid scenario job row: {index}")
        job_id = str(job.get("job_id") or job.get("id") or f"index_{index}")
        block_set_id = _require_text(
            job.get("block_set_id") or job.get("block_set_key"),
            "block_set_id",
            job_id,
        )
        if block_set_id not in assignment_by_block:
            print(
                "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
                f"cause=missing_phase1_assignment job_id={job_id} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"missing Phase 1 assignment: {block_set_id}")
        assigned_bay = assignment_by_block[block_set_id]
        if assignment_mode == "cut_bay":
            job["cut_bay"] = assigned_bay
        else:
            job["allowed_bay_ids"] = [assigned_bay]
        assigned_blocks.add(block_set_id)

    metadata = phase2_scenario.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        print(
            "[ERROR][phase1_bay_balancer.apply_phase1_plan_to_scenario] "
            f"cause=invalid_metadata metadata_type={type(metadata).__name__}"
        )
        raise RuntimeError("scenario metadata must be a mapping")
    metadata.update(
        {
            "phase1_applied": True,
            "phase1_assignment_mode": assignment_mode,
            "phase1_plan_algorithm": str(plan.get("algorithm") or ""),
            "phase1_assigned_block_count": len(assigned_blocks),
            "phase1_assigned_job_count": len(jobs),
        }
    )
    return {
        "scenario": phase2_scenario,
        "summary": {
            "assignment_mode": assignment_mode,
            "plan_algorithm": str(plan.get("algorithm") or ""),
            "assigned_job_count": len(jobs),
            "assigned_block_count": len(assigned_blocks),
            "plan_assignment_count": len(assignment_by_block),
        },
    }


def write_phase1_bay_plan(plan: Mapping, output_dir: str | Path) -> Dict[str, str]:
    """MIXED Phase 1 plan을 JSON과 block/Bay CSV로 저장한다."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "phase1_block_bay_plan.json"
    assignments_path = output_path / "phase1_block_assignments.csv"
    bay_loads_path = output_path / "phase1_bay_loads.csv"
    json_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assignment_fields = [
        "block_set_id",
        "project_no",
        "series",
        "block_no",
        "assigned_bay",
        "candidate_bays",
        "job_ids",
        "wo_count",
        "steel_quantity_sum",
        "cut_length_sum",
        "bevel_quantity_sum",
        "width_max",
        "long_cut_over_1000",
        "wide_plate_over_4500",
        "cnt_block",
        "bay_mask_reason_codes",
    ]
    _write_csv(assignments_path, assignment_fields, list(plan.get("assignments", [])))

    bay_rows: List[Dict[str, int | float | str]] = []
    for bay_id, loads in plan.get("bay_loads", {}).items():
        weight = _require_capacity_weight(loads, f"write_plan:{bay_id}")
        bay_rows.append(
            {
                "bay_id": str(bay_id),
                **dict(loads),
                "wo_count_per_capacity": _round_score(float(loads["wo_count"]) / weight),
                "cut_length_sum_per_capacity": _round_score(
                    float(loads["cut_length_sum"]) / weight
                ),
                "bevel_quantity_sum_per_capacity": _round_score(
                    float(loads["bevel_quantity_sum"]) / weight
                ),
            }
        )
    group_fields = [
        f"group_{group.lower()}_{metric}"
        for group in PHASE1_BALANCING_GROUP_ORDER
        for metric in ("wo_count", "cut_length_sum", "bevel_quantity_sum")
    ]
    bay_fields = [
        "bay_id",
        "capacity_weight",
        "wo_count",
        "wo_count_per_capacity",
        "cut_length_sum",
        "cut_length_sum_per_capacity",
        "bevel_quantity_sum",
        "bevel_quantity_sum_per_capacity",
        "steel_quantity_sum",
        "block_count",
        *group_fields,
    ]
    _write_csv(bay_loads_path, bay_fields, bay_rows)
    return {
        "json": str(json_path),
        "assignments_csv": str(assignments_path),
        "bay_loads_csv": str(bay_loads_path),
    }


def _validate_multi_series_plan_scope(
    blocks: Sequence[Phase1Block],
    bay_ids: Tuple[str, ...],
    capacity_weights: Mapping[str, float],
) -> None:
    """joint problem이 세 그룹과 정확한 다섯 Bay capacity를 사용하는지 검사한다."""

    groups = {block.balancing_group for block in blocks}
    if not groups or not groups <= set(PHASE1_BALANCING_GROUP_ORDER):
        print(
            "[ERROR][phase1_bay_balancer._validate_multi_series_plan_scope] "
            f"cause=invalid_balancing_groups groups={sorted(groups)}"
        )
        raise RuntimeError(f"invalid MIXED Phase 1 groups: {sorted(groups)}")
    expected = joint_phase1_bay_capacity_weights()
    if tuple(bay_ids) != tuple(expected):
        print(
            "[ERROR][phase1_bay_balancer._validate_multi_series_plan_scope] "
            f"cause=joint_bay_scope_mismatch expected={list(expected)} actual={list(bay_ids)}"
        )
        raise RuntimeError("MIXED Phase 1 requires the joint five-Bay scope")
    for bay_id, expected_weight in expected.items():
        actual_weight = float(capacity_weights.get(bay_id, math.nan))
        if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-9):
            print(
                "[ERROR][phase1_bay_balancer._validate_multi_series_plan_scope] "
                f"cause=capacity_weight_mismatch bay_id={bay_id} "
                f"expected={expected_weight} actual={actual_weight}"
            )
            raise RuntimeError(f"Phase 1 capacity weight mismatch: {bay_id}")


def _normalize_bay_ids(bay_ids: Sequence[str]) -> Tuple[str, ...]:
    """중복 없는 Bay ID를 입력 순서대로 반환한다."""

    normalized = tuple(str(value).strip() for value in bay_ids if str(value).strip())
    if not normalized or len(set(normalized)) != len(normalized):
        print(
            "[ERROR][phase1_bay_balancer._normalize_bay_ids] "
            f"cause=empty_or_duplicate_bay_ids bay_ids={normalized}"
        )
        raise RuntimeError("Phase 1 Bay IDs must be non-empty and unique")
    return normalized


def _normalize_bay_capacity_weights(
    bay_ids: Tuple[str, ...],
    bay_capacity_weights: Mapping[str, int | float] | None,
) -> Dict[str, float]:
    """다섯 Bay 설비 수를 누락 없이 숫자로 검증한다."""

    if bay_capacity_weights is None:
        print(
            "[ERROR][phase1_bay_balancer._normalize_bay_capacity_weights] "
            "cause=missing_capacity_weights"
        )
        raise RuntimeError("MIXED Phase 1 requires explicit Bay capacity weights")
    normalized = {str(key): float(value) for key, value in bay_capacity_weights.items()}
    if set(normalized) != set(bay_ids):
        print(
            "[ERROR][phase1_bay_balancer._normalize_bay_capacity_weights] "
            f"cause=capacity_scope_mismatch bay_ids={list(bay_ids)} "
            f"weight_bays={sorted(normalized)}"
        )
        raise RuntimeError("Phase 1 Bay capacity scope does not match Bay IDs")
    for bay_id, weight in normalized.items():
        if not math.isfinite(weight) or weight <= 0:
            print(
                "[ERROR][phase1_bay_balancer._normalize_bay_capacity_weights] "
                f"cause=invalid_capacity_weight bay_id={bay_id} value={weight}"
            )
            raise RuntimeError(f"invalid Phase 1 capacity weight: {bay_id}")
    return {bay_id: normalized[bay_id] for bay_id in bay_ids}


def _empty_phase1_bay_loads(
    bay_ids: Tuple[str, ...],
    bay_capacity_weights: Mapping[str, int | float] | None,
) -> Dict[str, Dict[str, int | float]]:
    """MIXED joint score에 필요한 Bay별/그룹별 누적 부하를 만든다."""

    weights = _normalize_bay_capacity_weights(bay_ids, bay_capacity_weights)
    loads: Dict[str, Dict[str, int | float]] = {
        bay_id: {
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "wo_count": 0,
            "block_count": 0,
            "capacity_weight": weights[bay_id],
        }
        for bay_id in bay_ids
    }
    initialize_multi_series_group_loads(loads)
    return loads


def _collect_blocks(
    jobs: Mapping[str, object],
    bay_ids: Tuple[str, ...],
) -> List[Phase1Block]:
    """W/O를 `PROJ_NO+GYEL+BLK_NO` 단위로 엄격히 집계한다."""

    if not jobs:
        print("[ERROR][phase1_bay_balancer._collect_blocks] cause=no_jobs")
        raise RuntimeError("Phase 1 requires at least one W/O")
    grouped: Dict[str, List[object]] = {}
    for job_key, job in jobs.items():
        block_set_id = _require_text(
            _job_attr(job, "block_set_id"), "block_set_id", str(job_key)
        )
        grouped.setdefault(block_set_id, []).append(job)

    blocks: List[Phase1Block] = []
    for block_set_id, block_jobs in sorted(grouped.items()):
        job_ids: List[str] = []
        families: set[str] = set()
        steel_quantity_sum = 0
        cut_length_sum = 0.0
        bevel_quantity_sum = 0
        lengths: List[float] = []
        thicknesses: List[float] = []
        widths: List[float] = []
        for job in block_jobs:
            job_id = _require_text(_job_attr(job, "job_id"), "job_id", block_set_id)
            job_ids.append(job_id)
            families.add(_require_text(_job_attr(job, "family"), "family", job_id).upper())
            steel_quantity_sum += _require_non_negative_int(
                _job_attr(job, "steel_quantity"), "steel_quantity", job_id
            )
            cut_length_sum += _require_non_negative_float(
                _job_attr(job, "cut_length"), "cut_length", job_id
            )
            bevel_quantity_sum += _require_non_negative_int(
                _job_attr(job, "bevel_quantity"), "bevel_quantity", job_id
            )
            lengths.append(
                _require_non_negative_float(_job_attr(job, "plate_length"), "plate_length", job_id)
            )
            thicknesses.append(
                _require_non_negative_float(_job_attr(job, "thickness"), "thickness", job_id)
            )
            widths.append(
                _require_non_negative_float(_job_attr(job, "plate_width"), "plate_width", job_id)
            )
        if len(families) != 1:
            print(
                "[ERROR][phase1_bay_balancer._collect_blocks] "
                f"cause=mixed_family_in_block block_set_id={block_set_id} families={sorted(families)}"
            )
            raise RuntimeError(f"mixed family in Phase 1 block-series: {block_set_id}")
        family = next(iter(families))
        project_no, block_no = _project_block_labels(block_set_id, block_jobs)
        parts = block_set_id.split("::")
        if len(parts) != 3 or parts[1].upper() != family:
            print(
                "[ERROR][phase1_bay_balancer._collect_blocks] "
                f"cause=block_identity_mismatch block_set_id={block_set_id} family={family}"
            )
            raise RuntimeError(f"invalid MIXED block identity: {block_set_id}")
        width_max = max(widths)
        explicit_bays = _explicit_allowed_bays(block_jobs, bay_ids, block_set_id)
        mask = apply_phase1_series_bay_mask(
            series=family,
            block_no=block_no,
            width_max=width_max,
            cut_length_sum=cut_length_sum,
            requested_bays=explicit_bays,
        )
        blocks.append(
            Phase1Block(
                block_set_id=block_set_id,
                project_no=project_no,
                block_no=block_no,
                job_ids=tuple(job_ids),
                wo_count=len(block_jobs),
                steel_quantity_sum=steel_quantity_sum,
                cut_length_sum=cut_length_sum,
                bevel_quantity_sum=bevel_quantity_sum,
                long_cut_over_1000=int(cut_length_sum >= 1000.0),
                allowed_bay_ids=mask.allowed_bay_ids,
                length_avg=sum(lengths) / len(lengths),
                thickness_avg=sum(thicknesses) / len(thicknesses),
                family=family,
                balancing_group=mask.balancing_group,
                width_max=width_max,
                wide_plate_over_4500=int(family == "NP" and width_max > 4500.0),
                cnt_block=int(family == "NP" and block_no.upper().startswith("CNT_BLK")),
                bay_mask_reason_codes=mask.reason_codes,
            )
        )
    return blocks


def _add_block_load(loads: Dict[str, int | float], bay_id: str, block: Phase1Block) -> None:
    """한 block-series의 W/O/CUT/BV 부하를 선택 Bay에 누적한다."""

    loads["steel_quantity_sum"] += block.steel_quantity_sum
    loads["cut_length_sum"] += block.cut_length_sum
    loads["bevel_quantity_sum"] += block.bevel_quantity_sum
    loads["wo_count"] += block.wo_count
    loads["block_count"] += 1
    add_multi_series_group_load(
        loads,
        group=block.balancing_group,
        wo_count=block.wo_count,
        cut_length_sum=block.cut_length_sum,
        bevel_quantity_sum=block.bevel_quantity_sum,
    )


def _multi_objective_load_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> Tuple[float, float, float, float, float, float]:
    """공유 설비군 전체 gap과 계열별 gap을 W/O -> CUT -> BV 순서로 반환한다."""

    shared_pool_scores = {
        "wo_count": 0.0,
        "cut_length_sum": 0.0,
        "bevel_quantity_sum": 0.0,
    }
    series_group_scores = {
        "wo_count": 0.0,
        "cut_length_sum": 0.0,
        "bevel_quantity_sum": 0.0,
    }
    for pool_bays in (("22", "23", "24"), ("25", "trans")):
        missing = sorted(set(pool_bays) - set(bay_loads))
        if missing:
            print(
                "[ERROR][phase1_bay_balancer._multi_objective_load_score] "
                f"cause=missing_shared_pool_bays bay_ids={missing}"
            )
            raise RuntimeError(f"missing Phase 1 shared-pool Bays: {missing}")
        for metric in shared_pool_scores:
            values = [
                _require_non_negative_float(
                    bay_loads[bay_id].get(metric),
                    metric,
                    f"shared_pool_score:{bay_id}",
                )
                / _require_capacity_weight(
                    bay_loads[bay_id], f"shared_pool_score:{bay_id}"
                )
                for bay_id in pool_bays
            ]
            shared_pool_scores[metric] += _gap(values)

    for group in PHASE1_BALANCING_GROUP_ORDER:
        group_weights = GROUP_BAY_CAPACITY_WEIGHTS[group]
        if not set(group_weights) <= set(bay_loads):
            missing = sorted(set(group_weights) - set(bay_loads))
            print(
                "[ERROR][phase1_bay_balancer._multi_objective_load_score] "
                f"cause=missing_group_bays group={group} bay_ids={missing}"
            )
            raise RuntimeError(f"missing Phase 1 group Bays: {group}/{missing}")
        for bay_id, expected_weight in group_weights.items():
            actual_weight = _require_capacity_weight(
                bay_loads[bay_id], f"score:{group}:{bay_id}"
            )
            if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1e-9):
                print(
                    "[ERROR][phase1_bay_balancer._multi_objective_load_score] "
                    f"cause=capacity_weight_mismatch group={group} bay_id={bay_id} "
                    f"expected={expected_weight} actual={actual_weight}"
                )
                raise RuntimeError(f"Phase 1 group capacity mismatch: {group}/{bay_id}")
        for metric in series_group_scores:
            values = [
                multi_series_group_load_value(bay_loads[bay_id], group, metric)
                / group_weights[bay_id]
                for bay_id in group_weights
            ]
            series_group_scores[metric] += _gap(values)
    return tuple(
        _round_score(score)
        for metric in ("wo_count", "cut_length_sum", "bevel_quantity_sum")
        for score in (shared_pool_scores[metric], series_group_scores[metric])
    )


def _require_capacity_weight(row: Mapping[str, int | float], location: str) -> float:
    """Bay 설비 수 분모를 누락 없이 읽는다."""

    try:
        weight = float(row["capacity_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_capacity_weight] "
            f"cause=missing_or_invalid_capacity_weight location={location}"
        )
        raise RuntimeError(f"invalid Bay capacity weight: {location}") from exc
    if not math.isfinite(weight) or weight <= 0:
        print(
            "[ERROR][phase1_bay_balancer._require_capacity_weight] "
            f"cause=non_positive_capacity_weight location={location} value={weight}"
        )
        raise RuntimeError(f"invalid Bay capacity weight: {location}")
    return weight


def _phase1_assignment_map(plan: Mapping) -> Dict[str, str]:
    rows = plan.get("assignments")
    if not isinstance(rows, list) or not rows:
        print("[ERROR][phase1_bay_balancer._phase1_assignment_map] cause=no_assignments")
        raise RuntimeError("Phase 1 plan requires non-empty assignments")
    result: Dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][phase1_bay_balancer._phase1_assignment_map] "
                f"cause=invalid_assignment_row index={index}"
            )
            raise RuntimeError(f"invalid Phase 1 assignment row: {index}")
        block_set_id = _require_text(row.get("block_set_id"), "block_set_id", str(index))
        assigned_bay = _require_text(row.get("assigned_bay"), "assigned_bay", block_set_id)
        if block_set_id in result:
            print(
                "[ERROR][phase1_bay_balancer._phase1_assignment_map] "
                f"cause=duplicate_assignment block_set_id={block_set_id}"
            )
            raise RuntimeError(f"duplicate Phase 1 assignment: {block_set_id}")
        result[block_set_id] = assigned_bay
    return result


def _explicit_allowed_bays(
    block_jobs: Sequence[object],
    bay_ids: Tuple[str, ...],
    block_set_id: str,
) -> Tuple[str, ...]:
    """명시된 planning whitelist만 교집합하고 실적 CUT_BAY는 사용하지 않는다."""

    allowed = set(bay_ids)
    for job in block_jobs:
        values = _job_attr(job, "allowed_bay_ids") or ()
        if values:
            allowed &= {str(value) for value in values}
    result = tuple(bay_id for bay_id in bay_ids if bay_id in allowed)
    if not result:
        print(
            "[ERROR][phase1_bay_balancer._explicit_allowed_bays] "
            f"cause=no_feasible_bay block_set_id={block_set_id}"
        )
        raise RuntimeError(f"no feasible Bay before MIXED hard mask: {block_set_id}")
    return result


def _project_block_labels(
    block_set_id: str,
    block_jobs: Sequence[object],
) -> Tuple[str, str]:
    parts = block_set_id.split("::")
    if len(parts) == 3 and parts[0] and parts[2]:
        return parts[0], parts[2]
    first_job = block_jobs[0]
    extra = _job_attr(first_job, "extra") or {}
    if isinstance(extra, Mapping):
        project_no = str(extra.get("source_project_no") or "").strip()
        block_no = str(extra.get("source_block_no") or "").strip()
        if project_no and block_no:
            return project_no, block_no
    print(
        "[ERROR][phase1_bay_balancer._project_block_labels] "
        f"cause=invalid_block_set_id block_set_id={block_set_id}"
    )
    raise RuntimeError(f"invalid MIXED block_set_id: {block_set_id}")


def _job_attr(job: object, name: str):
    return job.get(name) if isinstance(job, Mapping) else getattr(job, name, None)


def _require_text(value: object, field_name: str, row_key: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][phase1_bay_balancer._require_text] "
            f"cause=missing_{field_name} row_key={row_key}"
        )
        raise RuntimeError(f"missing {field_name}: {row_key}")
    return text


def _require_non_negative_int(value: object, field_name: str, row_key: str) -> int:
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_int] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid {field_name}: {row_key}") from exc
    parsed = int(parsed_float)
    if not math.isfinite(parsed_float) or parsed_float != parsed or parsed < 0:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_int] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid {field_name}: {row_key}")
    return parsed


def _require_non_negative_float(value: object, field_name: str, row_key: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_float] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid {field_name}: {row_key}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        print(
            "[ERROR][phase1_bay_balancer._require_non_negative_float] "
            f"cause=invalid_{field_name} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid {field_name}: {row_key}")
    return parsed


def _gap(values: Sequence[int | float]) -> float:
    return max(values) - min(values) if values else 0.0


def _round_score(value: int | float) -> float:
    return round(float(value), 9)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
