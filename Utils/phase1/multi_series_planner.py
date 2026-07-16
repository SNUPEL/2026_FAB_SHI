"""검증된 다계열 데이터에서 일별 Phase 1 문제를 구성한다."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from Utils.data.cutting_start_date import audit_cutting_start_dates
from Utils.data.multi_series_cutting_data import MultiSeriesCuttingData
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_BALANCING_GROUP_ORDER,
    balancing_group_for_series,
    joint_phase1_bay_capacity_weights,
)
from Phase1.heuristics import PHASE1_HEURISTIC_BANK, run_phase1_heuristic_candidate
from Phase1.orchestrator import candidate_to_phase1_plan


def build_multi_series_phase1_training_problem(jobs: Mapping[str, object]) -> dict:
    """모든 계열 W/O를 유지한 하나의 다섯-Bay joint 학습 문제를 만든다."""

    if not isinstance(jobs, Mapping) or not jobs:
        print(
            "[ERROR][multi_series_planner.build_multi_series_phase1_training_problem] "
            "cause=empty_or_invalid_jobs"
        )
        raise RuntimeError("multi-series Phase 1 training requires non-empty jobs")
    present_groups: set[str] = set()
    block_ids: set[str] = set()
    for job_key, job in jobs.items():
        family = _job_required_text(job, "family", str(job_key)).upper()
        present_groups.add(balancing_group_for_series(family))
        block_ids.add(_job_required_text(job, "block_set_id", str(job_key)))
    ordered_groups = tuple(
        group for group in PHASE1_BALANCING_GROUP_ORDER if group in present_groups
    )
    if not ordered_groups:
        print(
            "[ERROR][multi_series_planner.build_multi_series_phase1_training_problem] "
            "cause=no_supported_balancing_group"
        )
        raise RuntimeError("multi-series Phase 1 episode has no supported balancing group")
    capacity_weights = joint_phase1_bay_capacity_weights()
    return {
        "balancing_groups": ordered_groups,
        "jobs": dict(jobs),
        "bay_ids": tuple(capacity_weights),
        "bay_capacity_weights": capacity_weights,
        "block_count": len(block_ids),
        "job_count": len(jobs),
    }


def _job_required_text(job: object, field: str, row_key: str) -> str:
    """dict/namespace Job의 필수 문자열을 대체값 없이 읽는다."""

    value = job.get(field) if isinstance(job, Mapping) else getattr(job, field, None)
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][multi_series_planner._job_required_text] "
            f"cause=missing_required_field field={field} row_key={row_key}"
        )
        raise RuntimeError(f"missing multi-series Phase 1 job field: {field}")
    return text


def build_multi_series_phase1_daily_plans(
    data: MultiSeriesCuttingData,
    *,
    holiday_dates: Iterable[object] | None = None,
    algorithm: str = PHASE1_HEURISTIC_BANK[0],
) -> dict:
    """계산 착수일별로 모든 계열을 포함한 다섯-Bay joint plan을 만든다.

    TACT_TIME은 Phase 1 입력이 아니다. FN/FL/NC에 NP Case 6 공통식을
    적용하는 생성 가정과 이 Bay 배정 로직은 독립적이다. 실적 `CUT_BAY`는
    source audit 필드로만 보존하고 계획 후보에는 사용하지 않는다.
    """

    _validate_data_contract(data)
    block_records = data.blocks.to_dict("records")
    date_audit = audit_cutting_start_dates(block_records, holiday_dates=holiday_dates)
    workday_by_block = dict(
        zip(date_audit["BLOCK_SET_ID"], date_audit["CALCULATED_ACT_ST_DT"])
    )

    grouped_jobs: dict[str, dict[str, dict]] = {}
    grouped_families: dict[str, set[str]] = {}
    grouped_balancing_groups: dict[str, set[str]] = {}
    for row_index, row in data.work_orders.iterrows():
        block_set_id = _required_text(row, "BLOCK_SET_ID", f"wo_row_{row_index}")
        workday = workday_by_block.get(block_set_id)
        if workday is None:
            print(
                "[ERROR][multi_series_planner.build_multi_series_phase1_daily_plans] "
                f"cause=missing_block_workday block_set_id={block_set_id}"
            )
            raise RuntimeError(f"missing calculated Phase 1 workday: {block_set_id}")
        series = _required_text(row, "GYEL", block_set_id).upper()
        group = balancing_group_for_series(series)
        problem_key = str(workday)
        job = _phase1_job_from_wo(row, row_index)
        if job["job_id"] in grouped_jobs.setdefault(problem_key, {}):
            print(
                "[ERROR][multi_series_planner.build_multi_series_phase1_daily_plans] "
                f"cause=duplicate_work_order job_id={job['job_id']}"
            )
            raise RuntimeError(f"duplicate W/O in Phase 1 daily problems: {job['job_id']}")
        grouped_jobs[problem_key][job["job_id"]] = job
        grouped_families.setdefault(problem_key, set()).add(series)
        grouped_balancing_groups.setdefault(problem_key, set()).add(group)

    problems = []
    for workday, jobs in sorted(grouped_jobs.items()):
        capacity_weights = joint_phase1_bay_capacity_weights()
        candidate = run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=tuple(capacity_weights),
            algorithm=algorithm,
            bay_capacity_weights=capacity_weights,
        )
        plan = candidate_to_phase1_plan(jobs, tuple(capacity_weights), candidate)
        problems.append(
            {
                "problem_id": str(workday),
                "workday": workday,
                "balancing_groups": [
                    group
                    for group in PHASE1_BALANCING_GROUP_ORDER
                    if group in grouped_balancing_groups[workday]
                ],
                "families": sorted(grouped_families[workday]),
                "block_count": int(plan["summary"]["block_count"]),
                "wo_count": int(plan["summary"]["job_count"]),
                "bay_capacity_weights": capacity_weights,
                "plan": plan,
            }
        )

    result = {
        "phase": "phase1_multi_series_daily",
        "rule_profile": MULTI_SERIES_RULE_PROFILE,
        "score_mode": "wo_first",
        "problem_count": len(problems),
        "block_count": int(len(data.blocks)),
        "wo_count": int(len(data.work_orders)),
        "problems": problems,
        "start_date_audit": {
            "row_count": int(len(date_audit)),
            "matched_source_count": int(date_audit["MATCHED_SOURCE_DATE"].sum()),
        },
    }
    print(
        "[VALIDATION][multi_series_planner.build_multi_series_phase1_daily_plans] "
        f"passed=true problems={len(problems)} blocks={len(data.blocks)} wos={len(data.work_orders)}"
    )
    return result


def write_multi_series_phase1_daily_plans(result: Mapping, output_dir: str | Path) -> dict[str, str]:
    """다계열 Phase 1 전체 JSON과 block/Bay CSV를 저장한다."""

    if "problems" not in result or not isinstance(result["problems"], list) or not result["problems"]:
        print(
            "[ERROR][multi_series_planner.write_multi_series_phase1_daily_plans] "
            "cause=missing_or_empty_problems"
        )
        raise RuntimeError("multi-series Phase 1 result requires non-empty problems")
    problems = result["problems"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "phase1_multi_series_daily_plans.json"
    assignment_path = output_path / "phase1_multi_series_assignments.csv"
    bay_load_path = output_path / "phase1_multi_series_bay_loads.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    assignment_rows = []
    bay_rows = []
    for problem in problems:
        common = {
            "problem_id": problem["problem_id"],
            "workday": problem["workday"],
            "balancing_groups": "|".join(problem["balancing_groups"]),
        }
        for row in problem["plan"]["assignments"]:
            assignment_rows.append({**common, **row})
        for bay_id, loads in problem["plan"]["bay_loads"].items():
            bay_rows.append({**common, "bay_id": bay_id, **loads})
    _write_rows(assignment_path, assignment_rows)
    _write_rows(bay_load_path, bay_rows)
    print(
        "[CHECK][multi_series_planner.write_multi_series_phase1_daily_plans] "
        f"assignments={len(assignment_rows)} bay_rows={len(bay_rows)} output={output_path}"
    )
    return {
        "json": str(json_path),
        "assignments_csv": str(assignment_path),
        "bay_loads_csv": str(bay_load_path),
    }


def _phase1_job_from_wo(row: pd.Series, row_index: object) -> dict:
    job_id = _required_text(row, "WK_ORD_NO", f"wo_row_{row_index}")
    return {
        "job_id": job_id,
        "block_set_id": _required_text(row, "BLOCK_SET_ID", job_id),
        "family": _required_text(row, "GYEL", job_id).upper(),
        "steel_quantity": _required_non_negative_number(row, "STL_QTY", job_id, integer=True),
        "cut_length": _required_non_negative_number(row, "CUT_LTH", job_id),
        "bevel_quantity": _required_non_negative_number(row, "BV_QTY", job_id, integer=True),
        "plate_length": _required_non_negative_number(row, "LTH", job_id),
        "plate_width": _required_non_negative_number(row, "BTH", job_id),
        "thickness": _required_non_negative_number(row, "THK", job_id),
        "cut_bay": None,
        "source_cut_bay": _required_text(row, "CUT_BAY", job_id),
        "allowed_bay_ids": (),
        "extra": {
            "source_project_no": _required_text(row, "PROJ_NO", job_id),
            "source_block_no": _required_text(row, "BLK_NO", job_id),
            "source_wk_ord_no": job_id,
        },
    }


def _validate_data_contract(data: MultiSeriesCuttingData) -> None:
    if not isinstance(data, MultiSeriesCuttingData):
        print(
            "[ERROR][multi_series_planner._validate_data_contract] "
            f"cause=invalid_data_type data_type={type(data).__name__}"
        )
        raise RuntimeError("multi-series Phase 1 requires MultiSeriesCuttingData")
    if data.blocks.empty or data.work_orders.empty:
        print("[ERROR][multi_series_planner._validate_data_contract] cause=empty_data")
        raise RuntimeError("multi-series Phase 1 requires non-empty block and W/O rows")
    block_required = {"BLOCK_SET_ID", "PROJ_NO", "GYEL", "BLK_NO", "ACT_ST_DT"}
    wo_required = {
        "WK_ORD_NO", "BLOCK_SET_ID", "PROJ_NO", "GYEL", "BLK_NO", "STL_QTY",
        "CUT_LTH", "BV_QTY", "LTH", "BTH", "THK", "CUT_BAY",
    }
    missing_blocks = sorted(block_required - set(data.blocks.columns))
    missing_wos = sorted(wo_required - set(data.work_orders.columns))
    if missing_blocks or missing_wos:
        print(
            "[ERROR][multi_series_planner._validate_data_contract] "
            f"cause=missing_columns block={missing_blocks} wo={missing_wos}"
        )
        raise RuntimeError("multi-series Phase 1 data contract is incomplete")


def _required_text(row: Mapping, field: str, row_key: str) -> str:
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][multi_series_planner._required_text] "
            f"cause=missing_text field={field} row_key={row_key}"
        )
        raise RuntimeError(f"missing multi-series Phase 1 field: {field}")
    return text


def _required_non_negative_number(
    row: Mapping,
    field: str,
    row_key: str,
    *,
    integer: bool = False,
) -> int | float:
    value = row.get(field)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][multi_series_planner._required_non_negative_number] "
            f"cause=invalid_number field={field} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid multi-series Phase 1 field: {field}") from exc
    if pd.isna(parsed) or parsed < 0:
        print(
            "[ERROR][multi_series_planner._required_non_negative_number] "
            f"cause=invalid_number field={field} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"invalid multi-series Phase 1 field: {field}")
    if integer:
        if not parsed.is_integer():
            print(
                "[ERROR][multi_series_planner._required_non_negative_number] "
                f"cause=non_integer field={field} row_key={row_key} value={value}"
            )
            raise RuntimeError(f"non-integer multi-series Phase 1 field: {field}")
        return int(parsed)
    return parsed


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(
            "[ERROR][multi_series_planner._write_rows] "
            f"cause=no_rows path={path}"
        )
        raise RuntimeError(f"cannot write empty Phase 1 result: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
