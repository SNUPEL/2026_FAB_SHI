"""Phase 2 actual-8days 후보 workbook을 W/O scheduling 문제로 변환한다.

핵심 기준:
- `착수일 후보 블록.xlsx`는 날짜별 block universe를 정의한다.
- Phase 2는 W/O batch-machine 문제이므로, 각 block에 속한 전체 W/O를
  기준 W/O 원본 파일에서 다시 찾아 scenario job으로 만든다.
- missing column, missing block, Bay scope 불일치는 fallback 없이 실패한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from Utils.data.cutting_scenario_builder import build_scenario_from_cutting_records
from Utils.data.multi_series_cutting_data import build_block_set_id
from Utils.phase1.phase1_episode_dataset import build_phase1_candidate_workbook_jobs


PHASE2_CANDIDATE_WO_REQUIRED_COLUMNS = (
    "PROJ_NO",
    "BLK_NO",
    "WK_ORD_NO",
    "GYEL_ACT_STDT",
    "GYEL_ACT_EDDT",
    "RT_CUT_ST_DTM",
    "RT_CUT_ED_DTM",
    "GYEL",
    "LTH",
    "THK",
    "CUT_LTH",
    "MARK_LTH",
    "BVL_LTH",
    "STL_QTY",
    "BV_QTY",
    "RT_EQP_NM",
    "CUT_BAY",
    "TACT_TIME",
    "ASS_ST_DT",
    "PTLST_QTY",
)


def load_phase2_candidate_workbook_problems(
    wo_path: str | Path,
    candidate_path: str | Path,
    workdays: Sequence[str],
    bay_ids: Sequence[str],
    gyel: str = "NP",
    factory_config: Mapping[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build merged Phase 2 problems from candidate block workbook + W/O source.

    반환 payload 하나가 날짜 하나의 문제다.
    - `phase1_jobs`: Phase 1 block->Bay 배정에 쓰는 block 단위 job
    - `scenario.jobs`: merged Phase 2가 스케줄링할 W/O 단위 job
    """

    source = Path(wo_path)
    candidate = Path(candidate_path)
    allowed_bays = tuple(str(bay_id).strip() for bay_id in bay_ids if str(bay_id).strip())
    requested_workdays = tuple(str(workday).strip() for workday in workdays if str(workday).strip())
    if not allowed_bays:
        print("[ERROR][phase2_candidate_workbook.load_phase2_candidate_workbook_problems] cause=no_bay_ids")
        raise RuntimeError("bay_ids are required")
    if not requested_workdays:
        print("[ERROR][phase2_candidate_workbook.load_phase2_candidate_workbook_problems] cause=no_workdays")
        raise RuntimeError("workdays are required")

    phase1_payloads = build_phase1_candidate_workbook_jobs(
        candidate_path=candidate,
        workdays=requested_workdays,
        bay_ids=allowed_bays,
    )
    raw_wo = _read_wo_table(source)
    _require_columns(raw_wo, PHASE2_CANDIDATE_WO_REQUIRED_COLUMNS, source)
    raw_wo = raw_wo.copy()
    raw_wo["__source_row_index"] = raw_wo.index.astype(int) + 2
    raw_wo["__block_key"] = raw_wo.apply(
        lambda row: build_block_set_id(row["PROJ_NO"], row["GYEL"], row["BLK_NO"]),
        axis=1,
    )
    raw_wo["__cut_bay"] = raw_wo["CUT_BAY"].map(_normalize_bay)

    filtered_by_gyel = raw_wo[raw_wo["GYEL"].astype(str).str.strip().eq(str(gyel))].copy()
    print(
        "[CHECK][phase2_candidate_workbook.load_phase2_candidate_workbook_problems] "
        f"gyel={gyel} rows_before={len(raw_wo)} rows_after={len(filtered_by_gyel)}"
    )
    if filtered_by_gyel.empty:
        print(
            "[ERROR][phase2_candidate_workbook.load_phase2_candidate_workbook_problems] "
            f"cause=no_rows_after_gyel_filter gyel={gyel} path={source}"
        )
        raise RuntimeError(f"no_rows_after_gyel_filter: {gyel}")

    payloads: List[Dict[str, Any]] = []
    for phase1_payload in phase1_payloads:
        workday = str(phase1_payload["workday"])
        block_keys = sorted({str(job.block_set_id) for job in phase1_payload["jobs"].values()})
        subset = filtered_by_gyel[filtered_by_gyel["__block_key"].isin(block_keys)].copy()
        _validate_workday_subset(
            subset=subset,
            block_keys=block_keys,
            allowed_bays=allowed_bays,
            source=source,
            workday=workday,
        )
        records = _records_from_wo_subset(subset, workday)
        scenario = build_scenario_from_cutting_records(
            records,
            factory_config=dict(factory_config) if factory_config is not None else None,
            source_file=f"{source}|candidate={candidate}|workday={workday}",
            process_time_source="tact_time",
        )
        scenario.setdefault("metadata", {})
        scenario["metadata"].update(
            {
                "input_mode": "phase2_candidate_workbook",
                "candidate_xlsx": str(candidate),
                "wo_xlsx": str(source),
                "workday": workday,
                "candidate_block_count": len(block_keys),
                "selected_wo_count": len(records),
            }
        )
        payloads.append(
            {
                "problem_id": f"ACTUAL_{workday}",
                "workday": workday,
                "candidate_block_count": len(block_keys),
                "wo_count": len(records),
                "candidate_block_keys": block_keys,
                "phase1_jobs": phase1_payload["jobs"],
                "phase1_metadata": dict(phase1_payload.get("metadata", {})),
                "scenario": scenario,
            }
        )

    print(
        "[VALIDATION][phase2_candidate_workbook.load_phase2_candidate_workbook_problems] "
        f"passed=true workdays={len(payloads)} source={source} candidate={candidate}"
    )
    return payloads


def _read_wo_table(path: Path) -> pd.DataFrame:
    """Read W/O source table. Unsupported or missing source fails immediately."""

    if not path.exists():
        print(f"[ERROR][phase2_candidate_workbook._read_wo_table] cause=missing_wo_source path={path}")
        raise RuntimeError(f"missing_wo_source: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    print(f"[ERROR][phase2_candidate_workbook._read_wo_table] cause=unsupported_suffix suffix={path.suffix}")
    raise RuntimeError(f"unsupported W/O source suffix: {path.suffix}")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], source: Path) -> None:
    """Validate required W/O columns; no alias lookup is used in this path."""

    missing = [column for column in required if column not in frame.columns]
    if missing:
        print(
            "[ERROR][phase2_candidate_workbook._require_columns] "
            f"cause=missing_required_columns missing={missing} path={source}"
        )
        raise RuntimeError(f"missing_required_columns: {missing}")


def _validate_workday_subset(
    subset: pd.DataFrame,
    block_keys: Sequence[str],
    allowed_bays: Sequence[str],
    source: Path,
    workday: str,
) -> None:
    """Ensure candidate blocks have W/O rows and stay inside the requested Bay scope."""

    if subset.empty:
        print(
            "[ERROR][phase2_candidate_workbook._validate_workday_subset] "
            f"cause=no_wo_rows_for_candidate_blocks workday={workday} path={source}"
        )
        raise RuntimeError(f"no_wo_rows_for_candidate_blocks: {workday}")
    present_blocks = set(subset["__block_key"].astype(str))
    missing_blocks = sorted(set(block_keys) - present_blocks)
    if missing_blocks:
        print(
            "[ERROR][phase2_candidate_workbook._validate_workday_subset] "
            f"cause=missing_candidate_blocks_in_wo_source workday={workday} "
            f"missing_count={len(missing_blocks)} examples={missing_blocks[:5]}"
        )
        raise RuntimeError(f"missing_candidate_blocks_in_wo_source: {workday}")
    outside = sorted(set(subset["__cut_bay"].astype(str)) - set(allowed_bays))
    if outside:
        print(
            "[ERROR][phase2_candidate_workbook._validate_workday_subset] "
            f"cause=wo_bay_outside_scope workday={workday} bays={outside}"
        )
        raise RuntimeError(f"wo_bay_outside_scope: {workday}")
    duplicate_wo = subset["WK_ORD_NO"].astype(str).str.strip().duplicated()
    if bool(duplicate_wo.any()):
        examples = subset.loc[duplicate_wo, "WK_ORD_NO"].astype(str).head(5).tolist()
        print(
            "[ERROR][phase2_candidate_workbook._validate_workday_subset] "
            f"cause=duplicate_wk_ord_no workday={workday} examples={examples}"
        )
        raise RuntimeError(f"duplicate_wk_ord_no: {workday}")


def _records_from_wo_subset(subset: pd.DataFrame, workday: str) -> List[Dict[str, Any]]:
    """Convert strict W/O rows to canonical records consumed by scenario builder."""

    records: List[Dict[str, Any]] = []
    ordered = subset.sort_values(["PROJ_NO", "GYEL", "BLK_NO", "WK_ORD_NO"]).reset_index(drop=True)
    for row_index, row in ordered.iterrows():
        row_key = f"{workday}:row{row_index + 1}"
        project_no = _required_text(row["PROJ_NO"], "PROJ_NO", row_key)
        series = _required_text(row["GYEL"], "GYEL", row_key)
        block_no = _required_text(row["BLK_NO"], "BLK_NO", row_key)
        work_order_no = _required_text(row["WK_ORD_NO"], "WK_ORD_NO", row_key)
        source_cut_bay = _normalize_bay(row["CUT_BAY"])
        records.append(
            {
                "project_no": project_no,
                "block_no": block_no,
                "work_order_no": work_order_no,
                "planned_start_date": _optional_text(row["GYEL_ACT_STDT"]),
                "planned_end_date": _optional_text(row["GYEL_ACT_EDDT"]),
                "actual_start_datetime": _optional_text(row["RT_CUT_ST_DTM"]),
                "actual_end_datetime": _optional_text(row["RT_CUT_ED_DTM"]),
                "series": series,
                "length": _required_positive_float(row["LTH"], "LTH", row_key),
                "thickness": _required_positive_float(row["THK"], "THK", row_key),
                "cut_length": _required_non_negative_float(row["CUT_LTH"], "CUT_LTH", row_key),
                "mark_length": _required_non_negative_float(row["MARK_LTH"], "MARK_LTH", row_key),
                "bevel_length": _required_non_negative_float(row["BVL_LTH"], "BVL_LTH", row_key),
                "steel_qty": _required_positive_int(row["STL_QTY"], "STL_QTY", row_key),
                "bevel_qty": _required_non_negative_int(row["BV_QTY"], "BV_QTY", row_key),
                "source_machine_id": _optional_text(row["RT_EQP_NM"]),
                "source_cut_bay": source_cut_bay,
                "tact_time": _required_positive_float(row["TACT_TIME"], "TACT_TIME", row_key),
                "downstream_date": _optional_text(row["ASS_ST_DT"]),
                "part_qty": _required_non_negative_int(row["PTLST_QTY"], "PTLST_QTY", row_key),
                "source_row_index": int(row["__source_row_index"]),
                "block_set_id": build_block_set_id(project_no, series, block_no),
            }
        )
    return records


def _required_text_cell(value: object) -> str:
    """Map helper for vectorized project/block key creation."""

    if pd.isna(value) or not str(value).strip():
        print("[ERROR][phase2_candidate_workbook._required_text_cell] cause=blank_key_cell")
        raise RuntimeError("blank key cell")
    return str(value).strip()


def _required_text(value: object, column: str, row_key: str) -> str:
    """Read a non-empty text field."""

    if pd.isna(value) or not str(value).strip():
        print(
            "[ERROR][phase2_candidate_workbook._required_text] "
            f"cause=blank_value column={column} row_key={row_key}"
        )
        raise RuntimeError(f"blank_value: {column} {row_key}")
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _optional_text(value: object) -> str:
    """Preserve optional date/timestamp cells without inventing replacements."""

    if pd.isna(value) or not str(value).strip():
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _required_positive_float(value: object, column: str, row_key: str) -> float:
    """Read a positive numeric field."""

    number = _required_number(value, column, row_key)
    if number <= 0:
        print(
            "[ERROR][phase2_candidate_workbook._required_positive_float] "
            f"cause=non_positive column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_positive: {column} {row_key}")
    return number


def _required_non_negative_float(value: object, column: str, row_key: str) -> float:
    """Read a non-negative numeric field."""

    number = _required_number(value, column, row_key)
    if number < 0:
        print(
            "[ERROR][phase2_candidate_workbook._required_non_negative_float] "
            f"cause=negative column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative: {column} {row_key}")
    return number


def _required_positive_int(value: object, column: str, row_key: str) -> int:
    """Read a positive integer-like numeric field."""

    parsed = _required_int_like(value, column, row_key)
    if parsed <= 0:
        print(
            "[ERROR][phase2_candidate_workbook._required_positive_int] "
            f"cause=non_positive column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_positive: {column} {row_key}")
    return parsed


def _required_non_negative_int(value: object, column: str, row_key: str) -> int:
    """Read a non-negative integer-like numeric field."""

    parsed = _required_int_like(value, column, row_key)
    if parsed < 0:
        print(
            "[ERROR][phase2_candidate_workbook._required_non_negative_int] "
            f"cause=negative column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"negative: {column} {row_key}")
    return parsed


def _required_number(value: object, column: str, row_key: str) -> float:
    """Convert a value to float; missing/non-numeric values fail."""

    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        print(
            "[ERROR][phase2_candidate_workbook._required_number] "
            f"cause=non_numeric_or_missing column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_numeric_or_missing: {column} {row_key}")
    return float(number)


def _required_int_like(value: object, column: str, row_key: str) -> int:
    """Convert an integer-like value without rounding."""

    number = _required_number(value, column, row_key)
    if not float(number).is_integer():
        print(
            "[ERROR][phase2_candidate_workbook._required_int_like] "
            f"cause=non_integer column={column} row_key={row_key} value={value}"
        )
        raise RuntimeError(f"non_integer: {column} {row_key}")
    return int(number)


def _normalize_bay(value: object) -> str:
    """Normalize Excel Bay values such as 22.0 to '22'."""

    if pd.isna(value) or not str(value).strip():
        print("[ERROR][phase2_candidate_workbook._normalize_bay] cause=missing_cut_bay")
        raise RuntimeError("missing_cut_bay")
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


__all__ = ["load_phase2_candidate_workbook_problems"]
