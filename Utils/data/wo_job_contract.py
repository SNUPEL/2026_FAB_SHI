# -*- coding: utf-8 -*-
"""합성 W/O ↔ Job/scenario 변환 계약.

생성기(params_generator)와 학습/스케줄 경계 사이의 순수 변환·검증 유틸이다.
생성식(계수·형태)은 전혀 포함하지 않으므로, 어떤 생성기가 wo_df를 만들든
동일하게 쓰인다. no-silent-fallback: 누락/비수치/음수/비정수는 raise한다.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from Utils.data.multi_series_cutting_data import build_block_set_id

# W/O 계약 컬럼 (params_generator 출력과 일치해야 한다).
WO_COLUMNS = (
    "PROJ_NO", "BLK_NO", "WK_ORD_NO", "GYEL", "LTH", "BTH", "THK",
    "CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY", "TACT_TIME", "PTLST_QTY",
)

# TACT_TIME = 0.3037*CUT_LTH + 0.1325*MARK_LTH + 0.4790*THK + 0.3840*PTLST_QTY
TACT_A_CUT = 0.3037
TACT_A_MARK = 0.1325
TACT_A_THK = 0.4790
TACT_A_PTLST = 0.3840


def _numeric_value(value: object, column: str, context: str) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        print(f"[ERROR][wo_job_contract._numeric_value] cause=non_numeric column={column} context={context} value={value}")
        raise RuntimeError(f"non-numeric {column}: {context}")
    return float(parsed)


def _positive_float(value: object, column: str, context: str) -> float:
    parsed = _numeric_value(value, column, context)
    if parsed <= 0:
        print(f"[ERROR][wo_job_contract._positive_float] cause=non_positive column={column} context={context} value={value}")
        raise RuntimeError(f"non-positive {column}: {context}")
    return parsed


def _non_negative_float(value: object, column: str, context: str) -> float:
    parsed = _numeric_value(value, column, context)
    if parsed < 0:
        print(f"[ERROR][wo_job_contract._non_negative_float] cause=negative column={column} context={context} value={value}")
        raise RuntimeError(f"negative {column}: {context}")
    return parsed


def _positive_int(value: object, column: str, context: str) -> int:
    parsed = _positive_float(value, column, context)
    if not float(parsed).is_integer():
        print(f"[ERROR][wo_job_contract._positive_int] cause=non_integer column={column} context={context} value={value}")
        raise RuntimeError(f"non-integer {column}: {context}")
    return int(parsed)


def _non_negative_int(value: object, column: str, context: str) -> int:
    parsed = _non_negative_float(value, column, context)
    if not float(parsed).is_integer():
        print(f"[ERROR][wo_job_contract._non_negative_int] cause=non_integer column={column} context={context} value={value}")
        raise RuntimeError(f"non-integer {column}: {context}")
    return int(parsed)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        print(f"[ERROR][wo_job_contract._require_columns] cause=missing_columns frame={frame_name} missing={missing}")
        raise RuntimeError(f"missing columns in {frame_name}: {missing}")


def _required_text(value: object, column: str, context: str) -> str:
    parsed = str(value or "").strip()
    if not parsed:
        print(f"[ERROR][wo_job_contract._required_text] cause=empty_text column={column} context={context} value={value}")
        raise RuntimeError(f"empty {column}: {context}")
    return parsed


def _job_field(job: object, field_name: str) -> object:
    if isinstance(job, Mapping):
        if field_name not in job:
            print(f"[ERROR][wo_job_contract._job_field] cause=missing_key field={field_name} job={job}")
            raise RuntimeError(f"generated job missing field: {field_name}")
        return job[field_name]
    if not hasattr(job, field_name):
        print(f"[ERROR][wo_job_contract._job_field] cause=missing_attr field={field_name} job={job}")
        raise RuntimeError(f"generated job missing field: {field_name}")
    return getattr(job, field_name)


def jobs_from_wo_df(wo_df: pd.DataFrame, episode_id: str) -> Dict[str, SimpleNamespace]:
    """생성한 W/O 행을 merged Phase 2가 쓰는 Job-like 객체로 변환한다."""

    _require_columns(wo_df, WO_COLUMNS, "wo_df")
    jobs: Dict[str, SimpleNamespace] = {}
    for _, row in wo_df.reset_index(drop=True).iterrows():
        job_id = f"{episode_id}_{str(row['WK_ORD_NO']).strip()}"
        block_set_id = build_block_set_id(row["PROJ_NO"], row["GYEL"], row["BLK_NO"])
        tact_time = _positive_float(row["TACT_TIME"], "TACT_TIME", job_id)
        jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            family=str(row["GYEL"]).strip(),
            block_set_id=block_set_id,
            steel_quantity=1,
            plate_length=_positive_float(row["LTH"], "LTH", job_id),
            plate_width=_positive_float(row["BTH"], "BTH", job_id),
            thickness=_positive_float(row["THK"], "THK", job_id),
            cut_length=_non_negative_float(row["CUT_LTH"], "CUT_LTH", job_id),
            marking_length=_non_negative_float(row["MARK_LTH"], "MARK_LTH", job_id),
            bevel_length=_non_negative_float(row["BVL_LTH"], "BVL_LTH", job_id),
            bevel_quantity=_non_negative_int(row["BV_QTY"], "BV_QTY", job_id),
            part_count=_positive_int(row["PTLST_QTY"], "PTLST_QTY", job_id),
            processing_time=tact_time,
            base_stage_minutes={"cut": tact_time},
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            extra={
                "source_project_no": str(row["PROJ_NO"]).strip(),
                "source_block_no": str(row["BLK_NO"]).strip(),
                "source_wk_ord_no": str(row["WK_ORD_NO"]).strip(),
                "data_role": "params_generator_synthetic_wo",
            },
        )
    if not jobs:
        print("[ERROR][wo_job_contract.jobs_from_wo_df] cause=no_jobs")
        raise RuntimeError("generated W/O data produced no jobs")
    return jobs


def scenario_jobs_from_jobs(jobs: Mapping[str, object]) -> List[Dict]:
    """생성 Job-like 객체를 scenario job dict 목록으로 변환한다(Phase 2 full-flow 계약)."""

    if not jobs:
        print("[ERROR][wo_job_contract.scenario_jobs_from_jobs] cause=no_jobs")
        raise RuntimeError("scenario conversion requires non-empty generated jobs")
    rows: List[Dict] = []
    for job_id, job in sorted(jobs.items()):
        rows.append(
            {
                "job_id": _required_text(_job_field(job, "job_id"), "job_id", str(job_id)),
                "family": _required_text(_job_field(job, "family"), "family", str(job_id)),
                "block_set_id": _required_text(_job_field(job, "block_set_id"), "block_set_id", str(job_id)),
                "steel_quantity": _positive_int(_job_field(job, "steel_quantity"), "steel_quantity", str(job_id)),
                "plate_length": _positive_float(_job_field(job, "plate_length"), "plate_length", str(job_id)),
                "plate_width": _positive_float(_job_field(job, "plate_width"), "plate_width", str(job_id)),
                "thickness": _positive_float(_job_field(job, "thickness"), "thickness", str(job_id)),
                "cut_length": _non_negative_float(_job_field(job, "cut_length"), "cut_length", str(job_id)),
                "marking_length": _non_negative_float(_job_field(job, "marking_length"), "marking_length", str(job_id)),
                "bevel_length": _non_negative_float(_job_field(job, "bevel_length"), "bevel_length", str(job_id)),
                "bevel_quantity": _non_negative_int(_job_field(job, "bevel_quantity"), "bevel_quantity", str(job_id)),
                "part_count": _positive_int(_job_field(job, "part_count"), "part_count", str(job_id)),
                "processing_time": _positive_float(_job_field(job, "processing_time"), "processing_time", str(job_id)),
                "base_stage_minutes": {
                    "cut": _positive_float(_job_field(job, "processing_time"), "processing_time", str(job_id))
                },
                "cut_bay": None,
                "source_cut_bay": None,
                "allowed_bay_ids": list(_job_field(job, "allowed_bay_ids")),
                "allowed_machine_ids": list(_job_field(job, "allowed_machine_ids")),
                "prohibited_machine_ids": list(_job_field(job, "prohibited_machine_ids")),
                "extra": dict(_job_field(job, "extra")),
            }
        )
    return rows


__all__ = [
    "WO_COLUMNS",
    "TACT_A_CUT", "TACT_A_MARK", "TACT_A_THK", "TACT_A_PTLST",
    "jobs_from_wo_df", "scenario_jobs_from_jobs",
]
