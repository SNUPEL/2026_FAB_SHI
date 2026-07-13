"""후공정 기준일에서 계열별 절단 착수일을 영업일로 계산한다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from Utils.data.multi_series_cutting_data import build_block_set_id


@dataclass(frozen=True)
class CuttingStartDateResult:
    series: str
    reference_date: date
    lead_working_days: int
    start_date: date
    reason_codes: tuple[str, ...]


def audit_cutting_start_dates(
    rows: Iterable[Mapping[str, object]],
    *,
    holiday_dates: Iterable[date | str | int] | None = None,
) -> pd.DataFrame:
    """원본 `ACT_ST_DT`와 산출 착수일을 분리 보존해 비교한다.

    이 함수는 산출일로 원본을 덮어쓰지 않는다. 현업 실적일에는 수식 외 조정이
    포함될 수 있으므로 두 날짜와 차이, 적용 규칙을 모두 audit 열로 남긴다.
    """

    row_list = list(rows)
    if not row_list:
        print("[ERROR][cutting_start_date.audit_cutting_start_dates] cause=no_rows")
        raise RuntimeError("start-date audit requires at least one row")
    resolved_holidays = _resolve_audit_holidays(row_list, holiday_dates)

    audit_rows = []
    for index, row in enumerate(row_list):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][cutting_start_date.audit_cutting_start_dates] "
                f"cause=invalid_row_type index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid start-date audit row: index={index}")
        source_date = _required_date(row, "ACT_ST_DT")
        calculated = calculate_cutting_start_date(row, holiday_dates=resolved_holidays)
        block_set_id = build_block_set_id(row.get("PROJ_NO"), calculated.series, row.get("BLK_NO"))
        audit_rows.append(
            {
                "BLOCK_SET_ID": block_set_id,
                "GYEL": calculated.series,
                "SOURCE_ACT_ST_DT": source_date.strftime("%Y%m%d"),
                "CALCULATED_ACT_ST_DT": calculated.start_date.strftime("%Y%m%d"),
                "MATCHED_SOURCE_DATE": source_date == calculated.start_date,
                "DELTA_CALENDAR_DAYS": (calculated.start_date - source_date).days,
                "REFERENCE_DATE": calculated.reference_date.strftime("%Y%m%d"),
                "LEAD_WORKING_DAYS": calculated.lead_working_days,
                "REASON_CODES": "|".join(calculated.reason_codes),
            }
        )
    result = pd.DataFrame(audit_rows)
    print(
        "[VALIDATION][cutting_start_date.audit_cutting_start_dates] "
        f"passed=true rows={len(result)} matched={int(result['MATCHED_SOURCE_DATE'].sum())}"
    )
    return result


def summarize_cutting_start_date_audit(audit: pd.DataFrame) -> dict:
    """착수일 audit의 전체 및 계열별 일치 count를 반환한다."""

    required = {"GYEL", "MATCHED_SOURCE_DATE"}
    if not isinstance(audit, pd.DataFrame) or audit.empty:
        print("[ERROR][cutting_start_date.summarize_cutting_start_date_audit] cause=empty_audit")
        raise RuntimeError("start-date audit summary requires non-empty audit rows")
    missing = sorted(required - set(audit.columns))
    if missing:
        print(
            "[ERROR][cutting_start_date.summarize_cutting_start_date_audit] "
            f"cause=missing_columns missing={missing}"
        )
        raise RuntimeError(f"missing start-date audit columns: {missing}")

    series_summary = {}
    for series, group in audit.groupby("GYEL", sort=True):
        matched_count = int(group["MATCHED_SOURCE_DATE"].astype(bool).sum())
        series_summary[str(series)] = {
            "row_count": int(len(group)),
            "matched_count": matched_count,
            "mismatched_count": int(len(group) - matched_count),
            "matched_ratio": round(matched_count / len(group), 6),
        }
    matched_count = int(audit["MATCHED_SOURCE_DATE"].astype(bool).sum())
    return {
        "row_count": int(len(audit)),
        "matched_count": matched_count,
        "mismatched_count": int(len(audit) - matched_count),
        "matched_ratio": round(matched_count / len(audit), 6),
        "series": series_summary,
    }


def write_cutting_start_date_audit(
    audit: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    """착수일 audit 원문 CSV와 동일 집계 JSON을 함께 저장한다."""

    summary = summarize_cutting_start_date_audit(audit)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "cutting_start_date_audit.csv"
    summary_path = output_path / "cutting_start_date_audit_summary.json"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[CHECK][cutting_start_date.write_cutting_start_date_audit] "
        f"rows={len(audit)} matched={summary['matched_count']} output={output_path}"
    )
    return {"csv": str(csv_path), "summary_json": str(summary_path)}


def calculate_cutting_start_date(
    row: Mapping[str, object],
    *,
    holiday_dates: Iterable[date | str | int] | None = None,
) -> CuttingStartDateResult:
    """확정된 NP/FN/FL/NC 규칙으로 한 block의 절단 착수일을 계산한다."""

    series = _required_text(row, "GYEL").upper()
    block_no = _required_text(row, "BLK_NO")
    if series == "NP":
        reference, lead_days, reasons = _np_rule(row, block_no)
    elif series == "FN":
        reference, lead_days, reasons = _fn_rule(row, block_no)
    elif series == "FL":
        reference, lead_days, reasons = _fl_rule(row)
    elif series == "NC":
        reference, lead_days, reasons = _nc_rule(row, block_no)
    else:
        print(
            "[ERROR][cutting_start_date.calculate_cutting_start_date] "
            f"cause=unsupported_series series={series}"
        )
        raise RuntimeError(f"unsupported_series: {series}")

    holidays = (
        korean_public_holiday_dates({reference.year - 1, reference.year})
        if holiday_dates is None
        else {_parse_date(value, "holiday_dates") for value in holiday_dates}
    )

    start_date = subtract_working_days(reference, lead_days, holiday_dates=holidays)
    return CuttingStartDateResult(
        series=series,
        reference_date=reference,
        lead_working_days=lead_days,
        start_date=start_date,
        reason_codes=reasons,
    )


def korean_public_holiday_dates(years: Iterable[int]) -> set[date]:
    """`holidays` 패키지에서 한국 법정·대체 공휴일을 읽는다."""

    normalized_years = sorted({int(year) for year in years})
    if not normalized_years:
        print("[ERROR][cutting_start_date.korean_public_holiday_dates] cause=no_years")
        raise RuntimeError("Korean holiday calendar requires at least one year")
    try:
        import holidays
    except ImportError as exc:
        print(
            "[ERROR][cutting_start_date.korean_public_holiday_dates] "
            "cause=missing_holidays_package install='python -m pip install holidays'"
        )
        raise RuntimeError("holidays package is required for Korean working-day calculation") from exc
    calendar = holidays.KR(years=normalized_years)
    result = set(calendar.keys())
    if not result:
        print(
            "[ERROR][cutting_start_date.korean_public_holiday_dates] "
            f"cause=empty_calendar years={normalized_years}"
        )
        raise RuntimeError(f"empty Korean holiday calendar: {normalized_years}")
    return result


def _resolve_audit_holidays(
    rows: Iterable[Mapping[str, object]],
    holiday_dates: Iterable[date | str | int] | None,
) -> set[date]:
    if holiday_dates is not None:
        return {_parse_date(value, "holiday_dates") for value in holiday_dates}
    years: set[int] = set()
    date_fields = (
        "ACT_ST_DT",
        "ACT_ED_DT",
        "SASS_ACT_STDT",
        "ASS_ACT_STDT",
        "PAN_ST_DT",
    )
    for row in rows:
        for field in date_fields:
            value = row.get(field)
            if value is None or str(value).strip().lower() in {"", "nan", "nat"}:
                continue
            parsed = _parse_date(value, field)
            years.update({parsed.year - 1, parsed.year})
    if not years:
        print("[ERROR][cutting_start_date._resolve_audit_holidays] cause=no_date_years")
        raise RuntimeError("start-date audit could not resolve holiday calendar years")
    return korean_public_holiday_dates(years)


def subtract_working_days(
    reference_date: date,
    working_days: int,
    *,
    holiday_dates: Iterable[date] = (),
) -> date:
    """토·일요일과 명시적 휴일을 제외하고 과거 영업일을 센다."""

    if working_days < 0:
        print(
            "[ERROR][cutting_start_date.subtract_working_days] "
            f"cause=negative_working_days value={working_days}"
        )
        raise RuntimeError("working_days must be non-negative")
    holidays = set(holiday_dates)
    current = reference_date
    counted = 0
    while counted < working_days:
        current -= timedelta(days=1)
        if current.weekday() >= 5 or current in holidays:
            continue
        counted += 1
    return current


def _np_rule(row: Mapping[str, object], block_no: str) -> tuple[date, int, tuple[str, ...]]:
    if _required_text(row, "ASS_YN").upper() == "Y":
        return _required_date(row, "ASS_ACT_STDT"), 4, ("np_ass_yn_y",)
    if _is_cnt_block(block_no):
        return _required_date(row, "ASS_ACT_STDT"), 7, ("np_cnt_block",)
    lead_days = 3
    reasons = ["np_base_3"]
    if _required_number(row, "CUT_LTH") >= 2000:
        lead_days += 1
        reasons.append("cut_lth_ge_2000")
    if _required_number(row, "BV_QTY") >= 30:
        lead_days += 1
        reasons.append("bv_qty_ge_30")
    return _required_date(row, "SASS_ACT_STDT"), lead_days, tuple(reasons)


def _fn_rule(row: Mapping[str, object], block_no: str) -> tuple[date, int, tuple[str, ...]]:
    if _is_cnt_block(block_no):
        return _required_date(row, "ASS_ACT_STDT"), 7, ("fn_cnt_block",)
    lead_days = 3
    reasons = ["fn_base_3"]
    if _required_number(row, "BV_QTY") >= 1:
        lead_days += 1
        reasons.append("bv_qty_ge_1")
    if _required_number(row, "CURVE_QTY") >= 1:
        lead_days += 1
        reasons.append("curve_qty_ge_1")
    return _required_date(row, "ASS_ACT_STDT"), lead_days, tuple(reasons)


def _fl_rule(row: Mapping[str, object]) -> tuple[date, int, tuple[str, ...]]:
    lead_days = 3
    reasons = ["fl_base_3"]
    if _required_number(row, "PTLST_QTY") >= 13 or _required_number(row, "CUT_LTH") >= 450:
        lead_days += 1
        reasons.append("ptlst_ge_13_or_cut_ge_450")
    if _required_number(row, "BVL_LTH") >= 100:
        lead_days += 1
        reasons.append("bvl_lth_ge_100")
    return _required_date(row, "PAN_ST_DT"), lead_days, tuple(reasons)


def _nc_rule(row: Mapping[str, object], block_no: str) -> tuple[date, int, tuple[str, ...]]:
    reference = _required_date(row, "ASS_ACT_STDT")
    normalized_block = block_no.upper()
    if normalized_block.startswith("OVER_CURVE_BLK"):
        return reference, 15, ("nc_over_curve_block",)
    if normalized_block.startswith("CURVE_BLK"):
        return reference, 10, ("nc_curve_block",)
    curve_qty = _required_number(row, "CURVE_QTY")
    lead_days = 9 if curve_qty >= 10 else 8
    reasons = ["nc_curve_qty_ge_10" if curve_qty >= 10 else "nc_curve_qty_lt_10"]
    if _required_number(row, "BV_QTY") >= 1:
        lead_days += 2
        reasons.append("bv_qty_ge_1_add_2")
    return reference, lead_days, tuple(reasons)


def _is_cnt_block(block_no: str) -> bool:
    return block_no.upper().startswith("CNT_BLK")


def _required_date(row: Mapping[str, object], field: str) -> date:
    value = row.get(field)
    if value in (None, ""):
        print(
            "[ERROR][cutting_start_date._required_date] "
            f"cause=missing_required_date field={field} block={row.get('PROJ_NO')}::{row.get('GYEL')}::{row.get('BLK_NO')}"
        )
        raise RuntimeError(f"missing_required_date: {field}")
    try:
        return _parse_date(value, field)
    except RuntimeError:
        raise


def _parse_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        print(
            "[ERROR][cutting_start_date._parse_date] "
            f"cause=invalid_compact_date field={field} value={value}"
        )
        raise RuntimeError(f"invalid_compact_date: {field}") from exc


def _required_number(row: Mapping[str, object], field: str) -> float:
    value = row.get(field)
    if value in (None, ""):
        print(
            "[ERROR][cutting_start_date._required_number] "
            f"cause=missing_required_number field={field} block={row.get('PROJ_NO')}::{row.get('GYEL')}::{row.get('BLK_NO')}"
        )
        raise RuntimeError(f"missing_required_number: {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][cutting_start_date._required_number] "
            f"cause=invalid_required_number field={field} value={value}"
        )
        raise RuntimeError(f"invalid_required_number: {field}") from exc
    if parsed < 0:
        print(
            "[ERROR][cutting_start_date._required_number] "
            f"cause=negative_required_number field={field} value={value}"
        )
        raise RuntimeError(f"negative_required_number: {field}")
    return parsed


def _required_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][cutting_start_date._required_text] "
            f"cause=missing_required_text field={field}"
        )
        raise RuntimeError(f"missing_required_text: {field}")
    return text
