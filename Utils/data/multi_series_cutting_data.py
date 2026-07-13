"""신규 NP/FN/FL/NC 절단 block/W/O 데이터의 엄격한 입력 계약."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SUPPORTED_SERIES = ("NP", "FN", "FL", "NC")

COMMON_REQUIRED_COLUMNS = (
    "PROJ_NO",
    "BLK_NO",
    "ACT_ST_DT",
    "ACT_ED_DT",
    "RT_CUT_ST_DTM",
    "RT_CUT_ED_DTM",
    "GYEL",
    "LTH",
    "BTH",
    "THK",
    "WGT",
    "MARK_LTH",
    "CUT_LTH",
    "BVL_LTH",
    "PTLST_QTY",
    "STL_QTY",
    "BV_QTY",
    "CURVE_QTY",
    "ASS_YN",
    "CUT_BAY",
    "EQP_NM",
    "SASS_WKA",
    "ASS_WKA",
    "SASS_ACT_STDT",
    "ASS_ACT_STDT",
    "PAN_ST_DT",
)
BLOCK_REQUIRED_COLUMNS = COMMON_REQUIRED_COLUMNS
WO_REQUIRED_COLUMNS = ("WK_ORD_NO",) + COMMON_REQUIRED_COLUMNS

NUMERIC_COLUMNS = (
    "LTH",
    "BTH",
    "THK",
    "WGT",
    "MARK_LTH",
    "CUT_LTH",
    "BVL_LTH",
    "PTLST_QTY",
    "STL_QTY",
    "BV_QTY",
    "CURVE_QTY",
)
POSITIVE_NUMERIC_COLUMNS = ("LTH", "BTH", "THK", "CUT_LTH", "PTLST_QTY")
NON_NEGATIVE_NUMERIC_COLUMNS = tuple(
    column for column in NUMERIC_COLUMNS if column not in POSITIVE_NUMERIC_COLUMNS
)

# block 파일의 값이 W/O 집계와 같아야 하는 산출 계약이다.
AGGREGATE_SUM_COLUMNS = (
    "CUT_LTH",
    "BVL_LTH",
    "PTLST_QTY",
    "STL_QTY",
    "BV_QTY",
    "CURVE_QTY",
)
AGGREGATE_MAX_COLUMNS = ("LTH", "BTH", "THK")


@dataclass(frozen=True)
class MultiSeriesCuttingData:
    """검증 완료된 block 및 W/O 표."""

    blocks: pd.DataFrame
    work_orders: pd.DataFrame
    block_source: str = ""
    wo_source: str = ""


def build_block_set_id(project_no: object, series: object, block_no: object) -> str:
    """다계열 block 의사결정 단위 `PROJ_NO::GYEL::BLK_NO`를 만든다."""

    values = {
        "PROJ_NO": _required_text(project_no, "PROJ_NO", "block_set_id"),
        "GYEL": _required_text(series, "GYEL", "block_set_id"),
        "BLK_NO": _required_text(block_no, "BLK_NO", "block_set_id"),
    }
    if values["GYEL"] not in SUPPORTED_SERIES:
        print(
            "[ERROR][multi_series_cutting_data.build_block_set_id] "
            f"cause=unsupported_series series={values['GYEL']}"
        )
        raise RuntimeError(f"unsupported_series: {values['GYEL']}")
    return f"{values['PROJ_NO']}::{values['GYEL']}::{values['BLK_NO']}"


def load_multi_series_cutting_data(
    block_path: str | Path,
    wo_path: str | Path,
) -> MultiSeriesCuttingData:
    """신규 Excel/CSV 두 파일을 읽고 참조·집계 무결성을 검증한다."""

    block_source = Path(block_path)
    wo_source = Path(wo_path)
    blocks = _read_table(block_source, "block")
    work_orders = _read_table(wo_source, "wo")
    result = prepare_multi_series_cutting_data(
        blocks,
        work_orders,
        block_source=str(block_source),
        wo_source=str(wo_source),
    )
    print(
        "[VALIDATION][multi_series_cutting_data.load_multi_series_cutting_data] "
        f"passed=true blocks={len(result.blocks)} wos={len(result.work_orders)} "
        f"block_source={block_source} wo_source={wo_source}"
    )
    return result


def prepare_multi_series_cutting_data(
    blocks: pd.DataFrame,
    work_orders: pd.DataFrame,
    *,
    block_source: str = "block_dataframe",
    wo_source: str = "wo_dataframe",
) -> MultiSeriesCuttingData:
    """메모리 표에도 파일 로더와 동일한 strict 계약을 적용한다."""

    block_rows = _validate_table(blocks, BLOCK_REQUIRED_COLUMNS, block_source, "block")
    wo_rows = _validate_table(work_orders, WO_REQUIRED_COLUMNS, wo_source, "wo")
    _validate_unique(wo_rows, "WK_ORD_NO", wo_source)

    block_rows["BLOCK_SET_ID"] = _block_ids(block_rows)
    wo_rows["BLOCK_SET_ID"] = _block_ids(wo_rows)
    _validate_unique(block_rows, "BLOCK_SET_ID", block_source)
    _validate_block_references(block_rows, wo_rows)
    block_rows = _validate_block_aggregates(block_rows, wo_rows)

    # RET_QTY는 현업 확인상 모델 입력이 아니다. 원본에 있어도 반환 표에서는 제거한다.
    wo_rows = wo_rows.drop(columns=["RET_QTY"], errors="ignore")
    return MultiSeriesCuttingData(
        blocks=block_rows.reset_index(drop=True),
        work_orders=wo_rows.reset_index(drop=True),
        block_source=block_source,
        wo_source=wo_source,
    )


def _read_table(path: Path, table_name: str) -> pd.DataFrame:
    if not path.exists():
        print(
            "[ERROR][multi_series_cutting_data._read_table] "
            f"cause=missing_source_file table={table_name} path={path}"
        )
        raise RuntimeError(f"missing_source_file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    print(
        "[ERROR][multi_series_cutting_data._read_table] "
        f"cause=unsupported_source_suffix table={table_name} suffix={suffix}"
    )
    raise RuntimeError(f"unsupported_source_suffix: {suffix}")


def _validate_table(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    source: str,
    table_name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        print(
            "[ERROR][multi_series_cutting_data._validate_table] "
            f"cause=empty_table table={table_name} source={source}"
        )
        raise RuntimeError(f"empty_table: {table_name}")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        print(
            "[ERROR][multi_series_cutting_data._validate_table] "
            f"cause=missing_required_columns table={table_name} missing={missing} source={source}"
        )
        raise RuntimeError(f"missing_required_columns: {','.join(missing)}")

    result = frame.copy()
    text_columns = ["PROJ_NO", "BLK_NO", "GYEL", "ASS_YN", "CUT_BAY", "EQP_NM"]
    if table_name == "wo":
        text_columns.append("WK_ORD_NO")
    for column in text_columns:
        blank = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if bool(blank.any()):
            count = int(blank.sum())
            print(
                "[ERROR][multi_series_cutting_data._validate_table] "
                f"cause=blank_required_value table={table_name} column={column} count={count} source={source}"
            )
            raise RuntimeError(f"blank_required_value: {table_name}.{column}")
        result[column] = result[column].astype(str).str.strip()

    unknown_series = sorted(set(result["GYEL"]) - set(SUPPORTED_SERIES))
    if unknown_series:
        print(
            "[ERROR][multi_series_cutting_data._validate_table] "
            f"cause=unsupported_series table={table_name} values={unknown_series} source={source}"
        )
        raise RuntimeError(f"unsupported_series: {unknown_series}")

    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].isna()
        if bool(invalid.any()):
            print(
                "[ERROR][multi_series_cutting_data._validate_table] "
                f"cause=non_numeric_or_missing table={table_name} column={column} "
                f"count={int(invalid.sum())} source={source}"
            )
            raise RuntimeError(f"non_numeric_or_missing: {table_name}.{column}")
    for column in POSITIVE_NUMERIC_COLUMNS:
        if bool((result[column] <= 0).any()):
            print(
                "[ERROR][multi_series_cutting_data._validate_table] "
                f"cause=non_positive_value table={table_name} column={column} source={source}"
            )
            raise RuntimeError(f"non_positive_value: {table_name}.{column}")
    for column in NON_NEGATIVE_NUMERIC_COLUMNS:
        if bool((result[column] < 0).any()):
            print(
                "[ERROR][multi_series_cutting_data._validate_table] "
                f"cause=negative_value table={table_name} column={column} source={source}"
            )
            raise RuntimeError(f"negative_value: {table_name}.{column}")
    return result


def _block_ids(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: build_block_set_id(row["PROJ_NO"], row["GYEL"], row["BLK_NO"]),
        axis=1,
    )


def _validate_unique(frame: pd.DataFrame, column: str, source: str) -> None:
    duplicate = frame[column].duplicated(keep=False)
    if bool(duplicate.any()):
        examples = frame.loc[duplicate, column].astype(str).head(5).tolist()
        print(
            "[ERROR][multi_series_cutting_data._validate_unique] "
            f"cause=duplicate_key column={column} count={int(duplicate.sum())} "
            f"examples={examples} source={source}"
        )
        raise RuntimeError(f"duplicate_key: {column}")


def _validate_block_references(blocks: pd.DataFrame, work_orders: pd.DataFrame) -> None:
    block_ids = set(blocks["BLOCK_SET_ID"])
    wo_block_ids = set(work_orders["BLOCK_SET_ID"])
    missing_in_wo = sorted(block_ids - wo_block_ids)
    missing_in_blocks = sorted(wo_block_ids - block_ids)
    if missing_in_wo or missing_in_blocks:
        print(
            "[ERROR][multi_series_cutting_data._validate_block_references] "
            f"cause=block_reference_mismatch missing_in_wo={missing_in_wo[:5]} "
            f"missing_in_blocks={missing_in_blocks[:5]}"
        )
        raise RuntimeError("block_reference_mismatch")


def _validate_block_aggregates(blocks: pd.DataFrame, work_orders: pd.DataFrame) -> pd.DataFrame:
    grouped = work_orders.groupby("BLOCK_SET_ID", sort=False)
    aggregate_spec = {column: "sum" for column in AGGREGATE_SUM_COLUMNS}
    aggregate_spec.update({column: "max" for column in AGGREGATE_MAX_COLUMNS})
    aggregate_spec["WK_ORD_NO"] = "count"
    aggregated = grouped.agg(aggregate_spec).rename(columns={"WK_ORD_NO": "WO_QTY"})

    result = blocks.copy().set_index("BLOCK_SET_ID", drop=False)
    for block_id, block_row in result.iterrows():
        wo_row = aggregated.loc[block_id]
        for column in AGGREGATE_SUM_COLUMNS + AGGREGATE_MAX_COLUMNS:
            actual = float(block_row[column])
            expected = float(wo_row[column])
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
                print(
                    "[ERROR][multi_series_cutting_data._validate_block_aggregates] "
                    f"cause=block_wo_aggregate_mismatch block_set_id={block_id} "
                    f"column={column} block_value={actual} wo_value={expected}"
                )
                raise RuntimeError(f"block_wo_aggregate_mismatch: {block_id}:{column}")
        result.at[block_id, "WO_QTY"] = int(wo_row["WO_QTY"])
    result["WO_QTY"] = result["WO_QTY"].astype(int)
    return result.reset_index(drop=True)


def _required_text(value: object, field: str, context: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        print(
            "[ERROR][multi_series_cutting_data._required_text] "
            f"cause=blank_required_value field={field} context={context}"
        )
        raise RuntimeError(f"blank_required_value: {field}")
    return text
