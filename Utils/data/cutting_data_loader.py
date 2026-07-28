"""절단 실적 Excel/CSV 로딩 및 1차 전처리.

이 파일의 책임:
- 현업 Excel/CSV 파일을 row dict로 읽는다.
- 한글/영문 컬럼명을 내부 canonical key로 맞춘다.
- NP 계열, 실적 착수/종료시간, duration 등 1차 사용 가능 row를 선별한다.
- 제외된 row는 삭제하지 않고 `excluded_records`에 사유와 함께 남긴다.

중요한 데이터 기준:
- `RT_CUT_ST_DTM`, `RT_CUT_ED_DTM`은 `YYYYMMDDHHMM` compact datetime이다.
- actual replay 검증 시간은 `실적 종료시간 - 실적 착수시간`으로 계산한다.
- `TACT_TIME`은 처리시간 후보일 뿐 actual replay identity 검증값이 아니다.
"""

# LINE-BY-LINE: `__future__` 모듈에서 `annotations`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from __future__ import annotations

# LINE-BY-LINE: `csv` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import csv
# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass
# LINE-BY-LINE: `datetime` 모듈에서 `datetime`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from datetime import datetime
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Optional, Sequence`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Optional, Sequence

# LINE-BY-LINE: `openpyxl` 모듈에서 `load_workbook`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from openpyxl import load_workbook

from Utils.data.multi_series_cutting_data import build_block_set_id


# LINE-BY-LINE: `COLUMN_ALIASES` 변수에 `{` 결과를 저장합니다. 의미: `COLUMN_ALIASES` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `("PROJ_NO", "호선번호")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
    "project_no": ("PROJ_NO", "호선번호"),
    # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `("BLK_NO", "블록명")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
    "block_no": ("BLK_NO", "블록명"),
    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `("WK_ORD_NO", "W/O 명", "WO 명", "작업오더")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
    "work_order_no": ("WK_ORD_NO", "W/O 명", "WO 명", "작업오더"),
    # LINE-BY-LINE: 딕셔너리 키 `planned_start_date`에는 `("GYEL_ACT_STDT", "계획 착수일")` 값을 넣습니다. 의미: 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
    "planned_start_date": ("ACT_ST_DT", "GYEL_ACT_STDT", "계획 착수일"),
    # LINE-BY-LINE: 딕셔너리 키 `planned_end_date`에는 `("GYEL_ACT_EDDT", "계획 종료일")` 값을 넣습니다. 의미: 계획 종료일 원본입니다. 예: `GYEL_ACT_EDDT`, 사용: 계획일 기반 검증 후보.
    "planned_end_date": ("ACT_ED_DT", "GYEL_ACT_EDDT", "계획 종료일"),
    # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `("RT_CUT_ST_DTM", "실적 착수시간")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
    "actual_start_datetime": ("RT_CUT_ST_DTM", "실적 착수시간", "실적 착수시간(기계가 움직임)"),
    # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `("RT_CUT_ED_DTM", "실적 종료시간")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
    "actual_end_datetime": ("RT_CUT_ED_DTM", "실적 종료시간", "실적 종료시간(기계가 멈춤)"),
    # LINE-BY-LINE: 딕셔너리 키 `series`에는 `("GYEL", "계열")` 값을 넣습니다. 의미: 원본 계열 값입니다. 예: `NP`, 사용: NP만 필터링할 때 비교.
    "series": ("GYEL", "계열"),
    # LINE-BY-LINE: 딕셔너리 키 `length`에는 `("LTH", "길이")` 값을 넣습니다. 의미: 원본 길이 LTH입니다. 사용: Job.plate_length로 변환되어 capacity 제약에 들어갑니다.
    "length": ("LTH", "길이"),
    # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `("THK", "두께")` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
    "thickness": ("THK", "두께"),
    # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `("CUT_LTH", "절단 길이")` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
    "cut_length": ("CUT_LTH", "절단 길이"),
    # LINE-BY-LINE: 딕셔너리 키 `mark_length`에는 `("MARK_LTH", "마킹 길이")` 값을 넣습니다. 의미: 마킹 길이입니다. 사용: tact time 분석 feature.
    "mark_length": ("MARK_LTH", "마킹 길이"),
    # LINE-BY-LINE: 딕셔너리 키 `bevel_length`에는 `("BVL_LTH", "베벨 길이")` 값을 넣습니다. 의미: 베벨 길이입니다. 사용: tact time 분석 feature.
    "bevel_length": ("BVL_LTH", "베벨 길이"),
    "bevel_qty": ("BV_QTY", "개선 수", "베벨 수량"),
    # LINE-BY-LINE: 딕셔너리 키 `steel_qty`에는 `("STL_QTY", "강재 수량")` 값을 넣습니다. 의미: 원본 강재 수량입니다. 예: `STL_QTY`, 사용: steel_quantity로 변환.
    "steel_qty": ("STL_QTY", "강재 수량"),
    # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `("RT_EQP_NM", "장비명")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
    "source_machine_id": ("EQP_NM", "RT_EQP_NM", "장비명"),
    # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `("CUT_BAY", "절단 베이")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
    "source_cut_bay": ("CUT_BAY", "절단 베이"),
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `tact_time` 키에 `("TACT_TIME", "택트타임")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "tact_time": ("TACT_TIME", "택트타임", "택트타임(아크로 절단 시간): 실제값"),
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `downstream_date` 키에 `("ASS_ST_DT", "후공정 소요일")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "downstream_date": ("ASS_ST_DT", "후공정 소요일"),
    # LINE-BY-LINE: 딕셔너리 키 `part_qty`에는 `("PTLST_QTY", "부재 수량")` 값을 넣습니다. 의미: 원본 부재 수량입니다. 예: `PTLST_QTY`, 사용: part_count로 변환.
    "part_qty": ("PTLST_QTY", "부재 수량"),
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `stl_req_date` 키에 `("STL_REQ_DT", "가공 요청일")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "stl_req_date": ("STL_REQ_DT", "가공 요청일"),
    # LINE-BY-LINE: `현재 dict`에서 반환/저장할 dict의 `wk_seq` 키에 `("WK_SEQ", "작업 순서")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
    "wk_seq": ("WK_SEQ", "작업 순서"),
}


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `CleanedCuttingData` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 프로젝트의 해당 모듈에서 재사용됩니다.
class CleanedCuttingData:
    """전처리 결과와 제외 로그.

    records:
      스케줄링/scenario 생성에 사용할 row 목록
    excluded_records:
      제외된 row 목록. `exclude_reason` 필드를 포함한다.
    source_path:
      원본 파일 경로. report/debug용이다.
    """

    # LINE-BY-LINE: `records`를 `List[Dict[str, Any]]` 타입으로 선언합니다. 의미/사용: `CleanedCuttingData.records` 필드/속성입니다. 사용: CleanedCuttingData 객체를 만들거나 이후 로직에서 참조합니다.
    records: List[Dict[str, Any]]
    # LINE-BY-LINE: `excluded_records`를 `List[Dict[str, Any]]` 타입으로 선언합니다. 의미/사용: `CleanedCuttingData.excluded_records` 필드/속성입니다. 사용: CleanedCuttingData 객체를 만들거나 이후 로직에서 참조합니다.
    excluded_records: List[Dict[str, Any]]
    # LINE-BY-LINE: `source_path`를 `str` 타입으로 선언합니다. 의미/사용: `CleanedCuttingData.source_path` 필드/속성입니다. 사용: CleanedCuttingData 객체를 만들거나 이후 로직에서 참조합니다.
    source_path: str


# LINE-BY-LINE: `_stringify_header(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _stringify_header(value: Any) -> str:
    """Excel header cell을 비교 가능한 문자열로 정규화한다."""

    # LINE-BY-LINE: 호출자에게 `str(value).strip() if value is not None else ""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return str(value).strip() if value is not None else ""


# LINE-BY-LINE: `_read_xlsx(path: Path, sheet_name: str = "Sheet")` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
def _read_xlsx(path: Path, sheet_name: str = "Sheet") -> List[Dict[str, Any]]:
    """xlsx/xlsm 파일을 첫 row header 기반 dict 목록으로 읽는다."""

    # LINE-BY-LINE: `workbook`에 `load_workbook(path, read_only=True, data_only=True)` 결과를 저장합니다. 의미/사용: `workbook`는 openpyxl workbook 객체입니다. Excel 파일의 sheet 목록과 cell 값을 읽습니다.
    workbook = load_workbook(path, read_only=True, data_only=True)
    # LINE-BY-LINE: `worksheet`에 `workbook[sheet_name] if sheet_name in workbook.sheetnames else workbook[workbook.sheetnames[0]]` 결과를 저장합니다. 의미/사용: `worksheet`는 Excel sheet 객체입니다. iter_rows로 row 데이터를 추출합니다.
    worksheet = workbook[sheet_name] if sheet_name in workbook.sheetnames else workbook[workbook.sheetnames[0]]
    # LINE-BY-LINE: `rows`에 `list(worksheet.iter_rows(values_only=True))` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(worksheet.iter_rows(values_only=True))
    # LINE-BY-LINE: 조건 `not rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows:
        # LINE-BY-LINE: 호출자에게 `[]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return []
    # LINE-BY-LINE: `header`에 `[_stringify_header(value) for value in rows[0]]` 결과를 저장합니다. 의미/사용: `header`는 CSV/Excel 첫 줄의 컬럼명 목록입니다. dict row의 key로 사용됩니다.
    header = [_stringify_header(value) for value in rows[0]]
    # LINE-BY-LINE: 호출자에게 `[dict(zip(header, row)) for row in rows[1:]]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [dict(zip(header, row)) for row in rows[1:]]


# LINE-BY-LINE: `_read_csv(path: Path)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
def _read_csv(path: Path) -> List[Dict[str, Any]]:
    """utf-8-sig CSV 파일을 dict row 목록으로 읽는다."""

    # LINE-BY-LINE: `path.open("r", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: 호출자에게 `list(csv.DictReader(handle))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return list(csv.DictReader(handle))


# LINE-BY-LINE: `load_raw_cutting_rows(path: str, sheet_name: str = "Sheet")` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def load_raw_cutting_rows(path: str, sheet_name: str = "Sheet") -> List[Dict[str, Any]]:
    """Excel 또는 CSV를 raw row dict 목록으로 읽는다."""

    # LINE-BY-LINE: `source_path`에 `Path(path)` 결과를 저장합니다. 의미/사용: `source_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    source_path = Path(path)
    # LINE-BY-LINE: `suffix`에 `source_path.suffix.lower()` 결과를 저장합니다. 의미/사용: `suffix` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    suffix = source_path.suffix.lower()
    # LINE-BY-LINE: 조건 `suffix in {".xlsx", ".xlsm"}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if suffix in {".xlsx", ".xlsm"}:
        # LINE-BY-LINE: 호출자에게 `_read_xlsx(source_path, sheet_name=sheet_name)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _read_xlsx(source_path, sheet_name=sheet_name)
    # LINE-BY-LINE: 조건 `suffix == ".csv"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if suffix == ".csv":
        # LINE-BY-LINE: 호출자에게 `_read_csv(source_path)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return _read_csv(source_path)
    # LINE-BY-LINE: `ValueError(f"unsupported cutting data file: {path}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise ValueError(f"unsupported cutting data file: {path}")


# LINE-BY-LINE: `_resolve_column(row: Dict[str, Any], canonical_name: str)` 함수를 정의합니다. 반환 타입: `Any`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _resolve_column(row: Dict[str, Any], canonical_name: str) -> Any:
    """canonical key에 대응하는 한글/영문 alias 컬럼 값을 찾는다.

    예:
    - canonical_name=`actual_start_datetime`
    - 허용 alias=`RT_CUT_ST_DTM`, `실적 착수시간`
    """

    # LINE-BY-LINE: `alias in COLUMN_ALIASES[canonical_name]` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for alias in COLUMN_ALIASES[canonical_name]:
        # LINE-BY-LINE: 조건 `alias in row`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if alias in row:
            # LINE-BY-LINE: 호출자에게 `row.get(alias)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return row.get(alias)
    # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return None


# LINE-BY-LINE: `standardize_cutting_row(row: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def standardize_cutting_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """한글/영문 컬럼을 canonical key로 통일한다."""

    # LINE-BY-LINE: `standardized`에 `{canonical: _resolve_column(row, canonical) for canonical in COLUMN_ALIASES}` 결과를 저장합니다. 의미/사용: `standardized` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    standardized = {canonical: _resolve_column(row, canonical) for canonical in COLUMN_ALIASES}
    # LINE-BY-LINE: `standardized["_raw"]`에 `row` 결과를 저장합니다. 의미/사용: `standardized["_raw"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    standardized["_raw"] = row
    # LINE-BY-LINE: 호출자에게 `standardized`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return standardized


# LINE-BY-LINE: `standardize_cutting_rows(rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def standardize_cutting_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """여러 raw row를 canonical key row 목록으로 변환한다."""

    # LINE-BY-LINE: 호출자에게 `[standardize_cutting_row(row) for row in rows]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [standardize_cutting_row(row) for row in rows]


# LINE-BY-LINE: `parse_compact_datetime(value: Any)` 함수를 정의합니다. 반환 타입: `Optional[datetime]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def parse_compact_datetime(value: Any) -> Optional[datetime]:
    """YYYYMMDD 또는 YYYYMMDDHHMM 값을 datetime으로 변환한다.

    반환:
    - 정상 파싱: `datetime`
    - 비어 있거나 형식 오류: None

    여기서는 None을 반환하고, 실제 제외/오류 판단은 caller가 수행한다.
    """

    # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value in (None, ""):
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `text`에 `str(value).strip()` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    text = str(value).strip()
    # LINE-BY-LINE: 조건 `text.endswith(".0")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if text.endswith(".0"):
        # LINE-BY-LINE: `text`에 `text[:-2]` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        text = text[:-2]
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 조건 `len(text) == 8 and text.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(text) == 8 and text.isdigit():
            # LINE-BY-LINE: 호출자에게 `datetime.strptime(text, "%Y%m%d")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return datetime.strptime(text, "%Y%m%d")
        # LINE-BY-LINE: 조건 `len(text) == 12 and text.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(text) == 12 and text.isdigit():
            # LINE-BY-LINE: 호출자에게 `datetime.strptime(text, "%Y%m%d%H%M")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return datetime.strptime(text, "%Y%m%d%H%M")
    # LINE-BY-LINE: `except ValueError:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except ValueError:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return None


# LINE-BY-LINE: `_to_float(value: Any, default: float = 0.0)` 함수를 정의합니다. 반환 타입: `float`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _to_float(value: Any, default: float = 0.0) -> float:
    """숫자 값을 float로 변환한다.

    주의:
    - 이 helper는 기존 전처리 호환을 위해 default를 지원한다.
    - 핵심 검증용 parser에서는 조용한 default 대체를 줄여야 한다.
    - 향후 `strict_to_float`로 교체하는 것이 남은 개선 항목이다.
    """

    # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value in (None, ""):
        # LINE-BY-LINE: 호출자에게 `default`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return default
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 호출자에게 `float(value)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return float(value)
    # LINE-BY-LINE: `except (TypeError, ValueError):` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except (TypeError, ValueError):
        # LINE-BY-LINE: 호출자에게 `default`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return default


# LINE-BY-LINE: `_to_int(value: Any, default: int = 0)` 함수를 정의합니다. 반환 타입: `int`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _to_int(value: Any, default: int = 0) -> int:
    """숫자 값을 int로 변환한다.

    `_to_float()`와 같은 이유로 현재는 legacy default를 지원한다.
    """

    # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value in (None, ""):
        # LINE-BY-LINE: 호출자에게 `default`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return default
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 호출자에게 `int(float(value))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return int(float(value))
    # LINE-BY-LINE: `except (TypeError, ValueError):` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except (TypeError, ValueError):
        # LINE-BY-LINE: 호출자에게 `default`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return default


# LINE-BY-LINE: `_exclude(row: Dict[str, Any], reason: str)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _exclude(row: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """제외 row에 제외 사유를 붙여 로그용 dict로 만든다."""

    # LINE-BY-LINE: `excluded`에 `dict(row)` 결과를 저장합니다. 의미/사용: `excluded` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    excluded = dict(row)
    # LINE-BY-LINE: `excluded["exclude_reason"]`에 `reason` 결과를 저장합니다. 의미/사용: `excluded["exclude_reason"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    excluded["exclude_reason"] = reason
    # LINE-BY-LINE: `excluded.pop("_raw", None)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    excluded.pop("_raw", None)
    # LINE-BY-LINE: 호출자에게 `excluded`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return excluded


# LINE-BY-LINE: `clean_cutting_records` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def clean_cutting_records(
    # LINE-BY-LINE: `rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `target_series` 변수에 `("NP",)` 결과를 저장합니다. 의미: `target_series` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_series: Sequence[str] = ("NP",),
    # LINE-BY-LINE: `max_records` 변수에 `None` 결과를 저장합니다. 의미: `max_records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_records: Optional[int] = None,
    # LINE-BY-LINE: `require_positive_elapsed` 변수에 `True` 결과를 저장합니다. 의미: `require_positive_elapsed` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    require_positive_elapsed: bool = True,
    # LINE-BY-LINE: `max_elapsed_minutes` 변수에 `24 * 60` 결과를 저장합니다. 의미: `max_elapsed_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_elapsed_minutes: Optional[float] = 24 * 60,
    # LINE-BY-LINE: `remove_missing_stl_req_dt_if_exists` 변수에 `True` 결과를 저장합니다. 의미: `remove_missing_stl_req_dt_if_exists` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    remove_missing_stl_req_dt_if_exists: bool = True,
# LINE-BY-LINE: `) -> CleanedCuttingData:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> CleanedCuttingData:
    """스케줄링 입력으로 쓸 row와 제외 로그를 만든다.

    입력:
    - rows: Excel/CSV에서 읽은 raw row 목록
    - target_series: 사용할 계열. 1차는 NP만 사용한다.
    - max_records: 100건 샘플 같은 제한이 필요할 때 사용한다.
    - require_positive_elapsed: actual duration이 0 이하인 row 제외 여부
    - max_elapsed_minutes: 지나치게 긴 duration 제외 기준

    출력:
    - `CleanedCuttingData(records, excluded_records, source_path)`

    제외는 데이터 삭제가 아니라 분석 로그다.
    그래서 excluded_records를 반드시 파일로 남겨야 한다.
    """

    # LINE-BY-LINE: `standardized_rows`에 `standardize_cutting_rows(rows)` 결과를 저장합니다. 의미/사용: `standardized_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    standardized_rows = standardize_cutting_rows(rows)
    # LINE-BY-LINE: `selected` 변수에 `[]` 결과를 저장합니다. 의미: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
    selected: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `excluded` 변수에 `[]` 결과를 저장합니다. 의미: `excluded` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    excluded: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `target_set`에 `{str(item) for item in target_series}` 결과를 저장합니다. 의미/사용: `target_set` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_set = {str(item) for item in target_series}

    # LINE-BY-LINE: `row_index, row in enumerate(standardized_rows, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row_index, row in enumerate(standardized_rows, start=1):
        # 원본 row 번호를 보존한다.
        # 나중에 오류가 나면 Excel 몇 번째 row인지 되짚기 위해 필요하다.
        # LINE-BY-LINE: `row["source_row_index"]`에 `row_index` 결과를 저장합니다. 의미/사용: `row["source_row_index"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        row["source_row_index"] = row_index
        # LINE-BY-LINE: `series`에 `str(row.get("series") or "").strip()` 결과를 저장합니다. 의미/사용: `series`는 원본 계열 값입니다. 예: `NP`, 사용: NP만 필터링할 때 비교.
        series = str(row.get("series") or "").strip()
        # 현재 1차 알고리즘은 NP만 대상으로 한다.
        # 파일명은 NP 물량이지만 실제로 FL 등이 섞여 있으므로 여기서 제외 로그를 남긴다.
        # LINE-BY-LINE: 조건 `series not in target_set`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if series not in target_set:
            # LINE-BY-LINE: `excluded.append(_exclude(row, f"series_not_in_target:{series}"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, f"series_not_in_target:{series}"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # LINE-BY-LINE: `has_stl_req_date_column`에 `any(alias in row["_raw"] for alias in COLUMN_ALIASES["stl_req_date"])` 결과를 저장합니다. 의미/사용: `has_stl_req_date_column` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        has_stl_req_date_column = any(alias in row["_raw"] for alias in COLUMN_ALIASES["stl_req_date"])
        # LINE-BY-LINE: 조건 `remove_missing_stl_req_dt_if_exists and has_stl_req_date_column and row.get("stl_req_date") in (N...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if remove_missing_stl_req_dt_if_exists and has_stl_req_date_column and row.get("stl_req_date") in (None, ""):
            # LINE-BY-LINE: `excluded.append(_exclude(row, "missing_stl_req_date"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, "missing_stl_req_date"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # actual replay identity 검증은 이 두 시각을 그대로 써야 한다.
        # TACT_TIME과 다르더라도 여기서 맞추거나 보정하지 않는다.
        # LINE-BY-LINE: `actual_start`에 `parse_compact_datetime(row.get("actual_start_datetime"))` 결과를 저장합니다. 의미/사용: `actual_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start = parse_compact_datetime(row.get("actual_start_datetime"))
        # LINE-BY-LINE: `actual_end`에 `parse_compact_datetime(row.get("actual_end_datetime"))` 결과를 저장합니다. 의미/사용: `actual_end` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_end = parse_compact_datetime(row.get("actual_end_datetime"))
        # LINE-BY-LINE: 조건 `actual_start is None or actual_end is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_start is None or actual_end is None:
            # LINE-BY-LINE: `excluded.append(_exclude(row, "invalid_actual_datetime"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, "invalid_actual_datetime"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # LINE-BY-LINE: `actual_duration`에 `(actual_end - actual_start).total_seconds() / 60.0` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = (actual_end - actual_start).total_seconds() / 60.0
        # LINE-BY-LINE: 조건 `actual_duration < 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_duration < 0:
            # LINE-BY-LINE: `excluded.append(_exclude(row, "negative_actual_duration"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, "negative_actual_duration"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: 조건 `require_positive_elapsed and actual_duration <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if require_positive_elapsed and actual_duration <= 0:
            # LINE-BY-LINE: `excluded.append(_exclude(row, "non_positive_actual_duration"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, "non_positive_actual_duration"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: 조건 `max_elapsed_minutes is not None and actual_duration > max_elapsed_minutes`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if max_elapsed_minutes is not None and actual_duration > max_elapsed_minutes:
            # LINE-BY-LINE: `excluded.append(_exclude(row, "actual_duration_over_limit"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            excluded.append(_exclude(row, "actual_duration_over_limit"))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # 여기부터는 scenario 생성에 바로 쓸 수 있게 타입을 맞춘다.
        # LINE-BY-LINE: `cleaned`에 `dict(row)` 결과를 저장합니다. 의미/사용: `cleaned` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned = dict(row)
        # LINE-BY-LINE: `cleaned.pop("_raw", None)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        cleaned.pop("_raw", None)
        # LINE-BY-LINE: `cleaned["project_no"]`에 `str(cleaned.get("project_no"))` 결과를 저장합니다. 의미/사용: `cleaned["project_no"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["project_no"] = str(cleaned.get("project_no"))
        # LINE-BY-LINE: `cleaned["block_no"]`에 `str(cleaned.get("block_no"))` 결과를 저장합니다. 의미/사용: `cleaned["block_no"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["block_no"] = str(cleaned.get("block_no"))
        # LINE-BY-LINE: `cleaned["work_order_no"]`에 `str(cleaned.get("work_order_no"))` 결과를 저장합니다. 의미/사용: `cleaned["work_order_no"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["work_order_no"] = str(cleaned.get("work_order_no"))
        # LINE-BY-LINE: `cleaned["series"]`에 `series` 결과를 저장합니다. 의미/사용: `cleaned["series"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["series"] = series
        # LINE-BY-LINE: `cleaned["source_machine_id"]`에 `str(cleaned.get("source_machine_id"))` 결과를 저장합니다. 의미/사용: `cleaned["source_machine_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["source_machine_id"] = str(cleaned.get("source_machine_id"))
        # LINE-BY-LINE: `cleaned["source_cut_bay"]`에 `str(cleaned.get("source_cut_bay"))` 결과를 저장합니다. 의미/사용: `cleaned["source_cut_bay"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["source_cut_bay"] = str(cleaned.get("source_cut_bay"))
        # LINE-BY-LINE: `cleaned["length"]`에 `_to_float(cleaned.get("length"))` 결과를 저장합니다. 의미/사용: `cleaned["length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["length"] = _to_float(cleaned.get("length"))
        # LINE-BY-LINE: `cleaned["thickness"]`에 `_to_float(cleaned.get("thickness"))` 결과를 저장합니다. 의미/사용: `cleaned["thickness"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["thickness"] = _to_float(cleaned.get("thickness"))
        # LINE-BY-LINE: `cleaned["cut_length"]`에 `_to_float(cleaned.get("cut_length"))` 결과를 저장합니다. 의미/사용: `cleaned["cut_length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["cut_length"] = _to_float(cleaned.get("cut_length"))
        # LINE-BY-LINE: `cleaned["mark_length"]`에 `_to_float(cleaned.get("mark_length"))` 결과를 저장합니다. 의미/사용: `cleaned["mark_length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["mark_length"] = _to_float(cleaned.get("mark_length"))
        # LINE-BY-LINE: `cleaned["bevel_length"]`에 `_to_float(cleaned.get("bevel_length"))` 결과를 저장합니다. 의미/사용: `cleaned["bevel_length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["bevel_length"] = _to_float(cleaned.get("bevel_length"))
        if cleaned.get("bevel_qty") in (None, ""):
            cleaned["bevel_qty"] = None
        else:
            try:
                cleaned["bevel_qty"] = int(float(cleaned.get("bevel_qty")))
            except (TypeError, ValueError):
                excluded.append(_exclude(row, "invalid_bevel_qty"))
                continue
            if cleaned["bevel_qty"] < 0:
                excluded.append(_exclude(row, "negative_bevel_qty"))
                continue
        # LINE-BY-LINE: 조건 `cleaned.get("steel_qty") in (None, "")`를 검사합니다. 참이면 강재 수량 누락 row를 제외합니다.
        if cleaned.get("steel_qty") in (None, ""):
            excluded.append(_exclude(row, "missing_steel_qty"))
            continue
        try:
            cleaned["steel_qty"] = int(float(cleaned.get("steel_qty")))
        except (TypeError, ValueError):
            excluded.append(_exclude(row, "invalid_steel_qty"))
            continue
        if cleaned["steel_qty"] <= 0:
            excluded.append(_exclude(row, "non_positive_steel_qty"))
            continue
        # LINE-BY-LINE: `cleaned["part_qty"]`에 `_to_int(cleaned.get("part_qty"), default=0)` 결과를 저장합니다. 의미/사용: `cleaned["part_qty"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["part_qty"] = _to_int(cleaned.get("part_qty"), default=0)
        # LINE-BY-LINE: `cleaned["tact_time"]`에 `_to_float(cleaned.get("tact_time"))` 결과를 저장합니다. 의미/사용: `cleaned["tact_time"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["tact_time"] = _to_float(cleaned.get("tact_time"))
        # LINE-BY-LINE: `cleaned["actual_duration_minutes"]`에 `round(actual_duration, 6)` 결과를 저장합니다. 의미/사용: `cleaned["actual_duration_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["actual_duration_minutes"] = round(actual_duration, 6)
        # 다계열 데이터에서는 같은 프로젝트/블록도 계열별로 서로 다른 Bay 의사결정 단위다.
        cleaned["block_set_id"] = build_block_set_id(
            cleaned["project_no"], cleaned["series"], cleaned["block_no"]
        )
        # LINE-BY-LINE: `cleaned["_actual_start_dt"]`에 `actual_start` 결과를 저장합니다. 의미/사용: `cleaned["_actual_start_dt"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["_actual_start_dt"] = actual_start
        # LINE-BY-LINE: `cleaned["_actual_end_dt"]`에 `actual_end` 결과를 저장합니다. 의미/사용: `cleaned["_actual_end_dt"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["_actual_end_dt"] = actual_end
        # LINE-BY-LINE: `cleaned["_planned_start_dt"]`에 `parse_compact_datetime(cleaned.get("planned_start_date"))` 결과를 저장합니다. 의미/사용: `cleaned["_planned_start_dt"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["_planned_start_dt"] = parse_compact_datetime(cleaned.get("planned_start_date"))
        # LINE-BY-LINE: `cleaned["_planned_end_dt"]`에 `parse_compact_datetime(cleaned.get("planned_end_date"))` 결과를 저장합니다. 의미/사용: `cleaned["_planned_end_dt"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["_planned_end_dt"] = parse_compact_datetime(cleaned.get("planned_end_date"))
        # LINE-BY-LINE: `cleaned["_downstream_dt"]`에 `parse_compact_datetime(cleaned.get("downstream_date"))` 결과를 저장합니다. 의미/사용: `cleaned["_downstream_dt"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cleaned["_downstream_dt"] = parse_compact_datetime(cleaned.get("downstream_date"))
        # LINE-BY-LINE: `selected.append(cleaned)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        selected.append(cleaned)

        # LINE-BY-LINE: 조건 `max_records is not None and len(selected) >= max_records`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if max_records is not None and len(selected) >= max_records:
            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            break

    # simulation 내부 시간은 "첫 계획 착수일/첫 실적 착수시각으로부터 몇 분"으로 표현한다.
    # actual replay는 actual_base_dt를 기준으로 start/finish minutes를 만든다.
    # LINE-BY-LINE: `planned_starts`에 `[row["_planned_start_dt"] for row in selected if row.get("_planned_start_dt") is not None]` 결과를 저장합니다. 의미/사용: `planned_starts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    planned_starts = [row["_planned_start_dt"] for row in selected if row.get("_planned_start_dt") is not None]
    # LINE-BY-LINE: `base_dt`에 `min(planned_starts) if planned_starts else None` 결과를 저장합니다. 의미/사용: `base_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    base_dt = min(planned_starts) if planned_starts else None
    # LINE-BY-LINE: `actual_starts`에 `[row["_actual_start_dt"] for row in selected if row.get("_actual_start_dt") is not None]` 결과를 저장합니다. 의미/사용: `actual_starts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_starts = [row["_actual_start_dt"] for row in selected if row.get("_actual_start_dt") is not None]
    # LINE-BY-LINE: `actual_base_dt`에 `min(actual_starts) if actual_starts else None` 결과를 저장합니다. 의미/사용: `actual_base_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_base_dt = min(actual_starts) if actual_starts else None

    # LINE-BY-LINE: `row in selected` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in selected:
        # LINE-BY-LINE: 조건 `base_dt is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if base_dt is not None:
            # LINE-BY-LINE: `key, output_key in (` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key, output_key in (
                # LINE-BY-LINE: `in(...)` 호출에 `("_planned_start_dt", "planned_start_minutes")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                ("_planned_start_dt", "planned_start_minutes"),
                # LINE-BY-LINE: `in(...)` 호출에 `("_planned_end_dt", "planned_finish_minutes")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                ("_planned_end_dt", "planned_finish_minutes"),
                # LINE-BY-LINE: `in(...)` 호출에 `("_downstream_dt", "due_date_minutes")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                ("_downstream_dt", "due_date_minutes"),
            # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            ):
                # LINE-BY-LINE: `value`에 `row.get(key)` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                value = row.get(key)
                # LINE-BY-LINE: `row[output_key]`에 `None if value is None else (value - base_dt).total_seconds() / 60.0` 결과를 저장합니다. 의미/사용: `row[output_key]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                row[output_key] = None if value is None else (value - base_dt).total_seconds() / 60.0
        # LINE-BY-LINE: 조건 `actual_base_dt is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_base_dt is not None:
            # LINE-BY-LINE: `row["actual_start_minutes"]`에 `(row["_actual_start_dt"] - actual_base_dt).total_seconds() / 60.0` 결과를 저장합니다. 의미/사용: `row["actual_start_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["actual_start_minutes"] = (row["_actual_start_dt"] - actual_base_dt).total_seconds() / 60.0
            # LINE-BY-LINE: `row["actual_finish_minutes"]`에 `(row["_actual_end_dt"] - actual_base_dt).total_seconds() / 60.0` 결과를 저장합니다. 의미/사용: `row["actual_finish_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["actual_finish_minutes"] = (row["_actual_end_dt"] - actual_base_dt).total_seconds() / 60.0

        # LINE-BY-LINE: `private_key in list(row)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for private_key in list(row):
            # LINE-BY-LINE: 조건 `private_key.startswith("_")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if private_key.startswith("_"):
                # LINE-BY-LINE: `row.pop(private_key, None)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                row.pop(private_key, None)

    # LINE-BY-LINE: 호출자에게 `CleanedCuttingData(records=selected, excluded_records=excluded, source_path="")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return CleanedCuttingData(records=selected, excluded_records=excluded, source_path="")


# LINE-BY-LINE: `load_and_clean_cutting_data` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def load_and_clean_cutting_data(
    # LINE-BY-LINE: `path`를 `str,` 타입으로 선언합니다. 의미/사용: `path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path: str,
    # LINE-BY-LINE: `sheet_name` 변수에 `"Sheet"` 결과를 저장합니다. 의미: `sheet_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    sheet_name: str = "Sheet",
    # LINE-BY-LINE: `target_series` 변수에 `("NP",)` 결과를 저장합니다. 의미: `target_series` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_series: Sequence[str] = ("NP",),
    # LINE-BY-LINE: `max_records` 변수에 `None` 결과를 저장합니다. 의미: `max_records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_records: Optional[int] = None,
# LINE-BY-LINE: `) -> CleanedCuttingData:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> CleanedCuttingData:
    """파일 로딩과 전처리를 한 번에 수행한다."""

    # LINE-BY-LINE: `rows`에 `load_raw_cutting_rows(path, sheet_name=sheet_name)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = load_raw_cutting_rows(path, sheet_name=sheet_name)
    # LINE-BY-LINE: `cleaned`에 `clean_cutting_records(rows, target_series=target_series, max_records=max_records)` 결과를 저장합니다. 의미/사용: `cleaned` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cleaned = clean_cutting_records(rows, target_series=target_series, max_records=max_records)
    # LINE-BY-LINE: `cleaned.source_path`에 `str(path)` 결과를 저장합니다. 의미/사용: `source_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cleaned.source_path = str(path)
    # LINE-BY-LINE: 호출자에게 `cleaned`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return cleaned


# LINE-BY-LINE: `write_records_csv(records: Iterable[Dict[str, Any]], output_path: str)` 함수를 정의합니다. 반환 타입: `None`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
def write_records_csv(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    """dict row 목록을 CSV로 저장한다."""

    # LINE-BY-LINE: `rows`에 `list(records)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(records)
    # LINE-BY-LINE: `output_file`에 `Path(output_path)` 결과를 저장합니다. 의미/사용: `output_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_file = Path(output_path)
    # LINE-BY-LINE: `output_file.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: 조건 `not rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows:
        # LINE-BY-LINE: `output_file.write_text("", encoding` 여러 변수에 `"utf-8-sig")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        output_file.write_text("", encoding="utf-8-sig")
        # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
        return

    # LINE-BY-LINE: `fieldnames` 변수에 `[]` 결과를 저장합니다. 의미: `fieldnames` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    fieldnames: List[str] = []
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `key in row` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for key in row:
            # LINE-BY-LINE: 조건 `key not in fieldnames`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key not in fieldnames:
                # LINE-BY-LINE: `fieldnames.append(key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                fieldnames.append(key)

    # LINE-BY-LINE: `output_file.open("w", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with output_file.open("w", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `writer`에 `csv.DictWriter(handle, fieldnames=fieldnames)` 결과를 저장합니다. 의미/사용: `writer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # LINE-BY-LINE: `writer.writeheader()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        writer.writeheader()
        # LINE-BY-LINE: `writer.writerows(rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        writer.writerows(rows)
