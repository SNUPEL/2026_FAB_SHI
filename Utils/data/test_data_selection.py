"""실적일 분포 기반 테스트 데이터 선정 유틸리티.

이 파일의 책임:
- 실적 착수일 기준으로 어느 날짜에 작업이 많은지/적은지 분석한다.
- 100건 회귀검증용 scenario를 날짜 분포가 깨지지 않게 추출한다.
- 사용자가 지정한 실적 착수시간 범위 예: 20260226~20260228에 해당하는 slice를 만든다.

중요:
- 시간 parsing은 strict하게 한다.
- `YYYYMMDDHHMM` 형식이 아니면 원인을 출력하고 실패시킨다.
"""

# LINE-BY-LINE: `__future__` 모듈에서 `annotations`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from __future__ import annotations

# LINE-BY-LINE: `csv` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import csv
# LINE-BY-LINE: `json` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import json
# LINE-BY-LINE: `math` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import math
# LINE-BY-LINE: `statistics` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import statistics
# LINE-BY-LINE: `collections` 모듈에서 `Counter, defaultdict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from collections import Counter, defaultdict
# LINE-BY-LINE: `copy` 모듈에서 `deepcopy`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from copy import deepcopy
# LINE-BY-LINE: `datetime` 모듈에서 `datetime, timedelta`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from datetime import datetime, timedelta
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Tuple

# LINE-BY-LINE: `.scenario_generator` 모듈에서 `save_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .scenario_generator import save_scenario


# LINE-BY-LINE: `_write_csv(rows: Iterable[Dict[str, Any]], path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
def _write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    """dict row 목록을 CSV로 저장한다."""

    # LINE-BY-LINE: `rows`에 `list(rows)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(rows)
    # LINE-BY-LINE: `path.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: 조건 `not rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows:
        # LINE-BY-LINE: `path.write_text("", encoding` 여러 변수에 `"utf-8-sig")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        path.write_text("", encoding="utf-8-sig")
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

    # LINE-BY-LINE: `path.open("w", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `writer`에 `csv.DictWriter(handle, fieldnames=fieldnames)` 결과를 저장합니다. 의미/사용: `writer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # LINE-BY-LINE: `writer.writeheader()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        writer.writeheader()
        # LINE-BY-LINE: `writer.writerows(rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        writer.writerows(rows)


# LINE-BY-LINE: `_write_json(data: Any, path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _write_json(data: Any, path: Path) -> None:
    """dict/list 데이터를 JSON으로 저장한다."""

    # LINE-BY-LINE: `path.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `path.write_text(json.dumps(data, ensure_ascii` 여러 변수에 `False, indent=2), encoding="utf-8")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# LINE-BY-LINE: `_parse_actual_datetime(job: Dict[str, Any], field_name: str)` 함수를 정의합니다. 반환 타입: `datetime`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _parse_actual_datetime(job: Dict[str, Any], field_name: str) -> datetime:
    """job의 실적시각 필드를 `YYYYMMDDHHMM` datetime으로 strict 파싱한다."""

    # LINE-BY-LINE: `raw_value`에 `str(job.get(field_name, "")).strip()` 결과를 저장합니다. 의미/사용: `raw_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    raw_value = str(job.get(field_name, "")).strip()
    # LINE-BY-LINE: `job_id`에 `job.get("job_id", "")` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
    job_id = job.get("job_id", "")
    # LINE-BY-LINE: `work_order_no`에 `job.get("source_wk_ord_no", "")` 결과를 저장합니다. 의미/사용: `work_order_no`는 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
    work_order_no = job.get("source_wk_ord_no", "")
    # LINE-BY-LINE: 조건 `len(raw_value) != 12 or not raw_value.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(raw_value) != 12 or not raw_value.isdigit():
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._parse_actual_datetime] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection._parse_actual_datetime] "
            # LINE-BY-LINE: `f"cause`에 `invalid_datetime field={field_name} value={raw_value} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=invalid_datetime field={field_name} value={raw_value} "
            # LINE-BY-LINE: `f"job_id`에 `{job_id} work_order_no={work_order_no}"` 결과를 저장합니다. 의미/사용: `f"job_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"job_id={job_id} work_order_no={work_order_no}"
        )
        # LINE-BY-LINE: `ValueError(f"{field_name} must be YYYYMMDDHHMM")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"{field_name} must be YYYYMMDDHHMM")
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 호출자에게 `datetime.strptime(raw_value, "%Y%m%d%H%M")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return datetime.strptime(raw_value, "%Y%m%d%H%M")
    # LINE-BY-LINE: `except ValueError as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except ValueError as exc:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._parse_actual_datetime] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection._parse_actual_datetime] "
            # LINE-BY-LINE: `f"cause`에 `{exc} field={field_name} value={raw_value} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause={exc} field={field_name} value={raw_value} "
            # LINE-BY-LINE: `f"job_id`에 `{job_id} work_order_no={work_order_no}"` 결과를 저장합니다. 의미/사용: `f"job_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"job_id={job_id} work_order_no={work_order_no}"
        )
        # LINE-BY-LINE: `RuntimeError("failed to parse actual datetime") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("failed to parse actual datetime") from exc


# LINE-BY-LINE: `_date_key(value: datetime)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _date_key(value: datetime) -> str:
    """datetime을 YYYYMMDD 문자열로 변환한다."""

    # LINE-BY-LINE: 호출자에게 `value.strftime("%Y%m%d")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return value.strftime("%Y%m%d")


# LINE-BY-LINE: `_iter_date_keys(start: datetime, finish: datetime)` 함수를 정의합니다. 반환 타입: `Iterable[str]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _iter_date_keys(start: datetime, finish: datetime) -> Iterable[str]:
    """start~finish 사이에 걸친 날짜 key를 모두 생성한다."""

    # LINE-BY-LINE: `current`에 `start.date()` 결과를 저장합니다. 의미/사용: `current` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    current = start.date()
    # LINE-BY-LINE: `finish_date`에 `finish.date()` 결과를 저장합니다. 의미/사용: `finish_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    finish_date = finish.date()
    # LINE-BY-LINE: 조건 `current <= finish_date`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while current <= finish_date:
        # LINE-BY-LINE: `yield current.strftime("%Y%m%d")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        yield current.strftime("%Y%m%d")
        # LINE-BY-LINE: `current` 값을 `timedelta(days=1)` 기준으로 누적/증가합니다. 의미/사용: `current` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current += timedelta(days=1)


# LINE-BY-LINE: `_job_sort_key(job: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Tuple[str, str, int, str]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _job_sort_key(job: Dict[str, Any]) -> Tuple[str, str, int, str]:
    """실적 착수/종료/원본 row 순으로 안정 정렬하는 key."""

    # LINE-BY-LINE: 호출자에게 `(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return (
        # LINE-BY-LINE: `return(...)` 호출에 `str(job.get("actual_start_datetime", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        str(job.get("actual_start_datetime", "")),
        # LINE-BY-LINE: `return(...)` 호출에 `str(job.get("actual_end_datetime", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        str(job.get("actual_end_datetime", "")),
        # LINE-BY-LINE: `return(...)` 호출에 `int(job.get("source_row_index") or 0)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        int(job.get("source_row_index") or 0),
        # LINE-BY-LINE: `return(...)` 호출에 `str(job.get("job_id", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        str(job.get("job_id", "")),
    )


# LINE-BY-LINE: `build_actual_day_distribution(scenario: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def build_actual_day_distribution(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """실적 착수일/종료일/overlap 기준 일별 분포를 만든다."""

    # LINE-BY-LINE: `start_counts` 변수에 `Counter()` 결과를 저장합니다. 의미: `start_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_counts: Counter[str] = Counter()
    # LINE-BY-LINE: `finish_counts` 변수에 `Counter()` 결과를 저장합니다. 의미: `finish_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    finish_counts: Counter[str] = Counter()
    # LINE-BY-LINE: `active_counts` 변수에 `Counter()` 결과를 저장합니다. 의미: `active_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    active_counts: Counter[str] = Counter()
    # LINE-BY-LINE: `start_machine_counts` 변수에 `defaultdict(Counter)` 결과를 저장합니다. 의미: `start_machine_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_machine_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    # LINE-BY-LINE: `start_bay_counts` 변수에 `defaultdict(Counter)` 결과를 저장합니다. 의미: `start_bay_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_bay_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    # LINE-BY-LINE: `first_start_by_day` 변수에 `{}` 결과를 저장합니다. 의미: `first_start_by_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    first_start_by_day: Dict[str, str] = {}
    # LINE-BY-LINE: `last_finish_by_day` 변수에 `{}` 결과를 저장합니다. 의미: `last_finish_by_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    last_finish_by_day: Dict[str, str] = {}
    # LINE-BY-LINE: `duration_by_start_day` 변수에 `defaultdict(float)` 결과를 저장합니다. 의미: `duration_by_start_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_by_start_day: Dict[str, float] = defaultdict(float)

    # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in scenario.get("jobs", []):
        # LINE-BY-LINE: `start_dt`에 `_parse_actual_datetime(job, "actual_start_datetime")` 결과를 저장합니다. 의미/사용: `start_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_dt = _parse_actual_datetime(job, "actual_start_datetime")
        # LINE-BY-LINE: `finish_dt`에 `_parse_actual_datetime(job, "actual_end_datetime")` 결과를 저장합니다. 의미/사용: `finish_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_dt = _parse_actual_datetime(job, "actual_end_datetime")
        # LINE-BY-LINE: 조건 `finish_dt < start_dt`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if finish_dt < start_dt:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection.build_actual_day_distribution] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][test_data_selection.build_actual_day_distribution] "
                # LINE-BY-LINE: `f"cause`에 `finish_before_start job_id={job.get('job_id', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=finish_before_start job_id={job.get('job_id', '')} "
                # LINE-BY-LINE: `f"start`에 `{job.get('actual_start_datetime', '')} finish={job.get('actual_end_datetime', '')}"` 결과를 저장합니다. 의미/사용: `f"start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"start={job.get('actual_start_datetime', '')} finish={job.get('actual_end_datetime', '')}"
            )
            # LINE-BY-LINE: `ValueError("actual finish datetime must be >= actual start datetime")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("actual finish datetime must be >= actual start datetime")

        # LINE-BY-LINE: `start_day`에 `_date_key(start_dt)` 결과를 저장합니다. 의미/사용: `start_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_day = _date_key(start_dt)
        # LINE-BY-LINE: `finish_day`에 `_date_key(finish_dt)` 결과를 저장합니다. 의미/사용: `finish_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_day = _date_key(finish_dt)
        # LINE-BY-LINE: `start_counts[start_day]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `start_counts[start_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_counts[start_day] += 1
        # LINE-BY-LINE: `finish_counts[finish_day]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `finish_counts[finish_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_counts[finish_day] += 1
        # LINE-BY-LINE: `start_machine_counts[start_day][str(job.get("source_machine_id", ""))]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `get("source_machine_id", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_machine_counts[start_day][str(job.get("source_machine_id", ""))] += 1
        # LINE-BY-LINE: `start_bay_counts[start_day][str(job.get("source_cut_bay", ""))]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `get("source_cut_bay", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_bay_counts[start_day][str(job.get("source_cut_bay", ""))] += 1
        # LINE-BY-LINE: `duration_by_start_day[start_day]` 값을 `float(job.get("actual_duration_minutes") or 0.0)` 기준으로 누적/증가합니다. 의미/사용: `duration_by_start_day[start_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        duration_by_start_day[start_day] += float(job.get("actual_duration_minutes") or 0.0)

        # LINE-BY-LINE: `first_start_by_day[start_day]`에 `min(first_start_by_day.get(start_day, job["actual_start_datetime"]), job["actual_start_datetime"])` 결과를 저장합니다. 의미/사용: `first_start_by_day[start_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        first_start_by_day[start_day] = min(first_start_by_day.get(start_day, job["actual_start_datetime"]), job["actual_start_datetime"])
        # LINE-BY-LINE: `last_finish_by_day[start_day]`에 `max(last_finish_by_day.get(start_day, job["actual_end_datetime"]), job["actual_end_datetime"])` 결과를 저장합니다. 의미/사용: `last_finish_by_day[start_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        last_finish_by_day[start_day] = max(last_finish_by_day.get(start_day, job["actual_end_datetime"]), job["actual_end_datetime"])

        # LINE-BY-LINE: `active_day in _iter_date_keys(start_dt, finish_dt)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for active_day in _iter_date_keys(start_dt, finish_dt):
            # LINE-BY-LINE: `active_counts[active_day]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `active_counts[active_day]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_counts[active_day] += 1

    # LINE-BY-LINE: `all_days`에 `sorted(set(start_counts) | set(finish_counts) | set(active_counts))` 결과를 저장합니다. 의미/사용: `all_days` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    all_days = sorted(set(start_counts) | set(finish_counts) | set(active_counts))
    # LINE-BY-LINE: `rows`에 `[]` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = []
    # LINE-BY-LINE: `day in all_days` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for day in all_days:
        # LINE-BY-LINE: `rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `actual_day` 키에 `day` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_day": day,
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `actual_day_iso` 키에 `f"{day[:4]}-{day[4:6]}-{day[6:8]}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_day_iso": f"{day[:4]}-{day[4:6]}-{day[6:8]}",
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `start_job_count` 키에 `start_counts.get(day, 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_job_count": start_counts.get(day, 0),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `finish_job_count` 키에 `finish_counts.get(day, 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_job_count": finish_counts.get(day, 0),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `active_overlap_job_count` 키에 `active_counts.get(day, 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "active_overlap_job_count": active_counts.get(day, 0),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `unique_start_machine_count` 키에 `len([key for key in start_machine_counts.get(day, {}) if key])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "unique_start_machine_count": len([key for key in start_machine_counts.get(day, {}) if key]),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `unique_start_bay_count` 키에 `len([key for key in start_bay_counts.get(day, {}) if key])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "unique_start_bay_count": len([key for key in start_bay_counts.get(day, {}) if key]),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `start_machine_counts` 키에 `"|".join(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_machine_counts": "|".join(
                    # LINE-BY-LINE: `f"{machine}:{count}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    f"{machine}:{count}"
                    # LINE-BY-LINE: `machine, count in sorted(start_machine_counts.get(day, {}).items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for machine, count in sorted(start_machine_counts.get(day, {}).items())
                    # LINE-BY-LINE: 조건 `machine`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if machine
                ),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `start_bay_counts` 키에 `"|".join(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_bay_counts": "|".join(
                    # LINE-BY-LINE: `f"{bay}:{count}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    f"{bay}:{count}"
                    # LINE-BY-LINE: `bay, count in sorted(start_bay_counts.get(day, {}).items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                    for bay, count in sorted(start_bay_counts.get(day, {}).items())
                    # LINE-BY-LINE: 조건 `bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if bay
                ),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `total_actual_duration_minutes` 키에 `round(duration_by_start_day.get(day, 0.0), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_actual_duration_minutes": round(duration_by_start_day.get(day, 0.0), 6),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `first_actual_start_datetime` 키에 `first_start_by_day.get(day, "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "first_actual_start_datetime": first_start_by_day.get(day, ""),
                # LINE-BY-LINE: `build_actual_day_distribution`에서 반환/저장할 dict의 `last_actual_finish_datetime` 키에 `last_finish_by_day.get(day, "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "last_actual_finish_datetime": last_finish_by_day.get(day, ""),
            }
        )
    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `_pick_day_rows(day_rows: List[Dict[str, Any]], low_usable_min_count: int)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _pick_day_rows(day_rows: List[Dict[str, Any]], low_usable_min_count: int) -> List[Dict[str, Any]]:
    """high/median/low/low_usable 대표 날짜를 고른다."""

    # LINE-BY-LINE: `rows_with_starts`에 `[row for row in day_rows if int(row["start_job_count"]) > 0]` 결과를 저장합니다. 의미/사용: `rows_with_starts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rows_with_starts = [row for row in day_rows if int(row["start_job_count"]) > 0]
    # LINE-BY-LINE: 조건 `not rows_with_starts`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows_with_starts:
        # LINE-BY-LINE: 콘솔에 `print("[ERROR][test_data_selection._pick_day_rows] cause=no_start_days")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[ERROR][test_data_selection._pick_day_rows] cause=no_start_days")
        # LINE-BY-LINE: `ValueError("at least one actual start day is required")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("at least one actual start day is required")

    # LINE-BY-LINE: `counts`에 `[int(row["start_job_count"]) for row in rows_with_starts]` 결과를 저장합니다. 의미/사용: `counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    counts = [int(row["start_job_count"]) for row in rows_with_starts]
    # LINE-BY-LINE: `median_count`에 `statistics.median(counts)` 결과를 저장합니다. 의미/사용: `median_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    median_count = statistics.median(counts)
    # LINE-BY-LINE: `high`에 `max(rows_with_starts, key=lambda row: (int(row["start_job_count"]), row["actual_day"]))` 결과를 저장합니다. 의미/사용: `high` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    high = max(rows_with_starts, key=lambda row: (int(row["start_job_count"]), row["actual_day"]))
    # LINE-BY-LINE: `low`에 `min(rows_with_starts, key=lambda row: (int(row["start_job_count"]), row["actual_day"]))` 결과를 저장합니다. 의미/사용: `low` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    low = min(rows_with_starts, key=lambda row: (int(row["start_job_count"]), row["actual_day"]))
    # LINE-BY-LINE: `median`에 `min(rows_with_starts, key=lambda row: (abs(int(row["start_job_count"]) - median_count), row["actu...` 결과를 저장합니다. 의미/사용: `median` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    median = min(rows_with_starts, key=lambda row: (abs(int(row["start_job_count"]) - median_count), row["actual_day"]))
    # LINE-BY-LINE: `usable_candidates`에 `[row for row in rows_with_starts if int(row["start_job_count"]) >= low_usable_min_count]` 결과를 저장합니다. 의미/사용: `usable_candidates` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    usable_candidates = [row for row in rows_with_starts if int(row["start_job_count"]) >= low_usable_min_count]
    # LINE-BY-LINE: `low_usable`에 `min(usable_candidates, key=lambda row: (int(row["start_job_count"]), row["actual_day"])) if usabl...` 결과를 저장합니다. 의미/사용: `low_usable` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    low_usable = min(usable_candidates, key=lambda row: (int(row["start_job_count"]), row["actual_day"])) if usable_candidates else None

    # LINE-BY-LINE: `selected`에 `[` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
    selected = [
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `{**high, "slice_name": "high_start_day", "selection_reason": "maximum actual-start job count"}` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        {**high, "slice_name": "high_start_day", "selection_reason": "maximum actual-start job count"},
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `{**median, "slice_name": "median_start_day", "selection_reason": "closest to median actual-start ...` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        {**median, "slice_name": "median_start_day", "selection_reason": "closest to median actual-start job count"},
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `{**low, "slice_name": "low_start_day", "selection_reason": "minimum nonzero actual-start job count"}` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        {**low, "slice_name": "low_start_day", "selection_reason": "minimum nonzero actual-start job count"},
    ]
    # LINE-BY-LINE: 조건 `low_usable is not None and low_usable["actual_day"] not in {row["actual_day"] for row in selected}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if low_usable is not None and low_usable["actual_day"] not in {row["actual_day"] for row in selected}:
        # LINE-BY-LINE: `selected.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        selected.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**low_usable` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                **low_usable,
                # LINE-BY-LINE: `_pick_day_rows`에서 반환/저장할 dict의 `slice_name` 키에 `"low_usable_start_day"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "slice_name": "low_usable_start_day",
                # LINE-BY-LINE: `_pick_day_rows`에서 반환/저장할 dict의 `selection_reason` 키에 `f"minimum actual-start job count >= {low_usable_min_count}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "selection_reason": f"minimum actual-start job count >= {low_usable_min_count}",
            }
        )
    # LINE-BY-LINE: 호출자에게 `selected`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return selected


# LINE-BY-LINE: `_jobs_started_on_day(scenario: Dict[str, Any], day_key: str)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _jobs_started_on_day(scenario: Dict[str, Any], day_key: str) -> List[Dict[str, Any]]:
    """특정 실적 착수일에 시작한 job 목록을 반환한다."""

    # LINE-BY-LINE: 호출자에게 `sorted(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(
        # LINE-BY-LINE: `[` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        [
            # LINE-BY-LINE: `job` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            job
            # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for job in scenario.get("jobs", [])
            # LINE-BY-LINE: 조건 `str(job.get("actual_start_datetime", "")).startswith(day_key)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if str(job.get("actual_start_datetime", "")).startswith(day_key)
        ],
        # LINE-BY-LINE: `key`에 `_job_sort_key` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key=_job_sort_key,
    )


# LINE-BY-LINE: `_evenly_pick(jobs: List[Dict[str, Any]], count: int)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _evenly_pick(jobs: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """정렬된 job 목록에서 앞쪽에 치우치지 않게 균등 샘플링한다."""

    # LINE-BY-LINE: 조건 `count <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if count <= 0:
        # LINE-BY-LINE: 호출자에게 `[]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return []
    # LINE-BY-LINE: 조건 `count >= len(jobs)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if count >= len(jobs):
        # LINE-BY-LINE: 호출자에게 `list(jobs)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return list(jobs)
    # LINE-BY-LINE: 조건 `count == 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if count == 1:
        # LINE-BY-LINE: 호출자에게 `[jobs[0]]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return [jobs[0]]
    # LINE-BY-LINE: `indexes`에 `sorted({round(index * (len(jobs) - 1) / (count - 1)) for index in range(count)})` 결과를 저장합니다. 의미/사용: `indexes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    indexes = sorted({round(index * (len(jobs) - 1) / (count - 1)) for index in range(count)})
    # LINE-BY-LINE: 호출자에게 `[jobs[index] for index in indexes[:count]]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [jobs[index] for index in indexes[:count]]


# LINE-BY-LINE: `_build_stratified_jobs(scenario: Dict[str, Any], target_count: int)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _build_stratified_jobs(scenario: Dict[str, Any], target_count: int) -> List[Dict[str, Any]]:
    """실적 착수일 분포를 보존하도록 target_count개 job을 층화 추출한다."""

    # LINE-BY-LINE: `jobs_by_day` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `jobs_by_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    jobs_by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in scenario.get("jobs", []):
        # LINE-BY-LINE: `day_key`에 `str(job.get("actual_start_datetime", ""))[:8]` 결과를 저장합니다. 의미/사용: `day_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        day_key = str(job.get("actual_start_datetime", ""))[:8]
        # LINE-BY-LINE: `jobs_by_day[day_key].append(job)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        jobs_by_day[day_key].append(job)

    # LINE-BY-LINE: `day_key in jobs_by_day` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for day_key in jobs_by_day:
        # LINE-BY-LINE: `jobs_by_day[day_key]`에 `sorted(jobs_by_day[day_key], key=_job_sort_key)` 결과를 저장합니다. 의미/사용: `jobs_by_day[day_key]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs_by_day[day_key] = sorted(jobs_by_day[day_key], key=_job_sort_key)

    # LINE-BY-LINE: `total_jobs`에 `sum(len(jobs) for jobs in jobs_by_day.values())` 결과를 저장합니다. 의미/사용: `total_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    total_jobs = sum(len(jobs) for jobs in jobs_by_day.values())
    # LINE-BY-LINE: 조건 `total_jobs < target_count`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if total_jobs < target_count:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._build_stratified_jobs] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection._build_stratified_jobs] "
            # LINE-BY-LINE: `f"cause`에 `target_exceeds_total target_count={target_count} total_jobs={total_jobs}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=target_exceeds_total target_count={target_count} total_jobs={total_jobs}"
        )
        # LINE-BY-LINE: `ValueError("target_count cannot exceed available jobs")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("target_count cannot exceed available jobs")

    # LINE-BY-LINE: `quota_rows`에 `[]` 결과를 저장합니다. 의미/사용: `quota_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    quota_rows = []
    # LINE-BY-LINE: `day_key, jobs in sorted(jobs_by_day.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for day_key, jobs in sorted(jobs_by_day.items()):
        # LINE-BY-LINE: `exact_quota`에 `len(jobs) * target_count / total_jobs` 결과를 저장합니다. 의미/사용: `exact_quota` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        exact_quota = len(jobs) * target_count / total_jobs
        # LINE-BY-LINE: `quota`에 `min(len(jobs), max(1, math.floor(exact_quota)))` 결과를 저장합니다. 의미/사용: `quota` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        quota = min(len(jobs), max(1, math.floor(exact_quota)))
        # LINE-BY-LINE: `quota_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        quota_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_build_stratified_jobs`에서 반환/저장할 dict의 `day_key` 키에 `day_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "day_key": day_key,
                # LINE-BY-LINE: `_build_stratified_jobs`에서 반환/저장할 dict의 `jobs` 키에 `jobs` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "jobs": jobs,
                # LINE-BY-LINE: `_build_stratified_jobs`에서 반환/저장할 dict의 `quota` 키에 `quota` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "quota": quota,
                # LINE-BY-LINE: `_build_stratified_jobs`에서 반환/저장할 dict의 `fraction` 키에 `exact_quota - math.floor(exact_quota)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "fraction": exact_quota - math.floor(exact_quota),
            }
        )

    # LINE-BY-LINE: 조건 `sum(row["quota"] for row in quota_rows) > target_count`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while sum(row["quota"] for row in quota_rows) > target_count:
        # LINE-BY-LINE: `candidates`에 `[row for row in quota_rows if row["quota"] > 1]` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates = [row for row in quota_rows if row["quota"] > 1]
        # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not candidates:
            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            break
        # LINE-BY-LINE: `target`에 `min(candidates, key=lambda row: (row["fraction"], row["day_key"]))` 결과를 저장합니다. 의미/사용: `target` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target = min(candidates, key=lambda row: (row["fraction"], row["day_key"]))
        # LINE-BY-LINE: `target["quota"]` 값을 `1` 기준으로 차감합니다. 의미/사용: `target["quota"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target["quota"] -= 1

    # LINE-BY-LINE: 조건 `sum(row["quota"] for row in quota_rows) < target_count`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while sum(row["quota"] for row in quota_rows) < target_count:
        # LINE-BY-LINE: `candidates`에 `[row for row in quota_rows if row["quota"] < len(row["jobs"])]` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates = [row for row in quota_rows if row["quota"] < len(row["jobs"])]
        # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not candidates:
            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            break
        # LINE-BY-LINE: `target`에 `max(candidates, key=lambda row: (row["fraction"], row["day_key"]))` 결과를 저장합니다. 의미/사용: `target` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target = max(candidates, key=lambda row: (row["fraction"], row["day_key"]))
        # LINE-BY-LINE: `target["quota"]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `target["quota"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target["quota"] += 1

    # LINE-BY-LINE: `selected_jobs` 변수에 `[]` 결과를 저장합니다. 의미: `selected_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    selected_jobs: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `row in quota_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in quota_rows:
        # LINE-BY-LINE: `selected_jobs.extend(_evenly_pick(row["jobs"], int(row["quota"])))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        selected_jobs.extend(_evenly_pick(row["jobs"], int(row["quota"])))

    # LINE-BY-LINE: `selected_jobs`에 `sorted(selected_jobs, key=_job_sort_key)` 결과를 저장합니다. 의미/사용: `selected_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    selected_jobs = sorted(selected_jobs, key=_job_sort_key)
    # LINE-BY-LINE: 조건 `len(selected_jobs) != target_count`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(selected_jobs) != target_count:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._build_stratified_jobs] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection._build_stratified_jobs] "
            # LINE-BY-LINE: `f"cause`에 `stratified_count_mismatch expected={target_count} actual={len(selected_jobs)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=stratified_count_mismatch expected={target_count} actual={len(selected_jobs)}"
        )
        # LINE-BY-LINE: `RuntimeError("stratified test job count mismatch")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("stratified test job count mismatch")
    # LINE-BY-LINE: 호출자에게 `selected_jobs`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return selected_jobs


# LINE-BY-LINE: `_normalize_range_bound(value: str, *, is_end: bool)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _normalize_range_bound(value: str, *, is_end: bool) -> str:
    """YYYYMMDD 또는 YYYYMMDDHHMM 범위 입력을 YYYYMMDDHHMM으로 정규화한다."""

    # LINE-BY-LINE: `text`에 `str(value or "").strip()` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    text = str(value or "").strip()
    # LINE-BY-LINE: 조건 `len(text) == 8 and text.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(text) == 8 and text.isdigit():
        # LINE-BY-LINE: 호출자에게 `f"{text}{'2359' if is_end else '0000'}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return f"{text}{'2359' if is_end else '0000'}"
    # LINE-BY-LINE: 조건 `len(text) == 12 and text.isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(text) == 12 and text.isdigit():
        # LINE-BY-LINE: 호출자에게 `text`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return text
    # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(
        # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._normalize_range_bound] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "[ERROR][test_data_selection._normalize_range_bound] "
        # LINE-BY-LINE: `f"cause`에 `invalid_range_bound value={value} is_end={is_end}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"cause=invalid_range_bound value={value} is_end={is_end}"
    )
    # LINE-BY-LINE: `ValueError("range bound must be YYYYMMDD or YYYYMMDDHHMM")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise ValueError("range bound must be YYYYMMDD or YYYYMMDDHHMM")


# LINE-BY-LINE: `_jobs_started_in_range(scenario: Dict[str, Any], start_value: str, end_value: str)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _jobs_started_in_range(scenario: Dict[str, Any], start_value: str, end_value: str) -> List[Dict[str, Any]]:
    """실적 착수시간이 지정 범위 안에 있는 job 목록을 반환한다."""

    # LINE-BY-LINE: `start_text`에 `_normalize_range_bound(start_value, is_end=False)` 결과를 저장합니다. 의미/사용: `start_text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_text = _normalize_range_bound(start_value, is_end=False)
    # LINE-BY-LINE: `end_text`에 `_normalize_range_bound(end_value, is_end=True)` 결과를 저장합니다. 의미/사용: `end_text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    end_text = _normalize_range_bound(end_value, is_end=True)
    # LINE-BY-LINE: `start_dt`에 `datetime.strptime(start_text, "%Y%m%d%H%M")` 결과를 저장합니다. 의미/사용: `start_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_dt = datetime.strptime(start_text, "%Y%m%d%H%M")
    # LINE-BY-LINE: `end_dt`에 `datetime.strptime(end_text, "%Y%m%d%H%M")` 결과를 저장합니다. 의미/사용: `end_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    end_dt = datetime.strptime(end_text, "%Y%m%d%H%M")
    # LINE-BY-LINE: 조건 `end_dt < start_dt`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if end_dt < start_dt:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._jobs_started_in_range] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection._jobs_started_in_range] "
            # LINE-BY-LINE: `f"cause`에 `end_before_start start={start_text} end={end_text}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=end_before_start start={start_text} end={end_text}"
        )
        # LINE-BY-LINE: `ValueError("actual start range end must be >= start")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("actual start range end must be >= start")

    # LINE-BY-LINE: `jobs`에 `[]` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    jobs = []
    # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in scenario.get("jobs", []):
        # LINE-BY-LINE: `actual_start`에 `_parse_actual_datetime(job, "actual_start_datetime")` 결과를 저장합니다. 의미/사용: `actual_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start = _parse_actual_datetime(job, "actual_start_datetime")
        # LINE-BY-LINE: 조건 `start_dt <= actual_start <= end_dt`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_dt <= actual_start <= end_dt:
            # LINE-BY-LINE: `jobs.append(job)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            jobs.append(job)
    # LINE-BY-LINE: 호출자에게 `sorted(jobs, key=_job_sort_key)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(jobs, key=_job_sort_key)


# LINE-BY-LINE: `_recompute_actual_minutes_for_jobs(jobs: List[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _recompute_actual_minutes_for_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """선택된 slice 내부 기준으로 actual minute offset을 다시 계산한다."""

    # LINE-BY-LINE: 조건 `not jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not jobs:
        # LINE-BY-LINE: 호출자에게 `[]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return []
    # LINE-BY-LINE: `parsed_rows`에 `[]` 결과를 저장합니다. 의미/사용: `parsed_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    parsed_rows = []
    # LINE-BY-LINE: `job in jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in jobs:
        # LINE-BY-LINE: `copied`에 `deepcopy(job)` 결과를 저장합니다. 의미/사용: `copied` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        copied = deepcopy(job)
        # LINE-BY-LINE: `start_dt`에 `_parse_actual_datetime(copied, "actual_start_datetime")` 결과를 저장합니다. 의미/사용: `start_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_dt = _parse_actual_datetime(copied, "actual_start_datetime")
        # LINE-BY-LINE: `finish_dt`에 `_parse_actual_datetime(copied, "actual_end_datetime")` 결과를 저장합니다. 의미/사용: `finish_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_dt = _parse_actual_datetime(copied, "actual_end_datetime")
        # LINE-BY-LINE: 조건 `finish_dt < start_dt`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if finish_dt < start_dt:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection._recompute_actual_minutes_for_jobs] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][test_data_selection._recompute_actual_minutes_for_jobs] "
                # LINE-BY-LINE: `f"cause`에 `finish_before_start job_id={copied.get('job_id', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=finish_before_start job_id={copied.get('job_id', '')} "
                # LINE-BY-LINE: `f"start`에 `{copied.get('actual_start_datetime', '')} finish={copied.get('actual_end_datetime', '')}"` 결과를 저장합니다. 의미/사용: `f"start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"start={copied.get('actual_start_datetime', '')} finish={copied.get('actual_end_datetime', '')}"
            )
            # LINE-BY-LINE: `ValueError("actual finish datetime must be >= actual start datetime")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("actual finish datetime must be >= actual start datetime")
        # LINE-BY-LINE: `parsed_rows.append((copied, start_dt, finish_dt))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        parsed_rows.append((copied, start_dt, finish_dt))

    # LINE-BY-LINE: `base_dt`에 `min(start_dt for _, start_dt, _ in parsed_rows)` 결과를 저장합니다. 의미/사용: `base_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    base_dt = min(start_dt for _, start_dt, _ in parsed_rows)
    # LINE-BY-LINE: `output_jobs`에 `[]` 결과를 저장합니다. 의미/사용: `output_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_jobs = []
    # LINE-BY-LINE: `copied, start_dt, finish_dt in parsed_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for copied, start_dt, finish_dt in parsed_rows:
        # LINE-BY-LINE: `copied["actual_start_minutes"]`에 `round((start_dt - base_dt).total_seconds() / 60.0, 6)` 결과를 저장합니다. 의미/사용: `copied["actual_start_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        copied["actual_start_minutes"] = round((start_dt - base_dt).total_seconds() / 60.0, 6)
        # LINE-BY-LINE: `copied["actual_finish_minutes"]`에 `round((finish_dt - base_dt).total_seconds() / 60.0, 6)` 결과를 저장합니다. 의미/사용: `copied["actual_finish_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        copied["actual_finish_minutes"] = round((finish_dt - base_dt).total_seconds() / 60.0, 6)
        # LINE-BY-LINE: `copied["actual_duration_minutes"]`에 `round((finish_dt - start_dt).total_seconds() / 60.0, 6)` 결과를 저장합니다. 의미/사용: `copied["actual_duration_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        copied["actual_duration_minutes"] = round((finish_dt - start_dt).total_seconds() / 60.0, 6)
        # LINE-BY-LINE: `output_jobs.append(copied)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        output_jobs.append(copied)
    # LINE-BY-LINE: 호출자에게 `output_jobs`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return output_jobs


# LINE-BY-LINE: `_scenario_with_jobs(base_scenario: Dict[str, Any], jobs: List[Dict[str, Any]], metadata: Dict[str...)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def _scenario_with_jobs(base_scenario: Dict[str, Any], jobs: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """기존 scenario 구조를 유지하면서 jobs만 선택 slice로 교체한다."""

    # LINE-BY-LINE: `scenario`에 `deepcopy(base_scenario)` 결과를 저장합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario = deepcopy(base_scenario)
    # LINE-BY-LINE: `scenario["jobs"]`에 `_recompute_actual_minutes_for_jobs(jobs)` 결과를 저장합니다. 의미/사용: `scenario["jobs"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario["jobs"] = _recompute_actual_minutes_for_jobs(jobs)
    # LINE-BY-LINE: `scenario_metadata`에 `dict(scenario.get("metadata", {}))` 결과를 저장합니다. 의미/사용: `scenario_metadata` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_metadata = dict(scenario.get("metadata", {}))
    # LINE-BY-LINE: `scenario_metadata.update(metadata)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    scenario_metadata.update(metadata)
    # LINE-BY-LINE: `scenario_metadata["job_count"]`에 `len(jobs)` 결과를 저장합니다. 의미/사용: `scenario_metadata["job_count"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_metadata["job_count"] = len(jobs)
    # LINE-BY-LINE: `scenario["metadata"]`에 `scenario_metadata` 결과를 저장합니다. 의미/사용: `scenario["metadata"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario["metadata"] = scenario_metadata
    # LINE-BY-LINE: 호출자에게 `scenario`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return scenario


# LINE-BY-LINE: `_job_rows(jobs: Iterable[Dict[str, Any]], slice_name: str)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _job_rows(jobs: Iterable[Dict[str, Any]], slice_name: str) -> List[Dict[str, Any]]:
    """선택된 jobs를 검토용 CSV row로 변환한다."""

    # LINE-BY-LINE: `rows`에 `[]` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = []
    # LINE-BY-LINE: `index, job in enumerate(jobs, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for index, job in enumerate(jobs, start=1):
        # LINE-BY-LINE: `rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_job_rows`에서 반환/저장할 dict의 `slice_name` 키에 `slice_name` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "slice_name": slice_name,
                # LINE-BY-LINE: `_job_rows`에서 반환/저장할 dict의 `sequence` 키에 `index` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "sequence": index,
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": job.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": job.get("source_wk_ord_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `job.get("source_project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                "project_no": job.get("source_project_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `job.get("source_block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                "block_no": job.get("source_block_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `job.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": job.get("actual_start_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `job.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": job.get("actual_end_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `job.get("actual_duration_minutes", "")` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": job.get("actual_duration_minutes", ""),
                # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `job.get("source_machine_id", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                "source_machine_id": job.get("source_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `job.get("source_cut_bay", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                "source_cut_bay": job.get("source_cut_bay", ""),
                # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `job.get("plate_length", "")` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                "plate_length": job.get("plate_length", ""),
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `job.get("thickness", "")` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": job.get("thickness", ""),
                # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `job.get("part_count", "")` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
                "part_count": job.get("part_count", ""),
            }
        )
    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `select_actual_day_test_data` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def select_actual_day_test_data(
    # LINE-BY-LINE: `scenario`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario: Dict[str, Any],
    # LINE-BY-LINE: `output_dir`를 `str,` 타입으로 선언합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir: str,
    # LINE-BY-LINE: `scenario_output_dir`를 `str,` 타입으로 선언합니다. 의미/사용: `scenario_output_dir` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_output_dir: str,
    # LINE-BY-LINE: `stratified_count` 변수에 `100` 결과를 저장합니다. 의미: `stratified_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    stratified_count: int = 100,
    # LINE-BY-LINE: `low_usable_min_count` 변수에 `20` 결과를 저장합니다. 의미: `low_usable_min_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    low_usable_min_count: int = 20,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """실적일 편차를 반영한 테스트 scenario를 생성한다."""

    # LINE-BY-LINE: `output_path`에 `Path(output_dir)` 결과를 저장합니다. 의미/사용: `output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path = Path(output_dir)
    # LINE-BY-LINE: `scenario_output_path`에 `Path(scenario_output_dir)` 결과를 저장합니다. 의미/사용: `scenario_output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_output_path = Path(scenario_output_dir)
    # LINE-BY-LINE: `output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `scenario_output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_output_path.mkdir(parents=True, exist_ok=True)

    # LINE-BY-LINE: `day_rows`에 `build_actual_day_distribution(scenario)` 결과를 저장합니다. 의미/사용: `day_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    day_rows = build_actual_day_distribution(scenario)
    # LINE-BY-LINE: `selected_day_rows`에 `_pick_day_rows(day_rows, low_usable_min_count=low_usable_min_count)` 결과를 저장합니다. 의미/사용: `selected_day_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    selected_day_rows = _pick_day_rows(day_rows, low_usable_min_count=low_usable_min_count)
    # LINE-BY-LINE: `_write_csv(day_rows, output_path / "actual_day_distribution.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(day_rows, output_path / "actual_day_distribution.csv")
    # LINE-BY-LINE: `_write_json(day_rows, output_path / "actual_day_distribution.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(day_rows, output_path / "actual_day_distribution.json")
    # LINE-BY-LINE: `_write_csv(selected_day_rows, output_path / "selected_test_days.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(selected_day_rows, output_path / "selected_test_days.csv")

    # LINE-BY-LINE: `slice_rows` 변수에 `[]` 결과를 저장합니다. 의미: `slice_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    slice_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `job_list_rows` 변수에 `[]` 결과를 저장합니다. 의미: `job_list_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    job_list_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `created_scenarios`에 `[]` 결과를 저장합니다. 의미/사용: `created_scenarios` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    created_scenarios = []
    # LINE-BY-LINE: `selected_day in selected_day_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for selected_day in selected_day_rows:
        # LINE-BY-LINE: `slice_name`에 `str(selected_day["slice_name"])` 결과를 저장합니다. 의미/사용: `slice_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slice_name = str(selected_day["slice_name"])
        # LINE-BY-LINE: `actual_day`에 `str(selected_day["actual_day"])` 결과를 저장합니다. 의미/사용: `actual_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_day = str(selected_day["actual_day"])
        # LINE-BY-LINE: `jobs`에 `_jobs_started_on_day(scenario, actual_day)` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs = _jobs_started_on_day(scenario, actual_day)
        # LINE-BY-LINE: 조건 `len(jobs) != int(selected_day["start_job_count"])`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(jobs) != int(selected_day["start_job_count"]):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection.select_actual_day_test_data] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][test_data_selection.select_actual_day_test_data] "
                # LINE-BY-LINE: `f"cause`에 `day_job_count_mismatch day={actual_day} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=day_job_count_mismatch day={actual_day} "
                # LINE-BY-LINE: `f"expected`에 `{selected_day['start_job_count']} actual={len(jobs)}"` 결과를 저장합니다. 의미/사용: `f"expected` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"expected={selected_day['start_job_count']} actual={len(jobs)}"
            )
            # LINE-BY-LINE: `RuntimeError("selected day job count mismatch")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("selected day job count mismatch")

        # LINE-BY-LINE: `scenario_name`에 `f"np_{slice_name}_{actual_day}.yaml"` 결과를 저장합니다. 의미/사용: `scenario_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        scenario_name = f"np_{slice_name}_{actual_day}.yaml"
        # LINE-BY-LINE: `scenario_path`에 `scenario_output_path / scenario_name` 결과를 저장합니다. 의미/사용: `scenario_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        scenario_path = scenario_output_path / scenario_name
        # LINE-BY-LINE: `slice_scenario`에 `_scenario_with_jobs(` 결과를 저장합니다. 의미/사용: `slice_scenario` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        slice_scenario = _scenario_with_jobs(
            # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            scenario,
            # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `jobs` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            jobs,
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_name` 키에 `slice_name` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "test_slice_name": slice_name,
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_basis` 키에 `"actual_start_datetime day"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "test_slice_basis": "actual_start_datetime day",
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_actual_day` 키에 `actual_day` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "test_slice_actual_day": actual_day,
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_reason` 키에 `selected_day["selection_reason"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "test_slice_reason": selected_day["selection_reason"],
            },
        )
        # LINE-BY-LINE: `save_scenario(slice_scenario, str(scenario_path))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        save_scenario(slice_scenario, str(scenario_path))
        # LINE-BY-LINE: `created_scenarios.append(str(scenario_path))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        created_scenarios.append(str(scenario_path))
        # LINE-BY-LINE: `slice_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        slice_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `slice_name` 키에 `slice_name` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "slice_name": slice_name,
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `actual_day` 키에 `actual_day` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_day": actual_day,
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `job_count` 키에 `len(jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "job_count": len(jobs),
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `scenario_path` 키에 `str(scenario_path)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "scenario_path": str(scenario_path),
                # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `selection_reason` 키에 `selected_day["selection_reason"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "selection_reason": selected_day["selection_reason"],
            }
        )
        # LINE-BY-LINE: `job_list_rows.extend(_job_rows(jobs, slice_name))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        job_list_rows.extend(_job_rows(jobs, slice_name))

    # LINE-BY-LINE: `stratified_jobs`에 `_build_stratified_jobs(scenario, stratified_count)` 결과를 저장합니다. 의미/사용: `stratified_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    stratified_jobs = _build_stratified_jobs(scenario, stratified_count)
    # LINE-BY-LINE: `stratified_path`에 `scenario_output_path / f"np_stratified_actual_start_{stratified_count}.yaml"` 결과를 저장합니다. 의미/사용: `stratified_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    stratified_path = scenario_output_path / f"np_stratified_actual_start_{stratified_count}.yaml"
    # LINE-BY-LINE: `stratified_scenario`에 `_scenario_with_jobs(` 결과를 저장합니다. 의미/사용: `stratified_scenario` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    stratified_scenario = _scenario_with_jobs(
        # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario,
        # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `stratified_jobs` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        stratified_jobs,
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_name` 키에 `f"stratified_actual_start_{stratified_count}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_name": f"stratified_actual_start_{stratified_count}",
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_basis` 키에 `"actual_start_datetime day stratified"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_basis": "actual_start_datetime day stratified",
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `test_slice_reason` 키에 `"deterministic proportional sample across actual start days"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_reason": "deterministic proportional sample across actual start days",
        },
    )
    # LINE-BY-LINE: `save_scenario(stratified_scenario, str(stratified_path))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    save_scenario(stratified_scenario, str(stratified_path))
    # LINE-BY-LINE: `created_scenarios.append(str(stratified_path))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    created_scenarios.append(str(stratified_path))
    # LINE-BY-LINE: `slice_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    slice_rows.append(
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `slice_name` 키에 `f"stratified_actual_start_{stratified_count}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "slice_name": f"stratified_actual_start_{stratified_count}",
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `actual_day` 키에 `"multiple"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_day": "multiple",
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `job_count` 키에 `len(stratified_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "job_count": len(stratified_jobs),
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `scenario_path` 키에 `str(stratified_path)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "scenario_path": str(stratified_path),
            # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `selection_reason` 키에 `"deterministic proportional sample across actual start days"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "selection_reason": "deterministic proportional sample across actual start days",
        }
    )
    # LINE-BY-LINE: `job_list_rows.extend(_job_rows(stratified_jobs, f"stratified_actual_start_{stratified_count}"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    job_list_rows.extend(_job_rows(stratified_jobs, f"stratified_actual_start_{stratified_count}"))

    # LINE-BY-LINE: `_write_csv(slice_rows, output_path / "test_slices_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(slice_rows, output_path / "test_slices_summary.csv")
    # LINE-BY-LINE: `_write_csv(job_list_rows, output_path / "test_slice_jobs.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(job_list_rows, output_path / "test_slice_jobs.csv")
    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `source_job_count` 키에 `len(scenario.get("jobs", []))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "source_job_count": len(scenario.get("jobs", [])),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `actual_day_count` 키에 `len(day_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_day_count": len(day_rows),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `start_job_count_min` 키에 `min(int(row["start_job_count"]) for row in day_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "start_job_count_min": min(int(row["start_job_count"]) for row in day_rows),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `start_job_count_max` 키에 `max(int(row["start_job_count"]) for row in day_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "start_job_count_max": max(int(row["start_job_count"]) for row in day_rows),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `start_job_count_median` 키에 `statistics.median(int(row["start_job_count"]) for row in day_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "start_job_count_median": statistics.median(int(row["start_job_count"]) for row in day_rows),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `stratified_count` 키에 `stratified_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "stratified_count": stratified_count,
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `low_usable_min_count` 키에 `low_usable_min_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "low_usable_min_count": low_usable_min_count,
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `created_scenarios` 키에 `created_scenarios` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "created_scenarios": created_scenarios,
        # LINE-BY-LINE: 딕셔너리 키 `output_dir`에는 `str(output_path)` 값을 넣습니다. 의미: 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        "output_dir": str(output_path),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `scenario_output_dir` 키에 `str(scenario_output_path)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "scenario_output_dir": str(scenario_output_path),
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `selection_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "selection_note": (
            # LINE-BY-LINE: 문자열 값 `"Daily test slices are based on actual_start_datetime. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Daily test slices are based on actual_start_datetime. "
            # LINE-BY-LINE: 문자열 값 `"Distribution report also includes actual_end_datetime and active-overlap counts."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Distribution report also includes actual_end_datetime and active-overlap counts."
        ),
    }
    # LINE-BY-LINE: `_write_json(summary, output_path / "test_data_selection_summary.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(summary, output_path / "test_data_selection_summary.json")
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `summary` 키에 `summary` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "summary": summary,
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `day_rows` 키에 `day_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "day_rows": day_rows,
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `selected_day_rows` 키에 `selected_day_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "selected_day_rows": selected_day_rows,
        # LINE-BY-LINE: `select_actual_day_test_data`에서 반환/저장할 dict의 `slice_rows` 키에 `slice_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "slice_rows": slice_rows,
    }


# LINE-BY-LINE: `select_actual_start_range_test_data` 함수를 정의합니다. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def select_actual_start_range_test_data(
    # LINE-BY-LINE: `scenario`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario: Dict[str, Any],
    # LINE-BY-LINE: `output_dir`를 `str,` 타입으로 선언합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir: str,
    # LINE-BY-LINE: `scenario_output_path`를 `str,` 타입으로 선언합니다. 의미/사용: `scenario_output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_output_path: str,
    # LINE-BY-LINE: `start_value`를 `str,` 타입으로 선언합니다. 의미/사용: `start_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_value: str,
    # LINE-BY-LINE: `end_value`를 `str,` 타입으로 선언합니다. 의미/사용: `end_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    end_value: str,
    # LINE-BY-LINE: `expected_count` 변수에 `None` 결과를 저장합니다. 의미: `expected_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    expected_count: int | None = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """실적 착수시간 range 기준으로 테스트 scenario를 생성한다."""

    # LINE-BY-LINE: `output_path`에 `Path(output_dir)` 결과를 저장합니다. 의미/사용: `output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path = Path(output_dir)
    # LINE-BY-LINE: `output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `selected_jobs`에 `_jobs_started_in_range(scenario, start_value, end_value)` 결과를 저장합니다. 의미/사용: `selected_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    selected_jobs = _jobs_started_in_range(scenario, start_value, end_value)
    # LINE-BY-LINE: 조건 `expected_count is not None and len(selected_jobs) != int(expected_count)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if expected_count is not None and len(selected_jobs) != int(expected_count):
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection.select_actual_start_range_test_data] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection.select_actual_start_range_test_data] "
            # LINE-BY-LINE: `f"cause`에 `count_mismatch start={start_value} end={end_value} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=count_mismatch start={start_value} end={end_value} "
            # LINE-BY-LINE: `f"expected`에 `{expected_count} actual={len(selected_jobs)}"` 결과를 저장합니다. 의미/사용: `f"expected` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"expected={expected_count} actual={len(selected_jobs)}"
        )
        # LINE-BY-LINE: `RuntimeError("actual start range selected job count mismatch")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("actual start range selected job count mismatch")

    # LINE-BY-LINE: `start_text`에 `_normalize_range_bound(start_value, is_end=False)` 결과를 저장합니다. 의미/사용: `start_text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_text = _normalize_range_bound(start_value, is_end=False)
    # LINE-BY-LINE: `end_text`에 `_normalize_range_bound(end_value, is_end=True)` 결과를 저장합니다. 의미/사용: `end_text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    end_text = _normalize_range_bound(end_value, is_end=True)
    # LINE-BY-LINE: 조건 `not selected_jobs`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not selected_jobs:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][test_data_selection.select_actual_start_range_test_data] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][test_data_selection.select_actual_start_range_test_data] "
            # LINE-BY-LINE: `f"cause`에 `no_jobs_selected start={start_text} end={end_text}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=no_jobs_selected start={start_text} end={end_text}"
        )
        # LINE-BY-LINE: `ValueError("actual start range selected no jobs")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("actual start range selected no jobs")

    # LINE-BY-LINE: `range_scenario`에 `_scenario_with_jobs(` 결과를 저장합니다. 의미/사용: `range_scenario` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    range_scenario = _scenario_with_jobs(
        # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario,
        # LINE-BY-LINE: `_scenario_with_jobs(...)` 호출에 `selected_jobs` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        selected_jobs,
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `test_slice_name` 키에 `f"actual_start_{start_text}_{end_text}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_name": f"actual_start_{start_text}_{end_text}",
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `test_slice_basis` 키에 `"actual_start_datetime inclusive range"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_basis": "actual_start_datetime inclusive range",
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `test_slice_start` 키에 `start_text` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_start": start_text,
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `test_slice_end` 키에 `end_text` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_end": end_text,
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `test_slice_reason` 키에 `"user-requested actual start datetime range"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "test_slice_reason": "user-requested actual start datetime range",
            # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `actual_minute_base_datetime` 키에 `min(str(job.get("actual_start_datetime", "")) for job in selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_minute_base_datetime": min(str(job.get("actual_start_datetime", "")) for job in selected_jobs),
        },
    )
    # LINE-BY-LINE: `save_scenario(range_scenario, scenario_output_path)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    save_scenario(range_scenario, scenario_output_path)
    # LINE-BY-LINE: `job_rows`에 `_job_rows(range_scenario["jobs"], f"actual_start_{start_text}_{end_text}")` 결과를 저장합니다. 의미/사용: `job_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    job_rows = _job_rows(range_scenario["jobs"], f"actual_start_{start_text}_{end_text}")
    # LINE-BY-LINE: `_write_csv(job_rows, output_path / "actual_start_range_jobs.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(job_rows, output_path / "actual_start_range_jobs.csv")

    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `range_start` 키에 `start_text` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "range_start": start_text,
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `range_end` 키에 `end_text` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "range_end": end_text,
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `range_start_parse` 키에 `datetime.strptime(start_text, "%Y%m%d%H%M").isoformat(timespec="minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "range_start_parse": datetime.strptime(start_text, "%Y%m%d%H%M").isoformat(timespec="minutes"),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `range_end_parse` 키에 `datetime.strptime(end_text, "%Y%m%d%H%M").isoformat(timespec="minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "range_end_parse": datetime.strptime(end_text, "%Y%m%d%H%M").isoformat(timespec="minutes"),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `job_count` 키에 `len(selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "job_count": len(selected_jobs),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `expected_count` 키에 `expected_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "expected_count": expected_count,
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `first_actual_start_datetime` 키에 `min(str(job.get("actual_start_datetime", "")) for job in selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "first_actual_start_datetime": min(str(job.get("actual_start_datetime", "")) for job in selected_jobs),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `last_actual_start_datetime` 키에 `max(str(job.get("actual_start_datetime", "")) for job in selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "last_actual_start_datetime": max(str(job.get("actual_start_datetime", "")) for job in selected_jobs),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `first_actual_end_datetime` 키에 `min(str(job.get("actual_end_datetime", "")) for job in selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "first_actual_end_datetime": min(str(job.get("actual_end_datetime", "")) for job in selected_jobs),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `last_actual_end_datetime` 키에 `max(str(job.get("actual_end_datetime", "")) for job in selected_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "last_actual_end_datetime": max(str(job.get("actual_end_datetime", "")) for job in selected_jobs),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `scenario_output_path` 키에 `scenario_output_path` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "scenario_output_path": scenario_output_path,
        # LINE-BY-LINE: 딕셔너리 키 `output_dir`에는 `str(output_path)` 값을 넣습니다. 의미: 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        "output_dir": str(output_path),
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `time_parse_note` 키에 `"YYYYMMDDHHMM, e.g. 202602260906 = 2026-02-26 09:06."` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "time_parse_note": "YYYYMMDDHHMM, e.g. 202602260906 = 2026-02-26 09:06.",
        # LINE-BY-LINE: `select_actual_start_range_test_data`에서 반환/저장할 dict의 `actual_minutes_note` 키에 `"actual_start_minutes and actual_finish_minutes are rebased to the first selected actual start da...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_minutes_note": "actual_start_minutes and actual_finish_minutes are rebased to the first selected actual start datetime.",
    }
    # LINE-BY-LINE: `_write_json(summary, output_path / "actual_start_range_summary.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(summary, output_path / "actual_start_range_summary.json")
    # LINE-BY-LINE: 호출자에게 `{"summary": summary, "jobs": range_scenario["jobs"]}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return {"summary": summary, "jobs": range_scenario["jobs"]}
