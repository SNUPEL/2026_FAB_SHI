"""스케줄 결과를 CSV/JSON/HTML playback으로 내보낸다.

이 파일의 책임:
- generated simulation schedule을 사람이 볼 수 있는 CSV/JSON/HTML로 저장한다.
- actual replay row를 같은 schema로 저장한다.
- schedule row와 event log의 identity를 검증한다.
- playback HTML에서 시간 흐름, 설비 상태, W/O 상태를 볼 수 있게 데이터를 묶는다.

중요:
- validation 실패를 숨기지 않는다.
- actual replay의 핵심 검증 기준은 원본 실적 시간/장비/Bay identity가 0 오차인지다.
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
# LINE-BY-LINE: `collections` 모듈에서 `defaultdict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from collections import defaultdict
# LINE-BY-LINE: `datetime` 모듈에서 `datetime`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from datetime import datetime
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Optional`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Optional

# LINE-BY-LINE: `Environment.events` 모듈에서 `괄호 안의 여러 symbol`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.events import (
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_CUT_BAY_ASSIGN,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_MACHINE_ASSIGN,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_FINISH,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_PROCESS_START,
    # LINE-BY-LINE: `import(...)` 호출에 `EVENT_VALIDATION_FAILURE` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    EVENT_VALIDATION_FAILURE,
    # LINE-BY-LINE: `import(...)` 호출에 `FactoryEvent` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    FactoryEvent,
    # LINE-BY-LINE: `import(...)` 호출에 `SOURCE_ACTUAL` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_ACTUAL,
    # LINE-BY-LINE: `import(...)` 호출에 `SOURCE_VALIDATION` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    SOURCE_VALIDATION,
    # LINE-BY-LINE: `import(...)` 호출에 `event_sort_key` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    event_sort_key,
    # LINE-BY-LINE: `import(...)` 호출에 `validate_event_required_fields` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
    validate_event_required_fields,
)


# LINE-BY-LINE: `_write_csv(rows: Iterable[Dict[str, Any]], path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
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


# LINE-BY-LINE: `_write_json(data: Any, path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _write_json(data: Any, path: Path) -> None:
    """dict/list 데이터를 JSON으로 저장한다."""

    # LINE-BY-LINE: `path.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `path.write_text(json.dumps(data, ensure_ascii` 여러 변수에 `False, indent=2), encoding="utf-8")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# LINE-BY-LINE: `_write_event_log_csv(event_log_rows: Iterable[Dict[str, Any]], path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def _write_event_log_csv(event_log_rows: Iterable[Dict[str, Any]], path: Path) -> None:
    """event log를 CSV로 저장한다. payload는 JSON string으로 명시 변환한다."""

    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: `csv_rows`에 `[]` 결과를 저장합니다. 의미/사용: `csv_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        csv_rows = []
        # LINE-BY-LINE: `row in event_log_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in event_log_rows:
            # LINE-BY-LINE: `csv_row`에 `dict(row)` 결과를 저장합니다. 의미/사용: `csv_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            csv_row = dict(row)
            # LINE-BY-LINE: `payload`에 `csv_row.get("payload", {})` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload = csv_row.get("payload", {})
            # LINE-BY-LINE: 조건 `isinstance(payload, (dict, list))`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if isinstance(payload, (dict, list)):
                # LINE-BY-LINE: `csv_row["payload"]`에 `json.dumps(payload, ensure_ascii=False, sort_keys=True)` 결과를 저장합니다. 의미/사용: `csv_row["payload"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                csv_row["payload"] = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            # LINE-BY-LINE: `csv_rows.append(csv_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            csv_rows.append(csv_row)
        # LINE-BY-LINE: `_write_csv(csv_rows, path)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        _write_csv(csv_rows, path)
    # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except Exception as exc:
        # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][playback_builder._write_event_log_csv] cause={exc} path={path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"[ERROR][playback_builder._write_event_log_csv] cause={exc} path={path}")
        # LINE-BY-LINE: `RuntimeError(f"failed to write event log csv: {path}") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError(f"failed to write event log csv: {path}") from exc


# LINE-BY-LINE: `_day_key(simulation, start_time: float)` 함수를 정의합니다. 반환 타입: `str`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _day_key(simulation, start_time: float) -> str:
    """simulation 내부 minute를 날짜 key로 변환한다."""

    # LINE-BY-LINE: 호출자에게 `simulation._current_day_key(at_time=start_time) # noqa: SLF001 - export utility`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return simulation._current_day_key(at_time=start_time)  # noqa: SLF001 - export utility


# LINE-BY-LINE: `_format_compact_date(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _format_compact_date(value: Any) -> str:
    """YYYYMMDD 계열 문자열을 YYYY-MM-DD 표시용 문자열로 변환한다."""

    # LINE-BY-LINE: `text`에 `str(value or "").strip()` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    text = str(value or "").strip()
    # LINE-BY-LINE: 조건 `len(text) >= 8 and text[:8].isdigit()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(text) >= 8 and text[:8].isdigit():
        # LINE-BY-LINE: 호출자에게 `f"{text[:4]}-{text[4:6]}-{text[6:8]}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    # LINE-BY-LINE: 호출자에게 `""`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return ""


# LINE-BY-LINE: `_parse_compact_datetime_for_display(value: Any)` 함수를 정의합니다. 반환 타입: `Optional[datetime]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _parse_compact_datetime_for_display(value: Any) -> Optional[datetime]:
    """compact datetime을 playback 표시용 datetime으로 변환한다."""

    # LINE-BY-LINE: `text`에 `str(value or "").strip()` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    text = str(value or "").strip()
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


# LINE-BY-LINE: `_build_playback_time_axis` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _build_playback_time_axis(
    # LINE-BY-LINE: `events`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `schedule_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation: Dict[str, Any],
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """event와 schedule row 기준 playback 시간축 정보를 만든다."""

    # LINE-BY-LINE: `event_rows`에 `list(events)` 결과를 저장합니다. 의미/사용: `event_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_rows = list(events)
    # LINE-BY-LINE: `rows`에 `list(schedule_rows)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(schedule_rows)
    # LINE-BY-LINE: `event_seconds`에 `[` 결과를 저장합니다. 의미/사용: `event_seconds` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_seconds = [
        # LINE-BY-LINE: `int(round(float(event["time_sec"])))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        int(round(float(event["time_sec"])))
        # LINE-BY-LINE: `event in event_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for event in event_rows
        # LINE-BY-LINE: 조건 `_to_float_or_none(event.get("time_sec")) is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if _to_float_or_none(event.get("time_sec")) is not None
    ]
    # LINE-BY-LINE: `schedule_start_seconds`에 `[` 결과를 저장합니다. 의미/사용: `schedule_start_seconds` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_start_seconds = [
        # LINE-BY-LINE: `int(round(float(row["start_min"]) * 60))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        int(round(float(row["start_min"]) * 60))
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows
        # LINE-BY-LINE: 조건 `_to_float_or_none(row.get("start_min")) is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if _to_float_or_none(row.get("start_min")) is not None
    ]
    # LINE-BY-LINE: `schedule_finish_seconds`에 `[` 결과를 저장합니다. 의미/사용: `schedule_finish_seconds` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_finish_seconds = [
        # LINE-BY-LINE: `int(round(float(row["finish_min"]) * 60))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        int(round(float(row["finish_min"]) * 60))
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows
        # LINE-BY-LINE: 조건 `_to_float_or_none(row.get("finish_min")) is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if _to_float_or_none(row.get("finish_min")) is not None
    ]
    # LINE-BY-LINE: `all_seconds`에 `[*event_seconds, *schedule_start_seconds, *schedule_finish_seconds]` 결과를 저장합니다. 의미/사용: `all_seconds` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    all_seconds = [*event_seconds, *schedule_start_seconds, *schedule_finish_seconds]
    # LINE-BY-LINE: `min_time_sec`에 `min(all_seconds) if all_seconds else 0` 결과를 저장합니다. 의미/사용: `min_time_sec` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    min_time_sec = min(all_seconds) if all_seconds else 0
    # LINE-BY-LINE: `max_time_sec`에 `max(all_seconds) if all_seconds else 1` 결과를 저장합니다. 의미/사용: `max_time_sec` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_time_sec = max(all_seconds) if all_seconds else 1
    # LINE-BY-LINE: `is_actual_replay`에 `validation.get("validation_type") == "actual_factory_replay_identity"` 결과를 저장합니다. 의미/사용: `is_actual_replay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    is_actual_replay = validation.get("validation_type") == "actual_factory_replay_identity"
    # LINE-BY-LINE: `axis`에 `{` 결과를 저장합니다. 의미/사용: `axis` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    axis = {
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `min_time_sec` 키에 `min_time_sec` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "min_time_sec": min_time_sec,
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `max_time_sec` 키에 `max(max_time_sec, min_time_sec + 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_time_sec": max(max_time_sec, min_time_sec + 1),
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `initial_time_sec` 키에 `min_time_sec` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "initial_time_sec": min_time_sec,
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `mode` 키에 `"relative"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mode": "relative",
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_base_datetime` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "display_base_datetime": "",
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_base_time_sec` 키에 `min_time_sec` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "display_base_time_sec": min_time_sec,
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_note` 키에 `"relative simulation time"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "display_note": "relative simulation time",
        # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `actual_datetime_source` 키에 `"none"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_datetime_source": "none",
    }

    # LINE-BY-LINE: `actual_values`에 `[` 결과를 저장합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_values = [
        # LINE-BY-LINE: `(row, str(row.get("actual_start_datetime", "")).strip())`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        (row, str(row.get("actual_start_datetime", "")).strip())
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows
        # LINE-BY-LINE: 조건 `str(row.get("actual_start_datetime", "")).strip()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if str(row.get("actual_start_datetime", "")).strip()
    ]
    # LINE-BY-LINE: 조건 `not actual_values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not actual_values:
        # LINE-BY-LINE: 호출자에게 `axis`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return axis

    # LINE-BY-LINE: `candidates`에 `[]` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
    candidates = []
    # LINE-BY-LINE: `invalid_values`에 `[]` 결과를 저장합니다. 의미/사용: `invalid_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    invalid_values = []
    # LINE-BY-LINE: `missing_actual_job_ids`에 `[]` 결과를 저장합니다. 의미/사용: `missing_actual_job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    missing_actual_job_ids = []
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `raw_value`에 `str(row.get("actual_start_datetime", "")).strip()` 결과를 저장합니다. 의미/사용: `raw_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        raw_value = str(row.get("actual_start_datetime", "")).strip()
        # LINE-BY-LINE: 조건 `not raw_value`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not raw_value:
            # LINE-BY-LINE: `missing_actual_job_ids.append(str(row.get("job_id", "")))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            missing_actual_job_ids.append(str(row.get("job_id", "")))
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `actual_start_dt`에 `_parse_compact_datetime_for_display(raw_value)` 결과를 저장합니다. 의미/사용: `actual_start_dt` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start_dt = _parse_compact_datetime_for_display(raw_value)
        # LINE-BY-LINE: 조건 `actual_start_dt is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_start_dt is None:
            # LINE-BY-LINE: `invalid_values.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            invalid_values.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": row.get("job_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                    "work_order_no": row.get("work_order_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `raw_value` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                    "actual_start_datetime": raw_value,
                }
            )
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `candidates.append((actual_start_dt, row))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        candidates.append((actual_start_dt, row))

    # LINE-BY-LINE: 조건 `invalid_values or missing_actual_job_ids`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if invalid_values or missing_actual_job_ids:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._build_playback_time_axis] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder._build_playback_time_axis] "
            # LINE-BY-LINE: `f"cause`에 `incomplete_or_invalid_actual_start_datetime "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=incomplete_or_invalid_actual_start_datetime "
            # LINE-BY-LINE: `f"invalid_count`에 `{len(invalid_values)} missing_count={len(missing_actual_job_ids)} "` 결과를 저장합니다. 의미/사용: `f"invalid_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"invalid_count={len(invalid_values)} missing_count={len(missing_actual_job_ids)} "
            # LINE-BY-LINE: `f"invalid_samples`에 `{invalid_values[:5]} missing_samples={missing_actual_job_ids[:5]}"` 결과를 저장합니다. 의미/사용: `f"invalid_samples` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"invalid_samples={invalid_values[:5]} missing_samples={missing_actual_job_ids[:5]}"
        )
        # LINE-BY-LINE: `RuntimeError("playback has incomplete or invalid actual_start_datetime values")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("playback has incomplete or invalid actual_start_datetime values")

    # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not candidates:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._build_playback_time_axis] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder._build_playback_time_axis] "
            # LINE-BY-LINE: `"cause`에 `missing_parseable_actual_start_datetime"` 결과를 저장합니다. 의미/사용: `"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            "cause=missing_parseable_actual_start_datetime"
        )
        # LINE-BY-LINE: `RuntimeError("playback HTML requires parseable actual_start_datetime when actual datetimes exist")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("playback HTML requires parseable actual_start_datetime when actual datetimes exist")

    # LINE-BY-LINE: `base_dt, base_row` 여러 변수에 `min(candidates, key=lambda item: item[0])` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    base_dt, base_row = min(candidates, key=lambda item: item[0])
    # LINE-BY-LINE: `axis.update(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    axis.update(
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `mode` 키에 `"actual_datetime"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mode": "actual_datetime",
            # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_base_datetime` 키에 `base_dt.strftime("%Y-%m-%dT%H:%M:%S")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "display_base_datetime": base_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_base_time_sec` 키에 `min_time_sec` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "display_base_time_sec": min_time_sec,
            # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `actual_datetime_source` 키에 `"actual_replay" if is_actual_replay else "generated_schedule_anchor"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_datetime_source": "actual_replay" if is_actual_replay else "generated_schedule_anchor",
            # LINE-BY-LINE: `_build_playback_time_axis`에서 반환/저장할 dict의 `display_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "display_note": (
                # LINE-BY-LINE: 문자열 값 `"clock displays calendar time anchored to the earliest RT_CUT_ST_DTM in this scenario; "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "clock displays calendar time anchored to the earliest RT_CUT_ST_DTM in this scenario; "
                # LINE-BY-LINE: `f"base_job_id`에 `{base_row.get('job_id', '')}"` 결과를 저장합니다. 의미/사용: `f"base_job_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"base_job_id={base_row.get('job_id', '')}"
            ),
        }
    )
    # LINE-BY-LINE: 호출자에게 `axis`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return axis


# LINE-BY-LINE: `_to_float_or_none(value: Any)` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _to_float_or_none(value: Any) -> Optional[float]:
    """값을 float로 바꾸되 표시/집계용 결측은 None으로 반환한다."""

    # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value in (None, ""):
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 호출자에게 `float(value)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return float(value)
    # LINE-BY-LINE: `except (TypeError, ValueError):` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except (TypeError, ValueError):
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None


# LINE-BY-LINE: `_pearson_corr(pairs: Iterable[tuple[float, float]])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _pearson_corr(pairs: Iterable[tuple[float, float]]) -> Optional[float]:
    """(x, y) pair 목록의 Pearson 상관계수를 계산한다."""

    # LINE-BY-LINE: `pairs`에 `list(pairs)` 결과를 저장합니다. 의미/사용: `pairs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    pairs = list(pairs)
    # LINE-BY-LINE: 조건 `len(pairs) < 2`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(pairs) < 2:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `xs`에 `[pair[0] for pair in pairs]` 결과를 저장합니다. 의미/사용: `xs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    xs = [pair[0] for pair in pairs]
    # LINE-BY-LINE: `ys`에 `[pair[1] for pair in pairs]` 결과를 저장합니다. 의미/사용: `ys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    ys = [pair[1] for pair in pairs]
    # LINE-BY-LINE: `mean_x`에 `statistics.mean(xs)` 결과를 저장합니다. 의미/사용: `mean_x` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mean_x = statistics.mean(xs)
    # LINE-BY-LINE: `mean_y`에 `statistics.mean(ys)` 결과를 저장합니다. 의미/사용: `mean_y` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mean_y = statistics.mean(ys)
    # LINE-BY-LINE: `variance_x`에 `sum((value - mean_x) ** 2 for value in xs)` 결과를 저장합니다. 의미/사용: `variance_x` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    variance_x = sum((value - mean_x) ** 2 for value in xs)
    # LINE-BY-LINE: `variance_y`에 `sum((value - mean_y) ** 2 for value in ys)` 결과를 저장합니다. 의미/사용: `variance_y` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    variance_y = sum((value - mean_y) ** 2 for value in ys)
    # LINE-BY-LINE: 조건 `variance_x <= 0 or variance_y <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if variance_x <= 0 or variance_y <= 0:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `covariance`에 `sum((x - mean_x) * (y - mean_y) for x, y in pairs)` 결과를 저장합니다. 의미/사용: `covariance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    # LINE-BY-LINE: 호출자에게 `covariance / math.sqrt(variance_x * variance_y)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return covariance / math.sqrt(variance_x * variance_y)


# LINE-BY-LINE: `_join_unique(values: Iterable[Any])` 함수를 정의합니다. 반환 타입: `str`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _join_unique(values: Iterable[Any]) -> str:
    """중복과 빈 값을 제거한 뒤 `|`로 연결한다."""

    # LINE-BY-LINE: `unique_values`에 `[]` 결과를 저장합니다. 의미/사용: `unique_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    unique_values = []
    # LINE-BY-LINE: `seen`에 `set()` 결과를 저장합니다. 의미/사용: `seen` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    seen = set()
    # LINE-BY-LINE: `value in values` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for value in values:
        # LINE-BY-LINE: `text`에 `str(value or "").strip()` 결과를 저장합니다. 의미/사용: `text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        text = str(value or "").strip()
        # LINE-BY-LINE: 조건 `not text or text in seen`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not text or text in seen:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `seen.add(text)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        seen.add(text)
        # LINE-BY-LINE: `unique_values.append(text)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        unique_values.append(text)
    # LINE-BY-LINE: 호출자에게 `"|".join(unique_values)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return "|".join(unique_values)


# LINE-BY-LINE: `_machine_day_buckets(rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _machine_day_buckets(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """schedule row를 날짜/설비/Bay bucket으로 묶어 요약한다."""

    # LINE-BY-LINE: `buckets` 변수에 `{}` 결과를 저장합니다. 의미: `buckets` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    buckets: Dict[tuple, Dict[str, Any]] = {}
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `bucket_date`에 `row.get("capacity_date") or row["scheduled_day"]` 결과를 저장합니다. 의미/사용: `bucket_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket_date = row.get("capacity_date") or row["scheduled_day"]
        # LINE-BY-LINE: `machine_id`에 `str(row["algorithm_machine_id"])` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id = str(row["algorithm_machine_id"])
        # LINE-BY-LINE: `bay_id`에 `str(row["algorithm_cut_bay"])` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(row["algorithm_cut_bay"])
        # LINE-BY-LINE: `key`에 `(bucket_date, machine_id, bay_id)` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key = (bucket_date, machine_id, bay_id)
        # LINE-BY-LINE: `bucket`에 `buckets.setdefault(` 결과를 저장합니다. 의미/사용: `bucket` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket = buckets.setdefault(
            # LINE-BY-LINE: `setdefault(...)` 호출에 `key` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            key,
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_machine_day_buckets`에서 반환/저장할 dict의 `date` 키에 `bucket_date` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "date": bucket_date,
                # LINE-BY-LINE: `_machine_day_buckets`에서 반환/저장할 dict의 `date_basis` 키에 `row.get("capacity_basis", "scheduled_day")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "date_basis": row.get("capacity_basis", "scheduled_day"),
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": machine_id,
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `bay_id` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": bay_id,
                # LINE-BY-LINE: `_machine_day_buckets`에서 반환/저장할 dict의 `num_jobs` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "num_jobs": 0,
                # LINE-BY-LINE: `_machine_day_buckets`에서 반환/저장할 dict의 `total_length` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_length": 0.0,
                # LINE-BY-LINE: `_machine_day_buckets`에서 반환/저장할 dict의 `total_process_time` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_process_time": 0.0,
                # LINE-BY-LINE: 딕셔너리 키 `rows`에는 `[]` 값을 넣습니다. 의미: CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
                "rows": [],
            },
        )
        # LINE-BY-LINE: `bucket["num_jobs"]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `bucket["num_jobs"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["num_jobs"] += 1
        # LINE-BY-LINE: `bucket["total_length"]` 값을 `_to_float_or_none(row.get("plate_length")) or 0.0` 기준으로 누적/증가합니다. 의미/사용: `bucket["total_length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["total_length"] += _to_float_or_none(row.get("plate_length")) or 0.0
        # LINE-BY-LINE: `bucket["total_process_time"]` 값을 `_to_float_or_none(row.get("process_min")) or 0.0` 기준으로 누적/증가합니다. 의미/사용: `bucket["total_process_time"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["total_process_time"] += _to_float_or_none(row.get("process_min")) or 0.0
        # LINE-BY-LINE: `bucket["rows"].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        bucket["rows"].append(row)

    # LINE-BY-LINE: `bucket in buckets.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bucket in buckets.values():
        # LINE-BY-LINE: `bucket_rows`에 `bucket["rows"]` 결과를 저장합니다. 의미/사용: `bucket_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket_rows = bucket["rows"]
        # LINE-BY-LINE: `bucket["job_ids"]`에 `_join_unique(row.get("job_id") for row in bucket_rows)` 결과를 저장합니다. 의미/사용: `bucket["job_ids"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["job_ids"] = _join_unique(row.get("job_id") for row in bucket_rows)
        # LINE-BY-LINE: `bucket["work_order_nos"]`에 `_join_unique(row.get("work_order_no") for row in bucket_rows)` 결과를 저장합니다. 의미/사용: `bucket["work_order_nos"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["work_order_nos"] = _join_unique(row.get("work_order_no") for row in bucket_rows)
        # LINE-BY-LINE: `bucket["first_start_min"]`에 `min((_to_float_or_none(row.get("start_min")) or 0.0 for row in bucket_rows), default=0.0)` 결과를 저장합니다. 의미/사용: `bucket["first_start_min"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["first_start_min"] = min((_to_float_or_none(row.get("start_min")) or 0.0 for row in bucket_rows), default=0.0)
        # LINE-BY-LINE: `bucket["last_finish_min"]`에 `max((_to_float_or_none(row.get("finish_min")) or 0.0 for row in bucket_rows), default=0.0)` 결과를 저장합니다. 의미/사용: `bucket["last_finish_min"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["last_finish_min"] = max((_to_float_or_none(row.get("finish_min")) or 0.0 for row in bucket_rows), default=0.0)

    # LINE-BY-LINE: 호출자에게 `list(buckets.values())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return list(buckets.values())


# LINE-BY-LINE: `_machine_concurrent_capacity_intervals` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _machine_concurrent_capacity_intervals(
    # LINE-BY-LINE: `rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `max_wo_count`를 `Optional[int],` 타입으로 선언합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_wo_count: Optional[int],
    # LINE-BY-LINE: `max_length_sum`를 `Optional[float],` 타입으로 선언합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_length_sum: Optional[float],
# LINE-BY-LINE: `) -> List[Dict[str, Any]]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> List[Dict[str, Any]]:
    """같은 설비에서 동시에 active인 W/O 묶음의 capacity 위반 interval."""

    # LINE-BY-LINE: `events_by_machine` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `events_by_machine` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events_by_machine: Dict[str, List[tuple[float, int, Dict[str, Any]]]] = defaultdict(list)
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `machine_id`에 `str(row.get("algorithm_machine_id", ""))` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id = str(row.get("algorithm_machine_id", ""))
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: 조건 `not machine_id or start_min is None or finish_min is None or finish_min <= start_min`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not machine_id or start_min is None or finish_min is None or finish_min <= start_min:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `events_by_machine[machine_id].append((start_min, 1, row))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events_by_machine[machine_id].append((start_min, 1, row))
        # LINE-BY-LINE: `events_by_machine[machine_id].append((finish_min, -1, row))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events_by_machine[machine_id].append((finish_min, -1, row))

    # LINE-BY-LINE: `violations` 변수에 `[]` 결과를 저장합니다. 의미: `violations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violations: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `machine_id, events in sorted(events_by_machine.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for machine_id, events in sorted(events_by_machine.items()):
        # LINE-BY-LINE: `events`에 `sorted(events, key=lambda item: (item[0], item[1]))` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        events = sorted(events, key=lambda item: (item[0], item[1]))
        # LINE-BY-LINE: `active` 변수에 `{}` 결과를 저장합니다. 의미: `active` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active: Dict[str, Dict[str, Any]] = {}
        # LINE-BY-LINE: `index`에 `0` 결과를 저장합니다. 의미/사용: `index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        index = 0
        # LINE-BY-LINE: 조건 `index < len(events)`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while index < len(events):
            # LINE-BY-LINE: `event_time`에 `events[index][0]` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            event_time = events[index][0]
            # LINE-BY-LINE: `same_time_events`에 `[]` 결과를 저장합니다. 의미/사용: `same_time_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            same_time_events = []
            # LINE-BY-LINE: 조건 `index < len(events) and events[index][0] == event_time`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
            while index < len(events) and events[index][0] == event_time:
                # LINE-BY-LINE: `same_time_events.append(events[index])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                same_time_events.append(events[index])
                # LINE-BY-LINE: `index` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                index += 1

            # [start, finish) 기준이므로 같은 시각에서는 finish를 먼저 제거한다.
            # LINE-BY-LINE: `_, delta, row in same_time_events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for _, delta, row in same_time_events:
                # LINE-BY-LINE: 조건 `delta < 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if delta < 0:
                    # LINE-BY-LINE: `active.pop(str(row.get("job_id", "")), None)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    active.pop(str(row.get("job_id", "")), None)
            # LINE-BY-LINE: `_, delta, row in same_time_events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for _, delta, row in same_time_events:
                # LINE-BY-LINE: 조건 `delta > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if delta > 0:
                    # LINE-BY-LINE: `active[str(row.get("job_id", ""))]` 여러 변수에 `row` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                    active[str(row.get("job_id", ""))] = row

            # LINE-BY-LINE: `next_time`에 `events[index][0] if index < len(events) else event_time` 결과를 저장합니다. 의미/사용: `next_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            next_time = events[index][0] if index < len(events) else event_time
            # LINE-BY-LINE: 조건 `next_time <= event_time or not active`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if next_time <= event_time or not active:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue

            # LINE-BY-LINE: `active_rows`에 `list(active.values())` 결과를 저장합니다. 의미/사용: `active_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_rows = list(active.values())
            # LINE-BY-LINE: `active_count`에 `len(active_rows)` 결과를 저장합니다. 의미/사용: `active_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_count = len(active_rows)
            # LINE-BY-LINE: `active_length`에 `sum(_to_float_or_none(row.get("plate_length")) or 0.0 for row in active_rows)` 결과를 저장합니다. 의미/사용: `active_length` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_length = sum(_to_float_or_none(row.get("plate_length")) or 0.0 for row in active_rows)
            # LINE-BY-LINE: `bay_ids`에 `_join_unique(row.get("algorithm_cut_bay") for row in active_rows)` 결과를 저장합니다. 의미/사용: `bay_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bay_ids = _join_unique(row.get("algorithm_cut_bay") for row in active_rows)
            # LINE-BY-LINE: `violation_base`에 `{` 결과를 저장합니다. 의미/사용: `violation_base` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            violation_base = {
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": machine_id,
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `bay_ids` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": bay_ids,
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `start_min` 키에 `round(event_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_min": round(event_time, 6),
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `finish_min` 키에 `round(next_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_min": round(next_time, 6),
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `active_job_count` 키에 `active_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "active_job_count": active_count,
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `active_length_sum` 키에 `round(active_length, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "active_length_sum": round(active_length, 6),
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `job_ids` 키에 `_join_unique(row.get("job_id") for row in active_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "job_ids": _join_unique(row.get("job_id") for row in active_rows),
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `work_order_nos` 키에 `_join_unique(row.get("work_order_no") for row in active_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "work_order_nos": _join_unique(row.get("work_order_no") for row in active_rows),
                # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `_active_rows` 키에 `active_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "_active_rows": active_rows,
            }
            # LINE-BY-LINE: 조건 `max_wo_count is not None and active_count > int(max_wo_count)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if max_wo_count is not None and active_count > int(max_wo_count):
                # LINE-BY-LINE: `violations.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                violations.append(
                    # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    {
                        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**violation_base` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        **violation_base,
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `rule` 키에 `"machine_concurrent_wo_count_limit"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "rule": "machine_concurrent_wo_count_limit",
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `max_wo_count` 키에 `int(max_wo_count)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "max_wo_count": int(max_wo_count),
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `excess_wo_count` 키에 `active_count - int(max_wo_count)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "excess_wo_count": active_count - int(max_wo_count),
                        # LINE-BY-LINE: 딕셔너리 키 `message`에는 `f"{active_count}>{int(max_wo_count)}"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                        "message": f"{active_count}>{int(max_wo_count)}",
                    }
                )
            # LINE-BY-LINE: 조건 `max_length_sum is not None and active_length > float(max_length_sum)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if max_length_sum is not None and active_length > float(max_length_sum):
                # LINE-BY-LINE: `violations.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                violations.append(
                    # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    {
                        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `**violation_base` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        **violation_base,
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `rule` 키에 `"machine_concurrent_length_sum_limit"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "rule": "machine_concurrent_length_sum_limit",
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `max_length_sum` 키에 `float(max_length_sum)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "max_length_sum": float(max_length_sum),
                        # LINE-BY-LINE: `_machine_concurrent_capacity_intervals`에서 반환/저장할 dict의 `excess_length` 키에 `round(active_length - float(max_length_sum), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "excess_length": round(active_length - float(max_length_sum), 6),
                        # LINE-BY-LINE: 딕셔너리 키 `message`에는 `f"{active_length:.2f}>{float(max_length_sum):.2f}"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                        "message": f"{active_length:.2f}>{float(max_length_sum):.2f}",
                    }
                )

    # LINE-BY-LINE: 호출자에게 `violations`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return violations


# LINE-BY-LINE: `build_concurrent_capacity_audit(schedule_rows: Iterable[Dict[str, Any]], config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_concurrent_capacity_audit(schedule_rows: Iterable[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """동일 설비 active interval audit을 interval/detail/summary로 분리한다.

    현업 확인 후 generated planning 기본 제약은 rolling active interval이 아니라 batch다.
    이 audit은 actual replay의 timestamp 품질 진단과 legacy 비교용으로만 사용한다.
    """

    # LINE-BY-LINE: `limits`에 `config.get("constraints", {}).get("machine_day_limits", {}) or {}` 결과를 저장합니다. 의미/사용: `limits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    limits = config.get("constraints", {}).get("machine_day_limits", {}) or {}
    # LINE-BY-LINE: `max_wo_count`에 `limits.get("max_wo_count")` 결과를 저장합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_wo_count = limits.get("max_wo_count")
    # LINE-BY-LINE: `max_length_sum`에 `limits.get("max_length_sum")` 결과를 저장합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_length_sum = limits.get("max_length_sum")
    # LINE-BY-LINE: `raw_intervals`에 `_machine_concurrent_capacity_intervals(schedule_rows, max_wo_count, max_length_sum)` 결과를 저장합니다. 의미/사용: `raw_intervals` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    raw_intervals = _machine_concurrent_capacity_intervals(schedule_rows, max_wo_count, max_length_sum)
    # LINE-BY-LINE: `interval_rows` 변수에 `[]` 결과를 저장합니다. 의미: `interval_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    interval_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `active_job_rows` 변수에 `[]` 결과를 저장합니다. 의미: `active_job_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    active_job_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `count_by_rule` 변수에 `defaultdict(int)` 결과를 저장합니다. 의미: `count_by_rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    count_by_rule: Dict[str, int] = defaultdict(int)
    # LINE-BY-LINE: `count_by_machine` 변수에 `defaultdict(int)` 결과를 저장합니다. 의미: `count_by_machine` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    count_by_machine: Dict[str, int] = defaultdict(int)

    # LINE-BY-LINE: `index, raw_interval in enumerate(raw_intervals, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for index, raw_interval in enumerate(raw_intervals, start=1):
        # LINE-BY-LINE: `active_rows`에 `raw_interval.get("_active_rows", [])` 결과를 저장합니다. 의미/사용: `active_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_rows = raw_interval.get("_active_rows", [])
        # LINE-BY-LINE: `interval_id`에 `f"concurrent_capacity_{index:04d}"` 결과를 저장합니다. 의미/사용: `interval_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        interval_id = f"concurrent_capacity_{index:04d}"
        # LINE-BY-LINE: `interval_row`에 `{` 결과를 저장합니다. 의미/사용: `interval_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        interval_row = {
            # LINE-BY-LINE: `key`를 `value` 타입으로 선언합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key: value
            # LINE-BY-LINE: `key, value in raw_interval.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key, value in raw_interval.items()
            # LINE-BY-LINE: 조건 `key != "_active_rows"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key != "_active_rows"
        }
        # LINE-BY-LINE: `interval_row["interval_id"]`에 `interval_id` 결과를 저장합니다. 의미/사용: `interval_row["interval_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        interval_row["interval_id"] = interval_id
        # LINE-BY-LINE: `interval_rows.append(interval_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        interval_rows.append(interval_row)
        # LINE-BY-LINE: `count_by_rule[str(interval_row.get("rule", ""))]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `get("rule", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        count_by_rule[str(interval_row.get("rule", ""))] += 1
        # LINE-BY-LINE: `count_by_machine[str(interval_row.get("machine_id", ""))]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `get("machine_id", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        count_by_machine[str(interval_row.get("machine_id", ""))] += 1

        # LINE-BY-LINE: `active_row in active_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for active_row in active_rows:
            # LINE-BY-LINE: `active_job_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            active_job_rows.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `interval_id` 키에 `interval_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "interval_id": interval_id,
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `rule` 키에 `interval_row.get("rule", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "rule": interval_row.get("rule", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `interval_row.get("machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": interval_row.get("machine_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `interval_row.get("bay_id", "")` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    "bay_id": interval_row.get("bay_id", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `interval_start_min` 키에 `interval_row.get("start_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "interval_start_min": interval_row.get("start_min", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `interval_finish_min` 키에 `interval_row.get("finish_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "interval_finish_min": interval_row.get("finish_min", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `active_job_count` 키에 `interval_row.get("active_job_count", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "active_job_count": interval_row.get("active_job_count", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `active_length_sum` 키에 `interval_row.get("active_length_sum", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "active_length_sum": interval_row.get("active_length_sum", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `active_row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": active_row.get("job_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `active_row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                    "work_order_no": active_row.get("work_order_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `active_row.get("project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                    "project_no": active_row.get("project_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `active_row.get("block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                    "block_no": active_row.get("block_no", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `job_start_min` 키에 `active_row.get("start_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "job_start_min": active_row.get("start_min", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `job_finish_min` 키에 `active_row.get("finish_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "job_finish_min": active_row.get("finish_min", ""),
                    # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `job_process_min` 키에 `active_row.get("process_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "job_process_min": active_row.get("process_min", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `active_row.get("plate_length", "")` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                    "plate_length": active_row.get("plate_length", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `active_row.get("thickness", "")` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                    "thickness": active_row.get("thickness", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `active_row.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                    "actual_start_datetime": active_row.get("actual_start_datetime", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `active_row.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                    "actual_end_datetime": active_row.get("actual_end_datetime", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `planned_start_date`에는 `active_row.get("planned_start_date", "")` 값을 넣습니다. 의미: 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
                    "planned_start_date": active_row.get("planned_start_date", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `planned_end_date`에는 `active_row.get("planned_end_date", "")` 값을 넣습니다. 의미: 계획 종료일 원본입니다. 예: `GYEL_ACT_EDDT`, 사용: 계획일 기반 검증 후보.
                    "planned_end_date": active_row.get("planned_end_date", ""),
                }
            )

    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `interval_count` 키에 `len(interval_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "interval_count": len(interval_rows),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `active_job_detail_row_count` 키에 `len(active_job_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "active_job_detail_row_count": len(active_job_rows),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `counts_by_rule` 키에 `dict(sorted(count_by_rule.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "counts_by_rule": dict(sorted(count_by_rule.items())),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `counts_by_machine` 키에 `dict(sorted(count_by_machine.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "counts_by_machine": dict(sorted(count_by_machine.items())),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `max_active_job_count` 키에 `max((int(row["active_job_count"]) for row in interval_rows), default=0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_active_job_count": max((int(row["active_job_count"]) for row in interval_rows), default=0),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `max_active_length_sum` 키에 `max((float(row["active_length_sum"]) for row in interval_rows), default=0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_active_length_sum": max((float(row["active_length_sum"]) for row in interval_rows), default=0.0),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `max_length_sum` 키에 `None if max_length_sum in (None, "") else float(max_length_sum)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_length_sum": None if max_length_sum in (None, "") else float(max_length_sum),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `max_wo_count` 키에 `None if max_wo_count in (None, "") else int(max_wo_count)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_wo_count": None if max_wo_count in (None, "") else int(max_wo_count),
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `interpretation_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "interpretation_note": (
            # LINE-BY-LINE: 문자열 값 `"This audit checks active interval overlap by machine for data-quality diagnosis. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "This audit checks active interval overlap by machine for data-quality diagnosis. "
            # LINE-BY-LINE: 문자열 값 `"Confirmed generated planning uses batch constraints: W/O count <=3 and length sum <=55000 per closed batch. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Confirmed generated planning uses batch constraints: W/O count <=3 and length sum <=55000 per closed batch. "
            # LINE-BY-LINE: 문자열 값 `"Identical actual start/end timestamps across many W/O are treated as source timestamp errors, not proof of rolling operation."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Identical actual start/end timestamps across many W/O are treated as source timestamp errors, not proof of rolling operation."
        ),
    }
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `intervals` 키에 `interval_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "intervals": interval_rows,
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `active_jobs` 키에 `active_job_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "active_jobs": active_job_rows,
        # LINE-BY-LINE: `build_concurrent_capacity_audit`에서 반환/저장할 dict의 `summary` 키에 `summary` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "summary": summary,
    }


# LINE-BY-LINE: `build_same_machine_time_groups` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_same_machine_time_groups(
    # LINE-BY-LINE: `schedule_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `min_group_size` 변수에 `2` 결과를 저장합니다. 의미: `min_group_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    min_group_size: int = 2,
# LINE-BY-LINE: `) -> List[Dict[str, Any]]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> List[Dict[str, Any]]:
    """같은 설비에 동일 실적 시작/종료시각이 반복된 W/O 그룹."""

    # LINE-BY-LINE: `groups` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    groups: Dict[tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `key`에 `(` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key = (
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row.get("algorithm_machine_id", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row.get("algorithm_machine_id", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row.get("algorithm_cut_bay", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row.get("algorithm_cut_bay", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row.get("actual_start_datetime", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row.get("actual_start_datetime", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row.get("actual_end_datetime", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row.get("actual_end_datetime", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row.get("start_min", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row.get("start_min", "")),
        )
        # LINE-BY-LINE: `groups[key].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        groups[key].append(row)

    # LINE-BY-LINE: `output_rows` 변수에 `[]` 결과를 저장합니다. 의미: `output_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `index, ((machine_id, bay_id, start_dt, end_dt, start_min), rows) in enumerate(` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for index, ((machine_id, bay_id, start_dt, end_dt, start_min), rows) in enumerate(
        # LINE-BY-LINE: `sorted(groups.items(), key` 여러 변수에 `lambda item: (item[0][0], item[0][2], item[0][3], item[0][4]))` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        sorted(groups.items(), key=lambda item: (item[0][0], item[0][2], item[0][3], item[0][4])),
        # LINE-BY-LINE: `start`에 `1` 결과를 저장합니다. 의미/사용: `start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start=1,
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        # LINE-BY-LINE: 조건 `len(rows) < min_group_size`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(rows) < min_group_size:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `finish_values`에 `_join_unique(row.get("finish_min") for row in rows)` 결과를 저장합니다. 의미/사용: `finish_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_values = _join_unique(row.get("finish_min") for row in rows)
        # LINE-BY-LINE: `output_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        output_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `same_time_group_id` 키에 `f"same_machine_time_{index:04d}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "same_time_group_id": f"same_machine_time_{index:04d}",
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": machine_id,
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `bay_id` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": bay_id,
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `start_dt` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": start_dt,
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `end_dt` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": end_dt,
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `start_min` 키에 `start_min` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_min": start_min,
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `finish_min_values` 키에 `finish_values` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_min_values": finish_values,
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `group_size` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "group_size": len(rows),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `total_length` 키에 `round(sum(_to_float_or_none(row.get("plate_length")) or 0.0 for row in rows), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_length": round(sum(_to_float_or_none(row.get("plate_length")) or 0.0 for row in rows), 6),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `min_process_min` 키에 `min((_to_float_or_none(row.get("process_min")) or 0.0 for row in rows), default=0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "min_process_min": min((_to_float_or_none(row.get("process_min")) or 0.0 for row in rows), default=0.0),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `max_process_min` 키에 `max((_to_float_or_none(row.get("process_min")) or 0.0 for row in rows), default=0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "max_process_min": max((_to_float_or_none(row.get("process_min")) or 0.0 for row in rows), default=0.0),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `work_order_nos` 키에 `_join_unique(row.get("work_order_no") for row in rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "work_order_nos": _join_unique(row.get("work_order_no") for row in rows),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `job_ids` 키에 `_join_unique(row.get("job_id") for row in rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "job_ids": _join_unique(row.get("job_id") for row in rows),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `planned_start_dates` 키에 `_join_unique(row.get("planned_start_date") for row in rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_start_dates": _join_unique(row.get("planned_start_date") for row in rows),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `planned_end_dates` 키에 `_join_unique(row.get("planned_end_date") for row in rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_end_dates": _join_unique(row.get("planned_end_date") for row in rows),
                # LINE-BY-LINE: `build_same_machine_time_groups`에서 반환/저장할 dict의 `interpretation` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "interpretation": (
                    # LINE-BY-LINE: 문자열 값 `"same machine and exact same actual start/end repeated across multiple W/O; "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "same machine and exact same actual start/end repeated across multiple W/O; "
                    # LINE-BY-LINE: 문자열 값 `"check whether this is a shared batch/window timestamp rather than individual machine occupancy"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "check whether this is a shared batch/window timestamp rather than individual machine occupancy"
                ),
            }
        )
    # LINE-BY-LINE: 호출자에게 `output_rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return output_rows


# LINE-BY-LINE: `build_job_schedule_rows(simulation)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_job_schedule_rows(simulation) -> List[Dict[str, Any]]:
    """simulation schedule을 job-level row로 변환."""

    # LINE-BY-LINE: `rows` 변수에 `[]` 결과를 저장합니다. 의미: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `operation in simulation.state.schedule` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for operation in simulation.state.schedule:
        # LINE-BY-LINE: `job`에 `simulation.jobs[operation.job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job = simulation.jobs[operation.job_id]
        # LINE-BY-LINE: `machine`에 `simulation.machines[operation.machine_id]` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine = simulation.machines[operation.machine_id]
        # LINE-BY-LINE: `cut_bay`에 `operation.cut_bay or machine.bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
        cut_bay = operation.cut_bay or machine.bay_id
        # LINE-BY-LINE: `tact_time`에 `job.extra.get("tact_time_minutes", job.base_stage_minutes.get("cut", ""))` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_time = job.extra.get("tact_time_minutes", job.base_stage_minutes.get("cut", ""))
        # LINE-BY-LINE: `rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job.job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": job.job_id,
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.extra.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": job.extra.get("source_wk_ord_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `job.extra.get("source_project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                "project_no": job.extra.get("source_project_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `job.extra.get("source_block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                "block_no": job.extra.get("source_block_no", ""),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `block_set_id` 키에 `job.block_set_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "block_set_id": job.block_set_id or "",
                # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `operation.batch_id or ""` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
                "batch_id": operation.batch_id or "",
                # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `_join_unique(operation.batch_job_ids or (operation.job_id,))` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
                "batch_job_ids": _join_unique(operation.batch_job_ids or (operation.job_id,)),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `batch_wo_count` 키에 `len(operation.batch_job_ids or (operation.job_id,))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "batch_wo_count": len(operation.batch_job_ids or (operation.job_id,)),
                # LINE-BY-LINE: 딕셔너리 키 `batch_length_sum`에는 `"" if operation.batch_length_sum is None else round(float(operation.batch_length_sum), 6)` 값을 넣습니다. 의미: batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
                "batch_length_sum": "" if operation.batch_length_sum is None else round(float(operation.batch_length_sum), 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `scheduled_day` 키에 `_day_key(simulation, operation.start_time)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "scheduled_day": _day_key(simulation, operation.start_time),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `capacity_date` 키에 `_day_key(simulation, operation.start_time)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "capacity_date": _day_key(simulation, operation.start_time),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `capacity_basis` 키에 `"algorithm_scheduled_day"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "capacity_basis": "algorithm_scheduled_day",
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `start_min` 키에 `round(operation.start_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_min": round(operation.start_time, 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `finish_min` 키에 `round(operation.finish_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_min": round(operation.finish_time, 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `process_min` 키에 `round(operation.processing_minutes, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "process_min": round(operation.processing_minutes, 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `tact_time_minutes` 키에 `tact_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time_minutes": tact_time,
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `algorithm_machine_id` 키에 `operation.machine_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "algorithm_machine_id": operation.machine_id,
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `algorithm_cut_bay` 키에 `cut_bay or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "algorithm_cut_bay": cut_bay or "",
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `machine_type` 키에 `machine.machine_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_type": machine.machine_type,
                # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `job.source_machine_id or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                "source_machine_id": job.source_machine_id or "",
                # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `job.source_cut_bay or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                "source_cut_bay": job.source_cut_bay or "",
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `machine_match` 키에 `str(operation.machine_id) == str(job.source_machine_id)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_match": str(operation.machine_id) == str(job.source_machine_id),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `bay_match` 키에 `str(cut_bay) == str(job.source_cut_bay)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "bay_match": str(cut_bay) == str(job.source_cut_bay),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `actual_start_min` 키에 `"" if job.actual_start_minutes is None else round(float(job.actual_start_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_start_min": "" if job.actual_start_minutes is None else round(float(job.actual_start_minutes), 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `actual_finish_min` 키에 `"" if job.actual_finish_minutes is None else round(float(job.actual_finish_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_finish_min": "" if job.actual_finish_minutes is None else round(float(job.actual_finish_minutes), 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `planned_start_min` 키에 `"" if job.planned_start_minutes is None else round(float(job.planned_start_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_start_min": "" if job.planned_start_minutes is None else round(float(job.planned_start_minutes), 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `planned_finish_min` 키에 `"" if job.planned_finish_minutes is None else round(float(job.planned_finish_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_finish_min": "" if job.planned_finish_minutes is None else round(float(job.planned_finish_minutes), 6),
                # LINE-BY-LINE: 딕셔너리 키 `due_date_minutes`에는 `"" if job.due_date_minutes is None else round(float(job.due_date_minutes), 6)` 값을 넣습니다. 의미: 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
                "due_date_minutes": "" if job.due_date_minutes is None else round(float(job.due_date_minutes), 6),
                # LINE-BY-LINE: `build_job_schedule_rows`에서 반환/저장할 dict의 `due_slack_min` 키에 `"" if job.due_date_minutes is None else round(float(job.due_date_minutes) - operation.finish_time...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "due_slack_min": "" if job.due_date_minutes is None else round(float(job.due_date_minutes) - operation.finish_time, 6),
                # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `job.plate_length` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                "plate_length": job.plate_length,
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `job.thickness` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": job.thickness,
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `"" if job.actual_duration_minutes is None else job.actual_duration_minutes` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": "" if job.actual_duration_minutes is None else job.actual_duration_minutes,
                # LINE-BY-LINE: 딕셔너리 키 `planned_start_date`에는 `job.extra.get("planned_start_date", "")` 값을 넣습니다. 의미: 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
                "planned_start_date": job.extra.get("planned_start_date", ""),
                # LINE-BY-LINE: 딕셔너리 키 `planned_end_date`에는 `job.extra.get("planned_end_date", "")` 값을 넣습니다. 의미: 계획 종료일 원본입니다. 예: `GYEL_ACT_EDDT`, 사용: 계획일 기반 검증 후보.
                "planned_end_date": job.extra.get("planned_end_date", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `job.extra.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": job.extra.get("actual_start_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `job.extra.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": job.extra.get("actual_end_datetime", ""),
            }
        )
    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `build_actual_replay_rows(scenario: Dict[str, Any], config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_actual_replay_rows(scenario: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """실적 데이터의 장비/Bay/시작/종료시간을 그대로 playback row로 변환."""

    # LINE-BY-LINE: `rows` 변수에 `[]` 결과를 저장합니다. 의미: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `jobs`에 `sorted(` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    jobs = sorted(
        # LINE-BY-LINE: `sorted(...)` 호출에 `scenario.get("jobs", [])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario.get("jobs", []),
        # LINE-BY-LINE: `key`에 `lambda job: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key=lambda job: (
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `_to_float_or_none(job.get("actual_start_minutes")) or 0.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            _to_float_or_none(job.get("actual_start_minutes")) or 0.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `_to_float_or_none(job.get("actual_finish_minutes")) or 0.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            _to_float_or_none(job.get("actual_finish_minutes")) or 0.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `_to_float_or_none(job.get("wk_seq")) or _to_float_or_none(job.get("source_row_index")) or 0.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            _to_float_or_none(job.get("wk_seq")) or _to_float_or_none(job.get("source_row_index")) or 0.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(job.get("job_id", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(job.get("job_id", "")),
        ),
    )
    # LINE-BY-LINE: `minutes_per_day`에 `int(config.get("simulation", {}).get("minutes_per_day", 1440))` 결과를 저장합니다. 의미/사용: `minutes_per_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    minutes_per_day = int(config.get("simulation", {}).get("minutes_per_day", 1440))

    # LINE-BY-LINE: `sequence, job in enumerate(jobs, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for sequence, job in enumerate(jobs, start=1):
        # LINE-BY-LINE: `actual_start`에 `_to_float_or_none(job.get("actual_start_minutes"))` 결과를 저장합니다. 의미/사용: `actual_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start = _to_float_or_none(job.get("actual_start_minutes"))
        # LINE-BY-LINE: `actual_finish`에 `_to_float_or_none(job.get("actual_finish_minutes"))` 결과를 저장합니다. 의미/사용: `actual_finish` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_finish = _to_float_or_none(job.get("actual_finish_minutes"))
        # LINE-BY-LINE: `actual_duration`에 `_to_float_or_none(job.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float_or_none(job.get("actual_duration_minutes"))
        # LINE-BY-LINE: 조건 `actual_start is None or actual_finish is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_start is None or actual_finish is None:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: 조건 `actual_duration is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_duration is None:
            # LINE-BY-LINE: `actual_duration`에 `actual_finish - actual_start` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            actual_duration = actual_finish - actual_start

        # LINE-BY-LINE: `actual_start_raw`에 `job.get("actual_start_datetime", "")` 결과를 저장합니다. 의미/사용: `actual_start_raw` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start_raw = job.get("actual_start_datetime", "")
        # LINE-BY-LINE: `scheduled_day`에 `_format_compact_date(actual_start_raw)` 결과를 저장합니다. 의미/사용: `scheduled_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        scheduled_day = _format_compact_date(actual_start_raw)
        # LINE-BY-LINE: 조건 `not scheduled_day`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not scheduled_day:
            # LINE-BY-LINE: `day_index`에 `int(actual_start // max(minutes_per_day, 1))` 결과를 저장합니다. 의미/사용: `day_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            day_index = int(actual_start // max(minutes_per_day, 1))
            # LINE-BY-LINE: `scheduled_day`에 `f"actual_day_{day_index}"` 결과를 저장합니다. 의미/사용: `scheduled_day` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            scheduled_day = f"actual_day_{day_index}"

        # LINE-BY-LINE: `planned_start_date`에 `job.get("planned_start_date", "")` 결과를 저장합니다. 의미/사용: `planned_start_date`는 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
        planned_start_date = job.get("planned_start_date", "")
        # LINE-BY-LINE: `capacity_date`에 `_format_compact_date(planned_start_date) or scheduled_day` 결과를 저장합니다. 의미/사용: `capacity_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        capacity_date = _format_compact_date(planned_start_date) or scheduled_day
        # LINE-BY-LINE: `source_machine_id`에 `str(job.get("source_machine_id", ""))` 결과를 저장합니다. 의미/사용: `source_machine_id`는 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
        source_machine_id = str(job.get("source_machine_id", ""))
        # LINE-BY-LINE: `source_cut_bay`에 `str(job.get("source_cut_bay", ""))` 결과를 저장합니다. 의미/사용: `source_cut_bay`는 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
        source_cut_bay = str(job.get("source_cut_bay", ""))
        # LINE-BY-LINE: `tact_time`에 `job.get("tact_time_minutes")` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_time = job.get("tact_time_minutes")
        # LINE-BY-LINE: 조건 `tact_time in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if tact_time in (None, ""):
            # LINE-BY-LINE: `tact_time`에 `(job.get("base_stage_minutes", {}) or {}).get("cut", "")` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_time = (job.get("base_stage_minutes", {}) or {}).get("cut", "")

        # LINE-BY-LINE: `rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `actual_sequence` 키에 `sequence` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_sequence": sequence,
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": job.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": job.get("source_wk_ord_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `job.get("source_project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                "project_no": job.get("source_project_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `job.get("source_block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                "block_no": job.get("source_block_no", ""),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `block_set_id` 키에 `job.get("block_set_id") or job.get("block_set_key") or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "block_set_id": job.get("block_set_id") or job.get("block_set_key") or "",
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `scheduled_day` 키에 `scheduled_day` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "scheduled_day": scheduled_day,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `capacity_date` 키에 `capacity_date` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "capacity_date": capacity_date,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `capacity_basis` 키에 `"GYEL_ACT_STDT"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "capacity_basis": "GYEL_ACT_STDT",
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `start_min` 키에 `round(actual_start, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_min": round(actual_start, 6),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `finish_min` 키에 `round(actual_finish, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_min": round(actual_finish, 6),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `process_min` 키에 `round(actual_duration, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "process_min": round(actual_duration, 6),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `tact_time_minutes` 키에 `tact_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time_minutes": tact_time,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `algorithm_machine_id` 키에 `source_machine_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "algorithm_machine_id": source_machine_id,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `algorithm_cut_bay` 키에 `source_cut_bay` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "algorithm_cut_bay": source_cut_bay,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `machine_type` 키에 `"plasma" if source_machine_id.startswith("PLS") else ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_type": "plasma" if source_machine_id.startswith("PLS") else "",
                # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `source_machine_id` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                "source_machine_id": source_machine_id,
                # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `source_cut_bay` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                "source_cut_bay": source_cut_bay,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `machine_match` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_match": True,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `bay_match` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "bay_match": True,
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `actual_start_min` 키에 `round(actual_start, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_start_min": round(actual_start, 6),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `actual_finish_min` 키에 `round(actual_finish, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_finish_min": round(actual_finish, 6),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `planned_start_min` 키에 `job.get("planned_start_minutes", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_start_min": job.get("planned_start_minutes", ""),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `planned_finish_min` 키에 `job.get("planned_finish_minutes", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_finish_min": job.get("planned_finish_minutes", ""),
                # LINE-BY-LINE: 딕셔너리 키 `due_date_minutes`에는 `job.get("due_date_minutes", "")` 값을 넣습니다. 의미: 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
                "due_date_minutes": job.get("due_date_minutes", ""),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `due_slack_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "due_slack_min": (
                    # LINE-BY-LINE: 문자열 값 `""`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    ""
                    # LINE-BY-LINE: 조건 `job.get("due_date_minutes") in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if job.get("due_date_minutes") in (None, "")
                    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                    else round(float(job.get("due_date_minutes")) - actual_finish, 6)
                ),
                # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `job.get("plate_length", 0.0)` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                "plate_length": job.get("plate_length", 0.0),
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `job.get("thickness", 0.0)` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": job.get("thickness", 0.0),
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `round(actual_duration, 6)` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": round(actual_duration, 6),
                # LINE-BY-LINE: 딕셔너리 키 `planned_start_date`에는 `planned_start_date` 값을 넣습니다. 의미: 계획 착수일 원본입니다. 예: `GYEL_ACT_STDT`, 사용: 계획일 기반 검증 후보.
                "planned_start_date": planned_start_date,
                # LINE-BY-LINE: 딕셔너리 키 `planned_end_date`에는 `job.get("planned_end_date", "")` 값을 넣습니다. 의미: 계획 종료일 원본입니다. 예: `GYEL_ACT_EDDT`, 사용: 계획일 기반 검증 후보.
                "planned_end_date": job.get("planned_end_date", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `actual_start_raw` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": actual_start_raw,
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `job.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": job.get("actual_end_datetime", ""),
                # LINE-BY-LINE: `build_actual_replay_rows`에서 반환/저장할 dict의 `replay_mode` 키에 `"actual_historical"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "replay_mode": "actual_historical",
            }
        )

    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `build_actual_factory_identity_validation(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_actual_factory_identity_validation(schedule_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """실적 replay가 원본 실적 시간/장비/Bay를 그대로 재현했는지 검증한다."""

    # LINE-BY-LINE: `rows`에 `list(schedule_rows)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(schedule_rows)
    # LINE-BY-LINE: `errors` 변수에 `[]` 결과를 저장합니다. 의미: `errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    errors: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `start_abs_errors` 변수에 `[]` 결과를 저장합니다. 의미: `start_abs_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_abs_errors: List[float] = []
    # LINE-BY-LINE: `finish_abs_errors` 변수에 `[]` 결과를 저장합니다. 의미: `finish_abs_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    finish_abs_errors: List[float] = []
    # LINE-BY-LINE: `duration_abs_errors` 변수에 `[]` 결과를 저장합니다. 의미: `duration_abs_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_abs_errors: List[float] = []
    # LINE-BY-LINE: `tolerance`에 `1e-6` 결과를 저장합니다. 의미/사용: `tolerance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tolerance = 1e-6

    # LINE-BY-LINE: `add_error(row: Dict[str, Any], rule: str, message: str, expected: Any, actual: Any)` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
    def add_error(row: Dict[str, Any], rule: str, message: str, expected: Any, actual: Any) -> None:
        """identity validation error row를 누적한다."""

        # LINE-BY-LINE: `errors.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        errors.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `add_error`에서 반환/저장할 dict의 `rule` 키에 `rule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "rule": rule,
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("algorithm_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row.get("algorithm_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row.get("algorithm_cut_bay", "")` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": row.get("algorithm_cut_bay", ""),
                # LINE-BY-LINE: `add_error`에서 반환/저장할 dict의 `expected` 키에 `expected` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "expected": expected,
                # LINE-BY-LINE: `add_error`에서 반환/저장할 dict의 `actual` 키에 `actual` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual": actual,
                # LINE-BY-LINE: 딕셔너리 키 `message`에는 `message` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                "message": message,
            }
        )

    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: `process_min`에 `_to_float_or_none(row.get("process_min"))` 결과를 저장합니다. 의미/사용: `process_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_min = _to_float_or_none(row.get("process_min"))
        # LINE-BY-LINE: `actual_start_min`에 `_to_float_or_none(row.get("actual_start_min"))` 결과를 저장합니다. 의미/사용: `actual_start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start_min = _to_float_or_none(row.get("actual_start_min"))
        # LINE-BY-LINE: `actual_finish_min`에 `_to_float_or_none(row.get("actual_finish_min"))` 결과를 저장합니다. 의미/사용: `actual_finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_finish_min = _to_float_or_none(row.get("actual_finish_min"))
        # LINE-BY-LINE: `actual_duration`에 `_to_float_or_none(row.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float_or_none(row.get("actual_duration_minutes"))

        # LINE-BY-LINE: 조건 `start_min is None or actual_start_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_min is None or actual_start_min is None:
            # LINE-BY-LINE: `add_error(row, "actual_start_missing", "actual replay start time is missing", actual_start_min, s...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_error(row, "actual_start_missing", "actual replay start time is missing", actual_start_min, start_min)
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `start_error`에 `abs(actual_start_min - start_min)` 결과를 저장합니다. 의미/사용: `start_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            start_error = abs(actual_start_min - start_min)
            # LINE-BY-LINE: `start_abs_errors.append(start_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            start_abs_errors.append(start_error)
            # LINE-BY-LINE: 조건 `start_error > tolerance`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if start_error > tolerance:
                # LINE-BY-LINE: `add_error(row, "actual_start_mismatch", "actual start minute was not reproduced", actual_start_mi...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                add_error(row, "actual_start_mismatch", "actual start minute was not reproduced", actual_start_min, start_min)

        # LINE-BY-LINE: 조건 `finish_min is None or actual_finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if finish_min is None or actual_finish_min is None:
            # LINE-BY-LINE: `add_error(row, "actual_finish_missing", "actual replay finish time is missing", actual_finish_min...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_error(row, "actual_finish_missing", "actual replay finish time is missing", actual_finish_min, finish_min)
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `finish_error`에 `abs(actual_finish_min - finish_min)` 결과를 저장합니다. 의미/사용: `finish_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            finish_error = abs(actual_finish_min - finish_min)
            # LINE-BY-LINE: `finish_abs_errors.append(finish_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            finish_abs_errors.append(finish_error)
            # LINE-BY-LINE: 조건 `finish_error > tolerance`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if finish_error > tolerance:
                # LINE-BY-LINE: `add_error(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                add_error(
                    # LINE-BY-LINE: `add_error(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    row,
                    # LINE-BY-LINE: 문자열 값 `"actual_finish_mismatch"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "actual_finish_mismatch",
                    # LINE-BY-LINE: 문자열 값 `"actual finish minute was not reproduced"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "actual finish minute was not reproduced",
                    # LINE-BY-LINE: `add_error(...)` 호출에 `actual_finish_min` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    actual_finish_min,
                    # LINE-BY-LINE: `add_error(...)` 호출에 `finish_min` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    finish_min,
                )

        # LINE-BY-LINE: 조건 `process_min is None or actual_duration is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if process_min is None or actual_duration is None:
            # LINE-BY-LINE: `add_error(row, "actual_duration_missing", "actual replay duration is missing", actual_duration, p...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_error(row, "actual_duration_missing", "actual replay duration is missing", actual_duration, process_min)
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `duration_error`에 `abs(actual_duration - process_min)` 결과를 저장합니다. 의미/사용: `duration_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            duration_error = abs(actual_duration - process_min)
            # LINE-BY-LINE: `duration_abs_errors.append(duration_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            duration_abs_errors.append(duration_error)
            # LINE-BY-LINE: 조건 `duration_error > tolerance`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if duration_error > tolerance:
                # LINE-BY-LINE: `add_error(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                add_error(
                    # LINE-BY-LINE: `add_error(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    row,
                    # LINE-BY-LINE: 문자열 값 `"actual_duration_mismatch"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "actual_duration_mismatch",
                    # LINE-BY-LINE: 문자열 값 `"actual duration was not reproduced"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "actual duration was not reproduced",
                    # LINE-BY-LINE: `add_error(...)` 호출에 `actual_duration` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    actual_duration,
                    # LINE-BY-LINE: `add_error(...)` 호출에 `process_min` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    process_min,
                )

        # LINE-BY-LINE: 조건 `str(row.get("algorithm_machine_id", "")) != str(row.get("source_machine_id", ""))`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if str(row.get("algorithm_machine_id", "")) != str(row.get("source_machine_id", "")):
            # LINE-BY-LINE: `add_error(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_error(
                # LINE-BY-LINE: `add_error(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row,
                # LINE-BY-LINE: 문자열 값 `"actual_machine_mismatch"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "actual_machine_mismatch",
                # LINE-BY-LINE: 문자열 값 `"actual machine was not reproduced"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "actual machine was not reproduced",
                # LINE-BY-LINE: `add_error(...)` 호출에 `row.get("source_machine_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row.get("source_machine_id", ""),
                # LINE-BY-LINE: `add_error(...)` 호출에 `row.get("algorithm_machine_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row.get("algorithm_machine_id", ""),
            )

        # LINE-BY-LINE: 조건 `str(row.get("algorithm_cut_bay", "")) != str(row.get("source_cut_bay", ""))`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if str(row.get("algorithm_cut_bay", "")) != str(row.get("source_cut_bay", "")):
            # LINE-BY-LINE: `add_error(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_error(
                # LINE-BY-LINE: `add_error(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row,
                # LINE-BY-LINE: 문자열 값 `"actual_bay_mismatch"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "actual_bay_mismatch",
                # LINE-BY-LINE: 문자열 값 `"actual cut bay was not reproduced"`를 `add_error(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "actual cut bay was not reproduced",
                # LINE-BY-LINE: `add_error(...)` 호출에 `row.get("source_cut_bay", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row.get("source_cut_bay", ""),
                # LINE-BY-LINE: `add_error(...)` 호출에 `row.get("algorithm_cut_bay", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                row.get("algorithm_cut_bay", ""),
            )

    # LINE-BY-LINE: `identity_error_count`에 `len(errors)` 결과를 저장합니다. 의미/사용: `identity_error_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    identity_error_count = len(errors)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `identity_error_count == 0`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": identity_error_count == 0,
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `identity_error_count`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": identity_error_count,
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `factory_replay_passed` 키에 `identity_error_count == 0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "factory_replay_passed": identity_error_count == 0,
        # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `identity_error_count`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "identity_error_count": identity_error_count,
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `record_count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "record_count": len(rows),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `start_time_mae_minutes` 키에 `None if not start_abs_errors else statistics.mean(start_abs_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "start_time_mae_minutes": None if not start_abs_errors else statistics.mean(start_abs_errors),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `finish_time_mae_minutes` 키에 `None if not finish_abs_errors else statistics.mean(finish_abs_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "finish_time_mae_minutes": None if not finish_abs_errors else statistics.mean(finish_abs_errors),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `duration_mae_minutes` 키에 `None if not duration_abs_errors else statistics.mean(duration_abs_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mae_minutes": None if not duration_abs_errors else statistics.mean(duration_abs_errors),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `machine_match_rate` 키에 `None if not rows else sum(1 for row in rows if row.get("machine_match") in (True, "True", "true")...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_match_rate": None if not rows else sum(1 for row in rows if row.get("machine_match") in (True, "True", "true")) / len(rows),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `bay_match_rate` 키에 `None if not rows else sum(1 for row in rows if row.get("bay_match") in (True, "True", "true")) / ...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bay_match_rate": None if not rows else sum(1 for row in rows if row.get("bay_match") in (True, "True", "true")) / len(rows),
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `violations` 키에 `errors` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "violations": errors,
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `validation_type` 키에 `"actual_factory_replay_identity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_type": "actual_factory_replay_identity",
        # LINE-BY-LINE: `build_actual_factory_identity_validation`에서 반환/저장할 dict의 `validation_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_note": (
            # LINE-BY-LINE: 문자열 값 `"This validation checks whether actual RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "This validation checks whether actual RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM "
            # LINE-BY-LINE: 문자열 값 `"were reproduced exactly. Planning constraints are written separately as constraint audit."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "were reproduced exactly. Planning constraints are written separately as constraint audit."
        ),
    }


# LINE-BY-LINE: `build_machine_state_intervals(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_machine_state_intervals(schedule_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """실적 replay용 설비 busy interval 목록."""

    # LINE-BY-LINE: `intervals`에 `[]` 결과를 저장합니다. 의미/사용: `intervals` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    intervals = []
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: 조건 `start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_min is None or finish_min is None:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `intervals.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        intervals.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("algorithm_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row.get("algorithm_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row.get("algorithm_cut_bay", "")` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": row.get("algorithm_cut_bay", ""),
                # LINE-BY-LINE: `build_machine_state_intervals`에서 반환/저장할 dict의 `state` 키에 `"busy"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "state": "busy",
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: `build_machine_state_intervals`에서 반환/저장할 dict의 `start_min` 키에 `round(start_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_min": round(start_min, 6),
                # LINE-BY-LINE: `build_machine_state_intervals`에서 반환/저장할 dict의 `finish_min` 키에 `round(finish_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_min": round(finish_min, 6),
                # LINE-BY-LINE: `build_machine_state_intervals`에서 반환/저장할 dict의 `start_sec` 키에 `int(round(start_min * 60))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "start_sec": int(round(start_min * 60)),
                # LINE-BY-LINE: `build_machine_state_intervals`에서 반환/저장할 dict의 `finish_sec` 키에 `int(round(finish_min * 60))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "finish_sec": int(round(finish_min * 60)),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `row.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": row.get("actual_start_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `row.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": row.get("actual_end_datetime", ""),
            }
        )
    # LINE-BY-LINE: 호출자에게 `sorted(intervals, key=lambda item: (item["machine_id"], item["start_min"], item["finish_min"]))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(intervals, key=lambda item: (item["machine_id"], item["start_min"], item["finish_min"]))


# LINE-BY-LINE: `build_bay_state_intervals(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_bay_state_intervals(schedule_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """실적 replay용 Bay active/idle interval 목록."""

    # LINE-BY-LINE: `events_by_bay` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `events_by_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events_by_bay: Dict[str, List[tuple[float, int]]] = defaultdict(list)
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `bay_id`에 `str(row.get("algorithm_cut_bay", ""))` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(row.get("algorithm_cut_bay", ""))
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: 조건 `not bay_id or start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not bay_id or start_min is None or finish_min is None:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `events_by_bay[bay_id].append((start_min, 1))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events_by_bay[bay_id].append((start_min, 1))
        # LINE-BY-LINE: `events_by_bay[bay_id].append((finish_min, -1))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events_by_bay[bay_id].append((finish_min, -1))

    # LINE-BY-LINE: `intervals` 변수에 `[]` 결과를 저장합니다. 의미: `intervals` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    intervals: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `bay_id, events in sorted(events_by_bay.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bay_id, events in sorted(events_by_bay.items()):
        # LINE-BY-LINE: `events`에 `sorted(events, key=lambda item: (item[0], -item[1]))` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        events = sorted(events, key=lambda item: (item[0], -item[1]))
        # LINE-BY-LINE: 조건 `not events`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not events:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `current_time`에 `events[0][0]` 결과를 저장합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        current_time = events[0][0]
        # LINE-BY-LINE: `active_count`에 `0` 결과를 저장합니다. 의미/사용: `active_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_count = 0
        # LINE-BY-LINE: `index`에 `0` 결과를 저장합니다. 의미/사용: `index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        index = 0
        # LINE-BY-LINE: 조건 `index < len(events)`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
        while index < len(events):
            # LINE-BY-LINE: `event_time`에 `events[index][0]` 결과를 저장합니다. 의미/사용: `event_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            event_time = events[index][0]
            # LINE-BY-LINE: 조건 `event_time > current_time`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if event_time > current_time:
                # LINE-BY-LINE: `intervals.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                intervals.append(
                    # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                    {
                        # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `bay_id` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                        "bay_id": bay_id,
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `state` 키에 `"active" if active_count > 0 else "idle"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "state": "active" if active_count > 0 else "idle",
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `active_job_count` 키에 `active_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "active_job_count": active_count,
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `start_min` 키에 `round(current_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "start_min": round(current_time, 6),
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `finish_min` 키에 `round(event_time, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "finish_min": round(event_time, 6),
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `start_sec` 키에 `int(round(current_time * 60))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "start_sec": int(round(current_time * 60)),
                        # LINE-BY-LINE: `build_bay_state_intervals`에서 반환/저장할 dict의 `finish_sec` 키에 `int(round(event_time * 60))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                        "finish_sec": int(round(event_time * 60)),
                    }
                )
                # LINE-BY-LINE: `current_time`에 `event_time` 결과를 저장합니다. 의미/사용: `current_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                current_time = event_time

            # LINE-BY-LINE: 조건 `index < len(events) and events[index][0] == event_time`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
            while index < len(events) and events[index][0] == event_time:
                # LINE-BY-LINE: `active_count` 값을 `events[index][1]` 기준으로 누적/증가합니다. 의미/사용: `active_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                active_count += events[index][1]
                # LINE-BY-LINE: `index` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                index += 1
            # LINE-BY-LINE: `active_count`에 `max(0, active_count)` 결과를 저장합니다. 의미/사용: `active_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            active_count = max(0, active_count)

    # LINE-BY-LINE: 호출자에게 `intervals`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return intervals


# LINE-BY-LINE: `build_actual_job_lifecycle_rows(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_actual_job_lifecycle_rows(schedule_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """현재 데이터에서 관측 가능한 W/O lifecycle 상태를 명시한다.

    현재 원천 데이터에는 queue, transfer, Bay IN/OUT, LM/크레인, 중단 이벤트가 없다.
    따라서 해당 상태는 추정하지 않고 unknown으로 남긴다.
    """

    # LINE-BY-LINE: `lifecycle_rows` 변수에 `[]` 결과를 저장합니다. 의미: `lifecycle_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    lifecycle_rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: `lifecycle_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        lifecycle_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `row.get("project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                "project_no": row.get("project_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `row.get("block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                "block_no": row.get("block_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("algorithm_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row.get("algorithm_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row.get("algorithm_cut_bay", "")` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": row.get("algorithm_cut_bay", ""),
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `known_lifecycle` 키에 `"processing_start->processing_finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "known_lifecycle": "processing_start->processing_finish",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `unknown_lifecycle` 키에 `"release->queue_enter->queue_exit->transfer_start->transfer_end->bay_in->bay_out"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "unknown_lifecycle": "release->queue_enter->queue_exit->transfer_start->transfer_end->bay_in->bay_out",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `queue_status` 키에 `"unknown_missing_source_event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "queue_status": "unknown_missing_source_event",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `transfer_status` 키에 `"unknown_missing_source_event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "transfer_status": "unknown_missing_source_event",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `bay_in_out_status` 키에 `"unknown_missing_source_event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "bay_in_out_status": "unknown_missing_source_event",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `lm_crane_status` 키에 `"unknown_missing_source_event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "lm_crane_status": "unknown_missing_source_event",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `stoppage_status` 키에 `"unknown_missing_source_event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "stoppage_status": "unknown_missing_source_event",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `nesting_batch_status` 키에 `"derived_by_machine_time_overlap_for_capacity_audit"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "nesting_batch_status": "derived_by_machine_time_overlap_for_capacity_audit",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `processing_start_min` 키에 `"" if start_min is None else round(start_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "processing_start_min": "" if start_min is None else round(start_min, 6),
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `processing_finish_min` 키에 `"" if finish_min is None else round(finish_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "processing_finish_min": "" if finish_min is None else round(finish_min, 6),
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `processing_duration_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "processing_duration_min": (
                    # LINE-BY-LINE: 문자열 값 `""`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    ""
                    # LINE-BY-LINE: 조건 `start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                    if start_min is None or finish_min is None
                    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
                    else round(finish_min - start_min, 6)
                ),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `row.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": row.get("actual_start_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `row.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": row.get("actual_end_datetime", ""),
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `factory_replay_scope` 키에 `"observed_processing_interval_only"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "factory_replay_scope": "observed_processing_interval_only",
                # LINE-BY-LINE: `build_actual_job_lifecycle_rows`에서 반환/저장할 dict의 `validation_scope` 키에 `"machine_bay_processing_start_finish_identity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "validation_scope": "machine_bay_processing_start_finish_identity",
            }
        )
    # LINE-BY-LINE: 호출자에게 `lifecycle_rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return lifecycle_rows


# LINE-BY-LINE: `build_missing_event_requirements()` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_missing_event_requirements() -> List[Dict[str, Any]]:
    """진짜 resource-flow DES로 확장할 때 필요한 이벤트의 현재 관측 상태."""

    # LINE-BY-LINE: 호출자에게 list 반환을 시작합니다. 사용: 여러 row 또는 후보 값을 순서 있는 목록으로 전달합니다.
    return [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"machine_assignment"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "machine_assignment",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"observed"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "observed",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"use RT_EQP_NM as actual machine_id"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "use RT_EQP_NM as actual machine_id",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_EQP_NM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_EQP_NM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"identity validation target"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "identity validation target",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"cut_bay_assignment"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "cut_bay_assignment",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"observed"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "observed",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"use CUT_BAY as actual cutting bay"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "use CUT_BAY as actual cutting bay",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"CUT_BAY"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "CUT_BAY",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"identity validation target"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "identity validation target",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"processing_start"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "processing_start",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"observed"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "observed",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"use RT_CUT_ST_DTM as processing start event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "use RT_CUT_ST_DTM as processing start event",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_CUT_ST_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_CUT_ST_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"identity validation target"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "identity validation target",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"processing_finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "processing_finish",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"observed"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "observed",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"use RT_CUT_ED_DTM as processing finish event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "use RT_CUT_ED_DTM as processing finish event",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"identity validation target"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "identity validation target",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"actual_processing_duration"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "actual_processing_duration",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"derived_from_observed_events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "derived_from_observed_events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"RT_CUT_ED_DTM - RT_CUT_ST_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "RT_CUT_ED_DTM - RT_CUT_ST_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_CUT_ST_DTM|RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_CUT_ST_DTM|RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"duration identity validation target"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "duration identity validation target",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"machine_busy_interval"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "machine_busy_interval",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"derived_from_observed_events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "derived_from_observed_events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"one busy interval per W/O processing start/finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "one busy interval per W/O processing start/finish",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"observable factory state"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "observable factory state",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"machine_idle_gap"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "machine_idle_gap",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"derived_from_observed_events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "derived_from_observed_events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"implicit gap between busy intervals; not a separate source event"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "implicit gap between busy intervals; not a separate source event",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"explicit idle reason code if available"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "explicit idle reason code if available",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"derived state only"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "derived state only",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"bay_active_load_interval"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "bay_active_load_interval",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"derived_from_observed_events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "derived_from_observed_events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"active W/O count by Bay from processing start/finish events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "active W/O count by Bay from processing start/finish events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"CUT_BAY|RT_CUT_ST_DTM|RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "CUT_BAY|RT_CUT_ST_DTM|RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"observable aggregate factory state"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "observable aggregate factory state",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"work_order_release"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "work_order_release",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"WO_RELEASE_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "WO_RELEASE_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for queue/waiting DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for queue/waiting DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"queue_enter"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "queue_enter",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"QUEUE_IN_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "QUEUE_IN_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for machine queue DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for machine queue DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"queue_exit"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "queue_exit",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"QUEUE_OUT_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "QUEUE_OUT_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for machine queue DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for machine queue DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"transfer_start"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "transfer_start",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"TRANSFER_ST_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "TRANSFER_ST_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for movement-time DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for movement-time DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"transfer_finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "transfer_finish",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"TRANSFER_ED_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "TRANSFER_ED_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for movement-time DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for movement-time DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"bay_in"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "bay_in",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"BAY_IN_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "BAY_IN_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for Bay IN/OUT flow DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for Bay IN/OUT flow DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"bay_out"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "bay_out",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"BAY_OUT_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "BAY_OUT_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for Bay IN/OUT flow DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for Bay IN/OUT flow DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"lm_crane_start"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "lm_crane_start",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"LM_OR_CRANE_ID|LM_CRANE_ST_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "LM_OR_CRANE_ID|LM_CRANE_ST_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for crane/LM resource DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for crane/LM resource DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"lm_crane_finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "lm_crane_finish",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"LM_OR_CRANE_ID|LM_CRANE_ED_DTM or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "LM_OR_CRANE_ID|LM_CRANE_ED_DTM or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed for crane/LM resource DES"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed for crane/LM resource DES",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"stoppage_start"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "stoppage_start",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"STOP_ST_DTM|STOP_REASON or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "STOP_ST_DTM|STOP_REASON or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed to separate cutting time from interruption time"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed to separate cutting time from interruption time",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"stoppage_finish"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "stoppage_finish",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"missing_in_current_data"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "missing_in_current_data",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"unknown; never inferred"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "unknown; never inferred",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"STOP_ED_DTM|STOP_REASON or equivalent"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "STOP_ED_DTM|STOP_REASON or equivalent",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"needed to separate cutting time from interruption time"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "needed to separate cutting time from interruption time",
        },
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `event_or_state` 키에 `"nesting_or_batch_group"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_or_state": "nesting_or_batch_group",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `status` 키에 `"derived_from_observed_events"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "status": "derived_from_observed_events",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `current_handling` 키에 `"derive concurrent active W/O set by machine from RT_CUT_ST_DTM/RT_CUT_ED_DTM"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "current_handling": "derive concurrent active W/O set by machine from RT_CUT_ST_DTM/RT_CUT_ED_DTM",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `source_columns` 키에 `"RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM|LTH"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "source_columns": "RT_EQP_NM|RT_CUT_ST_DTM|RT_CUT_ED_DTM|LTH",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `required_columns_for_full_des` 키에 `"optional explicit NEST_ID or BATCH_ID if 현업 wants named batch tracking"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "required_columns_for_full_des": "optional explicit NEST_ID or BATCH_ID if 현업 wants named batch tracking",
            # LINE-BY-LINE: `build_missing_event_requirements`에서 반환/저장할 dict의 `validation_impact` 키에 `"3-W/O/55000 audit uses concurrent active interval, not daily total"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "validation_impact": "3-W/O/55000 audit uses concurrent active interval, not daily total",
        },
    ]


# LINE-BY-LINE: `build_time_validation(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_time_validation(schedule_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """스케줄 시간과 실적/계획 시간의 1차 비교 결과를 만든다.

    주의:
    - `process_min`은 현재 scenario의 예상 처리시간이다. NP 100건에서는 TACT_TIME 기반이다.
    - `actual_duration_minutes`는 RT_CUT_ST_DTM~RT_CUT_ED_DTM 차이다.
    - `actual_start_min`은 실적 시작시각 기준의 상대분이고, `start_min`은 계획 시작일 기준의
      스케줄 상대분이다. 기준일이 다르므로 시작/완료시각은 절대 차이보다 상대 순서 상관을 본다.
    """

    # LINE-BY-LINE: `rows` 변수에 `[]` 결과를 저장합니다. 의미: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `duration_errors` 변수에 `[]` 결과를 저장합니다. 의미: `duration_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_errors: List[float] = []
    # LINE-BY-LINE: `duration_abs_errors` 변수에 `[]` 결과를 저장합니다. 의미: `duration_abs_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_abs_errors: List[float] = []
    # LINE-BY-LINE: `duration_pct_errors` 변수에 `[]` 결과를 저장합니다. 의미: `duration_pct_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_pct_errors: List[float] = []
    # LINE-BY-LINE: `tact_errors` 변수에 `[]` 결과를 저장합니다. 의미: `tact_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_errors: List[float] = []
    # LINE-BY-LINE: `tact_abs_errors` 변수에 `[]` 결과를 저장합니다. 의미: `tact_abs_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_abs_errors: List[float] = []
    # LINE-BY-LINE: `tact_pct_errors` 변수에 `[]` 결과를 저장합니다. 의미: `tact_pct_errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_pct_errors: List[float] = []
    # LINE-BY-LINE: `start_pairs` 변수에 `[]` 결과를 저장합니다. 의미: `start_pairs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_pairs: List[tuple[float, float]] = []
    # LINE-BY-LINE: `finish_pairs` 변수에 `[]` 결과를 저장합니다. 의미: `finish_pairs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    finish_pairs: List[tuple[float, float]] = []
    # LINE-BY-LINE: `planned_start_late_count`에 `0` 결과를 저장합니다. 의미/사용: `planned_start_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    planned_start_late_count = 0
    # LINE-BY-LINE: `planned_finish_late_count`에 `0` 결과를 저장합니다. 의미/사용: `planned_finish_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    planned_finish_late_count = 0
    # LINE-BY-LINE: `due_late_count`에 `0` 결과를 저장합니다. 의미/사용: `due_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    due_late_count = 0
    # LINE-BY-LINE: `process_over_actual_count`에 `0` 결과를 저장합니다. 의미/사용: `process_over_actual_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    process_over_actual_count = 0
    # LINE-BY-LINE: `process_under_actual_count`에 `0` 결과를 저장합니다. 의미/사용: `process_under_actual_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    process_under_actual_count = 0
    # LINE-BY-LINE: `within_tolerance_count`에 `0` 결과를 저장합니다. 의미/사용: `within_tolerance_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    within_tolerance_count = 0

    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `process_min`에 `_to_float_or_none(row.get("process_min"))` 결과를 저장합니다. 의미/사용: `process_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_min = _to_float_or_none(row.get("process_min"))
        # LINE-BY-LINE: `tact_time`에 `_to_float_or_none(row.get("tact_time_minutes"))` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_time = _to_float_or_none(row.get("tact_time_minutes"))
        # LINE-BY-LINE: `actual_duration`에 `_to_float_or_none(row.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float_or_none(row.get("actual_duration_minutes"))
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: `actual_start_min`에 `_to_float_or_none(row.get("actual_start_min"))` 결과를 저장합니다. 의미/사용: `actual_start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start_min = _to_float_or_none(row.get("actual_start_min"))
        # LINE-BY-LINE: `actual_finish_min`에 `_to_float_or_none(row.get("actual_finish_min"))` 결과를 저장합니다. 의미/사용: `actual_finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_finish_min = _to_float_or_none(row.get("actual_finish_min"))
        # LINE-BY-LINE: `planned_start_min`에 `_to_float_or_none(row.get("planned_start_min"))` 결과를 저장합니다. 의미/사용: `planned_start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        planned_start_min = _to_float_or_none(row.get("planned_start_min"))
        # LINE-BY-LINE: `planned_finish_min`에 `_to_float_or_none(row.get("planned_finish_min"))` 결과를 저장합니다. 의미/사용: `planned_finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        planned_finish_min = _to_float_or_none(row.get("planned_finish_min"))
        # LINE-BY-LINE: `due_slack_min`에 `_to_float_or_none(row.get("due_slack_min"))` 결과를 저장합니다. 의미/사용: `due_slack_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        due_slack_min = _to_float_or_none(row.get("due_slack_min"))

        # LINE-BY-LINE: `duration_error`에 `None` 결과를 저장합니다. 의미/사용: `duration_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        duration_error = None
        # LINE-BY-LINE: `duration_abs_error`에 `None` 결과를 저장합니다. 의미/사용: `duration_abs_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        duration_abs_error = None
        # LINE-BY-LINE: `duration_pct_error`에 `None` 결과를 저장합니다. 의미/사용: `duration_pct_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        duration_pct_error = None
        # LINE-BY-LINE: `tact_error`에 `None` 결과를 저장합니다. 의미/사용: `tact_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_error = None
        # LINE-BY-LINE: `tact_abs_error`에 `None` 결과를 저장합니다. 의미/사용: `tact_abs_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_abs_error = None
        # LINE-BY-LINE: `tact_pct_error`에 `None` 결과를 저장합니다. 의미/사용: `tact_pct_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_pct_error = None
        # LINE-BY-LINE: `duration_status`에 `"missing_actual_or_process"` 결과를 저장합니다. 의미/사용: `duration_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        duration_status = "missing_actual_or_process"
        # LINE-BY-LINE: 조건 `process_min is not None and actual_duration is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if process_min is not None and actual_duration is not None:
            # LINE-BY-LINE: `duration_error`에 `actual_duration - process_min` 결과를 저장합니다. 의미/사용: `duration_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            duration_error = actual_duration - process_min
            # LINE-BY-LINE: `duration_abs_error`에 `abs(duration_error)` 결과를 저장합니다. 의미/사용: `duration_abs_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            duration_abs_error = abs(duration_error)
            # LINE-BY-LINE: `duration_pct_error`에 `duration_abs_error / actual_duration if actual_duration > 0 else None` 결과를 저장합니다. 의미/사용: `duration_pct_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            duration_pct_error = duration_abs_error / actual_duration if actual_duration > 0 else None
            # LINE-BY-LINE: `duration_errors.append(duration_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            duration_errors.append(duration_error)
            # LINE-BY-LINE: `duration_abs_errors.append(duration_abs_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            duration_abs_errors.append(duration_abs_error)
            # LINE-BY-LINE: 조건 `duration_pct_error is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if duration_pct_error is not None:
                # LINE-BY-LINE: `duration_pct_errors.append(duration_pct_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                duration_pct_errors.append(duration_pct_error)
            # LINE-BY-LINE: 조건 `process_min > actual_duration`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if process_min > actual_duration:
                # LINE-BY-LINE: `process_over_actual_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `process_over_actual_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                process_over_actual_count += 1
            # LINE-BY-LINE: 앞선 조건이 거짓일 때 `process_min < actual_duration`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
            elif process_min < actual_duration:
                # LINE-BY-LINE: `process_under_actual_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `process_under_actual_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                process_under_actual_count += 1

            # LINE-BY-LINE: `tolerance`에 `max(10.0, actual_duration * 0.2)` 결과를 저장합니다. 의미/사용: `tolerance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tolerance = max(10.0, actual_duration * 0.2)
            # LINE-BY-LINE: 조건 `duration_abs_error <= tolerance`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if duration_abs_error <= tolerance:
                # LINE-BY-LINE: `within_tolerance_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `within_tolerance_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                within_tolerance_count += 1
                # LINE-BY-LINE: `duration_status`에 `"within_20pct_or_10min"` 결과를 저장합니다. 의미/사용: `duration_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                duration_status = "within_20pct_or_10min"
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else:
                # LINE-BY-LINE: `duration_status`에 `"outside_20pct_or_10min"` 결과를 저장합니다. 의미/사용: `duration_status` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                duration_status = "outside_20pct_or_10min"

        # LINE-BY-LINE: 조건 `tact_time is not None and actual_duration is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if tact_time is not None and actual_duration is not None:
            # LINE-BY-LINE: `tact_error`에 `actual_duration - tact_time` 결과를 저장합니다. 의미/사용: `tact_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_error = actual_duration - tact_time
            # LINE-BY-LINE: `tact_abs_error`에 `abs(tact_error)` 결과를 저장합니다. 의미/사용: `tact_abs_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_abs_error = abs(tact_error)
            # LINE-BY-LINE: `tact_pct_error`에 `tact_abs_error / actual_duration if actual_duration > 0 else None` 결과를 저장합니다. 의미/사용: `tact_pct_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_pct_error = tact_abs_error / actual_duration if actual_duration > 0 else None
            # LINE-BY-LINE: `tact_errors.append(tact_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            tact_errors.append(tact_error)
            # LINE-BY-LINE: `tact_abs_errors.append(tact_abs_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            tact_abs_errors.append(tact_abs_error)
            # LINE-BY-LINE: 조건 `tact_pct_error is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if tact_pct_error is not None:
                # LINE-BY-LINE: `tact_pct_errors.append(tact_pct_error)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                tact_pct_errors.append(tact_pct_error)

        # LINE-BY-LINE: 조건 `start_min is not None and actual_start_min is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_min is not None and actual_start_min is not None:
            # LINE-BY-LINE: `start_pairs.append((start_min, actual_start_min))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            start_pairs.append((start_min, actual_start_min))
        # LINE-BY-LINE: 조건 `finish_min is not None and actual_finish_min is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if finish_min is not None and actual_finish_min is not None:
            # LINE-BY-LINE: `finish_pairs.append((finish_min, actual_finish_min))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            finish_pairs.append((finish_min, actual_finish_min))
        # LINE-BY-LINE: 조건 `start_min is not None and planned_start_min is not None and start_min > planned_start_min`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if start_min is not None and planned_start_min is not None and start_min > planned_start_min:
            # LINE-BY-LINE: `planned_start_late_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `planned_start_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            planned_start_late_count += 1
        # LINE-BY-LINE: 조건 `finish_min is not None and planned_finish_min is not None and finish_min > planned_finish_min`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if finish_min is not None and planned_finish_min is not None and finish_min > planned_finish_min:
            # LINE-BY-LINE: `planned_finish_late_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `planned_finish_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            planned_finish_late_count += 1
        # LINE-BY-LINE: 조건 `due_slack_min is not None and due_slack_min < 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if due_slack_min is not None and due_slack_min < 0:
            # LINE-BY-LINE: `due_late_count` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `due_late_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            due_late_count += 1

        # LINE-BY-LINE: `output_row`에 `dict(row)` 결과를 저장합니다. 의미/사용: `output_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        output_row = dict(row)
        # LINE-BY-LINE: `output_row.update(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        output_row.update(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_error_actual_minus_process` 키에 `"" if duration_error is None else round(duration_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "duration_error_actual_minus_process": "" if duration_error is None else round(duration_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_abs_error` 키에 `"" if duration_abs_error is None else round(duration_abs_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "duration_abs_error": "" if duration_abs_error is None else round(duration_abs_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_abs_pct_error` 키에 `"" if duration_pct_error is None else round(duration_pct_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "duration_abs_pct_error": "" if duration_pct_error is None else round(duration_pct_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_error_actual_minus_tact` 키에 `"" if tact_error is None else round(tact_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_error_actual_minus_tact": "" if tact_error is None else round(tact_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_abs_error` 키에 `"" if tact_abs_error is None else round(tact_abs_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_abs_error": "" if tact_abs_error is None else round(tact_abs_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_abs_pct_error` 키에 `"" if tact_pct_error is None else round(tact_pct_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_abs_pct_error": "" if tact_pct_error is None else round(tact_pct_error, 6),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_status` 키에 `duration_status` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "duration_status": duration_status,
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `planned_start_delta_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_start_delta_min": (
                    # LINE-BY-LINE: `"" if start_min is None or planned_start_min is None else round(start_min - planned_start_min, 6)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    "" if start_min is None or planned_start_min is None else round(start_min - planned_start_min, 6)
                ),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `planned_finish_delta_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "planned_finish_delta_min": (
                    # LINE-BY-LINE: `"" if finish_min is None or planned_finish_min is None else round(finish_min - planned_finish_min...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    "" if finish_min is None or planned_finish_min is None else round(finish_min - planned_finish_min, 6)
                ),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `actual_start_relative_delta_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_start_relative_delta_min": (
                    # LINE-BY-LINE: `"" if start_min is None or actual_start_min is None else round(actual_start_min - start_min, 6)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    "" if start_min is None or actual_start_min is None else round(actual_start_min - start_min, 6)
                ),
                # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `actual_finish_relative_delta_min` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_finish_relative_delta_min": (
                    # LINE-BY-LINE: `"" if finish_min is None or actual_finish_min is None else round(actual_finish_min - finish_min, 6)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    "" if finish_min is None or actual_finish_min is None else round(actual_finish_min - finish_min, 6)
                ),
            }
        )
        # LINE-BY-LINE: `rows.append(output_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(output_row)

    # LINE-BY-LINE: `duration_count`에 `len(duration_errors)` 결과를 저장합니다. 의미/사용: `duration_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    duration_count = len(duration_errors)
    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `record_count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "record_count": len(rows),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_compare_count` 키에 `duration_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_compare_count": duration_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_mae_minutes` 키에 `None if not duration_abs_errors else statistics.mean(duration_abs_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mae_minutes": None if not duration_abs_errors else statistics.mean(duration_abs_errors),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_bias_actual_minus_process_minutes` 키에 `None if not duration_errors else statistics.mean(duration_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_bias_actual_minus_process_minutes": None if not duration_errors else statistics.mean(duration_errors),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_rmse_minutes` 키에 `None if not duration_errors else math.sqrt(statistics.mean(error**2 for error in duration_errors))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_rmse_minutes": None if not duration_errors else math.sqrt(statistics.mean(error**2 for error in duration_errors)),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_mape_percent` 키에 `None if not duration_pct_errors else statistics.mean(duration_pct_errors) * 100.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mape_percent": None if not duration_pct_errors else statistics.mean(duration_pct_errors) * 100.0,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_compare_count` 키에 `len(tact_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_compare_count": len(tact_errors),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_mae_minutes` 키에 `None if not tact_abs_errors else statistics.mean(tact_abs_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mae_minutes": None if not tact_abs_errors else statistics.mean(tact_abs_errors),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_bias_actual_minus_tact_minutes` 키에 `None if not tact_errors else statistics.mean(tact_errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_bias_actual_minus_tact_minutes": None if not tact_errors else statistics.mean(tact_errors),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_rmse_minutes` 키에 `None if not tact_errors else math.sqrt(statistics.mean(error**2 for error in tact_errors))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_rmse_minutes": None if not tact_errors else math.sqrt(statistics.mean(error**2 for error in tact_errors)),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `tact_mape_percent` 키에 `None if not tact_pct_errors else statistics.mean(tact_pct_errors) * 100.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mape_percent": None if not tact_pct_errors else statistics.mean(tact_pct_errors) * 100.0,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_within_20pct_or_10min_count` 키에 `within_tolerance_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_within_20pct_or_10min_count": within_tolerance_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `duration_within_20pct_or_10min_rate` 키에 `None if duration_count == 0 else within_tolerance_count / duration_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_within_20pct_or_10min_rate": None if duration_count == 0 else within_tolerance_count / duration_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `process_over_actual_count` 키에 `process_over_actual_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "process_over_actual_count": process_over_actual_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `process_under_actual_count` 키에 `process_under_actual_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "process_under_actual_count": process_under_actual_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `relative_start_correlation` 키에 `_pearson_corr(start_pairs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "relative_start_correlation": _pearson_corr(start_pairs),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `relative_finish_correlation` 키에 `_pearson_corr(finish_pairs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "relative_finish_correlation": _pearson_corr(finish_pairs),
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `planned_start_late_count` 키에 `planned_start_late_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "planned_start_late_count": planned_start_late_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `planned_finish_late_count` 키에 `planned_finish_late_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "planned_finish_late_count": planned_finish_late_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `due_late_count` 키에 `due_late_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "due_late_count": due_late_count,
        # LINE-BY-LINE: `build_time_validation`에서 반환/저장할 dict의 `time_base_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "time_base_note": (
            # LINE-BY-LINE: 문자열 값 `"actual replay uses RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM directly. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual replay uses RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM directly. "
            # LINE-BY-LINE: 문자열 값 `"process_min vs actual_duration_minutes checks replay or algorithm duration. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "process_min vs actual_duration_minutes checks replay or algorithm duration. "
            # LINE-BY-LINE: 문자열 값 `"tact_time_minutes vs actual_duration_minutes checks the provided TACT_TIME."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "tact_time_minutes vs actual_duration_minutes checks the provided TACT_TIME."
        ),
    }
    # LINE-BY-LINE: 호출자에게 `{"rows": rows, "summary": summary}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return {"rows": rows, "summary": summary}


# LINE-BY-LINE: `build_action_trace_rows(trace_records: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
def build_action_trace_rows(trace_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """decision trace records를 CSV 저장 가능한 list로 반환한다."""

    # LINE-BY-LINE: 호출자에게 `list(trace_records)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return list(trace_records)


# LINE-BY-LINE: `build_machine_day_summary(rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_machine_day_summary(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """설비-일자별 작업 수/길이/처리시간 summary를 만든다."""

    # LINE-BY-LINE: `summary_rows`에 `[]` 결과를 저장합니다. 의미/사용: `summary_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    summary_rows = []
    # LINE-BY-LINE: `bucket in _machine_day_buckets(rows)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bucket in _machine_day_buckets(rows):
        # LINE-BY-LINE: `summary`에 `dict(bucket)` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
        summary = dict(bucket)
        # LINE-BY-LINE: `summary.pop("rows", None)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        summary.pop("rows", None)
        # LINE-BY-LINE: `summary_rows.append(summary)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        summary_rows.append(summary)
    # LINE-BY-LINE: 호출자에게 `summary_rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return summary_rows


# LINE-BY-LINE: `build_bay_day_summary(rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_bay_day_summary(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bay-일자별 작업 수/길이/처리시간 summary를 만든다."""

    # LINE-BY-LINE: `buckets` 변수에 `{}` 결과를 저장합니다. 의미: `buckets` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    buckets: Dict[tuple, Dict[str, Any]] = {}
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `bucket_date`에 `row.get("capacity_date") or row["scheduled_day"]` 결과를 저장합니다. 의미/사용: `bucket_date` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket_date = row.get("capacity_date") or row["scheduled_day"]
        # LINE-BY-LINE: `key`에 `(bucket_date, row["algorithm_cut_bay"])` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key = (bucket_date, row["algorithm_cut_bay"])
        # LINE-BY-LINE: `bucket`에 `buckets.setdefault(` 결과를 저장합니다. 의미/사용: `bucket` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket = buckets.setdefault(
            # LINE-BY-LINE: `setdefault(...)` 호출에 `key` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            key,
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `build_bay_day_summary`에서 반환/저장할 dict의 `date` 키에 `bucket_date` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "date": bucket_date,
                # LINE-BY-LINE: `build_bay_day_summary`에서 반환/저장할 dict의 `date_basis` 키에 `row.get("capacity_basis", "scheduled_day")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "date_basis": row.get("capacity_basis", "scheduled_day"),
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row["algorithm_cut_bay"]` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": row["algorithm_cut_bay"],
                # LINE-BY-LINE: `build_bay_day_summary`에서 반환/저장할 dict의 `num_jobs` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "num_jobs": 0,
                # LINE-BY-LINE: `build_bay_day_summary`에서 반환/저장할 dict의 `total_length` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_length": 0.0,
                # LINE-BY-LINE: `build_bay_day_summary`에서 반환/저장할 dict의 `total_process_time` 키에 `0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_process_time": 0.0,
            },
        )
        # LINE-BY-LINE: `bucket["num_jobs"]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `bucket["num_jobs"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["num_jobs"] += 1
        # LINE-BY-LINE: `bucket["total_length"]` 값을 `float(row["plate_length"])` 기준으로 누적/증가합니다. 의미/사용: `bucket["total_length"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["total_length"] += float(row["plate_length"])
        # LINE-BY-LINE: `bucket["total_process_time"]` 값을 `float(row["process_min"])` 기준으로 누적/증가합니다. 의미/사용: `bucket["total_process_time"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bucket["total_process_time"] += float(row["process_min"])
    # LINE-BY-LINE: 호출자에게 `list(buckets.values())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return list(buckets.values())


# LINE-BY-LINE: `validate_schedule_rows` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def validate_schedule_rows(
    # LINE-BY-LINE: `schedule_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `config`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config: Dict[str, Any],
    # LINE-BY-LINE: `machine_to_bay` 변수에 `None` 결과를 저장합니다. 의미: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_to_bay: Optional[Dict[str, str]] = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """schedule row 목록에 대해 1차 hard validation을 수행."""

    hard_enabled = config.get("constraints", {}).get("hard_enabled", {})

    def hard_rule_enabled(rule_id: str) -> bool:
        if rule_id not in hard_enabled:
            print(
                "[ERROR][playback_builder.validate_schedule_rows] "
                f"cause=missing hard_enabled rule key={rule_id}"
            )
            raise RuntimeError(f"missing hard_enabled rule: {rule_id}")
        return bool(hard_enabled[rule_id])

    # LINE-BY-LINE: `violations` 변수에 `[]` 결과를 저장합니다. 의미: `violations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violations: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `rows`에 `list(schedule_rows)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(schedule_rows)
    # LINE-BY-LINE: `affected_job_ids` 변수에 `set()` 결과를 저장합니다. 의미: `affected_job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    affected_job_ids: set[str] = set()
    # LINE-BY-LINE: `affected_work_orders` 변수에 `set()` 결과를 저장합니다. 의미: `affected_work_orders` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    affected_work_orders: set[str] = set()
    # LINE-BY-LINE: `violation_counts_by_rule` 변수에 `defaultdict(int)` 결과를 저장합니다. 의미: `violation_counts_by_rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violation_counts_by_rule: Dict[str, int] = defaultdict(int)
    # LINE-BY-LINE: `violating_machine_day_buckets` 변수에 `set()` 결과를 저장합니다. 의미: `violating_machine_day_buckets` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violating_machine_day_buckets: set[tuple[str, str, str]] = set()
    # LINE-BY-LINE: `hard_violation_excess_wo_count`에 `0` 결과를 저장합니다. 의미/사용: `hard_violation_excess_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hard_violation_excess_wo_count = 0
    # LINE-BY-LINE: `hard_violation_length_excess_total`에 `0.0` 결과를 저장합니다. 의미/사용: `hard_violation_length_excess_total` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hard_violation_length_excess_total = 0.0

    # LINE-BY-LINE: `add_violation(violation: Dict[str, Any], affected_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
    def add_violation(violation: Dict[str, Any], affected_rows: Iterable[Dict[str, Any]]) -> None:
        """hard validation 위반 1건과 영향받은 W/O 목록을 누적한다."""

        # LINE-BY-LINE: `affected_rows`에 `list(affected_rows)` 결과를 저장합니다. 의미/사용: `affected_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        affected_rows = list(affected_rows)
        # LINE-BY-LINE: `job_ids`에 `[str(row.get("job_id", "")).strip() for row in affected_rows if row.get("job_id")]` 결과를 저장합니다. 의미/사용: `job_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        job_ids = [str(row.get("job_id", "")).strip() for row in affected_rows if row.get("job_id")]
        # LINE-BY-LINE: `work_order_nos`에 `[` 결과를 저장합니다. 의미/사용: `work_order_nos` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        work_order_nos = [
            # LINE-BY-LINE: `str(row.get("work_order_no", "")).strip()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            str(row.get("work_order_no", "")).strip()
            # LINE-BY-LINE: `row in affected_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for row in affected_rows
            # LINE-BY-LINE: 조건 `row.get("work_order_no")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if row.get("work_order_no")
        ]
        # LINE-BY-LINE: `job_id in job_ids` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job_id in job_ids:
            # LINE-BY-LINE: `affected_job_ids.add(job_id)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            affected_job_ids.add(job_id)
        # LINE-BY-LINE: `work_order_no in work_order_nos` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for work_order_no in work_order_nos:
            # LINE-BY-LINE: `affected_work_orders.add(work_order_no)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            affected_work_orders.add(work_order_no)

        # LINE-BY-LINE: `rule`에 `str(violation.get("rule", "unknown"))` 결과를 저장합니다. 의미/사용: `rule` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rule = str(violation.get("rule", "unknown"))
        # LINE-BY-LINE: `violation_counts_by_rule[rule]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `violation_counts_by_rule[rule]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        violation_counts_by_rule[rule] += 1
        # LINE-BY-LINE: `violation.setdefault("severity", "hard")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        violation.setdefault("severity", "hard")
        # LINE-BY-LINE: `violation["affected_job_count"]`에 `len(set(job_ids))` 결과를 저장합니다. 의미/사용: `violation["affected_job_count"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        violation["affected_job_count"] = len(set(job_ids))
        # LINE-BY-LINE: `violation["affected_work_order_count"]`에 `len(set(work_order_nos))` 결과를 저장합니다. 의미/사용: `violation["affected_work_order_count"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        violation["affected_work_order_count"] = len(set(work_order_nos))
        # LINE-BY-LINE: `violation["job_ids"]`에 `_join_unique(job_ids)` 결과를 저장합니다. 의미/사용: `violation["job_ids"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        violation["job_ids"] = _join_unique(job_ids)
        # LINE-BY-LINE: `violation["work_order_nos"]`에 `_join_unique(work_order_nos)` 결과를 저장합니다. 의미/사용: `violation["work_order_nos"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        violation["work_order_nos"] = _join_unique(work_order_nos)
        # LINE-BY-LINE: `violations.append(violation)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        violations.append(violation)

    if hard_rule_enabled("machine_bay_consistency"):
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows:
            # LINE-BY-LINE: `expected_bay`에 `None if machine_to_bay is None else machine_to_bay.get(str(row["algorithm_machine_id"]))` 결과를 저장합니다. 의미/사용: `expected_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            expected_bay = None if machine_to_bay is None else machine_to_bay.get(str(row["algorithm_machine_id"]))
            # LINE-BY-LINE: 조건 `expected_bay is not None and str(expected_bay) != str(row["algorithm_cut_bay"])`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if expected_bay is None or str(expected_bay) == str(row["algorithm_cut_bay"]):
                continue
            # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_violation(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `rule` 키에 `"machine_bay_consistency"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "rule": "machine_bay_consistency",
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row["job_id"]` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": row["job_id"],
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row["algorithm_machine_id"]` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": row["algorithm_machine_id"],
                    # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row["algorithm_cut_bay"]` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    "bay_id": row["algorithm_cut_bay"],
                    # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `expected_bay_id` 키에 `expected_bay` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "expected_bay_id": expected_bay,
                    # LINE-BY-LINE: 딕셔너리 키 `message`에는 `f"machine bay {expected_bay} != operation bay {row['algorithm_cut_bay']}"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                    "message": f"machine bay {expected_bay} != operation bay {row['algorithm_cut_bay']}",
                },
                # LINE-BY-LINE: `add_violation(...)` 호출에 `[row]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                [row],
            )

    if hard_rule_enabled("block_set_same_bay"):
        # LINE-BY-LINE: `block_to_bays` 변수에 `defaultdict(set)` 결과를 저장합니다. 의미: `block_to_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        block_to_bays: Dict[str, set] = defaultdict(set)
        # LINE-BY-LINE: `block_to_rows` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `block_to_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        block_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows:
            # LINE-BY-LINE: 조건 `row["block_set_id"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if row["block_set_id"]:
                # LINE-BY-LINE: `block_to_bays[row["block_set_id"]].add(str(row["algorithm_cut_bay"]))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                block_to_bays[row["block_set_id"]].add(str(row["algorithm_cut_bay"]))
                # LINE-BY-LINE: `block_to_rows[row["block_set_id"]].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                block_to_rows[row["block_set_id"]].append(row)
        # LINE-BY-LINE: `block_set_id, bay_ids in block_to_bays.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for block_set_id, bay_ids in block_to_bays.items():
            # LINE-BY-LINE: 조건 `len(bay_ids) > 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if len(bay_ids) <= 1:
                continue
            # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            add_violation(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `rule` 키에 `"block_set_same_bay"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "rule": "block_set_same_bay",
                    # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `block_set_id` 키에 `block_set_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "block_set_id": block_set_id,
                    # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `assigned_bay_ids` 키에 `"|".join(sorted(bay_ids))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "assigned_bay_ids": "|".join(sorted(bay_ids)),
                    # LINE-BY-LINE: 딕셔너리 키 `message`에는 `f"assigned bays={sorted(bay_ids)}"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                    "message": f"assigned bays={sorted(bay_ids)}",
                },
                # LINE-BY-LINE: `add_violation(...)` 호출에 `block_to_rows[block_set_id]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                block_to_rows[block_set_id],
            )

    # LINE-BY-LINE: `limits`에 `config.get("constraints", {}).get("machine_day_limits", {}) or {}` 결과를 저장합니다. 의미/사용: `limits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    limits = config.get("constraints", {}).get("machine_day_limits", {}) or {}
    # LINE-BY-LINE: `max_wo_count`에 `limits.get("max_wo_count")` 결과를 저장합니다. 의미/사용: `max_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_wo_count = limits.get("max_wo_count") if hard_rule_enabled("machine_day_wo_count_limit") else None
    # LINE-BY-LINE: `max_length_sum`에 `limits.get("max_length_sum")` 결과를 저장합니다. 의미/사용: `max_length_sum` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    max_length_sum = limits.get("max_length_sum") if hard_rule_enabled("machine_day_length_sum_limit") else None
    # LINE-BY-LINE: `violation in _machine_concurrent_capacity_intervals(rows, max_wo_count, max_length_sum)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for violation in _machine_concurrent_capacity_intervals(rows, max_wo_count, max_length_sum):
        # LINE-BY-LINE: `active_rows`에 `violation.pop("_active_rows", [])` 결과를 저장합니다. 의미/사용: `active_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        active_rows = violation.pop("_active_rows", [])
        # LINE-BY-LINE: `interval_key`에 `(` 결과를 저장합니다. 의미/사용: `interval_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        interval_key = (
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(violation.get("machine_id", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(violation.get("machine_id", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(violation.get("start_min", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(violation.get("start_min", "")),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(violation.get("finish_min", ""))` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(violation.get("finish_min", "")),
        )
        # LINE-BY-LINE: `violating_machine_day_buckets.add(interval_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        violating_machine_day_buckets.add(interval_key)
        # LINE-BY-LINE: `hard_violation_excess_wo_count` 값을 `int(violation.get("excess_wo_count", 0) or 0)` 기준으로 누적/증가합니다. 의미/사용: `hard_violation_excess_wo_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hard_violation_excess_wo_count += int(violation.get("excess_wo_count", 0) or 0)
        # LINE-BY-LINE: `hard_violation_length_excess_total` 값을 `float(violation.get("excess_length", 0.0) or 0.0)` 기준으로 누적/증가합니다. 의미/사용: `hard_violation_length_excess_total` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hard_violation_length_excess_total += float(violation.get("excess_length", 0.0) or 0.0)
        # LINE-BY-LINE: `add_violation(violation, active_rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        add_violation(violation, active_rows)

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `len(violations) == 0`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": len(violations) == 0,
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `len(violations)`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": len(violations),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_rule_count` 키에 `len(violations)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_rule_count": len(violations),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_bucket_count` 키에 `len(violating_machine_day_buckets)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_bucket_count": len(violating_machine_day_buckets),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_affected_job_count` 키에 `len(affected_job_ids)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_affected_job_count": len(affected_job_ids),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_affected_work_order_count` 키에 `len(affected_work_orders)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_affected_work_order_count": len(affected_work_orders),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_excess_wo_count` 키에 `hard_violation_excess_wo_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_excess_wo_count": hard_violation_excess_wo_count,
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_length_excess_total` 키에 `round(hard_violation_length_excess_total, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_length_excess_total": round(hard_violation_length_excess_total, 6),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `hard_violation_counts_by_rule` 키에 `dict(sorted(violation_counts_by_rule.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_counts_by_rule": dict(sorted(violation_counts_by_rule.items())),
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `violations` 키에 `violations` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "violations": violations,
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `expected_hard_violations` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "expected_hard_violations": 0,
        # LINE-BY-LINE: `validate_schedule_rows`에서 반환/저장할 dict의 `validation_count_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_count_note": (
            # LINE-BY-LINE: 문자열 값 `"hard_violation_count is rule-instance count. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "hard_violation_count is rule-instance count. "
            # LINE-BY-LINE: 문자열 값 `"affected/excess fields report unique W/O impact and capacity excess separately. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "affected/excess fields report unique W/O impact and capacity excess separately. "
            # LINE-BY-LINE: 문자열 값 `"Generated planning capacity is checked on closed batch W/O count/length. Legacy concurrent active interval checks only run when machine_day_* flags are enabled."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Generated planning capacity is checked on closed batch W/O count/length. Legacy concurrent active interval checks only run when machine_day_* flags are enabled."
        ),
    }


# LINE-BY-LINE: `validate_schedule(simulation, config: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def validate_schedule(simulation, config: Dict[str, Any]) -> Dict[str, Any]:
    """현재 simulation schedule에 대해 1차 hard validation을 수행."""

    # LINE-BY-LINE: `machine_to_bay`에 `{` 결과를 저장합니다. 의미/사용: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_to_bay = {
        # LINE-BY-LINE: `machine_id`를 `str(machine.bay_id)` 타입으로 선언합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id: str(machine.bay_id)
        # LINE-BY-LINE: `machine_id, machine in simulation.machines.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine_id, machine in simulation.machines.items()
        # LINE-BY-LINE: 조건 `machine.bay_id is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine.bay_id is not None
    }
    # LINE-BY-LINE: 호출자에게 `validate_schedule_rows(build_job_schedule_rows(simulation), config, machine_to_bay=machine_to_bay)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return validate_schedule_rows(build_job_schedule_rows(simulation), config, machine_to_bay=machine_to_bay)


# LINE-BY-LINE: `build_factory_layout(simulation)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_factory_layout(simulation) -> Dict[str, Any]:
    """playback용 factory layout."""

    # LINE-BY-LINE: `bays` 변수에 `{}` 결과를 저장합니다. 의미: `bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bays: Dict[str, Dict[str, Any]] = {}
    # LINE-BY-LINE: `machine in simulation.machines.values()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for machine in simulation.machines.values():
        # LINE-BY-LINE: `bay_id`에 `str(machine.bay_id or "unknown")` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(machine.bay_id or "unknown")
        # LINE-BY-LINE: `bay`에 `bays.setdefault(bay_id, {"bay_id": bay_id, "machines": []})` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
        bay = bays.setdefault(bay_id, {"bay_id": bay_id, "machines": []})
        # LINE-BY-LINE: `bay["machines"].append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        bay["machines"].append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `machine.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": machine.machine_id,
                # LINE-BY-LINE: `build_factory_layout`에서 반환/저장할 dict의 `machine_type` 키에 `machine.machine_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_type": machine.machine_type,
                # LINE-BY-LINE: `build_factory_layout`에서 반환/저장할 dict의 `position` 키에 `list(machine.position)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "position": list(machine.position),
                # LINE-BY-LINE: `build_factory_layout`에서 반환/저장할 dict의 `parallel_capacity` 키에 `machine.parallel_capacity` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "parallel_capacity": machine.parallel_capacity,
            }
        )

    # LINE-BY-LINE: 호출자에게 `{"bays": list(sorted(bays.values(), key=lambda item: item["bay_id"]))}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return {"bays": list(sorted(bays.values(), key=lambda item: item["bay_id"]))}


# LINE-BY-LINE: `build_factory_layout_from_scenario(scenario: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def build_factory_layout_from_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """scenario의 machine 정의에서 playback용 layout을 만든다."""

    # LINE-BY-LINE: `bays` 변수에 `{}` 결과를 저장합니다. 의미: `bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bays: Dict[str, Dict[str, Any]] = {}
    # LINE-BY-LINE: `machine in scenario.get("machines", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for machine in scenario.get("machines", []):
        # LINE-BY-LINE: `bay_id`에 `str(machine.get("bay_id") or "unknown")` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(machine.get("bay_id") or "unknown")
        # LINE-BY-LINE: `bay`에 `bays.setdefault(bay_id, {"bay_id": bay_id, "machines": []})` 결과를 저장합니다. 의미/사용: `bay`는 DownstreamBay 또는 cut bay dict입니다. Bay capacity와 표시 정보에 사용됩니다.
        bay = bays.setdefault(bay_id, {"bay_id": bay_id, "machines": []})
        # LINE-BY-LINE: `bay["machines"].append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        bay["machines"].append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `machine.get("machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": machine.get("machine_id", ""),
                # LINE-BY-LINE: `build_factory_layout_from_scenario`에서 반환/저장할 dict의 `machine_type` 키에 `machine.get("machine_type", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "machine_type": machine.get("machine_type", ""),
                # LINE-BY-LINE: `build_factory_layout_from_scenario`에서 반환/저장할 dict의 `position` 키에 `list(machine.get("position", [0.0, 0.0]))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "position": list(machine.get("position", [0.0, 0.0])),
                # LINE-BY-LINE: `build_factory_layout_from_scenario`에서 반환/저장할 dict의 `parallel_capacity` 키에 `machine.get("parallel_capacity", 1)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "parallel_capacity": machine.get("parallel_capacity", 1),
            }
        )
    # LINE-BY-LINE: 호출자에게 `{"bays": list(sorted(bays.values(), key=lambda item: item["bay_id"]))}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return {"bays": list(sorted(bays.values(), key=lambda item: item["bay_id"]))}


# LINE-BY-LINE: `_operation_id_from_row(row: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `str`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _operation_id_from_row(row: Dict[str, Any]) -> str:
    """schedule row와 FactoryEvent를 연결하는 operation id."""

    # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_min = _to_float_or_none(row.get("start_min"))
    # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    finish_min = _to_float_or_none(row.get("finish_min"))
    # LINE-BY-LINE: 조건 `start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if start_min is None or finish_min is None:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._operation_id_from_row] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder._operation_id_from_row] "
            # LINE-BY-LINE: `f"cause`에 `missing_start_or_finish job_id={row.get('job_id', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=missing_start_or_finish job_id={row.get('job_id', '')} "
            # LINE-BY-LINE: `f"machine_id`에 `{row.get('algorithm_machine_id', '')} "` 결과를 저장합니다. 의미/사용: `f"machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"machine_id={row.get('algorithm_machine_id', '')} "
            # LINE-BY-LINE: `f"start_min`에 `{row.get('start_min', '')} finish_min={row.get('finish_min', '')}"` 결과를 저장합니다. 의미/사용: `f"start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"start_min={row.get('start_min', '')} finish_min={row.get('finish_min', '')}"
        )
        # LINE-BY-LINE: `ValueError("schedule row requires start_min and finish_min")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("schedule row requires start_min and finish_min")
    # LINE-BY-LINE: 호출자에게 `(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return (
        # LINE-BY-LINE: `f"{row.get('job_id', '')}@{row.get('algorithm_machine_id', '')}@"`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        f"{row.get('job_id', '')}@{row.get('algorithm_machine_id', '')}@"
        # LINE-BY-LINE: `f"{start_min:.6f}-{finish_min:.6f}"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        f"{start_min:.6f}-{finish_min:.6f}"
    )


# LINE-BY-LINE: `_event_payload(event_row: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def _event_payload(event_row: Dict[str, Any]) -> Dict[str, Any]:
    """event row의 payload를 dict로 복원한다."""

    # LINE-BY-LINE: `payload`에 `event_row.get("payload", {})` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
    payload = event_row.get("payload", {})
    # LINE-BY-LINE: 조건 `payload in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if payload in (None, ""):
        # LINE-BY-LINE: 호출자에게 `{}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return {}
    # LINE-BY-LINE: 조건 `isinstance(payload, dict)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(payload, dict):
        # LINE-BY-LINE: 호출자에게 `payload`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return payload
    # LINE-BY-LINE: 조건 `isinstance(payload, str)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if isinstance(payload, str):
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: `decoded`에 `json.loads(payload)` 결과를 저장합니다. 의미/사용: `decoded` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            decoded = json.loads(payload)
        # LINE-BY-LINE: `except json.JSONDecodeError as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except json.JSONDecodeError as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._event_payload] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder._event_payload] "
                # LINE-BY-LINE: `f"cause`에 `{exc} event_id={event_row.get('event_id', '')}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} event_id={event_row.get('event_id', '')}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to decode event payload json") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to decode event payload json") from exc
        # LINE-BY-LINE: 조건 `not isinstance(decoded, dict)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not isinstance(decoded, dict):
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._event_payload] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder._event_payload] "
                # LINE-BY-LINE: `f"cause`에 `decoded_payload_not_dict event_id={event_row.get('event_id', '')}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=decoded_payload_not_dict event_id={event_row.get('event_id', '')}"
            )
            # LINE-BY-LINE: `TypeError("decoded event payload must be a dict")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise TypeError("decoded event payload must be a dict")
        # LINE-BY-LINE: 호출자에게 `decoded`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return decoded
    # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(
        # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder._event_payload] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "[ERROR][playback_builder._event_payload] "
        # LINE-BY-LINE: `f"cause`에 `payload_not_dict_or_json event_id={event_row.get('event_id', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"cause=payload_not_dict_or_json event_id={event_row.get('event_id', '')} "
        # LINE-BY-LINE: `f"payload_type`에 `{type(payload).__name__}"` 결과를 저장합니다. 의미/사용: `f"payload_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        f"payload_type={type(payload).__name__}"
    )
    # LINE-BY-LINE: `TypeError("event payload must be a dict or JSON object string")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
    raise TypeError("event payload must be a dict or JSON object string")


# LINE-BY-LINE: `build_schedule_events(simulation)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_schedule_events(simulation) -> List[Dict[str, Any]]:
    """FactoryEvent log를 playback start/finish event로 변환한다."""

    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: `event_log_rows`에 `simulation.export_event_log()` 결과를 저장합니다. 의미/사용: `event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        event_log_rows = simulation.export_event_log()
    # LINE-BY-LINE: `except AttributeError as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except AttributeError as exc:
        # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][playback_builder.build_schedule_events] cause={exc}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"[ERROR][playback_builder.build_schedule_events] cause={exc}")
        # LINE-BY-LINE: `RuntimeError("simulation does not expose export_event_log()") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("simulation does not expose export_event_log()") from exc

    # LINE-BY-LINE: 조건 `not event_log_rows and simulation.state.schedule`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not event_log_rows and simulation.state.schedule:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_events] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder.build_schedule_events] "
            # LINE-BY-LINE: `f"cause`에 `empty_event_log schedule_count={len(simulation.state.schedule)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=empty_event_log schedule_count={len(simulation.state.schedule)}"
        )
        # LINE-BY-LINE: `RuntimeError("generated simulation event log is empty")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("generated simulation event log is empty")

    # LINE-BY-LINE: 호출자에게 `build_schedule_events_from_event_log(event_log_rows)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return build_schedule_events_from_event_log(event_log_rows)


# LINE-BY-LINE: `build_schedule_events_from_operations_legacy(simulation)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_schedule_events_from_operations_legacy(simulation) -> List[Dict[str, Any]]:
    """operation schedule을 playback event로 변환하는 legacy compatibility path."""

    # LINE-BY-LINE: `events` 변수에 `[]` 결과를 저장합니다. 의미: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `operation in simulation.state.schedule` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for operation in simulation.state.schedule:
        # LINE-BY-LINE: `job`에 `simulation.jobs[operation.job_id]` 결과를 저장합니다. 의미/사용: `job`는 Job 객체 또는 job dict입니다. 예: W/O 1건의 속성.
        job = simulation.jobs[operation.job_id]
        # LINE-BY-LINE: `machine`에 `simulation.machines[operation.machine_id]` 결과를 저장합니다. 의미/사용: `machine`는 Machine 객체 또는 machine dict입니다. 예: PLS21 설비 정보.
        machine = simulation.machines[operation.machine_id]
        # LINE-BY-LINE: `cut_bay`에 `operation.cut_bay or machine.bay_id` 결과를 저장합니다. 의미/사용: `cut_bay`는 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
        cut_bay = operation.cut_bay or machine.bay_id
        # LINE-BY-LINE: `common`에 `{` 결과를 저장합니다. 의미/사용: `common` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        common = {
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `operation.job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": operation.job_id,
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `operation.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": operation.machine_id,
            # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `str(cut_bay)` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            "bay_id": str(cut_bay),
            # LINE-BY-LINE: `build_schedule_events_from_operations_legacy`에서 반환/저장할 dict의 `block_set_id` 키에 `job.block_set_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "block_set_id": job.block_set_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `job.source_machine_id or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
            "source_machine_id": job.source_machine_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `job.source_cut_bay or ""` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
            "source_cut_bay": job.source_cut_bay or "",
        }
        # LINE-BY-LINE: `events.append({"time_sec": int(round(operation.start_time * 60)), "event_type": "start", **common})`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events.append({"time_sec": int(round(operation.start_time * 60)), "event_type": "start", **common})
        # LINE-BY-LINE: `events.append({"time_sec": int(round(operation.finish_time * 60)), "event_type": "finish", **comm...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events.append({"time_sec": int(round(operation.finish_time * 60)), "event_type": "finish", **common})
    # LINE-BY-LINE: 호출자에게 `sorted(events, key=lambda event: (event["time_sec"], event["event_type"]))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(events, key=lambda event: (event["time_sec"], event["event_type"]))


# LINE-BY-LINE: `build_schedule_events_from_event_log(event_log_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_schedule_events_from_event_log(event_log_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FactoryEvent log에서 playback HTML이 사용하는 start/finish event를 만든다."""

    # LINE-BY-LINE: `events` 변수에 `[]` 결과를 저장합니다. 의미: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `event_row in sorted(list(event_log_rows), key=event_sort_key)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for event_row in sorted(list(event_log_rows), key=event_sort_key):
        # LINE-BY-LINE: `event_type`에 `str(event_row.get("event_type", ""))` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
        event_type = str(event_row.get("event_type", ""))
        # LINE-BY-LINE: 조건 `event_type not in {EVENT_PROCESS_START, EVENT_PROCESS_FINISH}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if event_type not in {EVENT_PROCESS_START, EVENT_PROCESS_FINISH}:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # LINE-BY-LINE: `payload`에 `_event_payload(event_row)` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
        payload = _event_payload(event_row)
        # LINE-BY-LINE: `playback_event_type`에 `"start" if event_type == EVENT_PROCESS_START else "finish"` 결과를 저장합니다. 의미/사용: `playback_event_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        playback_event_type = "start" if event_type == EVENT_PROCESS_START else "finish"
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: `events.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            events.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `build_schedule_events_from_event_log`에서 반환/저장할 dict의 `time_sec` 키에 `int(event_row["time_sec"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "time_sec": int(event_row["time_sec"]),
                    # LINE-BY-LINE: 딕셔너리 키 `time_min`에는 `float(event_row["time_min"])` 값을 넣습니다. 의미: simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
                    "time_min": float(event_row["time_min"]),
                    # LINE-BY-LINE: 딕셔너리 키 `event_type`에는 `playback_event_type` 값을 넣습니다. 의미: event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                    "event_type": playback_event_type,
                    # LINE-BY-LINE: `build_schedule_events_from_event_log`에서 반환/저장할 dict의 `factory_event_type` 키에 `event_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "factory_event_type": event_type,
                    # LINE-BY-LINE: 딕셔너리 키 `event_id`에는 `event_row.get("event_id", "")` 값을 넣습니다. 의미: event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
                    "event_id": event_row.get("event_id", ""),
                    # LINE-BY-LINE: `build_schedule_events_from_event_log`에서 반환/저장할 dict의 `operation_id` 키에 `event_row.get("operation_id", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "operation_id": event_row.get("operation_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `event_row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": event_row.get("job_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `event_row.get("machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": event_row.get("machine_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `str(event_row.get("bay_id", ""))` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    "bay_id": str(event_row.get("bay_id", "")),
                    # LINE-BY-LINE: `build_schedule_events_from_event_log`에서 반환/저장할 dict의 `block_set_id` 키에 `payload.get("block_set_id", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "block_set_id": payload.get("block_set_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `payload.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                    "work_order_no": payload.get("work_order_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `payload.get("project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                    "project_no": payload.get("project_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `payload.get("block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                    "block_no": payload.get("block_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `payload.get("source_machine_id", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                    "source_machine_id": payload.get("source_machine_id", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `payload.get("source_cut_bay", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                    "source_cut_bay": payload.get("source_cut_bay", ""),
                    # LINE-BY-LINE: `build_schedule_events_from_event_log`에서 반환/저장할 dict의 `source` 키에 `event_row.get("source", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "source": event_row.get("source", ""),
                }
            )
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_events_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder.build_schedule_events_from_event_log] "
                # LINE-BY-LINE: `f"cause`에 `{exc} event_id={event_row.get('event_id', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} event_id={event_row.get('event_id', '')} "
                # LINE-BY-LINE: `f"event_type`에 `{event_type} job_id={event_row.get('job_id', '')}"` 결과를 저장합니다. 의미/사용: `f"event_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"event_type={event_type} job_id={event_row.get('job_id', '')}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to build playback event from factory event log") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to build playback event from factory event log") from exc

    # LINE-BY-LINE: 호출자에게 `sorted(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(
        # LINE-BY-LINE: `sorted(...)` 호출에 `events` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        events,
        # LINE-BY-LINE: `key`에 `lambda event: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key=lambda event: (
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `event["time_sec"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            event["time_sec"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `0 if event["event_type"] == "start" else 1` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            0 if event["event_type"] == "start" else 1,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `event.get("event_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            event.get("event_id", ""),
        ),
    )


# LINE-BY-LINE: `build_schedule_events_from_rows(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_schedule_events_from_rows(schedule_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """schedule row를 playback event로 변환하는 legacy compatibility path."""

    # LINE-BY-LINE: `events` 변수에 `[]` 결과를 저장합니다. 의미: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `common`에 `{` 결과를 저장합니다. 의미/사용: `common` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        common = {
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": row.get("job_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("algorithm_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": row.get("algorithm_machine_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `str(row.get("algorithm_cut_bay", ""))` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            "bay_id": str(row.get("algorithm_cut_bay", "")),
            # LINE-BY-LINE: `build_schedule_events_from_rows`에서 반환/저장할 dict의 `block_set_id` 키에 `row.get("block_set_id", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "block_set_id": row.get("block_set_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `row.get("source_machine_id", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
            "source_machine_id": row.get("source_machine_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `row.get("source_cut_bay", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
            "source_cut_bay": row.get("source_cut_bay", ""),
        }
        # LINE-BY-LINE: `events.append({"time_sec": int(round(float(row["start_min"]) * 60)), "event_type": "start", **com...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events.append({"time_sec": int(round(float(row["start_min"]) * 60)), "event_type": "start", **common})
        # LINE-BY-LINE: `events.append({"time_sec": int(round(float(row["finish_min"]) * 60)), "event_type": "finish", **c...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events.append({"time_sec": int(round(float(row["finish_min"]) * 60)), "event_type": "finish", **common})
    # LINE-BY-LINE: 호출자에게 `sorted(events, key=lambda event: (event["time_sec"], event["event_type"]))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(events, key=lambda event: (event["time_sec"], event["event_type"]))


# LINE-BY-LINE: `build_actual_event_log_rows(schedule_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_actual_event_log_rows(schedule_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """실적 replay row를 표준 FactoryEvent log로 변환한다."""

    # LINE-BY-LINE: `event_log` 변수에 `[]` 결과를 저장합니다. 의미: `event_log`는 FactoryEvent list입니다. generated/actual DES 상태 변화를 시간순으로 기록합니다.
    event_log: List[FactoryEvent] = []
    # LINE-BY-LINE: `event_seq`에 `0` 결과를 저장합니다. 의미/사용: `event_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_seq = 0
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `job_id`에 `str(row.get("job_id", ""))` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
        job_id = str(row.get("job_id", ""))
        # LINE-BY-LINE: `machine_id`에 `str(row.get("algorithm_machine_id", ""))` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id = str(row.get("algorithm_machine_id", ""))
        # LINE-BY-LINE: `bay_id`에 `str(row.get("algorithm_cut_bay", ""))` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
        bay_id = str(row.get("algorithm_cut_bay", ""))
        # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_min = _to_float_or_none(row.get("start_min"))
        # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        finish_min = _to_float_or_none(row.get("finish_min"))
        # LINE-BY-LINE: `operation_id`에 `""` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_id = ""
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: 조건 `not job_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not job_id:
                # LINE-BY-LINE: `ValueError("missing job_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("missing job_id")
            # LINE-BY-LINE: 조건 `not machine_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not machine_id:
                # LINE-BY-LINE: `ValueError("missing machine_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("missing machine_id")
            # LINE-BY-LINE: 조건 `not bay_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not bay_id:
                # LINE-BY-LINE: `ValueError("missing bay_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("missing bay_id")
            # LINE-BY-LINE: 조건 `start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if start_min is None or finish_min is None:
                # LINE-BY-LINE: `ValueError("missing start_min or finish_min")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("missing start_min or finish_min")
            # LINE-BY-LINE: 조건 `finish_min < start_min`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if finish_min < start_min:
                # LINE-BY-LINE: `ValueError(f"finish before start: start_min={start_min}, finish_min={finish_min}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError(f"finish before start: start_min={start_min}, finish_min={finish_min}")

            # LINE-BY-LINE: `operation_id`에 `_operation_id_from_row(row)` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id = _operation_id_from_row(row)
            # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload = {
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `action_id` 키에 `f"actual:{job_id}@{machine_id}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_id": f"actual:{job_id}@{machine_id}",
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `actual_sequence` 키에 `row.get("actual_sequence", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_sequence": row.get("actual_sequence", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `row.get("project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                "project_no": row.get("project_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `row.get("block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                "block_no": row.get("block_no", ""),
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `block_set_id` 키에 `row.get("block_set_id", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "block_set_id": row.get("block_set_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `row.get("source_machine_id", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                "source_machine_id": row.get("source_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `row.get("source_cut_bay", "")` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                "source_cut_bay": row.get("source_cut_bay", ""),
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `processing_minutes` 키에 `row.get("process_min", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "processing_minutes": row.get("process_min", ""),
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `tact_time_minutes` 키에 `row.get("tact_time_minutes", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time_minutes": row.get("tact_time_minutes", ""),
                # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `row.get("plate_length", "")` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                "plate_length": row.get("plate_length", ""),
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `row.get("thickness", "")` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": row.get("thickness", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `row.get("actual_start_datetime", "")` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": row.get("actual_start_datetime", ""),
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `row.get("actual_end_datetime", "")` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": row.get("actual_end_datetime", ""),
                # LINE-BY-LINE: `build_actual_event_log_rows`에서 반환/저장할 dict의 `replay_mode` 키에 `row.get("replay_mode", "actual_historical")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "replay_mode": row.get("replay_mode", "actual_historical"),
            }

            # LINE-BY-LINE: `specs`에 `(` 결과를 저장합니다. 의미/사용: `specs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            specs = (
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `(EVENT_MACHINE_ASSIGN, start_min, "actual machine assignment")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                (EVENT_MACHINE_ASSIGN, start_min, "actual machine assignment"),
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `(EVENT_CUT_BAY_ASSIGN, start_min, "actual cut bay assignment")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                (EVENT_CUT_BAY_ASSIGN, start_min, "actual cut bay assignment"),
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `(EVENT_PROCESS_START, start_min, "actual processing start")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                (EVENT_PROCESS_START, start_min, "actual processing start"),
                # LINE-BY-LINE: `현재 표현식(...)` 호출에 `(EVENT_PROCESS_FINISH, finish_min, "actual processing finish")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                (EVENT_PROCESS_FINISH, finish_min, "actual processing finish"),
            )
            # LINE-BY-LINE: `event_type, time_min, message in specs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for event_type, time_min, message in specs:
                # LINE-BY-LINE: `event_seq` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `event_seq` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                event_seq += 1
                # LINE-BY-LINE: `event`에 `FactoryEvent(` 결과를 저장합니다. 의미/사용: `event`는 FactoryEvent 객체 또는 event row입니다. playback과 validation에 사용됩니다.
                event = FactoryEvent(
                    # LINE-BY-LINE: `event_id`에 `f"A{event_seq:08d}"` 결과를 저장합니다. 의미/사용: `event_id`는 event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
                    event_id=f"A{event_seq:08d}",
                    # LINE-BY-LINE: `event_type`에 `event_type` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
                    event_type=event_type,
                    # LINE-BY-LINE: `time_min`에 `float(time_min)` 결과를 저장합니다. 의미/사용: `time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
                    time_min=float(time_min),
                    # LINE-BY-LINE: `job_id`에 `job_id` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    job_id=job_id,
                    # LINE-BY-LINE: `machine_id`에 `machine_id` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    machine_id=machine_id,
                    # LINE-BY-LINE: `bay_id`에 `bay_id` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                    bay_id=bay_id,
                    # LINE-BY-LINE: `operation_id`에 `operation_id` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    operation_id=operation_id,
                    # LINE-BY-LINE: `source`에 `SOURCE_ACTUAL` 결과를 저장합니다. 의미/사용: `source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    source=SOURCE_ACTUAL,
                    # LINE-BY-LINE: `payload`에 `payload` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
                    payload=payload,
                    # LINE-BY-LINE: `message`에 `message` 결과를 저장합니다. 의미/사용: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                    message=message,
                )
                # LINE-BY-LINE: `validate_event_required_fields(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                validate_event_required_fields(event)
                # LINE-BY-LINE: `event_log.append(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                event_log.append(event)
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_actual_event_log_rows] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder.build_actual_event_log_rows] "
                # LINE-BY-LINE: `f"cause`에 `{exc} job_id={job_id} work_order_no={row.get('work_order_no', '')} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} job_id={job_id} work_order_no={row.get('work_order_no', '')} "
                # LINE-BY-LINE: `f"machine_id`에 `{machine_id} bay_id={bay_id} "` 결과를 저장합니다. 의미/사용: `f"machine_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"machine_id={machine_id} bay_id={bay_id} "
                # LINE-BY-LINE: `f"start_min`에 `{row.get('start_min', '')} finish_min={row.get('finish_min', '')} "` 결과를 저장합니다. 의미/사용: `f"start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"start_min={row.get('start_min', '')} finish_min={row.get('finish_min', '')} "
                # LINE-BY-LINE: `f"operation_id`에 `{operation_id}"` 결과를 저장합니다. 의미/사용: `f"operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"operation_id={operation_id}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to convert actual replay row to FactoryEvent") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to convert actual replay row to FactoryEvent") from exc

    # LINE-BY-LINE: 호출자에게 `[event.to_dict() for event in sorted(event_log, key=event_sort_key)]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [event.to_dict() for event in sorted(event_log, key=event_sort_key)]


# LINE-BY-LINE: `build_schedule_rows_from_event_log(event_log_rows: Iterable[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다. 예: `PROCESS_START` event를 생성/검증.
def build_schedule_rows_from_event_log(event_log_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FactoryEvent log에서 validation용 schedule row를 복원한다."""

    # LINE-BY-LINE: `events`에 `list(event_log_rows)` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events = list(event_log_rows)
    # LINE-BY-LINE: `events_by_operation` 변수에 `defaultdict(lambda: defaultdict(list))` 결과를 저장합니다. 의미: `events_by_operation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events_by_operation: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    # LINE-BY-LINE: `event_row in events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for event_row in events:
        # LINE-BY-LINE: `event_type`에 `str(event_row.get("event_type", ""))` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
        event_type = str(event_row.get("event_type", ""))
        # LINE-BY-LINE: 조건 `event_type not in {EVENT_MACHINE_ASSIGN, EVENT_CUT_BAY_ASSIGN, EVENT_PROCESS_START, EVENT_PROCESS...`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if event_type not in {EVENT_MACHINE_ASSIGN, EVENT_CUT_BAY_ASSIGN, EVENT_PROCESS_START, EVENT_PROCESS_FINISH}:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `operation_id`에 `str(event_row.get("operation_id", ""))` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        operation_id = str(event_row.get("operation_id", ""))
        # LINE-BY-LINE: 조건 `not operation_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not operation_id:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                # LINE-BY-LINE: `f"cause`에 `missing_operation_id event_id={event_row.get('event_id', '')} event_type={event_type}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=missing_operation_id event_id={event_row.get('event_id', '')} event_type={event_type}"
            )
            # LINE-BY-LINE: `ValueError("required event is missing operation_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("required event is missing operation_id")
        # LINE-BY-LINE: `events_by_operation[operation_id][event_type].append(event_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        events_by_operation[operation_id][event_type].append(event_row)

    # LINE-BY-LINE: `rows` 변수에 `[]` 결과를 저장합니다. 의미: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `required_event_types`에 `(EVENT_MACHINE_ASSIGN, EVENT_CUT_BAY_ASSIGN, EVENT_PROCESS_START, EVENT_PROCESS_FINISH)` 결과를 저장합니다. 의미/사용: `required_event_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_event_types = (EVENT_MACHINE_ASSIGN, EVENT_CUT_BAY_ASSIGN, EVENT_PROCESS_START, EVENT_PROCESS_FINISH)
    # LINE-BY-LINE: `operation_id, grouped in sorted(events_by_operation.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for operation_id, grouped in sorted(events_by_operation.items()):
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: `missing`에 `[event_type for event_type in required_event_types if not grouped.get(event_type)]` 결과를 저장합니다. 의미/사용: `missing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            missing = [event_type for event_type in required_event_types if not grouped.get(event_type)]
            # LINE-BY-LINE: `duplicated`에 `[event_type for event_type in required_event_types if len(grouped.get(event_type, [])) > 1]` 결과를 저장합니다. 의미/사용: `duplicated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            duplicated = [event_type for event_type in required_event_types if len(grouped.get(event_type, [])) > 1]
            # LINE-BY-LINE: 조건 `missing or duplicated`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if missing or duplicated:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                    # LINE-BY-LINE: `f"cause`에 `missing_or_duplicate_events operation_id={operation_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=missing_or_duplicate_events operation_id={operation_id} "
                    # LINE-BY-LINE: `f"missing`에 `{missing} duplicated={duplicated}"` 결과를 저장합니다. 의미/사용: `f"missing` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"missing={missing} duplicated={duplicated}"
                )
                # LINE-BY-LINE: `ValueError("operation event set must contain exactly one required event each")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("operation event set must contain exactly one required event each")

            # LINE-BY-LINE: `assign_event`에 `grouped[EVENT_MACHINE_ASSIGN][0]` 결과를 저장합니다. 의미/사용: `assign_event` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            assign_event = grouped[EVENT_MACHINE_ASSIGN][0]
            # LINE-BY-LINE: `bay_event`에 `grouped[EVENT_CUT_BAY_ASSIGN][0]` 결과를 저장합니다. 의미/사용: `bay_event` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            bay_event = grouped[EVENT_CUT_BAY_ASSIGN][0]
            # LINE-BY-LINE: `start_event`에 `grouped[EVENT_PROCESS_START][0]` 결과를 저장합니다. 의미/사용: `start_event` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            start_event = grouped[EVENT_PROCESS_START][0]
            # LINE-BY-LINE: `finish_event`에 `grouped[EVENT_PROCESS_FINISH][0]` 결과를 저장합니다. 의미/사용: `finish_event` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            finish_event = grouped[EVENT_PROCESS_FINISH][0]
            # LINE-BY-LINE: `payload`에 `_event_payload(start_event)` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload = _event_payload(start_event)
            # LINE-BY-LINE: `finish_payload`에 `_event_payload(finish_event)` 결과를 저장합니다. 의미/사용: `finish_payload` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            finish_payload = _event_payload(finish_event)
            # LINE-BY-LINE: `start_min`에 `_to_float_or_none(start_event.get("time_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            start_min = _to_float_or_none(start_event.get("time_min"))
            # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(finish_event.get("time_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            finish_min = _to_float_or_none(finish_event.get("time_min"))
            # LINE-BY-LINE: 조건 `start_min is None or finish_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if start_min is None or finish_min is None:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                    # LINE-BY-LINE: `f"cause`에 `missing_start_or_finish_time operation_id={operation_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=missing_start_or_finish_time operation_id={operation_id} "
                    # LINE-BY-LINE: `f"start`에 `{start_event.get('time_min', '')} finish={finish_event.get('time_min', '')}"` 결과를 저장합니다. 의미/사용: `f"start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"start={start_event.get('time_min', '')} finish={finish_event.get('time_min', '')}"
                )
                # LINE-BY-LINE: `ValueError("event log operation requires PROCESS_START and PROCESS_FINISH time")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("event log operation requires PROCESS_START and PROCESS_FINISH time")
            # LINE-BY-LINE: 조건 `finish_min < start_min`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if finish_min < start_min:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                    # LINE-BY-LINE: `f"cause`에 `finish_before_start operation_id={operation_id} start={start_min} finish={finish_min}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=finish_before_start operation_id={operation_id} start={start_min} finish={finish_min}"
                )
                # LINE-BY-LINE: `ValueError("PROCESS_FINISH time must be greater than or equal to PROCESS_START time")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("PROCESS_FINISH time must be greater than or equal to PROCESS_START time")

            # LINE-BY-LINE: `machine_id`에 `str(assign_event.get("machine_id") or start_event.get("machine_id") or "")` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id = str(assign_event.get("machine_id") or start_event.get("machine_id") or "")
            # LINE-BY-LINE: `bay_id`에 `str(bay_event.get("bay_id") or start_event.get("bay_id") or "")` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id = str(bay_event.get("bay_id") or start_event.get("bay_id") or "")
            # LINE-BY-LINE: `job_id`에 `str(start_event.get("job_id") or assign_event.get("job_id") or "")` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id = str(start_event.get("job_id") or assign_event.get("job_id") or "")
            # LINE-BY-LINE: 조건 `not machine_id or not bay_id or not job_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not machine_id or not bay_id or not job_id:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                    # LINE-BY-LINE: `f"cause`에 `missing_job_machine_or_bay operation_id={operation_id} "` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=missing_job_machine_or_bay operation_id={operation_id} "
                    # LINE-BY-LINE: `f"job_id`에 `{job_id} machine_id={machine_id} bay_id={bay_id}"` 결과를 저장합니다. 의미/사용: `f"job_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"job_id={job_id} machine_id={machine_id} bay_id={bay_id}"
                )
                # LINE-BY-LINE: `ValueError("event log operation requires job_id, machine_id, bay_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("event log operation requires job_id, machine_id, bay_id")

            # LINE-BY-LINE: `process_min`에 `finish_min - start_min` 결과를 저장합니다. 의미/사용: `process_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            process_min = finish_min - start_min
            # LINE-BY-LINE: `rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            rows.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": job_id,
                    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `payload.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                    "work_order_no": payload.get("work_order_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `payload.get("project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
                    "project_no": payload.get("project_no", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `payload.get("block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
                    "block_no": payload.get("block_no", ""),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `block_set_id` 키에 `payload.get("block_set_id", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "block_set_id": payload.get("block_set_id", ""),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `scheduled_day` 키에 `f"event_day_{int(start_min // 1440)}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "scheduled_day": f"event_day_{int(start_min // 1440)}",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `capacity_date` 키에 `f"event_day_{int(start_min // 1440)}"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "capacity_date": f"event_day_{int(start_min // 1440)}",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `capacity_basis` 키에 `"event_log_process_start"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "capacity_basis": "event_log_process_start",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `start_min` 키에 `round(start_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "start_min": round(start_min, 6),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `finish_min` 키에 `round(finish_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "finish_min": round(finish_min, 6),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `process_min` 키에 `round(process_min, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "process_min": round(process_min, 6),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `tact_time_minutes` 키에 `payload.get("tact_time_minutes", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "tact_time_minutes": payload.get("tact_time_minutes", ""),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `algorithm_machine_id` 키에 `machine_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "algorithm_machine_id": machine_id,
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `algorithm_cut_bay` 키에 `bay_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "algorithm_cut_bay": bay_id,
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `machine_type` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_type": "",
                    # LINE-BY-LINE: 딕셔너리 키 `source_machine_id`에는 `payload.get("source_machine_id", finish_payload.get("source_machine_id", ""))` 값을 넣습니다. 의미: 실적 데이터의 원본 RT_EQP_NM입니다. 예: `PLS21`, 사용: actual replay 장비 identity 검증.
                    "source_machine_id": payload.get("source_machine_id", finish_payload.get("source_machine_id", "")),
                    # LINE-BY-LINE: 딕셔너리 키 `source_cut_bay`에는 `payload.get("source_cut_bay", finish_payload.get("source_cut_bay", ""))` 값을 넣습니다. 의미: 실적 데이터의 원본 CUT_BAY입니다. 예: `22`, 사용: actual replay 비교 기준.
                    "source_cut_bay": payload.get("source_cut_bay", finish_payload.get("source_cut_bay", "")),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `machine_match` 키에 `str(machine_id) == str(payload.get("source_machine_id", ""))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "machine_match": str(machine_id) == str(payload.get("source_machine_id", "")),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `bay_match` 키에 `str(bay_id) == str(payload.get("source_cut_bay", ""))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "bay_match": str(bay_id) == str(payload.get("source_cut_bay", "")),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `actual_start_min` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "actual_start_min": "",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `actual_finish_min` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "actual_finish_min": "",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `planned_start_min` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "planned_start_min": "",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `planned_finish_min` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "planned_finish_min": "",
                    # LINE-BY-LINE: 딕셔너리 키 `due_date_minutes`에는 `""` 값을 넣습니다. 의미: 기준 시점 대비 납기/후공정 시각입니다. 사용: due_date_urgency penalty.
                    "due_date_minutes": "",
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `due_slack_min` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "due_slack_min": "",
                    # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `payload.get("plate_length", "")` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
                    "plate_length": payload.get("plate_length", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `payload.get("thickness", "")` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                    "thickness": payload.get("thickness", ""),
                    # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `payload.get("actual_duration_minutes", "")` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                    "actual_duration_minutes": payload.get("actual_duration_minutes", ""),
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `operation_id` 키에 `operation_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "operation_id": operation_id,
                    # LINE-BY-LINE: `build_schedule_rows_from_event_log`에서 반환/저장할 dict의 `event_source` 키에 `start_event.get("source", "")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "event_source": start_event.get("source", ""),
                }
            )
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_schedule_rows_from_event_log] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder.build_schedule_rows_from_event_log] "
                # LINE-BY-LINE: `f"cause`에 `{exc} operation_id={operation_id}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} operation_id={operation_id}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to reconstruct schedule row from event log") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to reconstruct schedule row from event log") from exc

    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `validate_event_log_identity` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def validate_event_log_identity(
    # LINE-BY-LINE: `event_log_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_log_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `schedule_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: Iterable[Dict[str, Any]],
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """FactoryEvent log가 schedule row의 배정/시작/종료를 그대로 담고 있는지 확인한다."""

    # LINE-BY-LINE: `rows`에 `list(schedule_rows)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = list(schedule_rows)
    # LINE-BY-LINE: `events`에 `list(event_log_rows)` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events = list(event_log_rows)
    # LINE-BY-LINE: `required_event_types`에 `(` 결과를 저장합니다. 의미/사용: `required_event_types` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    required_event_types = (
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_MACHINE_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        EVENT_MACHINE_ASSIGN,
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_CUT_BAY_ASSIGN` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        EVENT_CUT_BAY_ASSIGN,
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_START` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        EVENT_PROCESS_START,
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `EVENT_PROCESS_FINISH` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        EVENT_PROCESS_FINISH,
    )
    # LINE-BY-LINE: `event_type_counts` 변수에 `defaultdict(int)` 결과를 저장합니다. 의미: `event_type_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_type_counts: Dict[str, int] = defaultdict(int)
    # LINE-BY-LINE: `events_by_key` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `events_by_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events_by_key: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `violations` 변수에 `[]` 결과를 저장합니다. 의미: `violations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violations: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `tolerance`에 `1e-6` 결과를 저장합니다. 의미/사용: `tolerance` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tolerance = 1e-6

    # LINE-BY-LINE: `event_row in events` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for event_row in events:
        # LINE-BY-LINE: `event_type`에 `str(event_row.get("event_type", ""))` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
        event_type = str(event_row.get("event_type", ""))
        # LINE-BY-LINE: `event_type_counts[event_type]` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `event_type_counts[event_type]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        event_type_counts[event_type] += 1
        # LINE-BY-LINE: 조건 `event_type in required_event_types`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if event_type in required_event_types:
            # LINE-BY-LINE: `operation_id`에 `str(event_row.get("operation_id", ""))` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id = str(event_row.get("operation_id", ""))
            # LINE-BY-LINE: 조건 `not operation_id`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not operation_id:
                # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
                print(
                    # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.validate_event_log_identity] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "[ERROR][playback_builder.validate_event_log_identity] "
                    # LINE-BY-LINE: `f"cause`에 `missing_operation_id event_id={event_row.get('event_id', '')}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    f"cause=missing_operation_id event_id={event_row.get('event_id', '')}"
                )
                # LINE-BY-LINE: `ValueError("required factory event is missing operation_id")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
                raise ValueError("required factory event is missing operation_id")
            # LINE-BY-LINE: `events_by_key[(operation_id, event_type)].append(event_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            events_by_key[(operation_id, event_type)].append(event_row)

    # LINE-BY-LINE: `add_violation(row: Dict[str, Any], rule: str, message: str, expected: Any, actual: Any)` 함수를 정의합니다. 반환 타입: `None`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
    def add_violation(row: Dict[str, Any], rule: str, message: str, expected: Any, actual: Any) -> None:
        """event log identity validation 위반을 누적한다."""

        # LINE-BY-LINE: `violations.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        violations.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `add_violation`에서 반환/저장할 dict의 `rule` 키에 `rule` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "rule": rule,
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("algorithm_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row.get("algorithm_machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `bay_id`에는 `row.get("algorithm_cut_bay", "")` 값을 넣습니다. 의미: Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
                "bay_id": row.get("algorithm_cut_bay", ""),
                # LINE-BY-LINE: `add_violation`에서 반환/저장할 dict의 `operation_id` 키에 `_operation_id_from_row(row)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "operation_id": _operation_id_from_row(row),
                # LINE-BY-LINE: `add_violation`에서 반환/저장할 dict의 `expected` 키에 `expected` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "expected": expected,
                # LINE-BY-LINE: `add_violation`에서 반환/저장할 dict의 `actual` 키에 `actual` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual": actual,
                # LINE-BY-LINE: 딕셔너리 키 `message`에는 `message` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
                "message": message,
            }
        )

    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows:
            # LINE-BY-LINE: `operation_id`에 `_operation_id_from_row(row)` 결과를 저장합니다. 의미/사용: `operation_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            operation_id = _operation_id_from_row(row)
            # LINE-BY-LINE: `expected_machine`에 `str(row.get("algorithm_machine_id", ""))` 결과를 저장합니다. 의미/사용: `expected_machine` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            expected_machine = str(row.get("algorithm_machine_id", ""))
            # LINE-BY-LINE: `expected_bay`에 `str(row.get("algorithm_cut_bay", ""))` 결과를 저장합니다. 의미/사용: `expected_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            expected_bay = str(row.get("algorithm_cut_bay", ""))
            # LINE-BY-LINE: `expected_job`에 `str(row.get("job_id", ""))` 결과를 저장합니다. 의미/사용: `expected_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            expected_job = str(row.get("job_id", ""))
            # LINE-BY-LINE: `start_min`에 `_to_float_or_none(row.get("start_min"))` 결과를 저장합니다. 의미/사용: `start_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            start_min = _to_float_or_none(row.get("start_min"))
            # LINE-BY-LINE: `finish_min`에 `_to_float_or_none(row.get("finish_min"))` 결과를 저장합니다. 의미/사용: `finish_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            finish_min = _to_float_or_none(row.get("finish_min"))
            # LINE-BY-LINE: `expected_times`에 `{` 결과를 저장합니다. 의미/사용: `expected_times` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            expected_times = {
                # LINE-BY-LINE: `EVENT_MACHINE_ASSIGN`를 `start_min,` 타입으로 선언합니다. 의미/사용: `EVENT_MACHINE_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                EVENT_MACHINE_ASSIGN: start_min,
                # LINE-BY-LINE: `EVENT_CUT_BAY_ASSIGN`를 `start_min,` 타입으로 선언합니다. 의미/사용: `EVENT_CUT_BAY_ASSIGN` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                EVENT_CUT_BAY_ASSIGN: start_min,
                # LINE-BY-LINE: `EVENT_PROCESS_START`를 `start_min,` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_START` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                EVENT_PROCESS_START: start_min,
                # LINE-BY-LINE: `EVENT_PROCESS_FINISH`를 `finish_min,` 타입으로 선언합니다. 의미/사용: `EVENT_PROCESS_FINISH` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                EVENT_PROCESS_FINISH: finish_min,
            }

            # LINE-BY-LINE: `event_type, expected_time in expected_times.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for event_type, expected_time in expected_times.items():
                # LINE-BY-LINE: `matching_events`에 `events_by_key.get((operation_id, event_type), [])` 결과를 저장합니다. 의미/사용: `matching_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                matching_events = events_by_key.get((operation_id, event_type), [])
                # LINE-BY-LINE: 조건 `len(matching_events) != 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if len(matching_events) != 1:
                    # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    add_violation(
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        row,
                        # LINE-BY-LINE: 문자열 값 `"event_count_mismatch"`를 `add_violation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                        "event_count_mismatch",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `f"{event_type} event count for operation is {len(matching_events)}"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        f"{event_type} event count for operation is {len(matching_events)}",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `1` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        1,
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `len(matching_events)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        len(matching_events),
                    )
                    # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                    continue

                # LINE-BY-LINE: `event_row`에 `matching_events[0]` 결과를 저장합니다. 의미/사용: `event_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                event_row = matching_events[0]
                # LINE-BY-LINE: `actual_time`에 `_to_float_or_none(event_row.get("time_min"))` 결과를 저장합니다. 의미/사용: `actual_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                actual_time = _to_float_or_none(event_row.get("time_min"))
                # LINE-BY-LINE: 조건 `expected_time is None or actual_time is None or abs(float(expected_time) - actual_time) > tolerance`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if expected_time is None or actual_time is None or abs(float(expected_time) - actual_time) > tolerance:
                    # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    add_violation(
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        row,
                        # LINE-BY-LINE: 문자열 값 `"event_time_mismatch"`를 `add_violation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                        "event_time_mismatch",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `f"{event_type} time does not match schedule row"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        f"{event_type} time does not match schedule row",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `expected_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        expected_time,
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `actual_time` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        actual_time,
                    )

                # LINE-BY-LINE: 조건 `str(event_row.get("job_id", "")) != expected_job`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if str(event_row.get("job_id", "")) != expected_job:
                    # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    add_violation(
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        row,
                        # LINE-BY-LINE: 문자열 값 `"event_job_mismatch"`를 `add_violation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                        "event_job_mismatch",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `f"{event_type} job_id does not match schedule row"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        f"{event_type} job_id does not match schedule row",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `expected_job` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        expected_job,
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `event_row.get("job_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        event_row.get("job_id", ""),
                    )

                # LINE-BY-LINE: 조건 `str(event_row.get("machine_id", "")) != expected_machine`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if str(event_row.get("machine_id", "")) != expected_machine:
                    # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    add_violation(
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        row,
                        # LINE-BY-LINE: 문자열 값 `"event_machine_mismatch"`를 `add_violation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                        "event_machine_mismatch",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `f"{event_type} machine_id does not match schedule row"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        f"{event_type} machine_id does not match schedule row",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `expected_machine` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        expected_machine,
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `event_row.get("machine_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        event_row.get("machine_id", ""),
                    )

                # LINE-BY-LINE: 조건 `str(event_row.get("bay_id", "")) != expected_bay`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if str(event_row.get("bay_id", "")) != expected_bay:
                    # LINE-BY-LINE: `add_violation(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    add_violation(
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `row` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        row,
                        # LINE-BY-LINE: 문자열 값 `"event_bay_mismatch"`를 `add_violation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                        "event_bay_mismatch",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `f"{event_type} bay_id does not match schedule row"` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        f"{event_type} bay_id does not match schedule row",
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `expected_bay` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        expected_bay,
                        # LINE-BY-LINE: `add_violation(...)` 호출에 `event_row.get("bay_id", "")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                        event_row.get("bay_id", ""),
                    )
    # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except Exception as exc:
        # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][playback_builder.validate_event_log_identity] cause={exc}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"[ERROR][playback_builder.validate_event_log_identity] cause={exc}")
        # LINE-BY-LINE: `RuntimeError("failed to validate event log identity") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("failed to validate event log identity") from exc

    # LINE-BY-LINE: `expected_required_event_count`에 `len(rows) * len(required_event_types)` 결과를 저장합니다. 의미/사용: `expected_required_event_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    expected_required_event_count = len(rows) * len(required_event_types)
    # LINE-BY-LINE: `actual_required_event_count`에 `sum(event_type_counts.get(event_type, 0) for event_type in required_event_types)` 결과를 저장합니다. 의미/사용: `actual_required_event_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_required_event_count = sum(event_type_counts.get(event_type, 0) for event_type in required_event_types)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `event_identity_passed` 키에 `len(violations) == 0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_identity_passed": len(violations) == 0,
        # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `len(violations)`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "identity_error_count": len(violations),
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `record_count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "record_count": len(rows),
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `event_count` 키에 `len(events)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_count": len(events),
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `required_event_count_expected` 키에 `expected_required_event_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "required_event_count_expected": expected_required_event_count,
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `required_event_count_actual` 키에 `actual_required_event_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "required_event_count_actual": actual_required_event_count,
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `event_type_counts` 키에 `dict(sorted(event_type_counts.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_type_counts": dict(sorted(event_type_counts.items())),
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `violations` 키에 `violations` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "violations": violations,
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `validation_type` 키에 `"factory_event_log_identity"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_type": "factory_event_log_identity",
        # LINE-BY-LINE: `validate_event_log_identity`에서 반환/저장할 dict의 `validation_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_note": (
            # LINE-BY-LINE: 문자열 값 `"Checks MACHINE_ASSIGN, CUT_BAY_ASSIGN, PROCESS_START, and PROCESS_FINISH "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Checks MACHINE_ASSIGN, CUT_BAY_ASSIGN, PROCESS_START, and PROCESS_FINISH "
            # LINE-BY-LINE: 문자열 값 `"events against schedule rows. Extra downstream/resource events are allowed."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "events against schedule rows. Extra downstream/resource events are allowed."
        ),
    }


# LINE-BY-LINE: `validate_event_log_constraints` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def validate_event_log_constraints(
    # LINE-BY-LINE: `event_log_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_log_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `config`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config: Dict[str, Any],
    # LINE-BY-LINE: `machine_to_bay` 변수에 `None` 결과를 저장합니다. 의미: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_to_bay: Optional[Dict[str, str]] = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """FactoryEvent log만으로 hard constraint audit을 수행한다."""

    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: `reconstructed_rows`에 `build_schedule_rows_from_event_log(event_log_rows)` 결과를 저장합니다. 의미/사용: `reconstructed_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        reconstructed_rows = build_schedule_rows_from_event_log(event_log_rows)
        # LINE-BY-LINE: `validation`에 `validate_schedule_rows(reconstructed_rows, config, machine_to_bay=machine_to_bay)` 결과를 저장합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        validation = validate_schedule_rows(reconstructed_rows, config, machine_to_bay=machine_to_bay)
        # LINE-BY-LINE: `validation["validation_type"]`에 `"factory_event_log_constraints"` 결과를 저장합니다. 의미/사용: `validation["validation_type"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        validation["validation_type"] = "factory_event_log_constraints"
        # LINE-BY-LINE: `validation["event_reconstructed_row_count"]`에 `len(reconstructed_rows)` 결과를 저장합니다. 의미/사용: `validation["event_reconstructed_row_count"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        validation["event_reconstructed_row_count"] = len(reconstructed_rows)
        # LINE-BY-LINE: `validation["validation_note"]`에 `(` 결과를 저장합니다. 의미/사용: `validation["validation_note"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        validation["validation_note"] = (
            # LINE-BY-LINE: 문자열 값 `"This validation reconstructs operation rows from FactoryEvent log and then applies "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "This validation reconstructs operation rows from FactoryEvent log and then applies "
            # LINE-BY-LINE: 문자열 값 `"the same hard constraint audit. It must match schedule-row validation for observable fields."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "the same hard constraint audit. It must match schedule-row validation for observable fields."
        )
        # LINE-BY-LINE: 호출자에게 `validation`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return validation
    # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except Exception as exc:
        # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][playback_builder.validate_event_log_constraints] cause={exc}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"[ERROR][playback_builder.validate_event_log_constraints] cause={exc}")
        # LINE-BY-LINE: `RuntimeError("failed to validate event log constraints") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("failed to validate event log constraints") from exc


# LINE-BY-LINE: `assert_event_constraint_validation_matches` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def assert_event_constraint_validation_matches(
    # LINE-BY-LINE: `schedule_validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `schedule_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_validation: Dict[str, Any],
    # LINE-BY-LINE: `event_validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `event_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_validation: Dict[str, Any],
    # LINE-BY-LINE: `label`를 `str,` 타입으로 선언합니다. 의미/사용: `label` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    label: str,
# LINE-BY-LINE: `) -> None:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> None:
    """schedule row validator와 event log validator의 핵심 결과 일치 확인."""

    # LINE-BY-LINE: `keys`에 `(` 결과를 저장합니다. 의미/사용: `keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    keys = (
        # LINE-BY-LINE: 문자열 값 `"hard_passed"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_passed",
        # LINE-BY-LINE: 문자열 값 `"hard_violation_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_violation_count",
        # LINE-BY-LINE: 문자열 값 `"hard_violation_affected_job_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_violation_affected_job_count",
        # LINE-BY-LINE: 문자열 값 `"hard_violation_excess_wo_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_violation_excess_wo_count",
        # LINE-BY-LINE: 문자열 값 `"hard_violation_length_excess_total"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_violation_length_excess_total",
        # LINE-BY-LINE: 문자열 값 `"hard_violation_counts_by_rule"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "hard_violation_counts_by_rule",
    )
    # LINE-BY-LINE: `mismatches`에 `[]` 결과를 저장합니다. 의미/사용: `mismatches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mismatches = []
    # LINE-BY-LINE: `key in keys` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for key in keys:
        # LINE-BY-LINE: 조건 `schedule_validation.get(key) != event_validation.get(key)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if schedule_validation.get(key) != event_validation.get(key):
            # LINE-BY-LINE: `mismatches.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            mismatches.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `assert_event_constraint_validation_matches`에서 반환/저장할 dict의 `key` 키에 `key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "key": key,
                    # LINE-BY-LINE: `assert_event_constraint_validation_matches`에서 반환/저장할 dict의 `schedule` 키에 `schedule_validation.get(key)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "schedule": schedule_validation.get(key),
                    # LINE-BY-LINE: `assert_event_constraint_validation_matches`에서 반환/저장할 dict의 `event` 키에 `event_validation.get(key)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "event": event_validation.get(key),
                }
            )

    # LINE-BY-LINE: 조건 `mismatches`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if mismatches:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.assert_event_constraint_validation_matches] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder.assert_event_constraint_validation_matches] "
            # LINE-BY-LINE: `f"cause`에 `validation_mismatch label={label} mismatches={mismatches}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=validation_mismatch label={label} mismatches={mismatches}"
        )
        # LINE-BY-LINE: `RuntimeError(f"schedule/event constraint validation mismatch: {label}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError(f"schedule/event constraint validation mismatch: {label}")


# LINE-BY-LINE: `build_validation_failure_event_rows` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_validation_failure_event_rows(
    # LINE-BY-LINE: `validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation: Dict[str, Any],
    # LINE-BY-LINE: `event_prefix`를 `str,` 타입으로 선언합니다. 의미/사용: `event_prefix` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_prefix: str,
    # LINE-BY-LINE: `start_index`를 `int,` 타입으로 선언합니다. 의미/사용: `start_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    start_index: int,
# LINE-BY-LINE: `) -> List[Dict[str, Any]]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> List[Dict[str, Any]]:
    """hard validation failure를 FactoryEvent log에 표시하기 위한 validation event 생성."""

    # LINE-BY-LINE: `failure_events` 변수에 `[]` 결과를 저장합니다. 의미: `failure_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    failure_events: List[FactoryEvent] = []
    # LINE-BY-LINE: `violations`에 `validation.get("violations", []) or []` 결과를 저장합니다. 의미/사용: `violations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    violations = validation.get("violations", []) or []
    # LINE-BY-LINE: `offset, violation in enumerate(violations, start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for offset, violation in enumerate(violations, start=1):
        # LINE-BY-LINE: `time_min`에 `_to_float_or_none(violation.get("start_min"))` 결과를 저장합니다. 의미/사용: `time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
        time_min = _to_float_or_none(violation.get("start_min"))
        # LINE-BY-LINE: 조건 `time_min is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if time_min is None:
            # LINE-BY-LINE: `time_min`에 `0.0` 결과를 저장합니다. 의미/사용: `time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
            time_min = 0.0
        # LINE-BY-LINE: `message`에 `str(violation.get("message") or violation.get("rule") or "hard validation failure")` 결과를 저장합니다. 의미/사용: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
        message = str(violation.get("message") or violation.get("rule") or "hard validation failure")
        # LINE-BY-LINE: `payload`에 `{` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
        payload = {
            # LINE-BY-LINE: `key`를 `value` 타입으로 선언합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key: value
            # LINE-BY-LINE: `key, value in violation.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key, value in violation.items()
            # LINE-BY-LINE: 조건 `not str(key).startswith("_")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not str(key).startswith("_")
        }
        # LINE-BY-LINE: `event`에 `FactoryEvent(` 결과를 저장합니다. 의미/사용: `event`는 FactoryEvent 객체 또는 event row입니다. playback과 validation에 사용됩니다.
        event = FactoryEvent(
            # LINE-BY-LINE: `event_id`에 `f"{event_prefix}{start_index + offset:08d}"` 결과를 저장합니다. 의미/사용: `event_id`는 event log 안에서 event 1건을 식별하는 ID입니다. 사용: playback/event validation.
            event_id=f"{event_prefix}{start_index + offset:08d}",
            # LINE-BY-LINE: `event_type`에 `EVENT_VALIDATION_FAILURE` 결과를 저장합니다. 의미/사용: `event_type`는 event 종류입니다. 예: `PROCESS_START`, 사용: event validation과 playback 상태 전환.
            event_type=EVENT_VALIDATION_FAILURE,
            # LINE-BY-LINE: `time_min`에 `float(time_min)` 결과를 저장합니다. 의미/사용: `time_min`는 simulation 기준 분 단위 시각입니다. 사용: event 정렬과 playback timeline.
            time_min=float(time_min),
            # LINE-BY-LINE: `job_id`에 `str(violation.get("job_id") or "").strip() or None` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            job_id=str(violation.get("job_id") or "").strip() or None,
            # LINE-BY-LINE: `machine_id`에 `str(violation.get("machine_id") or "").strip() or None` 결과를 저장합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            machine_id=str(violation.get("machine_id") or "").strip() or None,
            # LINE-BY-LINE: `bay_id`에 `str(violation.get("bay_id") or "").strip() or None` 결과를 저장합니다. 의미/사용: `bay_id`는 Bay ID입니다. 예: `22`, 사용: 설비 위치와 Bay별 부하 집계.
            bay_id=str(violation.get("bay_id") or "").strip() or None,
            # LINE-BY-LINE: `source`에 `SOURCE_VALIDATION` 결과를 저장합니다. 의미/사용: `source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            source=SOURCE_VALIDATION,
            # LINE-BY-LINE: `payload`에 `payload` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
            payload=payload,
            # LINE-BY-LINE: `message`에 `message` 결과를 저장합니다. 의미/사용: `message`는 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            message=message,
        )
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: `validate_event_required_fields(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            validate_event_required_fields(event)
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.build_validation_failure_event_rows] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][playback_builder.build_validation_failure_event_rows] "
                # LINE-BY-LINE: `f"cause`에 `{exc} rule={violation.get('rule', '')} message={message}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause={exc} rule={violation.get('rule', '')} message={message}"
            )
            # LINE-BY-LINE: `RuntimeError("failed to build validation failure event") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("failed to build validation failure event") from exc
        # LINE-BY-LINE: `failure_events.append(event)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        failure_events.append(event)

    # LINE-BY-LINE: 호출자에게 `[event.to_dict() for event in sorted(failure_events, key=event_sort_key)]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [event.to_dict() for event in sorted(failure_events, key=event_sort_key)]


# LINE-BY-LINE: `build_metrics(simulation, validation: Dict[str, Any], rows: List[Dict[str, Any]], time_summ...)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _metric_range(values: Dict[str, Any]) -> float:
    """max-min 편차 metric을 계산한다.

    입력 예:
    - 설비별 처리시간 부하: {"PLS21": 120.5, "PLS22": 80.0}
    - 설비별 할당 W/O 수: {"PLS21": 12, "PLS22": 9}

    빈 dict는 평가 대상이 없는 것이므로 조용히 0으로 처리하지 않고 실패시킨다.
    """

    if not values:
        print("[ERROR][playback_builder._metric_range] cause=empty_values")
        raise ValueError("cannot compute range from empty values")
    numeric_values = [float(value) for value in values.values()]
    return max(numeric_values) - min(numeric_values)


# LINE-BY-LINE: `build_metrics(simulation, validation: Dict[str, Any], rows: List[Dict[str, Any]], time_summ...)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_metrics(simulation, validation: Dict[str, Any], rows: List[Dict[str, Any]], time_summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """simulation 객체와 schedule row로 playback summary metrics를 만든다."""

    # LINE-BY-LINE: `machine_loads`에 `dict(simulation.state.machine_loads)` 결과를 저장합니다. 의미/사용: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_loads = dict(simulation.state.machine_loads)
    # LINE-BY-LINE: `bay_loads` 변수에 `defaultdict(float)` 결과를 저장합니다. 의미: `bay_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_loads: Dict[str, float] = defaultdict(float)
    machine_job_counts: Dict[str, int] = {machine_id: 0 for machine_id in simulation.machines}
    bay_job_counts: Dict[str, int] = {}
    for machine_id, machine in simulation.machines.items():
        if machine.bay_id is None or str(machine.bay_id).strip() == "":
            print(
                "[ERROR][playback_builder.build_metrics] "
                f"cause=missing_machine_bay machine_id={machine_id}"
            )
            raise ValueError(f"missing bay_id for machine: {machine_id}")
        bay_job_counts.setdefault(str(machine.bay_id), 0)
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `bay_loads[str(row["algorithm_cut_bay"])]` 값을 `float(row["process_min"])` 기준으로 누적/증가합니다. 의미/사용: `bay_loads[str(row["algorithm_cut_bay"])]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bay_loads[str(row["algorithm_cut_bay"])] += float(row["process_min"])
        machine_id = str(row["algorithm_machine_id"])
        bay_id = str(row["algorithm_cut_bay"])
        if machine_id not in machine_job_counts:
            print(
                "[ERROR][playback_builder.build_metrics] "
                f"cause=unknown_schedule_machine machine_id={machine_id}"
            )
            raise KeyError(f"unknown schedule machine: {machine_id}")
        machine_job_counts[machine_id] += 1
        bay_job_counts[bay_id] = bay_job_counts.get(bay_id, 0) + 1

    # LINE-BY-LINE: `machine_matches`에 `sum(1 for row in rows if row["machine_match"])` 결과를 저장합니다. 의미/사용: `machine_matches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_matches = sum(1 for row in rows if row["machine_match"])
    # LINE-BY-LINE: `bay_matches`에 `sum(1 for row in rows if row["bay_match"])` 결과를 저장합니다. 의미/사용: `bay_matches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_matches = sum(1 for row in rows if row["bay_match"])
    # LINE-BY-LINE: `total`에 `max(len(rows), 1)` 결과를 저장합니다. 의미/사용: `total` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    total = max(len(rows), 1)
    # LINE-BY-LINE: `metrics`에 `{` 결과를 저장합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics = {
        # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `len(rows)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "scheduled_jobs": len(rows),
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `makespan_minutes` 키에 `simulation.get_makespan()` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "makespan_minutes": simulation.get_makespan(),
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `machine_loads` 키에 `machine_loads` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_loads": machine_loads,
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `bay_process_loads` 키에 `dict(bay_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bay_process_loads": dict(bay_loads),
        "machine_load_imbalance": _metric_range(machine_loads),
        "bay_process_load_imbalance": _metric_range(bay_loads),
        "machine_job_counts": dict(machine_job_counts),
        "bay_job_counts": dict(bay_job_counts),
        "machine_job_count_imbalance": _metric_range(machine_job_counts),
        "bay_job_count_imbalance": _metric_range(bay_job_counts),
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `machine_match_rate` 키에 `machine_matches / total` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_match_rate": machine_matches / total,
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `bay_match_rate` 키에 `bay_matches / total` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bay_match_rate": bay_matches / total,
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `validation["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": validation["hard_passed"],
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `validation["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": validation["hard_violation_count"],
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `hard_violation_affected_job_count` 키에 `validation.get("hard_violation_affected_job_count", 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_affected_job_count": validation.get("hard_violation_affected_job_count", 0),
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `hard_violation_excess_wo_count` 키에 `validation.get("hard_violation_excess_wo_count", 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_excess_wo_count": validation.get("hard_violation_excess_wo_count", 0),
        # LINE-BY-LINE: `build_metrics`에서 반환/저장할 dict의 `hard_violation_length_excess_total` 키에 `validation.get("hard_violation_length_excess_total", 0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_length_excess_total": validation.get("hard_violation_length_excess_total", 0.0),
    }
    # LINE-BY-LINE: 조건 `time_summary is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if time_summary is not None:
        # LINE-BY-LINE: `metrics["time_validation"]`에 `time_summary` 결과를 저장합니다. 의미/사용: `metrics["time_validation"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        metrics["time_validation"] = time_summary
    # LINE-BY-LINE: 호출자에게 `metrics`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return metrics


# LINE-BY-LINE: `build_metrics_from_rows` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def build_metrics_from_rows(
    # LINE-BY-LINE: `schedule_rows`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: List[Dict[str, Any]],
    # LINE-BY-LINE: `validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation: Dict[str, Any],
    # LINE-BY-LINE: `time_summary` 변수에 `None` 결과를 저장합니다. 의미: `time_summary` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    time_summary: Dict[str, Any] | None = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """simulation 없이 row 기반 metrics를 만든다."""

    # LINE-BY-LINE: `machine_loads` 변수에 `defaultdict(float)` 결과를 저장합니다. 의미: `machine_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_loads: Dict[str, float] = defaultdict(float)
    # LINE-BY-LINE: `bay_loads` 변수에 `defaultdict(float)` 결과를 저장합니다. 의미: `bay_loads` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_loads: Dict[str, float] = defaultdict(float)
    machine_job_counts: Dict[str, int] = defaultdict(int)
    bay_job_counts: Dict[str, int] = defaultdict(int)
    # LINE-BY-LINE: `row in schedule_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in schedule_rows:
        # LINE-BY-LINE: `process_min`에 `float(row.get("process_min", 0.0) or 0.0)` 결과를 저장합니다. 의미/사용: `process_min` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_min = float(row.get("process_min", 0.0) or 0.0)
        # LINE-BY-LINE: `machine_loads[str(row.get("algorithm_machine_id", ""))]` 값을 `process_min` 기준으로 누적/증가합니다. 의미/사용: `get("algorithm_machine_id", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_id = str(row.get("algorithm_machine_id", ""))
        machine_loads[machine_id] += process_min
        # LINE-BY-LINE: `bay_loads[str(row.get("algorithm_cut_bay", ""))]` 값을 `process_min` 기준으로 누적/증가합니다. 의미/사용: `get("algorithm_cut_bay", ""))]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bay_id = str(row.get("algorithm_cut_bay", ""))
        bay_loads[bay_id] += process_min
        machine_job_counts[machine_id] += 1
        bay_job_counts[bay_id] += 1

    # LINE-BY-LINE: `total`에 `max(len(schedule_rows), 1)` 결과를 저장합니다. 의미/사용: `total` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    total = max(len(schedule_rows), 1)
    # LINE-BY-LINE: `machine_matches`에 `sum(1 for row in schedule_rows if row.get("machine_match") in (True, "True", "true"))` 결과를 저장합니다. 의미/사용: `machine_matches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_matches = sum(1 for row in schedule_rows if row.get("machine_match") in (True, "True", "true"))
    # LINE-BY-LINE: `bay_matches`에 `sum(1 for row in schedule_rows if row.get("bay_match") in (True, "True", "true"))` 결과를 저장합니다. 의미/사용: `bay_matches` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_matches = sum(1 for row in schedule_rows if row.get("bay_match") in (True, "True", "true"))
    # LINE-BY-LINE: `makespan`에 `max((float(row["finish_min"]) for row in schedule_rows), default=0.0)` 결과를 저장합니다. 의미/사용: `makespan` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    makespan = max((float(row["finish_min"]) for row in schedule_rows), default=0.0)
    # LINE-BY-LINE: `metrics`에 `{` 결과를 저장합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics = {
        # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `len(schedule_rows)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "scheduled_jobs": len(schedule_rows),
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `makespan_minutes` 키에 `makespan` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "makespan_minutes": makespan,
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `machine_loads` 키에 `dict(machine_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_loads": dict(machine_loads),
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `bay_process_loads` 키에 `dict(bay_loads)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bay_process_loads": dict(bay_loads),
        "machine_load_imbalance": _metric_range(machine_loads),
        "bay_process_load_imbalance": _metric_range(bay_loads),
        "machine_job_counts": dict(machine_job_counts),
        "bay_job_counts": dict(bay_job_counts),
        "machine_job_count_imbalance": _metric_range(machine_job_counts),
        "bay_job_count_imbalance": _metric_range(bay_job_counts),
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `machine_match_rate` 키에 `machine_matches / total` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "machine_match_rate": machine_matches / total,
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `bay_match_rate` 키에 `bay_matches / total` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bay_match_rate": bay_matches / total,
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `validation["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": validation["hard_passed"],
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `validation["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": validation["hard_violation_count"],
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `hard_violation_affected_job_count` 키에 `validation.get("hard_violation_affected_job_count", 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_affected_job_count": validation.get("hard_violation_affected_job_count", 0),
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `hard_violation_excess_wo_count` 키에 `validation.get("hard_violation_excess_wo_count", 0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_excess_wo_count": validation.get("hard_violation_excess_wo_count", 0),
        # LINE-BY-LINE: `build_metrics_from_rows`에서 반환/저장할 dict의 `hard_violation_length_excess_total` 키에 `validation.get("hard_violation_length_excess_total", 0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_length_excess_total": validation.get("hard_violation_length_excess_total", 0.0),
    }
    # LINE-BY-LINE: 조건 `time_summary is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if time_summary is not None:
        # LINE-BY-LINE: `metrics["time_validation"]`에 `time_summary` 결과를 저장합니다. 의미/사용: `metrics["time_validation"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        metrics["time_validation"] = time_summary
    # LINE-BY-LINE: 호출자에게 `metrics`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return metrics


# LINE-BY-LINE: `_html_document(layout: Dict[str, Any], events: List[Dict[str, Any]], schedule_rows: List[Dic...)` 함수를 정의합니다. 반환 타입: `str`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _html_document(layout: Dict[str, Any], events: List[Dict[str, Any]], schedule_rows: List[Dict[str, Any]], validation: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """브라우저에서 바로 열 수 있는 self-contained playback HTML 문자열을 만든다."""

    # LINE-BY-LINE: `embedded`에 `{` 결과를 저장합니다. 의미/사용: `embedded` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    embedded = {
        # LINE-BY-LINE: `_html_document`에서 반환/저장할 dict의 `layout` 키에 `layout` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "layout": layout,
        # LINE-BY-LINE: `_html_document`에서 반환/저장할 dict의 `events` 키에 `events` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "events": events,
        # LINE-BY-LINE: `_html_document`에서 반환/저장할 dict의 `schedule` 키에 `schedule_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "schedule": schedule_rows,
        # LINE-BY-LINE: `_html_document`에서 반환/저장할 dict의 `validation` 키에 `validation` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation": validation,
        # LINE-BY-LINE: `_html_document`에서 반환/저장할 dict의 `metrics` 키에 `metrics` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "metrics": metrics,
    }
    # LINE-BY-LINE: `payload`에 `json.dumps(embedded, ensure_ascii=False)` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
    payload = json.dumps(embedded, ensure_ascii=False)
    # LINE-BY-LINE: 호출자에게 `f"""<!doctype html>`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\" />
  <title>Cutting Factory Playback</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #f6f2ea; color: #1d2528; }}
    header {{ padding: 18px 24px; background: #12343b; color: white; }}
    main {{ padding: 18px 24px; display: grid; gap: 16px; }}
    .panel {{ background: white; border: 1px solid #d8d1c5; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,.06); }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    button, select, input {{ font-size: 14px; padding: 7px 10px; }}
    #timeline {{ width: min(900px, 90vw); }}
    .factory {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .bay {{ border: 2px solid #b58b35; border-radius: 12px; padding: 14px; background: #fffaf0; }}
    .bay h3 {{ margin: 0 0 12px; }}
    .machine {{ border: 1px solid #b8c2c7; border-radius: 10px; padding: 10px; margin: 8px 0; background: #edf4f5; min-height: 72px; }}
    .machine.active {{ background: #d7f0dc; border-color: #4f9d61; }}
    .machine .name {{ font-weight: 700; display: flex; justify-content: space-between; }}
    .job {{ margin-top: 8px; padding: 8px; border-radius: 8px; background: #246a73; color: white; }}
    .progress {{ margin-top: 8px; height: 8px; background: rgba(255,255,255,.35); border-radius: 99px; overflow: hidden; }}
    .bar {{ height: 100%; background: #f6d365; width: 0%; }}
    .bad {{ color: #b00020; font-weight: 700; }}
    .ok {{ color: #1b7f3a; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4ded2; padding: 6px 8px; text-align: left; }}
  </style>
</head>
<body>
  <header>
    <h1>Cutting Factory Playback</h1>
    <div id=\"clock\">00:00:00</div>
  </header>
  <main>
    <section class=\"panel controls\">
      <button id=\"playBtn\">Play</button>
      <button id=\"backBtn\">-1s</button>
      <button id=\"stepBtn\">+1s</button>
      <label>Speed
        <select id=\"speed\">
          <option value=\"1\">x1</option>
          <option value=\"10\">x10</option>
          <option value=\"60\" selected>x60</option>
          <option value=\"600\">x600</option>
        </select>
      </label>
      <input id=\"timeline\" type=\"range\" min=\"0\" value=\"0\" />
      <input id=\"search\" placeholder=\"W/O or job search\" />
      <span id=\"validation\"></span>
    </section>
    <section class=\"panel\">
      <h2>Factory</h2>
      <div id=\"factory\" class=\"factory\"></div>
    </section>
    <section class=\"panel\">
      <h2>Metrics</h2>
      <div id=\"metrics\"></div>
    </section>
    <section class=\"panel\">
      <h2>Current Jobs</h2>
      <table>
        <thead><tr><th>Machine</th><th>Job</th><th>Bay</th><th>Progress</th><th>Source</th></tr></thead>
        <tbody id=\"activeTable\"></tbody>
      </table>
    </section>
  </main>
  <script>
    const DATA = {payload};
    let current = 0;
    let playing = false;
    let timer = null;
    const maxTime = Math.max(...DATA.events.map(e => e.time_sec), 1);
    const timeline = document.getElementById('timeline');
    timeline.max = maxTime;

    function fmt(sec) {{
      const d = Math.floor(sec / 86400);
      const rem = sec % 86400;
      const h = String(Math.floor(rem / 3600)).padStart(2, '0');
      const m = String(Math.floor((rem % 3600) / 60)).padStart(2, '0');
      const s = String(Math.floor(rem % 60)).padStart(2, '0');
      return `day ${{d}} ${{h}}:${{m}}:${{s}}`;
    }}

    function activeOpsAt(t) {{
      return DATA.schedule.filter(row => row.start_min * 60 <= t && t < row.finish_min * 60);
    }}

    function renderFactory(activeOps) {{
      const activeByMachine = new Map(activeOps.map(row => [row.algorithm_machine_id, row]));
      const search = document.getElementById('search').value.trim();
      const root = document.getElementById('factory');
      root.innerHTML = '';
      DATA.layout.bays.forEach(bay => {{
        const bayEl = document.createElement('div');
        bayEl.className = 'bay';
        bayEl.innerHTML = `<h3>Bay ${{bay.bay_id}}</h3>`;
        bay.machines.forEach(machine => {{
          const row = activeByMachine.get(machine.machine_id);
          const machineEl = document.createElement('div');
          machineEl.className = 'machine' + (row ? ' active' : '');
          const progress = row ? Math.max(0, Math.min(100, ((current - row.start_min * 60) / ((row.finish_min - row.start_min) * 60)) * 100)) : 0;
          const hit = row && search && JSON.stringify(row).includes(search);
          machineEl.innerHTML = `
            <div class=\"name\"><span>${{machine.machine_id}}</span><span>${{machine.machine_type}}</span></div>
            ${{row ? `<div class=\"job\" style=\"outline:${{hit ? '3px solid #f6d365' : 'none'}}\">
              <div>${{row.job_id}}</div>
              <div>WO: ${{row.work_order_no || '-'}}</div>
              <div class=\"progress\"><div class=\"bar\" style=\"width:${{progress}}%\"></div></div>
            </div>` : '<div style=\"margin-top:8px;color:#607078\">idle</div>'}}
          `;
          bayEl.appendChild(machineEl);
        }});
        root.appendChild(bayEl);
      }});
    }}

    function render() {{
      current = Math.max(0, Math.min(maxTime, current));
      timeline.value = current;
      document.getElementById('clock').textContent = fmt(current);
      const active = activeOpsAt(current);
      renderFactory(active);
      const isFactoryReplay = DATA.validation.validation_type === 'actual_factory_replay_identity';
      const scope = DATA.metrics.factory_replay_scope;
      const scopeText = scope
        ? ` |
        observed_events=${{scope.observed_event_count}} |
        derived_states=${{scope.derived_state_count}} |
        missing_events=${{scope.missing_event_count}} |
        missing_events_inferred=${{scope.missing_events_are_inferred}}`
        : '';
      document.getElementById('validation').innerHTML = DATA.validation.hard_passed
        ? '<span class=\"ok\">' + (isFactoryReplay ? 'Factory replay identity passed' : 'Hard validation passed') + '</span>'
        : '<span class=\"bad\">' + (isFactoryReplay ? 'Factory replay identity failed: ' : 'Hard validation failed: ') + DATA.validation.hard_violation_count + '</span>';
      document.getElementById('metrics').innerHTML = `
        scheduled_jobs=${{DATA.metrics.scheduled_jobs}} |
        makespan_min=${{DATA.metrics.makespan_minutes.toFixed(2)}} |
        machine_match_rate=${{(DATA.metrics.machine_match_rate * 100).toFixed(1)}}% |
        bay_match_rate=${{(DATA.metrics.bay_match_rate * 100).toFixed(1)}}% |
        duration_MAE=${{DATA.metrics.time_validation.duration_mae_minutes.toFixed(2)}}min |
        duration_MAPE=${{DATA.metrics.time_validation.duration_mape_percent.toFixed(1)}}% |
        tact_MAE=${{DATA.metrics.time_validation.tact_mae_minutes.toFixed(2)}}min |
        tact_MAPE=${{DATA.metrics.time_validation.tact_mape_percent.toFixed(1)}}% |
        affected_jobs=${{DATA.metrics.hard_violation_affected_job_count}} |
        excess_WO=${{DATA.metrics.hard_violation_excess_wo_count}}${{scopeText}}
      `;
      document.getElementById('activeTable').innerHTML = active.map(row => {{
        const progress = Math.max(0, Math.min(100, ((current - row.start_min * 60) / ((row.finish_min - row.start_min) * 60)) * 100));
        return `<tr><td>${{row.algorithm_machine_id}}</td><td>${{row.job_id}}</td><td>${{row.algorithm_cut_bay}}</td><td>${{progress.toFixed(1)}}%</td><td>${{row.source_machine_id}} / ${{row.source_cut_bay}}</td></tr>`;
      }}).join('');
    }}

    function tick() {{
      const speed = Number(document.getElementById('speed').value);
      current += speed;
      if (current >= maxTime) {{
        playing = false;
        clearInterval(timer);
        document.getElementById('playBtn').textContent = 'Play';
      }}
      render();
    }}

    document.getElementById('playBtn').onclick = () => {{
      playing = !playing;
      document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
      if (playing) timer = setInterval(tick, 1000);
      else clearInterval(timer);
    }};
    document.getElementById('stepBtn').onclick = () => {{ current += 1; render(); }};
    document.getElementById('backBtn').onclick = () => {{ current -= 1; render(); }};
    timeline.oninput = () => {{ current = Number(timeline.value); render(); }};
    document.getElementById('search').oninput = render;
    render();
  </script>
</body>
</html>"""


# LINE-BY-LINE: `_factory_simulator_document` 함수를 정의합니다. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def _factory_simulator_document(
    # LINE-BY-LINE: `layout`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `layout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    layout: Dict[str, Any],
    # LINE-BY-LINE: `events`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events: List[Dict[str, Any]],
    # LINE-BY-LINE: `schedule_rows`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows: List[Dict[str, Any]],
    # LINE-BY-LINE: `validation`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation: Dict[str, Any],
    # LINE-BY-LINE: `metrics`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics: Dict[str, Any],
# LINE-BY-LINE: `) -> str:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> str:
    """초보자가 공장 내부 흐름을 확인할 수 있는 공장형 playback HTML."""

    # LINE-BY-LINE: `time_axis`에 `_build_playback_time_axis(events, schedule_rows, validation)` 결과를 저장합니다. 의미/사용: `time_axis` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    time_axis = _build_playback_time_axis(events, schedule_rows, validation)
    # LINE-BY-LINE: `embedded`에 `{` 결과를 저장합니다. 의미/사용: `embedded` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    embedded = {
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `layout` 키에 `layout` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "layout": layout,
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `events` 키에 `events` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "events": events,
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `schedule` 키에 `schedule_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "schedule": schedule_rows,
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `validation` 키에 `validation` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation": validation,
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `metrics` 키에 `metrics` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "metrics": metrics,
        # LINE-BY-LINE: `_factory_simulator_document`에서 반환/저장할 dict의 `time_axis` 키에 `time_axis` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "time_axis": time_axis,
    }
    # LINE-BY-LINE: `payload`에 `json.dumps(embedded, ensure_ascii=False)` 결과를 저장합니다. 의미/사용: `payload`는 event별 추가 정보 dict입니다. 예: processing_minutes, action_id.
    payload = json.dumps(embedded, ensure_ascii=False)
    # LINE-BY-LINE: `html`에 `"""<!doctype html>` 결과를 저장합니다. 의미/사용: `html` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    html = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Cutting Factory Simulator</title>
  <style>
    :root {
      --ink: #172126;
      --muted: #66767d;
      --line: #ccd7da;
      --floor: #eef1e8;
      --bay22: #e6f0d8;
      --bay23: #dcecf2;
      --machine: #f8fbfb;
      --active: #cdebd7;
      --warn: #fff0c7;
      --bad: #b42318;
      --ok: #167044;
      --steel: #2f6470;
      --token: #173f4f;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Georgia, "Times New Roman", serif; background: #f6f2ea; color: var(--ink); }
    header { padding: 18px 24px; background: linear-gradient(135deg, #102f38, #1d4d57); color: white; }
    header h1 { margin: 0 0 6px; font-size: 26px; }
    header .sub { color: #cfe2e6; font-size: 14px; }
    main { padding: 18px 24px; display: grid; gap: 16px; }
    .panel { background: rgba(255,255,255,.94); border: 1px solid #d8d1c5; border-radius: 14px; padding: 16px; box-shadow: 0 2px 12px rgba(0,0,0,.07); }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    button, select, input { font-size: 14px; padding: 8px 10px; border: 1px solid #bbc6c9; border-radius: 9px; background: white; }
    button { cursor: pointer; background: #173f4f; color: white; border-color: #173f4f; }
    #timeline { width: min(980px, 90vw); accent-color: #173f4f; }
    .dashboard { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; }
    .stat { border: 1px solid #d7dedf; border-radius: 12px; padding: 12px; background: #fbfcfa; }
    .stat .label { color: var(--muted); font-size: 12px; }
    .stat .value { font-size: 22px; font-weight: 700; margin-top: 3px; }
    .plant { display: grid; grid-template-columns: 170px 1fr 170px; gap: 14px; align-items: stretch; }
    .yard { border: 2px dashed #b8c2c7; border-radius: 16px; padding: 12px; background: #fbfaf5; min-height: 420px; }
    .yard h3, .floor h3 { margin: 0 0 10px; font-size: 16px; }
    .floor { position: relative; border: 2px solid #9eaaa8; border-radius: 18px; padding: 14px; background:
      linear-gradient(90deg, rgba(255,255,255,.4) 1px, transparent 1px),
      linear-gradient(0deg, rgba(255,255,255,.4) 1px, transparent 1px),
      var(--floor); background-size: 32px 32px; min-height: 420px; overflow: hidden; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; font-size: 12px; color: var(--muted); }
    .legend span { display: inline-flex; gap: 5px; align-items: center; }
    .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: #aaa; }
    .dot.active { background: var(--ok); }
    .dot.warn { background: #d99612; }
    .dot.bad { background: var(--bad); }
    .flow-note { margin: 0 0 10px; padding: 9px 11px; border-radius: 10px; background: #fff7dc; color: #6f4d00; font-size: 13px; border: 1px solid #efd58b; }
    .flow-map-wrap { border: 1px solid #b7c2c4; border-radius: 16px; background: rgba(255,255,255,.72); padding: 8px; margin-bottom: 14px; overflow-x: auto; }
    #flowMap { width: 100%; min-width: 760px; height: 360px; display: block; }
    .svg-zone { fill: #fbfaf5; stroke: #8ca0a5; stroke-width: 2; }
    .svg-bay22 { fill: #dcecc9; stroke: #7d9a61; stroke-width: 2; }
    .svg-bay23 { fill: #d4e8ef; stroke: #6c94a3; stroke-width: 2; }
    .svg-machine { fill: #f7fbfb; stroke: #70848a; stroke-width: 2; }
    .svg-machine.active { fill: #cdebd7; stroke: #237447; }
    .svg-machine.violation { fill: #ffe3de; stroke: #b42318; stroke-width: 3; }
    .svg-token { fill: var(--token); stroke: white; stroke-width: 2; }
    .svg-token.violation { fill: #b42318; }
    .svg-label { font-family: Georgia, "Times New Roman", serif; fill: #172126; font-size: 13px; font-weight: 700; }
    .svg-small { font-family: Georgia, "Times New Roman", serif; fill: #56666b; font-size: 11px; }
    .svg-arrow { stroke: #789098; stroke-width: 3; fill: none; marker-end: url(#arrowHead); opacity: .65; }
    .bay-row { border: 2px solid #94a69f; border-radius: 16px; padding: 12px; margin-bottom: 14px; position: relative; background: rgba(255,255,255,.58); }
    .bay-row.bay-22 { background: color-mix(in srgb, var(--bay22) 70%, white); }
    .bay-row.bay-23 { background: color-mix(in srgb, var(--bay23) 70%, white); }
    .bay-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
    .bay-title { font-size: 19px; font-weight: 800; }
    .lane { display: grid; grid-template-columns: repeat(auto-fit, minmax(172px, 1fr)); gap: 10px; position: relative; }
    .machine { border: 1px solid var(--line); border-radius: 14px; padding: 10px; background: var(--machine); min-height: 178px; position: relative; overflow: hidden; }
    .machine.active { background: var(--active); border-color: #6aa77d; box-shadow: inset 0 0 0 2px rgba(36,120,64,.12); }
    .machine.violation { background: #ffe9e6; border-color: var(--bad); }
    .machine .name { display: flex; justify-content: space-between; font-weight: 800; font-size: 14px; }
    .machine .meta { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .machine .capacity { position: absolute; top: 38px; right: 10px; font-size: 12px; padding: 3px 7px; border-radius: 999px; background: #e8eef0; }
    .machine.violation .capacity { background: var(--bad); color: white; }
    .slots { margin-top: 16px; display: grid; gap: 6px; }
    .slot { min-height: 38px; border: 1px dashed #aab6ba; border-radius: 10px; background: rgba(255,255,255,.65); padding: 5px; }
    .job-token { border-radius: 9px; padding: 6px 7px; background: var(--token); color: white; font-size: 12px; line-height: 1.25; box-shadow: 0 2px 5px rgba(0,0,0,.15); }
    .job-token.search-hit { outline: 3px solid #f6c343; }
    .progress { height: 7px; border-radius: 999px; background: rgba(255,255,255,.35); overflow: hidden; margin-top: 5px; }
    .bar { height: 100%; background: #f6c343; width: 0%; }
    .queue-list { display: grid; gap: 6px; max-height: 340px; overflow: auto; }
    .queue-item { font-size: 12px; padding: 7px; border-radius: 8px; background: #eef3f4; border-left: 4px solid #8aa0a7; }
    .queue-item.done { opacity: .62; border-left-color: var(--ok); }
    .queue-item.active { background: #d8efe2; border-left-color: var(--ok); }
    .status-line { display: grid; gap: 6px; font-size: 14px; }
    .ok { color: var(--ok); font-weight: 800; }
    .bad { color: var(--bad); font-weight: 800; }
    .warn { color: #9b6400; font-weight: 800; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #e4ded2; padding: 7px 8px; text-align: left; vertical-align: top; }
    @media (max-width: 980px) {
      .plant { grid-template-columns: 1fr; }
      .dashboard { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Cutting Factory Simulator</h1>
    <div class="sub">Bay 22/23 PLS 설비 기준. 현재 데이터에서 관측된 것은 장비, Bay, 절단 시작, 절단 종료입니다.</div>
  </header>
  <main>
    <section class="panel controls">
      <button id="playBtn">Play</button>
      <button id="backSecBtn">-1s</button>
      <button id="stepSecBtn">+1s</button>
      <button id="backMinBtn">-1m</button>
      <button id="stepMinBtn">+1m</button>
      <label>Speed
        <select id="speed">
          <option value="1">x1</option>
          <option value="10">x10</option>
          <option value="60" selected>x60</option>
          <option value="600">x600</option>
          <option value="3600">x3600</option>
        </select>
      </label>
      <input id="timeline" type="range" min="0" value="0" />
      <input id="search" placeholder="W/O, Job, Machine 검색" />
    </section>
    <section class="panel">
      <div class="dashboard" id="dashboard"></div>
    </section>
    <section class="panel">
      <div class="status-line" id="statusLine"></div>
    </section>
    <section class="plant">
      <aside class="yard">
        <h3>Input / Not Started</h3>
        <div class="queue-list" id="inputQueue"></div>
      </aside>
      <section class="floor">
        <h3>Factory Floor</h3>
        <div class="legend">
          <span><i class="dot active"></i>processing</span>
          <span><i class="dot warn"></i>observed replay audit issue</span>
          <span><i class="dot bad"></i>hard capacity violation at current time</span>
        </div>
        <div class="flow-note">
          이 지도는 Gantt가 아니라 현재 시각의 공장 상태입니다.
          데이터에 이동/대기/크레인 시간이 없으므로 W/O는 절단 시작 전 Input, 절단 중 해당 PLS, 종료 후 Output으로만 표시합니다.
        </div>
        <div class="flow-map-wrap">
          <svg id="flowMap" viewBox="0 0 1000 420" role="img" aria-label="factory flow map"></svg>
        </div>
        <div id="factory"></div>
      </section>
      <aside class="yard">
        <h3>Output / Finished</h3>
        <div class="queue-list" id="outputQueue"></div>
      </aside>
    </section>
    <section class="panel">
      <h2>현재 작업</h2>
      <table>
        <thead><tr><th>Machine</th><th>Active W/O</th><th>Bay</th><th>Progress</th><th>Scheduled / Actual Window</th></tr></thead>
        <tbody id="activeTable"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>데이터 범위</h2>
      <div id="scopeBox"></div>
    </section>
  </main>
  <script>
    const DATA = __PAYLOAD__;
    let playing = false;
    let timer = null;
    const maxFromEvents = DATA.events.length ? Math.max(...DATA.events.map(e => Number(e.time_sec || 0))) : 0;
    const maxFromSchedule = DATA.schedule.length ? Math.max(...DATA.schedule.map(r => Number(r.finish_min || 0) * 60)) : 1;
    const minFromEvents = DATA.events.length ? Math.min(...DATA.events.map(e => Number(e.time_sec || 0))) : 0;
    const minFromSchedule = DATA.schedule.length ? Math.min(...DATA.schedule.map(r => Number(r.start_min || 0) * 60)) : 0;
    const axis = DATA.time_axis || {};
    const minTime = Number.isFinite(Number(axis.min_time_sec))
      ? Number(axis.min_time_sec)
      : Math.min(minFromEvents, minFromSchedule, 0);
    const maxTime = Number.isFinite(Number(axis.max_time_sec))
      ? Number(axis.max_time_sec)
      : Math.max(maxFromEvents, maxFromSchedule, minTime + 1);
    let current = Number.isFinite(Number(axis.initial_time_sec)) ? Number(axis.initial_time_sec) : minTime;
    const maxWoCount = DATA.metrics?.concurrent_capacity_audit?.max_wo_count || DATA.metrics?.concurrent_capacity_audit?.max_wo_count === 0
      ? DATA.metrics.concurrent_capacity_audit.max_wo_count
      : 3;
    const timeline = document.getElementById('timeline');
    timeline.min = Math.floor(minTime);
    timeline.max = Math.ceil(maxTime);
    timeline.value = Math.floor(current);

    function num(value, fallback = 0) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }
    function pad2(value) {
      return String(value).padStart(2, '0');
    }
    function compactDateTimeLabel(value) {
      const text = String(value || '').trim().replace(/\.0$/, '');
      if (/^\d{12}$/.test(text)) {
        return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)} ${text.slice(8, 10)}:${text.slice(10, 12)}:00`;
      }
      if (/^\d{8}$/.test(text)) {
        return `${text.slice(0, 4)}-${text.slice(4, 6)}-${text.slice(6, 8)} 00:00:00`;
      }
      return text || '-';
    }
    function fmt(sec) {
      sec = Math.floor(sec);
      if (axis.mode === 'actual_datetime' && axis.display_base_datetime) {
        const base = new Date(axis.display_base_datetime);
        if (!Number.isNaN(base.getTime())) {
          const shifted = new Date(base.getTime() + (sec - num(axis.display_base_time_sec)) * 1000);
          return `${shifted.getFullYear()}-${pad2(shifted.getMonth() + 1)}-${pad2(shifted.getDate())} ${pad2(shifted.getHours())}:${pad2(shifted.getMinutes())}:${pad2(shifted.getSeconds())}`;
        }
      }
      sec = Math.max(0, sec - minTime);
      const d = Math.floor(sec / 86400);
      const rem = sec % 86400;
      const h = pad2(Math.floor(rem / 3600));
      const m = pad2(Math.floor((rem % 3600) / 60));
      const s = pad2(rem % 60);
      return `day ${d} ${h}:${m}:${s}`;
    }
    function isActualReplay() {
      return DATA.validation?.validation_type === 'actual_factory_replay_identity';
    }
    function scheduledTimeLabel(row) {
      return `${fmt(num(row.start_min) * 60)} ~ ${fmt(num(row.finish_min) * 60)}`;
    }
    function actualTimeLabel(row) {
      if (row.actual_start_datetime || row.actual_end_datetime) {
        return `${compactDateTimeLabel(row.actual_start_datetime)} ~ ${compactDateTimeLabel(row.actual_end_datetime)}`;
      }
      return `${num(row.start_min).toFixed(1)} ~ ${num(row.finish_min).toFixed(1)} min`;
    }
    function compareTimeLabel(row) {
      if (isActualReplay()) {
        return `ACT ${actualTimeLabel(row)}`;
      }
      return `SCH ${scheduledTimeLabel(row)} | ACT ${actualTimeLabel(row)}`;
    }
    function progress(row) {
      const start = num(row.start_min) * 60;
      const finish = num(row.finish_min) * 60;
      if (finish <= start) return current >= finish ? 100 : 0;
      return Math.max(0, Math.min(100, ((current - start) / (finish - start)) * 100));
    }
    function activeOpsAt(t) {
      return DATA.schedule.filter(row => num(row.start_min) * 60 <= t && t < num(row.finish_min) * 60);
    }
    function rowsByMachine(rows) {
      const map = new Map();
      rows.forEach(row => {
        const key = String(row.algorithm_machine_id || '');
        if (!map.has(key)) map.set(key, []);
        map.get(key).push(row);
      });
      return map;
    }
    function machinePositions() {
      const positions = new Map();
      DATA.layout.bays.forEach((bay, bayIndex) => {
        const bayId = String(bay.bay_id || 'unknown');
        const machines = [...bay.machines].sort((a, b) => String(a.machine_id).localeCompare(String(b.machine_id)));
        const y = bayIndex === 0 ? 104 : 250;
        const gap = 122;
        machines.forEach((machine, index) => {
          positions.set(String(machine.machine_id), {
            bayId,
            x: 260 + index * gap,
            y,
            w: 104,
            h: 72,
            machine,
          });
        });
      });
      return positions;
    }
    function svgEl(tag, attrs = {}) {
      const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
      return element;
    }
    function appendText(root, text, x, y, className = 'svg-label') {
      const node = svgEl('text', { x, y, class: className });
      node.textContent = text;
      root.appendChild(node);
      return node;
    }
    function renderFlowMap(activeOps) {
      const svg = document.getElementById('flowMap');
      svg.innerHTML = '';
      const defs = svgEl('defs');
      const marker = svgEl('marker', {
        id: 'arrowHead',
        markerWidth: '8',
        markerHeight: '8',
        refX: '6',
        refY: '3',
        orient: 'auto',
      });
      marker.appendChild(svgEl('path', { d: 'M0,0 L0,6 L7,3 z', fill: '#789098' }));
      defs.appendChild(marker);
      svg.appendChild(defs);

      svg.appendChild(svgEl('rect', { x: 20, y: 42, width: 150, height: 318, rx: 18, class: 'svg-zone' }));
      appendText(svg, 'INPUT', 62, 76);
      appendText(svg, 'not started', 50, 96, 'svg-small');
      svg.appendChild(svgEl('rect', { x: 830, y: 42, width: 150, height: 318, rx: 18, class: 'svg-zone' }));
      appendText(svg, 'OUTPUT', 870, 76);
      appendText(svg, 'finished', 876, 96, 'svg-small');
      svg.appendChild(svgEl('path', { d: 'M175,200 C215,200 220,142 252,142', class: 'svg-arrow' }));
      svg.appendChild(svgEl('path', { d: 'M752,142 C790,142 790,200 825,200', class: 'svg-arrow' }));

      const bayDefs = [
        { bayId: '22', y: 72, labelY: 95, cls: 'svg-bay22' },
        { bayId: '23', y: 218, labelY: 241, cls: 'svg-bay23' },
      ];
      bayDefs.forEach(def => {
        svg.appendChild(svgEl('rect', { x: 220, y: def.y, width: 580, height: 118, rx: 18, class: def.cls }));
        appendText(svg, `Bay ${def.bayId}`, 236, def.labelY);
      });

      const positions = machinePositions();
      const activeByMachine = rowsByMachine(activeOps);
      positions.forEach((pos, machineId) => {
        const activeRows = activeByMachine.get(machineId) || [];
        const violation = maxWoCount !== null && maxWoCount !== undefined && activeRows.length > Number(maxWoCount);
        const rect = svgEl('rect', {
          x: pos.x,
          y: pos.y,
          width: pos.w,
          height: pos.h,
          rx: 12,
          class: `svg-machine ${activeRows.length ? 'active' : ''} ${violation ? 'violation' : ''}`,
        });
        svg.appendChild(rect);
        appendText(svg, machineId, pos.x + 12, pos.y + 21);
        appendText(svg, `${activeRows.length}/${maxWoCount ?? '-'} W/O`, pos.x + 12, pos.y + 39, violation ? 'svg-label' : 'svg-small');
        if (!activeRows.length) {
          appendText(svg, 'idle', pos.x + 12, pos.y + 58, 'svg-small');
        }
        activeRows.forEach((row, index) => {
          const cx = pos.x + 22 + (index % 3) * 28;
          const cy = pos.y + 54 + Math.floor(index / 3) * 20;
          const tokenNode = svgEl('circle', {
            cx,
            cy,
            r: 10,
            class: `svg-token ${violation ? 'violation' : ''}`,
          });
          tokenNode.appendChild(svgEl('title')).textContent = `${row.work_order_no || row.job_id} | ${compareTimeLabel(row)}`;
          svg.appendChild(tokenNode);
        });
      });

      const notStarted = DATA.schedule.filter(row => num(row.start_min) * 60 > current).length;
      const finished = DATA.schedule.filter(row => num(row.finish_min) * 60 <= current).length;
      appendText(svg, String(notStarted), 78, 190);
      appendText(svg, 'W/O waiting', 50, 214, 'svg-small');
      appendText(svg, String(finished), 888, 190);
      appendText(svg, 'W/O done', 874, 214, 'svg-small');
      if (activeOps.length) {
        appendText(svg, `processing ${activeOps.length}`, 446, 392);
      } else {
        appendText(svg, 'no active processing at current time', 392, 392, 'svg-small');
      }
    }
    function queueRows(t, mode) {
      const rows = DATA.schedule.filter(row => {
        const start = num(row.start_min) * 60;
        const finish = num(row.finish_min) * 60;
        if (mode === 'input') return start > t;
        if (mode === 'output') return finish <= t;
        return start <= t && t < finish;
      });
      return rows.sort((a, b) => mode === 'output'
        ? num(b.finish_min) - num(a.finish_min)
        : num(a.start_min) - num(b.start_min)
      ).slice(0, 28);
    }
    function token(row, hit) {
      const p = progress(row);
      const timeLines = isActualReplay()
        ? `<div>ACT ${actualTimeLabel(row)}</div>`
        : `<div>SCH ${scheduledTimeLabel(row)}</div><div>ACT ${actualTimeLabel(row)}</div>`;
      return `<div class="job-token ${hit ? 'search-hit' : ''}">
        <div><b>${row.work_order_no || row.job_id}</b></div>
        <div>${row.project_no || '-'} / ${row.block_no || '-'}</div>
        <div>${num(row.plate_length).toLocaleString()}mm, THK ${row.thickness ?? '-'}</div>
        ${timeLines}
        <div class="progress"><div class="bar" style="width:${p}%"></div></div>
      </div>`;
    }
    function renderQueue(id, rows, mode) {
      const root = document.getElementById(id);
      const search = document.getElementById('search').value.trim();
      root.innerHTML = rows.map(row => {
        const hit = search && JSON.stringify(row).includes(search);
        return `<div class="queue-item ${mode}" style="${hit ? 'outline:3px solid #f6c343' : ''}">
          <b>${row.work_order_no || row.job_id}</b><br>
          ${row.algorithm_machine_id || '-'} / Bay ${row.algorithm_cut_bay || '-'}<br>
          ${compareTimeLabel(row)}
        </div>`;
      }).join('') || '<div class="queue-item">표시할 W/O 없음</div>';
    }
    function renderFactory(activeOps) {
      const activeByMachine = rowsByMachine(activeOps);
      const search = document.getElementById('search').value.trim();
      const root = document.getElementById('factory');
      root.innerHTML = '';
      DATA.layout.bays.forEach(bay => {
        const bayEl = document.createElement('div');
        const bayId = String(bay.bay_id || 'unknown');
        bayEl.className = `bay-row bay-${bayId}`;
        const bayActive = activeOps.filter(row => String(row.algorithm_cut_bay) === bayId);
        bayEl.innerHTML = `<div class="bay-head">
          <div class="bay-title">Bay ${bayId}</div>
          <div>${bayActive.length} active W/O</div>
        </div><div class="lane"></div>`;
        const lane = bayEl.querySelector('.lane');
        bay.machines.forEach(machine => {
          const activeRows = activeByMachine.get(String(machine.machine_id)) || [];
          const violation = maxWoCount !== null && maxWoCount !== undefined && activeRows.length > Number(maxWoCount);
          const machineEl = document.createElement('div');
          machineEl.className = `machine ${activeRows.length ? 'active' : ''} ${violation ? 'violation' : ''}`;
          const slotCount = Math.max(activeRows.length, Number(maxWoCount || 3), 1);
          const slots = Array.from({ length: slotCount }, (_, index) => {
            const row = activeRows[index];
            if (!row) return '<div class="slot"></div>';
            const hit = search && JSON.stringify(row).includes(search);
            return `<div class="slot">${token(row, hit)}</div>`;
          }).join('');
          machineEl.innerHTML = `<div class="name"><span>${machine.machine_id}</span><span>${machine.machine_type || 'machine'}</span></div>
            <div class="meta">Bay ${bayId} | capacity ${machine.parallel_capacity || 1} physical slot</div>
            <div class="capacity">${activeRows.length}/${maxWoCount ?? '-'} W/O</div>
            <div class="slots">${slots}</div>`;
          lane.appendChild(machineEl);
        });
        root.appendChild(bayEl);
      });
    }
    function renderDashboard(active) {
      const notStarted = DATA.schedule.filter(row => num(row.start_min) * 60 > current).length;
      const finished = DATA.schedule.filter(row => num(row.finish_min) * 60 <= current).length;
      const activeMachines = new Set(active.map(row => row.algorithm_machine_id)).size;
      const stats = [
        ['Clock', fmt(current)],
        ['Not Started', notStarted],
        ['Processing', active.length],
        ['Finished', finished],
        ['Active Machines', activeMachines],
        ['Total W/O', DATA.schedule.length],
        ['Max Time', fmt(maxTime)],
        ['Hard Violations', DATA.validation?.hard_violation_count ?? 0],
      ];
      document.getElementById('dashboard').innerHTML = stats.map(([label, value]) =>
        `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`
      ).join('');
    }
    function renderStatus() {
      const isActual = DATA.validation?.validation_type === 'actual_factory_replay_identity';
      const identityPassed = DATA.validation?.hard_passed === true;
      const constraintAudit = DATA.metrics?.constraint_audit;
      const eventLog = DATA.metrics?.event_log;
      const rows = [];
      if (isActual) {
        rows.push(identityPassed
          ? '<span class="ok">Actual replay identity passed: 원본 장비/Bay/시작/종료시간 재현 성공</span>'
          : `<span class="bad">Actual replay identity failed: ${DATA.validation?.hard_violation_count ?? 0}</span>`);
        if (constraintAudit) {
          rows.push(constraintAudit.hard_passed
            ? '<span class="ok">Planning constraint audit passed on observed actual intervals</span>'
            : `<span class="warn">Observed actual intervals have planning-audit issues: ${constraintAudit.hard_violation_count}</span>`);
        }
      } else {
        rows.push(identityPassed
          ? '<span class="ok">Generated schedule hard validation passed</span>'
          : `<span class="bad">Generated schedule hard validation failed: ${DATA.validation?.hard_violation_count ?? 0}</span>`);
      }
      if (eventLog) {
        rows.push(`event_identity=${eventLog.event_identity_passed}, event_constraint=${eventLog.event_constraint_passed}, event_count=${eventLog.event_count}`);
      }
      document.getElementById('statusLine').innerHTML = rows.join('<br>');
    }
    function renderScope() {
      const scope = DATA.metrics?.factory_replay_scope;
      const same = DATA.metrics?.same_machine_time_groups;
      const text = [];
      if (DATA.time_axis) {
        text.push(`<b>Clock axis</b>: mode=${DATA.time_axis.mode}, source=${DATA.time_axis.actual_datetime_source || 'none'}, base=${DATA.time_axis.display_base_datetime || 'relative'}, note=${DATA.time_axis.display_note || ''}`);
      }
      if (scope) {
        text.push(`<b>Observed event scope</b>: observed=${scope.observed_event_count}, derived=${scope.derived_state_count}, missing=${scope.missing_event_count}, inferred_missing=${scope.missing_events_are_inferred}`);
        text.push(scope.scope_note || '');
      }
      if (same) {
        text.push(`<b>Same timestamp groups</b>: group>=2 ${same.group_count_min_2}, group>3 ${same.group_count_over_3}, max size ${same.max_group_size}`);
        text.push(same.note || '');
      }
      document.getElementById('scopeBox').innerHTML = text.join('<br>') || 'Generated schedule playback';
    }
    function renderActiveTable(active) {
      document.getElementById('activeTable').innerHTML = active.map(row => {
        const p = progress(row);
        return `<tr><td>${row.algorithm_machine_id}</td><td>${row.work_order_no || row.job_id}</td><td>${row.algorithm_cut_bay}</td><td>${p.toFixed(1)}%</td><td>${compareTimeLabel(row)}</td></tr>`;
      }).join('') || '<tr><td colspan="5">현재 처리 중인 W/O 없음</td></tr>';
    }
    function render() {
      current = Math.max(minTime, Math.min(maxTime, current));
      timeline.value = Math.floor(current);
      const active = activeOpsAt(current);
      renderDashboard(active);
      renderStatus();
      renderFlowMap(active);
      renderFactory(active);
      renderQueue('inputQueue', queueRows(current, 'input'), 'input');
      renderQueue('outputQueue', queueRows(current, 'output'), 'done');
      renderScope();
      renderActiveTable(active);
    }
    function tick() {
      current += Number(document.getElementById('speed').value);
      if (current >= maxTime) {
        playing = false;
        clearInterval(timer);
        document.getElementById('playBtn').textContent = 'Play';
      }
      render();
    }
    document.getElementById('playBtn').onclick = () => {
      playing = !playing;
      document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
      if (playing) timer = setInterval(tick, 1000);
      else clearInterval(timer);
    };
    document.getElementById('stepSecBtn').onclick = () => { current += 1; render(); };
    document.getElementById('backSecBtn').onclick = () => { current -= 1; render(); };
    document.getElementById('stepMinBtn').onclick = () => { current += 60; render(); };
    document.getElementById('backMinBtn').onclick = () => { current -= 60; render(); };
    timeline.oninput = () => { current = Number(timeline.value); render(); };
    document.getElementById('search').oninput = render;
    render();
  </script>
</body>
</html>"""
    # LINE-BY-LINE: 호출자에게 `html.replace("__PAYLOAD__", payload)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return html.replace("__PAYLOAD__", payload)


# LINE-BY-LINE: `write_playback_artifacts(simulation, config: Dict[str, Any], trace_records: Iterable[Dict[str, Any]], ...)` 함수를 정의합니다. 반환 타입: `Dict[str, str]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def write_playback_artifacts(simulation, config: Dict[str, Any], trace_records: Iterable[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """playback에 필요한 모든 산출물을 저장한다."""

    # LINE-BY-LINE: `output_path`에 `Path(output_dir)` 결과를 저장합니다. 의미/사용: `output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path = Path(output_dir)
    # LINE-BY-LINE: `output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path.mkdir(parents=True, exist_ok=True)

    # LINE-BY-LINE: `schedule_rows`에 `build_job_schedule_rows(simulation)` 결과를 저장합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows = build_job_schedule_rows(simulation)
    # LINE-BY-LINE: `trace_rows`에 `build_action_trace_rows(trace_records)` 결과를 저장합니다. 의미/사용: `trace_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trace_rows = build_action_trace_rows(trace_records)
    # LINE-BY-LINE: `machine_summary`에 `build_machine_day_summary(schedule_rows)` 결과를 저장합니다. 의미/사용: `machine_summary` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_summary = build_machine_day_summary(schedule_rows)
    # LINE-BY-LINE: `bay_summary`에 `build_bay_day_summary(schedule_rows)` 결과를 저장합니다. 의미/사용: `bay_summary` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_summary = build_bay_day_summary(schedule_rows)
    # LINE-BY-LINE: `validation`에 `validate_schedule(simulation, config)` 결과를 저장합니다. 의미/사용: `validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation = validate_schedule(simulation, config)
    # LINE-BY-LINE: `time_validation`에 `build_time_validation(schedule_rows)` 결과를 저장합니다. 의미/사용: `time_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    time_validation = build_time_validation(schedule_rows)
    # LINE-BY-LINE: `layout`에 `build_factory_layout(simulation)` 결과를 저장합니다. 의미/사용: `layout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    layout = build_factory_layout(simulation)
    # LINE-BY-LINE: `event_log_rows`에 `simulation.export_event_log()` 결과를 저장합니다. 의미/사용: `event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_log_rows = simulation.export_event_log()
    # LINE-BY-LINE: 조건 `not event_log_rows and schedule_rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not event_log_rows and schedule_rows:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][playback_builder.write_playback_artifacts] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][playback_builder.write_playback_artifacts] "
            # LINE-BY-LINE: `f"cause`에 `empty_event_log schedule_count={len(schedule_rows)}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=empty_event_log schedule_count={len(schedule_rows)}"
        )
        # LINE-BY-LINE: `RuntimeError("generated simulation event log is empty")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise RuntimeError("generated simulation event log is empty")
    # LINE-BY-LINE: `event_identity`에 `validate_event_log_identity(event_log_rows, schedule_rows)` 결과를 저장합니다. 의미/사용: `event_identity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_identity = validate_event_log_identity(event_log_rows, schedule_rows)
    # LINE-BY-LINE: `machine_to_bay`에 `{` 결과를 저장합니다. 의미/사용: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_to_bay = {
        # LINE-BY-LINE: `machine_id`를 `str(machine.bay_id)` 타입으로 선언합니다. 의미/사용: `machine_id`는 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
        machine_id: str(machine.bay_id)
        # LINE-BY-LINE: `machine_id, machine in simulation.machines.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine_id, machine in simulation.machines.items()
        # LINE-BY-LINE: 조건 `machine.bay_id is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine.bay_id is not None
    }
    # LINE-BY-LINE: `event_constraint_validation`에 `validate_event_log_constraints(event_log_rows, config, machine_to_bay=machine_to_bay)` 결과를 저장합니다. 의미/사용: `event_constraint_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_constraint_validation = validate_event_log_constraints(event_log_rows, config, machine_to_bay=machine_to_bay)
    # LINE-BY-LINE: `assert_event_constraint_validation_matches(validation, event_constraint_validation, "generated_pl...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    assert_event_constraint_validation_matches(validation, event_constraint_validation, "generated_playback")
    # LINE-BY-LINE: `validation_failure_events`에 `build_validation_failure_event_rows(` 결과를 저장합니다. 의미/사용: `validation_failure_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_failure_events = build_validation_failure_event_rows(
        # LINE-BY-LINE: `build_validation_failure_event_rows(...)` 호출에 `event_constraint_validation` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        event_constraint_validation,
        # LINE-BY-LINE: `event_prefix`에 `"V"` 결과를 저장합니다. 의미/사용: `event_prefix` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        event_prefix="V",
        # LINE-BY-LINE: `start_index`에 `len(event_log_rows)` 결과를 저장합니다. 의미/사용: `start_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_index=len(event_log_rows),
    )
    # LINE-BY-LINE: `export_event_log_rows`에 `sorted([*event_log_rows, *validation_failure_events], key=event_sort_key)` 결과를 저장합니다. 의미/사용: `export_event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    export_event_log_rows = sorted([*event_log_rows, *validation_failure_events], key=event_sort_key)
    # LINE-BY-LINE: `events`에 `build_schedule_events_from_event_log(event_log_rows)` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events = build_schedule_events_from_event_log(event_log_rows)
    # LINE-BY-LINE: `metrics`에 `build_metrics(simulation, validation, schedule_rows, time_validation["summary"])` 결과를 저장합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics = build_metrics(simulation, validation, schedule_rows, time_validation["summary"])
    # LINE-BY-LINE: `metrics["event_log"]`에 `{` 결과를 저장합니다. 의미/사용: `metrics["event_log"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["event_log"] = {
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `source` 키에 `"generated"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "source": "generated",
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_count` 키에 `len(export_event_log_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_count": len(export_event_log_rows),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `factory_event_count` 키에 `len(event_log_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "factory_event_count": len(event_log_rows),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `validation_failure_event_count` 키에 `len(validation_failure_events)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_failure_event_count": len(validation_failure_events),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `playback_event_count` 키에 `len(events)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_event_count": len(events),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `event_identity["event_identity_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_identity_passed": event_identity["event_identity_passed"],
        # LINE-BY-LINE: validation 결과의 `event_identity_error_count` 항목에 `event_identity["identity_error_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "event_identity_error_count": event_identity["identity_error_count"],
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_constraint_passed` 키에 `event_constraint_validation["hard_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_passed": event_constraint_validation["hard_passed"],
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_constraint_violation_count` 키에 `event_constraint_validation["hard_violation_count"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_violation_count": event_constraint_validation["hard_violation_count"],
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_type_counts` 키에 `event_identity["event_type_counts"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_type_counts": event_identity["event_type_counts"],
    }

    # LINE-BY-LINE: `_write_csv(schedule_rows, output_path / "job_schedule.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(schedule_rows, output_path / "job_schedule.csv")
    # LINE-BY-LINE: `_write_csv(schedule_rows, output_path / "actual_vs_schedule.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(schedule_rows, output_path / "actual_vs_schedule.csv")
    # LINE-BY-LINE: `_write_csv(machine_summary, output_path / "machine_day_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(machine_summary, output_path / "machine_day_summary.csv")
    # LINE-BY-LINE: `_write_csv(bay_summary, output_path / "bay_day_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(bay_summary, output_path / "bay_day_summary.csv")
    # LINE-BY-LINE: `_write_csv(trace_rows, output_path / "action_trace.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(trace_rows, output_path / "action_trace.csv")
    # LINE-BY-LINE: `_write_csv(time_validation["rows"], output_path / "time_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(time_validation["rows"], output_path / "time_validation.csv")
    # LINE-BY-LINE: `validation_rows`에 `validation["violations"] or [` 결과를 저장합니다. 의미/사용: `validation_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_rows = validation["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `validation["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_passed": validation["hard_passed"],
            # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `validation["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_violation_count": validation["hard_violation_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"no hard violations"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "no hard violations",
        }
    ]
    # LINE-BY-LINE: `_write_csv(validation_rows, output_path / "validation_result.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(validation_rows, output_path / "validation_result.csv")
    # LINE-BY-LINE: `event_identity_rows`에 `event_identity["violations"] or [` 결과를 저장합니다. 의미/사용: `event_identity_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_identity_rows = event_identity["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `event_identity["event_identity_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_identity_passed": event_identity["event_identity_passed"],
            # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `event_identity["identity_error_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "identity_error_count": event_identity["identity_error_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"generated event log identity passed"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "generated event log identity passed",
        }
    ]
    # LINE-BY-LINE: `_write_csv(event_identity_rows, output_path / "event_log_identity_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(event_identity_rows, output_path / "event_log_identity_validation.csv")
    # LINE-BY-LINE: `_write_json(validation, output_path / "validation_result.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(validation, output_path / "validation_result.json")
    # LINE-BY-LINE: `_write_json(event_identity, output_path / "event_log_identity_validation.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(event_identity, output_path / "event_log_identity_validation.json")
    # LINE-BY-LINE: `event_constraint_rows`에 `event_constraint_validation["violations"] or [` 결과를 저장합니다. 의미/사용: `event_constraint_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_constraint_rows = event_constraint_validation["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `event_constraint_validation["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_passed": event_constraint_validation["hard_passed"],
            # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `event_constraint_validation["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_violation_count": event_constraint_validation["hard_violation_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"generated event log constraint validation passed"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "generated event log constraint validation passed",
        }
    ]
    # LINE-BY-LINE: `_write_csv(event_constraint_rows, output_path / "event_log_constraint_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(event_constraint_rows, output_path / "event_log_constraint_validation.csv")
    # LINE-BY-LINE: `_write_json(event_constraint_validation, output_path / "event_log_constraint_validation.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(event_constraint_validation, output_path / "event_log_constraint_validation.json")
    # LINE-BY-LINE: `_write_json(layout, output_path / "factory_layout.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(layout, output_path / "factory_layout.json")
    # LINE-BY-LINE: `_write_json(export_event_log_rows, output_path / "event_log.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(export_event_log_rows, output_path / "event_log.json")
    # LINE-BY-LINE: `_write_event_log_csv(export_event_log_rows, output_path / "event_log.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_event_log_csv(export_event_log_rows, output_path / "event_log.csv")
    # LINE-BY-LINE: `_write_json(events, output_path / "schedule_events.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(events, output_path / "schedule_events.json")
    # LINE-BY-LINE: `_write_json(time_validation["summary"], output_path / "time_validation_summary.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(time_validation["summary"], output_path / "time_validation_summary.json")
    # LINE-BY-LINE: `_write_json(metrics, output_path / "metrics.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(metrics, output_path / "metrics.json")
    # LINE-BY-LINE: `(output_path / "playback.html").write_text(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    (output_path / "playback.html").write_text(
        # LINE-BY-LINE: `write_text(...)` 호출에 `_factory_simulator_document(layout, events, schedule_rows, validation, metrics)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        _factory_simulator_document(layout, events, schedule_rows, validation, metrics),
        # LINE-BY-LINE: `encoding`에 `"utf-8"` 결과를 저장합니다. 의미/사용: `encoding` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        encoding="utf-8",
    )

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: 딕셔너리 키 `output_dir`에는 `str(output_path)` 값을 넣습니다. 의미: 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        "output_dir": str(output_path),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `playback_html` 키에 `str(output_path / "playback.html")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_html": str(output_path / "playback.html"),
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `str(validation["hard_passed"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": str(validation["hard_passed"]),
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `str(validation["hard_violation_count"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": str(validation["hard_violation_count"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `hard_violation_affected_job_count` 키에 `str(validation.get("hard_violation_affected_job_count", 0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_affected_job_count": str(validation.get("hard_violation_affected_job_count", 0)),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `hard_violation_excess_wo_count` 키에 `str(validation.get("hard_violation_excess_wo_count", 0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_excess_wo_count": str(validation.get("hard_violation_excess_wo_count", 0)),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `hard_violation_length_excess_total` 키에 `str(validation.get("hard_violation_length_excess_total", 0.0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_length_excess_total": str(validation.get("hard_violation_length_excess_total", 0.0)),
        # LINE-BY-LINE: 실행 summary의 `event_log_count` 항목에 `str(len(export_event_log_rows))`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "event_log_count": str(len(export_event_log_rows)),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `playback_event_count` 키에 `str(len(events))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_event_count": str(len(events)),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `str(event_identity["event_identity_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_identity_passed": str(event_identity["event_identity_passed"]),
        # LINE-BY-LINE: validation 결과의 `event_identity_error_count` 항목에 `str(event_identity["identity_error_count"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "event_identity_error_count": str(event_identity["identity_error_count"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_constraint_passed` 키에 `str(event_constraint_validation["hard_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_passed": str(event_constraint_validation["hard_passed"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `event_constraint_violation_count` 키에 `str(event_constraint_validation["hard_violation_count"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_violation_count": str(event_constraint_validation["hard_violation_count"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `duration_mae_minutes` 키에 `str(time_validation["summary"]["duration_mae_minutes"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mae_minutes": str(time_validation["summary"]["duration_mae_minutes"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `duration_mape_percent` 키에 `str(time_validation["summary"]["duration_mape_percent"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mape_percent": str(time_validation["summary"]["duration_mape_percent"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `tact_mae_minutes` 키에 `str(time_validation["summary"]["tact_mae_minutes"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mae_minutes": str(time_validation["summary"]["tact_mae_minutes"]),
        # LINE-BY-LINE: `write_playback_artifacts`에서 반환/저장할 dict의 `tact_mape_percent` 키에 `str(time_validation["summary"]["tact_mape_percent"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mape_percent": str(time_validation["summary"]["tact_mape_percent"]),
    }


# LINE-BY-LINE: `write_actual_replay_artifacts(scenario: Dict[str, Any], config: Dict[str, Any], output_dir: str)` 함수를 정의합니다. 반환 타입: `Dict[str, str]`. 사용: CSV/JSON/HTML playback과 validation 산출물을 만들 때 사용됩니다.
def write_actual_replay_artifacts(scenario: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """실적 데이터를 그대로 재생하는 산출물을 저장한다."""

    # LINE-BY-LINE: `output_path`에 `Path(output_dir)` 결과를 저장합니다. 의미/사용: `output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path = Path(output_dir)
    # LINE-BY-LINE: `output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path.mkdir(parents=True, exist_ok=True)

    # LINE-BY-LINE: `schedule_rows`에 `build_actual_replay_rows(scenario, config)` 결과를 저장합니다. 의미/사용: `schedule_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    schedule_rows = build_actual_replay_rows(scenario, config)
    # LINE-BY-LINE: `machine_summary`에 `build_machine_day_summary(schedule_rows)` 결과를 저장합니다. 의미/사용: `machine_summary` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_summary = build_machine_day_summary(schedule_rows)
    # LINE-BY-LINE: `bay_summary`에 `build_bay_day_summary(schedule_rows)` 결과를 저장합니다. 의미/사용: `bay_summary` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_summary = build_bay_day_summary(schedule_rows)
    # LINE-BY-LINE: `machine_to_bay`에 `{` 결과를 저장합니다. 의미/사용: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_to_bay = {
        # LINE-BY-LINE: `str(machine["machine_id"]): str(machine.get("bay_id"))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        str(machine["machine_id"]): str(machine.get("bay_id"))
        # LINE-BY-LINE: `machine in scenario.get("machines", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in scenario.get("machines", [])
        # LINE-BY-LINE: 조건 `machine.get("bay_id") is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if machine.get("bay_id") is not None
    }
    # LINE-BY-LINE: `replay_validation`에 `build_actual_factory_identity_validation(schedule_rows)` 결과를 저장합니다. 의미/사용: `replay_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    replay_validation = build_actual_factory_identity_validation(schedule_rows)
    # LINE-BY-LINE: `constraint_audit`에 `validate_schedule_rows(schedule_rows, config, machine_to_bay=machine_to_bay)` 결과를 저장합니다. 의미/사용: `constraint_audit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    constraint_audit = validate_schedule_rows(schedule_rows, config, machine_to_bay=machine_to_bay)
    # LINE-BY-LINE: `time_validation`에 `build_time_validation(schedule_rows)` 결과를 저장합니다. 의미/사용: `time_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    time_validation = build_time_validation(schedule_rows)
    # LINE-BY-LINE: `layout`에 `build_factory_layout_from_scenario(scenario)` 결과를 저장합니다. 의미/사용: `layout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    layout = build_factory_layout_from_scenario(scenario)
    # LINE-BY-LINE: `actual_event_log_rows`에 `build_actual_event_log_rows(schedule_rows)` 결과를 저장합니다. 의미/사용: `actual_event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_event_log_rows = build_actual_event_log_rows(schedule_rows)
    # LINE-BY-LINE: `event_identity`에 `validate_event_log_identity(actual_event_log_rows, schedule_rows)` 결과를 저장합니다. 의미/사용: `event_identity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_identity = validate_event_log_identity(actual_event_log_rows, schedule_rows)
    # LINE-BY-LINE: `event_constraint_validation`에 `validate_event_log_constraints(` 결과를 저장합니다. 의미/사용: `event_constraint_validation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_constraint_validation = validate_event_log_constraints(
        # LINE-BY-LINE: `validate_event_log_constraints(...)` 호출에 `actual_event_log_rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        actual_event_log_rows,
        # LINE-BY-LINE: `validate_event_log_constraints(...)` 호출에 `config` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        config,
        # LINE-BY-LINE: `machine_to_bay`에 `machine_to_bay` 결과를 저장합니다. 의미/사용: `machine_to_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_to_bay=machine_to_bay,
    )
    # LINE-BY-LINE: `assert_event_constraint_validation_matches(constraint_audit, event_constraint_validation, "actual...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    assert_event_constraint_validation_matches(constraint_audit, event_constraint_validation, "actual_replay")
    # LINE-BY-LINE: `validation_failure_events`에 `build_validation_failure_event_rows(` 결과를 저장합니다. 의미/사용: `validation_failure_events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_failure_events = build_validation_failure_event_rows(
        # LINE-BY-LINE: `build_validation_failure_event_rows(...)` 호출에 `event_constraint_validation` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        event_constraint_validation,
        # LINE-BY-LINE: `event_prefix`에 `"AV"` 결과를 저장합니다. 의미/사용: `event_prefix` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        event_prefix="AV",
        # LINE-BY-LINE: `start_index`에 `len(actual_event_log_rows)` 결과를 저장합니다. 의미/사용: `start_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_index=len(actual_event_log_rows),
    )
    # LINE-BY-LINE: `export_actual_event_log_rows`에 `sorted([*actual_event_log_rows, *validation_failure_events], key=event_sort_key)` 결과를 저장합니다. 의미/사용: `export_actual_event_log_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    export_actual_event_log_rows = sorted([*actual_event_log_rows, *validation_failure_events], key=event_sort_key)
    # LINE-BY-LINE: `events`에 `build_schedule_events_from_event_log(actual_event_log_rows)` 결과를 저장합니다. 의미/사용: `events` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    events = build_schedule_events_from_event_log(actual_event_log_rows)
    # LINE-BY-LINE: `machine_intervals`에 `build_machine_state_intervals(schedule_rows)` 결과를 저장합니다. 의미/사용: `machine_intervals` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_intervals = build_machine_state_intervals(schedule_rows)
    # LINE-BY-LINE: `bay_intervals`에 `build_bay_state_intervals(schedule_rows)` 결과를 저장합니다. 의미/사용: `bay_intervals` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_intervals = build_bay_state_intervals(schedule_rows)
    # LINE-BY-LINE: `lifecycle_rows`에 `build_actual_job_lifecycle_rows(schedule_rows)` 결과를 저장합니다. 의미/사용: `lifecycle_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    lifecycle_rows = build_actual_job_lifecycle_rows(schedule_rows)
    # LINE-BY-LINE: `missing_event_requirements`에 `build_missing_event_requirements()` 결과를 저장합니다. 의미/사용: `missing_event_requirements` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    missing_event_requirements = build_missing_event_requirements()
    # LINE-BY-LINE: `concurrent_capacity_audit`에 `build_concurrent_capacity_audit(schedule_rows, config)` 결과를 저장합니다. 의미/사용: `concurrent_capacity_audit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    concurrent_capacity_audit = build_concurrent_capacity_audit(schedule_rows, config)
    # LINE-BY-LINE: `same_time_groups`에 `build_same_machine_time_groups(schedule_rows, min_group_size=2)` 결과를 저장합니다. 의미/사용: `same_time_groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    same_time_groups = build_same_machine_time_groups(schedule_rows, min_group_size=2)
    # LINE-BY-LINE: `same_time_groups_over3`에 `[row for row in same_time_groups if int(row["group_size"]) > 3]` 결과를 저장합니다. 의미/사용: `same_time_groups_over3` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    same_time_groups_over3 = [row for row in same_time_groups if int(row["group_size"]) > 3]
    # LINE-BY-LINE: `metrics`에 `build_metrics_from_rows(schedule_rows, replay_validation, time_validation["summary"])` 결과를 저장합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics = build_metrics_from_rows(schedule_rows, replay_validation, time_validation["summary"])
    # LINE-BY-LINE: `metrics["event_log"]`에 `{` 결과를 저장합니다. 의미/사용: `metrics["event_log"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["event_log"] = {
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `source` 키에 `"actual"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "source": "actual",
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_count` 키에 `len(export_actual_event_log_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_count": len(export_actual_event_log_rows),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `factory_event_count` 키에 `len(actual_event_log_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "factory_event_count": len(actual_event_log_rows),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `validation_failure_event_count` 키에 `len(validation_failure_events)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_failure_event_count": len(validation_failure_events),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `playback_event_count` 키에 `len(events)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_event_count": len(events),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `event_identity["event_identity_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_identity_passed": event_identity["event_identity_passed"],
        # LINE-BY-LINE: validation 결과의 `event_identity_error_count` 항목에 `event_identity["identity_error_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "event_identity_error_count": event_identity["identity_error_count"],
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_constraint_passed` 키에 `event_constraint_validation["hard_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_passed": event_constraint_validation["hard_passed"],
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_constraint_violation_count` 키에 `event_constraint_validation["hard_violation_count"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_violation_count": event_constraint_validation["hard_violation_count"],
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_type_counts` 키에 `event_identity["event_type_counts"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_type_counts": event_identity["event_type_counts"],
    }
    # LINE-BY-LINE: `metrics["constraint_audit"]`에 `{` 결과를 저장합니다. 의미/사용: `metrics["constraint_audit"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["constraint_audit"] = {
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `constraint_audit["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": constraint_audit["hard_passed"],
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `constraint_audit["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": constraint_audit["hard_violation_count"],
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `hard_violation_counts_by_rule` 키에 `constraint_audit.get("hard_violation_counts_by_rule", {})` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "hard_violation_counts_by_rule": constraint_audit.get("hard_violation_counts_by_rule", {}),
    }
    # LINE-BY-LINE: `metrics["factory_replay_scope"]`에 `{` 결과를 저장합니다. 의미/사용: `metrics["factory_replay_scope"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["factory_replay_scope"] = {
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `observed_event_count` 키에 `sum(1 for row in missing_event_requirements if row["status"] == "observed")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "observed_event_count": sum(1 for row in missing_event_requirements if row["status"] == "observed"),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `derived_state_count` 키에 `sum(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "derived_state_count": sum(
            # LINE-BY-LINE: `1 for row in missing_event_requirements if row["status"] == "derived_from_observed_events"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            1 for row in missing_event_requirements if row["status"] == "derived_from_observed_events"
        ),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `missing_event_count` 키에 `sum(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "missing_event_count": sum(
            # LINE-BY-LINE: `1 for row in missing_event_requirements if row["status"] == "missing_in_current_data"` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            1 for row in missing_event_requirements if row["status"] == "missing_in_current_data"
        ),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `missing_events_are_inferred` 키에 `False` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "missing_events_are_inferred": False,
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `scope_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "scope_note": (
            # LINE-BY-LINE: 문자열 값 `"Actual factory replay reproduces observed processing intervals exactly. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Actual factory replay reproduces observed processing intervals exactly. "
            # LINE-BY-LINE: 문자열 값 `"Queue, transfer, Bay IN/OUT, LM/crane, and stoppage events are unknown until source data exists. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Queue, transfer, Bay IN/OUT, LM/crane, and stoppage events are unknown until source data exists. "
            # LINE-BY-LINE: 문자열 값 `"Identical timestamp overlap is kept as data-quality audit evidence; confirmed planning capacity is batch-based."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Identical timestamp overlap is kept as data-quality audit evidence; confirmed planning capacity is batch-based."
        ),
    }
    # LINE-BY-LINE: `metrics["concurrent_capacity_audit"]`에 `concurrent_capacity_audit["summary"]` 결과를 저장합니다. 의미/사용: `metrics["concurrent_capacity_audit"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["concurrent_capacity_audit"] = concurrent_capacity_audit["summary"]
    # LINE-BY-LINE: `metrics["same_machine_time_groups"]`에 `{` 결과를 저장합니다. 의미/사용: `metrics["same_machine_time_groups"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    metrics["same_machine_time_groups"] = {
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `group_count_min_2` 키에 `len(same_time_groups)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "group_count_min_2": len(same_time_groups),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `group_count_over_3` 키에 `len(same_time_groups_over3)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "group_count_over_3": len(same_time_groups_over3),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `max_group_size` 키에 `max((int(row["group_size"]) for row in same_time_groups), default=0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_group_size": max((int(row["group_size"]) for row in same_time_groups), default=0),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "note": (
            # LINE-BY-LINE: 문자열 값 `"Repeated exact actual start/end timestamps on the same machine are a data-pattern signal. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Repeated exact actual start/end timestamps on the same machine are a data-pattern signal. "
            # LINE-BY-LINE: 문자열 값 `"They may represent shared batch/window timestamps rather than individual W/O machine occupancy."`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "They may represent shared batch/window timestamps rather than individual W/O machine occupancy."
        ),
    }

    # LINE-BY-LINE: `_write_csv(schedule_rows, output_path / "actual_replay_schedule.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(schedule_rows, output_path / "actual_replay_schedule.csv")
    # LINE-BY-LINE: `_write_csv(schedule_rows, output_path / "actual_sequence.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(schedule_rows, output_path / "actual_sequence.csv")
    # LINE-BY-LINE: `_write_csv(machine_summary, output_path / "actual_machine_day_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(machine_summary, output_path / "actual_machine_day_summary.csv")
    # LINE-BY-LINE: `_write_csv(bay_summary, output_path / "actual_bay_day_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(bay_summary, output_path / "actual_bay_day_summary.csv")
    # LINE-BY-LINE: `_write_csv(time_validation["rows"], output_path / "actual_time_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(time_validation["rows"], output_path / "actual_time_validation.csv")
    # LINE-BY-LINE: `_write_csv(machine_intervals, output_path / "actual_machine_state_intervals.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(machine_intervals, output_path / "actual_machine_state_intervals.csv")
    # LINE-BY-LINE: `_write_csv(bay_intervals, output_path / "actual_bay_state_intervals.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(bay_intervals, output_path / "actual_bay_state_intervals.csv")
    # LINE-BY-LINE: `_write_csv(lifecycle_rows, output_path / "actual_job_lifecycle.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(lifecycle_rows, output_path / "actual_job_lifecycle.csv")
    # LINE-BY-LINE: `_write_csv(missing_event_requirements, output_path / "actual_missing_event_requirements.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(missing_event_requirements, output_path / "actual_missing_event_requirements.csv")
    # LINE-BY-LINE: `_write_csv(concurrent_capacity_audit["intervals"], output_path / "actual_concurrent_capacity_inte...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(concurrent_capacity_audit["intervals"], output_path / "actual_concurrent_capacity_intervals.csv")
    # LINE-BY-LINE: `_write_csv(concurrent_capacity_audit["active_jobs"], output_path / "actual_concurrent_capacity_ac...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(concurrent_capacity_audit["active_jobs"], output_path / "actual_concurrent_capacity_active_jobs.csv")
    # LINE-BY-LINE: `_write_csv(same_time_groups, output_path / "actual_same_machine_time_groups.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(same_time_groups, output_path / "actual_same_machine_time_groups.csv")
    # LINE-BY-LINE: `_write_csv(same_time_groups_over3, output_path / "actual_same_machine_time_groups_over3.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(same_time_groups_over3, output_path / "actual_same_machine_time_groups_over3.csv")
    # LINE-BY-LINE: `replay_validation_rows`에 `replay_validation["violations"] or [` 결과를 저장합니다. 의미/사용: `replay_validation_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    replay_validation_rows = replay_validation["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `factory_replay_passed` 키에 `replay_validation["factory_replay_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "factory_replay_passed": replay_validation["factory_replay_passed"],
            # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `replay_validation["identity_error_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "identity_error_count": replay_validation["identity_error_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"actual factory replay identity passed"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "actual factory replay identity passed",
        }
    ]
    # LINE-BY-LINE: `constraint_audit_rows`에 `constraint_audit["violations"] or [` 결과를 저장합니다. 의미/사용: `constraint_audit_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    constraint_audit_rows = constraint_audit["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `constraint_audit["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_passed": constraint_audit["hard_passed"],
            # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `constraint_audit["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_violation_count": constraint_audit["hard_violation_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"no planning constraint audit violations"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "no planning constraint audit violations",
        }
    ]
    # LINE-BY-LINE: `_write_csv(replay_validation_rows, output_path / "actual_factory_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(replay_validation_rows, output_path / "actual_factory_validation.csv")
    # LINE-BY-LINE: `_write_json(replay_validation, output_path / "actual_factory_validation.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(replay_validation, output_path / "actual_factory_validation.json")
    # LINE-BY-LINE: `_write_csv(replay_validation_rows, output_path / "actual_validation_result.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(replay_validation_rows, output_path / "actual_validation_result.csv")
    # LINE-BY-LINE: `_write_json(replay_validation, output_path / "actual_validation_result.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(replay_validation, output_path / "actual_validation_result.json")
    # LINE-BY-LINE: `event_identity_rows`에 `event_identity["violations"] or [` 결과를 저장합니다. 의미/사용: `event_identity_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_identity_rows = event_identity["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `event_identity["event_identity_passed"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "event_identity_passed": event_identity["event_identity_passed"],
            # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `event_identity["identity_error_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "identity_error_count": event_identity["identity_error_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"actual event log identity passed"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "actual event log identity passed",
        }
    ]
    # LINE-BY-LINE: `_write_csv(event_identity_rows, output_path / "actual_event_log_identity_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(event_identity_rows, output_path / "actual_event_log_identity_validation.csv")
    # LINE-BY-LINE: `_write_json(event_identity, output_path / "actual_event_log_identity_validation.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(event_identity, output_path / "actual_event_log_identity_validation.json")
    # LINE-BY-LINE: `event_constraint_rows`에 `event_constraint_validation["violations"] or [` 결과를 저장합니다. 의미/사용: `event_constraint_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    event_constraint_rows = event_constraint_validation["violations"] or [
        # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        {
            # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `rule` 키에 `"_summary"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule": "_summary",
            # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `event_constraint_validation["hard_passed"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_passed": event_constraint_validation["hard_passed"],
            # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `event_constraint_validation["hard_violation_count"]`를 저장합니다. 사용: replay/report 성공 여부 판단.
            "hard_violation_count": event_constraint_validation["hard_violation_count"],
            # LINE-BY-LINE: 딕셔너리 키 `message`에는 `"actual event log constraint validation passed"` 값을 넣습니다. 의미: 사람이 읽을 validation/error 설명입니다. 사용: VALIDATION_FAILURE 표시.
            "message": "actual event log constraint validation passed",
        }
    ]
    # LINE-BY-LINE: `_write_csv(event_constraint_rows, output_path / "actual_event_log_constraint_validation.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(event_constraint_rows, output_path / "actual_event_log_constraint_validation.csv")
    # LINE-BY-LINE: `_write_json(event_constraint_validation, output_path / "actual_event_log_constraint_validation.js...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(event_constraint_validation, output_path / "actual_event_log_constraint_validation.json")
    # LINE-BY-LINE: `_write_csv(constraint_audit_rows, output_path / "actual_constraint_audit_result.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(constraint_audit_rows, output_path / "actual_constraint_audit_result.csv")
    # LINE-BY-LINE: `_write_json(constraint_audit, output_path / "actual_constraint_audit_result.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(constraint_audit, output_path / "actual_constraint_audit_result.json")
    # LINE-BY-LINE: `_write_json(layout, output_path / "factory_layout.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(layout, output_path / "factory_layout.json")
    # LINE-BY-LINE: `_write_json(export_actual_event_log_rows, output_path / "actual_event_log.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(export_actual_event_log_rows, output_path / "actual_event_log.json")
    # LINE-BY-LINE: `_write_event_log_csv(export_actual_event_log_rows, output_path / "actual_event_log.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_event_log_csv(export_actual_event_log_rows, output_path / "actual_event_log.csv")
    # LINE-BY-LINE: `_write_json(events, output_path / "actual_schedule_events.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(events, output_path / "actual_schedule_events.json")
    # LINE-BY-LINE: `_write_json(machine_intervals, output_path / "actual_machine_state_intervals.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(machine_intervals, output_path / "actual_machine_state_intervals.json")
    # LINE-BY-LINE: `_write_json(bay_intervals, output_path / "actual_bay_state_intervals.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(bay_intervals, output_path / "actual_bay_state_intervals.json")
    # LINE-BY-LINE: `_write_json(lifecycle_rows, output_path / "actual_job_lifecycle.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(lifecycle_rows, output_path / "actual_job_lifecycle.json")
    # LINE-BY-LINE: `_write_json(missing_event_requirements, output_path / "actual_missing_event_requirements.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(missing_event_requirements, output_path / "actual_missing_event_requirements.json")
    # LINE-BY-LINE: `_write_json(concurrent_capacity_audit["intervals"], output_path / "actual_concurrent_capacity_int...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(concurrent_capacity_audit["intervals"], output_path / "actual_concurrent_capacity_intervals.json")
    # LINE-BY-LINE: `_write_json(concurrent_capacity_audit["active_jobs"], output_path / "actual_concurrent_capacity_a...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(concurrent_capacity_audit["active_jobs"], output_path / "actual_concurrent_capacity_active_jobs.json")
    # LINE-BY-LINE: `_write_json(concurrent_capacity_audit["summary"], output_path / "actual_concurrent_capacity_summa...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(concurrent_capacity_audit["summary"], output_path / "actual_concurrent_capacity_summary.json")
    # LINE-BY-LINE: `_write_json(same_time_groups, output_path / "actual_same_machine_time_groups.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(same_time_groups, output_path / "actual_same_machine_time_groups.json")
    # LINE-BY-LINE: `_write_json(same_time_groups_over3, output_path / "actual_same_machine_time_groups_over3.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(same_time_groups_over3, output_path / "actual_same_machine_time_groups_over3.json")
    # LINE-BY-LINE: `_write_json(time_validation["summary"], output_path / "actual_time_validation_summary.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(time_validation["summary"], output_path / "actual_time_validation_summary.json")
    # LINE-BY-LINE: `_write_json(metrics, output_path / "actual_metrics.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(metrics, output_path / "actual_metrics.json")
    # LINE-BY-LINE: `(output_path / "actual_replay.html").write_text(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    (output_path / "actual_replay.html").write_text(
        # LINE-BY-LINE: `write_text(...)` 호출에 `_factory_simulator_document(layout, events, schedule_rows, replay_validation, metrics)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        _factory_simulator_document(layout, events, schedule_rows, replay_validation, metrics),
        # LINE-BY-LINE: `encoding`에 `"utf-8"` 결과를 저장합니다. 의미/사용: `encoding` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        encoding="utf-8",
    )

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: 딕셔너리 키 `output_dir`에는 `str(output_path)` 값을 넣습니다. 의미: 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        "output_dir": str(output_path),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `playback_html` 키에 `str(output_path / "actual_replay.html")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_html": str(output_path / "actual_replay.html"),
        # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `str(len(schedule_rows))`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "scheduled_jobs": str(len(schedule_rows)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `factory_replay_passed` 키에 `str(replay_validation["factory_replay_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "factory_replay_passed": str(replay_validation["factory_replay_passed"]),
        # LINE-BY-LINE: validation 결과의 `identity_error_count` 항목에 `str(replay_validation["identity_error_count"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "identity_error_count": str(replay_validation["identity_error_count"]),
        # LINE-BY-LINE: 실행 summary의 `event_log_count` 항목에 `str(len(export_actual_event_log_rows))`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "event_log_count": str(len(export_actual_event_log_rows)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `playback_event_count` 키에 `str(len(events))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "playback_event_count": str(len(events)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_identity_passed` 키에 `str(event_identity["event_identity_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_identity_passed": str(event_identity["event_identity_passed"]),
        # LINE-BY-LINE: validation 결과의 `event_identity_error_count` 항목에 `str(event_identity["identity_error_count"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "event_identity_error_count": str(event_identity["identity_error_count"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_constraint_passed` 키에 `str(event_constraint_validation["hard_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_passed": str(event_constraint_validation["hard_passed"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `event_constraint_violation_count` 키에 `str(event_constraint_validation["hard_violation_count"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "event_constraint_violation_count": str(event_constraint_validation["hard_violation_count"]),
        # LINE-BY-LINE: validation 결과의 `hard_passed` 항목에 `str(replay_validation["hard_passed"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_passed": str(replay_validation["hard_passed"]),
        # LINE-BY-LINE: validation 결과의 `hard_violation_count` 항목에 `str(replay_validation["hard_violation_count"])`를 저장합니다. 사용: replay/report 성공 여부 판단.
        "hard_violation_count": str(replay_validation["hard_violation_count"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `constraint_audit_passed` 키에 `str(constraint_audit["hard_passed"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "constraint_audit_passed": str(constraint_audit["hard_passed"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `constraint_audit_violation_count` 키에 `str(constraint_audit["hard_violation_count"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "constraint_audit_violation_count": str(constraint_audit["hard_violation_count"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `constraint_audit_affected_job_count` 키에 `str(constraint_audit.get("hard_violation_affected_job_count", 0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "constraint_audit_affected_job_count": str(constraint_audit.get("hard_violation_affected_job_count", 0)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `constraint_audit_excess_wo_count` 키에 `str(constraint_audit.get("hard_violation_excess_wo_count", 0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "constraint_audit_excess_wo_count": str(constraint_audit.get("hard_violation_excess_wo_count", 0)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `constraint_audit_length_excess_total` 키에 `str(constraint_audit.get("hard_violation_length_excess_total", 0.0))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "constraint_audit_length_excess_total": str(constraint_audit.get("hard_violation_length_excess_total", 0.0)),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `missing_event_count` 키에 `str(metrics["factory_replay_scope"]["missing_event_count"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "missing_event_count": str(metrics["factory_replay_scope"]["missing_event_count"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `missing_events_are_inferred` 키에 `str(metrics["factory_replay_scope"]["missing_events_are_inferred"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "missing_events_are_inferred": str(metrics["factory_replay_scope"]["missing_events_are_inferred"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `duration_mae_minutes` 키에 `str(time_validation["summary"]["duration_mae_minutes"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mae_minutes": str(time_validation["summary"]["duration_mae_minutes"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `duration_mape_percent` 키에 `str(time_validation["summary"]["duration_mape_percent"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "duration_mape_percent": str(time_validation["summary"]["duration_mape_percent"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `tact_mae_minutes` 키에 `str(time_validation["summary"]["tact_mae_minutes"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mae_minutes": str(time_validation["summary"]["tact_mae_minutes"]),
        # LINE-BY-LINE: `write_actual_replay_artifacts`에서 반환/저장할 dict의 `tact_mape_percent` 키에 `str(time_validation["summary"]["tact_mape_percent"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_mape_percent": str(time_validation["summary"]["tact_mape_percent"]),
    }
