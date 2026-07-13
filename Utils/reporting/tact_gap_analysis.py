"""TACT_TIME과 실적 elapsed time 차이 원인 분석.

이 파일의 책임:
- 제공 TACT_TIME과 `실적 종료시간 - 실적 착수시간`의 차이를 분석한다.
- 같은 설비/같은 착수/같은 종료 timestamp group을 찾아 batch/window 의심 패턴을 표시한다.
- 학습 target 후보를 만들 때 사용할 진단용 row와 summary를 저장한다.

중요:
- 여기서 만드는 batch/window 보정 target은 진단/학습 후보일 뿐이다.
- actual replay DES 검증 시간은 원본 elapsed time 그대로 유지한다.
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
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Optional, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Optional, Tuple


# LINE-BY-LINE: `_to_float(value: Any, field_name: str = "value", context: str = "")` 함수를 정의합니다. 반환 타입: `float`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _to_float(value: Any, field_name: str = "value", context: str = "") -> float:
    """숫자 필드를 strict하게 float로 변환한다.

    실패하면 원인을 print하고 예외를 발생시킨다.
    이 분석은 fallback으로 0을 넣으면 상관관계와 오차가 왜곡된다.
    """

    # LINE-BY-LINE: 조건 `value in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value in (None, ""):
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][tact_gap_analysis._to_float] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][tact_gap_analysis._to_float] "
            # LINE-BY-LINE: `f"cause`에 `missing_numeric_value field={field_name} context={context}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=missing_numeric_value field={field_name} context={context}"
        )
        # LINE-BY-LINE: `ValueError(f"missing numeric value: {field_name}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"missing numeric value: {field_name}")
    # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
    try:
        # LINE-BY-LINE: 호출자에게 `float(value)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return float(value)
    # LINE-BY-LINE: `except (TypeError, ValueError):` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
    except (TypeError, ValueError):
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: 문자열 값 `"[ERROR][tact_gap_analysis._to_float] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "[ERROR][tact_gap_analysis._to_float] "
            # LINE-BY-LINE: `f"cause`에 `invalid_numeric_value field={field_name} value={value} context={context}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"cause=invalid_numeric_value field={field_name} value={value} context={context}"
        )
        # LINE-BY-LINE: `ValueError(f"invalid numeric value for {field_name}: {value}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"invalid numeric value for {field_name}: {value}")


# LINE-BY-LINE: `_safe_div(numerator: float, denominator: float)` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    """0 나눗셈을 피해서 비율을 계산한다."""

    # LINE-BY-LINE: 조건 `abs(denominator) <= 1e-12`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if abs(denominator) <= 1e-12:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: 호출자에게 `numerator / denominator`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return numerator / denominator


# LINE-BY-LINE: `_write_csv(rows: Iterable[Dict[str, Any]], path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다. 예: 현업 Excel/CSV row를 dict로 변환.
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


# LINE-BY-LINE: `_write_json(data: Any, path: Path)` 함수를 정의합니다. 반환 타입: `None`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _write_json(data: Any, path: Path) -> None:
    """분석 summary를 JSON으로 저장한다."""

    # LINE-BY-LINE: `path.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `path.write_text(json.dumps(data, ensure_ascii` 여러 변수에 `False, indent=2), encoding="utf-8")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# LINE-BY-LINE: `_pearson(xs: List[float], ys: List[float])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """두 변수의 Pearson 상관계수를 계산한다."""

    # LINE-BY-LINE: 조건 `len(xs) < 2 or len(xs) != len(ys)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(xs) < 2 or len(xs) != len(ys):
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `mean_x`에 `statistics.mean(xs)` 결과를 저장합니다. 의미/사용: `mean_x` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mean_x = statistics.mean(xs)
    # LINE-BY-LINE: `mean_y`에 `statistics.mean(ys)` 결과를 저장합니다. 의미/사용: `mean_y` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mean_y = statistics.mean(ys)
    # LINE-BY-LINE: `var_x`에 `sum((x - mean_x) ** 2 for x in xs)` 결과를 저장합니다. 의미/사용: `var_x` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    var_x = sum((x - mean_x) ** 2 for x in xs)
    # LINE-BY-LINE: `var_y`에 `sum((y - mean_y) ** 2 for y in ys)` 결과를 저장합니다. 의미/사용: `var_y` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    var_y = sum((y - mean_y) ** 2 for y in ys)
    # LINE-BY-LINE: 조건 `var_x <= 0 or var_y <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if var_x <= 0 or var_y <= 0:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `cov`에 `sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))` 결과를 저장합니다. 의미/사용: `cov` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    # LINE-BY-LINE: 호출자에게 `cov / math.sqrt(var_x * var_y)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return cov / math.sqrt(var_x * var_y)


# LINE-BY-LINE: `_summary_stats(values: List[float])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _summary_stats(values: List[float]) -> Dict[str, Any]:
    """count/mean/median/std/min/max 요약 통계를 계산한다."""

    # LINE-BY-LINE: 조건 `not values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not values:
        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `count` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "count": 0,
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `mean` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mean": None,
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `median` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "median": None,
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `std` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "std": None,
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `min` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "min": None,
            # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `max` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "max": None,
        }
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `count` 키에 `len(values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "count": len(values),
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `mean` 키에 `statistics.mean(values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mean": statistics.mean(values),
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `median` 키에 `statistics.median(values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "median": statistics.median(values),
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `std` 키에 `statistics.stdev(values) if len(values) >= 2 else 0.0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "std": statistics.stdev(values) if len(values) >= 2 else 0.0,
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `min` 키에 `min(values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "min": min(values),
        # LINE-BY-LINE: `_summary_stats`에서 반환/저장할 dict의 `max` 키에 `max(values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max": max(values),
    }


# LINE-BY-LINE: `_pct(numerator: int, denominator: int)` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _pct(numerator: int, denominator: int) -> Optional[float]:
    """분모가 0일 수 있는 비율을 percent로 계산한다."""

    # LINE-BY-LINE: 조건 `denominator <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if denominator <= 0:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: 호출자에게 `numerator / denominator * 100.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return numerator / denominator * 100.0


# LINE-BY-LINE: `_thickness_bucket(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _thickness_bucket(value: Any) -> str:
    """두께 값을 분석용 구간 label로 변환한다."""

    # LINE-BY-LINE: `thickness`에 `_to_float(value)` 결과를 저장합니다. 의미/사용: `thickness`는 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
    thickness = _to_float(value)
    # LINE-BY-LINE: 조건 `thickness <= 12`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if thickness <= 12:
        # LINE-BY-LINE: 호출자에게 `"THK_<=12"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "THK_<=12"
    # LINE-BY-LINE: 조건 `thickness <= 20`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if thickness <= 20:
        # LINE-BY-LINE: 호출자에게 `"THK_12_20"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "THK_12_20"
    # LINE-BY-LINE: 조건 `thickness <= 30`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if thickness <= 30:
        # LINE-BY-LINE: 호출자에게 `"THK_20_30"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "THK_20_30"
    # LINE-BY-LINE: 호출자에게 `"THK_>30"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return "THK_>30"


# LINE-BY-LINE: `_length_bucket(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _length_bucket(value: Any) -> str:
    """판 길이 값을 분석용 구간 label로 변환한다."""

    # LINE-BY-LINE: `length`에 `_to_float(value)` 결과를 저장합니다. 의미/사용: `length`는 원본 길이 LTH입니다. 사용: Job.plate_length로 변환되어 capacity 제약에 들어갑니다.
    length = _to_float(value)
    # LINE-BY-LINE: 조건 `length <= 5000`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if length <= 5000:
        # LINE-BY-LINE: 호출자에게 `"LTH_<=5000"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "LTH_<=5000"
    # LINE-BY-LINE: 조건 `length <= 10000`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if length <= 10000:
        # LINE-BY-LINE: 호출자에게 `"LTH_5000_10000"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "LTH_5000_10000"
    # LINE-BY-LINE: 조건 `length <= 15000`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if length <= 15000:
        # LINE-BY-LINE: 호출자에게 `"LTH_10000_15000"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "LTH_10000_15000"
    # LINE-BY-LINE: 호출자에게 `"LTH_>15000"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return "LTH_>15000"


# LINE-BY-LINE: `_cut_length_bucket(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _cut_length_bucket(value: Any) -> str:
    """절단 길이 값을 분석용 구간 label로 변환한다."""

    # LINE-BY-LINE: `cut_length`에 `_to_float(value)` 결과를 저장합니다. 의미/사용: `cut_length`는 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
    cut_length = _to_float(value)
    # LINE-BY-LINE: 조건 `cut_length <= 50`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if cut_length <= 50:
        # LINE-BY-LINE: 호출자에게 `"CUT_LTH_<=50"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "CUT_LTH_<=50"
    # LINE-BY-LINE: 조건 `cut_length <= 100`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if cut_length <= 100:
        # LINE-BY-LINE: 호출자에게 `"CUT_LTH_50_100"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "CUT_LTH_50_100"
    # LINE-BY-LINE: 조건 `cut_length <= 200`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if cut_length <= 200:
        # LINE-BY-LINE: 호출자에게 `"CUT_LTH_100_200"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "CUT_LTH_100_200"
    # LINE-BY-LINE: 호출자에게 `"CUT_LTH_>200"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return "CUT_LTH_>200"


# LINE-BY-LINE: `_group_size_bucket(value: int)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _group_size_bucket(value: int) -> str:
    """동일 timestamp group 크기를 분석용 label로 변환한다."""

    # LINE-BY-LINE: 조건 `value <= 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value <= 1:
        # LINE-BY-LINE: 호출자에게 `"group_1"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "group_1"
    # LINE-BY-LINE: 조건 `value <= 3`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value <= 3:
        # LINE-BY-LINE: 호출자에게 `"group_2_3"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "group_2_3"
    # LINE-BY-LINE: 조건 `value <= 10`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if value <= 10:
        # LINE-BY-LINE: 호출자에게 `"group_4_10"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "group_4_10"
    # LINE-BY-LINE: 호출자에게 `"group_>10"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return "group_>10"


# LINE-BY-LINE: `_build_base_rows(scenario: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _build_base_rows(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    """scenario jobs를 tact gap 분석용 row 목록으로 변환한다.

    핵심 산출:
    - actual_duration_minutes
    - tact_time
    - actual_minus_tact
    - same_machine_time_group_id
    - learning_target_candidate_duration
    """

    # LINE-BY-LINE: `rows`에 `[]` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = []
    # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in scenario.get("jobs", []):
        # LINE-BY-LINE: `job_id`에 `str(job.get("job_id", ""))` 결과를 저장합니다. 의미/사용: `job_id`는 W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
        job_id = str(job.get("job_id", ""))
        # LINE-BY-LINE: `context`에 `f"job_id={job_id}"` 결과를 저장합니다. 의미/사용: `context` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        context = f"job_id={job_id}"
        # LINE-BY-LINE: `base_stage`에 `job.get("base_stage_minutes", {}) or {}` 결과를 저장합니다. 의미/사용: `base_stage` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_stage = job.get("base_stage_minutes", {}) or {}
        # LINE-BY-LINE: `tact_source`에 `job.get("tact_time_minutes")` 결과를 저장합니다. 의미/사용: `tact_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_source = job.get("tact_time_minutes")
        # LINE-BY-LINE: 조건 `tact_source in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if tact_source in (None, ""):
            # LINE-BY-LINE: `tact_source`에 `base_stage.get("cut")` 결과를 저장합니다. 의미/사용: `tact_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_source = base_stage.get("cut")
        # LINE-BY-LINE: `tact_time`에 `_to_float(tact_source, "tact_time_minutes", context)` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_time = _to_float(tact_source, "tact_time_minutes", context)
        # LINE-BY-LINE: `actual_duration`에 `_to_float(job.get("actual_duration_minutes"), "actual_duration_minutes", context)` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float(job.get("actual_duration_minutes"), "actual_duration_minutes", context)
        # LINE-BY-LINE: `gap`에 `actual_duration - tact_time` 결과를 저장합니다. 의미/사용: `gap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        gap = actual_duration - tact_time
        # LINE-BY-LINE: `abs_gap`에 `abs(gap)` 결과를 저장합니다. 의미/사용: `abs_gap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        abs_gap = abs(gap)
        # LINE-BY-LINE: `ratio`에 `_safe_div(actual_duration, tact_time)` 결과를 저장합니다. 의미/사용: `ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        ratio = _safe_div(actual_duration, tact_time)
        # LINE-BY-LINE: `actual_start`에 `str(job.get("actual_start_datetime", ""))` 결과를 저장합니다. 의미/사용: `actual_start` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_start = str(job.get("actual_start_datetime", ""))
        # LINE-BY-LINE: `actual_end`에 `str(job.get("actual_end_datetime", ""))` 결과를 저장합니다. 의미/사용: `actual_end` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_end = str(job.get("actual_end_datetime", ""))
        # LINE-BY-LINE: 조건 `not actual_start or not actual_end`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not actual_start or not actual_end:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][tact_gap_analysis._build_base_rows] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][tact_gap_analysis._build_base_rows] "
                # LINE-BY-LINE: `f"cause`에 `missing_actual_datetime {context} actual_start={actual_start} actual_end={actual_end}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=missing_actual_datetime {context} actual_start={actual_start} actual_end={actual_end}"
            )
            # LINE-BY-LINE: `ValueError("actual datetime is required for tact gap analysis")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("actual datetime is required for tact gap analysis")
        # LINE-BY-LINE: `row`에 `{` 결과를 저장합니다. 의미/사용: `row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        row = {
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": job_id,
            # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
            "work_order_no": job.get("source_wk_ord_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `project_no`에는 `job.get("source_project_no", "")` 값을 넣습니다. 의미: 호선번호입니다. 예: `PROJ_NO`, 사용: block_set_id와 원본 추적.
            "project_no": job.get("source_project_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `block_no`에는 `job.get("source_block_no", "")` 값을 넣습니다. 의미: 블록명입니다. 예: `BLK_NO`, 사용: block_set_id 구성과 동일 block Bay 제약.
            "block_no": job.get("source_block_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `job.get("source_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": job.get("source_machine_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `str(job.get("source_cut_bay", ""))` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
            "cut_bay": str(job.get("source_cut_bay", "")),
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `actual_day` 키에 `actual_start[:8]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_day": actual_start[:8],
            # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `actual_start` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
            "actual_start_datetime": actual_start,
            # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `actual_end` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
            "actual_end_datetime": actual_end,
            # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `actual_duration` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
            "actual_duration_minutes": actual_duration,
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `tact_time` 키에 `tact_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "tact_time": tact_time,
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `actual_minus_tact` 키에 `gap` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_minus_tact": gap,
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `abs_gap_minutes` 키에 `abs_gap` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "abs_gap_minutes": abs_gap,
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `actual_to_tact_ratio` 키에 `"" if ratio is None else ratio` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_to_tact_ratio": "" if ratio is None else ratio,
            # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `_to_float(job.get("plate_length"), "plate_length", context)` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
            "plate_length": _to_float(job.get("plate_length"), "plate_length", context),
            # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `_to_float(job.get("thickness"), "thickness", context)` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
            "thickness": _to_float(job.get("thickness"), "thickness", context),
            # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `_to_float(job.get("cut_length"), "cut_length", context)` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
            "cut_length": _to_float(job.get("cut_length"), "cut_length", context),
            # LINE-BY-LINE: 딕셔너리 키 `mark_length`에는 `_to_float(job.get("mark_length"), "mark_length", context)` 값을 넣습니다. 의미: 마킹 길이입니다. 사용: tact time 분석 feature.
            "mark_length": _to_float(job.get("mark_length"), "mark_length", context),
            # LINE-BY-LINE: 딕셔너리 키 `bevel_length`에는 `_to_float(job.get("bevel_length"), "bevel_length", context)` 값을 넣습니다. 의미: 베벨 길이입니다. 사용: tact time 분석 feature.
            "bevel_length": _to_float(job.get("bevel_length"), "bevel_length", context),
            # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `_to_float(job.get("part_count"), "part_count", context)` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
            "part_count": _to_float(job.get("part_count"), "part_count", context),
            # LINE-BY-LINE: 딕셔너리 키 `steel_qty`에는 `_to_float(job.get("steel_qty"), "steel_qty", context)` 값을 넣습니다. 의미: 원본 강재 수량입니다. 예: `STL_QTY`, 사용: steel_quantity로 변환.
            "steel_qty": _to_float(job.get("steel_qty"), "steel_qty", context),
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `thickness_bucket` 키에 `_thickness_bucket(job.get("thickness"))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "thickness_bucket": _thickness_bucket(job.get("thickness")),
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `length_bucket` 키에 `_length_bucket(job.get("plate_length"))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "length_bucket": _length_bucket(job.get("plate_length")),
            # LINE-BY-LINE: `_build_base_rows`에서 반환/저장할 dict의 `cut_length_bucket` 키에 `_cut_length_bucket(job.get("cut_length"))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "cut_length_bucket": _cut_length_bucket(job.get("cut_length")),
        }
        # LINE-BY-LINE: `rows.append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(row)

    # LINE-BY-LINE: `groups` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `groups[(row["machine_id"], row["actual_start_datetime"], row["actual_end_datetime"])].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        groups[(row["machine_id"], row["actual_start_datetime"], row["actual_end_datetime"])].append(row)
    # LINE-BY-LINE: `index, (group_key, group_rows) in enumerate(sorted(groups.items()), start=1)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for index, (group_key, group_rows) in enumerate(sorted(groups.items()), start=1):
        # LINE-BY-LINE: `group_id`에 `f"same_machine_time_{index:05d}"` 결과를 저장합니다. 의미/사용: `group_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        group_id = f"same_machine_time_{index:05d}"
        # LINE-BY-LINE: `total_tact`에 `sum(_to_float(row["tact_time"]) for row in group_rows)` 결과를 저장합니다. 의미/사용: `total_tact` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        total_tact = sum(_to_float(row["tact_time"]) for row in group_rows)
        # LINE-BY-LINE: `max_abs_gap`에 `max((_to_float(row["abs_gap_minutes"]) for row in group_rows), default=0.0)` 결과를 저장합니다. 의미/사용: `max_abs_gap` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_abs_gap = max((_to_float(row["abs_gap_minutes"]) for row in group_rows), default=0.0)
        # LINE-BY-LINE: `group_size`에 `len(group_rows)` 결과를 저장합니다. 의미/사용: `group_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        group_size = len(group_rows)
        # LINE-BY-LINE: `group_window_duration`에 `_to_float(group_rows[0]["actual_duration_minutes"])` 결과를 저장합니다. 의미/사용: `group_window_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        group_window_duration = _to_float(group_rows[0]["actual_duration_minutes"])
        # LINE-BY-LINE: 조건 `group_size >= 2 and total_tact <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if group_size >= 2 and total_tact <= 0:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][tact_gap_analysis._build_base_rows] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][tact_gap_analysis._build_base_rows] "
                # LINE-BY-LINE: `f"cause`에 `non_positive_group_total_tact group_id={group_id} group_key={group_key}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=non_positive_group_total_tact group_id={group_id} group_key={group_key}"
            )
            # LINE-BY-LINE: `ValueError("batch/window group requires positive total tact for weighted allocation")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError("batch/window group requires positive total tact for weighted allocation")
        # LINE-BY-LINE: `row in group_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in group_rows:
            # LINE-BY-LINE: `tact_time`에 `_to_float(row["tact_time"])` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            tact_time = _to_float(row["tact_time"])
            # LINE-BY-LINE: 조건 `group_size >= 2`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if group_size >= 2:
                # LINE-BY-LINE: `equal_share_duration`에 `group_window_duration / group_size` 결과를 저장합니다. 의미/사용: `equal_share_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                equal_share_duration = group_window_duration / group_size
                # LINE-BY-LINE: `tact_weighted_duration`에 `group_window_duration * tact_time / total_tact` 결과를 저장합니다. 의미/사용: `tact_weighted_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                tact_weighted_duration = group_window_duration * tact_time / total_tact
                # LINE-BY-LINE: `target_candidate`에 `tact_weighted_duration` 결과를 저장합니다. 의미/사용: `target_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                target_candidate = tact_weighted_duration
                # LINE-BY-LINE: `target_candidate_method`에 `"batch_window_tact_weighted"` 결과를 저장합니다. 의미/사용: `target_candidate_method` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                target_candidate_method = "batch_window_tact_weighted"
            # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
            else:
                # LINE-BY-LINE: `equal_share_duration`에 `group_window_duration` 결과를 저장합니다. 의미/사용: `equal_share_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                equal_share_duration = group_window_duration
                # LINE-BY-LINE: `tact_weighted_duration`에 `group_window_duration` 결과를 저장합니다. 의미/사용: `tact_weighted_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                tact_weighted_duration = group_window_duration
                # LINE-BY-LINE: `target_candidate`에 `group_window_duration` 결과를 저장합니다. 의미/사용: `target_candidate` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                target_candidate = group_window_duration
                # LINE-BY-LINE: `target_candidate_method`에 `"single_observed_elapsed"` 결과를 저장합니다. 의미/사용: `target_candidate_method` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                target_candidate_method = "single_observed_elapsed"

            # LINE-BY-LINE: `row["same_machine_time_group_id"]`에 `group_id` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_id"] = group_id
            # LINE-BY-LINE: `row["same_machine_time_group_size"]`에 `group_size` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_size"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_size"] = group_size
            # LINE-BY-LINE: `row["same_machine_time_group_total_tact"]`에 `round(total_tact, 6)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_total_tact"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_total_tact"] = round(total_tact, 6)
            # LINE-BY-LINE: `row["same_machine_time_group_max_abs_gap"]`에 `round(max_abs_gap, 6)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_max_abs_gap"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_max_abs_gap"] = round(max_abs_gap, 6)
            # LINE-BY-LINE: `row["same_machine_time_group_window_duration"]`에 `round(group_window_duration, 6)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_window_duration"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_window_duration"] = round(group_window_duration, 6)
            # LINE-BY-LINE: `row["same_machine_time_group_equal_share_duration"]`에 `round(equal_share_duration, 6)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_equal_share_duration"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_equal_share_duration"] = round(equal_share_duration, 6)
            # LINE-BY-LINE: `row["same_machine_time_group_tact_weighted_duration"]`에 `round(tact_weighted_duration, 6)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_tact_weighted_duration"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_tact_weighted_duration"] = round(tact_weighted_duration, 6)
            # LINE-BY-LINE: `row["learning_target_candidate_duration"]`에 `round(target_candidate, 6)` 결과를 저장합니다. 의미/사용: `row["learning_target_candidate_duration"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["learning_target_candidate_duration"] = round(target_candidate, 6)
            # LINE-BY-LINE: `row["learning_target_candidate_method"]`에 `target_candidate_method` 결과를 저장합니다. 의미/사용: `row["learning_target_candidate_method"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["learning_target_candidate_method"] = target_candidate_method
            # LINE-BY-LINE: `row["learning_target_candidate_minus_tact"]`에 `round(target_candidate - tact_time, 6)` 결과를 저장합니다. 의미/사용: `row["learning_target_candidate_minus_tact"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["learning_target_candidate_minus_tact"] = round(target_candidate - tact_time, 6)
            # LINE-BY-LINE: `row["learning_target_candidate_abs_gap_to_tact"]`에 `round(abs(target_candidate - tact_time), 6)` 결과를 저장합니다. 의미/사용: `row["learning_target_candidate_abs_gap_to_tact"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["learning_target_candidate_abs_gap_to_tact"] = round(abs(target_candidate - tact_time), 6)
            # LINE-BY-LINE: `row["same_machine_time_group_window_to_total_tact_ratio"]`에 `(` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_window_to_total_tact_ratio"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_window_to_total_tact_ratio"] = (
                # LINE-BY-LINE: `"" if total_tact <= 0 else round(group_window_duration / total_tact, 6)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                "" if total_tact <= 0 else round(group_window_duration / total_tact, 6)
            )
            # LINE-BY-LINE: `row["same_machine_time_group_bucket"]`에 `_group_size_bucket(group_size)` 결과를 저장합니다. 의미/사용: `row["same_machine_time_group_bucket"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["same_machine_time_group_bucket"] = _group_size_bucket(group_size)
    # LINE-BY-LINE: 호출자에게 `rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows


# LINE-BY-LINE: `_metric_row(group_name: str, group_value: str, rows: List[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _metric_row(group_name: str, group_value: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """특정 그룹의 tact/actual 차이 지표 1행을 만든다."""

    # LINE-BY-LINE: `count`에 `len(rows)` 결과를 저장합니다. 의미/사용: `count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    count = len(rows)
    # LINE-BY-LINE: `actual_values`에 `[_to_float(row["actual_duration_minutes"]) for row in rows]` 결과를 저장합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_values = [_to_float(row["actual_duration_minutes"]) for row in rows]
    # LINE-BY-LINE: `tact_values`에 `[_to_float(row["tact_time"]) for row in rows]` 결과를 저장합니다. 의미/사용: `tact_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_values = [_to_float(row["tact_time"]) for row in rows]
    # LINE-BY-LINE: `gaps`에 `[_to_float(row["actual_minus_tact"]) for row in rows]` 결과를 저장합니다. 의미/사용: `gaps` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    gaps = [_to_float(row["actual_minus_tact"]) for row in rows]
    # LINE-BY-LINE: `abs_gaps`에 `[_to_float(row["abs_gap_minutes"]) for row in rows]` 결과를 저장합니다. 의미/사용: `abs_gaps` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    abs_gaps = [_to_float(row["abs_gap_minutes"]) for row in rows]
    # LINE-BY-LINE: `positive_actual_values`에 `[value for value in actual_values if value > 0]` 결과를 저장합니다. 의미/사용: `positive_actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    positive_actual_values = [value for value in actual_values if value > 0]
    # LINE-BY-LINE: `mape_values`에 `[` 결과를 저장합니다. 의미/사용: `mape_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mape_values = [
        # LINE-BY-LINE: `abs(_to_float(row["actual_minus_tact"])) / _to_float(row["actual_duration_minutes"])`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        abs(_to_float(row["actual_minus_tact"])) / _to_float(row["actual_duration_minutes"])
        # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in rows
        # LINE-BY-LINE: 조건 `_to_float(row["actual_duration_minutes"]) > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if _to_float(row["actual_duration_minutes"]) > 0
    ]
    # LINE-BY-LINE: `actual_gt_tact`에 `sum(1 for row in rows if _to_float(row["actual_minus_tact"]) > 0)` 결과를 저장합니다. 의미/사용: `actual_gt_tact` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_gt_tact = sum(1 for row in rows if _to_float(row["actual_minus_tact"]) > 0)
    # LINE-BY-LINE: `tact_gt_actual`에 `sum(1 for row in rows if _to_float(row["actual_minus_tact"]) < 0)` 결과를 저장합니다. 의미/사용: `tact_gt_actual` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_gt_actual = sum(1 for row in rows if _to_float(row["actual_minus_tact"]) < 0)
    # LINE-BY-LINE: `same`에 `sum(1 for row in rows if abs(_to_float(row["actual_minus_tact"])) <= 1e-9)` 결과를 저장합니다. 의미/사용: `same` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    same = sum(1 for row in rows if abs(_to_float(row["actual_minus_tact"])) <= 1e-9)
    # LINE-BY-LINE: `within_5`에 `sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 5)` 결과를 저장합니다. 의미/사용: `within_5` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    within_5 = sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 5)
    # LINE-BY-LINE: `within_10`에 `sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 10)` 결과를 저장합니다. 의미/사용: `within_10` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    within_10 = sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 10)
    # LINE-BY-LINE: `within_20`에 `sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 20)` 결과를 저장합니다. 의미/사용: `within_20` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    within_20 = sum(1 for row in rows if _to_float(row["abs_gap_minutes"]) <= 20)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `group_name` 키에 `group_name` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "group_name": group_name,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `group_value` 키에 `group_value` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "group_value": group_value,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `count` 키에 `count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "count": count,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `actual_duration_mean` 키에 `statistics.mean(actual_values) if actual_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_duration_mean": statistics.mean(actual_values) if actual_values else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `actual_duration_median` 키에 `statistics.median(actual_values) if actual_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_duration_median": statistics.median(actual_values) if actual_values else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `tact_time_mean` 키에 `statistics.mean(tact_values) if tact_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_time_mean": statistics.mean(tact_values) if tact_values else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `tact_time_median` 키에 `statistics.median(tact_values) if tact_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_time_median": statistics.median(tact_values) if tact_values else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `bias_actual_minus_tact` 키에 `statistics.mean(gaps) if gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bias_actual_minus_tact": statistics.mean(gaps) if gaps else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `mae_minutes` 키에 `statistics.mean(abs_gaps) if abs_gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mae_minutes": statistics.mean(abs_gaps) if abs_gaps else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `rmse_minutes` 키에 `math.sqrt(statistics.mean(gap**2 for gap in gaps)) if gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "rmse_minutes": math.sqrt(statistics.mean(gap**2 for gap in gaps)) if gaps else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `mape_percent` 키에 `statistics.mean(mape_values) * 100.0 if mape_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mape_percent": statistics.mean(mape_values) * 100.0 if mape_values else None,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `actual_gt_tact_count` 키에 `actual_gt_tact` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_gt_tact_count": actual_gt_tact,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `tact_gt_actual_count` 키에 `tact_gt_actual` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_gt_actual_count": tact_gt_actual,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `exact_match_count` 키에 `same` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "exact_match_count": same,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `within_5min_count` 키에 `within_5` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "within_5min_count": within_5,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `within_10min_count` 키에 `within_10` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "within_10min_count": within_10,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `within_20min_count` 키에 `within_20` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "within_20min_count": within_20,
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `within_20min_rate` 키에 `_pct(within_20, count)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "within_20min_rate": _pct(within_20, count),
        # LINE-BY-LINE: `_metric_row`에서 반환/저장할 dict의 `zero_actual_duration_count` 키에 `count - len(positive_actual_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "zero_actual_duration_count": count - len(positive_actual_values),
    }


# LINE-BY-LINE: `_group_summary(rows: List[Dict[str, Any]], group_name: str, key: str)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _group_summary(rows: List[Dict[str, Any]], group_name: str, key: str) -> List[Dict[str, Any]]:
    """key별로 row를 묶어 `_metric_row` summary를 만든다."""

    # LINE-BY-LINE: `grouped` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `grouped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `grouped[str(row.get(key, ""))].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        grouped[str(row.get(key, ""))].append(row)
    # LINE-BY-LINE: 호출자에게 list 반환을 시작합니다. 사용: 여러 row 또는 후보 값을 순서 있는 목록으로 전달합니다.
    return [
        # LINE-BY-LINE: `_metric_row(group_name, group_value, group_rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        _metric_row(group_name, group_value, group_rows)
        # LINE-BY-LINE: `group_value, group_rows in sorted(grouped.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for group_value, group_rows in sorted(grouped.items())
    ]


# LINE-BY-LINE: `_correlation_rows(rows: List[Dict[str, Any]], variables: List[str])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _correlation_rows(rows: List[Dict[str, Any]], variables: List[str]) -> List[Dict[str, Any]]:
    """변수쌍별 Pearson 상관계수 행을 만든다."""

    # LINE-BY-LINE: `output`에 `[]` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output = []
    # LINE-BY-LINE: `left in variables` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for left in variables:
        # LINE-BY-LINE: `right in variables` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for right in variables:
            # LINE-BY-LINE: `pairs`에 `[` 결과를 저장합니다. 의미/사용: `pairs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            pairs = [
                # LINE-BY-LINE: `(_to_float(row.get(left)), _to_float(row.get(right)))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                (_to_float(row.get(left)), _to_float(row.get(right)))
                # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for row in rows
                # LINE-BY-LINE: 조건 `row.get(left, "") not in (None, "") and row.get(right, "") not in (None, "")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if row.get(left, "") not in (None, "") and row.get(right, "") not in (None, "")
            ]
            # LINE-BY-LINE: `xs`에 `[pair[0] for pair in pairs]` 결과를 저장합니다. 의미/사용: `xs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            xs = [pair[0] for pair in pairs]
            # LINE-BY-LINE: `ys`에 `[pair[1] for pair in pairs]` 결과를 저장합니다. 의미/사용: `ys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            ys = [pair[1] for pair in pairs]
            # LINE-BY-LINE: `output.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            output.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `_correlation_rows`에서 반환/저장할 dict의 `variable_x` 키에 `left` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "variable_x": left,
                    # LINE-BY-LINE: `_correlation_rows`에서 반환/저장할 dict의 `variable_y` 키에 `right` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "variable_y": right,
                    # LINE-BY-LINE: `_correlation_rows`에서 반환/저장할 dict의 `count` 키에 `len(pairs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "count": len(pairs),
                    # LINE-BY-LINE: `_correlation_rows`에서 반환/저장할 dict의 `pearson_corr` 키에 `_pearson(xs, ys)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "pearson_corr": _pearson(xs, ys),
                }
            )
    # LINE-BY-LINE: 호출자에게 `output`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return output


# LINE-BY-LINE: `_same_time_group_rows(rows: List[Dict[str, Any]], min_group_size: int = 2)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _same_time_group_rows(rows: List[Dict[str, Any]], min_group_size: int = 2) -> List[Dict[str, Any]]:
    """같은 설비/같은 착수/같은 종료 timestamp group summary를 만든다."""

    # LINE-BY-LINE: `grouped` 변수에 `defaultdict(list)` 결과를 저장합니다. 의미: `grouped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `grouped[str(row["same_machine_time_group_id"])].append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        grouped[str(row["same_machine_time_group_id"])].append(row)

    # LINE-BY-LINE: `output`에 `[]` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output = []
    # LINE-BY-LINE: `group_id, group_rows in sorted(grouped.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for group_id, group_rows in sorted(grouped.items()):
        # LINE-BY-LINE: 조건 `len(group_rows) < min_group_size`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(group_rows) < min_group_size:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `first`에 `group_rows[0]` 결과를 저장합니다. 의미/사용: `first` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        first = group_rows[0]
        # LINE-BY-LINE: `output.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        output.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `same_machine_time_group_id` 키에 `group_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "same_machine_time_group_id": group_id,
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `first["machine_id"]` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": first["machine_id"],
                # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `first["cut_bay"]` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                "cut_bay": first["cut_bay"],
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `first["actual_start_datetime"]` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": first["actual_start_datetime"],
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `first["actual_end_datetime"]` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": first["actual_end_datetime"],
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `group_size` 키에 `len(group_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "group_size": len(group_rows),
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `first["actual_duration_minutes"]` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": first["actual_duration_minutes"],
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `total_tact_time` 키에 `round(sum(_to_float(row["tact_time"]) for row in group_rows), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "total_tact_time": round(sum(_to_float(row["tact_time"]) for row in group_rows), 6),
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `window_to_total_tact_ratio` 키에 `first["same_machine_time_group_window_to_total_tact_ratio"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "window_to_total_tact_ratio": first["same_machine_time_group_window_to_total_tact_ratio"],
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `equal_share_duration` 키에 `first["same_machine_time_group_equal_share_duration"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "equal_share_duration": first["same_machine_time_group_equal_share_duration"],
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `mean_abs_gap_minutes` 키에 `round(statistics.mean(_to_float(row["abs_gap_minutes"]) for row in group_rows), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "mean_abs_gap_minutes": round(statistics.mean(_to_float(row["abs_gap_minutes"]) for row in group_rows), 6),
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `max_abs_gap_minutes` 키에 `round(max(_to_float(row["abs_gap_minutes"]) for row in group_rows), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "max_abs_gap_minutes": round(max(_to_float(row["abs_gap_minutes"]) for row in group_rows), 6),
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `mean_learning_target_abs_gap_to_tact` 키에 `round(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "mean_learning_target_abs_gap_to_tact": round(
                    # LINE-BY-LINE: `round(...)` 호출에 `statistics.mean(_to_float(row["learning_target_candidate_abs_gap_to_tact"]) for row in group_rows)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    statistics.mean(_to_float(row["learning_target_candidate_abs_gap_to_tact"]) for row in group_rows),
                    # LINE-BY-LINE: `round(...)` 호출에 `6` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                    6,
                ),
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `work_order_nos` 키에 `"|".join(str(row["work_order_no"]) for row in group_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "work_order_nos": "|".join(str(row["work_order_no"]) for row in group_rows),
                # LINE-BY-LINE: `_same_time_group_rows`에서 반환/저장할 dict의 `job_ids` 키에 `"|".join(str(row["job_id"]) for row in group_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "job_ids": "|".join(str(row["job_id"]) for row in group_rows),
            }
        )
    # LINE-BY-LINE: 호출자에게 `sorted(output, key=lambda row: (-int(row["group_size"]), row["machine_id"], row["actual_start_dat...`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(output, key=lambda row: (-int(row["group_size"]), row["machine_id"], row["actual_start_datetime"]))


# LINE-BY-LINE: `_top_gap_rows(rows: List[Dict[str, Any]], limit: int = 50)` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _top_gap_rows(rows: List[Dict[str, Any]], limit: int = 50) -> List[Dict[str, Any]]:
    """TACT_TIME과 실적 elapsed 차이가 큰 row 상위 목록을 반환한다."""

    # LINE-BY-LINE: 호출자에게 `sorted(rows, key=lambda row: (-_to_float(row["abs_gap_minutes"]), row["job_id"]))[:limit]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(rows, key=lambda row: (-_to_float(row["abs_gap_minutes"]), row["job_id"]))[:limit]


# LINE-BY-LINE: `_target_candidate_metric_row(rows: List[Dict[str, Any]], target_key: str, target_name: str)` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _target_candidate_metric_row(rows: List[Dict[str, Any]], target_key: str, target_name: str) -> Dict[str, Any]:
    """학습 target 후보와 TACT/actual 사이의 오차 지표를 계산한다."""

    # LINE-BY-LINE: `target_values`에 `[_to_float(row[target_key]) for row in rows]` 결과를 저장합니다. 의미/사용: `target_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_values = [_to_float(row[target_key]) for row in rows]
    # LINE-BY-LINE: `tact_values`에 `[_to_float(row["tact_time"]) for row in rows]` 결과를 저장합니다. 의미/사용: `tact_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_values = [_to_float(row["tact_time"]) for row in rows]
    # LINE-BY-LINE: `gaps`에 `[target - tact for target, tact in zip(target_values, tact_values)]` 결과를 저장합니다. 의미/사용: `gaps` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    gaps = [target - tact for target, tact in zip(target_values, tact_values)]
    # LINE-BY-LINE: `abs_gaps`에 `[abs(gap) for gap in gaps]` 결과를 저장합니다. 의미/사용: `abs_gaps` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    abs_gaps = [abs(gap) for gap in gaps]
    # LINE-BY-LINE: `mape_values`에 `[` 결과를 저장합니다. 의미/사용: `mape_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    mape_values = [
        # LINE-BY-LINE: `abs(target - tact) / target`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        abs(target - tact) / target
        # LINE-BY-LINE: `target, tact in zip(target_values, tact_values)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for target, tact in zip(target_values, tact_values)
        # LINE-BY-LINE: 조건 `target > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if target > 0
    ]
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_name` 키에 `target_name` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_name": target_name,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_key` 키에 `target_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_key": target_key,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "count": len(rows),
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_mean` 키에 `statistics.mean(target_values) if target_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_mean": statistics.mean(target_values) if target_values else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_median` 키에 `statistics.median(target_values) if target_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_median": statistics.median(target_values) if target_values else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_vs_tact_mae_minutes` 키에 `statistics.mean(abs_gaps) if abs_gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_vs_tact_mae_minutes": statistics.mean(abs_gaps) if abs_gaps else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_vs_tact_bias_minutes` 키에 `statistics.mean(gaps) if gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_vs_tact_bias_minutes": statistics.mean(gaps) if gaps else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_vs_tact_rmse_minutes` 키에 `math.sqrt(statistics.mean(gap**2 for gap in gaps)) if gaps else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_vs_tact_rmse_minutes": math.sqrt(statistics.mean(gap**2 for gap in gaps)) if gaps else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `target_vs_tact_mape_percent` 키에 `statistics.mean(mape_values) * 100.0 if mape_values else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_vs_tact_mape_percent": statistics.mean(mape_values) * 100.0 if mape_values else None,
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `corr_target_with_tact_time` 키에 `_pearson(target_values, tact_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "corr_target_with_tact_time": _pearson(target_values, tact_values),
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `corr_target_with_cut_length` 키에 `_pearson(target_values, [_to_float(row["cut_length"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "corr_target_with_cut_length": _pearson(target_values, [_to_float(row["cut_length"]) for row in rows]),
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `corr_target_with_part_count` 키에 `_pearson(target_values, [_to_float(row["part_count"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "corr_target_with_part_count": _pearson(target_values, [_to_float(row["part_count"]) for row in rows]),
        # LINE-BY-LINE: `_target_candidate_metric_row`에서 반환/저장할 dict의 `corr_target_with_group_size` 키에 `_pearson(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "corr_target_with_group_size": _pearson(
            # LINE-BY-LINE: `_pearson(...)` 호출에 `target_values` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            target_values,
            # LINE-BY-LINE: `_pearson(...)` 호출에 `[_to_float(row["same_machine_time_group_size"]) for row in rows]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            [_to_float(row["same_machine_time_group_size"]) for row in rows],
        ),
    }


# LINE-BY-LINE: `_target_candidate_metrics(rows: List[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _target_candidate_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """여러 target 후보의 성능 지표를 한 번에 만든다."""

    # LINE-BY-LINE: 호출자에게 list 반환을 시작합니다. 사용: 여러 row 또는 후보 값을 순서 있는 목록으로 전달합니다.
    return [
        # LINE-BY-LINE: `현재 표현식(...)` 호출에 `_target_candidate_metric_row(rows, "actual_duration_minutes", "observed_elapsed_for_des_replay")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        _target_candidate_metric_row(rows, "actual_duration_minutes", "observed_elapsed_for_des_replay"),
        # LINE-BY-LINE: `_target_candidate_metric_row(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        _target_candidate_metric_row(
            # LINE-BY-LINE: `_target_candidate_metric_row(...)` 호출에 `rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            rows,
            # LINE-BY-LINE: 문자열 값 `"same_machine_time_group_equal_share_duration"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "same_machine_time_group_equal_share_duration",
            # LINE-BY-LINE: 문자열 값 `"batch_window_equal_share_candidate"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "batch_window_equal_share_candidate",
        ),
        # LINE-BY-LINE: `_target_candidate_metric_row(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        _target_candidate_metric_row(
            # LINE-BY-LINE: `_target_candidate_metric_row(...)` 호출에 `rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            rows,
            # LINE-BY-LINE: 문자열 값 `"same_machine_time_group_tact_weighted_duration"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "same_machine_time_group_tact_weighted_duration",
            # LINE-BY-LINE: 문자열 값 `"batch_window_tact_weighted_candidate"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "batch_window_tact_weighted_candidate",
        ),
        # LINE-BY-LINE: `_target_candidate_metric_row(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        _target_candidate_metric_row(
            # LINE-BY-LINE: `_target_candidate_metric_row(...)` 호출에 `rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            rows,
            # LINE-BY-LINE: 문자열 값 `"learning_target_candidate_duration"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "learning_target_candidate_duration",
            # LINE-BY-LINE: 문자열 값 `"recommended_learning_target_candidate"`를 `_target_candidate_metric_row(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "recommended_learning_target_candidate",
        ),
    ]


# LINE-BY-LINE: `_batch_window_issue_rows(rows: List[Dict[str, Any]])` 함수를 정의합니다. 반환 타입: `List[Dict[str, Any]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _batch_window_issue_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """batch/window 의심 group을 사람이 검토하기 좋은 row로 만든다."""

    # LINE-BY-LINE: `issues`에 `[]` 결과를 저장합니다. 의미/사용: `issues` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    issues = []
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `group_size`에 `int(row["same_machine_time_group_size"])` 결과를 저장합니다. 의미/사용: `group_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        group_size = int(row["same_machine_time_group_size"])
        # LINE-BY-LINE: `window_ratio`에 `row["same_machine_time_group_window_to_total_tact_ratio"]` 결과를 저장합니다. 의미/사용: `window_ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        window_ratio = row["same_machine_time_group_window_to_total_tact_ratio"]
        # LINE-BY-LINE: 조건 `group_size <= 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if group_size <= 1:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `issue_type`에 `"batch_window_same_timestamp"` 결과를 저장합니다. 의미/사용: `issue_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        issue_type = "batch_window_same_timestamp"
        # LINE-BY-LINE: 조건 `window_ratio != "" and float(window_ratio) > 1.5`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if window_ratio != "" and float(window_ratio) > 1.5:
            # LINE-BY-LINE: `issue_type`에 `"batch_window_duration_exceeds_total_tact"` 결과를 저장합니다. 의미/사용: `issue_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            issue_type = "batch_window_duration_exceeds_total_tact"
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `window_ratio != "" and float(window_ratio) < 0.5`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif window_ratio != "" and float(window_ratio) < 0.5:
            # LINE-BY-LINE: `issue_type`에 `"batch_window_duration_below_total_tact"` 결과를 저장합니다. 의미/사용: `issue_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            issue_type = "batch_window_duration_below_total_tact"
        # LINE-BY-LINE: `issues.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        issues.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `issue_type` 키에 `issue_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "issue_type": issue_type,
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `group_id` 키에 `row["same_machine_time_group_id"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "group_id": row["same_machine_time_group_id"],
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row["job_id"]` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row["job_id"],
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row["work_order_no"]` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row["work_order_no"],
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row["machine_id"]` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row["machine_id"],
                # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `row["cut_bay"]` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                "cut_bay": row["cut_bay"],
                # LINE-BY-LINE: 딕셔너리 키 `actual_start_datetime`에는 `row["actual_start_datetime"]` 값을 넣습니다. 의미: 실적 착수시간 원본입니다. 예: `202602260906`, 사용: actual replay 시작 시각.
                "actual_start_datetime": row["actual_start_datetime"],
                # LINE-BY-LINE: 딕셔너리 키 `actual_end_datetime`에는 `row["actual_end_datetime"]` 값을 넣습니다. 의미: 실적 종료시간 원본입니다. 예: `202602261015`, 사용: actual replay 종료 시각.
                "actual_end_datetime": row["actual_end_datetime"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `group_size` 키에 `group_size` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "group_size": group_size,
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `window_duration` 키에 `row["same_machine_time_group_window_duration"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "window_duration": row["same_machine_time_group_window_duration"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `group_total_tact` 키에 `row["same_machine_time_group_total_tact"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "group_total_tact": row["same_machine_time_group_total_tact"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `window_to_total_tact_ratio` 키에 `window_ratio` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "window_to_total_tact_ratio": window_ratio,
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `original_actual_duration` 키에 `row["actual_duration_minutes"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "original_actual_duration": row["actual_duration_minutes"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `tact_time` 키에 `row["tact_time"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time": row["tact_time"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `equal_share_duration` 키에 `row["same_machine_time_group_equal_share_duration"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "equal_share_duration": row["same_machine_time_group_equal_share_duration"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `tact_weighted_duration` 키에 `row["same_machine_time_group_tact_weighted_duration"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_weighted_duration": row["same_machine_time_group_tact_weighted_duration"],
                # LINE-BY-LINE: `_batch_window_issue_rows`에서 반환/저장할 dict의 `recommended_learning_target` 키에 `row["learning_target_candidate_duration"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "recommended_learning_target": row["learning_target_candidate_duration"],
            }
        )
    # LINE-BY-LINE: 호출자에게 `sorted(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return sorted(
        # LINE-BY-LINE: `sorted(...)` 호출에 `issues` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        issues,
        # LINE-BY-LINE: `key`에 `lambda row: (` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        key=lambda row: (
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row["group_id"])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row["group_id"]),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row["actual_start_datetime"])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row["actual_start_datetime"]),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `str(row["work_order_no"])` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            str(row["work_order_no"]),
        ),
    )


# LINE-BY-LINE: `_excluded_np_summary(excluded_csv_path: Optional[str])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _excluded_np_summary(excluded_csv_path: Optional[str]) -> Dict[str, Any]:
    """NP 제외 로그가 있으면 제외 사유별 count를 요약한다."""

    # LINE-BY-LINE: 조건 `not excluded_csv_path`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not excluded_csv_path:
        # LINE-BY-LINE: 호출자에게 `{"provided": False}`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return {"provided": False}
    # LINE-BY-LINE: `path`에 `Path(excluded_csv_path)` 결과를 저장합니다. 의미/사용: `path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path = Path(excluded_csv_path)
    # LINE-BY-LINE: 조건 `not path.exists()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not path.exists():
        # LINE-BY-LINE: 콘솔에 `print(f"[ERROR][tact_gap_analysis._excluded_np_summary] cause=missing_file path={excluded_csv_pat...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"[ERROR][tact_gap_analysis._excluded_np_summary] cause=missing_file path={excluded_csv_path}")
        # LINE-BY-LINE: `FileNotFoundError(excluded_csv_path)` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise FileNotFoundError(excluded_csv_path)

    # LINE-BY-LINE: `path.open("r", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `rows`에 `list(csv.DictReader(handle))` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
        rows = list(csv.DictReader(handle))

    # LINE-BY-LINE: `np_rows`에 `[row for row in rows if str(row.get("series", "")) == "NP"]` 결과를 저장합니다. 의미/사용: `np_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    np_rows = [row for row in rows if str(row.get("series", "")) == "NP"]
    # LINE-BY-LINE: `reason_counts`에 `Counter(row.get("exclude_reason", "") for row in np_rows)` 결과를 저장합니다. 의미/사용: `reason_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    reason_counts = Counter(row.get("exclude_reason", "") for row in np_rows)
    # LINE-BY-LINE: `bay_counts`에 `Counter(str(row.get("source_cut_bay", "")) for row in np_rows)` 결과를 저장합니다. 의미/사용: `bay_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_counts = Counter(str(row.get("source_cut_bay", "")) for row in np_rows)
    # LINE-BY-LINE: `zero_duration_rows`에 `[` 결과를 저장합니다. 의미/사용: `zero_duration_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    zero_duration_rows = [
        # LINE-BY-LINE: `row for row in np_rows` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        row for row in np_rows
        # LINE-BY-LINE: 조건 `row.get("exclude_reason") in {"non_positive_actual_duration", "negative_actual_duration"}`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if row.get("exclude_reason") in {"non_positive_actual_duration", "negative_actual_duration"}
    ]
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `provided` 키에 `True` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "provided": True,
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `path` 키에 `str(path)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "path": str(path),
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `excluded_np_count` 키에 `len(np_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "excluded_np_count": len(np_rows),
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `excluded_np_reason_counts` 키에 `dict(sorted(reason_counts.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "excluded_np_reason_counts": dict(sorted(reason_counts.items())),
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `excluded_np_bay_counts` 키에 `dict(sorted(bay_counts.items()))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "excluded_np_bay_counts": dict(sorted(bay_counts.items())),
        # LINE-BY-LINE: `_excluded_np_summary`에서 반환/저장할 dict의 `zero_or_non_positive_duration_count` 키에 `len(zero_duration_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "zero_or_non_positive_duration_count": len(zero_duration_rows),
    }


# LINE-BY-LINE: `build_tact_gap_analysis` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def build_tact_gap_analysis(
    # LINE-BY-LINE: `scenario`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario: Dict[str, Any],
    # LINE-BY-LINE: `output_dir`를 `str,` 타입으로 선언합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir: str,
    # LINE-BY-LINE: `excluded_csv_path` 변수에 `None` 결과를 저장합니다. 의미: `excluded_csv_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    excluded_csv_path: Optional[str] = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """TACT_TIME과 actual elapsed 차이의 규칙성을 분석한다."""

    # LINE-BY-LINE: `output_path`에 `Path(output_dir)` 결과를 저장합니다. 의미/사용: `output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path = Path(output_dir)
    # LINE-BY-LINE: `output_path.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `rows`에 `_build_base_rows(scenario)` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = _build_base_rows(scenario)
    # LINE-BY-LINE: 조건 `not rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows:
        # LINE-BY-LINE: 콘솔에 `print("[ERROR][tact_gap_analysis.build_tact_gap_analysis] cause=no_jobs")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[ERROR][tact_gap_analysis.build_tact_gap_analysis] cause=no_jobs")
        # LINE-BY-LINE: `ValueError("scenario has no jobs")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError("scenario has no jobs")

    # LINE-BY-LINE: `variables`에 `[` 결과를 저장합니다. 의미/사용: `variables` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    variables = [
        # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "actual_duration_minutes",
        # LINE-BY-LINE: 문자열 값 `"tact_time"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "tact_time",
        # LINE-BY-LINE: 문자열 값 `"actual_minus_tact"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "actual_minus_tact",
        # LINE-BY-LINE: 문자열 값 `"abs_gap_minutes"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "abs_gap_minutes",
        # LINE-BY-LINE: 문자열 값 `"learning_target_candidate_duration"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "learning_target_candidate_duration",
        # LINE-BY-LINE: 문자열 값 `"plate_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "plate_length",
        # LINE-BY-LINE: 문자열 값 `"thickness"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "thickness",
        # LINE-BY-LINE: 문자열 값 `"cut_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "cut_length",
        # LINE-BY-LINE: 문자열 값 `"mark_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "mark_length",
        # LINE-BY-LINE: 문자열 값 `"bevel_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "bevel_length",
        # LINE-BY-LINE: 문자열 값 `"part_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "part_count",
        # LINE-BY-LINE: 문자열 값 `"same_machine_time_group_size"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "same_machine_time_group_size",
    ]
    # LINE-BY-LINE: `correlation_rows`에 `_correlation_rows(rows, variables)` 결과를 저장합니다. 의미/사용: `correlation_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    correlation_rows = _correlation_rows(rows, variables)
    # LINE-BY-LINE: `same_time_groups`에 `_same_time_group_rows(rows)` 결과를 저장합니다. 의미/사용: `same_time_groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    same_time_groups = _same_time_group_rows(rows)
    # LINE-BY-LINE: `target_candidate_metrics`에 `_target_candidate_metrics(rows)` 결과를 저장합니다. 의미/사용: `target_candidate_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_candidate_metrics = _target_candidate_metrics(rows)
    # LINE-BY-LINE: `batch_window_issues`에 `_batch_window_issue_rows(rows)` 결과를 저장합니다. 의미/사용: `batch_window_issues` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    batch_window_issues = _batch_window_issue_rows(rows)
    # LINE-BY-LINE: `summary_rows`에 `[]` 결과를 저장합니다. 의미/사용: `summary_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    summary_rows = []
    # LINE-BY-LINE: `summary_rows.append(_metric_row("overall", "all", rows))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    summary_rows.append(_metric_row("overall", "all", rows))
    # LINE-BY-LINE: `group_name, key in (` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for group_name, key in (
        # LINE-BY-LINE: `in(...)` 호출에 `("cut_bay", "cut_bay")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("cut_bay", "cut_bay"),
        # LINE-BY-LINE: `in(...)` 호출에 `("machine_id", "machine_id")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("machine_id", "machine_id"),
        # LINE-BY-LINE: `in(...)` 호출에 `("actual_day", "actual_day")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("actual_day", "actual_day"),
        # LINE-BY-LINE: `in(...)` 호출에 `("thickness_bucket", "thickness_bucket")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("thickness_bucket", "thickness_bucket"),
        # LINE-BY-LINE: `in(...)` 호출에 `("length_bucket", "length_bucket")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("length_bucket", "length_bucket"),
        # LINE-BY-LINE: `in(...)` 호출에 `("cut_length_bucket", "cut_length_bucket")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("cut_length_bucket", "cut_length_bucket"),
        # LINE-BY-LINE: `in(...)` 호출에 `("same_machine_time_group_bucket", "same_machine_time_group_bucket")` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        ("same_machine_time_group_bucket", "same_machine_time_group_bucket"),
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        # LINE-BY-LINE: `summary_rows.extend(_group_summary(rows, group_name, key))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        summary_rows.extend(_group_summary(rows, group_name, key))

    # LINE-BY-LINE: `rows_in_same_time_groups`에 `sum(1 for row in rows if int(row["same_machine_time_group_size"]) >= 2)` 결과를 저장합니다. 의미/사용: `rows_in_same_time_groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rows_in_same_time_groups = sum(1 for row in rows if int(row["same_machine_time_group_size"]) >= 2)
    # LINE-BY-LINE: `rows_in_large_groups`에 `sum(1 for row in rows if int(row["same_machine_time_group_size"]) > 3)` 결과를 저장합니다. 의미/사용: `rows_in_large_groups` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rows_in_large_groups = sum(1 for row in rows if int(row["same_machine_time_group_size"]) > 3)
    # LINE-BY-LINE: `overall`에 `summary_rows[0]` 결과를 저장합니다. 의미/사용: `overall` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    overall = summary_rows[0]
    # LINE-BY-LINE: `tact_actual_corr`에 `next(` 결과를 저장합니다. 의미/사용: `tact_actual_corr` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_actual_corr = next(
        # LINE-BY-LINE: `row["pearson_corr"]` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        row["pearson_corr"]
        # LINE-BY-LINE: `row in correlation_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in correlation_rows
        # LINE-BY-LINE: 조건 `row["variable_x"] == "tact_time" and row["variable_y"] == "actual_duration_minutes"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if row["variable_x"] == "tact_time" and row["variable_y"] == "actual_duration_minutes"
    )
    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `record_count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "record_count": len(rows),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `clean_positive_elapsed_record_count` 키에 `len([row for row in rows if _to_float(row["actual_duration_minutes"]) > 0])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "clean_positive_elapsed_record_count": len([row for row in rows if _to_float(row["actual_duration_minutes"]) > 0]),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `tact_actual_pearson_corr` 키에 `tact_actual_corr` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_actual_pearson_corr": tact_actual_corr,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `overall` 키에 `overall` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "overall": overall,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `actual_duration_stats` 키에 `_summary_stats([_to_float(row["actual_duration_minutes"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "actual_duration_stats": _summary_stats([_to_float(row["actual_duration_minutes"]) for row in rows]),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `tact_time_stats` 키에 `_summary_stats([_to_float(row["tact_time"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_time_stats": _summary_stats([_to_float(row["tact_time"]) for row in rows]),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `gap_actual_minus_tact_stats` 키에 `_summary_stats([_to_float(row["actual_minus_tact"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "gap_actual_minus_tact_stats": _summary_stats([_to_float(row["actual_minus_tact"]) for row in rows]),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `abs_gap_stats` 키에 `_summary_stats([_to_float(row["abs_gap_minutes"]) for row in rows])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "abs_gap_stats": _summary_stats([_to_float(row["abs_gap_minutes"]) for row in rows]),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_machine_time_group_count_min_2` 키에 `len(same_time_groups)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_machine_time_group_count_min_2": len(same_time_groups),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_machine_time_group_rows_min_2` 키에 `rows_in_same_time_groups` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_machine_time_group_rows_min_2": rows_in_same_time_groups,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_machine_time_group_rows_min_2_rate` 키에 `_pct(rows_in_same_time_groups, len(rows))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_machine_time_group_rows_min_2_rate": _pct(rows_in_same_time_groups, len(rows)),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_machine_time_group_rows_over_3` 키에 `rows_in_large_groups` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_machine_time_group_rows_over_3": rows_in_large_groups,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_machine_time_group_rows_over_3_rate` 키에 `_pct(rows_in_large_groups, len(rows))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_machine_time_group_rows_over_3_rate": _pct(rows_in_large_groups, len(rows)),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `max_same_machine_time_group_size` 키에 `max((int(row["same_machine_time_group_size"]) for row in rows), default=0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "max_same_machine_time_group_size": max((int(row["same_machine_time_group_size"]) for row in rows), default=0),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `target_candidate_metrics` 키에 `target_candidate_metrics` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target_candidate_metrics": target_candidate_metrics,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `batch_window_issue_count` 키에 `len(batch_window_issues)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "batch_window_issue_count": len(batch_window_issues),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `excluded_np_summary` 키에 `_excluded_np_summary(excluded_csv_path)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "excluded_np_summary": _excluded_np_summary(excluded_csv_path),
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `interpretation` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "interpretation": {
            # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `actual_elapsed_definition` 키에 `"RT_CUT_ED_DTM - RT_CUT_ST_DTM; observed machine time interval, not guaranteed pure cutting time."` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_elapsed_definition": "RT_CUT_ED_DTM - RT_CUT_ST_DTM; observed machine time interval, not guaranteed pure cutting time.",
            # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `tact_time_definition_inferred` 키에 `"TACT_TIME behaves like a planning/standard processing-time candidate; it is not identical to obs...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "tact_time_definition_inferred": "TACT_TIME behaves like a planning/standard processing-time candidate; it is not identical to observed elapsed timestamp difference.",
            # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `rule_signal` 키에 `"If same machine has identical start/end for multiple W/O, actual elapsed is likely a shared batc...` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rule_signal": "If same machine has identical start/end for multiple W/O, actual elapsed is likely a shared batch/window timestamp or input pattern.",
            # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `recommended_learning_target` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "recommended_learning_target": (
                # LINE-BY-LINE: 문자열 값 `"Use observed elapsed for single rows. For same-machine/same-start/same-end groups, "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "Use observed elapsed for single rows. For same-machine/same-start/same-end groups, "
                # LINE-BY-LINE: 문자열 값 `"use tact-weighted allocation as a diagnostic learning target candidate until 현업 confirms batch s...`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "use tact-weighted allocation as a diagnostic learning target candidate until 현업 confirms batch semantics."
            ),
        },
    }

    # LINE-BY-LINE: `_write_csv(rows, output_path / "tact_gap_rows.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(rows, output_path / "tact_gap_rows.csv")
    # LINE-BY-LINE: `_write_csv(summary_rows, output_path / "tact_gap_group_summary.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(summary_rows, output_path / "tact_gap_group_summary.csv")
    # LINE-BY-LINE: `_write_csv(correlation_rows, output_path / "tact_gap_correlations.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(correlation_rows, output_path / "tact_gap_correlations.csv")
    # LINE-BY-LINE: `_write_csv(same_time_groups, output_path / "same_machine_time_groups.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(same_time_groups, output_path / "same_machine_time_groups.csv")
    # LINE-BY-LINE: `_write_csv(target_candidate_metrics, output_path / "target_candidate_metrics.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(target_candidate_metrics, output_path / "target_candidate_metrics.csv")
    # LINE-BY-LINE: `_write_csv(batch_window_issues, output_path / "batch_window_issue_rows.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(batch_window_issues, output_path / "batch_window_issue_rows.csv")
    # LINE-BY-LINE: `_write_csv(_top_gap_rows(rows), output_path / "top_abs_gap_rows.csv")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_csv(_top_gap_rows(rows), output_path / "top_abs_gap_rows.csv")
    # LINE-BY-LINE: `_write_json(summary, output_path / "tact_gap_summary.json")`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    _write_json(summary, output_path / "tact_gap_summary.json")
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `summary` 키에 `summary` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "summary": summary,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `summary_rows` 키에 `summary_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "summary_rows": summary_rows,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `correlation_rows` 키에 `correlation_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "correlation_rows": correlation_rows,
        # LINE-BY-LINE: `build_tact_gap_analysis`에서 반환/저장할 dict의 `same_time_groups` 키에 `same_time_groups` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "same_time_groups": same_time_groups,
    }
