"""택트타임 분석/추정.

이 파일의 책임:
- 제공 `TACT_TIME`과 실적 elapsed time의 차이를 정량화한다.
- CUT_LTH, THK, PTLST_QTY 등 feature로 선형 처리시간 후보식을 만든다.
- 전체/holdout/segment별 MAE, RMSE, MAPE를 계산한다.

중요:
- 이 산식은 actual replay identity 검증용이 아니다.
- actual replay는 항상 원본 `실적 종료시간 - 실적 착수시간`을 그대로 사용한다.
- 산식은 generated simulation, 학습데이터 생성, 휴리스틱 비교용 처리시간 후보로 사용한다.
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass
# LINE-BY-LINE: `math` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import math
# LINE-BY-LINE: `statistics` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import statistics
# LINE-BY-LINE: `typing` 모듈에서 `Any, Dict, Iterable, List, Optional`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Any, Dict, Iterable, List, Optional


# LINE-BY-LINE: `TACT_FEATURE_KEYS`에 `[` 결과를 저장합니다. 의미/사용: `TACT_FEATURE_KEYS` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
TACT_FEATURE_KEYS = [
    # LINE-BY-LINE: 문자열 값 `"cut_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "cut_length",
    # LINE-BY-LINE: 문자열 값 `"mark_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "mark_length",
    # LINE-BY-LINE: 문자열 값 `"bevel_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "bevel_length",
    # LINE-BY-LINE: 문자열 값 `"plate_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "plate_length",
    # LINE-BY-LINE: 문자열 값 `"thickness"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "thickness",
    # LINE-BY-LINE: 문자열 값 `"part_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "part_count",
    # LINE-BY-LINE: 문자열 값 `"steel_qty"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "steel_qty",
]

# LINE-BY-LINE: `CORE_TACT_FEATURE_KEYS`에 `[` 결과를 저장합니다. 의미/사용: `CORE_TACT_FEATURE_KEYS` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
CORE_TACT_FEATURE_KEYS = [
    # LINE-BY-LINE: 문자열 값 `"cut_length"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "cut_length",
    # LINE-BY-LINE: 문자열 값 `"thickness"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "thickness",
    # LINE-BY-LINE: 문자열 값 `"part_count"`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
    "part_count",
]


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `TactTimeRecord` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
class TactTimeRecord:
    """실적 데이터 1건을 표현하는 예시 자료형."""

    # LINE-BY-LINE: `machine_type`를 `str` 타입으로 선언합니다. 의미/사용: `TactTimeRecord.machine_type` 필드/속성입니다. 사용: TactTimeRecord 객체를 만들거나 이후 로직에서 참조합니다.
    machine_type: str
    # LINE-BY-LINE: `family`를 `str` 타입으로 선언합니다. 의미/사용: `TactTimeRecord.family`는 계열 값입니다. 예: `NP`, 사용: 설비 eligibility와 priority 판단.
    family: str
    # LINE-BY-LINE: `thickness`를 `float` 타입으로 선언합니다. 의미/사용: `TactTimeRecord.thickness`는 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
    thickness: float
    # LINE-BY-LINE: `plate_length`를 `float` 타입으로 선언합니다. 의미/사용: `TactTimeRecord.plate_length`는 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
    plate_length: float
    # LINE-BY-LINE: `actual_minutes`를 `float` 타입으로 선언합니다. 의미/사용: `TactTimeRecord.actual_minutes` 필드/속성입니다. 사용: TactTimeRecord 객체를 만들거나 이후 로직에서 참조합니다.
    actual_minutes: float


# LINE-BY-LINE: `TactTimeEstimator` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
class TactTimeEstimator:
    """택트타임 추정기.

    현재는 간단한 통계 기반 placeholder이지만,
    인터페이스는 실제 추정기처럼 유지합니다.
    """

    # LINE-BY-LINE: `__init__(self)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
    def __init__(self):
        """추정기 내부 상태를 초기화한다."""

        # 실제 구현에서는 여기서 통계량, 회귀계수, 모델 객체 등을 보관하면 됩니다.
        # LINE-BY-LINE: 현재 객체의 `fitted` 속성에 `False` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.fitted = False
        # LINE-BY-LINE: 현재 객체의 `summary` 속성에 `{}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.summary = {}

    # LINE-BY-LINE: `fit(self, records: Iterable[TactTimeRecord])` 함수를 정의합니다. 반환 타입: `None`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
    def fit(self, records: Iterable[TactTimeRecord]) -> None:
        """실적 데이터를 받아 기본 통계량을 계산합니다.

        현재는 평균 수준의 요약만 저장합니다.
        """

        # LINE-BY-LINE: `records`에 `list(records)` 결과를 저장합니다. 의미/사용: `records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        records = list(records)
        # LINE-BY-LINE: 조건 `not records`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not records:
            # LINE-BY-LINE: 현재 객체의 `summary` 속성에 `{"count": 0}` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.summary = {"count": 0}
            # LINE-BY-LINE: 현재 객체의 `fitted` 속성에 `True` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
            self.fitted = True
            # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
            return

        # LINE-BY-LINE: `avg_minutes`에 `sum(record.actual_minutes for record in records) / len(records)` 결과를 저장합니다. 의미/사용: `avg_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        avg_minutes = sum(record.actual_minutes for record in records) / len(records)
        # LINE-BY-LINE: 현재 객체의 `summary` 속성에 `{` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.summary = {
            # LINE-BY-LINE: `fit`에서 반환/저장할 dict의 `count` 키에 `len(records)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "count": len(records),
            # LINE-BY-LINE: `fit`에서 반환/저장할 dict의 `average_actual_minutes` 키에 `avg_minutes` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "average_actual_minutes": avg_minutes,
        }
        # LINE-BY-LINE: 현재 객체의 `fitted` 속성에 `True` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.fitted = True

    # LINE-BY-LINE: `estimate(self, job: Dict, machine: Dict)` 함수를 정의합니다. 반환 타입: `float`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
    def estimate(self, job: Dict, machine: Dict) -> float:
        """작업-설비 조합의 처리시간을 추정합니다.

        현재는 실제 모델이 없으므로 매우 단순한 placeholder만 둡니다.
        향후에는 아래 항목을 반영하면 됩니다.
        - 장비 타입
        - 계열
        - 두께
        - 길이 / 면적
        - 셋업 조건
        """

        # LINE-BY-LINE: `base_minutes`에 `float(job.get("base_cut_minutes", 60.0))` 결과를 저장합니다. 의미/사용: `base_minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_minutes = float(job.get("base_cut_minutes", 60.0))
        # LINE-BY-LINE: `speed_factor`에 `float(machine.get("cut_speed_factor", 1.0))` 결과를 저장합니다. 의미/사용: `speed_factor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        speed_factor = float(machine.get("cut_speed_factor", 1.0))
        # LINE-BY-LINE: 호출자에게 `base_minutes * speed_factor`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return base_minutes * speed_factor


# LINE-BY-LINE: `_to_float(value: Any, default: float = 0.0)` 함수를 정의합니다. 반환 타입: `float`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _to_float(value: Any, default: float = 0.0) -> float:
    """분석용 숫자 변환 helper.

    현재 tact 분석은 과거 리포트 호환 때문에 default를 허용한다.
    단, DES 환경이나 actual replay 검증에서 이 helper를 무비판적으로 쓰면 안 된다.
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


# LINE-BY-LINE: `_mean_absolute_error(actual: List[float], predicted: List[float])` 함수를 정의합니다. 반환 타입: `float`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _mean_absolute_error(actual: List[float], predicted: List[float]) -> float:
    """MAE, 즉 평균 절대 오차를 계산한다."""

    # LINE-BY-LINE: 호출자에게 `statistics.mean(abs(a - p) for a, p in zip(actual, predicted))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return statistics.mean(abs(a - p) for a, p in zip(actual, predicted))


# LINE-BY-LINE: `_root_mean_squared_error(actual: List[float], predicted: List[float])` 함수를 정의합니다. 반환 타입: `float`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _root_mean_squared_error(actual: List[float], predicted: List[float]) -> float:
    """RMSE, 즉 평균제곱근오차를 계산한다."""

    # LINE-BY-LINE: 호출자에게 `math.sqrt(statistics.mean((a - p) ** 2 for a, p in zip(actual, predicted)))`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return math.sqrt(statistics.mean((a - p) ** 2 for a, p in zip(actual, predicted)))


# LINE-BY-LINE: `_mean_absolute_percentage_error(actual: List[float], predicted: List[float])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _mean_absolute_percentage_error(actual: List[float], predicted: List[float]) -> Optional[float]:
    """MAPE를 percent 단위로 계산한다. actual이 0인 row는 제외한다."""

    # LINE-BY-LINE: `values`에 `[abs(a - p) / a for a, p in zip(actual, predicted) if a > 0]` 결과를 저장합니다. 의미/사용: `values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    values = [abs(a - p) / a for a, p in zip(actual, predicted) if a > 0]
    # LINE-BY-LINE: 조건 `not values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not values:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: 호출자에게 `statistics.mean(values) * 100.0`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return statistics.mean(values) * 100.0


# LINE-BY-LINE: `_pearson(xs: List[float], ys: List[float])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """두 숫자 배열의 Pearson 상관계수를 계산한다."""

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


# LINE-BY-LINE: `_solve_linear_system(matrix: List[List[float]], vector: List[float])` 함수를 정의합니다. 반환 타입: `List[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    """작은 normal equation을 Gaussian elimination으로 푼다."""

    # LINE-BY-LINE: `n`에 `len(vector)` 결과를 저장합니다. 의미/사용: `n` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    n = len(vector)
    # LINE-BY-LINE: `augmented`에 `[row[:] + [value] for row, value in zip(matrix, vector)]` 결과를 저장합니다. 의미/사용: `augmented` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    # LINE-BY-LINE: `pivot_index in range(n)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for pivot_index in range(n):
        # LINE-BY-LINE: `pivot_row`에 `max(range(pivot_index, n), key=lambda row: abs(augmented[row][pivot_index]))` 결과를 저장합니다. 의미/사용: `pivot_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        pivot_row = max(range(pivot_index, n), key=lambda row: abs(augmented[row][pivot_index]))
        # LINE-BY-LINE: 조건 `abs(augmented[pivot_row][pivot_index]) < 1e-12`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `augmented[pivot_index], augmented[pivot_row]` 여러 변수에 `augmented[pivot_row], augmented[pivot_index]` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        # LINE-BY-LINE: `pivot`에 `augmented[pivot_index][pivot_index]` 결과를 저장합니다. 의미/사용: `pivot` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        pivot = augmented[pivot_index][pivot_index]
        # LINE-BY-LINE: `augmented[pivot_index]`에 `[value / pivot for value in augmented[pivot_index]]` 결과를 저장합니다. 의미/사용: `augmented[pivot_index]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]

        # LINE-BY-LINE: `row_index in range(n)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row_index in range(n):
            # LINE-BY-LINE: 조건 `row_index == pivot_index`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if row_index == pivot_index:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: `factor`에 `augmented[row_index][pivot_index]` 결과를 저장합니다. 의미/사용: `factor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            factor = augmented[row_index][pivot_index]
            # LINE-BY-LINE: `augmented[row_index]`에 `[` 결과를 저장합니다. 의미/사용: `augmented[row_index]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            augmented[row_index] = [
                # LINE-BY-LINE: `value - factor * pivot_value` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                value - factor * pivot_value
                # LINE-BY-LINE: `value, pivot_value in zip(augmented[row_index], augmented[pivot_index])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for value, pivot_value in zip(augmented[row_index], augmented[pivot_index])
            ]

    # LINE-BY-LINE: 호출자에게 `[augmented[index][-1] for index in range(n)]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return [augmented[index][-1] for index in range(n)]


# LINE-BY-LINE: `_non_constant_features(rows: List[Dict[str, float]], feature_keys: List[str])` 함수를 정의합니다. 반환 타입: `tuple[List[str], List[str]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _non_constant_features(rows: List[Dict[str, float]], feature_keys: List[str]) -> tuple[List[str], List[str]]:
    """상수 feature를 회귀식에서 제외한다.

    모든 값이 같은 feature는 normal equation을 불안정하게 만들 수 있다.
    """

    # LINE-BY-LINE: `used`에 `[]` 결과를 저장합니다. 의미/사용: `used` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    used = []
    # LINE-BY-LINE: `dropped`에 `[]` 결과를 저장합니다. 의미/사용: `dropped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    dropped = []
    # LINE-BY-LINE: `feature_key in feature_keys` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for feature_key in feature_keys:
        # LINE-BY-LINE: `values`에 `[row[feature_key] for row in rows]` 결과를 저장합니다. 의미/사용: `values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        values = [row[feature_key] for row in rows]
        # LINE-BY-LINE: 조건 `not values or max(values) - min(values) <= 1e-9`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not values or max(values) - min(values) <= 1e-9:
            # LINE-BY-LINE: `dropped.append(feature_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            dropped.append(feature_key)
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `used.append(feature_key)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            used.append(feature_key)
    # LINE-BY-LINE: 호출자에게 `used, dropped`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return used, dropped


# LINE-BY-LINE: `_fit_linear_formula` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _fit_linear_formula(
    # LINE-BY-LINE: `rows`를 `List[Dict[str, float]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, float]],
    # LINE-BY-LINE: `target_key`를 `str,` 타입으로 선언합니다. 의미/사용: `target_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_key: str,
    # LINE-BY-LINE: `feature_keys`를 `List[str],` 타입으로 선언합니다. 의미/사용: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_keys: List[str],
    # LINE-BY-LINE: `formula_label`를 `str,` 타입으로 선언합니다. 의미/사용: `formula_label` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    formula_label: str,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """TACT_TIME 산출식 후보를 선형식으로 맞춘다."""

    # LINE-BY-LINE: 조건 `not rows`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not rows:
        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `formula` 키에 `""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "formula": "",
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `target` 키에 `target_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "target": target_key,
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `features_used` 키에 `[]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "features_used": [],
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `features_dropped` 키에 `feature_keys` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "features_dropped": feature_keys,
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `coefficients` 키에 `{}` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "coefficients": {},
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `metrics` 키에 `{"count": 0}` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "metrics": {"count": 0},
        }

    # LINE-BY-LINE: `active_features, dropped_features` 여러 변수에 `_non_constant_features(rows, feature_keys)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    active_features, dropped_features = _non_constant_features(rows, feature_keys)
    # LINE-BY-LINE: `names`에 `["intercept", *active_features]` 결과를 저장합니다. 의미/사용: `names` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    names = ["intercept", *active_features]
    # LINE-BY-LINE: `x_rows`에 `[[1.0, *[row[key] for key in active_features]] for row in rows]` 결과를 저장합니다. 의미/사용: `x_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    x_rows = [[1.0, *[row[key] for key in active_features]] for row in rows]
    # LINE-BY-LINE: `y_values`에 `[row[target_key] for row in rows]` 결과를 저장합니다. 의미/사용: `y_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    y_values = [row[target_key] for row in rows]
    # LINE-BY-LINE: `dim`에 `len(names)` 결과를 저장합니다. 의미/사용: `dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    dim = len(names)
    # LINE-BY-LINE: `ridge`에 `1e-8` 결과를 저장합니다. 의미/사용: `ridge` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    ridge = 1e-8

    # LINE-BY-LINE: `xtx`에 `[[0.0 for _ in range(dim)] for _ in range(dim)]` 결과를 저장합니다. 의미/사용: `xtx` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    # LINE-BY-LINE: `xty`에 `[0.0 for _ in range(dim)]` 결과를 저장합니다. 의미/사용: `xty` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    xty = [0.0 for _ in range(dim)]
    # LINE-BY-LINE: `x_row, y_value in zip(x_rows, y_values)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for x_row, y_value in zip(x_rows, y_values):
        # LINE-BY-LINE: `i in range(dim)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for i in range(dim):
            # LINE-BY-LINE: `xty[i]` 값을 `x_row[i] * y_value` 기준으로 누적/증가합니다. 의미/사용: `xty[i]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            xty[i] += x_row[i] * y_value
            # LINE-BY-LINE: `j in range(dim)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for j in range(dim):
                # LINE-BY-LINE: `xtx[i][j]` 값을 `x_row[i] * x_row[j]` 기준으로 누적/증가합니다. 의미/사용: `xtx[i][j]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                xtx[i][j] += x_row[i] * x_row[j]
    # LINE-BY-LINE: `i in range(dim)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for i in range(dim):
        # LINE-BY-LINE: `xtx[i][i]` 값을 `ridge` 기준으로 누적/증가합니다. 의미/사용: `xtx[i][i]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        xtx[i][i] += ridge

    # LINE-BY-LINE: `coefficients`에 `_solve_linear_system(xtx, xty)` 결과를 저장합니다. 의미/사용: `coefficients` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    coefficients = _solve_linear_system(xtx, xty)
    # LINE-BY-LINE: `predictions`에 `[` 결과를 저장합니다. 의미/사용: `predictions` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predictions = [
        # LINE-BY-LINE: `sum(coef * value for coef, value in zip(coefficients, x_row))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        sum(coef * value for coef, value in zip(coefficients, x_row))
        # LINE-BY-LINE: `x_row in x_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for x_row in x_rows
    ]
    # LINE-BY-LINE: `coefficient_map`에 `{name: coef for name, coef in zip(names, coefficients)}` 결과를 저장합니다. 의미/사용: `coefficient_map` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    coefficient_map = {name: coef for name, coef in zip(names, coefficients)}
    # LINE-BY-LINE: `terms`에 `[f"{coefficient_map['intercept']:.6g}"]` 결과를 저장합니다. 의미/사용: `terms` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    terms = [f"{coefficient_map['intercept']:.6g}"]
    # LINE-BY-LINE: `terms.extend(f"{coefficient_map[name]:+.6g}*{name}" for name in active_features)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    terms.extend(f"{coefficient_map[name]:+.6g}*{name}" for name in active_features)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `formula` 키에 `f"{formula_label} = " + " ".join(terms)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "formula": f"{formula_label} = " + " ".join(terms),
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `target` 키에 `target_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target": target_key,
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `features_used` 키에 `active_features` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "features_used": active_features,
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `features_dropped` 키에 `dropped_features` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "features_dropped": dropped_features,
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `coefficients` 키에 `coefficient_map` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "coefficients": coefficient_map,
        # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `metrics` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "metrics": {
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "count": len(rows),
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `mae_minutes` 키에 `_mean_absolute_error(y_values, predictions)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mae_minutes": _mean_absolute_error(y_values, predictions),
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `rmse_minutes` 키에 `_root_mean_squared_error(y_values, predictions)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rmse_minutes": _root_mean_squared_error(y_values, predictions),
            # LINE-BY-LINE: `_fit_linear_formula`에서 반환/저장할 dict의 `mape_percent` 키에 `_mean_absolute_percentage_error(y_values, predictions)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mape_percent": _mean_absolute_percentage_error(y_values, predictions),
        },
    }


# LINE-BY-LINE: `_prediction_metrics` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _prediction_metrics(
    # LINE-BY-LINE: `actual_values`를 `List[float],` 타입으로 선언합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_values: List[float],
    # LINE-BY-LINE: `predicted_values`를 `List[float],` 타입으로 선언합니다. 의미/사용: `predicted_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_values: List[float],
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """예측값과 실제값의 기본 오차 지표를 계산한다."""

    # LINE-BY-LINE: 조건 `not actual_values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not actual_values:
        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `count` 키에 `0` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "count": 0,
            # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `mae_minutes` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mae_minutes": None,
            # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `bias_actual_minus_predicted_minutes` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "bias_actual_minus_predicted_minutes": None,
            # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `rmse_minutes` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rmse_minutes": None,
            # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `mape_percent` 키에 `None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mape_percent": None,
        }
    # LINE-BY-LINE: `errors`에 `[actual - predicted for actual, predicted in zip(actual_values, predicted_values)]` 결과를 저장합니다. 의미/사용: `errors` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    errors = [actual - predicted for actual, predicted in zip(actual_values, predicted_values)]
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `count` 키에 `len(actual_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "count": len(actual_values),
        # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `mae_minutes` 키에 `_mean_absolute_error(actual_values, predicted_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mae_minutes": _mean_absolute_error(actual_values, predicted_values),
        # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `bias_actual_minus_predicted_minutes` 키에 `statistics.mean(errors)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "bias_actual_minus_predicted_minutes": statistics.mean(errors),
        # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `rmse_minutes` 키에 `_root_mean_squared_error(actual_values, predicted_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "rmse_minutes": _root_mean_squared_error(actual_values, predicted_values),
        # LINE-BY-LINE: `_prediction_metrics`에서 반환/저장할 dict의 `mape_percent` 키에 `_mean_absolute_percentage_error(actual_values, predicted_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "mape_percent": _mean_absolute_percentage_error(actual_values, predicted_values),
    }


# LINE-BY-LINE: `_evaluate_formula` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _evaluate_formula(
    # LINE-BY-LINE: `rows`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]],
    # LINE-BY-LINE: `target_key`를 `str,` 타입으로 선언합니다. 의미/사용: `target_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_key: str,
    # LINE-BY-LINE: `formula`를 `Dict[str, Any],` 타입으로 선언합니다. 의미/사용: `formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    formula: Dict[str, Any],
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """선형식 1개를 row 목록에 적용하고 row별 오차를 반환한다."""

    # LINE-BY-LINE: `actual_values`에 `[]` 결과를 저장합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_values = []
    # LINE-BY-LINE: `predicted_values`에 `[]` 결과를 저장합니다. 의미/사용: `predicted_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    predicted_values = []
    # LINE-BY-LINE: `output_rows`에 `[]` 결과를 저장합니다. 의미/사용: `output_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_rows = []
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `prediction`에 `_predict_linear(row, formula)` 결과를 저장합니다. 의미/사용: `prediction` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        prediction = _predict_linear(row, formula)
        # LINE-BY-LINE: 조건 `prediction is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if prediction is None:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `actual_value`에 `_to_float(row.get(target_key))` 결과를 저장합니다. 의미/사용: `actual_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_value = _to_float(row.get(target_key))
        # LINE-BY-LINE: 조건 `actual_value <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_value <= 0:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `error`에 `actual_value - prediction` 결과를 저장합니다. 의미/사용: `error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        error = actual_value - prediction
        # LINE-BY-LINE: `pct_error`에 `abs(error) / actual_value if actual_value > 0 else None` 결과를 저장합니다. 의미/사용: `pct_error` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        pct_error = abs(error) / actual_value if actual_value > 0 else None
        # LINE-BY-LINE: `actual_values.append(actual_value)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        actual_values.append(actual_value)
        # LINE-BY-LINE: `predicted_values.append(prediction)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        predicted_values.append(prediction)
        # LINE-BY-LINE: `output_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        output_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                "job_id": row.get("job_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row.get("work_order_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                "work_order_no": row.get("work_order_no", ""),
                # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row.get("machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                "machine_id": row.get("machine_id", ""),
                # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `row.get("cut_bay", "")` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                "cut_bay": row.get("cut_bay", ""),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `target` 키에 `target_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "target": target_key,
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `actual_minutes` 키에 `round(actual_value, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "actual_minutes": round(actual_value, 6),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `predicted_minutes` 키에 `round(prediction, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "predicted_minutes": round(prediction, 6),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `error_actual_minus_predicted` 키에 `round(error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "error_actual_minus_predicted": round(error, 6),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `abs_error_minutes` 키에 `round(abs(error), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "abs_error_minutes": round(abs(error), 6),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `abs_pct_error` 키에 `"" if pct_error is None else round(pct_error, 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "abs_pct_error": "" if pct_error is None else round(pct_error, 6),
                # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `row.get("cut_length", 0.0)` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
                "cut_length": row.get("cut_length", 0.0),
                # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `row.get("thickness", 0.0)` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
                "thickness": row.get("thickness", 0.0),
                # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `row.get("part_count", 0.0)` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
                "part_count": row.get("part_count", 0.0),
                # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `tact_time` 키에 `row.get("tact_time", 0.0)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "tact_time": row.get("tact_time", 0.0),
                # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `row.get("actual_duration_minutes", 0.0)` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
                "actual_duration_minutes": row.get("actual_duration_minutes", 0.0),
            }
        )
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_evaluate_formula`에서 반환/저장할 dict의 `metrics` 키에 `_prediction_metrics(actual_values, predicted_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "metrics": _prediction_metrics(actual_values, predicted_values),
        # LINE-BY-LINE: 딕셔너리 키 `rows`에는 `output_rows` 값을 넣습니다. 의미: CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
        "rows": output_rows,
    }


# LINE-BY-LINE: `_split_train_validation(rows: List[Dict[str, Any]], validation_ratio: float = 0.2)` 함수를 정의합니다. 반환 타입: `tuple[List[Dict[str, Any]], List[Dict[str, Any]]]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _split_train_validation(rows: List[Dict[str, Any]], validation_ratio: float = 0.2) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """시간순 row 목록을 train/validation으로 단순 분리한다."""

    # LINE-BY-LINE: 조건 `len(rows) < 5`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if len(rows) < 5:
        # LINE-BY-LINE: 호출자에게 `rows, []`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return rows, []
    # LINE-BY-LINE: `validation_count`에 `max(1, int(round(len(rows) * validation_ratio)))` 결과를 저장합니다. 의미/사용: `validation_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_count = max(1, int(round(len(rows) * validation_ratio)))
    # LINE-BY-LINE: `split_index`에 `max(1, len(rows) - validation_count)` 결과를 저장합니다. 의미/사용: `split_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    split_index = max(1, len(rows) - validation_count)
    # LINE-BY-LINE: 호출자에게 `rows[:split_index], rows[split_index:]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return rows[:split_index], rows[split_index:]


# LINE-BY-LINE: `_build_holdout_validation` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _build_holdout_validation(
    # LINE-BY-LINE: `rows`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]],
    # LINE-BY-LINE: `target_key`를 `str,` 타입으로 선언합니다. 의미/사용: `target_key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_key: str,
    # LINE-BY-LINE: `feature_keys`를 `List[str],` 타입으로 선언합니다. 의미/사용: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_keys: List[str],
    # LINE-BY-LINE: `formula_label`를 `str,` 타입으로 선언합니다. 의미/사용: `formula_label` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    formula_label: str,
    # LINE-BY-LINE: `validation_ratio` 변수에 `0.2` 결과를 저장합니다. 의미: `validation_ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_ratio: float = 0.2,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """train 구간으로 산식을 만들고 validation 구간에서 오차를 측정한다."""

    # LINE-BY-LINE: `train_rows, validation_rows` 여러 변수에 `_split_train_validation(rows, validation_ratio=validation_ratio)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    train_rows, validation_rows = _split_train_validation(rows, validation_ratio=validation_ratio)
    # LINE-BY-LINE: `train_formula`에 `_fit_linear_formula(train_rows, target_key, feature_keys, formula_label)` 결과를 저장합니다. 의미/사용: `train_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    train_formula = _fit_linear_formula(train_rows, target_key, feature_keys, formula_label)
    # LINE-BY-LINE: `train_eval`에 `_evaluate_formula(train_rows, target_key, train_formula)` 결과를 저장합니다. 의미/사용: `train_eval` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    train_eval = _evaluate_formula(train_rows, target_key, train_formula)
    # LINE-BY-LINE: `validation_eval`에 `_evaluate_formula(validation_rows, target_key, train_formula)` 결과를 저장합니다. 의미/사용: `validation_eval` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_eval = _evaluate_formula(validation_rows, target_key, train_formula)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `target` 키에 `target_key` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "target": target_key,
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `features` 키에 `feature_keys` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "features": feature_keys,
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `train_count` 키에 `len(train_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "train_count": len(train_rows),
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `validation_count` 키에 `len(validation_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_count": len(validation_rows),
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `validation_ratio` 키에 `validation_ratio` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_ratio": validation_ratio,
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `train_formula` 키에 `train_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "train_formula": train_formula,
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `train_metrics` 키에 `train_eval["metrics"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "train_metrics": train_eval["metrics"],
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `validation_metrics` 키에 `validation_eval["metrics"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_metrics": validation_eval["metrics"],
        # LINE-BY-LINE: `_build_holdout_validation`에서 반환/저장할 dict의 `validation_rows` 키에 `validation_eval["rows"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_rows": validation_eval["rows"],
    }


# LINE-BY-LINE: `_thickness_bucket(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _thickness_bucket(value: Any) -> str:
    """두께 값을 구간 label로 변환한다."""

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


# LINE-BY-LINE: `_cut_length_bucket(value: Any)` 함수를 정의합니다. 반환 타입: `str`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _cut_length_bucket(value: Any) -> str:
    """절단 길이 값을 구간 label로 변환한다."""

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


# LINE-BY-LINE: `_fit_segment_formulas` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _fit_segment_formulas(
    # LINE-BY-LINE: `rows`를 `List[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]],
    # LINE-BY-LINE: `segment_type`를 `str,` 타입으로 선언합니다. 의미/사용: `segment_type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    segment_type: str,
    # LINE-BY-LINE: `min_count` 변수에 `30` 결과를 저장합니다. 의미: `min_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    min_count: int = 30,
# LINE-BY-LINE: `) -> List[Dict[str, Any]]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> List[Dict[str, Any]]:
    """장비/두께구간/절단길이구간별 선형식 후보를 만든다."""

    # LINE-BY-LINE: `grouped` 변수에 `{}` 결과를 저장합니다. 의미: `grouped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: 조건 `segment_type == "machine_id"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if segment_type == "machine_id":
            # LINE-BY-LINE: `segment_value`에 `str(row.get("machine_id", "unknown") or "unknown")` 결과를 저장합니다. 의미/사용: `segment_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            segment_value = str(row.get("machine_id", "unknown") or "unknown")
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `segment_type == "thickness_bucket"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif segment_type == "thickness_bucket":
            # LINE-BY-LINE: `segment_value`에 `_thickness_bucket(row.get("thickness"))` 결과를 저장합니다. 의미/사용: `segment_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            segment_value = _thickness_bucket(row.get("thickness"))
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `segment_type == "cut_length_bucket"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif segment_type == "cut_length_bucket":
            # LINE-BY-LINE: `segment_value`에 `_cut_length_bucket(row.get("cut_length"))` 결과를 저장합니다. 의미/사용: `segment_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            segment_value = _cut_length_bucket(row.get("cut_length"))
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `ValueError(f"unknown segment_type: {segment_type}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise ValueError(f"unknown segment_type: {segment_type}")
        # LINE-BY-LINE: `grouped.setdefault(segment_value, []).append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        grouped.setdefault(segment_value, []).append(row)

    # LINE-BY-LINE: `segment_rows`에 `[]` 결과를 저장합니다. 의미/사용: `segment_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    segment_rows = []
    # LINE-BY-LINE: `segment_value, segment_rows_raw in sorted(grouped.items())` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for segment_value, segment_rows_raw in sorted(grouped.items()):
        # LINE-BY-LINE: 조건 `len(segment_rows_raw) < min_count`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if len(segment_rows_raw) < min_count:
            # LINE-BY-LINE: `segment_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            segment_rows.append(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `segment_type` 키에 `segment_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "segment_type": segment_type,
                    # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `segment_value` 키에 `segment_value` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "segment_value": segment_value,
                    # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `count` 키에 `len(segment_rows_raw)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "count": len(segment_rows_raw),
                    # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `status` 키에 `"insufficient_count"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "status": "insufficient_count",
                    # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `min_count` 키에 `min_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                    "min_count": min_count,
                }
            )
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `formula`에 `_fit_linear_formula(` 결과를 저장합니다. 의미/사용: `formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        formula = _fit_linear_formula(
            # LINE-BY-LINE: `_fit_linear_formula(...)` 호출에 `segment_rows_raw` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            segment_rows_raw,
            # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `_fit_linear_formula(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual_duration_minutes",
            # LINE-BY-LINE: `_fit_linear_formula(...)` 호출에 `CORE_TACT_FEATURE_KEYS` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            CORE_TACT_FEATURE_KEYS,
            # LINE-BY-LINE: 문자열 값 `"ESTIMATED_TACT_MINUTES"`를 `_fit_linear_formula(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "ESTIMATED_TACT_MINUTES",
        )
        # LINE-BY-LINE: `holdout`에 `_build_holdout_validation(` 결과를 저장합니다. 의미/사용: `holdout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        holdout = _build_holdout_validation(
            # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `segment_rows_raw` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            segment_rows_raw,
            # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "actual_duration_minutes",
            # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `CORE_TACT_FEATURE_KEYS` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            CORE_TACT_FEATURE_KEYS,
            # LINE-BY-LINE: 문자열 값 `"ESTIMATED_TACT_MINUTES"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "ESTIMATED_TACT_MINUTES",
        )
        # LINE-BY-LINE: `metrics`에 `formula["metrics"]` 결과를 저장합니다. 의미/사용: `metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        metrics = formula["metrics"]
        # LINE-BY-LINE: `holdout_metrics`에 `holdout["validation_metrics"]` 결과를 저장합니다. 의미/사용: `holdout_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        holdout_metrics = holdout["validation_metrics"]
        # LINE-BY-LINE: `coefficients`에 `formula["coefficients"]` 결과를 저장합니다. 의미/사용: `coefficients` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        coefficients = formula["coefficients"]
        # LINE-BY-LINE: `segment_rows.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        segment_rows.append(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `segment_type` 키에 `segment_type` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "segment_type": segment_type,
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `segment_value` 키에 `segment_value` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "segment_value": segment_value,
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `count` 키에 `len(segment_rows_raw)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "count": len(segment_rows_raw),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `status` 키에 `"ok"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "status": "ok",
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `min_count` 키에 `min_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "min_count": min_count,
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `formula` 키에 `formula["formula"]` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "formula": formula["formula"],
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `intercept` 키에 `coefficients.get("intercept")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "intercept": coefficients.get("intercept"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `cut_length_coef` 키에 `coefficients.get("cut_length")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "cut_length_coef": coefficients.get("cut_length"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `thickness_coef` 키에 `coefficients.get("thickness")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "thickness_coef": coefficients.get("thickness"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `part_count_coef` 키에 `coefficients.get("part_count")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "part_count_coef": coefficients.get("part_count"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `fit_mae_minutes` 키에 `metrics.get("mae_minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "fit_mae_minutes": metrics.get("mae_minutes"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `fit_rmse_minutes` 키에 `metrics.get("rmse_minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "fit_rmse_minutes": metrics.get("rmse_minutes"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `fit_mape_percent` 키에 `metrics.get("mape_percent")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "fit_mape_percent": metrics.get("mape_percent"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `holdout_count` 키에 `holdout_metrics.get("count")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "holdout_count": holdout_metrics.get("count"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `holdout_mae_minutes` 키에 `holdout_metrics.get("mae_minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "holdout_mae_minutes": holdout_metrics.get("mae_minutes"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `holdout_rmse_minutes` 키에 `holdout_metrics.get("rmse_minutes")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "holdout_rmse_minutes": holdout_metrics.get("rmse_minutes"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `holdout_mape_percent` 키에 `holdout_metrics.get("mape_percent")` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "holdout_mape_percent": holdout_metrics.get("mape_percent"),
                # LINE-BY-LINE: `_fit_segment_formulas`에서 반환/저장할 dict의 `holdout_bias_actual_minus_predicted_minutes` 키에 `holdout_metrics.get(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "holdout_bias_actual_minus_predicted_minutes": holdout_metrics.get(
                    # LINE-BY-LINE: 문자열 값 `"bias_actual_minus_predicted_minutes"`를 `get(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                    "bias_actual_minus_predicted_minutes"
                ),
            }
        )
    # LINE-BY-LINE: 호출자에게 `segment_rows`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return segment_rows


# LINE-BY-LINE: `_predict_linear(row: Dict[str, Any], formula: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def _predict_linear(row: Dict[str, Any], formula: Dict[str, Any]) -> Optional[float]:
    """저장된 coefficients를 사용해 row 1건의 예측값을 계산한다."""

    # LINE-BY-LINE: `coefficients`에 `formula.get("coefficients", {})` 결과를 저장합니다. 의미/사용: `coefficients` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    coefficients = formula.get("coefficients", {})
    # LINE-BY-LINE: 조건 `"intercept" not in coefficients`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if "intercept" not in coefficients:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None
    # LINE-BY-LINE: `prediction`에 `float(coefficients["intercept"])` 결과를 저장합니다. 의미/사용: `prediction` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    prediction = float(coefficients["intercept"])
    # LINE-BY-LINE: `feature_key in formula.get("features_used", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for feature_key in formula.get("features_used", []):
        # LINE-BY-LINE: `prediction` 값을 `float(coefficients[feature_key]) * _to_float(row.get(feature_key))` 기준으로 누적/증가합니다. 의미/사용: `prediction` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        prediction += float(coefficients[feature_key]) * _to_float(row.get(feature_key))
    # LINE-BY-LINE: 호출자에게 `prediction`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return prediction


# LINE-BY-LINE: `fit_actual_duration_formula` 함수를 정의합니다. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def fit_actual_duration_formula(
    # LINE-BY-LINE: `feature_rows`를 `Iterable[Dict[str, Any]],` 타입으로 선언합니다. 의미/사용: `feature_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_rows: Iterable[Dict[str, Any]],
    # LINE-BY-LINE: `feature_keys` 변수에 `None` 결과를 저장합니다. 의미: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_keys: Optional[List[str]] = None,
# LINE-BY-LINE: `) -> Dict[str, Any]:` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
) -> Dict[str, Any]:
    """실적 elapsed time을 target으로 하는 처리시간 선형식 후보를 맞춘다."""

    # LINE-BY-LINE: `feature_keys`에 `feature_keys or TACT_FEATURE_KEYS` 결과를 저장합니다. 의미/사용: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_keys = feature_keys or TACT_FEATURE_KEYS
    # LINE-BY-LINE: `numeric_rows`에 `[]` 결과를 저장합니다. 의미/사용: `numeric_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    numeric_rows = []
    # LINE-BY-LINE: `row in feature_rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in feature_rows:
        # LINE-BY-LINE: `actual_duration`에 `_to_float(row.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float(row.get("actual_duration_minutes"))
        # LINE-BY-LINE: 조건 `actual_duration <= 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if actual_duration <= 0:
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue
        # LINE-BY-LINE: `numeric_row`에 `{key: _to_float(row.get(key)) for key in feature_keys}` 결과를 저장합니다. 의미/사용: `numeric_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        numeric_row = {key: _to_float(row.get(key)) for key in feature_keys}
        # LINE-BY-LINE: `numeric_row["actual_duration_minutes"]`에 `actual_duration` 결과를 저장합니다. 의미/사용: `numeric_row["actual_duration_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        numeric_row["actual_duration_minutes"] = actual_duration
        # LINE-BY-LINE: `numeric_rows.append(numeric_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        numeric_rows.append(numeric_row)
    # LINE-BY-LINE: 호출자에게 `_fit_linear_formula(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return _fit_linear_formula(
        # LINE-BY-LINE: `_fit_linear_formula(...)` 호출에 `numeric_rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        numeric_rows,
        # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `_fit_linear_formula(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "actual_duration_minutes",
        # LINE-BY-LINE: `_fit_linear_formula(...)` 호출에 `feature_keys` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        feature_keys,
        # LINE-BY-LINE: 문자열 값 `"ESTIMATED_TACT_MINUTES"`를 `_fit_linear_formula(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "ESTIMATED_TACT_MINUTES",
    )


# LINE-BY-LINE: `predict_actual_duration_from_formula(feature_row: Dict[str, Any], formula: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Optional[float]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다.
def predict_actual_duration_from_formula(feature_row: Dict[str, Any], formula: Dict[str, Any]) -> Optional[float]:
    """fit_actual_duration_formula 결과로 1개 row의 처리시간을 추정한다."""

    # LINE-BY-LINE: 호출자에게 `_predict_linear(feature_row, formula)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return _predict_linear(feature_row, formula)


# LINE-BY-LINE: `build_tact_time_analysis_from_scenario(scenario: Dict[str, Any])` 함수를 정의합니다. 반환 타입: `Dict[str, Any]`. 사용: TACT_TIME과 실적 elapsed 분석/산식 생성에서 사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def build_tact_time_analysis_from_scenario(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """scenario에 저장된 원천 컬럼으로 TACT_TIME 분석 리포트를 만든다."""

    # LINE-BY-LINE: `feature_keys`에 `TACT_FEATURE_KEYS` 결과를 저장합니다. 의미/사용: `feature_keys` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    feature_keys = TACT_FEATURE_KEYS
    # LINE-BY-LINE: `rows` 변수에 `[]` 결과를 저장합니다. 의미: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows: List[Dict[str, Any]] = []
    # LINE-BY-LINE: `numeric_rows` 변수에 `[]` 결과를 저장합니다. 의미: `numeric_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    numeric_rows: List[Dict[str, float]] = []

    # LINE-BY-LINE: `job in scenario.get("jobs", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for job in scenario.get("jobs", []):
        # LINE-BY-LINE: `base_stage`에 `job.get("base_stage_minutes", {}) or {}` 결과를 저장합니다. 의미/사용: `base_stage` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        base_stage = job.get("base_stage_minutes", {}) or {}
        # LINE-BY-LINE: `tact_time`에 `_to_float(job.get("tact_time_minutes", base_stage.get("cut", 0.0)))` 결과를 저장합니다. 의미/사용: `tact_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_time = _to_float(job.get("tact_time_minutes", base_stage.get("cut", 0.0)))
        # LINE-BY-LINE: `actual_duration`에 `_to_float(job.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_duration = _to_float(job.get("actual_duration_minutes"))
        # LINE-BY-LINE: `row`에 `{` 결과를 저장합니다. 의미/사용: `row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        row = {
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `job.get("job_id", "")` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": job.get("job_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `job.get("source_wk_ord_no", "")` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
            "work_order_no": job.get("source_wk_ord_no", ""),
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `job.get("source_machine_id", "")` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": job.get("source_machine_id", ""),
            # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `job.get("source_cut_bay", "")` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
            "cut_bay": job.get("source_cut_bay", ""),
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `tact_time` 키에 `tact_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "tact_time": tact_time,
            # LINE-BY-LINE: 딕셔너리 키 `actual_duration_minutes`에는 `actual_duration` 값을 넣습니다. 의미: 실적 종료-착수 elapsed time입니다. 사용: actual replay 검증과 tact 산식 target.
            "actual_duration_minutes": actual_duration,
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `actual_minus_tact` 키에 `actual_duration - tact_time` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "actual_minus_tact": actual_duration - tact_time,
            # LINE-BY-LINE: 딕셔너리 키 `cut_length`에는 `_to_float(job.get("cut_length"))` 값을 넣습니다. 의미: 절단 길이입니다. 예: `120.5`, 사용: tact time 산식 feature.
            "cut_length": _to_float(job.get("cut_length")),
            # LINE-BY-LINE: 딕셔너리 키 `mark_length`에는 `_to_float(job.get("mark_length"))` 값을 넣습니다. 의미: 마킹 길이입니다. 사용: tact time 분석 feature.
            "mark_length": _to_float(job.get("mark_length")),
            # LINE-BY-LINE: 딕셔너리 키 `bevel_length`에는 `_to_float(job.get("bevel_length"))` 값을 넣습니다. 의미: 베벨 길이입니다. 사용: tact time 분석 feature.
            "bevel_length": _to_float(job.get("bevel_length")),
            # LINE-BY-LINE: 딕셔너리 키 `plate_length`에는 `_to_float(job.get("plate_length"))` 값을 넣습니다. 의미: 판/작업 길이입니다. 예: `9000.0`, 사용: table length와 rolling 55000 길이합 제약.
            "plate_length": _to_float(job.get("plate_length")),
            # LINE-BY-LINE: 딕셔너리 키 `thickness`에는 `_to_float(job.get("thickness"))` 값을 넣습니다. 의미: 두께 값입니다. 예: `12.0`, 사용: thickness_range 제약과 tact 산식.
            "thickness": _to_float(job.get("thickness")),
            # LINE-BY-LINE: 딕셔너리 키 `part_count`에는 `_to_float(job.get("part_count"))` 값을 넣습니다. 의미: 부재 수량입니다. 예: `PTLST_QTY`, 사용: tact time 산식 feature.
            "part_count": _to_float(job.get("part_count")),
            # LINE-BY-LINE: 딕셔너리 키 `steel_qty`에는 `_to_float(job.get("steel_qty"), default=1.0)` 값을 넣습니다. 의미: 원본 강재 수량입니다. 예: `STL_QTY`, 사용: steel_quantity로 변환.
            "steel_qty": _to_float(job.get("steel_qty"), default=1.0),
        }
        # LINE-BY-LINE: `rows.append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        rows.append(row)
        # LINE-BY-LINE: 조건 `tact_time > 0 and actual_duration > 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if tact_time > 0 and actual_duration > 0:
            # LINE-BY-LINE: `numeric_rows.append({key: _to_float(row[key]) for key in ["tact_time", "actual_duration_minutes",...`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            numeric_rows.append({key: _to_float(row[key]) for key in ["tact_time", "actual_duration_minutes", *feature_keys]})
            # LINE-BY-LINE: `numeric_rows[-1].update(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            numeric_rows[-1].update(
                # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
                {
                    # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `row["job_id"]` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
                    "job_id": row["job_id"],
                    # LINE-BY-LINE: 딕셔너리 키 `work_order_no`에는 `row["work_order_no"]` 값을 넣습니다. 의미: 현업 W/O 번호입니다. 예: `WK_ORD_NO`, 사용: report와 원본 데이터 추적.
                    "work_order_no": row["work_order_no"],
                    # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `row["machine_id"]` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
                    "machine_id": row["machine_id"],
                    # LINE-BY-LINE: 딕셔너리 키 `cut_bay`에는 `row["cut_bay"]` 값을 넣습니다. 의미: 절단 Bay입니다. 예: `22`, 사용: Machine.bay_id와 비교해 Bay 일관성 검증.
                    "cut_bay": row["cut_bay"],
                }
            )

    # LINE-BY-LINE: `correlations`에 `{}` 결과를 저장합니다. 의미/사용: `correlations` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    correlations = {}
    # LINE-BY-LINE: `feature_key in feature_keys` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for feature_key in feature_keys:
        # LINE-BY-LINE: `xs`에 `[row[feature_key] for row in numeric_rows]` 결과를 저장합니다. 의미/사용: `xs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        xs = [row[feature_key] for row in numeric_rows]
        # LINE-BY-LINE: `tact_values`에 `[row["tact_time"] for row in numeric_rows]` 결과를 저장합니다. 의미/사용: `tact_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tact_values = [row["tact_time"] for row in numeric_rows]
        # LINE-BY-LINE: `actual_values`에 `[row["actual_duration_minutes"] for row in numeric_rows]` 결과를 저장합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        actual_values = [row["actual_duration_minutes"] for row in numeric_rows]
        # LINE-BY-LINE: `correlations[feature_key]`에 `{` 결과를 저장합니다. 의미/사용: `correlations[feature_key]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        correlations[feature_key] = {
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `corr_with_tact_time` 키에 `_pearson(xs, tact_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "corr_with_tact_time": _pearson(xs, tact_values),
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `corr_with_actual_duration` 키에 `_pearson(xs, actual_values)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "corr_with_actual_duration": _pearson(xs, actual_values),
        }

    # LINE-BY-LINE: `tact_values`에 `[row["tact_time"] for row in numeric_rows]` 결과를 저장합니다. 의미/사용: `tact_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_values = [row["tact_time"] for row in numeric_rows]
    # LINE-BY-LINE: `actual_values`에 `[row["actual_duration_minutes"] for row in numeric_rows]` 결과를 저장합니다. 의미/사용: `actual_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_values = [row["actual_duration_minutes"] for row in numeric_rows]
    # LINE-BY-LINE: `tact_formula`에 `_fit_linear_formula(numeric_rows, "tact_time", feature_keys, "TACT_TIME")` 결과를 저장합니다. 의미/사용: `tact_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_formula = _fit_linear_formula(numeric_rows, "tact_time", feature_keys, "TACT_TIME")
    # LINE-BY-LINE: `core_tact_formula`에 `_fit_linear_formula(numeric_rows, "tact_time", CORE_TACT_FEATURE_KEYS, "TACT_TIME")` 결과를 저장합니다. 의미/사용: `core_tact_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    core_tact_formula = _fit_linear_formula(numeric_rows, "tact_time", CORE_TACT_FEATURE_KEYS, "TACT_TIME")
    # LINE-BY-LINE: `actual_formula`에 `fit_actual_duration_formula(numeric_rows)` 결과를 저장합니다. 의미/사용: `actual_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_formula = fit_actual_duration_formula(numeric_rows)
    # LINE-BY-LINE: `core_actual_formula`에 `fit_actual_duration_formula(numeric_rows, CORE_TACT_FEATURE_KEYS)` 결과를 저장합니다. 의미/사용: `core_actual_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    core_actual_formula = fit_actual_duration_formula(numeric_rows, CORE_TACT_FEATURE_KEYS)
    # LINE-BY-LINE: `core_holdout`에 `_build_holdout_validation(` 결과를 저장합니다. 의미/사용: `core_holdout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    core_holdout = _build_holdout_validation(
        # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `numeric_rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        numeric_rows,
        # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "actual_duration_minutes",
        # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `CORE_TACT_FEATURE_KEYS` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        CORE_TACT_FEATURE_KEYS,
        # LINE-BY-LINE: 문자열 값 `"ESTIMATED_TACT_MINUTES"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "ESTIMATED_TACT_MINUTES",
    )
    # LINE-BY-LINE: `extended_holdout`에 `_build_holdout_validation(` 결과를 저장합니다. 의미/사용: `extended_holdout` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    extended_holdout = _build_holdout_validation(
        # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `numeric_rows` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        numeric_rows,
        # LINE-BY-LINE: 문자열 값 `"actual_duration_minutes"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "actual_duration_minutes",
        # LINE-BY-LINE: `_build_holdout_validation(...)` 호출에 `TACT_FEATURE_KEYS` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        TACT_FEATURE_KEYS,
        # LINE-BY-LINE: 문자열 값 `"ESTIMATED_TACT_MINUTES"`를 `_build_holdout_validation(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "ESTIMATED_TACT_MINUTES",
    )
    # LINE-BY-LINE: `segment_formula_rows`에 `[]` 결과를 저장합니다. 의미/사용: `segment_formula_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    segment_formula_rows = []
    # LINE-BY-LINE: `segment_type in ("machine_id", "thickness_bucket", "cut_length_bucket")` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for segment_type in ("machine_id", "thickness_bucket", "cut_length_bucket"):
        # LINE-BY-LINE: `segment_formula_rows.extend(_fit_segment_formulas(numeric_rows, segment_type))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        segment_formula_rows.extend(_fit_segment_formulas(numeric_rows, segment_type))

    # LINE-BY-LINE: `row in rows` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in rows:
        # LINE-BY-LINE: `estimated_actual_duration`에 `_predict_linear(row, actual_formula)` 결과를 저장합니다. 의미/사용: `estimated_actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        estimated_actual_duration = _predict_linear(row, actual_formula)
        # LINE-BY-LINE: `core_estimated_actual_duration`에 `_predict_linear(row, core_actual_formula)` 결과를 저장합니다. 의미/사용: `core_estimated_actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        core_estimated_actual_duration = _predict_linear(row, core_actual_formula)
        # LINE-BY-LINE: 조건 `estimated_actual_duration is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if estimated_actual_duration is None:
            # LINE-BY-LINE: `row["estimated_tact_minutes_from_actual_model"]`에 `""` 결과를 저장합니다. 의미/사용: `row["estimated_tact_minutes_from_actual_model"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_tact_minutes_from_actual_model"] = ""
            # LINE-BY-LINE: `row["estimated_model_error_actual_minus_estimate"]`에 `""` 결과를 저장합니다. 의미/사용: `row["estimated_model_error_actual_minus_estimate"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_model_error_actual_minus_estimate"] = ""
            # LINE-BY-LINE: `row["estimated_model_abs_error"]`에 `""` 결과를 저장합니다. 의미/사용: `row["estimated_model_abs_error"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_model_abs_error"] = ""
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `actual_duration`에 `_to_float(row.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            actual_duration = _to_float(row.get("actual_duration_minutes"))
            # LINE-BY-LINE: `row["estimated_tact_minutes_from_actual_model"]`에 `round(estimated_actual_duration, 6)` 결과를 저장합니다. 의미/사용: `row["estimated_tact_minutes_from_actual_model"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_tact_minutes_from_actual_model"] = round(estimated_actual_duration, 6)
            # LINE-BY-LINE: `row["estimated_model_error_actual_minus_estimate"]`에 `round(actual_duration - estimated_actual_duration, 6)` 결과를 저장합니다. 의미/사용: `row["estimated_model_error_actual_minus_estimate"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_model_error_actual_minus_estimate"] = round(actual_duration - estimated_actual_duration, 6)
            # LINE-BY-LINE: `row["estimated_model_abs_error"]`에 `round(abs(actual_duration - estimated_actual_duration), 6)` 결과를 저장합니다. 의미/사용: `row["estimated_model_abs_error"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["estimated_model_abs_error"] = round(abs(actual_duration - estimated_actual_duration), 6)
        # LINE-BY-LINE: 조건 `core_estimated_actual_duration is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if core_estimated_actual_duration is None:
            # LINE-BY-LINE: `row["core_estimated_tact_minutes"]`에 `""` 결과를 저장합니다. 의미/사용: `row["core_estimated_tact_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_tact_minutes"] = ""
            # LINE-BY-LINE: `row["core_estimated_error_actual_minus_estimate"]`에 `""` 결과를 저장합니다. 의미/사용: `row["core_estimated_error_actual_minus_estimate"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_error_actual_minus_estimate"] = ""
            # LINE-BY-LINE: `row["core_estimated_abs_error"]`에 `""` 결과를 저장합니다. 의미/사용: `row["core_estimated_abs_error"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_abs_error"] = ""
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `actual_duration`에 `_to_float(row.get("actual_duration_minutes"))` 결과를 저장합니다. 의미/사용: `actual_duration` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            actual_duration = _to_float(row.get("actual_duration_minutes"))
            # LINE-BY-LINE: `row["core_estimated_tact_minutes"]`에 `round(core_estimated_actual_duration, 6)` 결과를 저장합니다. 의미/사용: `row["core_estimated_tact_minutes"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_tact_minutes"] = round(core_estimated_actual_duration, 6)
            # LINE-BY-LINE: `row["core_estimated_error_actual_minus_estimate"]`에 `round(actual_duration - core_estimated_actual_duration, 6)` 결과를 저장합니다. 의미/사용: `row["core_estimated_error_actual_minus_estimate"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_error_actual_minus_estimate"] = round(actual_duration - core_estimated_actual_duration, 6)
            # LINE-BY-LINE: `row["core_estimated_abs_error"]`에 `round(abs(actual_duration - core_estimated_actual_duration), 6)` 결과를 저장합니다. 의미/사용: `row["core_estimated_abs_error"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            row["core_estimated_abs_error"] = round(abs(actual_duration - core_estimated_actual_duration), 6)

    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `record_count` 키에 `len(rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "record_count": len(rows),
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `formula_target` 키에 `"actual_duration_minutes"` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "formula_target": "actual_duration_minutes",
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `formula_features` 키에 `feature_keys` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "formula_features": feature_keys,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `tact_vs_actual` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "tact_vs_actual": {
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `compare_count` 키에 `len(numeric_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "compare_count": len(numeric_rows),
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `mae_minutes` 키에 `_mean_absolute_error(actual_values, tact_values) if numeric_rows else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mae_minutes": _mean_absolute_error(actual_values, tact_values) if numeric_rows else None,
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `bias_actual_minus_tact_minutes` 키에 `statistics.mean(a - t for a, t in zip(actual_values, tact_values)) if numeric_rows else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "bias_actual_minus_tact_minutes": statistics.mean(a - t for a, t in zip(actual_values, tact_values)) if numeric_rows else None,
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `rmse_minutes` 키에 `_root_mean_squared_error(actual_values, tact_values) if numeric_rows else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "rmse_minutes": _root_mean_squared_error(actual_values, tact_values) if numeric_rows else None,
            # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `mape_percent` 키에 `_mean_absolute_percentage_error(actual_values, tact_values) if numeric_rows else None` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "mape_percent": _mean_absolute_percentage_error(actual_values, tact_values) if numeric_rows else None,
        },
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `correlations` 키에 `correlations` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "correlations": correlations,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `requested_core_features` 키에 `CORE_TACT_FEATURE_KEYS` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "requested_core_features": CORE_TACT_FEATURE_KEYS,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `requested_core_correlations` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "requested_core_correlations": {
            # LINE-BY-LINE: `key`를 `correlations[key]` 타입으로 선언합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key: correlations[key]
            # LINE-BY-LINE: `key in CORE_TACT_FEATURE_KEYS` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key in CORE_TACT_FEATURE_KEYS
            # LINE-BY-LINE: 조건 `key in correlations`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key in correlations
        },
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `linear_formula_fit_to_tact_time_core_features` 키에 `core_tact_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "linear_formula_fit_to_tact_time_core_features": core_tact_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `linear_formula_fit_to_actual_duration_core_features` 키에 `core_actual_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "linear_formula_fit_to_actual_duration_core_features": core_actual_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `linear_formula_fit_to_tact_time` 키에 `tact_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "linear_formula_fit_to_tact_time": tact_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `linear_formula_fit_to_actual_duration` 키에 `actual_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "linear_formula_fit_to_actual_duration": actual_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `recommended_tact_time_formula` 키에 `core_actual_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "recommended_tact_time_formula": core_actual_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `recommended_process_time_model` 키에 `core_actual_formula` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "recommended_process_time_model": core_actual_formula,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `core_formula_holdout_validation` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "core_formula_holdout_validation": {
            # LINE-BY-LINE: `key`를 `value` 타입으로 선언합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key: value
            # LINE-BY-LINE: `key, value in core_holdout.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key, value in core_holdout.items()
            # LINE-BY-LINE: 조건 `key != "validation_rows"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key != "validation_rows"
        },
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `extended_formula_holdout_validation` 키에 `{` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "extended_formula_holdout_validation": {
            # LINE-BY-LINE: `key`를 `value` 타입으로 선언합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            key: value
            # LINE-BY-LINE: `key, value in extended_holdout.items()` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for key, value in extended_holdout.items()
            # LINE-BY-LINE: 조건 `key != "validation_rows"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if key != "validation_rows"
        },
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `segment_formula_count` 키에 `len(segment_formula_rows)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "segment_formula_count": len(segment_formula_rows),
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `segment_formula_note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "segment_formula_note": (
            # LINE-BY-LINE: 문자열 값 `"Segment formulas are diagnostic candidates. They are not used automatically unless explicitly se...`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Segment formulas are diagnostic candidates. They are not used automatically unless explicitly selected."
        ),
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `note` 키에 `(` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "note": (
            # LINE-BY-LINE: 문자열 값 `"linear_formula_fit_to_tact_time only reproduces the provided TACT_TIME column. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "linear_formula_fit_to_tact_time only reproduces the provided TACT_TIME column. "
            # LINE-BY-LINE: 문자열 값 `"recommended_tact_time_formula fits actual RT_CUT_ST_DTM~RT_CUT_ED_DTM elapsed time "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "recommended_tact_time_formula fits actual RT_CUT_ST_DTM~RT_CUT_ED_DTM elapsed time "
            # LINE-BY-LINE: 문자열 값 `"from the requested core features: CUT_LTH, THK, PTLST_QTY. "`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "from the requested core features: CUT_LTH, THK, PTLST_QTY. "
            # LINE-BY-LINE: 문자열 값 `"Holdout validation rows show W/O-level differences between actual elapsed time and formula predi...`를 `현재 표현식(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
            "Holdout validation rows show W/O-level differences between actual elapsed time and formula prediction."
        ),
    }
    # LINE-BY-LINE: `validation_rows`에 `[]` 결과를 저장합니다. 의미/사용: `validation_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_rows = []
    # LINE-BY-LINE: `validation_row in core_holdout["validation_rows"]` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for validation_row in core_holdout["validation_rows"]:
        # LINE-BY-LINE: `output_row`에 `dict(validation_row)` 결과를 저장합니다. 의미/사용: `output_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        output_row = dict(validation_row)
        # LINE-BY-LINE: `output_row["formula_set"]`에 `"core_cut_length_thickness_part_count"` 결과를 저장합니다. 의미/사용: `output_row["formula_set"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        output_row["formula_set"] = "core_cut_length_thickness_part_count"
        # LINE-BY-LINE: `validation_rows.append(output_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        validation_rows.append(output_row)
    # LINE-BY-LINE: `validation_row in extended_holdout["validation_rows"]` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for validation_row in extended_holdout["validation_rows"]:
        # LINE-BY-LINE: `output_row`에 `dict(validation_row)` 결과를 저장합니다. 의미/사용: `output_row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        output_row = dict(validation_row)
        # LINE-BY-LINE: `output_row["formula_set"]`에 `"extended_available_features"` 결과를 저장합니다. 의미/사용: `output_row["formula_set"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        output_row["formula_set"] = "extended_available_features"
        # LINE-BY-LINE: `validation_rows.append(output_row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        validation_rows.append(output_row)
    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `summary` 키에 `summary` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "summary": summary,
        # LINE-BY-LINE: 딕셔너리 키 `rows`에는 `rows` 값을 넣습니다. 의미: CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
        "rows": rows,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `validation_rows` 키에 `validation_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "validation_rows": validation_rows,
        # LINE-BY-LINE: `build_tact_time_analysis_from_scenario`에서 반환/저장할 dict의 `segment_formula_rows` 키에 `segment_formula_rows` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "segment_formula_rows": segment_formula_rows,
    }
