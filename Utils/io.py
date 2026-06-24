"""입출력 유틸.

이 모듈은 두 가지 scenario 입력 방식을 지원합니다.

1. 기존 방식: `paths.scenario_path`가 가리키는 YAML scenario를 그대로 읽습니다.
2. 현재 권장 방식: `paths.source_data_path` 원본 Excel/CSV를 읽고 메모리에서 scenario dict를 만듭니다.

두 번째 방식은 `input/`에 거대한 YAML을 저장하지 않기 위한 구조입니다.
예: 전체 NP scenario YAML은 20만 줄 이상이 될 수 있으므로 config에는 원본 파일과 record 수만 둡니다.
"""

# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Dict, Tuple

# LINE-BY-LINE: `yaml` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import yaml

from .cutting_data_loader import load_and_clean_cutting_data
from .cutting_scenario_builder import build_scenario_from_cutting_records


# LINE-BY-LINE: `load_scenario(scenario_path: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def load_scenario(scenario_path: str):
    """시나리오 yaml을 읽습니다.

    시나리오는 현재 프로젝트의 가장 바깥 입력 단위입니다.
    즉, 설비 / bay / 작업 목록을 한 번에 읽어 환경으로 넘깁니다.
    """

    # LINE-BY-LINE: `Path(scenario_path).open("r", encoding="utf-8") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with Path(scenario_path).open("r", encoding="utf-8") as handle:
        # LINE-BY-LINE: 호출자에게 `yaml.safe_load(handle)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return yaml.safe_load(handle)


def _as_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    """config 값을 문자열 tuple로 정규화합니다.

    입력 예:
    - "NP"
    - "NP,FL"
    - ["NP", "FL"]

    비어 있으면 target 계열을 알 수 없으므로 예외를 발생시킵니다.
    """

    if isinstance(value, str):
        result = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, Iterable):
        result = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        result = ()
    if not result:
        print(f"[ERROR][Utils.io._as_tuple] cause=empty field={field_name} input={value}")
        raise ValueError(f"{field_name} must contain at least one value")
    return result


def _exclude_for_process_time(record: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """처리시간 출처별 추가 제외 row를 만든다.

    예:
    - process_time_source=tact_time인데 `tact_time`이 0 이하인 W/O는 planning 처리시간이 없으므로 제외한다.
    - actual replay는 `actual_duration_minutes`를 쓰므로 같은 W/O를 여기서 제외하지 않는다.
    """

    excluded = dict(record)
    excluded["exclude_reason"] = reason
    return excluded


def _filter_records_for_process_time_source(cleaned: Any, process_time_source: str) -> Any:
    """선택한 처리시간 출처로 scenario 생성 가능한 row만 남긴다.

    이 함수는 fallback이 아니다.
    `tact_time` planning에서는 TACT_TIME 0 이하를 다른 값으로 대체하지 않고 제외 로그에 남긴다.
    `actual_duration` replay에서는 TACT_TIME이 필요 없으므로 이 필터를 적용하지 않는다.
    """

    if process_time_source != "tact_time":
        print(
            "[CHECK][Utils.io._filter_records_for_process_time_source] "
            f"process_time_source={process_time_source} selected={len(cleaned.records)} "
            "excluded=0 reason=no_tact_filter_required"
        )
        return cleaned

    selected = []
    excluded = list(cleaned.excluded_records)
    process_time_excluded_count = 0
    for record in cleaned.records:
        work_order_no = record.get("work_order_no")
        source_row_index = record.get("source_row_index")
        tact_time = record.get("tact_time")
        try:
            tact_minutes = float(tact_time)
        except (TypeError, ValueError) as exc:
            print(
                "[ERROR][Utils.io._filter_records_for_process_time_source] "
                f"cause=invalid_tact_time key={work_order_no} "
                f"source_row_index={source_row_index} tact_time={tact_time}"
            )
            raise ValueError(f"invalid tact_time for W/O {work_order_no}: {tact_time}") from exc
        if tact_minutes <= 0:
            excluded.append(_exclude_for_process_time(record, "non_positive_tact_time_for_tact_time_source"))
            process_time_excluded_count += 1
            continue
        selected.append(record)

    cleaned.records = selected
    cleaned.excluded_records = excluded
    print(
        "[CHECK][Utils.io._filter_records_for_process_time_source] "
        f"process_time_source=tact_time selected={len(selected)} "
        f"excluded={process_time_excluded_count}"
    )
    return cleaned


def _load_scenario_from_source_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """원본 Excel/CSV에서 scenario dict를 메모리로 생성합니다.

    이 함수는 fallback이 아닙니다.
    config에서 `paths.scenario_path`를 비워 두고 `paths.source_data_path`를 명시했을 때 사용하는
    정상 입력 경로입니다.
    """

    paths = config.get("paths", {})
    data_config = config.get("data", {})
    source_data_path = paths.get("source_data_path")
    if not source_data_path:
        print("[ERROR][Utils.io._load_scenario_from_source_data] cause=missing paths.source_data_path")
        raise ValueError("paths.source_data_path is required when paths.scenario_path is empty")

    sheet_name = str(data_config.get("sheet_name") or paths.get("source_sheet_name") or "Sheet")
    target_series = _as_tuple(data_config.get("target_series", ("NP",)), "data.target_series")
    max_records = data_config.get("max_records")
    process_time_source_value = data_config.get("process_time_source")
    if not process_time_source_value:
        print(
            "[ERROR][Utils.io._load_scenario_from_source_data] "
            "cause=missing data.process_time_source "
            "fix=set data.process_time_source=tact_time"
        )
        raise ValueError("data.process_time_source is required; use tact_time for current PMSP validation")
    process_time_source = str(process_time_source_value)

    print(
        "[CHECK][Utils.io._load_scenario_from_source_data] "
        f"source_data_path={source_data_path} sheet={sheet_name} "
        f"target_series={','.join(target_series)} max_records={max_records} "
        f"process_time_source={process_time_source}"
    )

    cleaned = load_and_clean_cutting_data(
        str(source_data_path),
        sheet_name=sheet_name,
        target_series=target_series,
        max_records=None if max_records in (None, "") else int(max_records),
    )
    cleaned = _filter_records_for_process_time_source(cleaned, process_time_source)
    if not cleaned.records:
        print(
            "[ERROR][Utils.io._load_scenario_from_source_data] "
            f"cause=no usable records source_data_path={source_data_path}"
        )
        raise ValueError(f"no usable records in source_data_path={source_data_path}")

    scenario = build_scenario_from_cutting_records(
        cleaned.records,
        factory_config=config.get("factory"),
        source_file=str(source_data_path),
        process_time_source=process_time_source,
    )
    scenario.setdefault("metadata", {})
    scenario["metadata"]["input_mode"] = "source_data"
    scenario["metadata"]["selected_records"] = len(cleaned.records)
    scenario["metadata"]["excluded_records"] = len(cleaned.excluded_records)
    print(
        "[CHECK][Utils.io._load_scenario_from_source_data] "
        f"selected={len(cleaned.records)} excluded={len(cleaned.excluded_records)}"
    )
    return scenario


def load_scenario_for_config(
    config: Dict[str, Any],
    scenario_path_override: str | None = None,
) -> Dict[str, Any]:
    """config 기준으로 scenario를 로드합니다.

    우선순위:
    1. CLI에서 `--scenario-path`를 명시하면 그 YAML을 읽습니다.
    2. config의 `paths.scenario_path`가 있으면 그 YAML을 읽습니다.
    3. config의 `paths.source_data_path`가 있으면 원본 데이터에서 scenario를 메모리로 생성합니다.

    2번의 파일이 없으면 3번으로 조용히 넘어가지 않습니다.
    명시한 YAML이 없다는 뜻이므로 원인을 출력하고 실패합니다.
    """

    if scenario_path_override:
        return load_scenario(scenario_path_override)

    scenario_path = config.get("paths", {}).get("scenario_path")
    if scenario_path:
        path = Path(str(scenario_path))
        if not path.exists():
            print(
                "[ERROR][Utils.io.load_scenario_for_config] "
                f"cause=scenario_path_not_found scenario_path={scenario_path}"
            )
            raise FileNotFoundError(f"scenario_path not found: {scenario_path}")
        return load_scenario(str(path))

    return _load_scenario_from_source_data(config)
