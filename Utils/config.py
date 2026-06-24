"""config 로딩 유틸.

초보자용 포인트:
- config.yaml 안의 경로는 보통 상대경로로 적습니다.
- 이 파일이 그 상대경로를 프로젝트 루트 기준 절대경로로 바꿔줍니다.
- 그래서 어디서 실행하든 같은 파일을 읽게 됩니다.
"""

# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path

# LINE-BY-LINE: `yaml` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import yaml


# LINE-BY-LINE: `project_root()` 함수를 정의합니다. 반환 타입: `Path`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def project_root() -> Path:
    """프로젝트 루트 경로를 반환합니다."""

    # LINE-BY-LINE: 호출자에게 `Path(__file__).resolve().parents[1]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return Path(__file__).resolve().parents[1]


# LINE-BY-LINE: `_resolve_path(value: str)` 함수를 정의합니다. 반환 타입: `str`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def _resolve_path(value: str) -> str:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환합니다."""

    # LINE-BY-LINE: `path`에 `Path(value)` 결과를 저장합니다. 의미/사용: `path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    path = Path(value)
    # LINE-BY-LINE: 조건 `path.is_absolute()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if path.is_absolute():
        # LINE-BY-LINE: 호출자에게 `str(path)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return str(path)
    # LINE-BY-LINE: 호출자에게 `str((project_root() / path).resolve())`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return str((project_root() / path).resolve())


def _resolve_optional_path(config: dict, section: str, key: str) -> None:
    """config[section][key]가 비어 있지 않을 때만 절대경로로 바꿉니다.

    예:
    - paths.scenario_path: 기존 YAML scenario를 직접 읽을 때 사용
    - paths.source_data_path: 원본 Excel/CSV에서 scenario를 메모리로 만들 때 사용

    None/빈 문자열은 의도적으로 "사용하지 않음"을 뜻하므로 경로 변환하지 않습니다.
    """

    value = config.get(section, {}).get(key)
    if value in (None, ""):
        return
    config[section][key] = _resolve_path(str(value))


# LINE-BY-LINE: `load_config(config_path: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다.
def load_config(config_path: str):
    """YAML config를 읽고 경로를 정규화합니다.

    주의:
    - scenario_path
    - output_dir
    이 두 값은 코드 내부에서 바로 쓰기 편하도록 절대경로로 바꿉니다.
    """

    # LINE-BY-LINE: `config_file`에 `Path(config_path)` 결과를 저장합니다. 의미/사용: `config_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    config_file = Path(config_path)
    # LINE-BY-LINE: 조건 `not config_file.is_absolute()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not config_file.is_absolute():
        # LINE-BY-LINE: `config_file`에 `(project_root() / config_file).resolve()` 결과를 저장합니다. 의미/사용: `config_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        config_file = (project_root() / config_file).resolve()

    # LINE-BY-LINE: `config_file.open("r", encoding="utf-8") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with config_file.open("r", encoding="utf-8") as handle:
        # LINE-BY-LINE: `config`에 `yaml.safe_load(handle)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
        config = yaml.safe_load(handle)

    config.setdefault("paths", {})
    _resolve_optional_path(config, "paths", "scenario_path")
    _resolve_optional_path(config, "paths", "source_data_path")
    _resolve_optional_path(config, "paths", "output_dir")
    _resolve_optional_path(config, "paths", "generated_dir")
    # LINE-BY-LINE: 호출자에게 `config`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return config
