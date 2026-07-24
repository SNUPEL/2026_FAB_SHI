"""config 로딩 유틸.

초보자용 포인트:
- 프로젝트 YAML config 안의 경로는 보통 상대경로로 적습니다.
- 이 파일이 그 상대경로를 프로젝트 루트 기준 절대경로로 바꿔줍니다.
- 그래서 어디서 실행하든 같은 파일을 읽게 됩니다.
"""

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """프로젝트 루트 경로를 반환합니다."""

    return Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> str:
    """상대경로를 프로젝트 루트 기준 절대경로로 변환합니다."""

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((project_root() / path).resolve())


def _resolve_optional_path(config: dict[str, Any], section: str, key: str) -> None:
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


def load_config(config_path: str) -> dict[str, Any]:
    """YAML config를 읽고 경로를 정규화합니다.

    주의:
    - scenario_path
    - output_dir
    이 두 값은 코드 내부에서 바로 쓰기 편하도록 절대경로로 바꿉니다.
    """

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (project_root() / config_file).resolve()
    if not config_file.is_file():
        print(f"[ERROR][Utils.config.load_config] cause=config_not_found path={config_file}")
        raise RuntimeError(f"config file not found: {config_file}")
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        print(f"[ERROR][Utils.config.load_config] cause=invalid_yaml path={config_file} detail={exc}")
        raise RuntimeError(f"invalid YAML config: {config_file}") from exc
    if not isinstance(config, dict):
        print(
            "[ERROR][Utils.config.load_config] "
            f"cause=config_not_mapping path={config_file} type={type(config).__name__}"
        )
        raise RuntimeError(f"config root must be a mapping: {config_file}")
    if "paths" not in config or not isinstance(config["paths"], dict):
        print(
            "[ERROR][Utils.config.load_config] "
            f"cause=paths_not_mapping path={config_file} type={type(config.get('paths')).__name__}"
        )
        raise RuntimeError(f"config.paths must be a mapping: {config_file}")
    _resolve_optional_path(config, "paths", "scenario_path")
    _resolve_optional_path(config, "paths", "source_data_path")
    _resolve_optional_path(config, "paths", "output_dir")
    _resolve_optional_path(config, "paths", "generated_dir")
    return config
