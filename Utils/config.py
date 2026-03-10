"""config 로딩 유틸.

초보자용 포인트:
- config.yaml 안의 경로는 보통 상대경로로 적습니다.
- 이 파일이 그 상대경로를 프로젝트 루트 기준 절대경로로 바꿔줍니다.
- 그래서 어디서 실행하든 같은 파일을 읽게 됩니다.
"""

from pathlib import Path

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


def load_config(config_path: str):
    """YAML config를 읽고 경로를 정규화합니다.

    주의:
    - scenario_path
    - output_dir
    이 두 값은 코드 내부에서 바로 쓰기 편하도록 절대경로로 바꿉니다.
    """

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (project_root() / config_file).resolve()

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["paths"]["scenario_path"] = _resolve_path(config["paths"]["scenario_path"])
    config["paths"]["output_dir"] = _resolve_path(config["paths"]["output_dir"])
    return config
