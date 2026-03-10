"""입출력 유틸.

현재는 시나리오 YAML 로딩만 담당하지만,
향후에는 아래도 같이 넣을 수 있습니다.

- 결과 CSV 저장
- 로그 요약 저장
- checkpoint 메타데이터 저장
"""

from pathlib import Path

import yaml


def load_scenario(scenario_path: str):
    """시나리오 yaml을 읽습니다.

    시나리오는 현재 프로젝트의 가장 바깥 입력 단위입니다.
    즉, 설비 / bay / 작업 목록을 한 번에 읽어 환경으로 넘깁니다.
    """

    with Path(scenario_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
