"""학습용 시나리오 생성기.

현재는 가장 단순한 버전만 구현했습니다.
즉, 기존 job을 복제해서 더 큰 학습 문제를 만드는 수준입니다.

하지만 구조상 이 파일이 향후 담당할 역할은 더 큽니다.

- 절단 실적 데이터 분포 기반 샘플링
- 설비 상태 랜덤화
- 날짜별 capacity variation 반영
- due date / priority / downstream bay 조합 생성
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict

import yaml


def generate_scenario_from_template(template_scenario: Dict, duplicate_jobs: int = 1) -> Dict:
    """샘플 시나리오를 바탕으로 학습용 episode를 생성합니다.

    현재는 기본 생성기이므로
    - 기존 job을 복제
    - job_id만 바꾸는 수준으로 단순화했습니다.

    실제 구현에서는 아래가 들어가야 합니다.
    - 택트타임 로직 기반 처리시간 생성
    - 작업 pool 크기 변경
    - 긴급도 / 납기 / 후공정 베이 조합 생성
    - 설비 상태와 일일 가용시간 변화
    """

    # 원본 template을 그대로 유지하기 위해 deepcopy를 사용합니다.
    scenario = deepcopy(template_scenario)
    original_jobs = list(scenario.get("jobs", []))
    generated_jobs = []

    for copy_index in range(duplicate_jobs):
        for job in original_jobs:
            new_job = deepcopy(job)
            # 같은 작업을 여러 번 복제할 때는 ID가 겹치면 안 됩니다.
            new_job["job_id"] = f"{job['job_id']}_copy{copy_index + 1}"
            generated_jobs.append(new_job)

    scenario["jobs"] = generated_jobs
    return scenario


def save_scenario(scenario: Dict, output_path: str) -> None:
    """생성한 시나리오를 yaml 파일로 저장합니다."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scenario, handle, allow_unicode=True, sort_keys=False)
