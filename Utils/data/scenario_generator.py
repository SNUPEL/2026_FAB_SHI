"""학습용 시나리오 생성기.

현재는 가장 단순한 버전만 구현했습니다.
즉, 기존 job을 복제해서 더 큰 학습 문제를 만드는 수준입니다.

하지만 구조상 이 파일이 향후 담당할 역할은 더 큽니다.

- 절단 실적 데이터 분포 기반 샘플링
- 설비 상태 랜덤화
- 날짜별 capacity variation 반영
- due date / priority / downstream bay 조합 생성
"""

# LINE-BY-LINE: `copy` 모듈에서 `deepcopy`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from copy import deepcopy
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict

# LINE-BY-LINE: `yaml` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import yaml


# LINE-BY-LINE: `generate_scenario_from_template(template_scenario: Dict, duplicate_jobs: int = 1)` 함수를 정의합니다. 반환 타입: `Dict`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
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
    # LINE-BY-LINE: `scenario`에 `deepcopy(template_scenario)` 결과를 저장합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario = deepcopy(template_scenario)
    # LINE-BY-LINE: `original_jobs`에 `list(scenario.get("jobs", []))` 결과를 저장합니다. 의미/사용: `original_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    original_jobs = list(scenario.get("jobs", []))
    # LINE-BY-LINE: `generated_jobs`에 `[]` 결과를 저장합니다. 의미/사용: `generated_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    generated_jobs = []

    # LINE-BY-LINE: `copy_index in range(duplicate_jobs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for copy_index in range(duplicate_jobs):
        # LINE-BY-LINE: `job in original_jobs` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for job in original_jobs:
            # LINE-BY-LINE: `new_job`에 `deepcopy(job)` 결과를 저장합니다. 의미/사용: `new_job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            new_job = deepcopy(job)
            # 같은 작업을 여러 번 복제할 때는 ID가 겹치면 안 됩니다.
            # LINE-BY-LINE: `new_job["job_id"]`에 `f"{job['job_id']}_copy{copy_index + 1}"` 결과를 저장합니다. 의미/사용: `new_job["job_id"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            new_job["job_id"] = f"{job['job_id']}_copy{copy_index + 1}"
            # LINE-BY-LINE: `generated_jobs.append(new_job)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            generated_jobs.append(new_job)

    # LINE-BY-LINE: `scenario["jobs"]`에 `generated_jobs` 결과를 저장합니다. 의미/사용: `scenario["jobs"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario["jobs"] = generated_jobs
    # LINE-BY-LINE: 호출자에게 `scenario`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return scenario


# LINE-BY-LINE: `save_scenario(scenario: Dict, output_path: str)` 함수를 정의합니다. 반환 타입: `None`. 사용: 프로젝트의 해당 모듈에서 재사용됩니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def save_scenario(scenario: Dict, output_path: str) -> None:
    """생성한 시나리오를 yaml 파일로 저장합니다."""

    # LINE-BY-LINE: `output_file`에 `Path(output_path)` 결과를 저장합니다. 의미/사용: `output_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_file = Path(output_path)
    # LINE-BY-LINE: `output_file.parent.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `output_file.open("w", encoding="utf-8") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with output_file.open("w", encoding="utf-8") as handle:
        # LINE-BY-LINE: `yaml.safe_dump(scenario, handle, allow_unicode` 여러 변수에 `True, sort_keys=False)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        yaml.safe_dump(scenario, handle, allow_unicode=True, sort_keys=False)
