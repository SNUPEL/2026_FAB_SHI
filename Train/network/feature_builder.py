"""observation dict를 torch tensor로 변환합니다.

핵심 원칙:
- 입력 차원은 step마다 바뀌면 안 됩니다.
- 따라서 family / machine_type / downstream_bay one-hot 크기는
  "현재 후보 집합"이 아니라 "에피소드 전체 vocabulary" 기준으로 고정합니다.
"""

# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass
# LINE-BY-LINE: `typing` 모듈에서 `Dict, List`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `TensorObservation` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class TensorObservation:
    """네트워크 입력용 tensor 묶음.

    candidate_features:
      shape = (N, D)
      N개 후보 action 각각의 feature vector

    env_features:
      shape = (E,)
      현재 step의 전역 상태 벡터

    adjacency:
      shape = (N, N)
      후보들 사이 연결관계를 표현하는 인접행렬
    """

    # LINE-BY-LINE: `candidate_features`를 `torch.Tensor` 타입으로 선언합니다. 의미/사용: `TensorObservation.candidate_features` 필드/속성입니다. 사용: TensorObservation 객체를 만들거나 이후 로직에서 참조합니다.
    candidate_features: torch.Tensor
    # LINE-BY-LINE: `env_features`를 `torch.Tensor` 타입으로 선언합니다. 의미/사용: `TensorObservation.env_features` 필드/속성입니다. 사용: TensorObservation 객체를 만들거나 이후 로직에서 참조합니다.
    env_features: torch.Tensor
    # LINE-BY-LINE: `adjacency`를 `torch.Tensor` 타입으로 선언합니다. 의미/사용: `TensorObservation.adjacency` 필드/속성입니다. 사용: TensorObservation 객체를 만들거나 이후 로직에서 참조합니다.
    adjacency: torch.Tensor
    # LINE-BY-LINE: `action_ids`를 `List[str]` 타입으로 선언합니다. 의미/사용: `TensorObservation.action_ids` 필드/속성입니다. 사용: TensorObservation 객체를 만들거나 이후 로직에서 참조합니다.
    action_ids: List[str]

# LINE-BY-LINE: `_one_hot(index: int, size: int)` 함수를 정의합니다. 반환 타입: `List[float]`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def _one_hot(index: int, size: int) -> List[float]:
    """정수 index를 고정 길이 one-hot vector로 변환한다."""

    # LINE-BY-LINE: `vector`에 `[0.0] * size` 결과를 저장합니다. 의미/사용: `vector` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    vector = [0.0] * size
    # LINE-BY-LINE: 조건 `0 <= index < size`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if 0 <= index < size:
        # LINE-BY-LINE: `vector[index]`에 `1.0` 결과를 저장합니다. 의미/사용: `vector[index]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        vector[index] = 1.0
    # LINE-BY-LINE: 호출자에게 `vector`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return vector


# LINE-BY-LINE: `build_tensor_observation(observation: Dict, device: torch.device)` 함수를 정의합니다. 반환 타입: `TensorObservation`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def build_tensor_observation(observation: Dict, device: torch.device) -> TensorObservation:
    """현재 observation을 single-step network 입력으로 바꿉니다."""

    # LINE-BY-LINE: `actions`에 `observation["available_actions"]` 결과를 저장합니다. 의미/사용: `actions` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actions = observation["available_actions"]
    # LINE-BY-LINE: `vocab_sizes`에 `observation["vocab_sizes"]` 결과를 저장합니다. 의미/사용: `vocab_sizes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    vocab_sizes = observation["vocab_sizes"]
    # LINE-BY-LINE: `family_size`에 `max(int(vocab_sizes["family"]), 1)` 결과를 저장합니다. 의미/사용: `family_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    family_size = max(int(vocab_sizes["family"]), 1)
    # LINE-BY-LINE: `machine_type_size`에 `max(int(vocab_sizes["machine_type"]), 1)` 결과를 저장합니다. 의미/사용: `machine_type_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machine_type_size = max(int(vocab_sizes["machine_type"]), 1)
    # LINE-BY-LINE: `bay_size`에 `max(int(vocab_sizes["bay"]), 1)` 결과를 저장합니다. 의미/사용: `bay_size` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    bay_size = max(int(vocab_sizes["bay"]), 1)
    # LINE-BY-LINE: `candidate_dim`에 `15 + family_size + machine_type_size + bay_size` 결과를 저장합니다. 의미/사용: `candidate_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_dim = 15 + family_size + machine_type_size + bay_size

    # LINE-BY-LINE: 조건 `not actions`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if not actions:
        # LINE-BY-LINE: 호출자에게 `TensorObservation(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return TensorObservation(
            # LINE-BY-LINE: `candidate_features`에 `torch.zeros((0, candidate_dim), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `candidate_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_features=torch.zeros((0, candidate_dim), dtype=torch.float32, device=device),
            # LINE-BY-LINE: `env_features`에 `torch.zeros((7,), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `env_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            env_features=torch.zeros((7,), dtype=torch.float32, device=device),
            # LINE-BY-LINE: `adjacency`에 `torch.zeros((0, 0), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `adjacency` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            adjacency=torch.zeros((0, 0), dtype=torch.float32, device=device),
            # LINE-BY-LINE: `action_ids`에 `[]` 결과를 저장합니다. 의미/사용: `action_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            action_ids=[],
        )

    # env_features는 모든 후보가 공유하는 전역 상태입니다.
    # LINE-BY-LINE: `env_raw`에 `observation["env_features"]` 결과를 저장합니다. 의미/사용: `env_raw` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    env_raw = observation["env_features"]
    # LINE-BY-LINE: `env_features`에 `torch.tensor(` 결과를 저장합니다. 의미/사용: `env_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    env_features = torch.tensor(
        # LINE-BY-LINE: `[` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        [
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["current_time_norm"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["current_time_norm"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["makespan_norm"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["makespan_norm"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["remaining_job_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["remaining_job_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["available_machine_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["available_machine_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["average_machine_load_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["average_machine_load_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["max_downstream_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["max_downstream_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `env_raw["available_action_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            env_raw["available_action_ratio"],
        ],
        # LINE-BY-LINE: `dtype`에 `torch.float32` 결과를 저장합니다. 의미/사용: `dtype` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        dtype=torch.float32,
        # LINE-BY-LINE: `device`에 `device` 결과를 저장합니다. 의미/사용: `device`는 torch 연산 장치입니다. 예: cpu 또는 cuda.
        device=device,
    )

    # LINE-BY-LINE: `candidate_rows` 변수에 `[]` 결과를 저장합니다. 의미: `candidate_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_rows: List[List[float]] = []
    # LINE-BY-LINE: `adjacency`에 `torch.eye(len(actions), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `adjacency` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    adjacency = torch.eye(len(actions), dtype=torch.float32, device=device)

    # LINE-BY-LINE: `i, action in enumerate(actions)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for i, action in enumerate(actions):
        # LINE-BY-LINE: `family_idx`에 `int(action["family_index"])` 결과를 저장합니다. 의미/사용: `family_idx` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        family_idx = int(action["family_index"])
        # LINE-BY-LINE: `machine_type_idx`에 `int(action["machine_type_index"])` 결과를 저장합니다. 의미/사용: `machine_type_idx` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_type_idx = int(action["machine_type_index"])
        # LINE-BY-LINE: `bay_idx`에 `int(action["downstream_bay_index"])` 결과를 저장합니다. 의미/사용: `bay_idx` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bay_idx = int(action["downstream_bay_index"])

        # 아래 row 1개가 "후보 action 1개"를 뜻합니다.
        # 즉, job-machine pair를 수치 벡터로 바꾼 결과입니다.
        # LINE-BY-LINE: `row`에 `[` 결과를 저장합니다. 의미/사용: `row` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        row = [
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["estimated_minutes"] / 600.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["estimated_minutes"] / 600.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["processing_minutes"] / 600.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["processing_minutes"] / 600.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["blocked_minutes"] / 600.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["blocked_minutes"] / 600.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["changeover_minutes"] / 120.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["changeover_minutes"] / 120.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["priority_weight"] / 10.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["priority_weight"] / 10.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["soft_penalty"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["soft_penalty"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["thickness"] / 100.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["thickness"] / 100.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["plate_length"] / 50.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["plate_length"] / 50.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["downstream_priority_rank"] / 10.0` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["downstream_priority_rank"] / 10.0,
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["downstream_load_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["downstream_load_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["downstream_arrival_time_norm"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["downstream_arrival_time_norm"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["machine_speed_factor"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["machine_speed_factor"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["machine_load_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["machine_load_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["remaining_machine_capacity_ratio"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["remaining_machine_capacity_ratio"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `action["due_date_slack_norm"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            action["due_date_slack_norm"],
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `*_one_hot(family_idx, family_size)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            *_one_hot(family_idx, family_size),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `*_one_hot(machine_type_idx, machine_type_size)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            *_one_hot(machine_type_idx, machine_type_size),
            # LINE-BY-LINE: `현재 표현식(...)` 호출에 `*_one_hot(bay_idx, bay_size)` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            *_one_hot(bay_idx, bay_size),
        ]
        # LINE-BY-LINE: `candidate_rows.append(row)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        candidate_rows.append(row)

        # adjacency는 완전한 공정 그래프가 아니라
        # "서로 관련 있는 후보끼리 연결"하는 단순한 그래프입니다.
        # LINE-BY-LINE: `j, other in enumerate(actions)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for j, other in enumerate(actions):
            # LINE-BY-LINE: 조건 `i == j`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if i == j:
                # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
                continue
            # LINE-BY-LINE: 조건 `action["job_id"] == other["job_id"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if action["job_id"] == other["job_id"]:
                # LINE-BY-LINE: `adjacency[i, j]` 여러 변수에 `1.0` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                adjacency[i, j] = 1.0
            # LINE-BY-LINE: 조건 `action["machine_id"] == other["machine_id"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if action["machine_id"] == other["machine_id"]:
                # LINE-BY-LINE: `adjacency[i, j]` 여러 변수에 `1.0` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                adjacency[i, j] = 1.0
            # LINE-BY-LINE: 조건 `action["downstream_bay"] == other["downstream_bay"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if action["downstream_bay"] == other["downstream_bay"]:
                # LINE-BY-LINE: `adjacency[i, j]` 여러 변수에 `1.0` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                adjacency[i, j] = 1.0
            # LINE-BY-LINE: 조건 `action["family"] == other["family"]`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if action["family"] == other["family"]:
                # LINE-BY-LINE: `adjacency[i, j]` 여러 변수에 `1.0` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                adjacency[i, j] = 1.0

    # LINE-BY-LINE: `candidate_features`에 `torch.tensor(candidate_rows, dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `candidate_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_features = torch.tensor(candidate_rows, dtype=torch.float32, device=device)
    # LINE-BY-LINE: 호출자에게 `TensorObservation(`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return TensorObservation(
        # LINE-BY-LINE: `candidate_features`에 `candidate_features` 결과를 저장합니다. 의미/사용: `candidate_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate_features=candidate_features,
        # LINE-BY-LINE: `env_features`에 `env_features` 결과를 저장합니다. 의미/사용: `env_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        env_features=env_features,
        # LINE-BY-LINE: `adjacency`에 `adjacency` 결과를 저장합니다. 의미/사용: `adjacency` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        adjacency=adjacency,
        # LINE-BY-LINE: `action_ids`에 `[action["action_id"] for action in actions]` 결과를 저장합니다. 의미/사용: `action_ids` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_ids=[action["action_id"] for action in actions],
    )
