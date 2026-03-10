"""observation dict를 torch tensor로 변환합니다.

핵심 원칙:
- 입력 차원은 step마다 바뀌면 안 됩니다.
- 따라서 family / machine_type / downstream_bay one-hot 크기는
  "현재 후보 집합"이 아니라 "에피소드 전체 vocabulary" 기준으로 고정합니다.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
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

    candidate_features: torch.Tensor
    env_features: torch.Tensor
    adjacency: torch.Tensor
    action_ids: List[str]

def _one_hot(index: int, size: int) -> List[float]:
    vector = [0.0] * size
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def build_tensor_observation(observation: Dict, device: torch.device) -> TensorObservation:
    """현재 observation을 single-step network 입력으로 바꿉니다."""

    actions = observation["available_actions"]
    vocab_sizes = observation["vocab_sizes"]
    family_size = max(int(vocab_sizes["family"]), 1)
    machine_type_size = max(int(vocab_sizes["machine_type"]), 1)
    bay_size = max(int(vocab_sizes["bay"]), 1)
    candidate_dim = 11 + family_size + machine_type_size + bay_size

    if not actions:
        return TensorObservation(
            candidate_features=torch.zeros((0, candidate_dim), dtype=torch.float32, device=device),
            env_features=torch.zeros((7,), dtype=torch.float32, device=device),
            adjacency=torch.zeros((0, 0), dtype=torch.float32, device=device),
            action_ids=[],
        )

    # env_features는 모든 후보가 공유하는 전역 상태입니다.
    env_raw = observation["env_features"]
    env_features = torch.tensor(
        [
            env_raw["current_time_norm"],
            env_raw["makespan_norm"],
            env_raw["remaining_job_ratio"],
            env_raw["available_machine_ratio"],
            env_raw["average_machine_load_ratio"],
            env_raw["max_downstream_ratio"],
            env_raw["available_action_ratio"],
        ],
        dtype=torch.float32,
        device=device,
    )

    candidate_rows: List[List[float]] = []
    adjacency = torch.eye(len(actions), dtype=torch.float32, device=device)

    for i, action in enumerate(actions):
        family_idx = int(action["family_index"])
        machine_type_idx = int(action["machine_type_index"])
        bay_idx = int(action["downstream_bay_index"])

        # 아래 row 1개가 "후보 action 1개"를 뜻합니다.
        # 즉, job-machine pair를 수치 벡터로 바꾼 결과입니다.
        row = [
            action["estimated_minutes"] / 600.0,
            action["priority_weight"] / 10.0,
            action["soft_penalty"],
            action["thickness"] / 100.0,
            action["plate_length"] / 50.0,
            action["downstream_priority_rank"] / 10.0,
            action["downstream_load_ratio"],
            action["machine_speed_factor"],
            action["machine_load_ratio"],
            action["remaining_machine_capacity_ratio"],
            action["due_date_slack_norm"],
            *_one_hot(family_idx, family_size),
            *_one_hot(machine_type_idx, machine_type_size),
            *_one_hot(bay_idx, bay_size),
        ]
        candidate_rows.append(row)

        # adjacency는 완전한 공정 그래프가 아니라
        # "서로 관련 있는 후보끼리 연결"하는 단순한 그래프입니다.
        for j, other in enumerate(actions):
            if i == j:
                continue
            if action["job_id"] == other["job_id"]:
                adjacency[i, j] = 1.0
            if action["machine_id"] == other["machine_id"]:
                adjacency[i, j] = 1.0
            if action["downstream_bay"] == other["downstream_bay"]:
                adjacency[i, j] = 1.0
            if action["family"] == other["family"]:
                adjacency[i, j] = 1.0

    candidate_features = torch.tensor(candidate_rows, dtype=torch.float32, device=device)
    return TensorObservation(
        candidate_features=candidate_features,
        env_features=env_features,
        adjacency=adjacency,
        action_ids=[action["action_id"] for action in actions],
    )
