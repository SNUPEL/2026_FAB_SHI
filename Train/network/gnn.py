"""PyTorch만으로 만든 간단한 candidate graph encoder.

외부 GNN 라이브러리를 쓰지 않고,
adjacency matrix 기반 평균 집계만으로 구현한 최소 버전입니다.

의미:
- candidate i 주변의 이웃 후보 정보를 평균내서
  i의 hidden vector를 업데이트합니다.
"""

import torch
import torch.nn as nn


class GraphMessagePassingLayer(nn.Module):
    """인접 행렬 기반 평균 집계 레이어."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.env_linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor, env_hidden: torch.Tensor) -> torch.Tensor:
        # 후보가 아예 없으면 그대로 반환합니다.
        if hidden.size(0) == 0:
            return hidden

        # adjacency @ hidden / degree 형태로
        # 이웃 hidden의 평균을 구합니다.
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbor_hidden = adjacency @ hidden / degree

        # env_hidden은 모든 후보가 공유하는 전역 상태이므로
        # candidate 개수만큼 복제해서 더해줍니다.
        env_expand = env_hidden.unsqueeze(0).expand_as(hidden)
        updated = self.self_linear(hidden) + self.neighbor_linear(neighbor_hidden) + self.env_linear(env_expand)
        return self.activation(updated)
