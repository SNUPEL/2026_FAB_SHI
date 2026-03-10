"""Candidate graph 기반 policy-value network.

구조 요약:
1. 후보 action feature -> candidate encoder MLP
2. 환경 feature -> env encoder MLP
3. 후보 간 관계 -> 간단 GNN message passing
4. 후보 전체 평균 -> global context
5. global context + candidate hidden -> pointer style logits
6. 같은 hidden으로 value head 계산
"""

from typing import Dict

import torch
import torch.nn as nn

from .feature_builder import build_tensor_observation
from .gnn import GraphMessagePassingLayer
from .mlp import build_mlp


class CandidateGraphPolicyValueNet(nn.Module):
    """PMSP형 action 후보를 직접 점수화하는 network.

    구조:
    1. candidate feature -> candidate encoder MLP
    2. env feature -> env encoder MLP
    3. 간단한 graph message passing
    4. global context(mean pooling) 생성
    5. pointer-style logits 계산
    6. value head 계산
    """

    def __init__(
        self,
        candidate_input_dim: int,
        env_input_dim: int,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.candidate_encoder = build_mlp(candidate_input_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.env_encoder = build_mlp(env_input_dim, [hidden_dim], hidden_dim)
        self.gnn_layers = nn.ModuleList([GraphMessagePassingLayer(hidden_dim) for _ in range(num_gnn_layers)])

        self.query_projection = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        self.candidate_projection = build_mlp(hidden_dim, [hidden_dim], hidden_dim)
        self.pointer_vector = nn.Linear(hidden_dim, 1)
        self.value_head = build_mlp(hidden_dim * 2, [hidden_dim], 1)

    def forward(self, observation: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        # observation dict를 먼저 tensor 묶음으로 바꿉니다.
        tensor_obs = build_tensor_observation(observation, device)
        candidate_features = tensor_obs.candidate_features
        env_features = tensor_obs.env_features
        adjacency = tensor_obs.adjacency

        if candidate_features.size(0) == 0:
            empty_logits = torch.zeros((0,), dtype=torch.float32, device=device)
            zero_value = torch.zeros((), dtype=torch.float32, device=device)
            return {
                "logits": empty_logits,
                "value": zero_value,
                "action_ids": tensor_obs.action_ids,
            }

        # 1) 후보별 feature를 hidden vector로 바꿉니다.
        candidate_hidden = self.candidate_encoder(candidate_features)

        # 2) 환경 feature도 같은 hidden 차원으로 바꿉니다.
        env_hidden = self.env_encoder(env_features)

        # 3) 후보끼리의 adjacency를 사용해서 여러 번 message passing 합니다.
        for layer in self.gnn_layers:
            candidate_hidden = layer(candidate_hidden, adjacency, env_hidden)

        # 4) 후보 전체를 평균내 전역 context를 만듭니다.
        global_hidden = candidate_hidden.mean(dim=0)

        # 5) query는 "이번 step의 전역 기준"입니다.
        query = self.query_projection(torch.cat([global_hidden, env_hidden], dim=-1))
        candidate_proj = self.candidate_projection(candidate_hidden)

        # 6) pointer logits:
        #    각 후보 hidden과 전역 query를 합쳐 후보별 점수를 만듭니다.
        logits = self.pointer_vector(torch.tanh(candidate_proj + query.unsqueeze(0))).squeeze(-1)

        # 7) value는 현재 상태 전체의 scalar 평가값입니다.
        value = self.value_head(torch.cat([global_hidden, env_hidden], dim=-1)).squeeze(-1)

        return {
            "logits": logits,
            "value": value,
            "action_ids": tensor_obs.action_ids,
        }
