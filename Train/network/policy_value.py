"""Candidate graph 기반 policy-value network.

구조 요약:
1. 후보 action feature -> candidate encoder MLP
2. 환경 feature -> env encoder MLP
3. 후보 간 관계 -> 간단 GNN message passing
4. 후보 전체 평균 -> global context
5. global context + candidate hidden -> pointer style logits
6. 같은 hidden으로 value head 계산
"""

# LINE-BY-LINE: `typing` 모듈에서 `Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn as nn` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn as nn

# LINE-BY-LINE: `.feature_builder` 모듈에서 `build_tensor_observation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .feature_builder import build_tensor_observation
# LINE-BY-LINE: `.gnn` 모듈에서 `GraphMessagePassingLayer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .gnn import GraphMessagePassingLayer
# LINE-BY-LINE: `.mlp` 모듈에서 `build_mlp`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .mlp import build_mlp


# LINE-BY-LINE: `CandidateGraphPolicyValueNet` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
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

    # LINE-BY-LINE: `__init__` 함수를 정의합니다. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def __init__(
        # LINE-BY-LINE: `__init__(...)` 호출에 `self` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        self,
        # LINE-BY-LINE: `candidate_input_dim`를 `int,` 타입으로 선언합니다. 의미/사용: `CandidateGraphPolicyValueNet.candidate_input_dim` 필드/속성입니다. 사용: CandidateGraphPolicyValueNet 객체를 만들거나 이후 로직에서 참조합니다.
        candidate_input_dim: int,
        # LINE-BY-LINE: `env_input_dim`를 `int,` 타입으로 선언합니다. 의미/사용: `CandidateGraphPolicyValueNet.env_input_dim` 필드/속성입니다. 사용: CandidateGraphPolicyValueNet 객체를 만들거나 이후 로직에서 참조합니다.
        env_input_dim: int,
        # LINE-BY-LINE: `hidden_dim` 변수에 `128` 결과를 저장합니다. 의미: `hidden_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hidden_dim: int = 128,
        # LINE-BY-LINE: `num_gnn_layers` 변수에 `2` 결과를 저장합니다. 의미: `num_gnn_layers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        num_gnn_layers: int = 2,
    # LINE-BY-LINE: `):` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
    ):
        """candidate/env encoder, GNN layer, policy/value head를 초기화한다."""

        # LINE-BY-LINE: `super().__init__()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        super().__init__()
        # LINE-BY-LINE: 현재 객체의 `hidden_dim` 속성에 `hidden_dim` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.hidden_dim = hidden_dim
        # LINE-BY-LINE: 현재 객체의 `candidate_encoder` 속성에 `build_mlp(candidate_input_dim, [hidden_dim, hidden_dim], hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.candidate_encoder = build_mlp(candidate_input_dim, [hidden_dim, hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `env_encoder` 속성에 `build_mlp(env_input_dim, [hidden_dim], hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.env_encoder = build_mlp(env_input_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `gnn_layers` 속성에 `nn.ModuleList([GraphMessagePassingLayer(hidden_dim) for _ in range(num_gnn_layers)])` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.gnn_layers = nn.ModuleList([GraphMessagePassingLayer(hidden_dim) for _ in range(num_gnn_layers)])

        # LINE-BY-LINE: 현재 객체의 `query_projection` 속성에 `build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.query_projection = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `candidate_projection` 속성에 `build_mlp(hidden_dim, [hidden_dim], hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.candidate_projection = build_mlp(hidden_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `pointer_vector` 속성에 `nn.Linear(hidden_dim, 1)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.pointer_vector = nn.Linear(hidden_dim, 1)
        # LINE-BY-LINE: 현재 객체의 `value_head` 속성에 `build_mlp(hidden_dim * 2, [hidden_dim], 1)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.value_head = build_mlp(hidden_dim * 2, [hidden_dim], 1)

    # LINE-BY-LINE: `forward(self, observation: Dict, device: torch.device)` 함수를 정의합니다. 반환 타입: `Dict[str, torch.Tensor]`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def forward(self, observation: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """observation dict를 받아 action logits와 state value를 계산한다."""

        # observation dict를 먼저 tensor 묶음으로 바꿉니다.
        # LINE-BY-LINE: `tensor_obs`에 `build_tensor_observation(observation, device)` 결과를 저장합니다. 의미/사용: `tensor_obs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        tensor_obs = build_tensor_observation(observation, device)
        # LINE-BY-LINE: `candidate_features`에 `tensor_obs.candidate_features` 결과를 저장합니다. 의미/사용: `candidate_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate_features = tensor_obs.candidate_features
        # LINE-BY-LINE: `env_features`에 `tensor_obs.env_features` 결과를 저장합니다. 의미/사용: `env_features` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        env_features = tensor_obs.env_features
        # LINE-BY-LINE: `adjacency`에 `tensor_obs.adjacency` 결과를 저장합니다. 의미/사용: `adjacency` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        adjacency = tensor_obs.adjacency

        # LINE-BY-LINE: 조건 `candidate_features.size(0) == 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if candidate_features.size(0) == 0:
            # LINE-BY-LINE: `empty_logits`에 `torch.zeros((0,), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `empty_logits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            empty_logits = torch.zeros((0,), dtype=torch.float32, device=device)
            # LINE-BY-LINE: `zero_value`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `zero_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            zero_value = torch.zeros((), dtype=torch.float32, device=device)
            # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
            return {
                # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `logits` 키에 `empty_logits` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "logits": empty_logits,
                # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `value` 키에 `zero_value` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "value": zero_value,
                # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `action_ids` 키에 `tensor_obs.action_ids` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "action_ids": tensor_obs.action_ids,
            }

        # 1) 후보별 feature를 hidden vector로 바꿉니다.
        # LINE-BY-LINE: `candidate_hidden`에 `self.candidate_encoder(candidate_features)` 결과를 저장합니다. 의미/사용: `candidate_hidden` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate_hidden = self.candidate_encoder(candidate_features)

        # 2) 환경 feature도 같은 hidden 차원으로 바꿉니다.
        # LINE-BY-LINE: `env_hidden`에 `self.env_encoder(env_features)` 결과를 저장합니다. 의미/사용: `env_hidden` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        env_hidden = self.env_encoder(env_features)

        # 3) 후보끼리의 adjacency를 사용해서 여러 번 message passing 합니다.
        # LINE-BY-LINE: `layer in self.gnn_layers` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for layer in self.gnn_layers:
            # LINE-BY-LINE: `candidate_hidden`에 `layer(candidate_hidden, adjacency, env_hidden)` 결과를 저장합니다. 의미/사용: `candidate_hidden` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            candidate_hidden = layer(candidate_hidden, adjacency, env_hidden)

        # 4) 후보 전체를 평균내 전역 context를 만듭니다.
        # LINE-BY-LINE: `global_hidden`에 `candidate_hidden.mean(dim=0)` 결과를 저장합니다. 의미/사용: `global_hidden` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        global_hidden = candidate_hidden.mean(dim=0)

        # 5) query는 "이번 step의 전역 기준"입니다.
        # LINE-BY-LINE: `query`에 `self.query_projection(torch.cat([global_hidden, env_hidden], dim=-1))` 결과를 저장합니다. 의미/사용: `query` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        query = self.query_projection(torch.cat([global_hidden, env_hidden], dim=-1))
        # LINE-BY-LINE: `candidate_proj`에 `self.candidate_projection(candidate_hidden)` 결과를 저장합니다. 의미/사용: `candidate_proj` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate_proj = self.candidate_projection(candidate_hidden)

        # 6) pointer logits:
        #    각 후보 hidden과 전역 query를 합쳐 후보별 점수를 만듭니다.
        # LINE-BY-LINE: `logits`에 `self.pointer_vector(torch.tanh(candidate_proj + query.unsqueeze(0))).squeeze(-1)` 결과를 저장합니다. 의미/사용: `logits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        logits = self.pointer_vector(torch.tanh(candidate_proj + query.unsqueeze(0))).squeeze(-1)

        # 7) value는 현재 상태 전체의 scalar 평가값입니다.
        # LINE-BY-LINE: `value`에 `self.value_head(torch.cat([global_hidden, env_hidden], dim=-1)).squeeze(-1)` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value = self.value_head(torch.cat([global_hidden, env_hidden], dim=-1)).squeeze(-1)

        # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
        return {
            # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `logits` 키에 `logits` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "logits": logits,
            # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `value` 키에 `value` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "value": value,
            # LINE-BY-LINE: `forward`에서 반환/저장할 dict의 `action_ids` 키에 `tensor_obs.action_ids` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_ids": tensor_obs.action_ids,
        }
