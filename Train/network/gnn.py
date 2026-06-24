"""PyTorch만으로 만든 간단한 candidate graph encoder.

외부 GNN 라이브러리를 쓰지 않고,
adjacency matrix 기반 평균 집계만으로 구현한 최소 버전입니다.

의미:
- candidate i 주변의 이웃 후보 정보를 평균내서
  i의 hidden vector를 업데이트합니다.
"""

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn as nn` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn as nn


# LINE-BY-LINE: `GraphMessagePassingLayer` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class GraphMessagePassingLayer(nn.Module):
    """인접 행렬 기반 평균 집계 레이어."""

    # LINE-BY-LINE: `__init__(self, hidden_dim: int)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def __init__(self, hidden_dim: int):
        """message passing layer의 linear projection들을 초기화한다."""

        # LINE-BY-LINE: `super().__init__()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        super().__init__()
        # LINE-BY-LINE: 현재 객체의 `self_linear` 속성에 `nn.Linear(hidden_dim, hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `neighbor_linear` 속성에 `nn.Linear(hidden_dim, hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `env_linear` 속성에 `nn.Linear(hidden_dim, hidden_dim)` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.env_linear = nn.Linear(hidden_dim, hidden_dim)
        # LINE-BY-LINE: 현재 객체의 `activation` 속성에 `nn.ReLU()` 값을 저장합니다. 사용: 이후 메서드들이 같은 상태를 공유합니다.
        self.activation = nn.ReLU()

    # LINE-BY-LINE: `forward(self, hidden: torch.Tensor, adjacency: torch.Tensor, env_hidden: torch.Tensor)` 함수를 정의합니다. 반환 타입: `torch.Tensor`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor, env_hidden: torch.Tensor) -> torch.Tensor:
        """candidate hidden과 adjacency를 받아 이웃 평균 정보를 반영한다."""

        # 후보가 아예 없으면 그대로 반환합니다.
        # LINE-BY-LINE: 조건 `hidden.size(0) == 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if hidden.size(0) == 0:
            # LINE-BY-LINE: 호출자에게 `hidden`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
            return hidden

        # adjacency @ hidden / degree 형태로
        # 이웃 hidden의 평균을 구합니다.
        # LINE-BY-LINE: `degree`에 `adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)` 결과를 저장합니다. 의미/사용: `degree` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # LINE-BY-LINE: `neighbor_hidden`에 `adjacency @ hidden / degree` 결과를 저장합니다. 의미/사용: `neighbor_hidden` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        neighbor_hidden = adjacency @ hidden / degree

        # env_hidden은 모든 후보가 공유하는 전역 상태이므로
        # candidate 개수만큼 복제해서 더해줍니다.
        # LINE-BY-LINE: `env_expand`에 `env_hidden.unsqueeze(0).expand_as(hidden)` 결과를 저장합니다. 의미/사용: `env_expand` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        env_expand = env_hidden.unsqueeze(0).expand_as(hidden)
        # LINE-BY-LINE: `updated`에 `self.self_linear(hidden) + self.neighbor_linear(neighbor_hidden) + self.env_linear(env_expand)` 결과를 저장합니다. 의미/사용: `updated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        updated = self.self_linear(hidden) + self.neighbor_linear(neighbor_hidden) + self.env_linear(env_expand)
        # LINE-BY-LINE: 호출자에게 `self.activation(updated)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return self.activation(updated)
