"""Pointer-style policy for Phase 1 block-to-Bay decisions.

This network mirrors the current Phase 1 MDP:
1. score remaining block candidates for `SELECT_BLOCK`;
2. score feasible Bay candidates for `SELECT_BAY`.

It is intentionally small. Training code can consume the JSONL trace from
`Utils.phase1_mdp` and use cross-entropy on the selected action index.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python 버전별 annotation 충돌을 줄입니다.
from __future__ import annotations

# LINE-BY-LINE: `torch`는 tensor 생성/연산에 사용합니다. 예: candidate feature matrix를 network input으로 넣습니다.
import torch
# LINE-BY-LINE: `torch.nn`은 PyTorch layer/module 정의에 사용합니다. 예: `nn.Module`, `nn.Linear`.
import torch.nn as nn

# LINE-BY-LINE: 프로젝트 공통 MLP builder를 가져옵니다. 사용: block/bay/env encoder를 같은 방식으로 생성합니다.
from .mlp import build_mlp


# LINE-BY-LINE: `Phase1PointerPolicy` class를 정의합니다. 용도: 과거 split action(`SELECT_BLOCK -> SELECT_BAY`) 학습용 network입니다.
class Phase1PointerPolicy(nn.Module):
    """Two-stage pointer network for Phase 1 self-labeling/imitation."""

    # LINE-BY-LINE: `__init__`은 입력 feature 차원과 hidden_dim을 받아 network layer를 생성합니다.
    def __init__(
        self,
        block_feature_dim: int,
        bay_feature_dim: int,
        env_feature_dim: int,
        hidden_dim: int = 128,
    ):
        """Create encoders and pointer scoring heads.

        Example:
        - block_feature_dim = len(PHASE1_BLOCK_FEATURE_NAMES)
        - bay_feature_dim = len(PHASE1_BAY_FEATURE_NAMES)
        - env_feature_dim = len(PHASE1_ENV_FEATURE_NAMES)
        """

        # LINE-BY-LINE: PyTorch 부모 class 초기화입니다. 이 호출이 있어야 layer 등록/저장이 정상 동작합니다.
        super().__init__()
        # LINE-BY-LINE: block 후보 feature 차원이 0 이하이면 학습 불가능하므로 즉시 실패시킵니다.
        _require_positive_dim(block_feature_dim, "block_feature_dim")
        # LINE-BY-LINE: Bay 후보 feature 차원이 0 이하이면 학습 불가능하므로 즉시 실패시킵니다.
        _require_positive_dim(bay_feature_dim, "bay_feature_dim")
        # LINE-BY-LINE: 환경 feature 차원이 0 이하이면 state encoder 입력이 없으므로 즉시 실패시킵니다.
        _require_positive_dim(env_feature_dim, "env_feature_dim")
        # LINE-BY-LINE: hidden_dim이 0 이하이면 MLP layer를 만들 수 없으므로 즉시 실패시킵니다.
        _require_positive_dim(hidden_dim, "hidden_dim")

        # LINE-BY-LINE: block feature 입력 차원을 checkpoint와 검증에서 재사용하기 위해 보존합니다.
        self.block_feature_dim = block_feature_dim
        # LINE-BY-LINE: Bay feature 입력 차원을 checkpoint와 검증에서 재사용하기 위해 보존합니다.
        self.bay_feature_dim = bay_feature_dim
        # LINE-BY-LINE: 환경 feature 입력 차원을 checkpoint와 검증에서 재사용하기 위해 보존합니다.
        self.env_feature_dim = env_feature_dim
        # LINE-BY-LINE: 모든 encoder/query의 hidden dimension입니다.
        self.hidden_dim = hidden_dim

        # LINE-BY-LINE: block 후보 1개의 feature vector를 hidden vector로 바꾸는 encoder입니다.
        self.block_encoder = build_mlp(block_feature_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: Bay 후보 1개의 feature vector를 hidden vector로 바꾸는 encoder입니다.
        self.bay_encoder = build_mlp(bay_feature_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 Bay 부하/진행률 등 환경 vector를 hidden vector로 바꾸는 encoder입니다.
        self.env_encoder = build_mlp(env_feature_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 전체 block 후보 평균 hidden과 환경 hidden을 합쳐 block 선택 query를 만듭니다.
        self.block_query = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: Bay 후보 평균, 환경, 선택된 block 정보를 합쳐 Bay 선택 query를 만듭니다.
        self.bay_query = build_mlp(hidden_dim * 3, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 각 block 후보 hidden을 scalar logit으로 바꾸는 pointer head입니다.
        self.block_pointer = nn.Linear(hidden_dim, 1)
        # LINE-BY-LINE: 각 Bay 후보 hidden을 scalar logit으로 바꾸는 pointer head입니다.
        self.bay_pointer = nn.Linear(hidden_dim, 1)

    # LINE-BY-LINE: `score_blocks`는 남은 block 후보 각각의 선택 점수(logit)를 반환합니다.
    def score_blocks(self, block_features: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        """Return logits for `SELECT_BLOCK` candidates.

        Input shapes:
        - `block_features`: `(num_block_candidates, block_feature_dim)`
        - `env_features`: `(env_feature_dim,)`

        Output shape:
        - `(num_block_candidates,)`
        """

        # LINE-BY-LINE: block 후보 matrix shape가 `(후보 수, block_feature_dim)`인지 검사합니다.
        _require_candidate_tensor(block_features, self.block_feature_dim, "block_features")
        # LINE-BY-LINE: 환경 vector shape가 `(env_feature_dim,)`인지 검사합니다.
        _require_env_tensor(env_features, self.env_feature_dim)
        # LINE-BY-LINE: 모든 block 후보를 같은 encoder에 통과시켜 hidden matrix를 만듭니다.
        block_hidden = self.block_encoder(block_features)
        # LINE-BY-LINE: 현재 환경 상태를 hidden vector로 만듭니다.
        env_hidden = self.env_encoder(env_features)
        # LINE-BY-LINE: 후보 전체의 평균을 전역 후보 context로 사용합니다.
        global_hidden = block_hidden.mean(dim=0)
        # LINE-BY-LINE: 전역 후보 context와 환경 context를 합쳐 현재 step의 query를 만듭니다.
        query = self.block_query(torch.cat([global_hidden, env_hidden], dim=-1))
        # LINE-BY-LINE: 각 후보 hidden과 query의 compatibility를 scalar logit으로 변환합니다.
        return self.block_pointer(torch.tanh(block_hidden + query.unsqueeze(0))).squeeze(-1)

    # LINE-BY-LINE: `score_bays`는 선택된 block을 어느 Bay에 넣을지 Bay 후보별 점수(logit)를 반환합니다.
    def score_bays(
        self,
        bay_features: torch.Tensor,
        env_features: torch.Tensor,
        selected_block_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits for `SELECT_BAY` candidates.

        Input shapes:
        - `bay_features`: `(num_bay_candidates, bay_feature_dim)`
        - `env_features`: `(env_feature_dim,)`
        - `selected_block_features`: `(block_feature_dim,)`

        Output shape:
        - `(num_bay_candidates,)`
        """

        # LINE-BY-LINE: Bay 후보 matrix shape가 `(후보 수, bay_feature_dim)`인지 검사합니다.
        _require_candidate_tensor(bay_features, self.bay_feature_dim, "bay_features")
        # LINE-BY-LINE: 환경 vector shape가 `(env_feature_dim,)`인지 검사합니다.
        _require_env_tensor(env_features, self.env_feature_dim)
        # LINE-BY-LINE: 선택된 block feature가 1차원 vector인지 검사합니다.
        _require_vector_tensor(selected_block_features, self.block_feature_dim, "selected_block_features")
        # LINE-BY-LINE: 모든 Bay 후보를 hidden matrix로 변환합니다.
        bay_hidden = self.bay_encoder(bay_features)
        # LINE-BY-LINE: 현재 환경 상태를 hidden vector로 변환합니다.
        env_hidden = self.env_encoder(env_features)
        # LINE-BY-LINE: 선택된 block feature도 block encoder에 통과시켜 같은 hidden space로 맞춥니다.
        block_hidden = self.block_encoder(selected_block_features.unsqueeze(0)).squeeze(0)
        # LINE-BY-LINE: Bay 후보 전체 평균을 Bay 후보 context로 사용합니다.
        bay_global_hidden = bay_hidden.mean(dim=0)
        # LINE-BY-LINE: Bay context, 환경 context, 선택 block context를 합쳐 Bay 선택 query를 만듭니다.
        query = self.bay_query(torch.cat([bay_global_hidden, env_hidden, block_hidden], dim=-1))
        # LINE-BY-LINE: 각 Bay 후보 hidden과 query의 compatibility를 scalar logit으로 변환합니다.
        return self.bay_pointer(torch.tanh(bay_hidden + query.unsqueeze(0))).squeeze(-1)


# LINE-BY-LINE: `Phase1PairPointerPolicy` class를 정의합니다. 용도: 현재 권장 구조인 직접 pair action `SELECT_PAIR(block,bay)`용 network입니다.
class Phase1PairPointerPolicy(nn.Module):
    """Pointer network for direct `SELECT_PAIR(block, bay)` decisions."""

    # LINE-BY-LINE: `__init__`은 pair feature 차원과 env feature 차원을 받아 pair scoring network를 만듭니다.
    def __init__(
        self,
        pair_feature_dim: int,
        env_feature_dim: int,
        hidden_dim: int = 128,
    ):
        """Create the pair encoder and pointer scorer."""

        # LINE-BY-LINE: PyTorch 부모 class 초기화입니다. layer 등록과 checkpoint 저장에 필요합니다.
        super().__init__()
        # LINE-BY-LINE: pair 후보 feature 차원이 유효한지 검사합니다. 예: 현재 Phase 1 pair feature는 21차원입니다.
        _require_positive_dim(pair_feature_dim, "pair_feature_dim")
        # LINE-BY-LINE: 환경 feature 차원이 유효한지 검사합니다. 예: 현재 Phase 1 env feature는 6차원입니다.
        _require_positive_dim(env_feature_dim, "env_feature_dim")
        # LINE-BY-LINE: hidden dimension이 유효한지 검사합니다.
        _require_positive_dim(hidden_dim, "hidden_dim")
        # LINE-BY-LINE: pair feature 차원을 저장해 inference/checkpoint 검증에 사용합니다.
        self.pair_feature_dim = pair_feature_dim
        # LINE-BY-LINE: 환경 feature 차원을 저장해 inference/checkpoint 검증에 사용합니다.
        self.env_feature_dim = env_feature_dim
        # LINE-BY-LINE: 내부 hidden vector 차원입니다.
        self.hidden_dim = hidden_dim
        # LINE-BY-LINE: `(block,bay)` 후보 feature를 hidden vector로 변환합니다.
        self.pair_encoder = build_mlp(pair_feature_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 현재 환경 feature를 hidden vector로 변환합니다.
        self.env_encoder = build_mlp(env_feature_dim, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 전체 pair 후보 context와 환경 context를 합쳐 pointer query를 만듭니다.
        self.query = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        # LINE-BY-LINE: 각 pair 후보 hidden을 scalar logit으로 바꾸는 pointer head입니다.
        self.pointer = nn.Linear(hidden_dim, 1)

    # LINE-BY-LINE: `score_pairs`는 현재 step에서 가능한 모든 `(block,bay)` 후보의 점수(logit)를 반환합니다.
    def score_pairs(self, pair_features: torch.Tensor, env_features: torch.Tensor) -> torch.Tensor:
        """Return logits for `(block, bay)` pair candidates."""

        # LINE-BY-LINE: pair 후보 matrix shape가 `(후보 수, pair_feature_dim)`인지 검사합니다.
        _require_candidate_tensor(pair_features, self.pair_feature_dim, "pair_features")
        # LINE-BY-LINE: 환경 vector shape가 `(env_feature_dim,)`인지 검사합니다.
        _require_env_tensor(env_features, self.env_feature_dim)
        # LINE-BY-LINE: 모든 pair 후보를 hidden matrix로 변환합니다.
        pair_hidden = self.pair_encoder(pair_features)
        # LINE-BY-LINE: 현재 환경 상태를 hidden vector로 변환합니다.
        env_hidden = self.env_encoder(env_features)
        # LINE-BY-LINE: 후보 전체 평균을 전역 pair 후보 context로 사용합니다.
        global_hidden = pair_hidden.mean(dim=0)
        # LINE-BY-LINE: pair context와 env context를 합쳐 현재 decision query를 만듭니다.
        query = self.query(torch.cat([global_hidden, env_hidden], dim=-1))
        # LINE-BY-LINE: 각 pair 후보 hidden과 query의 compatibility를 scalar logit으로 변환합니다.
        return self.pointer(torch.tanh(pair_hidden + query.unsqueeze(0))).squeeze(-1)


# LINE-BY-LINE: dimension 값이 양수인지 검사하는 공통 guard입니다. 입력 오류를 fallback 없이 즉시 드러냅니다.
def _require_positive_dim(value: int, name: str) -> None:
    """Reject invalid model dimensions at construction time."""

    # LINE-BY-LINE: 0 이하 차원은 layer 생성이 불가능하므로 원인을 출력하고 예외를 발생시킵니다.
    if int(value) <= 0:
        print(f"[ERROR][phase1_pointer._require_positive_dim] cause=non_positive_dim name={name} value={value}")
        raise ValueError(f"{name} must be positive")


# LINE-BY-LINE: 후보 feature tensor가 2차원 matrix인지 검사하는 공통 guard입니다.
def _require_candidate_tensor(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Validate candidate feature matrix shape."""

    # LINE-BY-LINE: 후보가 없거나 마지막 차원이 feature 차원과 다르면 학습/추론이 불가능하므로 실패시킵니다.
    if tensor.dim() != 2 or tensor.size(-1) != expected_dim or tensor.size(0) == 0:
        print(
            "[ERROR][phase1_pointer._require_candidate_tensor] "
            f"cause=invalid_candidate_tensor name={name} shape={tuple(tensor.shape)} "
            f"expected_last_dim={expected_dim}"
        )
        raise RuntimeError(f"invalid candidate tensor: {name}")


# LINE-BY-LINE: 환경 feature tensor가 1차원 vector인지 검사하는 wrapper입니다.
def _require_env_tensor(tensor: torch.Tensor, expected_dim: int) -> None:
    """Validate environment feature vector shape."""

    # LINE-BY-LINE: 실제 검증은 vector 공통 guard에 위임합니다.
    _require_vector_tensor(tensor, expected_dim, "env_features")


# LINE-BY-LINE: 단일 feature vector shape를 검사하는 공통 guard입니다.
def _require_vector_tensor(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Validate one feature vector shape."""

    # LINE-BY-LINE: 1차원 vector가 아니거나 길이가 다르면 원인을 출력하고 예외를 발생시킵니다.
    if tensor.dim() != 1 or tensor.size(0) != expected_dim:
        print(
            "[ERROR][phase1_pointer._require_vector_tensor] "
            f"cause=invalid_vector_tensor name={name} shape={tuple(tensor.shape)} "
            f"expected_dim={expected_dim}"
        )
        raise RuntimeError(f"invalid vector tensor: {name}")
