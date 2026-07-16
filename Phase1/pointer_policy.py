"""MIXED Phase 1의 직접 ``SELECT_PAIR(block, bay)`` pointer policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from Train.network.mlp import build_mlp
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
)


class Phase1PairPointerPolicy(nn.Module):
    """가변 길이 ``(물리 블록-계열, Bay)`` 후보를 직접 점수화한다."""

    def __init__(
        self,
        pair_feature_dim: int,
        env_feature_dim: int,
        hidden_dim: int = 128,
        rule_profile: str = MULTI_SERIES_RULE_PROFILE,
        score_mode: str = "wo_first",
    ) -> None:
        super().__init__()
        _require_positive_dim(pair_feature_dim, "pair_feature_dim")
        _require_positive_dim(env_feature_dim, "env_feature_dim")
        _require_positive_dim(hidden_dim, "hidden_dim")
        if rule_profile != MULTI_SERIES_RULE_PROFILE or score_mode != "wo_first":
            print(
                "[ERROR][phase1_pointer.Phase1PairPointerPolicy] "
                f"cause=unsupported_policy_contract rule_profile={rule_profile} "
                f"score_mode={score_mode}"
            )
            raise RuntimeError("Phase 1 pair policy supports the MIXED wo_first contract only")

        self.pair_feature_dim = pair_feature_dim
        self.env_feature_dim = env_feature_dim
        self.hidden_dim = hidden_dim
        self.rule_profile = MULTI_SERIES_RULE_PROFILE
        self.score_mode = "wo_first"
        self.episode_scope_version = PHASE1_MULTI_SERIES_SCOPE_VERSION

        self.pair_encoder = build_mlp(pair_feature_dim, [hidden_dim], hidden_dim)
        self.env_encoder = build_mlp(env_feature_dim, [hidden_dim], hidden_dim)
        self.query = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        self.pointer = nn.Linear(hidden_dim, 1)

    def score_pairs(
        self,
        pair_features: torch.Tensor,
        env_features: torch.Tensor,
    ) -> torch.Tensor:
        """현재 step의 모든 feasible pair 후보 logit을 반환한다."""

        _require_candidate_tensor(pair_features, self.pair_feature_dim, "pair_features")
        _require_vector_tensor(env_features, self.env_feature_dim, "env_features")
        pair_hidden = self.pair_encoder(pair_features)
        env_hidden = self.env_encoder(env_features)
        query = self.query(torch.cat([pair_hidden.mean(dim=0), env_hidden], dim=-1))
        return self.pointer(torch.tanh(pair_hidden + query.unsqueeze(0))).squeeze(-1)


def _require_positive_dim(value: int, name: str) -> None:
    if int(value) <= 0:
        print(
            "[ERROR][phase1_pointer._require_positive_dim] "
            f"cause=non_positive_dim name={name} value={value}"
        )
        raise ValueError(f"{name} must be positive")


def _require_candidate_tensor(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str,
) -> None:
    if tensor.dim() != 2 or tensor.size(0) == 0 or tensor.size(-1) != expected_dim:
        print(
            "[ERROR][phase1_pointer._require_candidate_tensor] "
            f"cause=invalid_candidate_tensor name={name} shape={tuple(tensor.shape)} "
            f"expected_last_dim={expected_dim}"
        )
        raise RuntimeError(f"invalid candidate tensor: {name}")


def _require_vector_tensor(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str,
) -> None:
    if tensor.dim() != 1 or tensor.size(0) != expected_dim:
        print(
            "[ERROR][phase1_pointer._require_vector_tensor] "
            f"cause=invalid_vector_tensor name={name} shape={tuple(tensor.shape)} "
            f"expected_dim={expected_dim}"
        )
        raise RuntimeError(f"invalid vector tensor: {name}")
