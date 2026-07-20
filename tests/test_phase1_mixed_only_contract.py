"""Phase 1 공개 학습 경로가 MIXED 계약 하나만 제공하는지 검증한다."""

from __future__ import annotations

import unittest

import Phase1.pointer_policy as pointer_policy
from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
    phase1_pair_feature_schema,
)
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Utils.learning.phase_graph_mdp import (
    PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES,
)
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
)


class Phase1MixedOnlyContractTests(unittest.TestCase):
    def test_split_action_pointer_is_not_public(self) -> None:
        self.assertFalse(hasattr(pointer_policy, "Phase1PointerPolicy"))

    def test_pair_schema_is_the_single_mixed_schema(self) -> None:
        self.assertEqual(phase1_pair_feature_schema(), {
            "pair": PHASE1_PAIR_FEATURE_NAMES,
            "env": PHASE1_PAIR_ENV_FEATURE_NAMES,
        })
        self.assertEqual(
            PHASE1_PAIR_FEATURE_NAMES,
            list(PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES),
        )

    def test_pair_policy_defaults_to_mixed_contract(self) -> None:
        model = Phase1PairPointerPolicy(
            pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
            env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
            hidden_dim=16,
        )
        self.assertEqual(model.rule_profile, MULTI_SERIES_RULE_PROFILE)
        self.assertEqual(model.score_mode, "wo_first")
        self.assertEqual(model.objective_scope, "shared_and_series")
        self.assertEqual(model.episode_scope_version, PHASE1_MULTI_SERIES_SCOPE_VERSION)


if __name__ == "__main__":
    unittest.main()
