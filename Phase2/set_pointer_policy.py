"""환경이 선택한 Machine context에서 W/O를 고르는 Phase 2 pointer policy."""

from __future__ import annotations

import torch
from torch import nn

from Phase2.state import (
    PHASE2_BAY_CONTEXT_FEATURE_NAMES,
    PHASE2_MACHINE_NODE_FEATURE_NAMES,
    PHASE2_OPEN_BATCH_FEATURE_NAMES,
    PHASE2_PROJECTED_FEATURE_NAMES,
    PHASE2_STATE_SCHEMA_VERSION,
    PHASE2_WO_NODE_FEATURE_NAMES,
    Phase2PolicyState,
)
from Train.network.mlp import build_mlp


class Phase2SetPointerPolicy(nn.Module):
    """Shared set encoders와 pointer compatibility로 action logit을 계산한다."""

    schema_version = PHASE2_STATE_SCHEMA_VERSION

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        if hidden_dim <= 0:
            print(f"[ERROR][Phase2SetPointerPolicy.__init__] cause=invalid_hidden_dim value={hidden_dim}")
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.machine_encoder = build_mlp(len(PHASE2_MACHINE_NODE_FEATURE_NAMES), [hidden_dim], hidden_dim)
        self.wo_encoder = build_mlp(len(PHASE2_WO_NODE_FEATURE_NAMES), [hidden_dim], hidden_dim)
        self.bay_encoder = build_mlp(len(PHASE2_BAY_CONTEXT_FEATURE_NAMES), [hidden_dim], hidden_dim)
        self.open_batch_encoder = build_mlp(len(PHASE2_OPEN_BATCH_FEATURE_NAMES), [hidden_dim], hidden_dim)
        self.projected_encoder = build_mlp(len(PHASE2_PROJECTED_FEATURE_NAMES), [hidden_dim], hidden_dim)
        self.context_encoder = build_mlp(hidden_dim * 6, [hidden_dim], hidden_dim)
        self.query_encoder = build_mlp(hidden_dim * 2, [hidden_dim], hidden_dim)
        self.pointer = nn.Linear(hidden_dim, 1)

    def forward(self, state: Phase2PolicyState) -> torch.Tensor:
        _validate_state(state)
        device = next(self.parameters()).device
        machine = torch.tensor(state.machine_node_features, dtype=torch.float32, device=device)
        wo = torch.tensor(state.wo_node_features, dtype=torch.float32, device=device)
        bay = torch.tensor(state.bay_context_features, dtype=torch.float32, device=device)
        open_batch = torch.tensor(state.open_batch_features, dtype=torch.float32, device=device)
        projected = torch.tensor(state.action_projected_features, dtype=torch.float32, device=device)

        machine_hidden = self.machine_encoder(machine)
        wo_hidden = self.wo_encoder(wo)
        bay_hidden = self.bay_encoder(bay)
        open_hidden = self.open_batch_encoder(open_batch)
        projected_hidden = self.projected_encoder(projected)
        context = self.context_encoder(
            torch.cat(
                [
                    bay_hidden,
                    open_hidden,
                    machine_hidden.mean(dim=0),
                    machine_hidden.max(dim=0).values,
                    wo_hidden.mean(dim=0),
                    wo_hidden.max(dim=0).values,
                ],
                dim=-1,
            )
        )
        indices = torch.tensor(state.action_candidate_node_indices, dtype=torch.long, device=device)
        candidate_hidden = wo_hidden.index_select(0, indices)
        selected_machine_hidden = machine_hidden[state.selected_machine_node_index]
        query = self.query_encoder(torch.cat([context, selected_machine_hidden], dim=-1))
        return self.pointer(
            torch.tanh(candidate_hidden + projected_hidden + query.unsqueeze(0))
        ).squeeze(-1)


def _validate_state(state: Phase2PolicyState) -> None:
    if state.schema_version != PHASE2_STATE_SCHEMA_VERSION:
        print(
            "[ERROR][Phase2.set_pointer_policy._validate_state] "
            f"cause=schema_mismatch actual={state.schema_version} expected={PHASE2_STATE_SCHEMA_VERSION}"
        )
        raise RuntimeError("Phase 2 state schema mismatch")
    action_count = len(state.action_ids)
    if action_count == 0 or len(state.action_projected_features) != action_count:
        print("[ERROR][Phase2.set_pointer_policy._validate_state] cause=invalid_action_count")
        raise RuntimeError("Phase 2 policy state has inconsistent action rows")
    if len(state.action_candidate_node_indices) != action_count:
        print("[ERROR][Phase2.set_pointer_policy._validate_state] cause=invalid_candidate_indices")
        raise RuntimeError("Phase 2 policy state has inconsistent candidate indices")
    node_count = len(state.wo_ids)
    if any(index < 0 or index >= node_count for index in state.action_candidate_node_indices):
        print("[ERROR][Phase2.set_pointer_policy._validate_state] cause=candidate_index_out_of_range")
        raise RuntimeError("Phase 2 action candidate node index is out of range")
    if not 0 <= state.selected_machine_node_index < len(state.machine_ids):
        print("[ERROR][Phase2.set_pointer_policy._validate_state] cause=selected_machine_index_out_of_range")
        raise RuntimeError("Phase 2 selected machine node index is out of range")
