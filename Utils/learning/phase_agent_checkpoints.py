"""Checkpoint loaders for Phase 1/2 full-flow inference.

These helpers validate checkpoint feature schemas before constructing a model.
There is no fallback path: a wrong checkpoint must fail loudly because a
feature mismatch changes the action meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import torch

from Phase2.run_spec import validate_phase2_run_spec
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase2.state import (
    PHASE2_SET_POINTER_POLICY_TYPE,
    PHASE2_STATE_SCHEMA_VERSION,
    phase2_state_feature_schema,
)
from Phase1.pair_self_labeling import (
    PHASE1_PAIR_ENV_FEATURE_NAMES,
    PHASE1_PAIR_FEATURE_NAMES,
)
from Phase1.pointer_policy import Phase1PairPointerPolicy


def load_phase1_pair_pointer_checkpoint(checkpoint_path: str | Path) -> Phase1PairPointerPolicy:
    """Load a Phase 1 pair-pointer checkpoint for inference."""

    path, checkpoint = _load_phase1_checkpoint_payload(checkpoint_path)
    hidden_dim = int(checkpoint.get("hidden_dim", 0))
    if hidden_dim <= 0:
        print(
            "[ERROR][phase_agent_checkpoints.load_phase1_pair_pointer_checkpoint] "
            f"cause=invalid_hidden_dim hidden_dim={hidden_dim} path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint hidden_dim is invalid")
    model = Phase1PairPointerPolicy(
        pair_feature_dim=len(PHASE1_PAIR_FEATURE_NAMES),
        env_feature_dim=len(PHASE1_PAIR_ENV_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        "[CHECK][phase_agent_checkpoints.load_phase1_pair_pointer_checkpoint] "
        f"path={path} hidden_dim={hidden_dim}"
    )
    return model


def load_phase1_feedback_contract(checkpoint_path: str | Path) -> dict[str, Any] | None:
    """Phase 1 checkpoint이 frozen Phase 2 feedback으로 학습됐는지 반환한다."""

    path, checkpoint = _load_phase1_checkpoint_payload(checkpoint_path)
    contract = checkpoint.get("phase2_feedback_contract")
    if contract is None:
        return None
    required = {"checkpoint", "checkpoint_sha256", "run_spec"}
    if not isinstance(contract, Mapping) or set(contract) != required:
        print(
            "[ERROR][phase_agent_checkpoints.load_phase1_feedback_contract] "
            f"cause=field_mismatch path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint has an invalid Phase 2 feedback contract")
    checkpoint_identity = str(contract["checkpoint"]).strip()
    digest = str(contract["checkpoint_sha256"]).strip().lower()
    run_spec = contract["run_spec"]
    if not checkpoint_identity or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        print(
            "[ERROR][phase_agent_checkpoints.load_phase1_feedback_contract] "
            f"cause=invalid_identity path={path} checkpoint={checkpoint_identity} sha256={digest}"
        )
        raise RuntimeError("Phase 1 checkpoint Phase 2 feedback identity is invalid")
    if not isinstance(run_spec, Mapping):
        print(
            "[ERROR][phase_agent_checkpoints.load_phase1_feedback_contract] "
            f"cause=missing_run_spec path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint Phase 2 feedback RunSpec is missing")
    validate_phase2_run_spec(run_spec)
    return {
        "checkpoint": checkpoint_identity,
        "checkpoint_sha256": digest,
        "run_spec": dict(run_spec),
    }


def _load_phase1_checkpoint_payload(
    checkpoint_path: str | Path,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=missing_checkpoint path={path}"
        )
        raise RuntimeError(f"missing Phase 1 checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=checkpoint_load_failed path={path} error={exc}"
        )
        raise RuntimeError(f"failed to load Phase 1 checkpoint: {path}") from exc
    if not isinstance(checkpoint, Mapping):
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=invalid_payload_type path={path} type={type(checkpoint).__name__}"
        )
        raise RuntimeError(f"invalid Phase 1 checkpoint payload: {path}")
    if checkpoint.get("pair_feature_names") != PHASE1_PAIR_FEATURE_NAMES:
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=pair_feature_mismatch path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint pair features do not match current code")
    if checkpoint.get("env_feature_names") != PHASE1_PAIR_ENV_FEATURE_NAMES:
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=env_feature_mismatch path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint env features do not match current code")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        print(
            "[ERROR][phase_agent_checkpoints._load_phase1_checkpoint_payload] "
            f"cause=missing_model_state path={path}"
        )
        raise RuntimeError("Phase 1 checkpoint model state is missing")
    return path, checkpoint


def load_phase2_set_pointer_checkpoint(
    checkpoint_path: str | Path,
    context: str,
) -> Phase2SetPointerPolicy:
    """Feature 계약이 정확히 일치하는 Phase 2 set-pointer를 로드한다."""

    path, checkpoint = _load_phase2_checkpoint_payload(checkpoint_path, context)
    if checkpoint.get("policy_type") != PHASE2_SET_POINTER_POLICY_TYPE:
        print(
            "[ERROR][phase_agent_checkpoints.load_phase2_set_pointer_checkpoint] "
            f"cause=policy_type_mismatch context={context} "
            f"checkpoint={checkpoint.get('policy_type')} expected={PHASE2_SET_POINTER_POLICY_TYPE} path={path}"
        )
        raise RuntimeError(f"{context} checkpoint policy type does not match current code")
    if checkpoint.get("feature_schema_version") != PHASE2_STATE_SCHEMA_VERSION:
        print(
            "[ERROR][phase_agent_checkpoints.load_phase2_set_pointer_checkpoint] "
            f"cause=feature_schema_version_mismatch context={context} "
            f"checkpoint={checkpoint.get('feature_schema_version')} expected={PHASE2_STATE_SCHEMA_VERSION} path={path}"
        )
        raise RuntimeError(f"{context} checkpoint feature schema version does not match current code")
    if checkpoint.get("feature_schema") != phase2_state_feature_schema():
        print(
            "[ERROR][phase_agent_checkpoints.load_phase2_set_pointer_checkpoint] "
            f"cause=feature_schema_mismatch context={context} path={path}"
        )
        raise RuntimeError(f"{context} checkpoint feature schema does not match current code")
    _phase2_run_spec_from_payload(checkpoint, path, context)
    hidden_dim = int(checkpoint.get("hidden_dim", 0))
    if hidden_dim <= 0:
        print(
            "[ERROR][phase_agent_checkpoints.load_phase2_set_pointer_checkpoint] "
            f"cause=invalid_hidden_dim context={context} hidden_dim={hidden_dim} path={path}"
        )
        raise RuntimeError(f"{context} checkpoint hidden_dim is invalid")
    model = Phase2SetPointerPolicy(hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        "[CHECK][phase_agent_checkpoints.load_phase2_set_pointer_checkpoint] "
        f"context={context} path={path} hidden_dim={hidden_dim} "
        f"feature_schema_version={PHASE2_STATE_SCHEMA_VERSION}"
    )
    return model


def load_phase2_checkpoint_run_spec(
    checkpoint_path: str | Path,
    context: str,
) -> dict[str, Any]:
    """Phase 2 checkpoint의 검증된 RunSpec을 반환한다."""

    path, checkpoint = _load_phase2_checkpoint_payload(checkpoint_path, context)
    return _phase2_run_spec_from_payload(checkpoint, path, context)


def _load_phase2_checkpoint_payload(
    checkpoint_path: str | Path,
    context: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        print(
            "[ERROR][phase_agent_checkpoints._load_phase2_checkpoint_payload] "
            f"cause=missing_checkpoint context={context} path={path}"
        )
        raise RuntimeError(f"missing {context} checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(
            "[ERROR][phase_agent_checkpoints._load_phase2_checkpoint_payload] "
            f"cause=checkpoint_load_failed context={context} path={path} error={exc}"
        )
        raise RuntimeError(f"failed to load {context} checkpoint: {path}") from exc
    if not isinstance(checkpoint, Mapping):
        print(
            "[ERROR][phase_agent_checkpoints._load_phase2_checkpoint_payload] "
            f"cause=invalid_payload_type context={context} path={path} type={type(checkpoint).__name__}"
        )
        raise RuntimeError(f"invalid {context} checkpoint payload: {path}")
    return path, checkpoint


def _phase2_run_spec_from_payload(
    checkpoint: Mapping[str, Any],
    path: Path,
    context: str,
) -> dict[str, Any]:
    run_spec = checkpoint.get("run_spec")
    if not isinstance(run_spec, Mapping):
        print(
            "[ERROR][phase_agent_checkpoints._phase2_run_spec_from_payload] "
            f"cause=missing_run_spec context={context} path={path}"
        )
        raise RuntimeError(f"{context} checkpoint is missing the Phase 2 RunSpec")
    validate_phase2_run_spec(run_spec)
    return dict(run_spec)
