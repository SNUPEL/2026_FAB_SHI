"""Phase 1 self-labeling trainer.

This is not fixed-teacher imitation. For each episode/problem it builds a
candidate bank:

1. current policy sampled rollouts;
2. heuristic rollout candidates such as LPT / long-cut-preferred.

The best candidate under the Phase 1 objective becomes the pseudo-label for one
teacher-forcing update. This follows the same idea as SLIM/self-labeling:
sample multiple solutions, score them with the problem objective, and learn from
the best one.

주의:
- 이 파일은 과거 split-action 구조(`SELECT_BLOCK -> SELECT_BAY`)용 self-labeling trainer다.
- 현재 주력 구조는 `phase1_pair_self_labeling.py`의 `SELECT_PAIR(block,bay)`다.
- 그래도 비교/ablation/legacy 실험용으로 남겨 두므로 입출력과 pseudo-label 흐름을 명확히 주석화한다.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python annotation 호환성 유지.
from __future__ import annotations

# LINE-BY-LINE: CSV metric/audit 파일 저장에 사용합니다.
import csv
# LINE-BY-LINE: JSON/JSONL action table과 summary 저장에 사용합니다.
import json
# LINE-BY-LINE: transition/candidate record class 정의에 사용합니다.
from dataclasses import dataclass
# LINE-BY-LINE: output/checkpoint path를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 인자/반환 타입을 명확히 하기 위한 typing import입니다.
from typing import Callable, Dict, List, Mapping, Sequence

# LINE-BY-LINE: PyTorch model, tensor, optimizer, checkpoint 저장에 사용합니다.
import torch
# LINE-BY-LINE: cross entropy loss 계산에 사용합니다.
import torch.nn.functional as F

# LINE-BY-LINE: split-action pointer network입니다. block 선택과 Bay 선택 head가 따로 있습니다.
from Train.network.phase1_pointer import Phase1PointerPolicy
# LINE-BY-LINE: Phase 1 휴리스틱 plan 생성과 Bay 부하 계산 helper를 재사용합니다.
from Utils.phase1_bay_balancer import (
    CANONICAL_PHASE1_HEURISTIC,
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _multi_objective_load_score,
    _normalize_bay_ids,
    build_phase1_bay_plan,
)
# LINE-BY-LINE: split-action MDP feature 이름과 후보 feature 생성 helper를 재사용합니다.
from Utils.phase1_mdp import (
    PHASE1_BAY_FEATURE_NAMES,
    PHASE1_BLOCK_FEATURE_NAMES,
    PHASE1_ENV_FEATURE_NAMES,
    _bay_candidate,
    _block_feature_vector,
    _empty_bay_loads,
    _env_features,
)


# LINE-BY-LINE: self-labeling 후보 bank에 넣는 8개 휴리스틱 이름입니다.
PHASE1_SELF_LABEL_HEURISTIC_BANK = (
    # LINE-BY-LINE: block 선택은 강재수량 우선, Bay 선택은 강재/절단장/베벨 균형 기준입니다.
    "steel_first_balanced",
    # LINE-BY-LINE: block 선택은 절단장 우선입니다.
    "cut_first_balanced",
    # LINE-BY-LINE: block 선택은 베벨수량 우선입니다.
    "bevel_first_balanced",
    # LINE-BY-LINE: block 선택은 장척 우선입니다.
    "long_cut_first_balanced",
    # LINE-BY-LINE: 강재수량 우선 block 순서에 장척 Bay24 비선호 목적을 결합합니다.
    "steel_first_long_cut_preferred",
    # LINE-BY-LINE: 절단장 우선 block 순서에 장척 Bay24 비선호 목적을 결합합니다.
    "cut_first_long_cut_preferred",
    # LINE-BY-LINE: 베벨수량 우선 block 순서에 장척 Bay24 비선호 목적을 결합합니다.
    "bevel_first_long_cut_preferred",
    # LINE-BY-LINE: 장척 우선 block 순서에 장척 Bay24 비선호 목적을 결합합니다.
    "long_cut_first_long_cut_preferred",
)


# LINE-BY-LINE: split-action self-labeling의 한 decision step을 담는 immutable record입니다.
@dataclass(frozen=True)
class Phase1SelfLabelTransition:
    """One trainable Phase 1 decision from a selected candidate sequence."""

    # LINE-BY-LINE: decision phase입니다. 값은 `SELECT_BLOCK` 또는 `SELECT_BAY`.
    phase: str
    # LINE-BY-LINE: 현재 phase에서 가능한 후보 feature matrix입니다.
    candidate_features: List[List[float]]
    # LINE-BY-LINE: 현재 Bay 부하/진행률 등 환경 feature vector입니다.
    env_features: List[float]
    # LINE-BY-LINE: pseudo-label로 선택된 후보 index입니다.
    selected_action_index: int
    # LINE-BY-LINE: `SELECT_BAY`일 때 선택된 block feature입니다. `SELECT_BLOCK`에서는 None입니다.
    selected_block_features: List[float] | None = None
    # LINE-BY-LINE: 후보 id 목록입니다. audit/debug용입니다.
    candidate_ids: List[str] | None = None
    # LINE-BY-LINE: 선택된 action id입니다. audit/debug용입니다.
    selected_action_id: str = ""
    # LINE-BY-LINE: 선택된 block_set_id입니다.
    selected_block_set_id: str = ""
    # LINE-BY-LINE: 선택된 Bay입니다.
    selected_bay: str = ""


# LINE-BY-LINE: 한 episode 전체 block->Bay assignment 후보를 담는 immutable record입니다.
@dataclass(frozen=True)
class Phase1SelfLabelCandidate:
    """One complete Phase 1 candidate assignment."""

    # LINE-BY-LINE: 후보 생성 source입니다. 예: `agent_sample_1`, `steel_first_balanced`.
    source: str
    # LINE-BY-LINE: complete assignment를 만드는 데 사용된 step별 transition입니다.
    transitions: List[Phase1SelfLabelTransition]
    # LINE-BY-LINE: 최종 block_set_id -> Bay 배정 결과입니다.
    assignments: Dict[str, str]
    # LINE-BY-LINE: 최종 Bay별 누적 부하입니다. 목적함수 평가에 사용합니다.
    bay_loads: Dict[str, Dict[str, int | float]]


# LINE-BY-LINE: split-action best-of-K self-labeling 학습 entry function입니다.
def train_phase1_pointer_self_labeling(
    jobs: Mapping[str, object] | None,
    bay_ids: Sequence[str],
    output_dir: str | Path,
    episodes: int = 20,
    rollout_samples: int = 4,
    heuristic_algorithms: Sequence[str] = PHASE1_SELF_LABEL_HEURISTIC_BANK,
    score_mode: str = "steel_first",
    lr: float = 1e-3,
    hidden_dim: int = 128,
    temperature: float = 1.0,
    seed: int = 0,
    episode_jobs: Sequence[Mapping[str, object]] | None = None,
    episode_metadata: Sequence[Mapping[str, object]] | None = None,
    metrics_callback: Callable[[List[Dict], List[Dict]], None] | None = None,
) -> Dict:
    """Train Phase 1 policy with best-of-K self-labeling.

    입력:
    - fixed jobs 또는 episode_jobs 중 하나.
    - Bay 후보.
    - rollout_samples와 heuristic bank.

    출력:
    - checkpoint, metrics.csv, candidate_summary.csv, best_action_table.jsonl, plot PNG.
    """

    # LINE-BY-LINE: episode 수, rollout 수, lr, temperature를 검증합니다.
    _validate_train_args(episodes, rollout_samples, lr, temperature)
    # LINE-BY-LINE: fixed jobs 모드와 sampled episode 모드 입력이 서로 맞는지 검증합니다.
    _validate_episode_inputs(jobs, episode_jobs, episode_metadata, episodes)
    # LINE-BY-LINE: PyTorch seed를 고정해 재현성을 확보합니다.
    torch.manual_seed(seed)
    # LINE-BY-LINE: Bay ID를 deterministic tuple로 정규화합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    # LINE-BY-LINE: output directory를 생성합니다.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: split-action pointer policy를 생성합니다.
    model = Phase1PointerPolicy(
        block_feature_dim=len(PHASE1_BLOCK_FEATURE_NAMES),
        bay_feature_dim=len(PHASE1_BAY_FEATURE_NAMES),
        env_feature_dim=len(PHASE1_ENV_FEATURE_NAMES),
        hidden_dim=hidden_dim,
    )
    # LINE-BY-LINE: model parameter를 학습할 Adam optimizer입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # LINE-BY-LINE: episode metric row를 누적합니다.
    metrics_rows: List[Dict] = []
    # LINE-BY-LINE: agent/heuristic 후보별 audit row를 누적합니다.
    candidate_rows: List[Dict] = []
    # LINE-BY-LINE: best pseudo-label action table row를 누적합니다.
    best_action_rows: List[Dict] = []
    checkpoint_path = output_path / "phase1_self_label_pointer.pt"
    metrics_csv = output_path / "metrics.csv"
    candidate_summary_csv = output_path / "candidate_summary.csv"
    best_action_table_jsonl = output_path / "best_action_table.jsonl"
    manifest_json = output_path / "learning_data_manifest.json"

    print("[phase1-train-self-labeling]")
    print(f"- episodes: {episodes}")
    print(f"- rollout_samples: {rollout_samples}")
    print(f"- heuristic_algorithms: {','.join(heuristic_algorithms)}")
    print(f"- score_mode: {score_mode}")
    print(f"- episode_mode: {'sampled' if episode_jobs is not None else 'fixed'}")

    for episode in range(1, episodes + 1):
        current_jobs, problem_id, block_count, problem_seed = _episode_problem(
            episode=episode,
            fixed_jobs=jobs,
            episode_jobs=episode_jobs,
            episode_metadata=episode_metadata,
        )
        candidates: List[Phase1SelfLabelCandidate] = []
        for sample_index in range(1, rollout_samples + 1):
            candidates.append(
                run_phase1_policy_rollout(
                    jobs=current_jobs,
                    bay_ids=normalized_bay_ids,
                    model=model,
                    temperature=temperature,
                    seed=seed + episode * 1000 + sample_index,
                    source=f"agent_sample_{sample_index}",
                )
            )
        for algorithm in heuristic_algorithms:
            candidates.append(
                run_phase1_heuristic_candidate(
                    jobs=current_jobs,
                    bay_ids=normalized_bay_ids,
                    algorithm=algorithm,
                )
            )

        best = _select_best_candidate(candidates, score_mode=score_mode)
        loss = _teacher_forcing_update(model, optimizer, best.transitions)
        score = _score_bay_loads(best.bay_loads, score_mode)
        episode_candidate_rows = [
            _candidate_summary_row(
                episode=episode,
                problem_id=problem_id,
                block_count=block_count,
                job_count=len(current_jobs),
                problem_seed=problem_seed,
                candidate_index=candidate_index,
                candidate=candidate,
                best=best,
                score_mode=score_mode,
            )
            for candidate_index, candidate in enumerate(candidates, start=1)
        ]
        candidate_rows.extend(episode_candidate_rows)
        best_action_rows.extend(
            _best_action_rows(
                episode=episode,
                problem_id=problem_id,
                block_count=block_count,
                job_count=len(current_jobs),
                problem_seed=problem_seed,
                best=best,
            )
        )
        metrics_rows.append(
            {
                "episode": episode,
                "problem_id": problem_id,
                "block_count": block_count,
                "job_count": len(current_jobs),
                "problem_seed": problem_seed,
                "best_source": best.source,
                "loss": loss,
                "score_json": json.dumps(list(score), ensure_ascii=False),
                "candidate_count": len(candidates),
            }
        )
        _write_metrics(metrics_csv, metrics_rows)
        _write_candidate_summary(candidate_summary_csv, candidate_rows)
        _write_jsonl(best_action_table_jsonl, best_action_rows)
        if metrics_callback is not None:
            metrics_callback(list(metrics_rows), list(candidate_rows))
        print(
            "[CHECK][phase1_self_labeling.train] "
            f"episode={episode} problem_id={problem_id} block_count={block_count} "
            f"best_source={best.source} loss={loss:.6f} score={score}"
        )

    plot_paths = _write_training_plots(output_path, metrics_rows, candidate_rows)
    summary_json = output_path / "summary.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "block_feature_names": PHASE1_BLOCK_FEATURE_NAMES,
            "bay_feature_names": PHASE1_BAY_FEATURE_NAMES,
            "env_feature_names": PHASE1_ENV_FEATURE_NAMES,
            "hidden_dim": hidden_dim,
        },
        checkpoint_path,
    )
    source_counts: Dict[str, int] = {}
    for row in metrics_rows:
        source = str(row["best_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "metrics_csv": str(metrics_csv),
        "candidate_summary_csv": str(candidate_summary_csv),
        "best_action_table_jsonl": str(best_action_table_jsonl),
        "learning_data_manifest_json": str(manifest_json),
        **plot_paths,
        "summary_json": str(summary_json),
        "episodes": episodes,
        "episode_mode": "sampled" if episode_jobs is not None else "fixed",
        "rollout_samples": rollout_samples,
        "heuristic_algorithms": list(heuristic_algorithms),
        "score_mode": score_mode,
        "best_source_counts": source_counts,
    }
    _write_manifest(manifest_json, summary, len(candidate_rows), len(best_action_rows))
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, sort_keys=True)
    return summary


def run_phase1_policy_rollout(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    model: Phase1PointerPolicy | None,
    temperature: float,
    seed: int,
    source: str,
) -> Phase1SelfLabelCandidate:
    """Sample one `SELECT_BLOCK -> SELECT_BAY` Phase 1 rollout."""

    if temperature <= 0:
        print(f"[ERROR][phase1_self_labeling.run_phase1_policy_rollout] cause=non_positive_temperature value={temperature}")
        raise ValueError("temperature must be positive")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids, require_multi_objective=True)
    bay_loads = _empty_bay_loads(normalized_bay_ids)
    remaining_blocks = {block.block_set_id: block for block in blocks}
    transitions: List[Phase1SelfLabelTransition] = []
    assignments: Dict[str, str] = {}

    for block_step in range(len(blocks)):
        block_candidates = sorted(remaining_blocks.values(), key=lambda block: block.block_set_id)
        env_dict = _env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
        )
        env_features = _feature_row(env_dict, PHASE1_ENV_FEATURE_NAMES)
        block_feature_rows = [_block_feature_vector(block, assigned=False) for block in block_candidates]
        block_candidate_ids = [block.block_set_id for block in block_candidates]
        block_index = _sample_index(
            logits=_score_block_candidates(model, block_feature_rows, env_features),
            temperature=temperature,
            generator=generator,
        )
        selected_block = block_candidates[block_index]
        transitions.append(
            Phase1SelfLabelTransition(
                phase="SELECT_BLOCK",
                candidate_features=block_feature_rows,
                env_features=env_features,
                selected_action_index=block_index,
                candidate_ids=block_candidate_ids,
                selected_action_id=f"select_block:{selected_block.block_set_id}",
                selected_block_set_id=selected_block.block_set_id,
            )
        )

        bay_candidates = [
            bay_id
            for bay_id in normalized_bay_ids
            if bay_id in selected_block.allowed_bay_ids
        ]
        bay_feature_rows = [
            _bay_candidate(
                bay_id=bay_id,
                loads=bay_loads[bay_id],
                candidate_allowed=True,
            )["features"]
            for bay_id in bay_candidates
        ]
        selected_block_features = _block_feature_vector(selected_block, assigned=False)
        bay_index = _sample_index(
            logits=_score_bay_candidates(model, bay_feature_rows, env_features, selected_block_features),
            temperature=temperature,
            generator=generator,
        )
        selected_bay = bay_candidates[bay_index]
        transitions.append(
            Phase1SelfLabelTransition(
                phase="SELECT_BAY",
                candidate_features=bay_feature_rows,
                env_features=env_features,
                selected_action_index=bay_index,
                selected_block_features=selected_block_features,
                candidate_ids=list(bay_candidates),
                selected_action_id=f"select_bay:{selected_bay}",
                selected_block_set_id=selected_block.block_set_id,
                selected_bay=selected_bay,
            )
        )
        assignments[selected_block.block_set_id] = selected_bay
        _add_block_load(bay_loads[selected_bay], selected_bay, selected_block)
        del remaining_blocks[selected_block.block_set_id]

    return Phase1SelfLabelCandidate(
        source=source,
        transitions=transitions,
        assignments=assignments,
        bay_loads=_plain_bay_loads(bay_loads),
    )


def run_phase1_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    long_cut_hard_mask: bool = True,
) -> Phase1SelfLabelCandidate:
    """Convert an existing Phase 1 heuristic plan into a self-label candidate."""

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    if algorithm in PHASE1_SELF_LABEL_HEURISTIC_BANK:
        return _run_phase1_greedy_bank_candidate(
            jobs=jobs,
            bay_ids=normalized_bay_ids,
            algorithm=algorithm,
            long_cut_hard_mask=long_cut_hard_mask,
        )
    plan = build_phase1_bay_plan(jobs=jobs, bay_ids=normalized_bay_ids, algorithm=algorithm)
    assignments = {str(row["block_set_id"]): str(row["assigned_bay"]) for row in plan["assignments"]}
    transitions = _transitions_from_assignment(jobs, normalized_bay_ids, assignments)
    return Phase1SelfLabelCandidate(
        source=algorithm,
        transitions=transitions,
        assignments=assignments,
        bay_loads=_plain_bay_loads(plan["bay_loads"]),
    )


def _run_phase1_greedy_bank_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    long_cut_hard_mask: bool = True,
) -> Phase1SelfLabelCandidate:
    """Run one block-order/Bay-score greedy heuristic from the 8-way bank."""

    order_name, bay_score_rule = _heuristic_bank_spec(algorithm)
    blocks = _sort_blocks_for_bank(
        _collect_blocks(
            jobs=jobs,
            bay_ids=tuple(bay_ids),
            require_multi_objective=True,
            long_cut_hard_mask=long_cut_hard_mask,
        ),
        order_name=order_name,
    )
    bay_loads = _empty_bay_loads(tuple(bay_ids))
    assignments: Dict[str, str] = {}
    for block in blocks:
        selected_bay = min(
            block.allowed_bay_ids,
            key=lambda bay_id: _projected_bank_score(bay_loads, bay_id, block, bay_score_rule),
        )
        assignments[block.block_set_id] = selected_bay
        _add_block_load(bay_loads[selected_bay], selected_bay, block)
    transitions = _transitions_from_assignment(jobs, bay_ids, assignments) if long_cut_hard_mask else []
    return Phase1SelfLabelCandidate(
        source=algorithm,
        transitions=transitions,
        assignments=assignments,
        bay_loads=_plain_bay_loads(bay_loads),
    )


def _heuristic_bank_spec(algorithm: str) -> tuple[str, str]:
    """Map a bank heuristic name to block order and Bay scoring mode."""

    if algorithm.endswith("_long_cut_preferred"):
        order_name = algorithm[: -len("_long_cut_preferred")]
        bay_score_rule = "long_cut_preferred"
    elif algorithm.endswith("_balanced"):
        order_name = algorithm[: -len("_balanced")]
        bay_score_rule = "multi_objective"
    else:
        print(f"[ERROR][phase1_self_labeling._heuristic_bank_spec] cause=unknown_bank_algorithm algorithm={algorithm}")
        raise RuntimeError(f"unknown Phase 1 self-label heuristic bank algorithm: {algorithm}")
    if order_name not in {"steel_first", "cut_first", "bevel_first", "long_cut_first"}:
        print(
            "[ERROR][phase1_self_labeling._heuristic_bank_spec] "
            f"cause=unknown_order_name algorithm={algorithm} order_name={order_name}"
        )
        raise RuntimeError(f"unknown Phase 1 self-label heuristic order: {order_name}")
    return order_name, bay_score_rule


def _sort_blocks_for_bank(blocks: Sequence[Phase1Block], order_name: str) -> List[Phase1Block]:
    """Return the block selection order used by one bank heuristic."""

    if order_name == "steel_first":
        return sorted(blocks, key=lambda block: (-block.steel_quantity_sum, -block.cut_length_sum, -block.bevel_quantity_sum, -block.long_cut_over_1000, block.block_set_id))
    if order_name == "cut_first":
        return sorted(blocks, key=lambda block: (-block.cut_length_sum, -block.steel_quantity_sum, -block.bevel_quantity_sum, -block.long_cut_over_1000, block.block_set_id))
    if order_name == "bevel_first":
        return sorted(blocks, key=lambda block: (-block.bevel_quantity_sum, -block.steel_quantity_sum, -block.cut_length_sum, -block.long_cut_over_1000, block.block_set_id))
    if order_name == "long_cut_first":
        return sorted(blocks, key=lambda block: (-block.long_cut_over_1000, -block.steel_quantity_sum, -block.cut_length_sum, -block.bevel_quantity_sum, block.block_set_id))
    print(f"[ERROR][phase1_self_labeling._sort_blocks_for_bank] cause=unknown_order_name order_name={order_name}")
    raise RuntimeError(f"unknown Phase 1 self-label heuristic order: {order_name}")


def _projected_bank_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    bay_id: str,
    block: Phase1Block,
    bay_score_rule: str,
) -> tuple:
    """Score one greedy insertion candidate."""

    projected = {current_bay_id: dict(loads) for current_bay_id, loads in bay_loads.items()}
    _add_block_load(projected[bay_id], bay_id, block)
    if bay_score_rule == "multi_objective":
        return _multi_objective_load_score(projected, score_mode="steel_first") + (str(bay_id),)
    if bay_score_rule == "long_cut_preferred":
        row = projected[str(bay_id)]
        return (
            int(row["long_cut_bay24_count"]),
            float(row["cut_length_sum"]),
            int(row["bevel_quantity_sum"]),
            int(row["steel_quantity_sum"]),
            str(bay_id),
        )
    print(
        "[ERROR][phase1_self_labeling._projected_bank_score] "
        f"cause=unknown_bay_score_rule bay_score_rule={bay_score_rule}"
    )
    raise RuntimeError(f"unknown_bay_score_rule: {bay_score_rule}")


def _transitions_from_assignment(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    assignments: Mapping[str, str],
) -> List[Phase1SelfLabelTransition]:
    """Replay a fixed assignment into trainable `SELECT_BLOCK/SELECT_BAY` rows."""

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids, require_multi_objective=True)
    block_by_id = {block.block_set_id: block for block in blocks}
    bay_loads = _empty_bay_loads(normalized_bay_ids)
    remaining_blocks = dict(block_by_id)
    transitions: List[Phase1SelfLabelTransition] = []

    for block_step, block_id in enumerate(assignments):
        selected_block = block_by_id[block_id]
        block_candidates = sorted(remaining_blocks.values(), key=lambda block: block.block_set_id)
        block_ids = [block.block_set_id for block in block_candidates]
        selected_block_index = _candidate_index(block_ids, selected_block.block_set_id, "selected_block_set_id")
        env_dict = _env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
        )
        env_features = _feature_row(env_dict, PHASE1_ENV_FEATURE_NAMES)
        transitions.append(
            Phase1SelfLabelTransition(
                phase="SELECT_BLOCK",
                candidate_features=[_block_feature_vector(block, assigned=False) for block in block_candidates],
                env_features=env_features,
                selected_action_index=selected_block_index,
                candidate_ids=block_ids,
                selected_action_id=f"select_block:{selected_block.block_set_id}",
                selected_block_set_id=selected_block.block_set_id,
            )
        )
        selected_bay = assignments[block_id]
        bay_candidates = [
            bay_id
            for bay_id in normalized_bay_ids
            if bay_id in selected_block.allowed_bay_ids
        ]
        selected_bay_index = _candidate_index(bay_candidates, selected_bay, "selected_bay")
        selected_block_features = _block_feature_vector(selected_block, assigned=False)
        transitions.append(
            Phase1SelfLabelTransition(
                phase="SELECT_BAY",
                candidate_features=[
                    _bay_candidate(bay_id=bay_id, loads=bay_loads[bay_id], candidate_allowed=True)["features"]
                    for bay_id in bay_candidates
                ],
                env_features=env_features,
                selected_action_index=selected_bay_index,
                selected_block_features=selected_block_features,
                candidate_ids=list(bay_candidates),
                selected_action_id=f"select_bay:{selected_bay}",
                selected_block_set_id=selected_block.block_set_id,
                selected_bay=selected_bay,
            )
        )
        _add_block_load(bay_loads[selected_bay], selected_bay, selected_block)
        del remaining_blocks[selected_block.block_set_id]

    return transitions


def _select_best_candidate(
    candidates: Sequence[Phase1SelfLabelCandidate],
    score_mode: str,
) -> Phase1SelfLabelCandidate:
    """Return the best candidate by objective only, not by source name."""

    if not candidates:
        print("[ERROR][phase1_self_labeling._select_best_candidate] cause=no_candidates")
        raise RuntimeError("Phase 1 self-labeling candidate bank is empty")
    return min(candidates, key=lambda candidate: _score_bay_loads(candidate.bay_loads, score_mode))


def _candidate_summary_row(
    episode: int,
    problem_id: str,
    block_count: int,
    job_count: int,
    problem_seed: int | str,
    candidate_index: int,
    candidate: Phase1SelfLabelCandidate,
    best: Phase1SelfLabelCandidate,
    score_mode: str,
) -> Dict:
    """Return one audit row for a self-label candidate."""

    score = _score_bay_loads(candidate.bay_loads, score_mode)
    return {
        "episode": episode,
        "problem_id": problem_id,
        "block_count": block_count,
        "job_count": job_count,
        "problem_seed": problem_seed,
        "candidate_index": candidate_index,
        "source": candidate.source,
        "is_best": int(candidate is best),
        "score_mode": score_mode,
        "score_json": json.dumps(list(score), ensure_ascii=False),
        "score_0": score[0] if len(score) > 0 else "",
        "score_1": score[1] if len(score) > 1 else "",
        "score_2": score[2] if len(score) > 2 else "",
        "score_3": score[3] if len(score) > 3 else "",
        "score_4": score[4] if len(score) > 4 else "",
        "score_5": score[5] if len(score) > 5 else "",
        "score_6": score[6] if len(score) > 6 else "",
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "bay_loads_json": json.dumps(candidate.bay_loads, ensure_ascii=False, sort_keys=True),
        "assignments_json": json.dumps(candidate.assignments, ensure_ascii=False, sort_keys=True),
    }


def _best_action_rows(
    episode: int,
    problem_id: str,
    block_count: int,
    job_count: int,
    problem_seed: int | str,
    best: Phase1SelfLabelCandidate,
) -> List[Dict]:
    """Return JSONL rows for the selected pseudo-label sequence."""

    rows: List[Dict] = []
    for step, transition in enumerate(best.transitions):
        rows.append(
            {
                "episode": episode,
                "problem_id": problem_id,
                "block_count": block_count,
                "job_count": job_count,
                "problem_seed": problem_seed,
                "step": step,
                "source": best.source,
                "phase": transition.phase,
                "selected_action_index": transition.selected_action_index,
                "selected_action_id": transition.selected_action_id,
                "selected_block_set_id": transition.selected_block_set_id,
                "selected_bay": transition.selected_bay,
                "candidate_ids": transition.candidate_ids or [],
                "candidate_features": transition.candidate_features,
                "env_features": transition.env_features,
                "selected_block_features": transition.selected_block_features,
            }
        )
    return rows


def _score_bay_loads(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    score_mode: str,
) -> tuple:
    """Score Phase 1 loads with the same lexicographic objective as heuristics."""

    return _multi_objective_load_score(bay_loads, score_mode=score_mode)


def _teacher_forcing_update(
    model: Phase1PointerPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Phase1SelfLabelTransition],
) -> float:
    """Cross-entropy update on the selected best candidate sequence."""

    if not transitions:
        print("[ERROR][phase1_self_labeling._teacher_forcing_update] cause=no_transitions")
        raise RuntimeError("best self-label candidate has no transitions")
    total_loss = torch.zeros((), dtype=torch.float32)
    for transition in transitions:
        candidate_tensor = torch.tensor(transition.candidate_features, dtype=torch.float32)
        env_tensor = torch.tensor(transition.env_features, dtype=torch.float32)
        if transition.phase == "SELECT_BLOCK":
            logits = model.score_blocks(candidate_tensor, env_tensor)
        elif transition.phase == "SELECT_BAY":
            if transition.selected_block_features is None:
                print("[ERROR][phase1_self_labeling._teacher_forcing_update] cause=missing_selected_block_features")
                raise RuntimeError("SELECT_BAY transition requires selected_block_features")
            logits = model.score_bays(
                bay_features=candidate_tensor,
                env_features=env_tensor,
                selected_block_features=torch.tensor(transition.selected_block_features, dtype=torch.float32),
            )
        else:
            print(f"[ERROR][phase1_self_labeling._teacher_forcing_update] cause=unknown_phase phase={transition.phase}")
            raise RuntimeError(f"unknown Phase 1 transition phase: {transition.phase}")
        if transition.selected_action_index >= logits.numel():
            print(
                "[ERROR][phase1_self_labeling._teacher_forcing_update] "
                f"cause=selected_index_out_of_range phase={transition.phase} "
                f"selected={transition.selected_action_index} candidate_count={logits.numel()}"
            )
            raise RuntimeError("selected action index out of range")
        total_loss = total_loss + F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([transition.selected_action_index], dtype=torch.long),
        )
    loss = total_loss / len(transitions)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def _score_block_candidates(
    model: Phase1PointerPolicy | None,
    candidate_features: List[List[float]],
    env_features: List[float],
) -> torch.Tensor:
    """Return policy logits, or uniform logits when no model is supplied."""

    if model is None:
        return torch.zeros((len(candidate_features),), dtype=torch.float32)
    return model.score_blocks(
        torch.tensor(candidate_features, dtype=torch.float32),
        torch.tensor(env_features, dtype=torch.float32),
    ).detach()


def _score_bay_candidates(
    model: Phase1PointerPolicy | None,
    candidate_features: List[List[float]],
    env_features: List[float],
    selected_block_features: List[float],
) -> torch.Tensor:
    """Return policy logits for Bay candidates, or uniform logits without model."""

    if model is None:
        return torch.zeros((len(candidate_features),), dtype=torch.float32)
    return model.score_bays(
        bay_features=torch.tensor(candidate_features, dtype=torch.float32),
        env_features=torch.tensor(env_features, dtype=torch.float32),
        selected_block_features=torch.tensor(selected_block_features, dtype=torch.float32),
    ).detach()


def _sample_index(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    """Sample one local action index from masked candidate logits."""

    if logits.numel() == 0:
        print("[ERROR][phase1_self_labeling._sample_index] cause=no_logits")
        raise RuntimeError("cannot sample from empty candidate logits")
    probabilities = torch.softmax(logits / temperature, dim=0)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def _candidate_index(candidates: Sequence[str], selected: str, field_name: str) -> int:
    """Find selected candidate index and fail loudly if a heuristic violates mask."""

    try:
        return list(candidates).index(selected)
    except ValueError as exc:
        print(
            "[ERROR][phase1_self_labeling._candidate_index] "
            f"cause=selected_not_in_candidates field={field_name} selected={selected} "
            f"candidates={list(candidates)}"
        )
        raise RuntimeError(f"selected_not_in_candidates: {field_name}={selected}") from exc


def _feature_row(feature_dict: Mapping[str, float], field_names: Sequence[str]) -> List[float]:
    """Convert named feature dict to ordered numeric row."""

    return [float(feature_dict[name]) for name in field_names]


def _plain_bay_loads(bay_loads: Mapping[str, Mapping[str, int | float]]) -> Dict[str, Dict[str, int | float]]:
    """Return JSON-safe Bay load mapping."""

    return {
        str(bay_id): {
            "steel_quantity_sum": int(loads["steel_quantity_sum"]),
            "cut_length_sum": float(loads["cut_length_sum"]),
            "bevel_quantity_sum": int(loads["bevel_quantity_sum"]),
            "long_cut_bay24_count": int(loads["long_cut_bay24_count"]),
            "wo_count": int(loads["wo_count"]),
            "block_count": int(loads["block_count"]),
        }
        for bay_id, loads in bay_loads.items()
    }


def _validate_train_args(episodes: int, rollout_samples: int, lr: float, temperature: float) -> None:
    """Reject invalid self-labeling train parameters."""

    if episodes <= 0:
        print(f"[ERROR][phase1_self_labeling._validate_train_args] cause=non_positive_episodes episodes={episodes}")
        raise ValueError("episodes must be positive")
    if rollout_samples <= 0:
        print(
            "[ERROR][phase1_self_labeling._validate_train_args] "
            f"cause=non_positive_rollout_samples rollout_samples={rollout_samples}"
        )
        raise ValueError("rollout_samples must be positive")
    if lr <= 0:
        print(f"[ERROR][phase1_self_labeling._validate_train_args] cause=non_positive_lr lr={lr}")
        raise ValueError("lr must be positive")
    if temperature <= 0:
        print(f"[ERROR][phase1_self_labeling._validate_train_args] cause=non_positive_temperature temperature={temperature}")
        raise ValueError("temperature must be positive")


def _validate_episode_inputs(
    jobs: Mapping[str, object] | None,
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    episodes: int,
) -> None:
    """Validate fixed or sampled training problem inputs."""

    if episode_jobs is None:
        if not jobs:
            print("[ERROR][phase1_self_labeling._validate_episode_inputs] cause=no_fixed_jobs")
            raise RuntimeError("fixed self-labeling mode requires jobs")
        return
    if jobs is not None:
        print("[ERROR][phase1_self_labeling._validate_episode_inputs] cause=jobs_and_episode_jobs_both_set")
        raise RuntimeError("provide either jobs or episode_jobs, not both")
    if len(episode_jobs) != episodes:
        print(
            "[ERROR][phase1_self_labeling._validate_episode_inputs] "
            f"cause=episode_job_count_mismatch episodes={episodes} episode_jobs={len(episode_jobs)}"
        )
        raise RuntimeError("episode_jobs length must equal episodes")
    if episode_metadata is not None and len(episode_metadata) != episodes:
        print(
            "[ERROR][phase1_self_labeling._validate_episode_inputs] "
            f"cause=episode_metadata_count_mismatch episodes={episodes} metadata={len(episode_metadata)}"
        )
        raise RuntimeError("episode_metadata length must equal episodes")
    for index, current_jobs in enumerate(episode_jobs, start=1):
        if not current_jobs:
            print(f"[ERROR][phase1_self_labeling._validate_episode_inputs] cause=empty_episode_jobs episode={index}")
            raise RuntimeError(f"empty episode_jobs at episode {index}")


def _episode_problem(
    episode: int,
    fixed_jobs: Mapping[str, object] | None,
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], str, int, int | str]:
    """Return the current episode problem and audit metadata."""

    if episode_jobs is None:
        if fixed_jobs is None:
            print("[ERROR][phase1_self_labeling._episode_problem] cause=no_fixed_jobs")
            raise RuntimeError("fixed self-labeling mode requires jobs")
        return fixed_jobs, "fixed_config", len(fixed_jobs), ""
    current_jobs = episode_jobs[episode - 1]
    metadata = dict(episode_metadata[episode - 1]) if episode_metadata is not None else {}
    problem_id = str(metadata.get("problem_id") or metadata.get("episode_id") or f"EP{episode:05d}")
    block_count = int(metadata.get("block_count") or len(current_jobs))
    problem_seed = metadata.get("seed", "")
    return current_jobs, problem_id, block_count, problem_seed


def _write_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write self-labeling metrics CSV."""

    fields = ["episode", "problem_id", "block_count", "job_count", "problem_seed", "best_source", "loss", "score_json", "candidate_count"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all self-label candidate scores for audit."""

    fields = [
        "episode",
        "problem_id",
        "block_count",
        "job_count",
        "problem_seed",
        "candidate_index",
        "source",
        "is_best",
        "score_mode",
        "score_json",
        "score_0",
        "score_1",
        "score_2",
        "score_3",
        "score_4",
        "score_5",
        "score_6",
        "assignment_count",
        "transition_count",
        "bay_loads_json",
        "assignments_json",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """Write JSONL rows with one selected pseudo-label action per line."""

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_manifest(path: Path, summary: Mapping, candidate_count: int, best_action_count: int) -> None:
    """Write a small manifest explaining the generated learning data."""

    manifest = {
        "type": "phase1_best_of_k_self_labeling_learning_data",
        "summary": dict(summary),
        "candidate_row_count": int(candidate_count),
        "best_action_row_count": int(best_action_count),
        "candidate_summary": "Every candidate generated in each episode, including agent samples and heuristic candidates.",
        "best_action_table": "Only the selected best candidate sequence used as pseudo-label for teacher-forcing update.",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)


def _write_training_plots(output_path: Path, metrics_rows: Sequence[Mapping], candidate_rows: Sequence[Mapping]) -> Dict[str, str]:
    """Write PNG plots for loss and best-source counts."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][phase1_self_labeling._write_training_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to write Phase 1 self-labeling plots") from exc

    loss_curve = output_path / "loss_curve.png"
    best_source_counts = output_path / "best_source_counts.png"
    score0_curve = output_path / "best_score0_curve.png"

    episodes = [int(row["episode"]) for row in metrics_rows]
    losses = [float(row["loss"]) for row in metrics_rows]
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, losses, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Teacher-forcing loss")
    plt.title("Phase 1 self-labeling loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_curve, dpi=160)
    plt.close()

    counts: Dict[str, int] = {}
    for row in metrics_rows:
        source = str(row["best_source"])
        counts[source] = counts.get(source, 0) + 1
    plt.figure(figsize=(8, 4))
    plt.bar(list(counts), list(counts.values()))
    plt.xlabel("Best source")
    plt.ylabel("Count")
    plt.title("Best candidate source counts")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(best_source_counts, dpi=160)
    plt.close()

    best_rows = [row for row in candidate_rows if int(row.get("is_best", 0)) == 1]
    best_episodes = [int(row["episode"]) for row in best_rows]
    score0 = [float(row["score_0"]) for row in best_rows]
    plt.figure(figsize=(8, 4))
    plt.plot(best_episodes, score0, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Best score[0]")
    plt.title("Best candidate primary objective")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(score0_curve, dpi=160)
    plt.close()

    return {
        "loss_curve_png": str(loss_curve),
        "best_source_counts_png": str(best_source_counts),
        "best_score0_curve_png": str(score0_curve),
    }
