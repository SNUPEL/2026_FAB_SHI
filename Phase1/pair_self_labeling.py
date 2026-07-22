"""MIXED Phase 1 direct pair-action self-labeling.

한 물리 episode를 `NP_NC(22/23/24)`와 `FN_FL(25/trans)` 두 자원군
서브문제로 나눈다. 하나의 공유 정책이 각 서브문제에서 `(block-series, Bay)`
pair를 선택하지만, 후보 bank·teacher 선정·CE update는 자원군별로 독립한다.
"""

# LINE-BY-LINE: 미래 타입 힌트를 문자열로 늦게 평가합니다. 사용: Python 버전별 annotation 충돌을 줄입니다.
from __future__ import annotations

# LINE-BY-LINE: CSV audit 파일을 읽고 쓰기 위해 사용합니다. 예: metrics.csv, validation_candidate_summary.csv.
import csv
# LINE-BY-LINE: NUL byte가 섞인 기존 CSV resume 파일을 메모리 문자열로 읽을 때 사용합니다.
import io
# LINE-BY-LINE: JSON/JSONL 저장에 사용합니다. 예: score_json, best_action_table.jsonl, summary.json.
import json
# LINE-BY-LINE: CSV field size limit을 플랫폼 안전하게 키울 때 사용합니다.
import sys
# LINE-BY-LINE: `dataclass`는 transition/candidate record class를 간결하게 정의하는 데 사용합니다.
from dataclasses import dataclass
# LINE-BY-LINE: output path/checkpoint path를 OS 독립적으로 다루기 위해 사용합니다.
from pathlib import Path
# LINE-BY-LINE: 함수 인자/반환 타입을 명확히 하기 위한 typing import입니다.
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

# LINE-BY-LINE: PyTorch tensor, model checkpoint, random seed, optimizer 실행에 사용합니다.
import torch
# LINE-BY-LINE: cross entropy loss 계산에 사용합니다.
import torch.nn.functional as F

from Environment.hierarchical import (
    HierarchicalPlanningState,
    apply_phase1_block_assignment,
    complete_phase1_planning,
    create_phase1_planning_state,
)
# LINE-BY-LINE: 직접 pair action을 scoring하는 pointer network입니다. 입력: pair feature matrix + env feature vector.
from Phase1.pointer_policy import Phase1PairPointerPolicy
from Phase1.orchestrator import candidate_to_phase1_plan
# LINE-BY-LINE: Phase 1 Bay 부하 계산과 목적함수 점수 계산에 필요한 기존 balancer 함수/타입입니다.
from Utils.phase1.phase1_bay_balancer import (
    Phase1Block,
    _add_block_load,
    _collect_blocks,
    _job_attr,
    _multi_objective_load_score,
    _normalize_bay_capacity_weights,
    _normalize_bay_ids,
    _require_capacity_weight,
    _validate_multi_series_plan_scope,
    write_phase1_bay_plan,
)
# LINE-BY-LINE: 가변 Bay 수를 지원하는 graph edge feature schema와 graph builder입니다. 고정 `bay_22_flag`를 대체합니다.
from Utils.learning.phase_graph_mdp import (
    PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES,
    build_phase1_block_bay_graph,
)
from Utils.phase1.multi_series_rules import (
    GROUP_BAY_CAPACITY_WEIGHTS,
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_BALANCING_GROUP_ORDER,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
    PHASE1_RESOURCE_POOL_BAYS,
    PHASE1_RESOURCE_POOL_ORDER,
    PHASE1_RESOURCE_POOL_SERIES,
    add_multi_series_group_load,
    initialize_multi_series_group_loads,
    joint_phase1_bay_capacity_weights,
    multi_series_group_load_value,
    normalize_phase1_objective_scope,
    phase1_resource_pool_for_series,
    phase1_objective_field_names,
    split_phase1_jobs_by_resource_pool,
)
# LINE-BY-LINE: 8개 휴리스틱 후보 bank와 complete heuristic assignment 생성 함수를 재사용합니다.
from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    run_phase1_heuristic_candidate,
)


# MIXED pair 후보는 W/O-first 부하, 계열 그룹, NP hard mask 상태를 포함한다.
PHASE1_PAIR_FEATURE_NAMES = list(PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES)
# 활성 자원군의 공유/계열별 W/O·CUT·BV gap과 진행률을 사용한다.
PHASE1_PAIR_ENV_FEATURE_NAMES = [
    "progress_ratio",
    "remaining_block_ratio",
    "shared_wo_gap_ratio",
    "series_wo_gap_ratio",
    "shared_cut_gap_ratio",
    "series_cut_gap_ratio",
    "shared_bevel_gap_ratio",
    "series_bevel_gap_ratio",
]
# LINE-BY-LINE: 최대 6개 capacity-normalized 목적함수를 CSV에 저장하는 고정 column 이름입니다.
PHASE1_SCORE_FIELD_NAMES = [f"score_{index}" for index in range(6)]
# LINE-BY-LINE: validation 그래프에서 agent_greedy와 agent_sample_* 중 최고 후보를 하나로 묶어 표시할 때 쓰는 source 이름입니다.
PHASE1_PROPOSED_BEST_OF_K_SOURCE = "proposed_best_of_k"
PHASE1_VALIDATION_VIEW_ORDER = ("NP", "NC", "NP_NC", "FN", "FL", "FN_FL")
PHASE1_VALIDATION_GAP_FIELDS = tuple(
    f"{scope}_{metric}_gap"
    for scope in ("np", "nc", "np_nc", "fn", "fl", "fn_fl")
    for metric in ("wo", "cut", "bv")
)


def phase1_pair_feature_schema() -> Dict[str, List[str]]:
    """유일한 MIXED pair/env feature 계약을 반환한다."""
    return {
        "pair": list(PHASE1_PAIR_FEATURE_NAMES),
        "env": list(PHASE1_PAIR_ENV_FEATURE_NAMES),
    }


    # LINE-BY-LINE: `Phase1PairTransition`은 학습 pseudo-label 한 step을 담는 immutable record입니다.
@dataclass(frozen=True)
class Phase1PairTransition:
    """One direct pair decision used for pair-policy training."""

    # LINE-BY-LINE: decision phase 이름입니다. 현재 pair action에서는 항상 `SELECT_PAIR`입니다.
    phase: str
    # LINE-BY-LINE: 현재 step의 feasible pair 후보 feature matrix입니다. shape: `(feasible pair 수, 20)`.
    candidate_features: List[List[float]]
    # LINE-BY-LINE: 현재 step의 환경 feature vector입니다. shape: `(8,)`.
    env_features: List[float]
    # LINE-BY-LINE: pseudo-label로 선택된 후보 index입니다. cross entropy target으로 사용합니다.
    selected_action_index: int
    # LINE-BY-LINE: 후보 id 목록입니다. 예: `PROJ_1::BLK_1@22`.
    candidate_ids: List[str]
    # LINE-BY-LINE: 실제 선택된 후보 id입니다. audit와 replay에 사용합니다.
    selected_action_id: str
    # LINE-BY-LINE: 선택된 block_set_id입니다. 예: `PROJ_1::BLK_1`.
    selected_block_set_id: str
    # LINE-BY-LINE: 선택된 Bay입니다. 예: `22`, `23`, `24`.
    selected_bay: str


# LINE-BY-LINE: `Phase1PairCandidate`는 한 서브문제 또는 병합 parent의 complete assignment입니다.
@dataclass(frozen=True)
class Phase1PairCandidate:
    """한 자원군 또는 병합된 Phase 1 완전 배정 후보."""

    # LINE-BY-LINE: 후보를 만든 방법입니다. 예: `agent_greedy`, `agent_sample_3`, `bevel_first_long_cut_preferred`.
    source: str
    # LINE-BY-LINE: 이 complete assignment를 만들 때 거친 step별 transition 목록입니다.
    transitions: List[Phase1PairTransition]
    # LINE-BY-LINE: 최종 block->Bay 배정 결과입니다. key=block_set_id, value=Bay.
    assignments: Dict[str, str]
    # LINE-BY-LINE: 최종 Bay별 누적 부하입니다. 목적함수 점수와 CSV report에 사용합니다.
    bay_loads: Dict[str, Dict[str, int | float]]


@dataclass(frozen=True)
class _Phase1PairEpisodeCache:
    """Phase 1 한 episode 안에서 변하지 않는 block 집계와 denominator cache."""

    # LINE-BY-LINE: 부모 문제의 정규화된 5개 Bay ID입니다.
    bay_ids: Tuple[str, ...]
    # LINE-BY-LINE: Bay별 설비 수/가중치입니다. score와 feature denominator에 동일하게 사용합니다.
    bay_capacity_weights: Dict[str, float]
    # LINE-BY-LINE: `_collect_blocks()` 결과입니다. 기존에는 매 step 다시 만들던 값을 episode당 1회만 만듭니다.
    blocks: Tuple[Phase1Block, ...]
    # LINE-BY-LINE: 선택된 block_id로 block feature를 O(1)에 찾기 위한 lookup입니다.
    block_by_id: Dict[str, Phase1Block]
    # LINE-BY-LINE: env feature denominator입니다. 기존 `_problem_totals(blocks)`와 동일합니다.
    env_totals: Dict[str, float]
    # LINE-BY-LINE: 현재 profile에서 model에 입력되는 pair feature 이름입니다.
    pair_feature_names: Tuple[str, ...]
    # LINE-BY-LINE: 현재 profile에서 model에 입력되는 env feature 이름입니다.
    env_feature_names: Tuple[str, ...]


Phase2FeedbackScorer = Callable[
    [Phase1PairCandidate, Mapping[str, object], Sequence[str]],
    Sequence[int | float],
]


def build_phase1_pair_candidates(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_loads: Mapping[str, Mapping[str, int | float]] | None = None,
    remaining_block_ids: Sequence[str] | None = None,
    assigned_block_count: int = 0,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
) -> List[Dict]:
    """현재 MIXED state에서 가능한 `(block-series, Bay)` 후보를 만든다."""

    # LINE-BY-LINE: Bay ID를 문자열 tuple로 정규화하고, 빈 Bay 입력이면 여기서 실패합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    normalized_capacity_weights = _normalize_bay_capacity_weights(
        normalized_bay_ids,
        bay_capacity_weights,
    )
    # LINE-BY-LINE: W/O job들을 block_set_id 기준으로 묶고, 강재수량/절단장/베벨수량 등을 block 단위로 합산합니다.
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids)
    _validate_multi_series_plan_scope(blocks, normalized_bay_ids, normalized_capacity_weights)
    # LINE-BY-LINE: 남은 block 목록이 외부에서 오면 그것만 사용하고, 없으면 전체 block을 남은 후보로 봅니다.
    remaining = set(remaining_block_ids) if remaining_block_ids is not None else {block.block_set_id for block in blocks}
    # LINE-BY-LINE: 현재 Bay별 부하를 복사합니다. 원본 dict를 직접 수정하지 않기 위해 새 dict를 만듭니다.
    current_loads = (
        _create_pair_planning_state(normalized_capacity_weights).bay_loads
        if bay_loads is None
        else {str(bay_id): dict(loads) for bay_id, loads in bay_loads.items()}
    )
    # LINE-BY-LINE: graph builder가 만든 edge feature를 그대로 재사용합니다. 여기서 Bay별 one-hot을 새로 만들지 않습니다.
    graph = build_phase1_block_bay_graph(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        bay_loads=current_loads,
        assigned_block_ids={block.block_set_id for block in blocks if block.block_set_id not in remaining},
        bay_capacity_weights=normalized_capacity_weights,
    )
    # LINE-BY-LINE: 반환할 pair 후보 row를 누적합니다.
    candidates: List[Dict] = []
    # LINE-BY-LINE: graph edge 순서를 deterministic하게 정렬해 학습/검증 재현성을 유지합니다.
    for edge in sorted(graph["candidate_edges"], key=lambda row: (str(row["source_block_id"]), str(row["target_bay_id"]))):
        block_id = str(edge["source_block_id"])
        bay_id = str(edge["target_bay_id"])
        # LINE-BY-LINE: action id, block id, Bay id, graph edge feature 이름, graph edge feature 값을 한 row로 저장합니다.
        candidates.append(
            {
                "action_id": f"{block_id}@{bay_id}",
                "block_set_id": block_id,
                "bay_id": bay_id,
                "feature_names": list(graph["feature_names"]["edge"]),
                "features": list(edge["features"]),
            }
        )
    # LINE-BY-LINE: 후보가 하나도 없으면 학습/추론을 계속할 수 없으므로 fallback 없이 실패시킵니다.
    if not candidates:
        print("[ERROR][phase1_pair_self_labeling.build_phase1_pair_candidates] cause=no_pair_candidates")
        raise RuntimeError("Phase 1 pair MDP has no feasible pair candidates")
    # LINE-BY-LINE: 현재 step에서 선택 가능한 모든 pair 후보를 반환합니다.
    return candidates


def _build_phase1_pair_episode_cache(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float] | None,
) -> _Phase1PairEpisodeCache:
    """Build immutable Phase 1 values once per rollout episode."""

    # LINE-BY-LINE: public builder와 같은 Bay 정규화 경로를 사용합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    # LINE-BY-LINE: 설비 수 가중치를 한 번만 정규화합니다. 이후 step에서는 이 값을 그대로 씁니다.
    capacity_weights = _normalize_bay_capacity_weights(normalized_bay_ids, bay_capacity_weights)
    # LINE-BY-LINE: 가장 비싼 block 집계를 episode 시작 시 1회만 수행합니다.
    blocks = tuple(_collect_blocks(jobs=jobs, bay_ids=normalized_bay_ids))
    _validate_multi_series_plan_scope(blocks, normalized_bay_ids, capacity_weights)
    # LINE-BY-LINE: block id 중복은 같은 block이 두 개의 의사결정 단위가 되는 오류라 즉시 실패시킵니다.
    block_by_id = {block.block_set_id: block for block in blocks}
    if len(block_by_id) != len(blocks):
        print("[ERROR][phase1_pair_self_labeling._build_phase1_pair_episode_cache] cause=duplicate_block_id")
        raise RuntimeError("duplicate Phase 1 block id")
    feature_schema = phase1_pair_feature_schema()
    return _Phase1PairEpisodeCache(
        bay_ids=normalized_bay_ids,
        bay_capacity_weights=capacity_weights,
        blocks=blocks,
        block_by_id=block_by_id,
        env_totals=_problem_totals(blocks, capacity_weights),
        pair_feature_names=tuple(feature_schema["pair"]),
        env_feature_names=tuple(feature_schema["env"]),
    )


def _build_phase1_pair_candidates_from_cache(
    cache: _Phase1PairEpisodeCache,
    bay_loads: Mapping[str, Mapping[str, int | float]],
    remaining_block_ids: Sequence[str],
) -> List[Dict]:
    """Build the same candidate rows as `build_phase1_pair_candidates` without rebuilding blocks."""

    # LINE-BY-LINE: Bay load 누락은 fallback 없이 실패합니다. graph builder의 missing load 검사와 같은 역할입니다.
    missing_bays = [bay_id for bay_id in cache.bay_ids if bay_id not in bay_loads]
    if missing_bays:
        print(
            "[ERROR][phase1_pair_self_labeling._build_phase1_pair_candidates_from_cache] "
            f"cause=missing_bay_loads bay_ids={missing_bays}"
        )
        raise RuntimeError(f"missing Bay loads: {missing_bays}")
    # LINE-BY-LINE: 기존 graph builder denominator와 동일하게 전체 block 합계 + 현재 누적 load를 사용합니다.
    totals = _phase1_pair_graph_totals(cache.blocks, bay_loads)
    candidates: List[Dict] = []
    for block_id in sorted(str(value) for value in remaining_block_ids):
        block = cache.block_by_id.get(block_id)
        if block is None:
            print(
                "[ERROR][phase1_pair_self_labeling._build_phase1_pair_candidates_from_cache] "
                f"cause=unknown_remaining_block block_id={block_id}"
            )
            raise RuntimeError(f"unknown remaining Phase 1 block: {block_id}")
        for bay_id in block.allowed_bay_ids:
            score = _phase1_projected_pair_score(
                bay_loads,
                bay_id,
                block,
            )
            group_flags = _phase1_pair_group_flags(block)
            group_wo_average = totals[
                _phase1_pair_group_average_key(block.balancing_group, "wo_count")
            ]
            group_cut_average = totals[
                _phase1_pair_group_average_key(block.balancing_group, "cut_length_sum")
            ]
            group_bevel_average = totals[
                _phase1_pair_group_average_key(block.balancing_group, "bevel_quantity_sum")
            ]
            features = [
                _phase1_pair_ratio(block.wo_count, totals["wo_count"]),
                _phase1_pair_ratio(block.cut_length_sum, totals["cut_length_sum"]),
                _phase1_pair_ratio(block.bevel_quantity_sum, totals["bevel_quantity_sum"]),
                *group_flags,
                float(block.wide_plate_over_4500),
                float(block.cnt_block),
                float(block.long_cut_over_1000),
                _phase1_group_capacity_load_ratio(
                    bay_loads[bay_id], block.balancing_group, "wo_count", group_wo_average
                ),
                _phase1_group_capacity_load_ratio(
                    bay_loads[bay_id], block.balancing_group, "cut_length_sum", group_cut_average
                ),
                _phase1_group_capacity_load_ratio(
                    bay_loads[bay_id], block.balancing_group, "bevel_quantity_sum", group_bevel_average
                ),
                _phase1_pair_ratio(
                    _require_capacity_weight(
                        bay_loads[bay_id], f"pair_candidate:{block_id}@{bay_id}"
                    ),
                    totals["capacity_weight_sum"],
                ),
                _phase1_pair_ratio(score[0], totals["wo_per_capacity_average"]),
                _phase1_pair_ratio(score[1], totals["wo_per_capacity_average"]),
                _phase1_pair_ratio(score[2], totals["cut_per_capacity_average"]),
                _phase1_pair_ratio(score[3], totals["cut_per_capacity_average"]),
                _phase1_pair_ratio(score[4], totals["bevel_per_capacity_average"]),
                _phase1_pair_ratio(score[5], totals["bevel_per_capacity_average"]),
            ]
            candidates.append(
                {
                    "action_id": f"{block.block_set_id}@{bay_id}",
                    "block_set_id": block.block_set_id,
                    "bay_id": bay_id,
                    "feature_names": list(cache.pair_feature_names),
                    "features": features,
                }
            )
    if not candidates:
        print("[ERROR][phase1_pair_self_labeling._build_phase1_pair_candidates_from_cache] cause=no_pair_candidates")
        raise RuntimeError("Phase 1 pair MDP has no feasible pair candidates")
    return candidates


def _phase1_projected_pair_score(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    bay_id: str,
    block: Phase1Block,
) -> tuple:
    """Return projected score after adding `block` to one Bay."""

    # LINE-BY-LINE: 한 후보 Bay만 가상 반영합니다. deepcopy 대신 shallow row copy로 같은 값을 더 적은 비용에 만듭니다.
    projected_loads = {str(current_bay): dict(loads) for current_bay, loads in bay_loads.items()}
    _add_block_load(projected_loads[bay_id], bay_id, block)
    return _multi_objective_load_score(projected_loads)


def _phase1_pair_graph_totals(
    blocks: Sequence[Phase1Block],
    bay_loads: Mapping[str, Mapping[str, int | float]],
) -> Dict[str, float]:
    """Return denominator values matching `Utils.learning.phase_graph_mdp._phase1_totals` for edge features."""

    capacity_weight_sum = max(
        1.0,
        sum(
            _require_capacity_weight(row, f"pair_totals:{bay_id}")
            for bay_id, row in bay_loads.items()
        ),
    )
    wo_count = _phase1_positive_total(
        sum(block.wo_count for block in blocks),
        "phase1 wo_count",
    )
    cut_length_sum = _phase1_positive_total(
        sum(block.cut_length_sum for block in blocks),
        "phase1 cut_length_sum",
    )
    bevel_quantity_sum = max(
        1.0,
        sum(block.bevel_quantity_sum for block in blocks),
    )
    result = {
        "wo_count": wo_count,
        "cut_length_sum": cut_length_sum,
        "bevel_quantity_sum": bevel_quantity_sum,
        "capacity_weight_sum": capacity_weight_sum,
        "wo_per_capacity_average": wo_count / capacity_weight_sum,
        "cut_per_capacity_average": cut_length_sum / capacity_weight_sum,
        "bevel_per_capacity_average": max(1.0, bevel_quantity_sum / capacity_weight_sum),
    }
    present_groups = {block.balancing_group for block in blocks}
    group_averages: Dict[str, Dict[str, float]] = {}
    for group in PHASE1_BALANCING_GROUP_ORDER:
        group_blocks = [block for block in blocks if block.balancing_group == group]
        group_capacity_sum = sum(GROUP_BAY_CAPACITY_WEIGHTS[group].values())
        raw_averages = {
            "wo_count": sum(block.wo_count for block in group_blocks) / group_capacity_sum,
            "cut_length_sum": sum(block.cut_length_sum for block in group_blocks) / group_capacity_sum,
            "bevel_quantity_sum": sum(block.bevel_quantity_sum for block in group_blocks) / group_capacity_sum,
        }
        group_averages[group] = raw_averages
        for metric, raw_average in raw_averages.items():
            result[_phase1_pair_group_average_key(group, metric)] = max(1.0, raw_average)
    result["wo_per_capacity_average"] = max(
        1.0, sum(group_averages[group]["wo_count"] for group in present_groups)
    )
    result["cut_per_capacity_average"] = max(
        1.0, sum(group_averages[group]["cut_length_sum"] for group in present_groups)
    )
    result["bevel_per_capacity_average"] = max(
        1.0, sum(group_averages[group]["bevel_quantity_sum"] for group in present_groups)
    )
    return result


def _phase1_pair_group_flags(block: Phase1Block) -> List[float]:
    """다계열 block의 평준화 그룹 one-hot을 반환한다."""

    groups = ("NP", "NC", "FN", "FL")
    if block.balancing_group not in groups:
        print(
            "[ERROR][phase1_pair_self_labeling._phase1_pair_group_flags] "
            f"cause=unknown_balancing_group block_set_id={block.block_set_id} group={block.balancing_group}"
        )
        raise RuntimeError(f"unknown Phase 1 balancing group: {block.balancing_group}")
    return [1.0 if block.balancing_group == group else 0.0 for group in groups]


def _phase1_group_capacity_load_ratio(
    loads: Mapping[str, int | float],
    group: str,
    metric: str,
    average_per_capacity: float,
) -> float:
    """현재 후보 block과 같은 평준화 그룹의 Bay 부하만 정규화한다."""

    capacity = _require_capacity_weight(loads, f"pair_group_load:{group}:{metric}")
    return _phase1_pair_ratio(
        multi_series_group_load_value(loads, group, metric) / capacity,
        average_per_capacity,
    )


def _phase1_pair_group_average_key(group: str, metric: str) -> str:
    return f"group_{group.lower()}_{metric}_per_capacity_average"


def _phase1_positive_total(value: int | float, label: str) -> float:
    """Return a positive denominator or fail with context."""

    parsed = float(value)
    if parsed <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._phase1_positive_total] "
            f"cause=non_positive_total label={label} value={value}"
        )
        raise RuntimeError(f"non_positive_total: {label}")
    return parsed


def _phase1_pair_ratio(value: int | float, denominator: int | float) -> float:
    """Match `Utils.learning.phase_graph_mdp._ratio` without importing its private helper."""

    return round(float(value) / float(denominator), 9)


# LINE-BY-LINE: Phase 1 pair-action self-labeling의 메인 학습 함수입니다. CLI `phase1-train-pair-self-labeling`이 호출합니다.
def train_phase1_pair_self_labeling(
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    bay_ids: Sequence[str],
    output_dir: str | Path,
    episodes: int,
    rollout_samples: int = 4,
    heuristic_algorithms: Sequence[str] = PHASE1_HEURISTIC_BANK,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    temperature: float = 1.0,
    seed: int = 0,
    episode_factory: Callable[[int], Mapping[str, object]] | None = None,
    checkpoint_every: int = 100,
    validation_every: int = 100,
    validation_episodes: int = 20,
    validation_episode_factory: Callable[[int], Mapping[str, object]] | None = None,
    validation_rollout_samples: int | None = None,
    resume_checkpoint: str | Path | None = None,
    phase2_feedback_scorer: Phase2FeedbackScorer | None = None,
    phase2_feedback_contract: Mapping[str, object] | None = None,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
    device: str = "cpu",
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Dict:
    """Train one shared pair policy with independent resource-pool teachers.

    학습 절차:
    1. 부모 episode를 `NP_NC`와 `FN_FL` 서브문제로 나눈다.
    2. 각 서브문제에서 policy greedy/sample K개와 dispatching 휴리스틱을 비교한다.
    3. 해당 자원군의 사전식 score로 teacher를 하나 선정한다.
    4. teacher sequence의 step별 index로 즉시 CE update한다.
    5. 두 자원군이 모두 있으면 하나의 부모 episode에서 update가 2번 일어난다.
    6. 병합된 배정은 parent audit/report에만 사용하고 teacher 비교에는 쓰지 않는다.
    """

    # LINE-BY-LINE: validation sampling 수를 별도 지정하지 않으면 기존 호환성을 위해 train rollout 수를 그대로 씁니다.
    resolved_validation_rollout_samples = rollout_samples if validation_rollout_samples is None else validation_rollout_samples
    # LINE-BY-LINE: 입력 episode 수, rollout 수, validation rollout 수, learning rate, checkpoint interval 등을 먼저 검증합니다.
    _validate_train_inputs(
        episode_jobs,
        episode_metadata,
        episodes,
        rollout_samples,
        resolved_validation_rollout_samples,
        lr,
        temperature,
        episode_factory,
        checkpoint_every,
        validation_every,
        validation_episodes,
        validation_episode_factory,
        phase2_feedback_scorer,
        phase2_feedback_contract,
    )
    normalized_feedback_contract = _normalize_phase2_feedback_contract(phase2_feedback_contract)
    normalized_objective_scope = normalize_phase1_objective_scope(objective_scope)
    feature_schema = phase1_pair_feature_schema()
    # LINE-BY-LINE: PyTorch 난수 seed를 고정해 같은 입력에서 같은 초기 모델/샘플링을 재현합니다.
    torch.manual_seed(seed)
    # LINE-BY-LINE: 학습 device를 검증합니다. CUDA 요청 시 사용 불가하면 CPU로 조용히 내려가지 않고 실패합니다.
    torch_device = _resolve_torch_device(device)
    # LINE-BY-LINE: 부모 episode의 5개 Bay ID를 정규화합니다.
    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    normalized_capacity_weights = _normalize_bay_capacity_weights(normalized_bay_ids, bay_capacity_weights)
    # LINE-BY-LINE: output_dir를 Path 객체로 바꾸고 없으면 생성합니다.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: graph edge 기반 pair feature와 전역 env feature를 받는 pointer policy를 생성합니다.
    model = Phase1PairPointerPolicy(
        pair_feature_dim=len(feature_schema["pair"]),
        env_feature_dim=len(feature_schema["env"]),
        hidden_dim=hidden_dim,
        rule_profile=MULTI_SERIES_RULE_PROFILE,
        score_mode="wo_first",
        objective_scope=normalized_objective_scope,
    ).to(torch_device)
    # LINE-BY-LINE: Adam optimizer를 생성합니다. 학습 대상은 model parameter 전체입니다.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # LINE-BY-LINE: 주기 checkpoint를 저장할 하위 디렉터리입니다.
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: resume 옵션이 있으면 실제 checkpoint path를 계산합니다. 예: `latest`.
    resume_path = _resolve_resume_checkpoint(output_path, resume_checkpoint)
    # LINE-BY-LINE: 새 학습이면 1 episode부터, resume이면 checkpoint 다음 episode부터 시작합니다.
    start_episode = 1
    previous_objective_scope = normalized_objective_scope
    objective_scope_transition = False
    if resume_path is not None:
        # LINE-BY-LINE: checkpoint에서 model/optimizer state를 복원하고 완료 episode 번호를 읽습니다.
        completed_episode, previous_objective_scope = _load_pair_checkpoint(
            model=model,
            optimizer=optimizer,
            path=resume_path,
            hidden_dim=hidden_dim,
            phase2_feedback_contract=normalized_feedback_contract,
            feature_schema=feature_schema,
            objective_scope=normalized_objective_scope,
        )
        start_episode = completed_episode + 1
        objective_scope_transition = previous_objective_scope != normalized_objective_scope
        _move_optimizer_state(optimizer, torch_device)
    # LINE-BY-LINE: resume 시 기존 metrics.csv에서 start_episode 이전 row만 보존합니다.
    metrics_rows: List[Dict] = _read_csv_rows(output_path / "metrics.csv", "episode", start_episode)
    # 한 parent episode의 자원군별 CE update를 독립 행으로 보존한다.
    subproblem_metric_rows: List[Dict] = _read_csv_rows(
        output_path / "subproblem_metrics.csv", "episode", start_episode
    )
    # LINE-BY-LINE: resume 시 기존 후보 audit CSV에서 start_episode 이전 row만 보존합니다.
    candidate_rows: List[Dict] = _read_csv_rows(output_path / "candidate_summary.csv", "episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 pseudo-label JSONL에서 start_episode 이전 row만 보존합니다.
    best_action_rows: List[Dict] = _read_jsonl_rows(output_path / "best_action_table.jsonl", "episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 validation summary에서 start_episode 이전 row만 보존합니다.
    validation_rows: List[Dict] = _read_csv_rows(output_path / "validation_summary.csv", "train_episode", start_episode)
    # LINE-BY-LINE: resume 시 기존 validation 후보별 score CSV에서 start_episode 이전 row만 보존합니다.
    validation_candidate_rows: List[Dict] = _read_csv_rows(output_path / "validation_candidate_summary.csv", "train_episode", start_episode)
    # LINE-BY-LINE: best validation checkpoint 비교를 위해 이전 summary의 best score를 읽습니다.
    best_validation_score: tuple | None = (
        _read_best_validation_score(output_path / "summary.json")
        if resume_path is not None and not objective_scope_transition
        else None
    )
    # LINE-BY-LINE: validation 기준으로 가장 좋은 모델을 저장할 path입니다.
    best_checkpoint_path = output_path / "phase1_pair_pointer_best.pt"
    # LINE-BY-LINE: validation을 한 번이라도 돌리면 여기에 PNG 경로들이 들어갑니다.
    validation_plot_paths: Dict[str, str] = {}

    print("[phase1-train-pair-self-labeling]")
    print(f"- episodes: {episodes}")
    print(f"- start_episode: {start_episode}")
    print(f"- rollout_samples: {rollout_samples}")
    print(f"- validation_rollout_samples: {resolved_validation_rollout_samples}")
    print(f"- heuristic_algorithms: {','.join(heuristic_algorithms)}")
    print(f"- rule_profile: {MULTI_SERIES_RULE_PROFILE}")
    print("- score_mode: wo_first")
    print(f"- objective_scope: {normalized_objective_scope}")
    print(f"- objective_scope_transition: {objective_scope_transition}")
    print(f"- previous_objective_scope: {previous_objective_scope}")
    print(f"- episode_mode: {'on_the_fly' if episode_factory is not None else 'prebuilt'}")
    print(f"- resume_checkpoint: {resume_path or ''}")
    print(f"- device: {torch_device}")

    # LINE-BY-LINE: start_episode부터 사용자가 요청한 episodes까지 학습 loop를 수행합니다.
    for episode in range(start_episode, episodes + 1):
        # LINE-BY-LINE: 현재 episode의 Job-like mapping과 metadata를 가져옵니다. on-the-fly factory도 여기서 호출됩니다.
        jobs, metadata = _episode_payload(episode, episode_jobs, episode_metadata, episode_factory)
        episode_bay_ids, episode_capacity_weights = _resolve_phase1_episode_scope(
            metadata=metadata,
            default_bay_ids=normalized_bay_ids,
            default_capacity_weights=normalized_capacity_weights,
        )
        # LINE-BY-LINE: problem_id는 출력 CSV/JSONL에서 episode 문제를 식별하는 이름입니다.
        problem_id = str(metadata.get("problem_id") or metadata.get("episode_id") or f"EP{episode:05d}")
        # LINE-BY-LINE: block_count는 로그와 CSV audit에 저장하는 문제 크기입니다.
        block_count = int(metadata.get("block_count") or len(jobs))
        # LINE-BY-LINE: synthetic generator seed를 audit에 남깁니다. 없으면 빈 문자열입니다.
        problem_seed = metadata.get("seed", "")
        # LINE-BY-LINE: synthetic episode가 일반 문제인지 hard-case 문제인지 metrics에 남깁니다.
        case_type = str(metadata.get("case_type") or "")
        # LINE-BY-LINE: hard-case mode입니다. 예: none, cut_shuffle.
        hard_case_mode = str(metadata.get("hard_case_mode") or "")
        # LINE-BY-LINE: hard-case 적용 전 corr(STL_QTY,CUT_LTH)입니다. 일반 episode는 빈 값입니다.
        hard_case_corr_before = metadata.get("hard_case_corr_steel_cut_before", "")
        # LINE-BY-LINE: hard-case 적용 후 corr(STL_QTY,CUT_LTH)입니다. 일반 episode는 빈 값입니다.
        hard_case_corr_after = metadata.get("hard_case_corr_steel_cut_after", "")
        checkpoint_due = episode % checkpoint_every == 0
        subproblems = split_phase1_jobs_by_resource_pool(jobs)
        selected_subproblems: List[tuple[str, Phase1PairCandidate]] = []
        selected_agent_subproblems: List[tuple[str, Phase1PairCandidate]] = []
        subproblem_losses: List[float] = []
        total_candidate_count = 0
        for pool_index, pool_id in enumerate(PHASE1_RESOURCE_POOL_ORDER):
            pool_jobs = subproblems.get(pool_id)
            if not pool_jobs:
                continue
            pool_block_count = len(_collect_blocks(pool_jobs, episode_bay_ids))
            candidates: List[Phase1PairCandidate] = []
            for sample_index in range(1, rollout_samples + 1):
                is_greedy = sample_index == 1
                candidates.append(
                    run_phase1_pair_policy_rollout(
                        jobs=pool_jobs,
                        bay_ids=episode_bay_ids,
                        model=model,
                        temperature=temperature,
                        seed=seed + episode * 10_000 + pool_index * 1_000 + sample_index,
                        source="agent_greedy" if is_greedy else f"agent_sample_{sample_index}",
                        selection="greedy" if is_greedy else "sample",
                        bay_capacity_weights=episode_capacity_weights,
                    )
                )
            for algorithm in heuristic_algorithms:
                candidates.append(
                    _run_phase1_pair_heuristic_candidate(
                        pool_jobs,
                        episode_bay_ids,
                        algorithm,
                        bay_capacity_weights=episode_capacity_weights,
                        objective_scope=normalized_objective_scope,
                    )
                )
            candidate_scores = {
                id(candidate): _candidate_score_components(
                    candidate=candidate,
                    jobs=pool_jobs,
                    bay_ids=episode_bay_ids,
                    phase2_feedback_scorer=phase2_feedback_scorer,
                    objective_scope=normalized_objective_scope,
                )
                for candidate in candidates
            }
            best = min(candidates, key=lambda candidate: candidate_scores[id(candidate)][2])
            if checkpoint_due:
                agent_candidates = [
                    candidate for candidate in candidates if _is_agent_source(candidate.source)
                ]
                if not agent_candidates:
                    print(
                        "[ERROR][phase1_pair_self_labeling.train] "
                        f"cause=no_agent_candidate episode={episode} subproblem_id={pool_id}"
                    )
                    raise RuntimeError("Phase 1 checkpoint solution requires an agent candidate")
                agent_best = min(
                    agent_candidates,
                    key=lambda candidate: candidate_scores[id(candidate)][2],
                )
                selected_agent_subproblems.append((pool_id, agent_best))
            loss = _teacher_forcing_update(model, optimizer, best.transitions)
            score, phase2_feedback_score, learning_score = candidate_scores[id(best)]
            selected_subproblems.append((pool_id, best))
            subproblem_losses.append(loss)
            total_candidate_count += len(candidates)
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_rows.append(
                    _candidate_summary_row(
                        episode=episode,
                        problem_id=problem_id,
                        block_count=pool_block_count,
                        problem_seed=problem_seed,
                        subproblem_id=pool_id,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        best=best,
                        score=candidate_scores[id(candidate)][0],
                        phase2_feedback_score=candidate_scores[id(candidate)][1],
                        objective_scope=normalized_objective_scope,
                    )
                )
            best_action_rows.extend(
                _best_action_rows(
                    episode=episode,
                    problem_id=problem_id,
                    block_count=pool_block_count,
                    problem_seed=problem_seed,
                    subproblem_id=pool_id,
                    best=best,
                    objective_scope=normalized_objective_scope,
                )
            )
            subproblem_metric_rows.append(
                {
                    "episode": episode,
                    "problem_id": problem_id,
                    "subproblem_id": pool_id,
                    "series": "|".join(PHASE1_RESOURCE_POOL_SERIES[pool_id]),
                    "block_count": pool_block_count,
                    "job_count": len(pool_jobs),
                    "best_source": best.source,
                    "objective_scope": normalized_objective_scope,
                    "loss": loss,
                    "score_json": json.dumps(list(score), ensure_ascii=False),
                    "phase2_feedback_score_json": json.dumps(list(phase2_feedback_score), ensure_ascii=False),
                    "learning_score_json": json.dumps(list(learning_score), ensure_ascii=False),
                    "candidate_count": len(candidates),
                }
            )
            print(
                "[CHECK][phase1_pair_self_labeling.train.subproblem] "
                f"episode={episode} subproblem_id={pool_id} block_count={pool_block_count} "
                f"best_source={best.source} loss={loss:.6f} phase1_score={score}"
            )

        combined_best = _merge_phase1_resource_pool_candidates(
            jobs=jobs,
            bay_ids=episode_bay_ids,
            bay_capacity_weights=episode_capacity_weights,
            selected=selected_subproblems,
        )
        combined_agent_best = (
            _merge_phase1_resource_pool_candidates(
                jobs=jobs,
                bay_ids=episode_bay_ids,
                bay_capacity_weights=episode_capacity_weights,
                selected=selected_agent_subproblems,
            )
            if checkpoint_due
            else None
        )
        loss = sum(subproblem_losses) / len(subproblem_losses)
        score = _score_bay_loads(combined_best.bay_loads, normalized_objective_scope)
        phase2_feedback_score = _candidate_phase2_feedback_score(
            candidate=combined_best,
            jobs=jobs,
            bay_ids=episode_bay_ids,
            phase2_feedback_scorer=phase2_feedback_scorer,
        )
        learning_score = phase2_feedback_score + score
        # LINE-BY-LINE: episode 단위 학습 metric row를 누적합니다.
        metrics_rows.append(
            {
                "episode": episode,
                "problem_id": problem_id,
                "block_count": block_count,
                "problem_seed": problem_seed,
                "case_type": case_type,
                "hard_case_mode": hard_case_mode,
                "hard_case_corr_steel_cut_before": hard_case_corr_before,
                "hard_case_corr_steel_cut_after": hard_case_corr_after,
                "best_source": combined_best.source,
                "score_mode": "wo_first",
                "objective_scope": normalized_objective_scope,
                "loss": loss,
                "score_json": json.dumps(list(score), ensure_ascii=False),
                "phase2_feedback_score_json": json.dumps(list(phase2_feedback_score), ensure_ascii=False),
                "learning_score_json": json.dumps(list(learning_score), ensure_ascii=False),
                "candidate_count": total_candidate_count,
                "subproblem_count": len(selected_subproblems),
            }
        )
        # LINE-BY-LINE: 현재까지 metrics를 즉시 파일에 씁니다. 중간 중단되어도 진행 상황이 남습니다.
        _write_metrics(output_path / "metrics.csv", metrics_rows)
        _write_subproblem_metrics(output_path / "subproblem_metrics.csv", subproblem_metric_rows)
        # LINE-BY-LINE: 현재까지 후보 audit row를 즉시 파일에 씁니다.
        _write_candidate_summary(output_path / "candidate_summary.csv", candidate_rows)
        # LINE-BY-LINE: 현재까지 pseudo-label action table을 즉시 파일에 씁니다.
        _write_jsonl(output_path / "best_action_table.jsonl", best_action_rows)
        # LINE-BY-LINE: 콘솔에 episode 진행 상황과 best source/score를 출력합니다.
        print(
            "[CHECK][phase1_pair_self_labeling.train] "
            f"episode={episode} problem_id={problem_id} block_count={block_count} "
            f"best_source={combined_best.source} subproblem_count={len(selected_subproblems)} "
            f"loss={loss:.6f} "
            f"learning_score={learning_score} phase1_score={score}"
        )
        # LINE-BY-LINE: checkpoint interval에 도달하면 주기 checkpoint를 저장합니다.
        if checkpoint_due:
            _save_pair_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoint_dir / f"phase1_pair_pointer_ep{episode:05d}.pt",
                hidden_dim=hidden_dim,
                episode=episode,
                validation_score=None,
                phase2_feedback_contract=normalized_feedback_contract,
                feature_schema=feature_schema,
            )
            if combined_agent_best is None:
                print(
                    "[ERROR][phase1_pair_self_labeling.train] "
                    f"cause=missing_checkpoint_agent_solution episode={episode}"
                )
                raise RuntimeError("Phase 1 checkpoint agent solution was not built")
            _write_phase1_checkpoint_solutions(
                checkpoint_dir=checkpoint_dir,
                episode=episode,
                jobs=jobs,
                bay_ids=episode_bay_ids,
                objective_scope=normalized_objective_scope,
                teacher_best=combined_best,
                agent_best=combined_agent_best,
            )
        # LINE-BY-LINE: validation factory가 있고 validation interval에 도달하면 holdout 검증을 실행합니다.
        if validation_episode_factory is not None and episode % validation_every == 0:
            # LINE-BY-LINE: validation 문제에서 agent_greedy와 heuristic bank를 비교합니다.
            validation = _validate_pair_policy(
                model=model,
                validation_episode_factory=validation_episode_factory,
                validation_episodes=validation_episodes,
                bay_ids=normalized_bay_ids,
                heuristic_algorithms=heuristic_algorithms,
                episode=episode,
                rollout_samples=resolved_validation_rollout_samples,
                bay_capacity_weights=normalized_capacity_weights,
                phase2_feedback_scorer=phase2_feedback_scorer,
                objective_scope=normalized_objective_scope,
            )
            # LINE-BY-LINE: validation episode별 agent 요약 row를 누적합니다.
            validation_rows.extend(validation["rows"])
            # LINE-BY-LINE: validation에서 평가한 모든 방법론 후보 row를 누적합니다.
            validation_candidate_rows.extend(validation["candidate_rows"])
            # LINE-BY-LINE: validation summary CSV를 저장합니다.
            _write_validation_summary(output_path / "validation_summary.csv", validation_rows)
            # LINE-BY-LINE: validation 후보별 score/rank CSV를 저장합니다.
            _write_validation_candidate_summary(output_path / "validation_candidate_summary.csv", validation_candidate_rows)
            # LINE-BY-LINE: 최신 validation 기준 scatter/bar plot PNG를 저장합니다.
            validation_plot_paths = _write_validation_plots(
                output_path=output_path,
                candidate_rows=validation_candidate_rows,
                summary_rows=validation_rows,
                objective_scope=normalized_objective_scope,
            )
            # LINE-BY-LINE: 모든 MIXED validation 문제의 best-of-K 평균을 checkpoint 기준으로 사용합니다.
            current_score = validation["agent_mean_score"]
            # LINE-BY-LINE: 이전 best보다 lexicographic score가 좋으면 best checkpoint를 갱신합니다.
            saved_best = best_validation_score is None or current_score < best_validation_score
            if saved_best:
                best_validation_score = current_score
                _save_pair_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path=best_checkpoint_path,
                    hidden_dim=hidden_dim,
                    episode=episode,
                    validation_score=current_score,
                    phase2_feedback_contract=normalized_feedback_contract,
                    feature_schema=feature_schema,
                )
            # LINE-BY-LINE: validation 요약을 콘솔에 출력합니다. best_checkpoint_saved 여부가 핵심입니다.
            print(
                "[VALIDATION][phase1_pair_self_labeling.validate] "
                f"episode={episode} validation_episodes={validation_episodes} "
                f"agent_best_rate={validation['agent_best_rate']:.6f} "
                f"checkpoint_score={current_score} best_checkpoint_saved={saved_best}"
            )

    # LINE-BY-LINE: 학습 종료 시점의 최종 모델 checkpoint path입니다.
    checkpoint_path = output_path / "phase1_pair_pointer.pt"
    # LINE-BY-LINE: 학습 결과 요약 JSON path입니다.
    summary_path = output_path / "summary.json"
    # LINE-BY-LINE: 마지막 episode의 model/optimizer state를 최종 checkpoint로 저장합니다.
    _save_pair_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        hidden_dim=hidden_dim,
        episode=episodes,
        validation_score=None,
        phase2_feedback_contract=normalized_feedback_contract,
        feature_schema=feature_schema,
    )
    # LINE-BY-LINE: Phase 2 feedback scorer 사용 여부와 score prefix 길이를 summary에 남깁니다. 사용: 서로 다른 학습 run 비교 시 같은 목적함수였는지 확인합니다.
    phase2_feedback_score_length = 0
    if phase2_feedback_scorer is not None and metrics_rows:
        try:
            phase2_feedback_score_length = len(json.loads(metrics_rows[-1]["phase2_feedback_score_json"]))
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            print(
                "[ERROR][phase1_pair_self_labeling.train] "
                f"cause=invalid_phase2_feedback_score_json episode={metrics_rows[-1].get('episode')}"
            )
            raise RuntimeError("invalid phase2_feedback_score_json in metrics rows") from exc
    # LINE-BY-LINE: CLI와 외부 notebook이 읽을 수 있는 summary dict를 구성합니다.
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_validation_score is not None else "",
        "metrics_csv": str(output_path / "metrics.csv"),
        "subproblem_metrics_csv": str(output_path / "subproblem_metrics.csv"),
        "candidate_summary_csv": str(output_path / "candidate_summary.csv"),
        "best_action_table_jsonl": str(output_path / "best_action_table.jsonl"),
        "checkpoint_solution_dir": str(checkpoint_dir / "solutions"),
        "validation_summary_csv": str(output_path / "validation_summary.csv") if validation_rows else "",
        "validation_candidate_summary_csv": str(output_path / "validation_candidate_summary.csv") if validation_candidate_rows else "",
        **validation_plot_paths,
        "summary_json": str(summary_path),
        "episodes": episodes,
        "rollout_samples": rollout_samples,
        "validation_rollout_samples": resolved_validation_rollout_samples,
        "bay_ids": list(normalized_bay_ids),
        "bay_capacity_weights": dict(normalized_capacity_weights),
        "heuristic_algorithms": list(heuristic_algorithms),
        "score_mode": "wo_first",
        "objective_scope": normalized_objective_scope,
        "objective_scope_transition": objective_scope_transition,
        "previous_objective_scope": previous_objective_scope,
        "rule_profile": MULTI_SERIES_RULE_PROFILE,
        "episode_scope_mode": "two_resource_pool_subproblems",
        "episode_scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
        "optimizer_update_count": len(subproblem_metric_rows),
        "pair_feature_names": list(feature_schema["pair"]),
        "env_feature_names": list(feature_schema["env"]),
        "score_field_names": list(
            PHASE1_SCORE_FIELD_NAMES[:len(phase1_objective_field_names(normalized_objective_scope))]
        ),
        "objective_field_names": list(phase1_objective_field_names(normalized_objective_scope)),
        "best_source_counts": _source_counts(metrics_rows),
        "episode_mode": "on_the_fly" if episode_factory is not None else "prebuilt",
        "checkpoint_every": checkpoint_every,
        "validation_every": validation_every,
        "validation_episodes": validation_episodes,
        "phase2_feedback_score_enabled": phase2_feedback_scorer is not None,
        "phase2_feedback_score_length": phase2_feedback_score_length,
        "phase2_feedback_contract": normalized_feedback_contract,
        "device": str(torch_device),
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
        "start_episode": start_episode,
        "resumed_from_episode": start_episode - 1 if resume_path is not None else 0,
        "best_validation_score": list(best_validation_score) if best_validation_score is not None else [],
    }
    # LINE-BY-LINE: summary.json을 UTF-8 JSON으로 저장합니다.
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    # LINE-BY-LINE: 호출자(main.py)가 출력할 수 있도록 summary dict를 반환합니다.
    return summary


def run_phase1_pair_policy_rollout(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    model: Phase1PairPointerPolicy | None,
    temperature: float,
    seed: int,
    source: str,
    selection: str = "sample",
    bay_capacity_weights: Mapping[str, int | float] | None = None,
) -> Phase1PairCandidate:
    """한 자원군의 hard mask를 지키는 complete pair assignment를 생성한다."""

    if temperature <= 0:
        print(f"[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_rollout] cause=non_positive_temperature value={temperature}")
        raise ValueError("temperature must be positive")
    if selection not in {"sample", "greedy"}:
        print(f"[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_rollout] cause=unknown_selection selection={selection}")
        raise RuntimeError(f"unknown pair rollout selection: {selection}")
    _validate_pair_policy_model_contract(model)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    cache = _build_phase1_pair_episode_cache(
        jobs=jobs,
        bay_ids=bay_ids,
        bay_capacity_weights=bay_capacity_weights,
    )
    blocks = list(cache.blocks)
    planning_state = _create_pair_planning_state(cache.bay_capacity_weights)
    bay_loads = planning_state.bay_loads
    remaining = {block.block_set_id for block in blocks}
    transitions: List[Phase1PairTransition] = []
    assignments = planning_state.block_to_bay

    for block_step in range(len(blocks)):
        candidates = _build_phase1_pair_candidates_from_cache(
            cache=cache,
            bay_loads=bay_loads,
            remaining_block_ids=sorted(remaining),
        )
        env_features = _pair_env_features(
            bay_loads=bay_loads,
            assigned_block_count=block_step,
            total_block_count=len(blocks),
            totals=cache.env_totals,
        )
        logits = _score_pair_candidates(model, [row["features"] for row in candidates], env_features)
        selected_index = int(torch.argmax(logits).item()) if selection == "greedy" else _sample_index(
            logits=logits,
            temperature=temperature,
            generator=generator,
        )
        selected = candidates[selected_index]
        selected_block_id = str(selected["block_set_id"])
        selected_bay = str(selected["bay_id"])
        transitions.append(
            Phase1PairTransition(
                phase="SELECT_PAIR",
                candidate_features=[row["features"] for row in candidates],
                env_features=env_features,
                selected_action_index=selected_index,
                candidate_ids=[str(row["action_id"]) for row in candidates],
                selected_action_id=str(selected["action_id"]),
                selected_block_set_id=selected_block_id,
                selected_bay=selected_bay,
            )
        )
        block = cache.block_by_id[selected_block_id]
        _apply_block_to_planning_state(planning_state, selected_bay, block)
        remaining.remove(selected_block_id)

    complete_phase1_planning(planning_state, set(cache.block_by_id))

    return Phase1PairCandidate(
        source=source,
        transitions=transitions,
        assignments=assignments,
        bay_loads=_plain_bay_loads(bay_loads),
    )


def run_phase1_pair_policy_resource_pool_best_of_k(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    model: Phase1PairPointerPolicy | None,
    sample_count: int,
    temperature: float,
    seed: int,
    bay_capacity_weights: Mapping[str, int | float] | None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Phase1PairCandidate:
    """두 자원군에서 best-of-K를 독립 선택한 뒤 하나의 완전 배정으로 병합한다."""

    if sample_count <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_resource_pool_best_of_k] "
            f"cause=non_positive_sample_count value={sample_count}"
        )
        raise RuntimeError("Phase 1 best-of-K sample_count must be positive")
    normalized_scope = normalize_phase1_objective_scope(objective_scope)
    if model is not None and model.objective_scope != normalized_scope:
        print(
            "[ERROR][phase1_pair_self_labeling.run_phase1_pair_policy_resource_pool_best_of_k] "
            f"cause=objective_scope_mismatch model={model.objective_scope} requested={normalized_scope}"
        )
        raise RuntimeError("Phase 1 best-of-K objective scope mismatch")
    subproblems = split_phase1_jobs_by_resource_pool(jobs)
    selected: List[tuple[str, Phase1PairCandidate]] = []
    for pool_index, pool_id in enumerate(PHASE1_RESOURCE_POOL_ORDER):
        pool_jobs = subproblems.get(pool_id)
        if not pool_jobs:
            continue
        candidates = []
        for sample_index in range(1, sample_count + 1):
            is_greedy = sample_index == 1
            candidates.append(
                run_phase1_pair_policy_rollout(
                    jobs=pool_jobs,
                    bay_ids=bay_ids,
                    model=model,
                    temperature=temperature,
                    seed=seed + pool_index * 1_000_000 + sample_index,
                    source="agent_greedy" if is_greedy else f"agent_sample_{sample_index}",
                    selection="greedy" if is_greedy else "sample",
                    bay_capacity_weights=bay_capacity_weights,
                )
            )
        best = min(
            candidates,
            key=lambda candidate: _score_bay_loads(
                candidate.bay_loads,
                objective_scope=normalized_scope,
            ),
        )
        selected.append((pool_id, best))
    return _merge_phase1_resource_pool_candidates(
        jobs=jobs,
        bay_ids=bay_ids,
        bay_capacity_weights=bay_capacity_weights,
        selected=selected,
    )


def _merge_phase1_resource_pool_candidates(
    *,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float] | None,
    selected: Sequence[tuple[str, Phase1PairCandidate]],
) -> Phase1PairCandidate:
    """독립 자원군 후보의 중복·누락을 검사하고 전체 Bay load를 재계산한다."""

    normalized_bays = _normalize_bay_ids(bay_ids)
    normalized_weights = _normalize_bay_capacity_weights(normalized_bays, bay_capacity_weights)
    subproblems = split_phase1_jobs_by_resource_pool(jobs)
    selected_by_pool: Dict[str, Phase1PairCandidate] = {}
    for pool_id, candidate in selected:
        if pool_id in selected_by_pool:
            print(
                "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
                f"cause=duplicate_subproblem_selection pool_id={pool_id}"
            )
            raise RuntimeError("duplicate Phase 1 resource-pool selection")
        selected_by_pool[pool_id] = candidate
    if set(selected_by_pool) != set(subproblems):
        print(
            "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
            f"cause=subproblem_scope_mismatch expected={sorted(subproblems)} "
            f"actual={sorted(selected_by_pool)}"
        )
        raise RuntimeError("Phase 1 resource-pool selections are incomplete")
    blocks = _collect_blocks(jobs=jobs, bay_ids=normalized_bays)
    expected_block_ids = {block.block_set_id for block in blocks}
    expected_blocks_by_pool = {
        pool_id: {
            block.block_set_id
            for block in blocks
            if phase1_resource_pool_for_series(block.family) == pool_id
        }
        for pool_id in subproblems
    }
    assignments: Dict[str, str] = {}
    transitions: List[Phase1PairTransition] = []
    sources: List[str] = []
    for pool_id in PHASE1_RESOURCE_POOL_ORDER:
        candidate = selected_by_pool.get(pool_id)
        if candidate is None:
            continue
        if set(candidate.assignments) != expected_blocks_by_pool[pool_id]:
            print(
                "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
                f"cause=assignment_scope_mismatch pool_id={pool_id} "
                f"expected={sorted(expected_blocks_by_pool[pool_id])} "
                f"actual={sorted(candidate.assignments)}"
            )
            raise RuntimeError(f"Phase 1 resource-pool assignment scope mismatch: {pool_id}")
        for block_id, assigned_bay in candidate.assignments.items():
            if assigned_bay not in PHASE1_RESOURCE_POOL_BAYS[pool_id]:
                print(
                    "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
                    f"cause=resource_pool_bay_mismatch pool_id={pool_id} "
                    f"block_set_id={block_id} bay_id={assigned_bay}"
                )
                raise RuntimeError(f"Phase 1 resource-pool Bay mismatch: {pool_id}")
        duplicate_ids = sorted(set(assignments) & set(candidate.assignments))
        if duplicate_ids:
            print(
                "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
                f"cause=duplicate_assignments pool_id={pool_id} block_ids={duplicate_ids}"
            )
            raise RuntimeError("duplicate Phase 1 resource-pool assignments")
        assignments.update(candidate.assignments)
        transitions.extend(candidate.transitions)
        sources.append(f"{pool_id}:{candidate.source}")
    if set(assignments) != expected_block_ids:
        missing = sorted(expected_block_ids - set(assignments))
        extra = sorted(set(assignments) - expected_block_ids)
        print(
            "[ERROR][phase1_pair_self_labeling._merge_phase1_resource_pool_candidates] "
            f"cause=incomplete_assignment missing={missing} extra={extra}"
        )
        raise RuntimeError("incomplete Phase 1 resource-pool assignment")

    planning_state = _create_pair_planning_state(normalized_weights)
    for block in blocks:
        _apply_block_to_planning_state(planning_state, assignments[block.block_set_id], block)
    complete_phase1_planning(planning_state, expected_block_ids)
    return Phase1PairCandidate(
        source="|".join(sources),
        transitions=transitions,
        assignments=dict(assignments),
        bay_loads=_plain_bay_loads(planning_state.bay_loads),
    )


def _validate_pair_policy_model_contract(model: Phase1PairPointerPolicy | None) -> None:
    """모델이 유일한 MIXED feature 계약과 정확히 일치하는지 검증한다."""

    if model is None:
        return
    model_rule_profile = getattr(model, "rule_profile", None)
    model_score_mode = getattr(model, "score_mode", None)
    model_objective_scope = getattr(model, "objective_scope", None)
    try:
        normalized_objective_scope = normalize_phase1_objective_scope(model_objective_scope)
    except RuntimeError as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_pair_policy_model_contract] "
            f"cause=model_objective_scope_mismatch value={model_objective_scope}"
        )
        raise RuntimeError("Phase 1 pair model objective scope is invalid") from exc
    if model_rule_profile != MULTI_SERIES_RULE_PROFILE or model_score_mode != "wo_first":
        print(
            "[ERROR][phase1_pair_self_labeling._validate_pair_policy_model_contract] "
            f"cause=model_contract_mismatch model_profile={model_rule_profile} "
            f"model_score={model_score_mode}"
        )
        raise RuntimeError("Phase 1 pair model must use the MIXED/wo_first contract")
    if model_objective_scope != normalized_objective_scope:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_pair_policy_model_contract] "
            f"cause=noncanonical_model_objective_scope value={model_objective_scope} "
            f"normalized={normalized_objective_scope}"
        )
        raise RuntimeError("Phase 1 pair model objective scope must be canonical")
    feature_schema = phase1_pair_feature_schema()
    if model.pair_feature_dim != len(feature_schema["pair"]) or model.env_feature_dim != len(feature_schema["env"]):
        print(
            "[ERROR][phase1_pair_self_labeling._validate_pair_policy_model_contract] "
            "cause=model_feature_dim_mismatch "
            f"pair={model.pair_feature_dim}/{len(feature_schema['pair'])} "
            f"env={model.env_feature_dim}/{len(feature_schema['env'])}"
        )
        raise RuntimeError("Phase 1 pair model dimensions do not match the MIXED schema")


def _validate_pair_policy(
    model: Phase1PairPointerPolicy,
    validation_episode_factory: Callable[[int], Mapping[str, object]],
    validation_episodes: int,
    bay_ids: Sequence[str],
    heuristic_algorithms: Sequence[str],
    episode: int,
    rollout_samples: int,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
    phase2_feedback_scorer: Phase2FeedbackScorer | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Dict:
    """Evaluate six series/resource-pool views without mixing independent teachers."""

    rows: List[Dict] = []
    candidate_rows: List[Dict] = []
    agent_scores: List[tuple] = []
    agent_best_count = 0
    for validation_index in range(1, validation_episodes + 1):
        jobs, metadata = _episode_payload(validation_index, None, None, validation_episode_factory)
        validation_bay_ids, validation_capacity_weights = _resolve_phase1_episode_scope(
            metadata=metadata,
            default_bay_ids=bay_ids,
            default_capacity_weights=_normalize_bay_capacity_weights(bay_ids, bay_capacity_weights),
        )
        parent_problem_id = str(
            metadata.get("problem_id") or metadata.get("episode_id") or f"VAL{validation_index:05d}"
        )
        validation_source = str(metadata.get("validation_source") or "synthetic")
        evaluation_input_type = _validation_input_type(metadata, validation_source)
        case_type = str(metadata.get("case_type") or "")
        hard_case_mode = str(metadata.get("hard_case_mode") or "")
        hard_case_corr_before = metadata.get("hard_case_corr_steel_cut_before", "")
        hard_case_corr_after = metadata.get("hard_case_corr_steel_cut_after", "")
        for view_index, (validation_view, view_jobs) in enumerate(
            _phase1_validation_views(jobs).items(), start=1
        ):
            problem_id = f"{parent_problem_id}::{validation_view}"
            first_view_job = next(iter(view_jobs.values()))
            subproblem_id = phase1_resource_pool_for_series(
                _job_attr(first_view_job, "family")
            )
            block_count = len(_collect_blocks(view_jobs, validation_bay_ids))
            candidates: List[Phase1PairCandidate] = []
            for sample_index in range(1, rollout_samples + 1):
                is_greedy = sample_index == 1
                candidates.append(
                    run_phase1_pair_policy_rollout(
                        jobs=view_jobs,
                        bay_ids=validation_bay_ids,
                        model=model,
                        temperature=1.0,
                        seed=(
                            episode * 1_000_000
                            + validation_index * 10_000
                            + view_index * 100
                            + sample_index
                        ),
                        source="agent_greedy" if is_greedy else f"agent_sample_{sample_index}",
                        selection="greedy" if is_greedy else "sample",
                        bay_capacity_weights=validation_capacity_weights,
                    )
                )
            for algorithm in heuristic_algorithms:
                candidates.append(
                    _run_phase1_pair_heuristic_candidate(
                        view_jobs,
                        validation_bay_ids,
                        algorithm,
                        bay_capacity_weights=validation_capacity_weights,
                        objective_scope=objective_scope,
                    )
                )
            candidate_scores = {
                id(candidate): _candidate_score_components(
                    candidate=candidate,
                    jobs=view_jobs,
                    bay_ids=validation_bay_ids,
                    phase2_feedback_scorer=phase2_feedback_scorer,
                    objective_scope=objective_scope,
                )
                for candidate in candidates
            }
            scored = sorted(
                ((candidate_scores[id(candidate)][2], candidate) for candidate in candidates),
                key=lambda item: item[0],
            )
            rank_by_source = {
                candidate.source: rank
                for rank, (_, candidate) in enumerate(scored, start=1)
            }
            agent_candidates = [candidate for candidate in candidates if _is_agent_source(candidate.source)]
            best_agent_learning_score, best_agent_candidate = min(
                ((candidate_scores[id(candidate)][2], candidate) for candidate in agent_candidates),
                key=lambda item: item[0],
            )
            agent_score = candidate_scores[id(best_agent_candidate)][0]
            best_learning_score, best_candidate = scored[0]
            best_score = candidate_scores[id(best_candidate)][0]
            agent_rank = rank_by_source[best_agent_candidate.source]
            agent_is_best = int(agent_rank == 1)
            agent_best_count += agent_is_best
            agent_scores.append(best_agent_learning_score)
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_rows.append(
                    _validation_candidate_summary_row(
                        train_episode=episode,
                        validation_episode=validation_index,
                        validation_source=validation_source,
                        evaluation_input_type=evaluation_input_type,
                        parent_problem_id=parent_problem_id,
                        problem_id=problem_id,
                        subproblem_id=subproblem_id,
                        validation_view=validation_view,
                        block_count=block_count,
                        candidate_index=candidate_index,
                        candidate=candidate,
                        best=best_candidate,
                        rank=rank_by_source[candidate.source],
                        score=candidate_scores[id(candidate)][0],
                        phase2_feedback_score=candidate_scores[id(candidate)][1],
                        objective_scope=objective_scope,
                        gap_breakdown=_phase1_validation_gap_breakdown(
                            candidate.bay_loads, view_jobs
                        ),
                    )
                )
            agent_gaps = _phase1_validation_gap_breakdown(
                best_agent_candidate.bay_loads, view_jobs
            )
            rows.append(
                {
                    "train_episode": episode,
                    "validation_episode": validation_index,
                    "validation_source": validation_source,
                    "evaluation_input_type": evaluation_input_type,
                    "parent_problem_id": parent_problem_id,
                    "problem_id": problem_id,
                    "subproblem_id": subproblem_id,
                    "validation_view": validation_view,
                    "block_count": block_count,
                    "case_type": case_type,
                    "hard_case_mode": hard_case_mode,
                    "hard_case_corr_steel_cut_before": hard_case_corr_before,
                    "hard_case_corr_steel_cut_after": hard_case_corr_after,
                    "agent_score_json": json.dumps(list(agent_score), ensure_ascii=False),
                    "agent_learning_score_json": json.dumps(
                        list(best_agent_learning_score), ensure_ascii=False
                    ),
                    "best_score_json": json.dumps(list(best_score), ensure_ascii=False),
                    "best_learning_score_json": json.dumps(
                        list(best_learning_score), ensure_ascii=False
                    ),
                    "best_source": best_candidate.source,
                    "agent_best_source": best_agent_candidate.source,
                    "agent_is_best": agent_is_best,
                    "agent_rank": agent_rank,
                    "candidate_count": len(candidates),
                    "objective_scope": objective_scope,
                    **agent_gaps,
                }
            )
    if not rows:
        print("[ERROR][phase1_pair_self_labeling._validate_pair_policy] cause=no_validation_views")
        raise RuntimeError("Phase 1 validation produced no resource-pool views")
    return {
        "rows": rows,
        "candidate_rows": candidate_rows,
        "agent_mean_score": _mean_score(agent_scores),
        "agent_best_rate": agent_best_count / len(rows),
    }


def _phase1_validation_views(jobs: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    """Build ordered single-series and mixed resource-pool validation views."""

    by_series: Dict[str, Dict[str, object]] = {series: {} for series in ("NP", "NC", "FN", "FL")}
    for job_key, job in jobs.items():
        family = str(_job_attr(job, "family") or "").strip().upper()
        if family not in by_series:
            print(
                "[ERROR][phase1_pair_self_labeling._phase1_validation_views] "
                f"cause=unsupported_series job_key={job_key} family={family}"
            )
            raise RuntimeError(f"unsupported Phase 1 validation series: {family}")
        by_series[family][str(job_key)] = job
    views: Dict[str, Dict[str, object]] = {}
    for view in PHASE1_VALIDATION_VIEW_ORDER:
        series_values = PHASE1_RESOURCE_POOL_SERIES.get(view, (view,))
        view_jobs = {
            job_key: job
            for series in series_values
            for job_key, job in by_series[series].items()
        }
        if view_jobs:
            views[view] = view_jobs
    return views


def _phase1_validation_gap_breakdown(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    jobs: Mapping[str, object],
) -> Dict[str, float | str]:
    """Return per-series and active-pool capacity-normalized gap diagnostics."""

    present_series = {
        str(_job_attr(job, "family") or "").strip().upper()
        for job in jobs.values()
    }
    result: Dict[str, float | str] = {field: "" for field in PHASE1_VALIDATION_GAP_FIELDS}
    metric_fields = {
        "wo": "wo_count",
        "cut": "cut_length_sum",
        "bv": "bevel_quantity_sum",
    }
    for series in ("NP", "NC", "FN", "FL"):
        if series not in present_series:
            continue
        weights = GROUP_BAY_CAPACITY_WEIGHTS[series]
        for metric_name, load_metric in metric_fields.items():
            values = [
                multi_series_group_load_value(bay_loads[bay_id], series, load_metric)
                / weights[bay_id]
                for bay_id in weights
            ]
            result[f"{series.lower()}_{metric_name}_gap"] = round(max(values) - min(values), 9)
    for pool_id in PHASE1_RESOURCE_POOL_ORDER:
        pool_series = PHASE1_RESOURCE_POOL_SERIES[pool_id]
        if not present_series.intersection(pool_series):
            continue
        pool_bays = tuple(GROUP_BAY_CAPACITY_WEIGHTS[pool_series[0]])
        for metric_name, load_metric in metric_fields.items():
            values = [
                sum(
                    multi_series_group_load_value(bay_loads[bay_id], series, load_metric)
                    for series in pool_series
                )
                / _require_capacity_weight(bay_loads[bay_id], f"validation_gap:{pool_id}:{bay_id}")
                for bay_id in pool_bays
            ]
            result[f"{pool_id.lower()}_{metric_name}_gap"] = round(
                max(values) - min(values), 9
            )
    return result


def _is_agent_source(source: str) -> bool:
    """Return True for policy-generated validation candidates."""

    return source == "agent_greedy" or source.startswith("agent_sample_")


def _write_phase1_checkpoint_solutions(
    *,
    checkpoint_dir: Path,
    episode: int,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    objective_scope: str,
    teacher_best: Phase1PairCandidate,
    agent_best: Phase1PairCandidate,
) -> None:
    """Checkpoint episode의 teacher/agent 전체 Bay 배정 해를 저장한다."""

    solution_root = checkpoint_dir / "solutions" / f"episode_{episode:05d}"
    for role, candidate in (("teacher_best", teacher_best), ("agent_best", agent_best)):
        plan = candidate_to_phase1_plan(
            jobs=jobs,
            bay_ids=bay_ids,
            candidate=candidate,
            objective_scope=objective_scope,
        )
        plan["checkpoint_episode"] = episode
        plan["solution_role"] = role
        paths = write_phase1_bay_plan(plan, solution_root / role)
        print(
            "[CHECK][phase1_pair_self_labeling._write_phase1_checkpoint_solutions] "
            f"episode={episode} role={role} source={candidate.source} "
            f"plan_json={paths['json']}"
        )


def _save_pair_checkpoint(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
    episode: int,
    validation_score: tuple | None,
    phase2_feedback_contract: Mapping[str, object] | None,
    feature_schema: Mapping[str, Sequence[str]],
) -> None:
    """Save an auditable pair-policy checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "pair_feature_names": list(feature_schema["pair"]),
            "env_feature_names": list(feature_schema["env"]),
            "rule_profile": MULTI_SERIES_RULE_PROFILE,
            "score_mode": "wo_first",
            "objective_scope": model.objective_scope,
            "episode_scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
            "hidden_dim": hidden_dim,
            "episode": episode,
            "validation_score": list(validation_score) if validation_score is not None else [],
            "phase2_feedback_contract": phase2_feedback_contract,
        },
        path,
    )


def _resolve_resume_checkpoint(output_path: Path, resume_checkpoint: str | Path | None) -> Path | None:
    """Resolve resume checkpoint path, supporting `latest` under output/checkpoints."""

    if resume_checkpoint in (None, ""):
        return None
    text = str(resume_checkpoint)
    if text == "latest":
        candidates = sorted((output_path / "checkpoints").glob("phase1_pair_pointer_ep*.pt"))
        if not candidates:
            print(
                "[ERROR][phase1_pair_self_labeling._resolve_resume_checkpoint] "
                f"cause=no_latest_checkpoint output_dir={output_path}"
            )
            raise RuntimeError(f"no latest checkpoint in {output_path / 'checkpoints'}")
        return candidates[-1]
    path = Path(text)
    if not path.exists():
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_resume_checkpoint] "
            f"cause=missing_resume_checkpoint path={path}"
        )
        raise RuntimeError(f"missing_resume_checkpoint: {path}")
    return path


def _load_pair_checkpoint(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
    phase2_feedback_contract: Mapping[str, object] | None,
    feature_schema: Mapping[str, Sequence[str]],
    objective_scope: str,
) -> tuple[int, str]:
    """Load model/optimizer state and return episode plus previous objective scope."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_hidden_dim = int(checkpoint.get("hidden_dim", -1))
    if checkpoint_hidden_dim != hidden_dim:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=hidden_dim_mismatch checkpoint={checkpoint_hidden_dim} requested={hidden_dim} path={path}"
        )
        raise RuntimeError("resume checkpoint hidden_dim mismatch")
    checkpoint_rule_profile = checkpoint.get("rule_profile")
    checkpoint_score_mode = checkpoint.get("score_mode")
    if checkpoint_rule_profile != MULTI_SERIES_RULE_PROFILE or checkpoint_score_mode != "wo_first":
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=policy_contract_mismatch path={path} "
            f"checkpoint_profile={checkpoint_rule_profile} checkpoint_score={checkpoint_score_mode}"
        )
        raise RuntimeError("only MIXED/wo_first Phase 1 checkpoints are supported")
    checkpoint_objective_scope = checkpoint.get("objective_scope")
    if checkpoint_objective_scope is None:
        checkpoint_objective_scope = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES
        print(
            "[CHECK][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"path={path} legacy_objective_scope={checkpoint_objective_scope}"
        )
    checkpoint_objective_scope = normalize_phase1_objective_scope(checkpoint_objective_scope)
    if checkpoint_objective_scope != objective_scope:
        print(
            "[CHECK][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"objective_scope_transition=true path={path} "
            f"checkpoint={checkpoint_objective_scope} requested={objective_scope} "
            "validation_best_reset=true"
        )
    checkpoint_scope_version = checkpoint.get("episode_scope_version")
    if checkpoint_scope_version != PHASE1_MULTI_SERIES_SCOPE_VERSION:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=episode_scope_contract_mismatch path={path} "
            f"checkpoint={checkpoint_scope_version} expected={PHASE1_MULTI_SERIES_SCOPE_VERSION}"
        )
        raise RuntimeError("resume checkpoint Phase 1 episode scope contract mismatch")
    if checkpoint.get("pair_feature_names") != list(feature_schema["pair"]):
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=pair_feature_names_mismatch path={path}"
        )
        raise RuntimeError("resume checkpoint pair features mismatch")
    if checkpoint.get("env_feature_names") != list(feature_schema["env"]):
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=env_feature_names_mismatch path={path}"
        )
        raise RuntimeError("resume checkpoint env features mismatch")
    checkpoint_feedback_contract = checkpoint.get("phase2_feedback_contract")
    if checkpoint_feedback_contract != phase2_feedback_contract:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=phase2_feedback_contract_mismatch path={path} "
            f"checkpoint_enabled={checkpoint_feedback_contract is not None} "
            f"requested_enabled={phase2_feedback_contract is not None}"
        )
        raise RuntimeError("resume checkpoint Phase 2 feedback contract mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    episode = int(checkpoint.get("episode", 0))
    if episode <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._load_pair_checkpoint] "
            f"cause=invalid_checkpoint_episode episode={episode} path={path}"
        )
        raise RuntimeError("invalid resume checkpoint episode")
    print(
        "[CHECK][phase1_pair_self_labeling._load_pair_checkpoint] "
        f"path={path} completed_episode={episode}"
    )
    return episode, checkpoint_objective_scope


def _resolve_torch_device(device: str) -> torch.device:
    """Resolve an explicit torch device without silent CPU fallback."""

    requested = str(device or "cpu").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[ERROR][phase1_pair_self_labeling._resolve_torch_device] cause=cuda_requested_but_unavailable")
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(
                "[ERROR][phase1_pair_self_labeling._resolve_torch_device] "
                f"cause=cuda_device_requested_but_unavailable device={device}"
            )
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        torch_device = torch.device(requested)
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            print(
                "[ERROR][phase1_pair_self_labeling._resolve_torch_device] "
                f"cause=cuda_index_out_of_range device={device} count={torch.cuda.device_count()}"
            )
            raise RuntimeError(f"CUDA device index out of range: {device}")
        return torch_device
    print(f"[ERROR][phase1_pair_self_labeling._resolve_torch_device] cause=unknown_device device={device}")
    raise RuntimeError(f"unknown torch device: {device}")


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer momentum/state tensors after loading a checkpoint."""

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _read_csv_rows(path: Path, episode_field: str, start_episode: int) -> List[Dict]:
    """Read existing CSV rows before `start_episode` for resume."""

    if not path.exists():
        return []
    _ensure_large_csv_field_limit()
    text = path.read_text(encoding="utf-8-sig")
    nul_count = text.count("\x00")
    if nul_count:
        print(
            "[CHECK][phase1_pair_self_labeling._read_csv_rows] "
            f"path={path} stripped_nul_bytes={nul_count}"
        )
        text = text.replace("\x00", "")
    try:
        rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    except csv.Error as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._read_csv_rows] "
            f"cause=csv_parse_failed path={path} error={exc}"
        )
        raise RuntimeError(f"failed to parse resume csv: {path}") from exc
    kept = [row for row in rows if _row_episode(row, episode_field, path) < start_episode]
    print(
        "[CHECK][phase1_pair_self_labeling._read_csv_rows] "
        f"path={path} loaded={len(rows)} kept={len(kept)} start_episode={start_episode}"
    )
    return kept


def _ensure_large_csv_field_limit() -> None:
    """Raise csv parser field size limit for resume files with large JSON fields."""

    # LINE-BY-LINE: candidate_summary.csv에는 과거 run의 긴 JSON 필드가 들어갈 수 있어 기본 131072 byte 제한을 넘습니다.
    limit = sys.maxsize
    # LINE-BY-LINE: Windows/Python 조합에 따라 sys.maxsize가 C long 범위를 넘으면 10분의 1씩 줄여 안전한 최댓값을 찾습니다.
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_jsonl_rows(path: Path, episode_field: str, start_episode: int) -> List[Dict]:
    """Read existing JSONL rows before `start_episode` for resume."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    nul_count = text.count("\x00")
    if nul_count:
        print(
            "[CHECK][phase1_pair_self_labeling._read_jsonl_rows] "
            f"path={path} stripped_nul_bytes={nul_count}"
        )
        text = text.replace("\x00", "")
    rows: List[Dict] = []
    invalid_line_count = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            invalid_line_count += 1
            print(
                "[CHECK][phase1_pair_self_labeling._read_jsonl_rows] "
                f"path={path} excluded_invalid_jsonl_line={line_number} "
                f"error={exc} preview={line[:120]!r}"
            )
    if invalid_line_count:
        print(
            "[CHECK][phase1_pair_self_labeling._read_jsonl_rows] "
            f"path={path} excluded_invalid_jsonl_lines={invalid_line_count}"
        )
    kept = [row for row in rows if _row_episode(row, episode_field, path) < start_episode]
    print(
        "[CHECK][phase1_pair_self_labeling._read_jsonl_rows] "
        f"path={path} loaded={len(rows)} kept={len(kept)} start_episode={start_episode}"
    )
    return kept


def _read_best_validation_score(path: Path) -> tuple | None:
    """Read previous best validation score when resuming."""

    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    score = summary.get("best_validation_score") or []
    if not score:
        return None
    return tuple(float(value) for value in score)


def _row_episode(row: Mapping, episode_field: str, path: Path) -> int:
    """Parse episode field from persisted row."""

    try:
        return int(row[episode_field])
    except (KeyError, TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._row_episode] "
            f"cause=invalid_episode_field field={episode_field} path={path} row={row}"
        )
        raise RuntimeError(f"invalid episode field in {path}") from exc


def _mean_score(scores: Sequence[tuple]) -> tuple:
    """Return mean objective tuple for validation checkpoint selection."""

    if not scores:
        print("[ERROR][phase1_pair_self_labeling._mean_score] cause=no_scores")
        raise RuntimeError("validation produced no scores")
    width = len(scores[0])
    return tuple(
        round(sum(float(score[index]) for score in scores) / len(scores), 12)
        for index in range(width)
    )


def _run_phase1_pair_heuristic_candidate(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    algorithm: str,
    bay_capacity_weights: Mapping[str, int | float] | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> Phase1PairCandidate:
    """Convert an existing complete heuristic assignment into pair transitions."""

    heuristic = run_phase1_heuristic_candidate(
        jobs=jobs,
        bay_ids=bay_ids,
        algorithm=algorithm,
        bay_capacity_weights=bay_capacity_weights,
        objective_scope=objective_scope,
    )
    transitions = _pair_transitions_from_assignment(
        jobs,
        bay_ids,
        heuristic.assignments,
        bay_capacity_weights=bay_capacity_weights,
    )
    return Phase1PairCandidate(
        source=algorithm,
        transitions=transitions,
        assignments=dict(heuristic.assignments),
        bay_loads=dict(heuristic.bay_loads),
    )


def _pair_transitions_from_assignment(
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    assignments: Mapping[str, str],
    bay_capacity_weights: Mapping[str, int | float] | None = None,
) -> List[Phase1PairTransition]:
    """Replay a complete assignment into one `SELECT_PAIR` row per block."""

    normalized_bay_ids = _normalize_bay_ids(bay_ids)
    cache = _build_phase1_pair_episode_cache(
        jobs=jobs,
        bay_ids=normalized_bay_ids,
        bay_capacity_weights=bay_capacity_weights,
    )
    blocks = list(cache.blocks)
    planning_state = _create_pair_planning_state(cache.bay_capacity_weights)
    bay_loads = planning_state.bay_loads
    remaining = set(cache.block_by_id)
    transitions: List[Phase1PairTransition] = []
    for block_step, block_id in enumerate(assignments):
        if block_id not in remaining:
            print(f"[ERROR][phase1_pair_self_labeling._pair_transitions_from_assignment] cause=bad_block_order block_id={block_id}")
            raise RuntimeError(f"bad heuristic block order: {block_id}")
        candidates = _build_phase1_pair_candidates_from_cache(
            cache=cache,
            bay_loads=bay_loads,
            remaining_block_ids=sorted(remaining),
        )
        selected_bay = str(assignments[block_id])
        selected_action_id = f"{block_id}@{selected_bay}"
        candidate_ids = [str(row["action_id"]) for row in candidates]
        selected_index = _candidate_index(candidate_ids, selected_action_id)
        transitions.append(
            Phase1PairTransition(
                phase="SELECT_PAIR",
                candidate_features=[row["features"] for row in candidates],
                env_features=_pair_env_features(
                    bay_loads=bay_loads,
                    assigned_block_count=block_step,
                    total_block_count=len(blocks),
                    totals=cache.env_totals,
                ),
                selected_action_index=selected_index,
                candidate_ids=candidate_ids,
                selected_action_id=selected_action_id,
                selected_block_set_id=str(block_id),
                selected_bay=selected_bay,
            )
        )
        _apply_block_to_planning_state(planning_state, selected_bay, cache.block_by_id[block_id])
        remaining.remove(block_id)
    complete_phase1_planning(planning_state, set(cache.block_by_id))
    return transitions


def _pair_env_features(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    assigned_block_count: int,
    total_block_count: int,
    totals: Mapping[str, float],
) -> List[float]:
    """Return normalized environment features before selecting a pair."""

    score = _multi_objective_load_score(bay_loads)
    return [
        assigned_block_count / total_block_count,
        (total_block_count - assigned_block_count) / total_block_count,
        float(score[0]) / totals["wo"],
        float(score[1]) / totals["wo"],
        float(score[2]) / totals["cut"],
        float(score[3]) / totals["cut"],
        float(score[4]) / totals["bevel"],
        float(score[5]) / totals["bevel"],
    ]


def _teacher_forcing_update(
    model: Phase1PairPointerPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Phase1PairTransition],
) -> float:
    """Cross-entropy update for selected pair sequence."""

    if not transitions:
        print("[ERROR][phase1_pair_self_labeling._teacher_forcing_update] cause=no_transitions")
        raise RuntimeError("best pair candidate has no transitions")
    device = next(model.parameters()).device
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    for transition in transitions:
        logits = model.score_pairs(
            torch.tensor(transition.candidate_features, dtype=torch.float32, device=device),
            torch.tensor(transition.env_features, dtype=torch.float32, device=device),
        )
        if transition.selected_action_index >= logits.numel():
            print(
                "[ERROR][phase1_pair_self_labeling._teacher_forcing_update] "
                f"cause=selected_index_out_of_range selected={transition.selected_action_index} candidate_count={logits.numel()}"
            )
            raise RuntimeError("selected pair index out of range")
        total_loss = total_loss + F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([transition.selected_action_index], dtype=torch.long, device=device),
        )
    loss = total_loss / len(transitions)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def _score_pair_candidates(
    model: Phase1PairPointerPolicy | None,
    candidate_features: List[List[float]],
    env_features: List[float],
) -> torch.Tensor:
    """Score pair candidates, or return uniform logits for random rollout."""

    if model is None:
        return torch.zeros((len(candidate_features),), dtype=torch.float32)
    device = next(model.parameters()).device
    return model.score_pairs(
        torch.tensor(candidate_features, dtype=torch.float32, device=device),
        torch.tensor(env_features, dtype=torch.float32, device=device),
    ).detach().cpu()


def _sample_index(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> int:
    """Sample one candidate index from logits."""

    if logits.numel() == 0:
        print("[ERROR][phase1_pair_self_labeling._sample_index] cause=no_logits")
        raise RuntimeError("cannot sample from empty pair logits")
    return int(torch.multinomial(torch.softmax(logits / temperature, dim=0), num_samples=1, generator=generator).item())


def _candidate_index(candidate_ids: Sequence[str], selected: str) -> int:
    """Return selected action index or fail if heuristic violates the mask."""

    try:
        return list(candidate_ids).index(selected)
    except ValueError as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._candidate_index] "
            f"cause=selected_not_in_pair_candidates selected={selected} candidates={list(candidate_ids)}"
        )
        raise RuntimeError(f"selected pair is not feasible: {selected}") from exc


def _score_bay_loads(
    bay_loads: Mapping[str, Mapping[str, int | float]],
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> tuple:
    """Score final Bay loads with the confirmed Phase 1 objective."""

    return _multi_objective_load_score(bay_loads, objective_scope)


def _candidate_learning_score(
    candidate: Phase1PairCandidate,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    phase2_feedback_scorer: Phase2FeedbackScorer | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> tuple:
    """Return the score used to choose the self-labeling teacher candidate."""

    return _candidate_score_components(
        candidate=candidate,
        jobs=jobs,
        bay_ids=bay_ids,
        phase2_feedback_scorer=phase2_feedback_scorer,
        objective_scope=objective_scope,
    )[2]


def _candidate_score_components(
    candidate: Phase1PairCandidate,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    phase2_feedback_scorer: Phase2FeedbackScorer | None = None,
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> tuple[tuple, tuple, tuple]:
    """Phase 1 score, optional Phase 2 feedback, combined teacher score를 한 번 계산한다."""

    phase1_score = _score_bay_loads(candidate.bay_loads, objective_scope)
    phase2_feedback_score = _candidate_phase2_feedback_score(
        candidate=candidate,
        jobs=jobs,
        bay_ids=bay_ids,
        phase2_feedback_scorer=phase2_feedback_scorer,
    )
    return phase1_score, phase2_feedback_score, phase2_feedback_score + phase1_score


def _candidate_phase2_feedback_score(
    candidate: Phase1PairCandidate,
    jobs: Mapping[str, object],
    bay_ids: Sequence[str],
    phase2_feedback_scorer: Phase2FeedbackScorer | None = None,
) -> tuple:
    """Return optional Phase 2 feedback score, or an empty tuple when disabled."""

    if phase2_feedback_scorer is None:
        return ()
    raw_score = phase2_feedback_scorer(candidate, jobs, bay_ids)
    if raw_score is None:
        print(
            "[ERROR][phase1_pair_self_labeling._candidate_phase2_feedback_score] "
            f"cause=none_feedback_score source={candidate.source}"
        )
        raise RuntimeError("Phase 2 feedback scorer returned None")
    try:
        return tuple(float(value) for value in raw_score)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._candidate_phase2_feedback_score] "
            f"cause=invalid_feedback_score source={candidate.source} raw_score={raw_score}"
        )
        raise RuntimeError("Phase 2 feedback score must be a numeric sequence") from exc


def _apply_block_to_planning_state(
    state: HierarchicalPlanningState,
    bay_id: str,
    block: Phase1Block,
) -> None:
    """기존 Phase 1 block aggregate를 공통 planning transition에 전달한다."""

    apply_phase1_block_assignment(
        state,
        block_set_id=block.block_set_id,
        bay_id=bay_id,
        steel_quantity_sum=block.steel_quantity_sum,
        cut_length_sum=block.cut_length_sum,
        bevel_quantity_sum=block.bevel_quantity_sum,
        wo_count=block.wo_count,
        long_cut_over_1000=block.long_cut_over_1000,
        allow_zero_steel_quantity=bool(block.balancing_group),
    )
    if block.balancing_group:
        add_multi_series_group_load(
            state.bay_loads[str(bay_id)],
            group=block.balancing_group,
            wo_count=block.wo_count,
            cut_length_sum=block.cut_length_sum,
            bevel_quantity_sum=block.bevel_quantity_sum,
        )


def _create_pair_planning_state(
    bay_capacity_weights: Mapping[str, int | float],
) -> HierarchicalPlanningState:
    """MIXED 그룹별 누적 부하를 포함한 planning state를 생성한다."""

    state = create_phase1_planning_state(bay_capacity_weights)
    initialize_multi_series_group_loads(state.bay_loads)
    return state


def _plain_bay_loads(bay_loads: Mapping[str, Mapping[str, int | float]]) -> Dict[str, Dict[str, int | float]]:
    """그룹별 누적 부하를 포함한 JSON-safe Bay load mapping을 반환한다."""

    return {str(bay_id): dict(loads) for bay_id, loads in bay_loads.items()}


def _problem_totals(
    blocks: Sequence[Phase1Block],
    bay_capacity_weights: Mapping[str, int | float],
) -> Dict[str, float]:
    """Return non-zero denominators for normalized pair features."""

    if not blocks:
        print("[ERROR][phase1_pair_self_labeling._problem_totals] cause=no_blocks")
        raise RuntimeError("Phase 1 pair MDP requires at least one block")
    capacity_sum = sum(float(value) for value in bay_capacity_weights.values())
    if capacity_sum <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._problem_totals] "
            f"cause=non_positive_capacity_sum value={capacity_sum}"
        )
        raise RuntimeError("Phase 1 pair MDP requires positive Bay capacity")
    present_groups = {block.balancing_group for block in blocks if block.balancing_group}
    group_metric_totals = {"wo": 0.0, "cut": 0.0, "bevel": 0.0}
    for group in present_groups:
        group_capacity_sum = sum(GROUP_BAY_CAPACITY_WEIGHTS[group].values())
        group_blocks = [block for block in blocks if block.balancing_group == group]
        group_metric_totals["wo"] += sum(block.wo_count for block in group_blocks) / group_capacity_sum
        group_metric_totals["cut"] += sum(block.cut_length_sum for block in group_blocks) / group_capacity_sum
        group_metric_totals["bevel"] += sum(block.bevel_quantity_sum for block in group_blocks) / group_capacity_sum
    return {
        "wo": max(1.0, group_metric_totals["wo"]),
        "cut": max(1.0, group_metric_totals["cut"]),
        "bevel": max(1.0, group_metric_totals["bevel"]),
        "block": max(1.0, float(len(blocks))),
    }


def _validate_train_inputs(
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    episodes: int,
    rollout_samples: int,
    validation_rollout_samples: int,
    lr: float,
    temperature: float,
    episode_factory: Callable[[int], Mapping[str, object]] | None,
    checkpoint_every: int,
    validation_every: int,
    validation_episodes: int,
    validation_episode_factory: Callable[[int], Mapping[str, object]] | None,
    phase2_feedback_scorer: Phase2FeedbackScorer | None,
    phase2_feedback_contract: Mapping[str, object] | None,
) -> None:
    """Validate pair self-labeling train controls."""

    if episodes <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=non_positive_episodes episodes={episodes}")
        raise RuntimeError("episodes must be positive")
    if episode_factory is None and (
        episode_jobs is None
        or episode_metadata is None
        or len(episode_jobs) != episodes
        or len(episode_metadata) != episodes
    ):
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=episode_count_mismatch episodes={episodes} "
            f"jobs={0 if episode_jobs is None else len(episode_jobs)} "
            f"metadata={0 if episode_metadata is None else len(episode_metadata)}"
        )
        raise RuntimeError("pair self-labeling requires one job mapping and metadata row per episode")
    if rollout_samples <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=non_positive_rollout_samples rollout_samples={rollout_samples}")
        raise RuntimeError("rollout_samples must be positive")
    if validation_rollout_samples <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=non_positive_validation_rollout_samples validation_rollout_samples={validation_rollout_samples}"
        )
        raise RuntimeError("validation_rollout_samples must be positive")
    if lr <= 0 or temperature <= 0:
        print(f"[ERROR][phase1_pair_self_labeling._validate_train_inputs] cause=invalid_lr_or_temperature lr={lr} temperature={temperature}")
        raise RuntimeError("lr and temperature must be positive")
    if checkpoint_every <= 0 or validation_every <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=invalid_checkpoint_or_validation_interval checkpoint_every={checkpoint_every} validation_every={validation_every}"
        )
        raise RuntimeError("checkpoint_every and validation_every must be positive")
    if validation_episode_factory is not None and validation_episodes <= 0:
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=non_positive_validation_episodes validation_episodes={validation_episodes}"
        )
        raise RuntimeError("validation_episodes must be positive when validation is enabled")
    if phase2_feedback_scorer is not None and not callable(phase2_feedback_scorer):
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=non_callable_phase2_feedback_scorer type={type(phase2_feedback_scorer).__name__}"
        )
        raise RuntimeError("phase2_feedback_scorer must be callable")
    if (phase2_feedback_scorer is None) != (phase2_feedback_contract is None):
        print(
            "[ERROR][phase1_pair_self_labeling._validate_train_inputs] "
            f"cause=feedback_scorer_contract_mismatch scorer={phase2_feedback_scorer is not None} "
            f"contract={phase2_feedback_contract is not None}"
        )
        raise RuntimeError("Phase 2 feedback scorer and contract must be provided together")


def _normalize_phase2_feedback_contract(
    contract: Mapping[str, object] | None,
) -> Dict[str, object] | None:
    """Phase 1 checkpoint에 저장할 frozen Phase 2 정체성을 엄격히 검증한다."""

    if contract is None:
        return None
    required = {"checkpoint", "checkpoint_sha256", "run_spec"}
    if not isinstance(contract, Mapping) or set(contract) != required:
        print(
            "[ERROR][phase1_pair_self_labeling._normalize_phase2_feedback_contract] "
            f"cause=field_mismatch fields={sorted(contract) if isinstance(contract, Mapping) else []}"
        )
        raise RuntimeError("Phase 2 feedback contract fields are invalid")
    checkpoint = str(contract["checkpoint"]).strip()
    digest = str(contract["checkpoint_sha256"]).strip().lower()
    run_spec = contract["run_spec"]
    if not checkpoint:
        print(
            "[ERROR][phase1_pair_self_labeling._normalize_phase2_feedback_contract] "
            "cause=empty_checkpoint"
        )
        raise RuntimeError("Phase 2 feedback checkpoint path is empty")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        print(
            "[ERROR][phase1_pair_self_labeling._normalize_phase2_feedback_contract] "
            f"cause=invalid_sha256 value={digest}"
        )
        raise RuntimeError("Phase 2 feedback checkpoint SHA256 is invalid")
    if not isinstance(run_spec, Mapping) or not run_spec:
        print(
            "[ERROR][phase1_pair_self_labeling._normalize_phase2_feedback_contract] "
            "cause=invalid_run_spec"
        )
        raise RuntimeError("Phase 2 feedback RunSpec is invalid")
    try:
        normalized_run_spec = json.loads(json.dumps(dict(run_spec), ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][phase1_pair_self_labeling._normalize_phase2_feedback_contract] "
            f"cause=non_json_run_spec error={exc}"
        )
        raise RuntimeError("Phase 2 feedback RunSpec must be JSON serializable") from exc
    return {
        "checkpoint": checkpoint,
        "checkpoint_sha256": digest,
        "run_spec": normalized_run_spec,
    }


def _episode_payload(
    episode: int,
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_metadata: Sequence[Mapping[str, object]] | None,
    episode_factory: Callable[[int], Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], Dict[str, object]]:
    """Return one training episode, generating it on demand when configured."""

    if episode_factory is not None:
        payload = episode_factory(episode)
        if not isinstance(payload, Mapping):
            print(
                "[ERROR][phase1_pair_self_labeling._episode_payload] "
                f"cause=invalid_factory_payload episode={episode} type={type(payload).__name__}"
            )
            raise RuntimeError("episode_factory must return a mapping")
        jobs = payload.get("jobs")
        metadata = payload.get("metadata")
    else:
        if episode_jobs is None or episode_metadata is None:
            print(
                "[ERROR][phase1_pair_self_labeling._episode_payload] "
                f"cause=missing_prebuilt_episode episode={episode}"
            )
            raise RuntimeError("prebuilt episode inputs are missing")
        jobs = episode_jobs[episode - 1]
        metadata = episode_metadata[episode - 1]
    if not isinstance(jobs, Mapping) or not jobs:
        print(
            "[ERROR][phase1_pair_self_labeling._episode_payload] "
            f"cause=invalid_jobs episode={episode} type={type(jobs).__name__}"
        )
        raise RuntimeError(f"invalid episode jobs: {episode}")
    if not isinstance(metadata, Mapping):
        print(
            "[ERROR][phase1_pair_self_labeling._episode_payload] "
            f"cause=invalid_metadata episode={episode} type={type(metadata).__name__}"
        )
        raise RuntimeError(f"invalid episode metadata: {episode}")
    return jobs, dict(metadata)


def _resolve_phase1_episode_scope(
    metadata: Mapping[str, object],
    default_bay_ids: Sequence[str],
    default_capacity_weights: Mapping[str, int | float],
) -> tuple[Tuple[str, ...], Dict[str, float]]:
    """분할 전 부모 episode가 모든 W/O와 확정 5-Bay 계약을 유지하는지 검증한다."""

    default_ids = _normalize_bay_ids(default_bay_ids)
    default_weights = _normalize_bay_capacity_weights(default_ids, default_capacity_weights)
    raw_groups = metadata.get("balancing_groups")
    raw_bay_ids = metadata.get("bay_ids")
    raw_capacity_weights = metadata.get("bay_capacity_weights")
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_phase1_episode_scope] "
            f"cause=missing_balancing_groups balancing_groups={raw_groups}"
        )
        raise RuntimeError("multi-series Phase 1 metadata requires balancing_groups")
    groups = tuple(str(group).strip().upper() for group in raw_groups)
    expected_group_order = tuple(
        group for group in PHASE1_BALANCING_GROUP_ORDER if group in set(groups)
    )
    if not groups or groups != expected_group_order:
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_phase1_episode_scope] "
            f"cause=invalid_balancing_groups groups={groups} expected_order={expected_group_order}"
        )
        raise RuntimeError("invalid multi-series Phase 1 balancing_groups")
    if not isinstance(raw_bay_ids, Sequence) or isinstance(raw_bay_ids, (str, bytes)):
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_phase1_episode_scope] "
            f"cause=missing_joint_bay_ids bay_ids={raw_bay_ids}"
        )
        raise RuntimeError("multi-series Phase 1 metadata requires joint bay_ids")
    if not isinstance(raw_capacity_weights, Mapping):
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_phase1_episode_scope] "
            "cause=missing_joint_capacity_weights"
        )
        raise RuntimeError("multi-series Phase 1 metadata requires bay_capacity_weights")
    expected_weights = joint_phase1_bay_capacity_weights()
    episode_ids = _normalize_bay_ids(raw_bay_ids)
    episode_weights = _normalize_bay_capacity_weights(episode_ids, raw_capacity_weights)
    if (
        default_ids != tuple(expected_weights)
        or default_weights != expected_weights
        or episode_ids != tuple(expected_weights)
        or episode_weights != expected_weights
    ):
        print(
            "[ERROR][phase1_pair_self_labeling._resolve_phase1_episode_scope] "
            "cause=joint_scope_mismatch "
            f"expected_bays={tuple(expected_weights)} actual_bays={episode_ids} "
            f"expected_weights={expected_weights} actual_weights={episode_weights} "
            f"default_bays={default_ids} default_weights={default_weights}"
        )
        raise RuntimeError("multi-series Phase 1 requires the joint five-Bay scope")
    return episode_ids, episode_weights


def _candidate_summary_row(
    episode: int,
    problem_id: str,
    block_count: int,
    problem_seed: int | str,
    subproblem_id: str,
    candidate_index: int,
    candidate: Phase1PairCandidate,
    best: Phase1PairCandidate,
    score: tuple,
    phase2_feedback_score: tuple,
    objective_scope: str,
) -> Dict:
    """Return one candidate audit row."""

    learning_score = phase2_feedback_score + score
    row = {
        "episode": episode,
        "problem_id": problem_id,
        "block_count": block_count,
        "problem_seed": problem_seed,
        "subproblem_id": subproblem_id,
        "candidate_index": candidate_index,
        "source": candidate.source,
        "is_best": int(candidate is best),
        "score_mode": "wo_first",
        "objective_scope": objective_scope,
        "score_json": json.dumps(list(score), ensure_ascii=False),
        "phase2_feedback_score_json": json.dumps(list(phase2_feedback_score), ensure_ascii=False),
        "learning_score_json": json.dumps(list(learning_score), ensure_ascii=False),
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "bay_loads_json": json.dumps(candidate.bay_loads, ensure_ascii=False, sort_keys=True),
    }
    row.update(_score_columns(score))
    return row


def _validation_candidate_summary_row(
    train_episode: int,
    validation_episode: int,
    validation_source: str,
    evaluation_input_type: str,
    parent_problem_id: str,
    problem_id: str,
    subproblem_id: str,
    validation_view: str,
    block_count: int,
    candidate_index: int,
    candidate: Phase1PairCandidate,
    best: Phase1PairCandidate,
    rank: int | str,
    score: tuple,
    phase2_feedback_score: tuple | None,
    objective_scope: str,
    gap_breakdown: Mapping[str, float | str],
) -> Dict:
    """Return one validation candidate audit row."""

    if phase2_feedback_score is None:
        print(
            "[ERROR][phase1_pair_self_labeling._validation_candidate_summary_row] "
            f"cause=missing_cached_feedback_score source={candidate.source}"
        )
        raise RuntimeError("validation candidate is missing its cached feedback score")
    learning_score = phase2_feedback_score + score
    row = {
        "train_episode": train_episode,
        "validation_episode": validation_episode,
        "validation_source": validation_source,
        "evaluation_input_type": evaluation_input_type,
        "parent_problem_id": parent_problem_id,
        "problem_id": problem_id,
        "subproblem_id": subproblem_id,
        "validation_view": validation_view,
        "block_count": block_count,
        "candidate_index": candidate_index,
        "source": candidate.source,
        "rank": rank,
        "is_best": int(candidate is best),
        "score_mode": "wo_first",
        "objective_scope": objective_scope,
        "score_json": json.dumps(list(score), ensure_ascii=False),
        "phase2_feedback_score_json": json.dumps(list(phase2_feedback_score), ensure_ascii=False),
        "learning_score_json": json.dumps(list(learning_score), ensure_ascii=False),
        "assignment_count": len(candidate.assignments),
        "transition_count": len(candidate.transitions),
        "bay_loads_json": json.dumps(candidate.bay_loads, ensure_ascii=False, sort_keys=True),
    }
    row.update(_score_columns(score))
    row.update(gap_breakdown)
    return row


def _validation_input_type(metadata: Mapping[str, object], validation_source: str) -> str:
    """Return explicit validation input type for audit CSVs."""

    input_type = str(metadata.get("evaluation_input_type") or "")
    if input_type:
        return input_type
    if validation_source == "synthetic":
        return "synthetic_generated"
    print(
        "[ERROR][phase1_pair_self_labeling._validation_input_type] "
        f"cause=missing_evaluation_input_type validation_source={validation_source} "
        f"metadata_keys={sorted(metadata.keys())}"
    )
    raise RuntimeError("evaluation_input_type is required for non-synthetic validation")


def _score_columns(score: Sequence[int | float]) -> Dict[str, int | float]:
    """Return stable score_N columns for the full objective tuple."""

    return {
        field: score[index] if index < len(score) else ""
        for index, field in enumerate(PHASE1_SCORE_FIELD_NAMES)
    }


def _best_action_rows(
    episode: int,
    problem_id: str,
    block_count: int,
    problem_seed: int | str,
    subproblem_id: str,
    best: Phase1PairCandidate,
    objective_scope: str,
) -> List[Dict]:
    """Return JSONL rows for selected pair pseudo-label sequence."""

    return [
        {
            "episode": episode,
            "problem_id": problem_id,
            "block_count": block_count,
            "problem_seed": problem_seed,
            "subproblem_id": subproblem_id,
            "step": step,
            "source": best.source,
            "objective_scope": objective_scope,
            "phase": transition.phase,
            "selected_action_index": transition.selected_action_index,
            "selected_action_id": transition.selected_action_id,
            "selected_block_set_id": transition.selected_block_set_id,
            "selected_bay": transition.selected_bay,
            "candidate_ids": transition.candidate_ids,
            "candidate_features": transition.candidate_features,
            "env_features": transition.env_features,
        }
        for step, transition in enumerate(best.transitions)
    ]


def _source_counts(rows: Sequence[Mapping]) -> Dict[str, int]:
    """Count best candidate sources."""

    counts: Dict[str, int] = {}
    for row in rows:
        source = str(row["best_source"])
        counts[source] = counts.get(source, 0) + 1
    return counts


def _write_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write per-episode train metrics."""

    _write_csv(
        path,
        [
            "episode",
            "problem_id",
            "block_count",
            "problem_seed",
            "case_type",
            "hard_case_mode",
            "hard_case_corr_steel_cut_before",
            "hard_case_corr_steel_cut_after",
            "best_source",
            "score_mode",
            "objective_scope",
            "loss",
            "score_json",
            "phase2_feedback_score_json",
            "learning_score_json",
            "candidate_count",
            "subproblem_count",
        ],
        rows,
    )


def _write_subproblem_metrics(path: Path, rows: Sequence[Mapping]) -> None:
    """Write one row for every independent resource-pool optimizer update."""

    _write_csv(
        path,
        [
            "episode",
            "problem_id",
            "subproblem_id",
            "series",
            "block_count",
            "job_count",
            "best_source",
            "objective_scope",
            "loss",
            "score_json",
            "phase2_feedback_score_json",
            "learning_score_json",
            "candidate_count",
        ],
        rows,
    )


def _write_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all complete candidates for audit."""

    _write_csv(
        path,
        [
            "episode",
            "problem_id",
            "block_count",
            "problem_seed",
            "subproblem_id",
            "candidate_index",
            "source",
            "is_best",
            "score_mode",
            "objective_scope",
            "score_json",
            "phase2_feedback_score_json",
            "learning_score_json",
            *PHASE1_SCORE_FIELD_NAMES,
            "assignment_count",
            "transition_count",
            "bay_loads_json",
        ],
        rows,
    )


def _write_validation_candidate_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write all method candidates evaluated during validation."""

    _write_csv(
        path,
        [
            "train_episode",
            "validation_episode",
            "validation_source",
            "evaluation_input_type",
            "parent_problem_id",
            "problem_id",
            "subproblem_id",
            "validation_view",
            "block_count",
            "candidate_index",
            "source",
            "rank",
            "is_best",
            "score_mode",
            "objective_scope",
            "score_json",
            "phase2_feedback_score_json",
            "learning_score_json",
            *PHASE1_SCORE_FIELD_NAMES,
            *PHASE1_VALIDATION_GAP_FIELDS,
            "assignment_count",
            "transition_count",
            "bay_loads_json",
        ],
        rows,
    )


def _write_validation_plots(
    output_path: Path,
    candidate_rows: Sequence[Mapping],
    summary_rows: Sequence[Mapping],
    objective_scope: str,
) -> Dict[str, str]:
    """Write validation comparison plots for the latest validation checkpoint."""

    if not candidate_rows:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][phase1_pair_self_labeling._write_validation_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to write Phase 1 pair validation plots") from exc

    latest_candidate_rows = _latest_rows(candidate_rows, "train_episode")
    latest_summary_rows = _latest_rows(summary_rows, "train_episode")
    latest_candidate_rows = _collapse_agent_samples_for_validation_plot(latest_candidate_rows)
    plot_paths: Dict[str, str] = {}
    for output_key, filename, score_field, title, ylabel in _validation_plot_specs(
        objective_scope
    ):
        path = output_path / filename
        _plot_validation_metric(
            plt=plt,
            path=path,
            rows=latest_candidate_rows,
            score_field=score_field,
            title=title,
            ylabel=ylabel,
        )
        plot_paths[output_key] = str(path)
    for gap_field in PHASE1_VALIDATION_GAP_FIELDS:
        metric_rows = [
            row
            for row in latest_candidate_rows
            if row.get(gap_field) not in (None, "")
        ]
        if not metric_rows:
            continue
        output_key = f"validation_{gap_field}_png"
        path = output_path / f"validation_{gap_field}.png"
        _plot_validation_metric(
            plt=plt,
            path=path,
            rows=metric_rows,
            score_field=gap_field,
            title=gap_field.replace("_", " ").upper(),
            ylabel="Gap",
        )
        plot_paths[output_key] = str(path)
    best_source_path = output_path / "validation_best_source_counts.png"
    _plot_validation_best_source_counts(plt, best_source_path, latest_summary_rows)
    plot_paths["validation_best_source_counts_png"] = str(best_source_path)
    agent_rank_path = output_path / "validation_agent_rank.png"
    _plot_validation_agent_rank(plt, agent_rank_path, latest_summary_rows)
    plot_paths["validation_agent_rank_png"] = str(agent_rank_path)
    return plot_paths


def _validation_plot_specs(
    objective_scope: str = PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES,
) -> List[tuple[str, str, str, str, str]]:
    """Return score-column mapping for validation plots."""

    normalized_scope = normalize_phase1_objective_scope(objective_scope)
    if normalized_scope == PHASE1_OBJECTIVE_SCOPE_SHARED_AND_SERIES:
        return [
            ("validation_wo_gap_png", "validation_wo_gap.png", "score_0", "Shared-pool W/O load gap", "Gap"),
            ("validation_series_wo_gap_png", "validation_series_wo_gap.png", "score_1", "Series-group W/O load gap", "Gap"),
            ("validation_cut_gap_png", "validation_cut_gap.png", "score_2", "Shared-pool cut length gap", "Gap"),
            ("validation_series_cut_gap_png", "validation_series_cut_gap.png", "score_3", "Series-group cut length gap", "Gap"),
            ("validation_bevel_gap_png", "validation_bevel_gap.png", "score_4", "Shared-pool bevel quantity gap", "Gap"),
            ("validation_series_bevel_gap_png", "validation_series_bevel_gap.png", "score_5", "Series-group bevel quantity gap", "Gap"),
        ]
    return [
        ("validation_series_wo_gap_png", "validation_series_wo_gap.png", "score_0", "Series-group W/O load gap", "Gap"),
        ("validation_series_cut_gap_png", "validation_series_cut_gap.png", "score_1", "Series-group cut length gap", "Gap"),
        ("validation_series_bevel_gap_png", "validation_series_bevel_gap.png", "score_2", "Series-group bevel quantity gap", "Gap"),
    ]


def _latest_rows(rows: Sequence[Mapping], episode_field: str) -> List[Mapping]:
    """Return rows from the latest train episode."""

    if not rows:
        return []
    latest_episode = max(int(row[episode_field]) for row in rows)
    return [row for row in rows if int(row[episode_field]) == latest_episode]


def _plot_validation_metric(plt, path: Path, rows: Sequence[Mapping], score_field: str, title: str, ylabel: str) -> None:
    """Scatter one validation metric by problem and method."""

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["validation_source"]),
            int(row["validation_episode"]),
            str(row["problem_id"]),
            str(row["source"]),
        ),
    )
    problem_keys = []
    for row in ordered_rows:
        key = (str(row["validation_source"]), str(row["problem_id"]))
        if key not in problem_keys:
            problem_keys.append(key)
    problem_index = {key: index + 1 for index, key in enumerate(problem_keys)}
    sources = _ordered_sources(row["source"] for row in ordered_rows)
    styles = _validation_plot_styles()
    offset_step = 0.48 / max(len(sources), 1)
    plt.figure(figsize=(11, 5))
    for source_index, source in enumerate(sources):
        source_rows = [row for row in ordered_rows if str(row["source"]) == source]
        x_offset = (source_index - (len(sources) - 1) / 2) * offset_step
        x_values = [
            problem_index[(str(row["validation_source"]), str(row["problem_id"]))] + x_offset
            for row in source_rows
        ]
        y_values = [float(row[score_field]) for row in source_rows]
        marker, color = styles.get(source, ("o", "#333333"))
        plt.scatter(
            x_values,
            y_values,
            label=_display_source_name(source),
            marker=marker,
            color=color,
            edgecolors="#222222",
            linewidths=0.35,
            s=42,
            alpha=1.0,
        )
    plt.xlabel("Validation problem")
    plt.ylabel(ylabel)
    plt.title(f"{title} by method (lower is better)")
    plt.xticks(
        list(problem_index.values()),
        [key[1] for key in problem_keys],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_best_source_counts(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot which source wins each latest validation problem."""

    counts: Dict[str, int] = {}
    for row in rows:
        source = _plot_source_key(str(row["best_source"]))
        counts[source] = counts.get(source, 0) + 1
    sources = _ordered_sources(counts)
    plt.figure(figsize=(9, 4))
    plt.bar([_display_source_name(source) for source in sources], [counts[source] for source in sources])
    plt.xlabel("Best source")
    plt.ylabel("Validation wins")
    plt.title("Best source counts in latest validation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_agent_rank(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot agent rank in each latest validation problem."""

    ordered_rows = sorted(
        rows,
        key=lambda row: (str(row["validation_source"]), int(row["validation_episode"]), str(row["problem_id"])),
    )
    x_values = list(range(1, len(ordered_rows) + 1))
    y_values = [int(row["agent_rank"]) for row in ordered_rows]
    labels = [str(row["problem_id"]) for row in ordered_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel("Validation problem")
    plt.ylabel("Proposed rank")
    plt.title("Proposed(best-of-K) rank in latest validation (1 is best)")
    plt.xticks(x_values, labels, rotation=45, ha="right", fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _ordered_sources(sources: Sequence[object]) -> List[str]:
    """Return stable method order with the learned policy first."""

    unique = {str(source) for source in sources}
    preferred = list(
        dict.fromkeys(
            [
                PHASE1_PROPOSED_BEST_OF_K_SOURCE,
                "agent_greedy",
                *PHASE1_HEURISTIC_BANK,
            ]
        )
    )
    return [source for source in preferred if source in unique] + sorted(unique - set(preferred))


def _display_source_name(source: str) -> str:
    """Shorten source names for plot legends."""

    names = {
        PHASE1_PROPOSED_BEST_OF_K_SOURCE: "Proposed(best-of-K)",
        "agent_greedy": "Proposed(greedy)",
        "wo_first_balanced": "W/O",
        "cut_first_balanced": "Cut",
        "bevel_first_balanced": "Bevel",
    }
    return names.get(source, source)


def _validation_plot_styles() -> Dict[str, tuple[str, str]]:
    """Return fixed marker/color mapping so legend and scatter points match exactly."""

    return {
        PHASE1_PROPOSED_BEST_OF_K_SOURCE: ("D", "#1f77b4"),
        "agent_greedy": ("D", "#1f77b4"),
        "wo_first_balanced": ("s", "#ff7f0e"),
        "cut_first_balanced": ("^", "#2ca02c"),
        "bevel_first_balanced": ("o", "#d62728"),
    }


def _plot_source_key(source: str) -> str:
    """Return the visible plot source key for raw policy candidates."""

    if _is_agent_source(source):
        return PHASE1_PROPOSED_BEST_OF_K_SOURCE
    return source


def _collapse_agent_samples_for_validation_plot(rows: Sequence[Mapping]) -> List[Mapping]:
    """Collapse raw agent validation rows into one best-of-K row per problem for plots only."""

    non_agent_rows: List[Mapping] = []
    best_agent_rows: Dict[tuple[str, str, str, str], Mapping] = {}
    for row in rows:
        source = str(row["source"])
        if not _is_agent_source(source):
            non_agent_rows.append(row)
            continue
        key = (
            str(row.get("validation_source", "")),
            str(row.get("evaluation_input_type", "")),
            str(row.get("validation_episode", "")),
            str(row.get("problem_id", "")),
        )
        current_best = best_agent_rows.get(key)
        if current_best is None or _phase1_plot_score_key(row) < _phase1_plot_score_key(current_best):
            best_agent_rows[key] = row

    collapsed_agent_rows: List[Mapping] = []
    for row in best_agent_rows.values():
        collapsed = dict(row)
        collapsed["source"] = PHASE1_PROPOSED_BEST_OF_K_SOURCE
        collapsed_agent_rows.append(collapsed)
    return non_agent_rows + collapsed_agent_rows


def _phase1_plot_score_key(row: Mapping) -> tuple[float, ...]:
    """Parse the learning score used for selecting the visible best-of-K agent point."""

    value = row.get("learning_score_json") or row.get("score_json")
    if value in (None, ""):
        print("[ERROR][phase1_pair_self_labeling._phase1_plot_score_key] cause=missing_score_json")
        raise RuntimeError("Phase 1 validation plot row has no score_json")
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        print(f"[ERROR][phase1_pair_self_labeling._phase1_plot_score_key] cause=invalid_score_json value={value}")
        raise RuntimeError("invalid Phase 1 validation plot score_json") from exc
    return tuple(float(item) for item in parsed)


def _write_validation_summary(path: Path, rows: Sequence[Mapping]) -> None:
    """Write periodic holdout validation summary."""

    _write_csv(
        path,
        [
            "train_episode",
            "validation_episode",
            "validation_source",
            "evaluation_input_type",
            "parent_problem_id",
            "problem_id",
            "subproblem_id",
            "validation_view",
            "block_count",
            "case_type",
            "hard_case_mode",
            "hard_case_corr_steel_cut_before",
            "hard_case_corr_steel_cut_after",
            "agent_score_json",
            "agent_learning_score_json",
            "best_score_json",
            "best_learning_score_json",
            "best_source",
            "agent_best_source",
            "agent_is_best",
            "agent_rank",
            "candidate_count",
            "objective_scope",
            *PHASE1_VALIDATION_GAP_FIELDS,
        ],
        rows,
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping]) -> None:
    """Write CSV rows with explicit fields."""

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    """Write JSONL rows."""

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
