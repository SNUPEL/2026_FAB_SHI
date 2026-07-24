"""Merged Phase 2: machine 선택 뒤 W/O를 순차 선택해 batch를 구성한다.

기존 분리 구조는 먼저 모든 W/O를 machine에 배정한 뒤 machine별 batch를
만들었다. 이 모듈은 중간 고정 배정을 없애되, 조합 폭발을 피하기 위해
환경이 설비를 결정하고 정책이 `SELECT_WO... -> 자동 batch close`로 schedule을 만든다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from Environment.metrics import (
    PHASE2_NORMALIZED_SCORE_FIELD_NAMES,
    PHASE2_RAW_SCORE_FIELD_NAMES,
    calculate_phase2_schedule_metrics,
)
from Environment.hierarchical import CommonHierarchicalEnvironment
from Environment.data import Machine
from Environment.constraints.profiles import (
    PhaseConstraintProfile,
    audit_phase2_schedule_constraints,
    default_phase2_constraint_profile,
    evaluate_phase2_action_constraints,
)
from Phase1.heuristics import run_phase1_resource_pool_heuristic_candidate
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase2.run_spec import (
    build_phase2_run_spec,
    require_matching_phase2_run_spec,
)
from Phase2.state import (
    PHASE2_SET_POINTER_POLICY_TYPE,
    PHASE2_STATE_SCHEMA_VERSION,
    Phase2PolicyState,
    build_phase2_policy_state,
    phase2_state_feature_schema,
)
from Utils.data.multi_series_cutting_data import MIXED_PLANNING_MACHINE_IDS_BY_BAY
from Utils.learning.phase_graph_mdp import _processing_time, _required_non_negative, build_phase2_wo_machine_graph


PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES = list(PHASE2_RAW_SCORE_FIELD_NAMES)
PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES = list(PHASE2_NORMALIZED_SCORE_FIELD_NAMES)
PHASE2_BATCH_MACHINE_SCORE_MODES = ("raw", "normalized")
PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES = [
    "raw_hard_violation_count",
    "raw_makespan",
    "raw_bay_internal_cut_length_gap",
    "raw_bay_internal_wo_count_gap",
    "raw_bay_internal_bevel_quantity_gap",
    "raw_bay_internal_occupancy_gap",
    "normalized_bay_internal_cut_length_gap",
    "normalized_bay_internal_wo_count_gap",
    "normalized_bay_internal_bevel_quantity_gap",
    "normalized_bay_internal_occupancy_gap",
]

PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK = (
    "workload_makespan_dispatch",
    "min_makespan",
    "lookahead_min_makespan",
    "best_fit_lth",
    "spt_batch",
    "lpt_batch",
)
PHASE2_PROPOSED_BEST_OF_K_SOURCE = "proposed_best_of_k"

MIXED_PHASE2_ELIGIBLE_FAMILIES = {
    "22": ("NP", "NC"),
    "23": ("NP", "NC"),
    "24": ("NP", "NC"),
    "25": ("FN", "FL"),
    "trans": ("FN", "FL"),
}


def build_mixed_phase2_training_machines() -> Dict[str, Machine]:
    """확정 EQP 매핑의 PLS/PLP 15대로 MIXED planning 설비를 만든다.

    EQP_3은 과거 NC/trans 실적 전용이므로 planning machine에는 포함하지 않는다.
    현재 미확정인 두께/정반 제약은 Phase 2 profile에서 비활성 상태이며 이
    함수는 임의 제약을 추가하지 않는다.
    """

    machines: Dict[str, Machine] = {}
    for bay_id, machine_ids in MIXED_PLANNING_MACHINE_IDS_BY_BAY.items():
        families = MIXED_PHASE2_ELIGIBLE_FAMILIES[bay_id]
        for machine_id in machine_ids:
            equipment_type = machine_id[:3]
            machines[machine_id] = Machine(
                machine_id=machine_id,
                machine_type="plasma",
                enabled=True,
                eligible_families=families,
                min_thickness=0.0,
                max_thickness=1_000.0,
                table_length_limit=55_000.0,
                cut_speed_factor=1.0,
                daily_capacity_minutes=1_000_000_000.0,
                parallel_capacity=1,
                bay_id=bay_id,
                equipment_type=equipment_type,
                max_batch_wo_count=3,
                max_batch_length_sum=55_000.0,
                priority_tiers_by_family={family: 0 for family in families},
            )
    expected_count = sum(len(machine_ids) for machine_ids in MIXED_PLANNING_MACHINE_IDS_BY_BAY.values())
    if len(machines) != expected_count:
        print(
            "[ERROR][Phase2.merged.build_mixed_phase2_training_machines] "
            f"cause=machine_count_mismatch expected={expected_count} actual={len(machines)}"
        )
        raise RuntimeError("MIXED Phase 2 mapped machine count mismatch")
    return machines


def phase2_score_field_names(score_mode: str) -> list[str]:
    """score_mode에 대응하는 사전식 score 필드 목록의 정본이다.

    학습(`Phase2.merged`)과 full-flow RunSpec(`main.py`)이 같은 매핑을 써야
    checkpoint RunSpec 대조가 성립한다.
    """

    mode = str(score_mode or "").strip().lower()
    if mode == "raw":
        return list(PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES)
    if mode == "normalized":
        return list(PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES)
    print(f"[ERROR][Phase2.merged.phase2_score_field_names] cause=unknown_score_mode value={score_mode}")
    raise RuntimeError(f"unknown Phase 2 score mode: {score_mode}")


# 기존 내부 호출부 호환용 별칭.
_score_field_names = phase2_score_field_names


@dataclass(frozen=True)
class Phase2BatchMachineTransition:
    """한 step의 CE 학습 target: 후보 action들 중 선택된 sequential action index."""

    policy_state: Phase2PolicyState
    selected_action_index: int
    selected_machine_id: str
    selected_job_ids: tuple[str, ...]
    action_type: str = ""
    target_batch_size: int = 0


@dataclass(frozen=True)
class Phase2BatchMachineCandidate:
    """한 episode 전체의 batch-machine schedule 후보."""

    source: str
    transitions: List[Phase2BatchMachineTransition]
    machine_assignments: Dict[str, str]
    machine_loads: Dict[str, Dict[str, int | float]]
    machine_bay_ids: Dict[str, str]
    batches: List[Dict]
    timeline: List[Dict]
    event_log: List[Dict]
    constraint_audit_rows: List[Dict]
    score_tuple: tuple
    score_details: Dict[str, int | float]
    subproblem_bay_id: str = ""


@dataclass(frozen=True)
class _Phase2DispatchCache:
    """Phase 2 후보 1개를 생성하는 동안 변하지 않는 값만 캐시한다."""

    machine_bay_ids: Dict[str, str]
    static_feasible_by_machine: Dict[str, tuple[str, ...]]
    processing_time_by_job: Dict[str, float]
    plate_length_by_job: Dict[str, float]
    cut_length_by_job: Dict[str, float]
    bevel_quantity_by_job: Dict[str, float]


def train_phase2_batch_machine_self_labeling(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    output_dir: str | Path,
    episodes: int = 10,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    seed: int = 0,
    heuristic_algorithms: Sequence[str] = PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK,
    rollout_samples: int = 1,
    max_wo_count: int = 3,
    max_length_sum: float = 55000.0,
    action_pool_limit: int | None = None,
    episode_jobs: Sequence[Mapping[str, object]] | None = None,
    episode_job_factory: Callable[[int], Mapping[str, object]] | None = None,
    phase1_heuristic: str | None = None,
    phase1_bay_ids: Sequence[str] | None = None,
    phase1_assignment_builder: Callable[[Mapping[str, object], int], Mapping[str, str]] | None = None,
    phase1_bay_capacity_weights: Mapping[str, int | float] | None = None,
    validation_every: int = 100,
    validation_episodes: int = 20,
    validation_rollout_samples: int | None = None,
    validation_episode_jobs: Sequence[Mapping[str, object]] | None = None,
    validation_episode_job_factory: Callable[[int], Mapping[str, object]] | None = None,
    device: str = "cpu",
    checkpoint_every: int = 0,
    write_candidate_summary: bool = False,
    score_mode: str = "raw",
    resume_checkpoint: str | Path | None = None,
    constraint_profile: PhaseConstraintProfile | None = None,
) -> Dict:
    """Self-labeling으로 통합 Phase 2 batch-machine policy를 학습한다."""

    resolved_validation_rollout_samples = rollout_samples if validation_rollout_samples is None else validation_rollout_samples
    resolved_constraint_profile = constraint_profile or default_phase2_constraint_profile()
    score_field_names = _score_field_names(score_mode)
    _validate_train_inputs(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        episodes=episodes,
        lr=lr,
        heuristic_algorithms=heuristic_algorithms,
        rollout_samples=rollout_samples,
        validation_rollout_samples=resolved_validation_rollout_samples,
        validation_every=validation_every,
        validation_episodes=validation_episodes,
        checkpoint_every=checkpoint_every,
        write_candidate_summary=write_candidate_summary,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        action_pool_limit=action_pool_limit,
        episode_jobs=episode_jobs,
        episode_job_factory=episode_job_factory,
        validation_episode_jobs=validation_episode_jobs,
        validation_episode_job_factory=validation_episode_job_factory,
        phase1_heuristic=phase1_heuristic,
        phase1_bay_ids=phase1_bay_ids,
        phase1_assignment_builder=phase1_assignment_builder,
        score_mode=score_mode,
    )
    torch.manual_seed(seed)
    torch_device = _resolve_torch_device(device)
    model = Phase2SetPointerPolicy(hidden_dim=hidden_dim).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resolved_phase1_capacity_weights = _resolve_phase1_capacity_weights_for_run_spec(
        machines=machines,
        phase1_bay_ids=phase1_bay_ids,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
    )
    run_spec = build_phase2_run_spec(
        score_mode=score_mode,
        score_fields=score_field_names,
        action_pool_limit=action_pool_limit,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        phase1_bay_capacity_weights=resolved_phase1_capacity_weights,
        constraint_profile=resolved_constraint_profile,
        heuristic_algorithms=heuristic_algorithms,
        train_rollout_samples=rollout_samples,
        validation_rollout_samples=resolved_validation_rollout_samples,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_path / "metrics.csv"
    subproblem_metrics_csv = output_path / "subproblem_metrics.csv"
    candidate_summary_csv = output_path / "candidate_summary.csv"
    best_batches_csv = output_path / "best_batches.csv"
    best_timeline_csv = output_path / "best_timeline.csv"
    best_assignment_csv = output_path / "best_machine_assignment.csv"
    validation_summary_csv = output_path / "validation_summary.csv"
    validation_candidate_summary_csv = output_path / "validation_candidate_summary.csv"
    checkpoint_path = output_path / "phase2_batch_machine_policy.pt"
    checkpoint_dir = output_path / "checkpoints"
    summary_json = output_path / "summary.json"
    if checkpoint_every > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resume_path = _resolve_phase2_resume_checkpoint(resume_checkpoint, output_path)
    resumed_from_episode = 0
    if resume_path is not None:
        resumed_from_episode = _load_phase2_batch_machine_checkpoint(
            model=model,
            optimizer=optimizer,
            path=resume_path,
            hidden_dim=hidden_dim,
            score_mode=score_mode,
            torch_device=torch_device,
            expected_run_spec=run_spec,
        )
        if resumed_from_episode >= episodes:
            print(
                "[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                f"cause=resume_episode_not_less_than_target resume_episode={resumed_from_episode} "
                f"target_episodes={episodes}"
            )
            raise RuntimeError("resume checkpoint already reached requested target episodes")
        _prepare_resume_csv(metrics_csv, _metrics_fields(score_field_names), "episode", resumed_from_episode)
        _prepare_resume_csv(subproblem_metrics_csv, _subproblem_metrics_fields(score_field_names), "episode", resumed_from_episode)
        if write_candidate_summary:
            _prepare_resume_csv(candidate_summary_csv, _candidate_fields(score_field_names), "episode", resumed_from_episode)
        elif candidate_summary_csv.exists():
            candidate_summary_csv.unlink()
            print(
                "[CHECK][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                f"removed_stale_candidate_summary={candidate_summary_csv}"
            )
        validation_rows = _prepare_resume_csv(
            validation_summary_csv,
            _validation_summary_fields(score_field_names),
            "train_episode",
            resumed_from_episode,
        )
        validation_candidate_rows = _prepare_resume_csv(
            validation_candidate_summary_csv,
            _validation_candidate_fields(score_field_names),
            "train_episode",
            resumed_from_episode,
        )
    else:
        for path in (metrics_csv, subproblem_metrics_csv, candidate_summary_csv, validation_summary_csv, validation_candidate_summary_csv):
            if path.exists():
                path.unlink()
        validation_rows: list[dict] = []
        validation_candidate_rows: list[dict] = []

    rows: list[dict] = []
    validation_plot_paths: Dict[str, str] = {}
    last_best: Phase2BatchMachineCandidate | None = None
    start_episode = resumed_from_episode + 1

    print("[phase2-train-batch-machine-self-labeling]")
    print(f"- episodes: {episodes}")
    print(f"- start_episode: {start_episode}")
    print(f"- resume_checkpoint: {str(resume_path) if resume_path is not None else ''}")
    print(f"- resumed_from_episode: {resumed_from_episode}")
    print(f"- machine_count: {len(machines)}")
    print(f"- score_mode: {score_mode}")
    print(f"- objective: {','.join(score_field_names)}")
    print(f"- heuristic_algorithms: {','.join(heuristic_algorithms)}")
    print(f"- rollout_samples: {rollout_samples}")
    print(f"- rollout_samples_validation: {resolved_validation_rollout_samples}")
    print(f"- validation_every: {validation_every}")
    print(f"- validation_episodes: {validation_episodes}")
    print(f"- checkpoint_every: {checkpoint_every}")
    print(f"- write_candidate_summary: {write_candidate_summary}")
    print(f"- device: {torch_device}")

    for episode in range(start_episode, episodes + 1):
        current_jobs = _episode_jobs(episode, jobs, episode_jobs, episode_job_factory)
        current_phase1 = _phase1_assignments_for_episode(
            jobs=current_jobs,
            fixed_assignments=phase1_assignments,
            phase1_heuristic=phase1_heuristic,
            phase1_bay_ids=phase1_bay_ids,
            phase1_assignment_builder=phase1_assignment_builder,
            phase1_assignment_seed=episode,
            phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        )
        loss, best, candidates, subproblem_rows = _train_one_episode(
            model=model,
            optimizer=optimizer,
            episode=episode,
            jobs=current_jobs,
            machines=machines,
            phase1_assignments=current_phase1,
            heuristic_algorithms=heuristic_algorithms,
            rollout_samples=rollout_samples,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            seed=seed + episode * 100_000,
            score_mode=score_mode,
            constraint_profile=resolved_constraint_profile,
        )
        last_best = best
        row = {
            "episode": episode,
            "job_count": len(current_jobs),
            "machine_count": len(machines),
            "selection_count": len(best.transitions),
            "best_source": best.source,
            "candidate_count": len(candidates),
            "loss": round(loss, 9),
            "score_json": json.dumps(list(best.score_tuple), ensure_ascii=False),
        }
        row.update({field: best.score_tuple[index] for index, field in enumerate(score_field_names)})
        row.update(best.score_details)
        rows.append(row)
        _append_rows(metrics_csv, [row], _metrics_fields(score_field_names))
        _append_rows(subproblem_metrics_csv, subproblem_rows, _subproblem_metrics_fields(score_field_names))
        if write_candidate_summary:
            episode_candidate_rows = [
                _candidate_summary_row(episode, rank, candidate, score_field_names)
                for rank, candidate in enumerate(sorted(candidates, key=_candidate_sort_key), start=1)
            ]
            _append_rows(candidate_summary_csv, episode_candidate_rows, _candidate_fields(score_field_names))
        print(
            "[CHECK][Phase2.merged.train_phase2_batch_machine_self_labeling] "
            f"episode={episode} best_source={best.source} loss={row['loss']} score={row['score_json']}"
        )
        if validation_episodes > 0 and episode % validation_every == 0:
            for validation_episode in range(1, validation_episodes + 1):
                validation_jobs = _validation_episode_jobs(
                    validation_episode,
                    current_jobs,
                    validation_episode_jobs,
                    validation_episode_job_factory,
                )
                validation_phase1 = _phase1_assignments_for_episode(
                    jobs=validation_jobs,
                    fixed_assignments=phase1_assignments,
                    phase1_heuristic=phase1_heuristic,
                    phase1_bay_ids=phase1_bay_ids,
                    phase1_assignment_builder=phase1_assignment_builder,
                    phase1_assignment_seed=episode * 1_000_000 + validation_episode,
                    phase1_bay_capacity_weights=phase1_bay_capacity_weights,
                )
                validation_candidates = build_phase2_batch_machine_candidate_bank(
                    jobs=validation_jobs,
                    machines=machines,
                    phase1_assignments=validation_phase1,
                    model=model,
                    heuristic_algorithms=heuristic_algorithms,
                    rollout_samples=resolved_validation_rollout_samples,
                    max_wo_count=max_wo_count,
                    max_length_sum=max_length_sum,
                    action_pool_limit=action_pool_limit,
                    seed=seed + episode * 1_000_000 + validation_episode,
                    score_mode=score_mode,
                    constraint_profile=resolved_constraint_profile,
                )
                ranked_candidates = sorted(validation_candidates, key=_candidate_sort_key)
                best_validation = ranked_candidates[0]
                proposed_rank, proposed_best = _agent_best_from_ranked(ranked_candidates)
                greedy_rank, greedy_best = _agent_greedy_from_ranked(ranked_candidates)
                validation_row = _validation_summary_row(
                    train_episode=episode,
                    validation_episode=validation_episode,
                    jobs=validation_jobs,
                    machines=machines,
                    candidate_count=len(ranked_candidates),
                    best=best_validation,
                    proposed_rank=proposed_rank,
                    proposed_best=proposed_best,
                    greedy_rank=greedy_rank,
                    greedy_best=greedy_best,
                    score_field_names=score_field_names,
                )
                validation_rows.append(validation_row)
                for rank, candidate in enumerate(ranked_candidates, start=1):
                    validation_candidate_rows.append(_validation_candidate_summary_row(episode, validation_episode, rank, candidate, score_field_names))
            _write_rows(validation_summary_csv, validation_rows, _validation_summary_fields(score_field_names))
            _write_rows(validation_candidate_summary_csv, validation_candidate_rows, _validation_candidate_fields(score_field_names))
            validation_plot_paths = _write_validation_plots(output_path, validation_candidate_rows, validation_rows)
            print(
                "[VALIDATION][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                f"episode={episode} validation_episodes={validation_episodes} "
                f"validation_rollout_samples={resolved_validation_rollout_samples}"
            )
        if checkpoint_every > 0 and episode % checkpoint_every == 0:
            checkpoint_file = checkpoint_dir / f"phase2_batch_machine_policy_ep{episode:05d}.pt"
            _save_phase2_batch_machine_checkpoint(
                model=model,
                optimizer=optimizer,
                path=checkpoint_file,
                hidden_dim=hidden_dim,
                heuristic_algorithms=heuristic_algorithms,
                rollout_samples=rollout_samples,
                validation_rollout_samples=resolved_validation_rollout_samples,
                validation_every=validation_every,
                validation_episodes=validation_episodes,
                episodes=episode,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                device=torch_device,
                score_mode=score_mode,
                run_spec=run_spec,
            )
            print(
                "[CHECK][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                f"checkpoint_saved={checkpoint_file}"
            )

    if last_best is None:
        print("[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] cause=no_best_candidate")
        raise RuntimeError("merged Phase 2 training produced no best candidate")
    _write_dict_rows(best_batches_csv, last_best.batches)
    _write_dict_rows(best_timeline_csv, last_best.timeline)
    _write_assignments(best_assignment_csv, last_best)
    _save_phase2_batch_machine_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        hidden_dim=hidden_dim,
        heuristic_algorithms=heuristic_algorithms,
        rollout_samples=rollout_samples,
        validation_rollout_samples=resolved_validation_rollout_samples,
        validation_every=validation_every,
        validation_episodes=validation_episodes,
        episodes=episodes,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        action_pool_limit=action_pool_limit,
        device=torch_device,
        score_mode=score_mode,
        run_spec=run_spec,
    )
    summary = {
        "episodes": episodes,
        "start_episode": start_episode,
        "resumed_from_episode": resumed_from_episode,
        "resume_checkpoint": str(resume_path) if resume_path is not None else "",
        "job_count": len(last_best.machine_assignments),
        "machine_count": len(machines),
        "policy_type": PHASE2_SET_POINTER_POLICY_TYPE,
        "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
        "feature_group_dims": {
            name: len(features) for name, features in phase2_state_feature_schema().items()
        },
        "score_mode": score_mode,
        "score_fields": list(score_field_names),
        "raw_score_fields": list(PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES),
        "normalized_score_fields": list(PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES),
        "metrics_csv": str(metrics_csv),
        "subproblem_metrics_csv": str(subproblem_metrics_csv),
        "candidate_summary_csv": str(candidate_summary_csv),
        "validation_summary_csv": str(validation_summary_csv),
        "validation_candidate_summary_csv": str(validation_candidate_summary_csv),
        **validation_plot_paths,
        "best_batches_csv": str(best_batches_csv),
        "best_timeline_csv": str(best_timeline_csv),
        "best_assignment_csv": str(best_assignment_csv),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_every > 0 else "",
        "checkpoint_every": checkpoint_every,
        "summary_json": str(summary_json),
        "heuristic_algorithms": list(heuristic_algorithms),
        "rollout_samples": rollout_samples,
        "validation_rollout_samples": resolved_validation_rollout_samples,
        "validation_every": validation_every,
        "validation_episodes": validation_episodes,
        "action_pool_limit": action_pool_limit,
        "write_candidate_summary": write_candidate_summary,
        "device": str(torch_device),
        "run_spec": run_spec,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_phase2_batch_machine_candidate_bank(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    model: Phase2SetPointerPolicy | None,
    heuristic_algorithms: Sequence[str] = PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK,
    rollout_samples: int = 1,
    max_wo_count: int = 3,
    max_length_sum: float = 55000.0,
    action_pool_limit: int | None = None,
    seed: int = 0,
    score_mode: str = "raw",
    constraint_profile: PhaseConstraintProfile | None = None,
    common_environment: CommonHierarchicalEnvironment | None = None,
) -> List[Phase2BatchMachineCandidate]:
    """통합 Phase 2 후보 bank를 만든다. 휴리스틱 후보와 agent 후보를 함께 비교한다."""

    _score_field_names(score_mode)
    resolved_constraint_profile = constraint_profile or default_phase2_constraint_profile()
    if model is None and rollout_samples != 0:
        print("[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] cause=missing_model")
        raise RuntimeError("merged Phase 2 sampled candidates require a model")
    if model is None and not heuristic_algorithms:
        print("[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] cause=no_candidate_source")
        raise RuntimeError("merged Phase 2 candidate bank requires a model or heuristic")
    if rollout_samples < 0:
        print(
            "[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] "
            f"cause=invalid_rollout_samples value={rollout_samples}"
        )
        raise RuntimeError("rollout_samples must be non-negative")
    base_environment = common_environment or _build_phase2_common_environment(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        constraint_profile=resolved_constraint_profile,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
    )
    _validate_pristine_phase2_environment(
        environment=base_environment,
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        constraint_profile=resolved_constraint_profile,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
    )
    candidates = [
        run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            source=algorithm,
            model=model,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            seed=seed,
            score_mode=score_mode,
            constraint_profile=resolved_constraint_profile,
            common_environment=base_environment,
        )
        for algorithm in heuristic_algorithms
    ]
    if model is None:
        return candidates
    candidates.append(
        run_phase2_batch_machine_candidate(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            source="agent_greedy",
            model=model,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            seed=seed,
            score_mode=score_mode,
            constraint_profile=resolved_constraint_profile,
            common_environment=base_environment,
        )
    )
    for sample_index in range(1, rollout_samples + 1):
        candidates.append(
            run_phase2_batch_machine_candidate(
                jobs=jobs,
                machines=machines,
                phase1_assignments=phase1_assignments,
                source=f"agent_sample_{sample_index}",
                model=model,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                seed=seed + sample_index,
                score_mode=score_mode,
                constraint_profile=resolved_constraint_profile,
                common_environment=base_environment,
            )
        )
    return candidates


def run_phase2_batch_machine_candidate(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    source: str,
    model: Phase2SetPointerPolicy | None,
    max_wo_count: int = 3,
    max_length_sum: float = 55000.0,
    action_pool_limit: int | None = None,
    seed: int = 0,
    score_mode: str = "raw",
    constraint_profile: PhaseConstraintProfile | None = None,
    common_environment: CommonHierarchicalEnvironment | None = None,
) -> Phase2BatchMachineCandidate:
    """하나의 통합 후보 schedule을 생성한다."""

    _validate_candidate_inputs(jobs, machines, phase1_assignments, max_wo_count, max_length_sum)
    resolved_constraint_profile = constraint_profile or default_phase2_constraint_profile()
    _validate_action_pool_limit(action_pool_limit)
    _score_field_names(score_mode)
    machine_bay_ids = _machine_bay_ids(machines)
    jobs_by_bay = _jobs_by_phase1_bay(jobs, phase1_assignments)
    machines_by_bay = _machines_by_bay(machines, machine_bay_ids)
    if common_environment is None:
        common_env = _build_phase2_common_environment(
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            constraint_profile=resolved_constraint_profile,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
        )
    else:
        _validate_pristine_phase2_environment(
            environment=common_environment,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            constraint_profile=resolved_constraint_profile,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
        )
        common_env = common_environment.fork()
    machine_loads = common_env.state.runtime.machine_loads
    machine_clock = common_env.state.runtime.machine_available_at
    machine_assignments: Dict[str, str] = {}
    transitions: List[Phase2BatchMachineTransition] = []
    step = 0

    remaining_jobs_by_bay: Dict[str, Dict[str, object]] = {}
    bay_machines_by_bay: Dict[str, Dict[str, object]] = {}
    dispatch_cache_by_bay: Dict[str, _Phase2DispatchCache] = {}
    for bay_id in sorted(jobs_by_bay):
        bay_view = common_env.phase2_bay_view(bay_id)
        bay_jobs = dict(bay_view.jobs)
        bay_machines = dict(bay_view.machines)
        if not bay_machines:
            print(f"[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] cause=no_machine_for_bay bay_id={bay_id}")
            raise RuntimeError(f"Phase 2 Bay subproblem has no machines: {bay_id}")
        remaining_jobs_by_bay[bay_id] = bay_jobs
        bay_machines_by_bay[bay_id] = bay_machines
        dispatch_cache_by_bay[bay_id] = _build_dispatch_cache(
            jobs=jobs_by_bay[bay_id],
            machines=bay_machines,
            phase1_assignments=phase1_assignments,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            constraint_profile=resolved_constraint_profile,
        )

    while remaining_jobs_by_bay:
        feasible_by_machine: Dict[str, Sequence[str]] = {
            str(machine_id): () for machine_id in machines
        }
        remaining_job_count_by_bay: Dict[str, int] = {}
        for bay_id in sorted(remaining_jobs_by_bay):
            bay_jobs = remaining_jobs_by_bay[bay_id]
            bay_machines = bay_machines_by_bay[bay_id]
            bay_machine_loads = {
                machine_id: machine_loads[machine_id] for machine_id in bay_machines
            }
            feasible_by_machine.update(
                _feasible_jobs_by_machine(
                    jobs=bay_jobs,
                    machines=bay_machines,
                    phase1_assignments=phase1_assignments,
                    machine_loads=bay_machine_loads,
                    dispatch_cache=dispatch_cache_by_bay[bay_id],
                )
            )
            remaining_job_count_by_bay[bay_id] = len(bay_jobs)

        machine_id, target_batch_size = common_env.select_phase2_machine(
            feasible_job_ids_by_machine=feasible_by_machine,
            remaining_job_count_by_bay=remaining_job_count_by_bay,
        )
        bay_id = machine_bay_ids[machine_id]
        if bay_id not in remaining_jobs_by_bay:
            print(
                "[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] "
                f"cause=selected_machine_bay_has_no_jobs machine_id={machine_id} bay_id={bay_id}"
            )
            raise RuntimeError("selected Phase 2 machine belongs to a Bay with no remaining W/O")
        bay_jobs = remaining_jobs_by_bay[bay_id]
        bay_machines = bay_machines_by_bay[bay_id]
        dispatch_cache = dispatch_cache_by_bay[bay_id]
        bay_machine_loads = {
            current_machine_id: machine_loads[current_machine_id]
            for current_machine_id in bay_machines
        }
        bay_machine_clock = {
            current_machine_id: machine_clock[current_machine_id]
            for current_machine_id in bay_machines
        }
        batch_start_time = float(common_env.state.events.current_time)
        open_batch_id = common_env.open_batch(machine_id, target_batch_size=target_batch_size)
        job_ids: List[str] = []
        open_length_sum = 0.0
        open_batch_duration = 0.0
        open_processing_sum = 0.0
        open_cut_length_sum = 0.0
        open_bevel_quantity_sum = 0.0
        for _slot in range(target_batch_size):
            wo_actions = _wo_select_actions(
                jobs=bay_jobs,
                machines=bay_machines,
                phase1_assignments=phase1_assignments,
                machine_loads=bay_machine_loads,
                machine_clock=bay_machine_clock,
                current_time=batch_start_time,
                selected_machine_id=machine_id,
                selected_job_ids=job_ids,
                target_batch_size=target_batch_size,
                open_length_sum=open_length_sum,
                open_batch_duration=open_batch_duration,
                open_processing_sum=open_processing_sum,
                open_cut_length_sum=open_cut_length_sum,
                open_bevel_quantity_sum=open_bevel_quantity_sum,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                dispatch_cache=dispatch_cache,
                constraint_profile=resolved_constraint_profile,
            )
            if not wo_actions:
                if job_ids:
                    break
                print(
                    "[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] "
                    f"cause=no_feasible_wo_after_machine_select bay_id={bay_id} machine_id={machine_id}"
                )
                raise RuntimeError("selected machine has no feasible W/O action")
            wo_policy_state = build_phase2_policy_state(
                environment=common_env,
                bay_id=bay_id,
                actions=wo_actions,
                selected_machine_id=machine_id,
                open_batch_id=open_batch_id,
            )
            selected_wo_index = _action_index(
                actions=wo_actions,
                source=source,
                model=model,
                seed=seed + step,
                policy_state=wo_policy_state,
            )
            selected_wo = wo_actions[selected_wo_index]
            selected_job_id = str(selected_wo["job_ids"][0])
            transitions.append(
                Phase2BatchMachineTransition(
                    policy_state=wo_policy_state,
                    selected_action_index=selected_wo_index,
                    selected_machine_id=machine_id,
                    selected_job_ids=(selected_job_id,),
                    action_type="select_wo",
                    target_batch_size=target_batch_size,
                )
            )
            common_env.add_wo(open_batch_id, selected_job_id)
            job_ids.append(selected_job_id)
            open_batch = common_env.state.runtime.open_batches[open_batch_id]
            open_length_sum = float(open_batch.length_sum)
            open_batch_duration = float(open_batch.max_processing_time)
            open_processing_sum = float(open_batch.processing_time_sum)
            open_cut_length_sum = float(open_batch.cut_length_sum)
            open_bevel_quantity_sum = float(open_batch.bevel_quantity_sum)
            step += 1
        if not job_ids:
            print(
                "[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] "
                f"cause=empty_auto_closed_batch bay_id={bay_id} machine_id={machine_id}"
            )
            raise RuntimeError("merged Phase 2 attempted to close an empty batch")
        common_env.close_batch(open_batch_id)
        for job_id in job_ids:
            if job_id not in bay_jobs:
                print(f"[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] cause=missing_remaining_job job_id={job_id}")
                raise RuntimeError(f"selected job missing from remaining jobs: {job_id}")
            machine_assignments[job_id] = machine_id
            del bay_jobs[job_id]
        if not bay_jobs:
            del remaining_jobs_by_bay[bay_id]

    common_env.advance_phase2_to_completion()
    batches = sorted(
        common_env.state.runtime.completed_batches,
        key=lambda row: (float(row["start_time"]), str(row["machine_id"]), str(row["batch_id"])),
    )
    timeline = sorted(
        common_env.state.events.timeline,
        key=lambda row: (float(row["start_time"]), str(row["machine_id"]), str(row["batch_id"])),
    )
    event_log = sorted(
        (event.to_dict() for event in common_env.state.events.event_log),
        key=lambda row: (float(row["time_min"]), str(row["event_id"])),
    )
    for event_index, row in enumerate(event_log, start=1):
        row["event_id"] = f"HE{event_index:08d}"
    constraint_audit = audit_phase2_schedule_constraints(
        profile=resolved_constraint_profile,
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        batches=batches,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
    )
    if int(constraint_audit["hard_violation_count"]) != 0:
        failed_rules = sorted(
            {
                str(row["rule_name"])
                for row in constraint_audit["rows"]
                if not bool(row["passed"])
            }
        )
        print(
            "[ERROR][Phase2.merged.run_phase2_batch_machine_candidate] "
            f"cause=final_hard_constraint_violation source={source} "
            f"count={constraint_audit['hard_violation_count']} rules={failed_rules}"
        )
        raise RuntimeError("generated Phase 2 candidate failed final hard-constraint audit")
    score_tuple, score_details = _schedule_score(
        batches=batches,
        machine_loads=machine_loads,
        machine_clock=machine_clock,
        machine_bay_ids=machine_bay_ids,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        score_mode=score_mode,
        hard_violation_count=int(constraint_audit["hard_violation_count"]),
    )
    return Phase2BatchMachineCandidate(
        source=source,
        transitions=transitions,
        machine_assignments=machine_assignments,
        machine_loads=machine_loads,
        machine_bay_ids=machine_bay_ids,
        batches=batches,
        timeline=timeline,
        event_log=event_log,
        constraint_audit_rows=list(constraint_audit["rows"]),
        score_tuple=score_tuple,
        score_details=score_details,
    )


def _train_one_episode(
    model: Phase2SetPointerPolicy,
    optimizer: torch.optim.Optimizer,
    episode: int,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    heuristic_algorithms: Sequence[str],
    rollout_samples: int,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    seed: int,
    score_mode: str,
    constraint_profile: PhaseConstraintProfile,
) -> tuple[float, Phase2BatchMachineCandidate, List[Phase2BatchMachineCandidate], List[Dict]]:
    machine_bay_ids = _machine_bay_ids(machines)
    jobs_by_bay = _jobs_by_phase1_bay(jobs, phase1_assignments)
    machines_by_bay = _machines_by_bay(machines, machine_bay_ids)
    all_candidates: List[Phase2BatchMachineCandidate] = []
    bay_bests: List[Phase2BatchMachineCandidate] = []
    bay_losses: List[float] = []
    subproblem_rows: List[Dict] = []
    score_field_names = _score_field_names(score_mode)

    for bay_index, bay_id in enumerate(sorted(jobs_by_bay), start=1):
        bay_machines = machines_by_bay.get(bay_id)
        if not bay_machines:
            print(f"[ERROR][Phase2.merged._train_one_episode] cause=no_machine_for_bay bay_id={bay_id}")
            raise RuntimeError(f"Phase 2 Bay subproblem has no machines: {bay_id}")
        bay_candidates = [
            _with_subproblem_bay(candidate, bay_id)
            for candidate in build_phase2_batch_machine_candidate_bank(
                jobs=jobs_by_bay[bay_id],
                machines=bay_machines,
                phase1_assignments=phase1_assignments,
                model=model,
                heuristic_algorithms=heuristic_algorithms,
                rollout_samples=rollout_samples,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                seed=seed + bay_index * 10_000,
                score_mode=score_mode,
                constraint_profile=constraint_profile,
            )
        ]
        best = min(bay_candidates, key=_candidate_sort_key)
        bay_loss = _update_from_candidate(model, optimizer, best)
        bay_losses.append(bay_loss)
        bay_bests.append(best)
        all_candidates.extend(bay_candidates)
        subproblem_rows.append(
            _subproblem_metrics_row(
                episode=episode,
                bay_id=bay_id,
                job_count=len(jobs_by_bay[bay_id]),
                machine_count=len(bay_machines),
                candidate_count=len(bay_candidates),
                best=best,
                loss=bay_loss,
                score_field_names=score_field_names,
            )
        )
        print(
            "[CHECK][Phase2.merged._train_one_episode.subproblem] "
            f"bay_id={bay_id} job_count={len(jobs_by_bay[bay_id])} machine_count={len(bay_machines)} "
            f"candidate_count={len(bay_candidates)} best_source={best.source} "
            f"loss={bay_loss:.9f} score={json.dumps(list(best.score_tuple), ensure_ascii=False)}",
            flush=True,
        )

    combined_best = _combine_subproblem_bests(
        bay_bests=bay_bests,
        machines=machines,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        score_mode=score_mode,
    )
    return sum(bay_losses) / float(len(bay_losses)), combined_best, all_candidates, subproblem_rows


def _update_from_candidate(
    model: Phase2SetPointerPolicy,
    optimizer: torch.optim.Optimizer,
    best: Phase2BatchMachineCandidate,
) -> float:
    if not best.transitions:
        print(f"[ERROR][Phase2.merged._update_from_candidate] cause=no_transitions best_source={best.source}")
        raise RuntimeError("best merged Phase 2 candidate has no transitions")
    device = next(model.parameters()).device
    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    for transition in best.transitions:
        logits = model(transition.policy_state)
        total_loss = total_loss + F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([transition.selected_action_index], dtype=torch.long, device=device),
        )
    mean_loss = total_loss / len(best.transitions)
    optimizer.zero_grad()
    mean_loss.backward()
    optimizer.step()
    return float(mean_loss.item())


def _with_subproblem_bay(candidate: Phase2BatchMachineCandidate, bay_id: str) -> Phase2BatchMachineCandidate:
    return Phase2BatchMachineCandidate(
        source=candidate.source,
        transitions=candidate.transitions,
        machine_assignments=candidate.machine_assignments,
        machine_loads=candidate.machine_loads,
        machine_bay_ids=candidate.machine_bay_ids,
        batches=candidate.batches,
        timeline=candidate.timeline,
        event_log=candidate.event_log,
        constraint_audit_rows=candidate.constraint_audit_rows,
        score_tuple=candidate.score_tuple,
        score_details=candidate.score_details,
        subproblem_bay_id=str(bay_id),
    )


def _combine_subproblem_bests(
    bay_bests: Sequence[Phase2BatchMachineCandidate],
    machines: Mapping[str, object],
    max_wo_count: int,
    max_length_sum: float,
    score_mode: str,
) -> Phase2BatchMachineCandidate:
    if not bay_bests:
        print("[ERROR][Phase2.merged._combine_subproblem_bests] cause=no_bay_bests")
        raise RuntimeError("merged Phase 2 training produced no Bay subproblem best candidates")
    machine_bay_ids = _machine_bay_ids(machines)
    machine_loads = _empty_machine_loads(machines)
    machine_clock = {str(machine_id): 0.0 for machine_id in machines}
    machine_assignments: Dict[str, str] = {}
    transitions: List[Phase2BatchMachineTransition] = []
    batches: List[Dict] = []
    timeline: List[Dict] = []
    event_log: List[Dict] = []
    constraint_audit_rows: List[Dict] = []
    source_parts: List[str] = []

    for candidate in bay_bests:
        if not candidate.subproblem_bay_id:
            print(f"[ERROR][Phase2.merged._combine_subproblem_bests] cause=missing_bay_id source={candidate.source}")
            raise RuntimeError("Bay subproblem best candidate has no bay id")
        source_parts.append(f"{candidate.subproblem_bay_id}:{candidate.source}")
        transitions.extend(candidate.transitions)
        batches.extend(candidate.batches)
        timeline.extend(candidate.timeline)
        event_log.extend(dict(row) for row in candidate.event_log)
        constraint_audit_rows.extend(dict(row) for row in candidate.constraint_audit_rows)
        for job_id, machine_id in candidate.machine_assignments.items():
            machine_assignments[str(job_id)] = str(machine_id)
        for machine_id, loads in candidate.machine_loads.items():
            machine_loads[str(machine_id)] = dict(loads)
        for row in candidate.timeline:
            machine_id = str(row["machine_id"])
            machine_clock[machine_id] = max(machine_clock[machine_id], float(row["finish_time"]))

    score_tuple, score_details = _schedule_score(
        batches=batches,
        machine_loads=machine_loads,
        machine_clock=machine_clock,
        machine_bay_ids=machine_bay_ids,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        score_mode=score_mode,
        hard_violation_count=sum(1 for row in constraint_audit_rows if not bool(row["passed"])),
    )
    event_log.sort(key=lambda row: (float(row["time_min"]), str(row["event_id"])))
    for event_index, row in enumerate(event_log, start=1):
        row["event_id"] = f"HE{event_index:08d}"
    return Phase2BatchMachineCandidate(
        source="|".join(source_parts),
        transitions=transitions,
        machine_assignments=machine_assignments,
        machine_loads=machine_loads,
        machine_bay_ids=machine_bay_ids,
        batches=batches,
        timeline=timeline,
        event_log=event_log,
        constraint_audit_rows=constraint_audit_rows,
        score_tuple=score_tuple,
        score_details=score_details,
    )


def _wo_select_actions(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_clock: Mapping[str, float],
    current_time: float,
    selected_machine_id: str,
    selected_job_ids: Sequence[str],
    target_batch_size: int,
    open_length_sum: float,
    open_batch_duration: float,
    open_processing_sum: float,
    open_cut_length_sum: float,
    open_bevel_quantity_sum: float,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    dispatch_cache: _Phase2DispatchCache | None = None,
    constraint_profile: PhaseConstraintProfile | None = None,
) -> List[Dict]:
    """선택된 machine의 열린 batch에 추가 가능한 W/O 후보를 만든다."""

    selected_machine_start_time = float(current_time)
    if not math.isfinite(selected_machine_start_time) or selected_machine_start_time < 0.0:
        print(
            "[ERROR][Phase2.merged._wo_select_actions] "
            f"cause=invalid_current_time value={current_time}"
        )
        raise RuntimeError("Phase 2 W/O action requires a finite non-negative event time")
    selected_machine_available_at = float(machine_clock[selected_machine_id])
    if selected_machine_available_at > selected_machine_start_time + 1e-9:
        print(
            "[ERROR][Phase2.merged._wo_select_actions] "
            f"cause=selected_machine_busy machine_id={selected_machine_id} "
            f"current_time={selected_machine_start_time} available_at={selected_machine_available_at}"
        )
        raise RuntimeError("Phase 2 W/O actions cannot be built for a busy machine")

    feasible_by_machine = _feasible_jobs_by_machine(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        machine_loads=machine_loads,
        dispatch_cache=dispatch_cache,
    )
    if selected_machine_id not in feasible_by_machine:
        print(
            "[ERROR][Phase2.merged._wo_select_actions] "
            f"cause=unknown_selected_machine machine_id={selected_machine_id}"
        )
        raise RuntimeError(f"unknown selected machine: {selected_machine_id}")
    resolved_constraint_profile = constraint_profile or default_phase2_constraint_profile()
    selected_set = {str(job_id) for job_id in selected_job_ids}
    feasible_job_ids: List[str] = []
    for job_id in _ordered_feasible_job_ids(
        jobs,
        feasible_by_machine[selected_machine_id],
        action_pool_limit,
        dispatch_cache,
    ):
        if job_id in selected_set:
            continue
        projected_job_ids = tuple([*selected_job_ids, job_id])
        projected_length_sum = open_length_sum + _job_length(jobs[job_id], job_id, dispatch_cache)
        constraint_result = evaluate_phase2_action_constraints(
            profile=resolved_constraint_profile,
            job=jobs[job_id],
            machine=machines[selected_machine_id],
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at=machine_clock,
            current_time=selected_machine_start_time,
            candidate_batch_job_ids=projected_job_ids,
            candidate_batch_length_sum=projected_length_sum,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
        )
        if constraint_result.hard_passed:
            feasible_job_ids.append(job_id)
    if not feasible_job_ids:
        return []
    machine_bay_ids = dispatch_cache.machine_bay_ids if dispatch_cache else _machine_bay_ids(machines)
    remaining_jobs_by_bay: Dict[str, set[str]] = {}
    for machine_id, job_ids in feasible_by_machine.items():
        bay_id = machine_bay_ids[machine_id]
        remaining_jobs_by_bay.setdefault(bay_id, set()).update(str(job_id) for job_id in job_ids)
    bay_remaining_processing_sum = {
        bay_id: sum(_processing_time_for_job(jobs[job_id], job_id, dispatch_cache) for job_id in job_ids)
        for bay_id, job_ids in remaining_jobs_by_bay.items()
    }
    actions: List[Dict] = []
    for job_id in feasible_job_ids:
        job = jobs[job_id]
        processing_time = _processing_time_for_job(job, job_id, dispatch_cache)
        cut_length = _cut_length_for_job(job, job_id, dispatch_cache)
        bevel_quantity = _bevel_quantity_for_job(job, job_id, dispatch_cache)
        projected_length = open_length_sum + _job_length(job, job_id, dispatch_cache)
        projected_duration = max(open_batch_duration, processing_time)
        duration_increment = projected_duration - open_batch_duration
        projected_processing_sum = open_processing_sum + processing_time
        projected_cut_length = open_cut_length_sum + cut_length
        projected_bevel_quantity = open_bevel_quantity_sum + bevel_quantity
        projected_finish = selected_machine_start_time + projected_duration
        projected_makespan = max(projected_finish, *(float(value) for value in machine_clock.values()))
        projected_job_ids = tuple([*selected_job_ids, job_id])
        lookahead_makespan = _lookahead_makespan_lower_bound(
            jobs=jobs,
            machine_clock=machine_clock,
            machine_bay_ids=machine_bay_ids,
            remaining_jobs_by_bay=remaining_jobs_by_bay,
            bay_remaining_processing_sum=bay_remaining_processing_sum,
            selected_machine_id=selected_machine_id,
            selected_machine_start_time=selected_machine_start_time,
            selected_job_ids=projected_job_ids,
            batch_duration=projected_duration,
            max_wo_count=max_wo_count,
            processing_time_by_job=dispatch_cache.processing_time_by_job if dispatch_cache else None,
        )
        machine_load = machine_loads[selected_machine_id]
        actions.append(
            {
                "action_type": "select_wo",
                "machine_id": selected_machine_id,
                "job_ids": (str(job_id),),
                "target_batch_size": target_batch_size,
                "wo_count": len(projected_job_ids),
                "open_batch_size": len(selected_job_ids),
                "remaining_batch_slots_after": max(0, int(target_batch_size) - len(projected_job_ids)),
                "length_sum": round(projected_length, 6),
                "batch_duration": round(projected_duration, 6),
                "duration_increment": round(duration_increment, 6),
                "job_processing_time_sum": round(projected_processing_sum, 6),
                "cut_length_sum": round(projected_cut_length, 6),
                "bevel_quantity_sum": round(projected_bevel_quantity, 6),
                "candidate_processing_time": round(processing_time, 6),
                "candidate_cut_length": round(cut_length, 6),
                "candidate_bevel_quantity": round(bevel_quantity, 6),
                "machine_clock": round(selected_machine_start_time, 6),
                "projected_finish_time": round(projected_finish, 6),
                "projected_makespan": round(projected_makespan, 6),
                "lookahead_makespan_lower_bound": round(lookahead_makespan, 6),
                "remaining_job_count": len(jobs),
            }
        )
    return actions


def _feasible_jobs_by_machine(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    machine_loads: Mapping[str, Mapping[str, int | float]],
    dispatch_cache: _Phase2DispatchCache | None = None,
) -> Dict[str, List[str]]:
    if dispatch_cache is not None:
        remaining_job_ids = {str(job_id) for job_id in jobs}
        return {
            str(machine_id): [job_id for job_id in dispatch_cache.static_feasible_by_machine[str(machine_id)] if job_id in remaining_job_ids]
            for machine_id in machines
        }
    graph = build_phase2_wo_machine_graph(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        machine_loads=machine_loads,
    )
    feasible_by_machine: Dict[str, list[str]] = {str(machine_id): [] for machine_id in machines}
    for edge in graph["candidate_edges"]:
        feasible_by_machine[str(edge["target_machine_id"])].append(str(edge["source_job_id"]))
    return feasible_by_machine


def _ordered_feasible_job_ids(
    jobs: Mapping[str, object],
    job_ids: Sequence[str],
    action_pool_limit: int | None,
    dispatch_cache: _Phase2DispatchCache | None = None,
) -> List[str]:
    ordered = sorted(
        {str(job_id) for job_id in job_ids},
        key=lambda job_id: (
            -_processing_time_for_job(jobs[job_id], job_id, dispatch_cache),
            -_cut_length_for_job(jobs[job_id], job_id, dispatch_cache),
            job_id,
        ),
    )
    if action_pool_limit is None:
        return ordered
    return ordered[:action_pool_limit]


def _lookahead_makespan_lower_bound(
    jobs: Mapping[str, object],
    machine_clock: Mapping[str, float],
    machine_bay_ids: Mapping[str, str],
    remaining_jobs_by_bay: Mapping[str, set[str]],
    bay_remaining_processing_sum: Mapping[str, float] | None,
    selected_machine_id: str,
    selected_machine_start_time: float,
    selected_job_ids: Sequence[str],
    batch_duration: float,
    max_wo_count: int,
    processing_time_by_job: Mapping[str, float] | None = None,
) -> float:
    """Estimate a cheap makespan lower bound after selecting one batch."""

    projected_clock = {
        str(machine_id): max(float(value), float(selected_machine_start_time))
        for machine_id, value in machine_clock.items()
    }
    projected_clock[str(selected_machine_id)] = float(selected_machine_start_time) + float(batch_duration)
    lower_bound = max(projected_clock.values())
    selected = {str(job_id) for job_id in selected_job_ids}
    machines_by_bay: Dict[str, List[str]] = {}
    for machine_id, bay_id in machine_bay_ids.items():
        machines_by_bay.setdefault(str(bay_id), []).append(str(machine_id))
    for bay_id, bay_machine_ids in machines_by_bay.items():
        bay_remaining_jobs = set(remaining_jobs_by_bay.get(bay_id, set())) - selected
        if not bay_remaining_jobs:
            continue
        if bay_remaining_processing_sum is None:
            remaining_processing_sum = sum(_processing_time(jobs[job_id]) for job_id in bay_remaining_jobs)
        else:
            raw_remaining_sum = float(bay_remaining_processing_sum.get(bay_id, 0.0))
            selected_sum = sum(
                _processing_time_for_job(jobs[job_id], job_id, None, processing_time_by_job)
                for job_id in selected
                if job_id in remaining_jobs_by_bay.get(bay_id, set())
            )
            remaining_processing_sum = max(0.0, raw_remaining_sum - selected_sum)
        remaining_equivalent_time = remaining_processing_sum / float(max_wo_count)
        bay_clock_sum = sum(projected_clock[machine_id] for machine_id in bay_machine_ids)
        lower_bound = max(lower_bound, (bay_clock_sum + remaining_equivalent_time) / float(len(bay_machine_ids)))
    return float(lower_bound)


def _action_index(
    actions: Sequence[Mapping],
    source: str,
    model: Phase2SetPointerPolicy | None,
    seed: int,
    policy_state: Phase2PolicyState,
) -> int:
    if source == "agent_greedy":
        return _model_action_index(actions, model, policy_state, sample=False, seed=seed)
    if source.startswith("agent_sample_"):
        return _model_action_index(actions, model, policy_state, sample=True, seed=seed)
    best: tuple | None = None
    best_index = -1
    for index, action in enumerate(actions):
        key = _heuristic_key(source, action)
        if best is None or key < best:
            best = key
            best_index = index
    if best_index < 0:
        print(f"[ERROR][Phase2.merged._action_index] cause=no_action source={source}")
        raise RuntimeError(f"merged Phase 2 candidate found no action: {source}")
    return best_index


def _heuristic_key(source: str, action: Mapping) -> tuple:
    action_type = str(action.get("action_type", "select_wo"))
    if action_type != "select_wo":
        print(
            "[ERROR][Phase2.merged._heuristic_key] "
            f"cause=non_wo_action source={source} action_type={action_type}"
        )
        raise RuntimeError("Phase 2 heuristics accept SELECT_WO actions only")
    job_ids = tuple(str(job_id) for job_id in action["job_ids"])
    common = (str(action["machine_id"]), job_ids)
    if source == "workload_makespan_dispatch":
        if int(action["open_batch_size"]) == 0:
            return (
                float(action["machine_clock"]),
                -float(action["candidate_processing_time"]),
                -float(action["candidate_cut_length"]),
                -float(action["candidate_bevel_quantity"]),
                *common,
            )
        return (
            float(action["duration_increment"]),
            float(action["lookahead_makespan_lower_bound"]),
            float(action["projected_makespan"]),
            float(action["projected_finish_time"]),
            int(action["remaining_batch_slots_after"]),
            -float(action["candidate_processing_time"]),
            -float(action["candidate_cut_length"]),
            -float(action["candidate_bevel_quantity"]),
            *common,
        )
    if source == "min_makespan":
        return (float(action["projected_makespan"]), float(action["projected_finish_time"]), -int(action["wo_count"]), *common)
    if source == "lookahead_min_makespan":
        return (
            float(action["lookahead_makespan_lower_bound"]),
            float(action["projected_makespan"]),
            float(action["projected_finish_time"]),
            -int(action["wo_count"]),
            *common,
        )
    if source == "best_fit_lth":
        return (-int(action["wo_count"]), -float(action["length_sum"]), float(action["projected_makespan"]), *common)
    if source == "spt_batch":
        return (float(action["candidate_processing_time"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "lpt_batch":
        return (-float(action["candidate_processing_time"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    print(f"[ERROR][Phase2.merged._heuristic_key] cause=unknown_source source={source}")
    raise RuntimeError(f"unknown merged Phase 2 candidate source: {source}")


def _model_action_index(
    actions: Sequence[Mapping],
    model: Phase2SetPointerPolicy | None,
    policy_state: Phase2PolicyState,
    sample: bool,
    seed: int,
) -> int:
    if model is None:
        print("[ERROR][Phase2.merged._model_action_index] cause=missing_model_for_agent_source")
        raise RuntimeError("merged Phase 2 agent candidate requires a model")
    with torch.no_grad():
        logits = model(policy_state).detach().cpu()
    if not sample:
        return int(torch.argmax(logits).item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    probabilities = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def _schedule_score(
    batches: Sequence[Mapping],
    machine_loads: Mapping[str, Mapping[str, int | float]],
    machine_clock: Mapping[str, float],
    machine_bay_ids: Mapping[str, str],
    max_wo_count: int,
    max_length_sum: float,
    score_mode: str,
    hard_violation_count: int,
) -> tuple[tuple, Dict[str, int | float]]:
    score_field_names = _score_field_names(score_mode)
    metrics = calculate_phase2_schedule_metrics(
        machine_loads=machine_loads,
        machine_clock=machine_clock,
        machine_bay_ids=machine_bay_ids,
        batches=batches,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        hard_violation_count=hard_violation_count,
    )
    score_key = "normalized_score" if score_field_names == PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES else "raw_score"
    return metrics[score_key], metrics["score_details"]


def _candidate_sort_key(candidate: Phase2BatchMachineCandidate) -> tuple:
    return (*candidate.score_tuple, candidate.source)


def _is_agent_source(source: str) -> bool:
    """Return True for candidates generated by the current policy."""

    return source == "agent_greedy" or source.startswith("agent_sample_")


def _agent_best_from_ranked(ranked_candidates: Sequence[Phase2BatchMachineCandidate]) -> tuple[int, Phase2BatchMachineCandidate | None]:
    """Return the best current-policy candidate from a ranked validation candidate list."""

    for rank, candidate in enumerate(ranked_candidates, start=1):
        if _is_agent_source(candidate.source):
            return rank, candidate
    return 0, None


def _agent_greedy_from_ranked(ranked_candidates: Sequence[Phase2BatchMachineCandidate]) -> tuple[int, Phase2BatchMachineCandidate | None]:
    """Return the deterministic greedy policy candidate from a ranked validation candidate list."""

    for rank, candidate in enumerate(ranked_candidates, start=1):
        if candidate.source == "agent_greedy":
            return rank, candidate
    return 0, None


def _empty_machine_loads(machines: Mapping[str, object]) -> Dict[str, Dict[str, int | float]]:
    return {
        str(machine_id): {
            "wo_count": 0,
            "processing_time_sum": 0.0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "occupancy_time_sum": 0.0,
            "batch_count": 0,
        }
        for machine_id in machines
    }


def _add_job_load(loads: Dict[str, int | float], job: object) -> None:
    loads["wo_count"] += 1
    loads["processing_time_sum"] += _processing_time(job)
    loads["cut_length_sum"] += _required_non_negative(job, "cut_length")
    loads["bevel_quantity_sum"] += int(_required_non_negative(job, "bevel_quantity"))


def _build_dispatch_cache(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    max_wo_count: int,
    max_length_sum: float,
    constraint_profile: PhaseConstraintProfile,
) -> _Phase2DispatchCache:
    """후보 생성 중 변하지 않는 feasible edge와 W/O 수치를 한 번만 계산한다."""

    machine_bay_ids = _machine_bay_ids(machines)
    graph = build_phase2_wo_machine_graph(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        machine_loads=_empty_machine_loads(machines),
    )
    feasible_by_machine: Dict[str, list[str]] = {str(machine_id): [] for machine_id in machines}
    initial_machine_clock = {str(machine_id): 0.0 for machine_id in machines}
    for edge in graph["candidate_edges"]:
        machine_id = str(edge["target_machine_id"])
        job_id = str(edge["source_job_id"])
        if machine_id not in feasible_by_machine:
            print(
                "[ERROR][Phase2.merged._build_dispatch_cache] "
                f"cause=edge_unknown_machine machine_id={machine_id} job_id={job_id}"
            )
            raise RuntimeError(f"Phase 2 graph returned unknown machine edge: {machine_id}")
        plate_length = _job_length(jobs[job_id], job_id)
        constraint_result = evaluate_phase2_action_constraints(
            profile=constraint_profile,
            job=jobs[job_id],
            machine=machines[machine_id],
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            machine_available_at=initial_machine_clock,
            current_time=0.0,
            candidate_batch_job_ids=(job_id,),
            candidate_batch_length_sum=plate_length,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
        )
        if constraint_result.hard_passed:
            feasible_by_machine[machine_id].append(job_id)
    return _Phase2DispatchCache(
        machine_bay_ids=machine_bay_ids,
        static_feasible_by_machine={machine_id: tuple(job_ids) for machine_id, job_ids in feasible_by_machine.items()},
        processing_time_by_job={str(job_id): _processing_time(job) for job_id, job in jobs.items()},
        plate_length_by_job={str(job_id): _job_length(job, str(job_id)) for job_id, job in jobs.items()},
        cut_length_by_job={str(job_id): _required_non_negative(job, "cut_length") for job_id, job in jobs.items()},
        bevel_quantity_by_job={str(job_id): _required_non_negative(job, "bevel_quantity") for job_id, job in jobs.items()},
    )


def _build_phase2_common_environment(
    *,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    constraint_profile: PhaseConstraintProfile,
    max_wo_count: int,
    max_length_sum: float,
) -> CommonHierarchicalEnvironment:
    """Phase 2 후보 하나가 사용할 공통 planning/runtime/event 상태를 만든다."""

    machine_bay_ids = _machine_bay_ids(machines)
    bay_capacity_weights: Dict[str, float] = {}
    for bay_id in machine_bay_ids.values():
        bay_capacity_weights[bay_id] = bay_capacity_weights.get(bay_id, 0.0) + 1.0
    environment = CommonHierarchicalEnvironment(
        jobs=jobs,
        machines=machines,
        bay_capacity_weights=bay_capacity_weights,
        constraint_profile=constraint_profile,
        max_batch_wo_count=max_wo_count,
        max_batch_length_sum=max_length_sum,
    )

    block_ids = {_job_block_set_id(job, str(job_id)) for job_id, job in jobs.items()}
    selected_assignments = {
        block_id: str(phase1_assignments[block_id])
        for block_id in sorted(block_ids)
        if block_id in phase1_assignments
    }
    if set(selected_assignments) != block_ids:
        missing = sorted(block_ids - set(selected_assignments))
        print(
            "[ERROR][Phase2.merged._build_phase2_common_environment] "
            f"cause=missing_phase1_assignments block_ids={missing}"
        )
        raise RuntimeError(f"missing Phase 1 assignments for common environment: {missing}")

    bay_loads = {
        bay_id: {
            "block_count": 0,
            "wo_count": 0,
            "steel_quantity_sum": 0,
            "cut_length_sum": 0.0,
            "bevel_quantity_sum": 0,
            "long_cut_bay24_count": 0,
            "capacity_weight": bay_capacity_weights[bay_id],
        }
        for bay_id in environment.bay_ids
    }
    blocks_by_bay: Dict[str, set[str]] = {bay_id: set() for bay_id in environment.bay_ids}
    block_cut_lengths: Dict[str, float] = {}
    for job_id, job in jobs.items():
        block_id = _job_block_set_id(job, str(job_id))
        bay_id = selected_assignments[block_id]
        if bay_id not in bay_loads:
            print(
                "[ERROR][Phase2.merged._build_phase2_common_environment] "
                f"cause=assigned_bay_without_machine block_id={block_id} bay_id={bay_id}"
            )
            raise RuntimeError(f"Phase 1 assigned Bay without Phase 2 machine: {bay_id}")
        cut_length = _required_non_negative(job, "cut_length")
        bevel_quantity = _required_non_negative(job, "bevel_quantity")
        row = bay_loads[bay_id]
        row["wo_count"] += 1
        row["steel_quantity_sum"] += 1
        row["cut_length_sum"] += cut_length
        row["bevel_quantity_sum"] += int(bevel_quantity)
        blocks_by_bay[bay_id].add(block_id)
        block_cut_lengths[block_id] = block_cut_lengths.get(block_id, 0.0) + cut_length
    for bay_id, assigned_blocks in blocks_by_bay.items():
        bay_loads[bay_id]["block_count"] = len(assigned_blocks)
        if bay_id == "24":
            bay_loads[bay_id]["long_cut_bay24_count"] = sum(
                1 for block_id in assigned_blocks if block_cut_lengths[block_id] > 1000.0
            )
    environment.commit_phase1(block_to_bay=selected_assignments, bay_loads=bay_loads)
    return environment


def _validate_pristine_phase2_environment(
    *,
    environment: CommonHierarchicalEnvironment,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    constraint_profile: PhaseConstraintProfile,
    max_wo_count: int,
    max_length_sum: float,
) -> None:
    """후보 bank가 공유할 Phase 1 직후 snapshot 계약을 검증한다."""

    if set(environment.jobs) != set(jobs) or set(environment.machines) != set(machines):
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            f"cause=static_scenario_mismatch environment_jobs={len(environment.jobs)} jobs={len(jobs)} "
            f"environment_machines={len(environment.machines)} machines={len(machines)}"
        )
        raise RuntimeError("Phase 2 common environment static scenario mismatch")
    if environment.constraint_profile != constraint_profile:
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            "cause=constraint_profile_mismatch"
        )
        raise RuntimeError("Phase 2 common environment constraint profile mismatch")
    if (
        environment.max_batch_wo_count != int(max_wo_count)
        or not math.isclose(
            environment.max_batch_length_sum,
            float(max_length_sum),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            f"cause=batch_limit_mismatch environment_count={environment.max_batch_wo_count} "
            f"requested_count={max_wo_count} environment_length={environment.max_batch_length_sum} "
            f"requested_length={max_length_sum}"
        )
        raise RuntimeError("Phase 2 common environment batch limit mismatch")

    expected_blocks = {_job_block_set_id(job, str(job_id)) for job_id, job in jobs.items()}
    expected_assignments = {
        block_id: str(phase1_assignments[block_id])
        for block_id in expected_blocks
        if block_id in phase1_assignments
    }
    if set(expected_assignments) != expected_blocks:
        missing = sorted(expected_blocks - set(expected_assignments))
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            f"cause=missing_phase1_assignment block_ids={missing}"
        )
        raise RuntimeError("Phase 2 common environment validation found missing Phase 1 assignments")
    if (
        not environment.state.planning.phase1_completed
        or environment.state.planning.block_to_bay != expected_assignments
    ):
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            "cause=phase1_snapshot_mismatch"
        )
        raise RuntimeError("Phase 2 common environment is not the requested post-Phase1 snapshot")

    runtime = environment.state.runtime
    events = environment.state.events
    has_nonzero_load = any(
        any(not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-9) for value in load.values())
        for load in runtime.machine_loads.values()
    )
    has_nonzero_clock = any(
        not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1e-9)
        for value in runtime.machine_available_at.values()
    )
    if (
        runtime.open_batches
        or runtime.completed_batches
        or runtime.scheduled_jobs
        or runtime.batch_seq != 0
        or has_nonzero_load
        or has_nonzero_clock
        or events.current_time != 0.0
        or events.timeline
        or events.event_log
        or events.event_seq != 0
    ):
        print(
            "[ERROR][Phase2.merged._validate_pristine_phase2_environment] "
            f"cause=runtime_not_pristine open_batches={len(runtime.open_batches)} "
            f"completed_batches={len(runtime.completed_batches)} scheduled_jobs={len(runtime.scheduled_jobs)} "
            f"event_count={len(events.event_log)}"
        )
        raise RuntimeError("Phase 2 candidate bank requires a pristine post-Phase1 runtime")


def _machine_bay_ids(machines: Mapping[str, object]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for machine_id, machine in machines.items():
        bay_id = machine.get("bay_id") if isinstance(machine, Mapping) else getattr(machine, "bay_id", None)
        if bay_id in (None, ""):
            print(f"[ERROR][Phase2.merged._machine_bay_ids] cause=missing_bay_id machine_id={machine_id}")
            raise RuntimeError(f"machine has no bay_id: {machine_id}")
        result[str(machine_id)] = str(bay_id)
    return result


def _resolve_phase1_capacity_weights_for_run_spec(
    machines: Mapping[str, object],
    phase1_bay_ids: Sequence[str] | None,
    phase1_bay_capacity_weights: Mapping[str, int | float] | None,
) -> Dict[str, float]:
    """RunSpec에 기록할 Phase 1 Bay 용량비를 추론 없이 확정한다."""

    machine_bay_ids = _machine_bay_ids(machines)
    machine_counts: Dict[str, int] = {}
    for bay_id in machine_bay_ids.values():
        machine_counts[bay_id] = machine_counts.get(bay_id, 0) + 1
    requested_bays = (
        tuple(str(bay_id) for bay_id in phase1_bay_ids)
        if phase1_bay_ids is not None
        else tuple(sorted(machine_counts))
    )
    if not requested_bays or len(set(requested_bays)) != len(requested_bays):
        print(
            "[ERROR][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
            f"cause=invalid_phase1_bay_ids bay_ids={requested_bays}"
        )
        raise RuntimeError("Phase 2 RunSpec requires unique Phase 1 Bay IDs")
    missing_machine_bays = sorted(set(requested_bays) - set(machine_counts))
    if missing_machine_bays:
        print(
            "[ERROR][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
            f"cause=bay_without_machine bay_ids={missing_machine_bays}"
        )
        raise RuntimeError(f"Phase 1 Bays have no machines: {missing_machine_bays}")

    if phase1_bay_capacity_weights is None:
        resolved = {bay_id: float(machine_counts[bay_id]) for bay_id in requested_bays}
        print(
            "[CHECK][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
            f"source=machine_count weights={resolved}"
        )
        return resolved

    provided = {str(bay_id): value for bay_id, value in phase1_bay_capacity_weights.items()}
    missing = sorted(set(requested_bays) - set(provided))
    extra = sorted(set(provided) - set(requested_bays))
    if missing or extra:
        print(
            "[ERROR][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
            f"cause=capacity_weight_bay_mismatch missing={missing} extra={extra}"
        )
        raise RuntimeError("Phase 1 Bay capacity weight keys do not match the active Bays")
    resolved: Dict[str, float] = {}
    for bay_id in requested_bays:
        value = provided[bay_id]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            print(
                "[ERROR][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
                f"cause=invalid_capacity_weight_type bay_id={bay_id} value={value}"
            )
            raise RuntimeError(f"invalid Phase 1 Bay capacity weight: {bay_id}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0:
            print(
                "[ERROR][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
                f"cause=invalid_capacity_weight bay_id={bay_id} value={value}"
            )
            raise RuntimeError(f"invalid Phase 1 Bay capacity weight: {bay_id}")
        resolved[bay_id] = numeric
    print(
        "[CHECK][Phase2.merged._resolve_phase1_capacity_weights_for_run_spec] "
        f"source=explicit weights={resolved}"
    )
    return resolved


def _machines_by_bay(machines: Mapping[str, object], machine_bay_ids: Mapping[str, str]) -> Dict[str, Dict[str, object]]:
    """Bay별 machine subproblem을 만들기 위해 machine을 Bay 단위로 나눈다."""

    result: Dict[str, Dict[str, object]] = {}
    for machine_id, machine in machines.items():
        bay_id = machine_bay_ids.get(str(machine_id))
        if bay_id in (None, ""):
            print(f"[ERROR][Phase2.merged._machines_by_bay] cause=missing_machine_bay machine_id={machine_id}")
            raise RuntimeError(f"missing Bay for machine: {machine_id}")
        result.setdefault(str(bay_id), {})[str(machine_id)] = machine
    return result


def _jobs_by_phase1_bay(jobs: Mapping[str, object], phase1_assignments: Mapping[str, str]) -> Dict[str, Dict[str, object]]:
    """Phase 1 block->Bay 결과에 따라 W/O를 Bay별 merged Phase 2 subproblem으로 분할한다."""

    result: Dict[str, Dict[str, object]] = {}
    for job_id, job in jobs.items():
        block_set_id = _job_block_set_id(job, str(job_id))
        if block_set_id not in phase1_assignments:
            print(
                "[ERROR][Phase2.merged._jobs_by_phase1_bay] "
                f"cause=missing_phase1_assignment job_id={job_id} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"missing Phase 1 assignment for block: {block_set_id}")
        bay_id = str(phase1_assignments[block_set_id])
        if not bay_id:
            print(
                "[ERROR][Phase2.merged._jobs_by_phase1_bay] "
                f"cause=empty_phase1_bay job_id={job_id} block_set_id={block_set_id}"
            )
            raise RuntimeError(f"empty Phase 1 Bay assignment for block: {block_set_id}")
        result.setdefault(bay_id, {})[str(job_id)] = job
    return result


def _job_block_set_id(job: object, job_id: str) -> str:
    block_set_id = job.get("block_set_id") if isinstance(job, Mapping) else getattr(job, "block_set_id", None)
    if block_set_id in (None, ""):
        print(f"[ERROR][Phase2.merged._job_block_set_id] cause=missing_block_set_id job_id={job_id}")
        raise RuntimeError(f"job has no block_set_id: {job_id}")
    return str(block_set_id)


def _processing_time_for_job(
    job: object,
    job_id: str,
    dispatch_cache: _Phase2DispatchCache | None = None,
    processing_time_by_job: Mapping[str, float] | None = None,
) -> float:
    if processing_time_by_job is not None and str(job_id) in processing_time_by_job:
        return float(processing_time_by_job[str(job_id)])
    if dispatch_cache is not None and str(job_id) in dispatch_cache.processing_time_by_job:
        return float(dispatch_cache.processing_time_by_job[str(job_id)])
    return float(_processing_time(job))


def _cut_length_for_job(job: object, job_id: str, dispatch_cache: _Phase2DispatchCache | None = None) -> float:
    if dispatch_cache is not None and str(job_id) in dispatch_cache.cut_length_by_job:
        return float(dispatch_cache.cut_length_by_job[str(job_id)])
    return float(_required_non_negative(job, "cut_length"))


def _bevel_quantity_for_job(job: object, job_id: str, dispatch_cache: _Phase2DispatchCache | None = None) -> float:
    if dispatch_cache is not None and str(job_id) in dispatch_cache.bevel_quantity_by_job:
        return float(dispatch_cache.bevel_quantity_by_job[str(job_id)])
    return float(_required_non_negative(job, "bevel_quantity"))


def _job_length(job: object, job_id: str, dispatch_cache: _Phase2DispatchCache | None = None) -> float:
    if dispatch_cache is not None and str(job_id) in dispatch_cache.plate_length_by_job:
        return float(dispatch_cache.plate_length_by_job[str(job_id)])
    length = _required_non_negative(job, "plate_length")
    if length <= 0:
        print(f"[ERROR][Phase2.merged._job_length] cause=non_positive_plate_length job_id={job_id} value={length}")
        raise RuntimeError(f"non-positive plate_length for {job_id}")
    return float(length)


def _episode_jobs(
    episode: int,
    fixed_jobs: Mapping[str, object],
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_job_factory: Callable[[int], Mapping[str, object]] | None,
) -> Mapping[str, object]:
    if episode_job_factory is not None:
        return episode_job_factory(episode)
    if episode_jobs is None:
        return fixed_jobs
    if not episode_jobs:
        print("[ERROR][Phase2.merged._episode_jobs] cause=empty_episode_jobs")
        raise RuntimeError("episode_jobs must not be empty")
    return episode_jobs[(episode - 1) % len(episode_jobs)]


def _validation_episode_jobs(
    validation_episode: int,
    fixed_jobs: Mapping[str, object],
    validation_episode_jobs: Sequence[Mapping[str, object]] | None,
    validation_episode_job_factory: Callable[[int], Mapping[str, object]] | None,
) -> Mapping[str, object]:
    """Return validation jobs; no holdout list means validate the current episode jobs."""

    if validation_episode_job_factory is not None:
        return validation_episode_job_factory(validation_episode)
    if validation_episode_jobs is None:
        return fixed_jobs
    if not validation_episode_jobs:
        print("[ERROR][Phase2.merged._validation_episode_jobs] cause=empty_validation_episode_jobs")
        raise RuntimeError("validation_episode_jobs must not be empty")
    return validation_episode_jobs[(validation_episode - 1) % len(validation_episode_jobs)]


def _phase1_assignments_for_episode(
    jobs: Mapping[str, object],
    fixed_assignments: Mapping[str, str],
    phase1_heuristic: str | None,
    phase1_bay_ids: Sequence[str] | None,
    phase1_assignment_builder: Callable[[Mapping[str, object], int], Mapping[str, str]] | None,
    phase1_assignment_seed: int,
    phase1_bay_capacity_weights: Mapping[str, int | float] | None,
) -> Mapping[str, str]:
    if phase1_assignment_builder is not None:
        if fixed_assignments or phase1_heuristic is not None:
            print("[ERROR][Phase2.merged._phase1_assignments_for_episode] cause=conflicting_phase1_assignment_sources")
            raise RuntimeError("use Phase 1 assignment builder without fixed assignments or phase1_heuristic")
        return phase1_assignment_builder(jobs, phase1_assignment_seed)
    if phase1_heuristic is None:
        return fixed_assignments
    if fixed_assignments:
        print("[ERROR][Phase2.merged._phase1_assignments_for_episode] cause=conflicting_phase1_assignment_sources")
        raise RuntimeError("use either phase1_heuristic or explicit Phase 1 assignments, not both")
    if not phase1_bay_ids:
        print("[ERROR][Phase2.merged._phase1_assignments_for_episode] cause=no_phase1_bay_ids")
        raise RuntimeError("phase1_bay_ids are required when phase1_heuristic is used")
    candidate = run_phase1_resource_pool_heuristic_candidate(
        jobs=jobs,
        bay_ids=tuple(str(bay_id) for bay_id in phase1_bay_ids),
        algorithm=phase1_heuristic,
        bay_capacity_weights=phase1_bay_capacity_weights,
    )
    return candidate.assignments


def _validate_candidate_inputs(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    max_wo_count: int,
    max_length_sum: float,
) -> None:
    if not jobs:
        print("[ERROR][Phase2.merged._validate_candidate_inputs] cause=no_jobs")
        raise RuntimeError("merged Phase 2 requires jobs")
    if not machines:
        print("[ERROR][Phase2.merged._validate_candidate_inputs] cause=no_machines")
        raise RuntimeError("merged Phase 2 requires machines")
    if not phase1_assignments:
        print("[ERROR][Phase2.merged._validate_candidate_inputs] cause=no_phase1_assignments")
        raise RuntimeError("merged Phase 2 requires Phase 1 block->Bay assignments")
    if max_wo_count <= 0:
        print(f"[ERROR][Phase2.merged._validate_candidate_inputs] cause=invalid_max_wo_count value={max_wo_count}")
        raise ValueError("max_wo_count must be positive")
    if max_length_sum <= 0:
        print(f"[ERROR][Phase2.merged._validate_candidate_inputs] cause=invalid_max_length_sum value={max_length_sum}")
        raise ValueError("max_length_sum must be positive")


def _validate_action_pool_limit(action_pool_limit: int | None) -> None:
    """Validate per-machine W/O pool size used before batch combination."""

    if action_pool_limit is None:
        return
    if action_pool_limit <= 0:
        print(f"[ERROR][Phase2.merged._validate_action_pool_limit] cause=invalid_action_pool_limit value={action_pool_limit}")
        raise ValueError("action_pool_limit must be positive")


def _validate_train_inputs(
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    episodes: int,
    lr: float,
    heuristic_algorithms: Sequence[str],
    rollout_samples: int,
    validation_rollout_samples: int,
    validation_every: int,
    validation_episodes: int,
    checkpoint_every: int,
    write_candidate_summary: bool,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    episode_jobs: Sequence[Mapping[str, object]] | None,
    episode_job_factory: Callable[[int], Mapping[str, object]] | None,
    validation_episode_jobs: Sequence[Mapping[str, object]] | None,
    validation_episode_job_factory: Callable[[int], Mapping[str, object]] | None,
    phase1_heuristic: str | None,
    phase1_bay_ids: Sequence[str] | None,
    phase1_assignment_builder: Callable[[Mapping[str, object], int], Mapping[str, str]] | None,
    score_mode: str,
) -> None:
    _score_field_names(score_mode)
    has_fixed_assignments = bool(phase1_assignments)
    phase1_source_count = int(has_fixed_assignments) + int(phase1_heuristic is not None) + int(phase1_assignment_builder is not None)
    if phase1_source_count != 1:
        print(
            "[ERROR][Phase2.merged._validate_train_inputs] "
            f"cause=phase1_source_cardinality fixed_assignments={has_fixed_assignments} "
            f"phase1_heuristic={phase1_heuristic} phase1_assignment_builder={phase1_assignment_builder is not None}"
        )
        raise RuntimeError("merged Phase 2 training requires exactly one Phase 1 assignment source")
    if phase1_heuristic is None and phase1_assignment_builder is None:
        _validate_candidate_inputs(jobs, machines, phase1_assignments, max_wo_count, max_length_sum)
    elif not phase1_bay_ids:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=no_phase1_bay_ids")
        raise RuntimeError("phase1_bay_ids are required when Phase 1 assignments are built per episode")
    _validate_action_pool_limit(action_pool_limit)
    if episodes <= 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_episodes value={episodes}")
        raise ValueError("episodes must be positive")
    if lr <= 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_lr value={lr}")
        raise ValueError("lr must be positive")
    if not heuristic_algorithms:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=no_heuristic_algorithms")
        raise RuntimeError("merged Phase 2 training requires at least one heuristic algorithm")
    unknown = [name for name in heuristic_algorithms if name not in PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK]
    if unknown:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=unknown_heuristic algorithms={unknown}")
        raise RuntimeError(f"unknown merged Phase 2 heuristic algorithm: {unknown[0]}")
    if rollout_samples < 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_rollout_samples value={rollout_samples}")
        raise ValueError("rollout_samples must be non-negative")
    if validation_rollout_samples < 0:
        print(
            "[ERROR][Phase2.merged._validate_train_inputs] "
            f"cause=invalid_validation_rollout_samples value={validation_rollout_samples}"
        )
        raise ValueError("validation_rollout_samples must be non-negative")
    if validation_every <= 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_validation_every value={validation_every}")
        raise ValueError("validation_every must be positive")
    if validation_episodes < 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_validation_episodes value={validation_episodes}")
        raise ValueError("validation_episodes must be non-negative")
    if checkpoint_every < 0:
        print(f"[ERROR][Phase2.merged._validate_train_inputs] cause=invalid_checkpoint_every value={checkpoint_every}")
        raise ValueError("checkpoint_every must be non-negative")
    if not isinstance(write_candidate_summary, bool):
        print(
            "[ERROR][Phase2.merged._validate_train_inputs] "
            f"cause=invalid_write_candidate_summary value={write_candidate_summary}"
        )
        raise ValueError("write_candidate_summary must be a bool")
    if episode_jobs is not None and episode_job_factory is not None:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=conflicting_episode_sources")
        raise RuntimeError("use either episode_jobs or episode_job_factory, not both")
    if validation_episode_jobs is not None and validation_episode_job_factory is not None:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=conflicting_validation_episode_sources")
        raise RuntimeError("use either validation_episode_jobs or validation_episode_job_factory, not both")
    if episode_jobs is not None and not episode_jobs:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=empty_episode_jobs")
        raise RuntimeError("episode_jobs must not be empty")
    if validation_episode_jobs is not None and not validation_episode_jobs:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=empty_validation_episode_jobs")
        raise RuntimeError("validation_episode_jobs must not be empty")


def _resolve_torch_device(device: str) -> torch.device:
    """Resolve an explicit torch device without silently falling back to CPU."""

    requested = str(device or "cpu").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("[ERROR][Phase2.merged._resolve_torch_device] cause=cuda_requested_but_unavailable")
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            print(
                "[ERROR][Phase2.merged._resolve_torch_device] "
                f"cause=cuda_device_requested_but_unavailable device={device}"
            )
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        torch_device = torch.device(requested)
        if torch_device.index is not None and torch_device.index >= torch.cuda.device_count():
            print(
                "[ERROR][Phase2.merged._resolve_torch_device] "
                f"cause=cuda_index_out_of_range device={device} count={torch.cuda.device_count()}"
            )
            raise RuntimeError(f"CUDA device index out of range: {device}")
        return torch_device
    print(f"[ERROR][Phase2.merged._resolve_torch_device] cause=unknown_device device={device}")
    raise RuntimeError(f"unknown torch device: {device}")


def _resolve_phase2_resume_checkpoint(resume_checkpoint: str | Path | None, output_path: Path) -> Path | None:
    """Resolve an explicit checkpoint path or output-dir/checkpoints latest file."""

    if resume_checkpoint is None:
        return None
    value = str(resume_checkpoint).strip()
    if not value:
        return None
    if value.lower() == "latest":
        checkpoint_dir = output_path / "checkpoints"
        if not checkpoint_dir.exists():
            print(
                "[ERROR][Phase2.merged._resolve_phase2_resume_checkpoint] "
                f"cause=missing_checkpoint_dir path={checkpoint_dir}"
            )
            raise RuntimeError(f"missing Phase 2 checkpoint dir: {checkpoint_dir}")
        candidates = list(checkpoint_dir.glob("phase2_batch_machine_policy_ep*.pt"))
        if not candidates:
            print(
                "[ERROR][Phase2.merged._resolve_phase2_resume_checkpoint] "
                f"cause=no_checkpoint_files path={checkpoint_dir}"
            )
            raise RuntimeError(f"no Phase 2 checkpoints in: {checkpoint_dir}")
        return max(candidates, key=_checkpoint_episode_from_path)
    path = Path(value)
    if not path.exists():
        print(f"[ERROR][Phase2.merged._resolve_phase2_resume_checkpoint] cause=missing_checkpoint path={path}")
        raise RuntimeError(f"missing Phase 2 checkpoint: {path}")
    return path


def _checkpoint_episode_from_path(path: Path) -> int:
    marker = "_ep"
    stem = path.stem
    if marker not in stem:
        print(f"[ERROR][Phase2.merged._checkpoint_episode_from_path] cause=invalid_checkpoint_name path={path}")
        raise RuntimeError(f"invalid Phase 2 checkpoint name: {path}")
    suffix = stem.rsplit(marker, 1)[1]
    try:
        return int(suffix)
    except ValueError as exc:
        print(
            "[ERROR][Phase2.merged._checkpoint_episode_from_path] "
            f"cause=invalid_checkpoint_episode path={path} suffix={suffix}"
        )
        raise RuntimeError(f"invalid Phase 2 checkpoint episode in: {path}") from exc


def _load_phase2_batch_machine_checkpoint(
    model: Phase2SetPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
    score_mode: str,
    torch_device: torch.device,
    expected_run_spec: Mapping[str, object],
) -> int:
    """Load Phase 2 model/optimizer state and return completed episode."""

    checkpoint = torch.load(path, map_location=torch_device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        print(f"[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] cause=invalid_payload path={path}")
        raise RuntimeError(f"invalid Phase 2 checkpoint payload: {path}")
    if checkpoint.get("policy_type") != PHASE2_SET_POINTER_POLICY_TYPE:
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=policy_type_mismatch checkpoint={checkpoint.get('policy_type')} "
            f"expected={PHASE2_SET_POINTER_POLICY_TYPE} path={path}"
        )
        raise RuntimeError("Phase 2 checkpoint policy type mismatch")
    if checkpoint.get("feature_schema_version") != PHASE2_STATE_SCHEMA_VERSION:
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=feature_schema_version_mismatch checkpoint={checkpoint.get('feature_schema_version')} "
            f"expected={PHASE2_STATE_SCHEMA_VERSION} path={path}"
        )
        raise RuntimeError("Phase 2 checkpoint feature schema version mismatch")
    if checkpoint.get("feature_schema") != phase2_state_feature_schema():
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=feature_schema_mismatch path={path}"
        )
        raise RuntimeError("Phase 2 checkpoint feature schema mismatch")
    if int(checkpoint.get("hidden_dim", -1)) != hidden_dim:
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=hidden_dim_mismatch checkpoint={checkpoint.get('hidden_dim')} requested={hidden_dim}"
        )
        raise RuntimeError("Phase 2 checkpoint hidden_dim mismatch")
    checkpoint_score_mode = _checkpoint_score_mode(checkpoint, path)
    if checkpoint_score_mode != score_mode:
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=score_mode_mismatch checkpoint={checkpoint_score_mode} requested={score_mode}"
        )
        raise RuntimeError("Phase 2 checkpoint score_mode mismatch")
    checkpoint_run_spec = checkpoint.get("run_spec")
    if not isinstance(checkpoint_run_spec, Mapping):
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=missing_run_spec path={path}"
        )
        raise RuntimeError("Phase 2 resume checkpoint missing RunSpec")
    require_matching_phase2_run_spec(
        checkpoint_run_spec,
        expected_run_spec,
        context="resume_training",
    )
    try:
        completed_episode = int(checkpoint["episodes"])
    except KeyError as exc:
        print(f"[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] cause=missing_episodes path={path}")
        raise RuntimeError("Phase 2 checkpoint missing episodes") from exc

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" not in checkpoint:
        print(
            "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
            f"cause=missing_optimizer_state path={path}"
        )
        raise RuntimeError("Phase 2 resume checkpoint missing optimizer state")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        "[CHECK][Phase2.merged._load_phase2_batch_machine_checkpoint] "
        f"path={path} completed_episode={completed_episode} score_mode={checkpoint_score_mode}"
    )
    return completed_episode


def _checkpoint_score_mode(checkpoint: Mapping, path: Path) -> str:
    mode = checkpoint.get("score_mode")
    if mode:
        return str(mode)
    print(f"[ERROR][Phase2.merged._checkpoint_score_mode] cause=missing_score_mode path={path}")
    raise RuntimeError("Phase 2 checkpoint missing score_mode")


def _prepare_resume_csv(path: Path, fields: Sequence[str], episode_field: str, completed_episode: int) -> list[dict]:
    """Align an existing CSV to the checkpoint episode and current schema."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    kept_rows: list[dict] = []
    dropped = 0
    for row in rows:
        try:
            episode = int(row[episode_field])
        except (KeyError, ValueError) as exc:
            print(
                "[ERROR][Phase2.merged._prepare_resume_csv] "
                f"cause=invalid_episode_field path={path} field={episode_field} row={row}"
            )
            raise RuntimeError(f"invalid resume CSV episode field: {path}") from exc
        if episode <= completed_episode:
            kept_rows.append(row)
        else:
            dropped += 1
    if kept_rows:
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept_rows)
    else:
        path.unlink()
    print(
        "[CHECK][Phase2.merged._prepare_resume_csv] "
        f"path={path} kept={len(kept_rows)} dropped_after_checkpoint={dropped} completed_episode={completed_episode}"
    )
    return kept_rows


def _save_phase2_batch_machine_checkpoint(
    model: Phase2SetPointerPolicy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    hidden_dim: int,
    heuristic_algorithms: Sequence[str],
    rollout_samples: int,
    validation_rollout_samples: int,
    validation_every: int,
    validation_episodes: int,
    episodes: int,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    device: torch.device,
    score_mode: str,
    run_spec: Mapping[str, object],
) -> None:
    """Save Phase 2 merged policy checkpoint with the feature contract."""

    score_field_names = _score_field_names(score_mode)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "policy_type": PHASE2_SET_POINTER_POLICY_TYPE,
            "feature_schema_version": PHASE2_STATE_SCHEMA_VERSION,
            "feature_schema": phase2_state_feature_schema(),
            "hidden_dim": hidden_dim,
            "score_mode": score_mode,
            "score_fields": list(score_field_names),
            "raw_score_fields": list(PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES),
            "normalized_score_fields": list(PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES),
            "heuristic_algorithms": list(heuristic_algorithms),
            "rollout_samples": rollout_samples,
            "validation_rollout_samples": validation_rollout_samples,
            "validation_every": validation_every,
            "validation_episodes": validation_episodes,
            "episodes": episodes,
            "max_wo_count": max_wo_count,
            "max_length_sum": max_length_sum,
            "action_pool_limit": action_pool_limit,
            "device": str(device),
            "run_spec": dict(run_spec),
        },
        path,
    )


def _candidate_summary_row(
    episode: int,
    rank: int,
    candidate: Phase2BatchMachineCandidate,
    score_field_names: Sequence[str],
) -> Dict:
    row = {
        "episode": episode,
        "rank": rank,
        "bay_id": candidate.subproblem_bay_id,
        "source": candidate.source,
        "selection_count": len(candidate.transitions),
        "batch_count": len(candidate.batches),
        "assignment_count": len(candidate.machine_assignments),
        "score_json": json.dumps(list(candidate.score_tuple), ensure_ascii=False),
    }
    row.update({field: candidate.score_tuple[index] for index, field in enumerate(score_field_names)})
    row.update(candidate.score_details)
    return row


def _subproblem_metrics_row(
    episode: int,
    bay_id: str,
    job_count: int,
    machine_count: int,
    candidate_count: int,
    best: Phase2BatchMachineCandidate,
    loss: float,
    score_field_names: Sequence[str],
) -> Dict:
    row = {
        "episode": episode,
        "bay_id": bay_id,
        "job_count": job_count,
        "machine_count": machine_count,
        "candidate_count": candidate_count,
        "best_source": best.source,
        "selection_count": len(best.transitions),
        "loss": round(loss, 9),
        "score_json": json.dumps(list(best.score_tuple), ensure_ascii=False),
    }
    row.update({field: best.score_tuple[index] for index, field in enumerate(score_field_names)})
    row.update(best.score_details)
    return row


def _validation_summary_row(
    train_episode: int,
    validation_episode: int,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    candidate_count: int,
    best: Phase2BatchMachineCandidate,
    proposed_rank: int,
    proposed_best: Phase2BatchMachineCandidate | None,
    greedy_rank: int,
    greedy_best: Phase2BatchMachineCandidate | None,
    score_field_names: Sequence[str],
) -> Dict:
    row = {
        "train_episode": train_episode,
        "validation_episode": validation_episode,
        "job_count": len(jobs),
        "machine_count": len(machines),
        "candidate_count": candidate_count,
        "best_source": best.source,
        "best_score_json": json.dumps(list(best.score_tuple), ensure_ascii=False),
        "agent_best_rank": proposed_rank,
        "agent_best_source": "" if proposed_best is None else proposed_best.source,
        "agent_best_score_json": "" if proposed_best is None else json.dumps(list(proposed_best.score_tuple), ensure_ascii=False),
        "proposed_best_rank": proposed_rank,
        "proposed_best_source": "" if proposed_best is None else proposed_best.source,
        "proposed_best_score_json": "" if proposed_best is None else json.dumps(list(proposed_best.score_tuple), ensure_ascii=False),
        "greedy_rank": greedy_rank,
        "greedy_source": "" if greedy_best is None else greedy_best.source,
        "greedy_score_json": "" if greedy_best is None else json.dumps(list(greedy_best.score_tuple), ensure_ascii=False),
    }
    row.update({field: best.score_tuple[index] for index, field in enumerate(score_field_names)})
    row.update(best.score_details)
    if proposed_best is None:
        row.update({f"agent_{field}": "" for field in score_field_names})
        row.update({f"proposed_{field}": "" for field in score_field_names})
    else:
        row.update({f"agent_{field}": proposed_best.score_tuple[index] for index, field in enumerate(score_field_names)})
        row.update({f"proposed_{field}": proposed_best.score_tuple[index] for index, field in enumerate(score_field_names)})
    if greedy_best is None:
        row.update({f"greedy_{field}": "" for field in score_field_names})
    else:
        row.update({f"greedy_{field}": greedy_best.score_tuple[index] for index, field in enumerate(score_field_names)})
    return row


def _validation_candidate_summary_row(
    train_episode: int,
    validation_episode: int,
    rank: int,
    candidate: Phase2BatchMachineCandidate,
    score_field_names: Sequence[str],
) -> Dict:
    row = _candidate_summary_row(validation_episode, rank, candidate, score_field_names)
    row["train_episode"] = train_episode
    row["validation_episode"] = validation_episode
    row.pop("episode", None)
    return row


def _metrics_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "episode",
        "job_count",
        "machine_count",
        "selection_count",
        "best_source",
        "candidate_count",
        "loss",
        "score_json",
        *score_field_names,
        *PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES,
    ]


def _subproblem_metrics_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "episode",
        "bay_id",
        "job_count",
        "machine_count",
        "candidate_count",
        "best_source",
        "selection_count",
        "loss",
        "score_json",
        *score_field_names,
        *PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES,
    ]


def _candidate_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "episode",
        "rank",
        "bay_id",
        "source",
        "selection_count",
        "batch_count",
        "assignment_count",
        "score_json",
        *score_field_names,
        *PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES,
    ]


def _validation_summary_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "train_episode",
        "validation_episode",
        "job_count",
        "machine_count",
        "candidate_count",
        "best_source",
        "best_score_json",
        "agent_best_rank",
        "agent_best_source",
        "agent_best_score_json",
        "proposed_best_rank",
        "proposed_best_source",
        "proposed_best_score_json",
        "greedy_rank",
        "greedy_source",
        "greedy_score_json",
        *score_field_names,
        *PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES,
        *[f"agent_{field}" for field in score_field_names],
        *[f"proposed_{field}" for field in score_field_names],
        *[f"greedy_{field}" for field in score_field_names],
    ]


def _validation_candidate_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "train_episode",
        "validation_episode",
        "rank",
        "bay_id",
        "source",
        "selection_count",
        "batch_count",
        "assignment_count",
        "score_json",
        *score_field_names,
        *PHASE2_BATCH_MACHINE_SCORE_DETAIL_FIELD_NAMES,
    ]


def _write_validation_plots(output_path: Path, candidate_rows: Sequence[Mapping], summary_rows: Sequence[Mapping]) -> Dict[str, str]:
    """Write Phase 2 validation PNGs for the latest validation checkpoint."""

    if not candidate_rows:
        return {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[ERROR][Phase2.merged._write_validation_plots] cause=matplotlib_import_failed error={exc}")
        raise RuntimeError("matplotlib is required to write Phase 2 validation plots") from exc

    latest_candidate_rows = _collapse_agent_samples_for_validation_plot(_latest_rows(candidate_rows, "train_episode"))
    latest_summary_rows = _latest_rows(summary_rows, "train_episode")
    plot_paths: Dict[str, str] = {}
    wo_gap_field = _available_metric_field(
        latest_candidate_rows,
        "bay_internal_wo_count_gap",
        "bay_internal_normalized_wo_count_gap",
    )
    cut_gap_field = _available_metric_field(
        latest_candidate_rows,
        "bay_internal_cut_length_gap",
        "bay_internal_normalized_cut_length_gap",
    )
    bevel_gap_field = _available_metric_field(
        latest_candidate_rows,
        "bay_internal_bevel_quantity_gap",
        "bay_internal_normalized_bevel_quantity_gap",
    )
    occupancy_gap_field = _available_metric_field(
        latest_candidate_rows,
        "bay_internal_occupancy_gap",
        "bay_internal_normalized_occupancy_gap",
    )
    plot_specs = [
        ("validation_hard_violation_png", "validation_hard_violation.png", "hard_violation_count", "Hard violation count", "Count"),
        ("validation_makespan_png", "validation_makespan.png", "makespan", "Makespan", "Minutes"),
        ("validation_wo_gap_png", "validation_wo_gap.png", wo_gap_field, "Bay-internal W/O count gap", "Gap"),
        ("validation_cut_gap_png", "validation_cut_gap.png", cut_gap_field, "Bay-internal cut length gap", "Gap"),
        ("validation_bevel_gap_png", "validation_bevel_gap.png", bevel_gap_field, "Bay-internal bevel quantity gap", "Gap"),
        ("validation_occupancy_gap_png", "validation_occupancy_gap.png", occupancy_gap_field, "Bay-internal occupancy gap", "Gap"),
    ]
    for output_key, filename, score_field, title, ylabel in plot_specs:
        path = output_path / filename
        _plot_validation_metric(plt, path, latest_candidate_rows, score_field, title, ylabel)
        plot_paths[output_key] = str(path)

    best_source_path = output_path / "validation_best_source_counts.png"
    _plot_validation_best_source_counts(plt, best_source_path, latest_summary_rows)
    plot_paths["validation_best_source_counts_png"] = str(best_source_path)

    rank_path = output_path / "validation_policy_rank.png"
    _plot_validation_policy_rank(plt, rank_path, latest_summary_rows)
    plot_paths["validation_policy_rank_png"] = str(rank_path)
    return plot_paths


def _available_metric_field(rows: Sequence[Mapping], raw_field: str, normalized_field: str) -> str:
    if not rows:
        return raw_field
    sample = rows[0]
    if raw_field in sample:
        return raw_field
    if normalized_field in sample:
        return normalized_field
    print(
        "[ERROR][Phase2.merged._available_metric_field] "
        f"cause=missing_metric_field raw={raw_field} normalized={normalized_field}"
    )
    raise RuntimeError(f"missing Phase 2 validation metric field: {raw_field}")


def _latest_rows(rows: Sequence[Mapping], episode_field: str) -> List[Mapping]:
    """Return rows from the latest train episode."""

    if not rows:
        return []
    latest_episode = max(int(row[episode_field]) for row in rows)
    return [row for row in rows if int(row[episode_field]) == latest_episode]


def _collapse_agent_samples_for_validation_plot(rows: Sequence[Mapping]) -> List[Mapping]:
    """Collapse raw Phase 2 agent validation rows into one best-of-K row per validation episode."""

    non_agent_rows: List[Mapping] = []
    best_agent_rows: Dict[str, Mapping] = {}
    for row in rows:
        source = str(row["source"])
        if not _is_agent_source(source):
            non_agent_rows.append(row)
            continue
        key = str(row["validation_episode"])
        current_best = best_agent_rows.get(key)
        if current_best is None or _phase2_plot_score_key(row) < _phase2_plot_score_key(current_best):
            best_agent_rows[key] = row
    collapsed_agent_rows: List[Mapping] = []
    for row in best_agent_rows.values():
        collapsed = dict(row)
        collapsed["source"] = PHASE2_PROPOSED_BEST_OF_K_SOURCE
        collapsed_agent_rows.append(collapsed)
    return non_agent_rows + collapsed_agent_rows


def _phase2_plot_score_key(row: Mapping) -> tuple[float, ...]:
    """Parse the Phase 2 score tuple used to choose the visible best-of-K point."""

    value = row.get("score_json")
    if value in (None, ""):
        print("[ERROR][Phase2.merged._phase2_plot_score_key] cause=missing_score_json")
        raise RuntimeError("Phase 2 validation plot row has no score_json")
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        print(f"[ERROR][Phase2.merged._phase2_plot_score_key] cause=invalid_score_json value={value}")
        raise RuntimeError("invalid Phase 2 validation plot score_json") from exc
    return tuple(float(item) for item in parsed)


def _plot_validation_metric(plt, path: Path, rows: Sequence[Mapping], score_field: str, title: str, ylabel: str) -> None:
    """Scatter one Phase 2 validation metric by validation episode and method."""

    ordered_rows = sorted(rows, key=lambda row: (int(row["validation_episode"]), str(row["source"])))
    validation_episodes = sorted({int(row["validation_episode"]) for row in ordered_rows})
    episode_index = {episode: index + 1 for index, episode in enumerate(validation_episodes)}
    sources = _ordered_sources(row["source"] for row in ordered_rows)
    markers = ["D", "s", "^", "o", "v", "P", "X", "*", "h", "p"]
    plt.figure(figsize=(11, 5))
    for source_index, source in enumerate(sources):
        source_rows = [row for row in ordered_rows if str(row["source"]) == source]
        x_values = [episode_index[int(row["validation_episode"])] for row in source_rows]
        y_values = [float(row[score_field]) for row in source_rows]
        plt.scatter(
            x_values,
            y_values,
            label=_display_source_name(source),
            marker=markers[source_index % len(markers)],
            s=42,
            alpha=0.85,
        )
    plt.xlabel("Validation episode")
    plt.ylabel(ylabel)
    plt.title(f"{title} by method (lower is better)")
    plt.xticks(list(episode_index.values()), [f"VAL{episode:02d}" for episode in validation_episodes], rotation=45, ha="right")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_best_source_counts(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot which source wins each latest Phase 2 validation problem."""

    counts: Dict[str, int] = {}
    for row in rows:
        source = _plot_source_key(str(row["best_source"]))
        counts[source] = counts.get(source, 0) + 1
    sources = _ordered_sources(counts)
    plt.figure(figsize=(9, 4))
    plt.bar([_display_source_name(source) for source in sources], [counts[source] for source in sources])
    plt.xlabel("Best source")
    plt.ylabel("Validation wins")
    plt.title("Best source counts in latest Phase 2 validation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_policy_rank(plt, path: Path, rows: Sequence[Mapping]) -> None:
    """Plot Proposed(best-of-K) rank and greedy-policy rank separately."""

    ordered_rows = sorted(rows, key=lambda row: int(row["validation_episode"]))
    x_values = list(range(1, len(ordered_rows) + 1))
    labels = [f"VAL{int(row['validation_episode']):02d}" for row in ordered_rows]
    proposed_ranks = [int(row["proposed_best_rank"]) for row in ordered_rows]
    greedy_ranks = [int(row["greedy_rank"]) for row in ordered_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x_values, proposed_ranks, marker="D", label="Proposed(best-of-K)")
    plt.plot(x_values, greedy_ranks, marker="o", label="Proposed(greedy)")
    plt.xlabel("Validation episode")
    plt.ylabel("Rank")
    plt.title("Policy rank in latest Phase 2 validation (1 is best)")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _ordered_sources(sources: Sequence[object]) -> List[str]:
    """Return stable Phase 2 method order with learned policy first."""

    unique = {str(source) for source in sources}
    preferred = [PHASE2_PROPOSED_BEST_OF_K_SOURCE, "agent_greedy", *PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK]
    return [source for source in preferred if source in unique] + sorted(unique - set(preferred))


def _display_source_name(source: str) -> str:
    """Shorten Phase 2 source names for plot legends."""

    names = {
        PHASE2_PROPOSED_BEST_OF_K_SOURCE: "Proposed(best-of-K)",
        "agent_greedy": "Proposed(greedy)",
        "workload_makespan_dispatch": "WorkloadMakespan",
        "min_makespan": "MinMakespan",
        "lookahead_min_makespan": "LookaheadMinMakespan",
        "best_fit_lth": "BestFitLTH",
        "spt_batch": "SPTBatch",
        "lpt_batch": "LPTBatch",
    }
    return names.get(source, source)


def _plot_source_key(source: str) -> str:
    """Return the visible plot source key for raw Phase 2 agent candidates."""

    if _is_agent_source(source):
        return PHASE2_PROPOSED_BEST_OF_K_SOURCE
    return source


def _write_assignments(path: Path, candidate: Phase2BatchMachineCandidate) -> None:
    rows = [
        {"source": candidate.source, "job_id": job_id, "machine_id": machine_id}
        for job_id, machine_id in sorted(candidate.machine_assignments.items())
    ]
    _write_rows(path, rows, ["source", "job_id", "machine_id"])


def _write_dict_rows(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        print(f"[ERROR][Phase2.merged._write_dict_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to write: {path}")
    _write_rows(path, rows, list(rows[0].keys()))


def _write_rows(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    if not rows:
        print(f"[ERROR][Phase2.merged._write_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _append_rows(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    """Append audit rows without rewriting the whole training history."""

    if not rows:
        print(f"[ERROR][Phase2.merged._append_rows] cause=no_rows path={path}")
        raise RuntimeError(f"no rows to append: {path}")
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK",
    "PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES",
    "Phase2BatchMachineCandidate",
    "Phase2BatchMachineTransition",
    "build_phase2_batch_machine_candidate_bank",
    "run_phase2_batch_machine_candidate",
    "train_phase2_batch_machine_self_labeling",
]
