"""Merged Phase 2: machine 선택 뒤 W/O를 순차 선택해 batch를 구성한다.

기존 분리 구조는 먼저 모든 W/O를 machine에 배정한 뒤 machine별 batch를
만들었다. 이 모듈은 중간 고정 배정을 없애되, 조합 폭발을 피하기 위해
환경이 설비를 결정하고 정책이 `SELECT_WO... -> 자동 batch close`로 schedule을 만든다.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Sequence

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
from Phase2.validation_grid import (
    Phase2ValidationProblem,
    phase2_validation_grid_contract,
)
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
from Utils.learning.run_manifest import summarize_validation_contract, write_run_manifest


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
    "long_cut_batch",      # 절단 길이 긴 W/O 우선
    "short_cut_batch",     # 절단 길이 짧은 W/O 우선
    "long_bevel_batch",    # 베벨 길이(BVL_LTH) 긴 W/O 우선
    "short_bevel_batch",   # 베벨 길이 짧은 W/O 우선
)
PHASE2_PROPOSED_BEST_OF_K_SOURCE = "proposed_best_of_k"
PHASE2_VALIDATION_PHASE1_SEED_OFFSET = 20_000_000
PHASE2_VALIDATION_CANDIDATE_SEED_OFFSET = 30_000_000

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
    target_action_indices: tuple[int, ...]
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
    bevel_length_by_job: Dict[str, float]


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
    validation_problems: Sequence[Phase2ValidationProblem] | None = None,
    secondary_validation_problems: Sequence[Phase2ValidationProblem] | None = None,
    device: str = "cpu",
    checkpoint_every: int = 0,
    write_candidate_summary: bool = False,
    score_mode: str = "raw",
    resume_checkpoint: str | Path | None = None,
    eval_only: bool = False,
    constraint_profile: PhaseConstraintProfile | None = None,
    candidate_workers: int = 1,
    candidate_executor: ProcessPoolExecutor | None = None,
    temperature: float = 1.0,
    temperature_min: float | None = None,
    temperature_anneal_episodes: int | None = None,
    validation_temperature: float | None = None,
    run_manifest_fields: Mapping[str, object] | None = None,
) -> Dict:
    """Self-labeling으로 통합 Phase 2 batch-machine policy를 학습한다."""

    _validate_candidate_workers(candidate_workers)
    if candidate_workers == 1 and candidate_executor is not None:
        print(
            "[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] "
            "cause=executor_with_single_worker"
        )
        raise RuntimeError("candidate_executor requires candidate_workers greater than 1")
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
        validation_problems=validation_problems,
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
    resolved_validation_problems = _resolve_validation_problems(
        validation_episodes=validation_episodes,
        default_jobs=jobs,
        validation_episode_jobs=validation_episode_jobs,
        validation_episode_job_factory=validation_episode_job_factory,
        validation_problems=validation_problems,
    )
    validation_contract = phase2_validation_grid_contract(resolved_validation_problems)
    # 부차(GENERALIZATION) validation 문제집합. best-checkpoint 선택/RunSpec 대조에는
    # 절대 참여하지 않으며 오직 리포팅 용도로 validation_generalization/에만 기록한다.
    resolved_secondary_validation_problems = tuple(secondary_validation_problems or ())
    if resolved_secondary_validation_problems:
        secondary_problem_ids = [
            problem.problem_id for problem in resolved_secondary_validation_problems
        ]
        if len(set(secondary_problem_ids)) != len(secondary_problem_ids):
            print(
                "[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                f"cause=duplicate_secondary_problem_ids values={secondary_problem_ids}"
            )
            raise RuntimeError("secondary validation problem IDs must be unique")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_path / "metrics.csv"
    subproblem_metrics_csv = output_path / "subproblem_metrics.csv"
    candidate_summary_csv = output_path / "candidate_summary.csv"
    best_batches_csv = output_path / "best_batches.csv"
    best_timeline_csv = output_path / "best_timeline.csv"
    best_assignment_csv = output_path / "best_machine_assignment.csv"
    validation_root = output_path / "validation"
    validation_summary_csv = validation_root / "validation_bay_history.csv"
    validation_candidate_summary_csv = validation_root / "validation_candidate_latest.csv"
    validation_parent_summary_csv = validation_root / "validation_parent_history.csv"
    generalization_root = output_path / "validation_generalization"
    generalization_summary_csv = generalization_root / "validation_bay_history.csv"
    generalization_candidate_summary_csv = generalization_root / "validation_candidate_latest.csv"
    generalization_parent_summary_csv = generalization_root / "validation_parent_history.csv"
    checkpoint_path = output_path / "phase2_batch_machine_policy.pt"
    best_checkpoint_path = output_path / "phase2_best.pt"
    checkpoint_dir = output_path / "checkpoints"
    summary_json = output_path / "summary.json"
    if checkpoint_every > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if resolved_validation_problems:
        _write_validation_problem_catalog(
            validation_root,
            resolved_validation_problems,
            validation_contract,
        )
    if resolved_secondary_validation_problems:
        _write_validation_problem_catalog(
            generalization_root,
            resolved_secondary_validation_problems,
            phase2_validation_grid_contract(resolved_secondary_validation_problems),
        )

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
            expected_validation_contract=validation_contract,
            eval_only=eval_only,
        )
        if resumed_from_episode >= episodes and not eval_only:
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
        validation_parent_rows = _prepare_resume_csv(
            validation_parent_summary_csv,
            _validation_parent_fields(score_field_names),
            "train_episode",
            resumed_from_episode,
        )
        secondary_validation_rows = _prepare_resume_csv(
            generalization_summary_csv,
            _validation_summary_fields(score_field_names),
            "train_episode",
            resumed_from_episode,
        )
        secondary_validation_parent_rows = _prepare_resume_csv(
            generalization_parent_summary_csv,
            _validation_parent_fields(score_field_names),
            "train_episode",
            resumed_from_episode,
        )
    else:
        for path in (
            metrics_csv,
            subproblem_metrics_csv,
            candidate_summary_csv,
            validation_summary_csv,
            validation_candidate_summary_csv,
            validation_parent_summary_csv,
            generalization_summary_csv,
            generalization_candidate_summary_csv,
            generalization_parent_summary_csv,
        ):
            if path.exists():
                path.unlink()
        if best_checkpoint_path.exists():
            best_checkpoint_path.unlink()
        validation_rows: list[dict] = []
        validation_parent_rows: list[dict] = []
        secondary_validation_rows: list[dict] = []
        secondary_validation_parent_rows: list[dict] = []

    best_validation_key = _resume_validation_checkpoint_key(
        validation_parent_rows,
        best_checkpoint_path,
    )
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
    print(f"- validation_types_per_size: {validation_episodes}")
    print(f"- validation_problem_count: {len(resolved_validation_problems)}")
    print(f"- secondary_validation_problem_count: {len(resolved_secondary_validation_problems)}")
    print(f"- checkpoint_every: {checkpoint_every}")
    print(f"- write_candidate_summary: {write_candidate_summary}")
    print(f"- device: {torch_device}")
    print(f"- candidate_workers: {candidate_workers}")
    resolved_temperature_min = temperature if temperature_min is None else temperature_min
    resolved_temperature_anneal_episodes = episodes if temperature_anneal_episodes is None else temperature_anneal_episodes
    print(f"- temperature: {temperature}")
    print(f"- temperature_min: {resolved_temperature_min}")
    print(f"- temperature_anneal_episodes: {resolved_temperature_anneal_episodes}")
    # validation/eval에서 agent_sample 후보를 뽑는 온도. None이면 기존처럼 학습 스케줄의
    # 현재 온도를 그대로 쓴다. 값을 주면 학습 arm이 달라도 채점 온도는 같아지므로,
    # 서로 다른 temperature 설정을 비교할 때 평가 자체가 교란되지 않는다.
    resolved_validation_temperature = _resolve_validation_temperature(validation_temperature)
    print(
        "- validation_temperature: "
        + (
            "follows_training_schedule"
            if resolved_validation_temperature is None
            else str(resolved_validation_temperature)
        )
    )

    def _validation_sampling_temperature(current_temperature: float) -> float:
        """평가용 sampling 온도를 고정값(지정 시) 또는 학습 스케줄 값으로 정한다."""

        if resolved_validation_temperature is None:
            return current_temperature
        return resolved_validation_temperature

    if run_manifest_fields is not None:
        write_run_manifest(
            output_path,
            cli_fields=run_manifest_fields,
            run_spec=run_spec,
            run_spec_source="phase2_run_spec",
            validation={
                "main": summarize_validation_contract(
                    validation_contract if resolved_validation_problems else None,
                    fixed_grid=bool(resolved_validation_problems),
                    seed=seed,
                    note="phase2 in-distribution grid; drives best-checkpoint selection",
                ),
                "generalization": summarize_validation_contract(
                    phase2_validation_grid_contract(resolved_secondary_validation_problems)
                    if resolved_secondary_validation_problems
                    else None,
                    fixed_grid=bool(resolved_secondary_validation_problems),
                    seed=seed,
                    note="phase2 reporting-only grid",
                ),
                "validation_temperature": resolved_validation_temperature,
                "validation_rollout_samples": resolved_validation_rollout_samples,
                "validation_every": validation_every,
            },
            extra={
                "device": str(torch_device),
                "seed": seed,
                "episodes": episodes,
                "start_episode": start_episode,
                "eval_only": bool(eval_only),
                "resume_checkpoint": str(resume_path) if resume_path is not None else "",
                "resumed_from_episode": resumed_from_episode,
                "temperature": temperature,
                "temperature_min": resolved_temperature_min,
                "temperature_anneal_episodes": resolved_temperature_anneal_episodes,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "score_mode": score_mode,
                "phase1_heuristic": phase1_heuristic or "",
                "candidate_workers": candidate_workers,
            },
        )

    def _run_phase2_validation_cycle(
        *,
        episode: int,
        current_temperature: float,
        problems: Sequence[Phase2ValidationProblem],
        root: Path,
        select_best: bool,
        accumulated_rows: list[dict],
        accumulated_parent_rows: list[dict],
        current_best_validation_key: tuple | None,
    ) -> dict:
        """한 checkpoint의 validation pass를 root 아래에 기록한다.

        select_best=True(=MAIN)만 best_checkpoint.json 기록과 best_checkpoint 저장을
        수행한다. select_best=False(=GENERALIZATION)는 리포팅만 하며 checkpoint 선택,
        best_checkpoint.json, resume-contract 대조에 절대 관여하지 않는다.
        """

        cycle_summary_csv = root / "validation_bay_history.csv"
        cycle_candidate_summary_csv = root / "validation_candidate_latest.csv"
        cycle_parent_summary_csv = root / "validation_parent_history.csv"
        current_validation_rows: list[dict] = []
        current_validation_candidate_rows: list[dict] = []
        current_validation_parent_rows: list[dict] = []
        current_validation_method_rows: list[dict] = []
        current_validation_bay_method_rows: list[dict] = []
        evaluation_root = root / "evaluations" / f"ep_{episode:06d}"
        for validation_episode, validation_problem in enumerate(
            problems,
            start=1,
        ):
            validation_jobs = validation_problem.jobs
            validation_phase1_seed = (
                seed
                + PHASE2_VALIDATION_PHASE1_SEED_OFFSET
                + validation_episode
            )
            validation_phase1 = _phase1_assignments_for_episode(
                jobs=validation_jobs,
                fixed_assignments=phase1_assignments,
                phase1_heuristic=phase1_heuristic,
                phase1_bay_ids=phase1_bay_ids,
                phase1_assignment_builder=phase1_assignment_builder,
                phase1_assignment_seed=validation_phase1_seed,
                phase1_bay_capacity_weights=phase1_bay_capacity_weights,
            )
            validation_seed = (
                seed
                + PHASE2_VALIDATION_CANDIDATE_SEED_OFFSET
                + validation_episode * 100_000
            )
            candidate_banks_by_bay: dict[
                str,
                tuple[
                    Mapping[str, object],
                    Mapping[str, object],
                    int,
                    list[Phase2BatchMachineCandidate],
                ],
            ] = {}
            for (
                bay_id,
                bay_jobs,
                bay_machines,
                bay_seed,
                validation_candidates,
            ) in _iter_phase2_bay_candidate_banks(
                jobs=validation_jobs,
                machines=machines,
                phase1_assignments=validation_phase1,
                model=model,
                heuristic_algorithms=heuristic_algorithms,
                rollout_samples=resolved_validation_rollout_samples,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                seed=validation_seed,
                score_mode=score_mode,
                constraint_profile=resolved_constraint_profile,
                candidate_workers=candidate_workers,
                candidate_executor=candidate_executor,
                temperature=current_temperature,
            ):
                ranked_candidates = sorted(validation_candidates, key=_candidate_sort_key)
                candidate_banks_by_bay[bay_id] = (
                    bay_jobs,
                    bay_machines,
                    bay_seed,
                    ranked_candidates,
                )
                best_validation = ranked_candidates[0]
                proposed_rank, proposed_best = _agent_best_from_ranked(ranked_candidates)
                greedy_rank, greedy_best = _agent_greedy_from_ranked(ranked_candidates)
                best_heuristic = _best_heuristic_from_ranked(ranked_candidates)
                validation_row = _validation_summary_row(
                    train_episode=episode,
                    validation_episode=validation_episode,
                    validation_phase1_seed=validation_phase1_seed,
                    validation_sampling_seed=bay_seed,
                    bay_id=bay_id,
                    jobs=bay_jobs,
                    machines=bay_machines,
                    candidate_count=len(ranked_candidates),
                    best=best_validation,
                    best_heuristic=best_heuristic,
                    proposed_rank=proposed_rank,
                    proposed_best=proposed_best,
                    greedy_rank=greedy_rank,
                    greedy_best=greedy_best,
                    score_field_names=score_field_names,
                )
                validation_row.update(
                    _validation_problem_columns(
                        validation_problem,
                    )
                )
                current_validation_rows.append(validation_row)
                for rank, candidate in enumerate(ranked_candidates, start=1):
                    candidate_row = _validation_candidate_summary_row(
                        episode,
                        validation_episode,
                        bay_seed,
                        rank,
                        candidate,
                        score_field_names,
                    )
                    candidate_row.update(
                        _validation_problem_columns(
                            validation_problem,
                        )
                    )
                    current_validation_candidate_rows.append(candidate_row)
                current_validation_bay_method_rows.extend(
                    _validation_bay_method_rows(
                        train_episode=episode,
                        validation_episode=validation_episode,
                        validation_problem=validation_problem,
                        bay_id=bay_id,
                        ranked_candidates=ranked_candidates,
                        heuristic_algorithms=heuristic_algorithms,
                        score_field_names=score_field_names,
                    )
                )

            parent_candidates = _validation_parent_candidates(
                candidate_banks_by_bay=candidate_banks_by_bay,
                machines=machines,
                heuristic_algorithms=heuristic_algorithms,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                score_mode=score_mode,
            )
            parent_row, method_rows = _validation_parent_rows(
                train_episode=episode,
                validation_episode=validation_episode,
                validation_problem=validation_problem,
                parent_candidates=parent_candidates,
                score_field_names=score_field_names,
            )
            current_validation_parent_rows.append(parent_row)
            current_validation_method_rows.extend(method_rows)
            _write_validation_problem_evaluation(
                evaluation_root=evaluation_root,
                validation_problem=validation_problem,
                phase1_assignments=validation_phase1,
                machines=machines,
                candidate_banks_by_bay=candidate_banks_by_bay,
                parent_candidates=parent_candidates,
                score_field_names=score_field_names,
            )

        validation_key = _validation_checkpoint_key(current_validation_parent_rows)
        if select_best:
            checkpoint_saved = (
                current_best_validation_key is None
                or validation_key < current_best_validation_key
            )
        else:
            checkpoint_saved = False
        selection_key_json = json.dumps(list(validation_key), ensure_ascii=False)
        for validation_row in (
            *current_validation_rows,
            *current_validation_parent_rows,
        ):
            validation_row["checkpoint_selection_key_json"] = selection_key_json
            validation_row["best_checkpoint_saved"] = int(checkpoint_saved)
        accumulated_rows.extend(current_validation_rows)
        accumulated_parent_rows.extend(current_validation_parent_rows)
        _write_rows(cycle_summary_csv, accumulated_rows, _validation_summary_fields(score_field_names))
        _write_rows(
            cycle_candidate_summary_csv,
            current_validation_candidate_rows,
            _validation_candidate_fields(score_field_names),
        )
        _write_rows(
            cycle_parent_summary_csv,
            accumulated_parent_rows,
            _validation_parent_fields(score_field_names),
        )
        cycle_plot_paths = _write_validation_graph_hierarchy(
            evaluation_root=evaluation_root,
            parent_problem_rows=current_validation_parent_rows,
            parent_method_rows=current_validation_method_rows,
            bay_method_rows=current_validation_bay_method_rows,
            score_field_names=score_field_names,
        )
        _write_json(
            root / "latest_evaluation.json",
            {
                "train_episode": episode,
                "evaluation_root": str(evaluation_root),
                "selection_key": list(validation_key),
            },
        )
        new_best_validation_key = current_best_validation_key
        if select_best and checkpoint_saved:
            _save_phase2_batch_machine_checkpoint(
                model=model,
                optimizer=optimizer,
                path=best_checkpoint_path,
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
                validation_contract=validation_contract,
            )
            new_best_validation_key = validation_key
            _write_json(
                root / "best_checkpoint.json",
                {
                    "train_episode": episode,
                    "checkpoint_path": str(best_checkpoint_path),
                    "selection_key": list(validation_key),
                    "parent_problem_count": len(current_validation_parent_rows),
                },
            )
        role = "main" if select_best else "generalization"
        print(
            "[VALIDATION][Phase2.merged.train_phase2_batch_machine_self_labeling] "
            f"role={role} episode={episode} validation_problems={len(problems)} "
            f"validation_rollout_samples={resolved_validation_rollout_samples} "
            f"selection_key={selection_key_json} best_checkpoint_saved={str(checkpoint_saved).lower()}"
        )
        return {
            "current_parent_rows": current_validation_parent_rows,
            "plot_paths": cycle_plot_paths,
            "best_validation_key": new_best_validation_key,
            "checkpoint_saved": checkpoint_saved,
        }

    if eval_only:
        # 순수 평가: 이미 학습된 checkpoint를 로드해 고정 validation 문제로 1회 평가만 한다.
        # 학습 루프도 optimizer step도 없다(가중치 불변). best-checkpoint 선택/저장도 하지 않는다.
        if resume_path is None:
            print(
                "[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                "cause=eval_only_requires_resume_checkpoint"
            )
            raise RuntimeError("eval_only requires --resume-checkpoint (an already-trained model)")
        if not resolved_validation_problems and not resolved_secondary_validation_problems:
            print(
                "[ERROR][Phase2.merged.train_phase2_batch_machine_self_labeling] "
                "cause=eval_only_requires_validation_problems"
            )
            raise RuntimeError("eval_only requires at least one validation problem set")
        eval_episode = resumed_from_episode if resumed_from_episode > 0 else start_episode
        eval_temperature = _validation_sampling_temperature(
            annealed_temperature(
                eval_episode, temperature, resolved_temperature_min, resolved_temperature_anneal_episodes
            )
        )
        print(
            "[CHECK][Phase2.merged.train_phase2_batch_machine_self_labeling] "
            f"eval_only=true eval_episode={eval_episode} eval_temperature={eval_temperature:.4f} "
            f"validation_temperature_fixed={str(resolved_validation_temperature is not None).lower()}"
        )
        if resolved_validation_problems:
            _run_phase2_validation_cycle(
                episode=eval_episode,
                current_temperature=eval_temperature,
                problems=resolved_validation_problems,
                root=validation_root,
                select_best=False,
                accumulated_rows=validation_rows,
                accumulated_parent_rows=validation_parent_rows,
                current_best_validation_key=None,
            )
        if resolved_secondary_validation_problems:
            _run_phase2_validation_cycle(
                episode=eval_episode,
                current_temperature=eval_temperature,
                problems=resolved_secondary_validation_problems,
                root=generalization_root,
                select_best=False,
                accumulated_rows=secondary_validation_rows,
                accumulated_parent_rows=secondary_validation_parent_rows,
                current_best_validation_key=None,
            )
        eval_summary = {
            "eval_only": True,
            "eval_episode": eval_episode,
            "resumed_from_episode": resumed_from_episode,
            "resume_checkpoint": str(resume_path),
            "validation_root": str(validation_root),
            "validation_problem_count": len(resolved_validation_problems),
            "generalization_validation_root": (
                str(generalization_root) if resolved_secondary_validation_problems else ""
            ),
            "secondary_validation_problem_count": len(resolved_secondary_validation_problems),
            "summary_json": str(summary_json),
            # 아래는 재평가 디렉터리가 자기 채점 조건을 스스로 증명하기 위한 값이다.
            # 이것이 없으면 eval 결과만 보고는 어떤 계약/온도로 채점했는지 알 수 없다.
            "run_spec": run_spec,
            "validation_contract": validation_contract,
            "score_mode": score_mode,
            "heuristic_algorithms": list(heuristic_algorithms),
            "rollout_samples": rollout_samples,
            "validation_rollout_samples": resolved_validation_rollout_samples,
            "temperature": temperature,
            "temperature_min": resolved_temperature_min,
            "temperature_anneal_episodes": resolved_temperature_anneal_episodes,
            "validation_temperature": resolved_validation_temperature,
            "eval_temperature": eval_temperature,
            "device": str(torch_device),
        }
        summary_json.write_text(
            json.dumps(eval_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            "[CHECK][Phase2.merged.train_phase2_batch_machine_self_labeling] "
            f"eval_only=true eval_episode={eval_episode} "
            f"validation_problems={len(resolved_validation_problems)} "
            f"generalization_problems={len(resolved_secondary_validation_problems)}"
        )
        return eval_summary

    for episode in range(start_episode, episodes + 1):
        current_temperature = annealed_temperature(
            episode, temperature, resolved_temperature_min, resolved_temperature_anneal_episodes
        )
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
            candidate_workers=candidate_workers,
            candidate_executor=candidate_executor,
            temperature=current_temperature,
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
            f"episode={episode} temperature={current_temperature:.4f} best_source={best.source} loss={row['loss']} score={row['score_json']}"
        )
        run_validation_now = episode % validation_every == 0
        if run_validation_now and resolved_validation_problems:
            # PRIMARY(MAIN, in-distribution): best-checkpoint 선택을 담당한다.
            primary_result = _run_phase2_validation_cycle(
                episode=episode,
                current_temperature=_validation_sampling_temperature(current_temperature),
                problems=resolved_validation_problems,
                root=validation_root,
                select_best=True,
                accumulated_rows=validation_rows,
                accumulated_parent_rows=validation_parent_rows,
                current_best_validation_key=best_validation_key,
            )
            best_validation_key = primary_result["best_validation_key"]
            validation_plot_paths = primary_result["plot_paths"]
        if run_validation_now and resolved_secondary_validation_problems:
            # SECONDARY(GENERALIZATION): 리포팅 전용. best-checkpoint 선택에 관여하지 않는다.
            _run_phase2_validation_cycle(
                episode=episode,
                current_temperature=_validation_sampling_temperature(current_temperature),
                problems=resolved_secondary_validation_problems,
                root=generalization_root,
                select_best=False,
                accumulated_rows=secondary_validation_rows,
                accumulated_parent_rows=secondary_validation_parent_rows,
                current_best_validation_key=None,
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
                validation_contract=validation_contract,
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
        validation_contract=validation_contract,
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
        "validation_root": str(validation_root),
        "validation_summary_csv": str(validation_summary_csv),
        "validation_candidate_summary_csv": str(validation_candidate_summary_csv),
        "validation_parent_summary_csv": str(validation_parent_summary_csv),
        "generalization_validation_root": (
            str(generalization_root) if resolved_secondary_validation_problems else ""
        ),
        "generalization_validation_parent_summary_csv": (
            str(generalization_parent_summary_csv)
            if resolved_secondary_validation_problems
            else ""
        ),
        "secondary_validation_problem_count": len(resolved_secondary_validation_problems),
        **validation_plot_paths,
        "best_batches_csv": str(best_batches_csv),
        "best_timeline_csv": str(best_timeline_csv),
        "best_assignment_csv": str(best_assignment_csv),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path.is_file() else "",
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_every > 0 else "",
        "checkpoint_every": checkpoint_every,
        "summary_json": str(summary_json),
        "heuristic_algorithms": list(heuristic_algorithms),
        "rollout_samples": rollout_samples,
        "validation_rollout_samples": resolved_validation_rollout_samples,
        "validation_every": validation_every,
        "validation_episodes": validation_episodes,
        "validation_problem_count": len(resolved_validation_problems),
        "validation_contract": validation_contract,
        "action_pool_limit": action_pool_limit,
        "write_candidate_summary": write_candidate_summary,
        "candidate_workers": candidate_workers,
        "temperature": temperature,
        "temperature_min": resolved_temperature_min,
        "temperature_anneal_episodes": resolved_temperature_anneal_episodes,
        "validation_temperature": resolved_validation_temperature,
        "device": str(torch_device),
        "run_spec": run_spec,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@dataclass(frozen=True)
class _Phase2CandidateWorkerPayload:
    """한 worker가 동일 model snapshot으로 처리할 candidate 묶음이다."""

    indexed_sources: tuple[tuple[int, str, int], ...]
    jobs: Mapping[str, object]
    machines: Mapping[str, object]
    phase1_assignments: Mapping[str, str]
    model_state_dict: Dict[str, torch.Tensor] | None
    hidden_dim: int
    device: str
    max_wo_count: int
    max_length_sum: float
    action_pool_limit: int | None
    score_mode: str
    constraint_profile: PhaseConstraintProfile
    temperature: float = 1.0


def create_phase2_candidate_executor(candidate_workers: int) -> ProcessPoolExecutor | None:
    """CUDA-safe spawn worker pool을 만든다. 1은 기존 순차 경로다."""

    _validate_candidate_workers(candidate_workers)
    if candidate_workers == 1:
        return None
    return ProcessPoolExecutor(
        max_workers=candidate_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_phase2_candidate_worker,
    )


def _resolve_validation_temperature(validation_temperature: float | None) -> float | None:
    """평가용 고정 sampling 온도를 검증한다. None이면 학습 스케줄을 그대로 따른다."""

    if validation_temperature is None:
        return None
    try:
        value = float(validation_temperature)
    except (TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.merged._resolve_validation_temperature] "
            f"cause=invalid_validation_temperature value={validation_temperature!r}"
        )
        raise ValueError("validation_temperature must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        print(
            "[ERROR][Phase2.merged._resolve_validation_temperature] "
            f"cause=non_positive_validation_temperature value={validation_temperature!r}"
        )
        raise ValueError("validation_temperature must be a positive number")
    return value


def annealed_temperature(episode: int, t0: float, t_min: float, anneal_episodes: int) -> float:
    """episode에 따른 지수 감쇠 temperature. t_min>=t0이면 상수(t0)로 동작(하위호환)."""

    if t0 <= 0 or t_min <= 0:
        print(f"[ERROR][Phase2.merged.annealed_temperature] cause=non_positive value t0={t0} t_min={t_min}")
        raise ValueError("temperature and temperature_min must be positive")
    if t_min >= t0 or anneal_episodes <= 1:
        return t0
    # T(ep) = t0 * (t_min/t0)^(min(ep, anneal)/anneal), clamp at t_min
    frac = min(max(episode, 0), anneal_episodes) / float(anneal_episodes)
    return max(t_min, t0 * (t_min / t0) ** frac)


def _validate_candidate_workers(candidate_workers: int) -> None:
    if isinstance(candidate_workers, bool) or not isinstance(candidate_workers, int) or candidate_workers <= 0:
        print(
            "[ERROR][Phase2.merged._validate_candidate_workers] "
            f"cause=invalid_candidate_workers value={candidate_workers}"
        )
        raise ValueError("candidate_workers must be a positive integer")


def _initialize_phase2_candidate_worker() -> None:
    """각 candidate process가 CPU thread를 중첩 생성하지 않게 한다."""

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _run_phase2_candidate_worker(
    payload: _Phase2CandidateWorkerPayload,
) -> List[tuple[int, Phase2BatchMachineCandidate]]:
    """동일 episode/model snapshot의 candidate 묶음을 한 process에서 실행한다."""

    model: Phase2SetPointerPolicy | None = None
    if payload.model_state_dict is not None:
        model = Phase2SetPointerPolicy(hidden_dim=payload.hidden_dim)
        model.load_state_dict(payload.model_state_dict)
        model.to(_resolve_torch_device(payload.device))
        model.eval()
    base_environment = _build_phase2_common_environment(
        jobs=payload.jobs,
        machines=payload.machines,
        phase1_assignments=payload.phase1_assignments,
        constraint_profile=payload.constraint_profile,
        max_wo_count=payload.max_wo_count,
        max_length_sum=payload.max_length_sum,
    )
    results: List[tuple[int, Phase2BatchMachineCandidate]] = []
    for candidate_index, source, candidate_seed in payload.indexed_sources:
        results.append(
            (
                candidate_index,
                run_phase2_batch_machine_candidate(
                    jobs=payload.jobs,
                    machines=payload.machines,
                    phase1_assignments=payload.phase1_assignments,
                    source=source,
                    model=model,
                    max_wo_count=payload.max_wo_count,
                    max_length_sum=payload.max_length_sum,
                    action_pool_limit=payload.action_pool_limit,
                    seed=candidate_seed,
                    score_mode=payload.score_mode,
                    constraint_profile=payload.constraint_profile,
                    common_environment=base_environment,
                    temperature=payload.temperature,
                ),
            )
        )
    return results


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
    candidate_workers: int = 1,
    candidate_executor: ProcessPoolExecutor | None = None,
    temperature: float = 1.0,
) -> List[Phase2BatchMachineCandidate]:
    """통합 Phase 2 후보 bank를 만든다. 휴리스틱 후보와 agent 후보를 함께 비교한다."""

    _score_field_names(score_mode)
    _validate_candidate_workers(candidate_workers)
    if candidate_workers == 1 and candidate_executor is not None:
        print(
            "[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] "
            "cause=executor_with_single_worker"
        )
        raise RuntimeError("candidate_executor requires candidate_workers greater than 1")
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
    indexed_sources: list[tuple[int, str, int]] = []
    for algorithm in heuristic_algorithms:
        indexed_sources.append((len(indexed_sources), algorithm, seed))
    if model is not None:
        # Phase 1과 동일한 방식으로 통일: sample_index==1을 greedy로 사용하며
        # 총 rollout_samples개(=greedy 1 + sample rollout_samples-1)의 agent 후보를 만든다.
        for sample_index in range(1, rollout_samples + 1):
            source = "agent_greedy" if sample_index == 1 else f"agent_sample_{sample_index}"
            indexed_sources.append((len(indexed_sources), source, seed + sample_index))
    if candidate_workers == 1:
        return [
            run_phase2_batch_machine_candidate(
                jobs=jobs,
                machines=machines,
                phase1_assignments=phase1_assignments,
                source=source,
                model=model,
                max_wo_count=max_wo_count,
                max_length_sum=max_length_sum,
                action_pool_limit=action_pool_limit,
                seed=candidate_seed,
                score_mode=score_mode,
                constraint_profile=resolved_constraint_profile,
                common_environment=base_environment,
                temperature=temperature,
            )
            for _, source, candidate_seed in indexed_sources
        ]

    worker_count = min(candidate_workers, len(indexed_sources))
    source_chunks = tuple(
        tuple(indexed_sources[worker_index::worker_count])
        for worker_index in range(worker_count)
    )
    model_state_dict = None
    hidden_dim = 1
    worker_device = "cpu"
    if model is not None:
        model_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        hidden_dim = model.hidden_dim
        worker_device = str(next(model.parameters()).device)
    payloads = [
        _Phase2CandidateWorkerPayload(
            indexed_sources=source_chunk,
            jobs=jobs,
            machines=machines,
            phase1_assignments=phase1_assignments,
            model_state_dict=model_state_dict,
            hidden_dim=hidden_dim,
            device=worker_device,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            action_pool_limit=action_pool_limit,
            score_mode=score_mode,
            constraint_profile=resolved_constraint_profile,
            temperature=temperature,
        )
        for source_chunk in source_chunks
    ]
    owned_executor = candidate_executor is None
    executor = candidate_executor or create_phase2_candidate_executor(candidate_workers)
    if executor is None:
        print(
            "[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] "
            "cause=missing_parallel_executor"
        )
        raise RuntimeError("parallel Phase 2 candidate bank requires an executor")
    futures = {
        executor.submit(_run_phase2_candidate_worker, payload): payload
        for payload in payloads
    }
    indexed_candidates: list[tuple[int, Phase2BatchMachineCandidate]] = []
    try:
        for future, payload in futures.items():
            try:
                indexed_candidates.extend(future.result())
            except Exception as exc:
                sources = [source for _, source, _ in payload.indexed_sources]
                print(
                    "[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] "
                    f"cause=candidate_worker_failed sources={sources} error={exc}"
                )
                raise RuntimeError("Phase 2 candidate worker failed") from exc
    finally:
        if owned_executor:
            executor.shutdown(wait=True, cancel_futures=True)
    indexed_candidates.sort(key=lambda item: item[0])
    if len(indexed_candidates) != len(indexed_sources):
        print(
            "[ERROR][Phase2.merged.build_phase2_batch_machine_candidate_bank] "
            f"cause=candidate_count_mismatch expected={len(indexed_sources)} actual={len(indexed_candidates)}"
        )
        raise RuntimeError("parallel Phase 2 candidate count mismatch")
    return [candidate for _, candidate in indexed_candidates]


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
    temperature: float = 1.0,
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
                temperature=temperature,
            )
            selected_wo = wo_actions[selected_wo_index]
            selected_job_id = str(selected_wo["job_ids"][0])
            transitions.append(
                Phase2BatchMachineTransition(
                    policy_state=wo_policy_state,
                    selected_action_index=selected_wo_index,
                    target_action_indices=_teacher_target_action_indices(
                        actions=wo_actions,
                        source=source,
                        selected_action_index=selected_wo_index,
                        policy_state=wo_policy_state,
                    ),
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


def _iter_phase2_bay_candidate_banks(
    *,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    phase1_assignments: Mapping[str, str],
    model: Phase2SetPointerPolicy,
    heuristic_algorithms: Sequence[str],
    rollout_samples: int,
    max_wo_count: int,
    max_length_sum: float,
    action_pool_limit: int | None,
    seed: int,
    score_mode: str,
    constraint_profile: PhaseConstraintProfile,
    candidate_workers: int = 1,
    candidate_executor: ProcessPoolExecutor | None = None,
    temperature: float = 1.0,
) -> Iterator[
    tuple[
        str,
        Mapping[str, object],
        Mapping[str, object],
        int,
        List[Phase2BatchMachineCandidate],
    ]
]:
    """학습과 validation이 공유하는 Bay별 후보 bank를 순서대로 만든다."""

    machine_bay_ids = _machine_bay_ids(machines)
    jobs_by_bay = _jobs_by_phase1_bay(jobs, phase1_assignments)
    machines_by_bay = _machines_by_bay(machines, machine_bay_ids)
    for bay_index, bay_id in enumerate(sorted(jobs_by_bay), start=1):
        bay_machines = machines_by_bay.get(bay_id)
        if not bay_machines:
            print(
                "[ERROR][Phase2.merged._iter_phase2_bay_candidate_banks] "
                f"cause=no_machine_for_bay bay_id={bay_id}"
            )
            raise RuntimeError(f"Phase 2 Bay subproblem has no machines: {bay_id}")
        bank_seed = seed + bay_index * 10_000
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
                seed=bank_seed,
                score_mode=score_mode,
                constraint_profile=constraint_profile,
                candidate_workers=candidate_workers,
                candidate_executor=candidate_executor,
                temperature=temperature,
            )
        ]
        yield bay_id, jobs_by_bay[bay_id], bay_machines, bank_seed, bay_candidates


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
    candidate_workers: int,
    candidate_executor: ProcessPoolExecutor | None,
    temperature: float = 1.0,
) -> tuple[float, Phase2BatchMachineCandidate, List[Phase2BatchMachineCandidate], List[Dict]]:
    all_candidates: List[Phase2BatchMachineCandidate] = []
    bay_bests: List[Phase2BatchMachineCandidate] = []
    bay_losses: List[float] = []
    subproblem_rows: List[Dict] = []
    score_field_names = _score_field_names(score_mode)

    for bay_id, bay_jobs, bay_machines, _bank_seed, bay_candidates in _iter_phase2_bay_candidate_banks(
        jobs=jobs,
        machines=machines,
        phase1_assignments=phase1_assignments,
        model=model,
        heuristic_algorithms=heuristic_algorithms,
        rollout_samples=rollout_samples,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        action_pool_limit=action_pool_limit,
        seed=seed,
        score_mode=score_mode,
        constraint_profile=constraint_profile,
        candidate_workers=candidate_workers,
        candidate_executor=candidate_executor,
        temperature=temperature,
    ):
        best = min(bay_candidates, key=_candidate_sort_key)
        bay_loss = _update_from_candidate(model, optimizer, best)
        bay_losses.append(bay_loss)
        bay_bests.append(best)
        all_candidates.extend(bay_candidates)
        subproblem_rows.append(
            _subproblem_metrics_row(
                episode=episode,
                bay_id=bay_id,
                job_count=len(bay_jobs),
                machine_count=len(bay_machines),
                candidate_count=len(bay_candidates),
                best=best,
                loss=bay_loss,
                score_field_names=score_field_names,
            )
        )
        print(
            "[CHECK][Phase2.merged._train_one_episode.subproblem] "
            f"bay_id={bay_id} job_count={len(bay_jobs)} machine_count={len(bay_machines)} "
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
        total_loss = total_loss + _multi_target_cross_entropy(
            logits,
            transition.target_action_indices,
        )
    mean_loss = total_loss / len(best.transitions)
    optimizer.zero_grad()
    mean_loss.backward()
    optimizer.step()
    return float(mean_loss.item())


def _multi_target_cross_entropy(
    logits: torch.Tensor,
    target_action_indices: Sequence[int],
) -> torch.Tensor:
    """복수 정답 후보에 할당된 확률 질량의 음의 로그를 반환한다."""

    if logits.ndim != 1 or logits.numel() == 0:
        print(
            "[ERROR][Phase2.merged._multi_target_cross_entropy] "
            f"cause=invalid_logits_shape shape={tuple(logits.shape)}"
        )
        raise RuntimeError("Phase 2 tie-aware CE requires one-dimensional non-empty logits")
    targets = tuple(dict.fromkeys(int(index) for index in target_action_indices))
    if not targets or any(index < 0 or index >= logits.numel() for index in targets):
        print(
            "[ERROR][Phase2.merged._multi_target_cross_entropy] "
            f"cause=invalid_targets targets={targets} action_count={logits.numel()}"
        )
        raise RuntimeError("Phase 2 tie-aware CE target indices are invalid")
    target_tensor = torch.tensor(targets, dtype=torch.long, device=logits.device)
    log_probabilities = F.log_softmax(logits, dim=0)
    return -torch.logsumexp(log_probabilities.index_select(0, target_tensor), dim=0)


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
        bevel_length = _bevel_length_for_job(job, job_id, dispatch_cache)
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
                "candidate_bevel_length": round(bevel_length, 6),
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
    temperature: float = 1.0,
) -> int:
    if source == "agent_greedy":
        return _model_action_index(actions, model, policy_state, sample=False, seed=seed)
    if source.startswith("agent_sample_"):
        return _model_action_index(actions, model, policy_state, sample=True, seed=seed, temperature=temperature)
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
    if source == "balanced_tact_load":
        return (float(action["projected_finish_time"]), float(action["machine_clock"]), -int(action["wo_count"]), *common)
    if source == "spt_batch":
        return (float(action["candidate_processing_time"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "lpt_batch":
        return (-float(action["candidate_processing_time"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "long_cut_batch":
        return (-float(action["candidate_cut_length"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "short_cut_batch":
        return (float(action["candidate_cut_length"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "long_bevel_batch":
        return (-float(action["candidate_bevel_length"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "short_bevel_batch":
        return (float(action["candidate_bevel_length"]), float(action["projected_makespan"]), -int(action["wo_count"]), *common)
    if source == "worst_fit_lth":
        # best_fit_lth의 반대: batch 길이합을 최소화 → 슬롯에서 가장 짧은 LTH W/O 우선.
        return (-int(action["wo_count"]), float(action["length_sum"]), float(action["projected_makespan"]), *common)
    print(f"[ERROR][Phase2.merged._heuristic_key] cause=unknown_source source={source}")
    raise RuntimeError(f"unknown merged Phase 2 candidate source: {source}")


def _teacher_target_action_indices(
    *,
    actions: Sequence[Mapping],
    source: str,
    selected_action_index: int,
    policy_state: Phase2PolicyState,
) -> tuple[int, ...]:
    """정책이 구분할 수 없는 휴리스틱 동률 후보를 모두 CE 정답으로 반환한다."""

    if not 0 <= selected_action_index < len(actions):
        print(
            "[ERROR][Phase2.merged._teacher_target_action_indices] "
            f"cause=selected_index_out_of_range index={selected_action_index} actions={len(actions)}"
        )
        raise RuntimeError("Phase 2 teacher selected action index is out of range")
    if _is_agent_source(source):
        return (selected_action_index,)

    selected_priority = _heuristic_priority_key(source, actions[selected_action_index])
    selected_observation = _policy_action_observation_key(policy_state, selected_action_index)
    tied = tuple(
        index
        for index, action in enumerate(actions)
        if _heuristic_priority_key(source, action) == selected_priority
        and _policy_action_observation_key(policy_state, index) == selected_observation
    )
    if selected_action_index not in tied:
        print(
            "[ERROR][Phase2.merged._teacher_target_action_indices] "
            f"cause=selected_index_missing_from_tie_set index={selected_action_index} tied={tied}"
        )
        raise RuntimeError("Phase 2 tie-aware target omitted the selected action")
    return tied


def _heuristic_priority_key(source: str, action: Mapping) -> tuple:
    """재현성용 machine/job ID를 제외한 휴리스틱의 실제 우선순위 key다."""

    key = _heuristic_key(source, action)
    if len(key) < 3:
        print(
            "[ERROR][Phase2.merged._heuristic_priority_key] "
            f"cause=invalid_heuristic_key source={source} key={key}"
        )
        raise RuntimeError("Phase 2 heuristic key has no removable identity tie-break")
    return key[:-2]


def _policy_action_observation_key(
    policy_state: Phase2PolicyState,
    action_index: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pointer score에 직접 들어가는 후보 W/O·projected feature를 반환한다."""

    if not 0 <= action_index < len(policy_state.action_candidate_node_indices):
        print(
            "[ERROR][Phase2.merged._policy_action_observation_key] "
            f"cause=action_index_out_of_range index={action_index}"
        )
        raise RuntimeError("Phase 2 action observation index is out of range")
    node_index = policy_state.action_candidate_node_indices[action_index]
    if not 0 <= node_index < len(policy_state.wo_node_features):
        print(
            "[ERROR][Phase2.merged._policy_action_observation_key] "
            f"cause=node_index_out_of_range index={node_index}"
        )
        raise RuntimeError("Phase 2 action observation node index is out of range")
    return (
        tuple(float(value) for value in policy_state.wo_node_features[node_index]),
        tuple(float(value) for value in policy_state.action_projected_features[action_index]),
    )


def _model_action_index(
    actions: Sequence[Mapping],
    model: Phase2SetPointerPolicy | None,
    policy_state: Phase2PolicyState,
    sample: bool,
    seed: int,
    temperature: float = 1.0,
) -> int:
    if model is None:
        print("[ERROR][Phase2.merged._model_action_index] cause=missing_model_for_agent_source")
        raise RuntimeError("merged Phase 2 agent candidate requires a model")
    if temperature <= 0:
        print(f"[ERROR][Phase2.merged._model_action_index] cause=non_positive_temperature value={temperature}")
        raise ValueError("temperature must be positive")
    with torch.no_grad():
        logits = model(policy_state).detach().cpu()
    if not sample:
        return int(torch.argmax(logits).item())
    generator = torch.Generator()
    generator.manual_seed(seed)
    probabilities = torch.softmax(logits / temperature, dim=0)
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


def _best_heuristic_from_ranked(
    ranked_candidates: Sequence[Phase2BatchMachineCandidate],
) -> Phase2BatchMachineCandidate:
    """검증 후보 중 사전식 score가 가장 좋은 dispatch heuristic을 반환한다."""

    for candidate in ranked_candidates:
        if not _is_agent_source(candidate.source):
            return candidate
    print("[ERROR][Phase2.merged._best_heuristic_from_ranked] cause=no_heuristic_candidate")
    raise RuntimeError("Phase 2 validation requires at least one heuristic candidate")


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
        bevel_length_by_job={str(job_id): _optional_bevel_length(job) for job_id, job in jobs.items()},
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


def _optional_bevel_length(job: object) -> float:
    """job의 베벨 길이(BVL_LTH). 최소 fixture처럼 필드가 없으면 0.0으로 관대하게 읽는다.
    (실적 job은 bevel_length가 항상 존재하며, 이 값이 long/short bevel 휴리스틱의 기준이 된다.)"""

    value = job.get("bevel_length") if isinstance(job, Mapping) else getattr(job, "bevel_length", None)
    if value is None:
        return 0.0
    length = float(value)
    return length if length >= 0.0 else 0.0


def _bevel_length_for_job(job: object, job_id: str, dispatch_cache: _Phase2DispatchCache | None = None) -> float:
    if dispatch_cache is not None and str(job_id) in dispatch_cache.bevel_length_by_job:
        return float(dispatch_cache.bevel_length_by_job[str(job_id)])
    return _optional_bevel_length(job)


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


def _resolve_validation_problems(
    validation_episodes: int,
    default_jobs: Mapping[str, object],
    validation_episode_jobs: Sequence[Mapping[str, object]] | None,
    validation_episode_job_factory: Callable[[int], Mapping[str, object]] | None,
    validation_problems: Sequence[Phase2ValidationProblem] | None,
) -> tuple[Phase2ValidationProblem, ...]:
    """모든 validation 입력 방식을 학습 시작 시 고정 문제 객체로 변환한다."""

    if validation_episodes == 0:
        if validation_problems is not None:
            print(
                "[ERROR][Phase2.merged._resolve_validation_problems] "
                "cause=validation_disabled_with_explicit_problems"
            )
            raise RuntimeError("validation_problems require validation_episodes > 0")
        return ()
    if validation_problems is not None:
        resolved = tuple(validation_problems)
    else:
        resolved = tuple(
            Phase2ValidationProblem(
                problem_id=f"VAL{index:03d}",
                block_count=_physical_block_count(jobs),
                distribution_type=index,
                generation_seed=index,
                target_distribution={},
                normalized_distribution={},
                actual_distribution={},
                jobs=jobs,
                metadata={
                    "problem_id": f"VAL{index:03d}",
                    "physical_block_count": _physical_block_count(jobs),
                    "distribution_type": index,
                    "input_contract": "legacy_fixed_validation",
                },
            )
            for index in range(1, validation_episodes + 1)
            for jobs in (
                _validation_episode_jobs(
                    index,
                    default_jobs,
                    validation_episode_jobs,
                    validation_episode_job_factory,
                ),
            )
        )
    problem_ids = [problem.problem_id for problem in resolved]
    if not resolved or len(set(problem_ids)) != len(problem_ids):
        print(
            "[ERROR][Phase2.merged._resolve_validation_problems] "
            f"cause=invalid_problem_ids values={problem_ids}"
        )
        raise RuntimeError("Phase 2 validation problem IDs must be non-empty and unique")
    return resolved


def _physical_block_count(jobs: Mapping[str, object]) -> int:
    block_ids = {
        str(getattr(job, "block_set_id", ""))
        for job in jobs.values()
    }
    if not block_ids or "" in block_ids:
        print(
            "[ERROR][Phase2.merged._physical_block_count] "
            "cause=missing_block_set_id"
        )
        raise RuntimeError("Phase 2 validation jobs require block_set_id")
    malformed = sorted(block_id for block_id in block_ids if len(block_id.split("::")) < 3)
    if malformed:
        print(
            "[ERROR][Phase2.merged._physical_block_count] "
            f"cause=invalid_block_set_ids values={malformed[:5]}"
        )
        raise RuntimeError("Phase 2 block_set_id must contain project, series, and block")
    physical_ids = {
        "::".join((parts[0], parts[-1]))
        for block_id in block_ids
        for parts in (block_id.split("::"),)
    }
    return len(physical_ids)


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
    validation_problems: Sequence[Phase2ValidationProblem] | None,
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
    validation_source_count = sum(
        source is not None
        for source in (
            validation_episode_jobs,
            validation_episode_job_factory,
            validation_problems,
        )
    )
    if validation_source_count > 1:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=conflicting_validation_episode_sources")
        raise RuntimeError("use exactly one explicit Phase 2 validation problem source")
    if episode_jobs is not None and not episode_jobs:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=empty_episode_jobs")
        raise RuntimeError("episode_jobs must not be empty")
    if validation_episode_jobs is not None and not validation_episode_jobs:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=empty_validation_episode_jobs")
        raise RuntimeError("validation_episode_jobs must not be empty")
    if validation_problems is not None and not validation_problems:
        print("[ERROR][Phase2.merged._validate_train_inputs] cause=empty_validation_problems")
        raise RuntimeError("validation_problems must not be empty")


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
    expected_validation_contract: Mapping[str, object],
    eval_only: bool = False,
) -> int:
    """Load Phase 2 model/optimizer state and return completed episode.

    eval_only=True는 이미 학습된 checkpoint를 (다를 수 있는) validation 문제로 평가만 하므로
    validation_contract 일치 검사를 건너뛴다. run_spec(정책/피처/휴리스틱 계약) 검사는 유지한다.
    """

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
    checkpoint_validation_contract = checkpoint.get("validation_contract")
    if checkpoint_validation_contract != expected_validation_contract:
        if eval_only:
            print(
                "[CHECK][Phase2.merged._load_phase2_batch_machine_checkpoint] "
                f"eval_only=true validation_contract_differs path={path} "
                "(평가 전용이므로 contract 불일치를 허용한다)"
            )
        else:
            print(
                "[ERROR][Phase2.merged._load_phase2_batch_machine_checkpoint] "
                f"cause=validation_contract_mismatch path={path}"
            )
            raise RuntimeError("Phase 2 resume checkpoint validation grid mismatch")
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
    validation_contract: Mapping[str, object],
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
            "validation_contract": dict(validation_contract),
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


def _score_relation(
    proposed: Phase2BatchMachineCandidate | None,
    heuristic: Phase2BatchMachineCandidate,
) -> str:
    """Proposed와 최상 휴리스틱의 사전식 score 관계를 반환한다."""

    if proposed is None:
        print("[ERROR][Phase2.merged._score_relation] cause=no_proposed_candidate")
        raise RuntimeError("Phase 2 validation requires a Proposed candidate")
    if proposed.score_tuple < heuristic.score_tuple:
        return "win"
    if proposed.score_tuple == heuristic.score_tuple:
        return "tie"
    return "loss"


def _validation_checkpoint_key(rows: Sequence[Mapping]) -> tuple[int, int, float]:
    """동일 가중치의 상위 `(block size, Type)` 문제로 best checkpoint를 고른다."""

    if not rows:
        print("[ERROR][Phase2.merged._validation_checkpoint_key] cause=no_rows")
        raise RuntimeError("Phase 2 best-checkpoint selection requires validation rows")
    relations = [str(row["proposed_vs_best_heuristic"]) for row in rows]
    invalid = sorted(set(relations) - {"win", "tie", "loss"})
    if invalid:
        print(
            "[ERROR][Phase2.merged._validation_checkpoint_key] "
            f"cause=invalid_relation values={invalid}"
        )
        raise RuntimeError("Phase 2 validation contains invalid comparison labels")
    ranks = [int(row["proposed_best_rank"]) for row in rows]
    if any(rank <= 0 for rank in ranks):
        print(
            "[ERROR][Phase2.merged._validation_checkpoint_key] "
            f"cause=invalid_proposed_rank ranks={ranks}"
        )
        raise RuntimeError("Phase 2 validation contains invalid Proposed ranks")
    return (
        relations.count("loss"),
        -relations.count("win"),
        sum(ranks) / float(len(ranks)),
    )


def _resume_validation_checkpoint_key(
    rows: Sequence[Mapping],
    checkpoint_path: Path,
) -> tuple[float, ...] | None:
    """재개 CSV의 신규 selection key가 있으면 기존 best 상태를 복원한다."""

    keys: list[tuple[float, ...]] = []
    for row in rows:
        raw = row.get("checkpoint_selection_key_json")
        if raw in (None, ""):
            continue
        try:
            parsed = json.loads(str(raw))
            key = tuple(float(value) for value in parsed)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(
                "[ERROR][Phase2.merged._resume_validation_checkpoint_key] "
                f"cause=invalid_selection_key value={raw}"
            )
            raise RuntimeError("invalid Phase 2 validation checkpoint key") from exc
        if len(key) != 3:
            print(
                "[ERROR][Phase2.merged._resume_validation_checkpoint_key] "
                f"cause=invalid_selection_key_length value={raw}"
            )
            raise RuntimeError("invalid Phase 2 validation checkpoint key length")
        keys.append(key)
    if not keys:
        return None
    if not checkpoint_path.is_file():
        print(
            "[ERROR][Phase2.merged._resume_validation_checkpoint_key] "
            f"cause=missing_best_checkpoint path={checkpoint_path}"
        )
        raise RuntimeError("Phase 2 validation history exists but best checkpoint is missing")
    return min(keys)


def _validation_summary_row(
    train_episode: int,
    validation_episode: int,
    validation_phase1_seed: int,
    validation_sampling_seed: int,
    bay_id: str,
    jobs: Mapping[str, object],
    machines: Mapping[str, object],
    candidate_count: int,
    best: Phase2BatchMachineCandidate,
    best_heuristic: Phase2BatchMachineCandidate,
    proposed_rank: int,
    proposed_best: Phase2BatchMachineCandidate | None,
    greedy_rank: int,
    greedy_best: Phase2BatchMachineCandidate | None,
    score_field_names: Sequence[str],
) -> Dict:
    row = {
        "train_episode": train_episode,
        "validation_episode": validation_episode,
        "validation_phase1_seed": validation_phase1_seed,
        "validation_sampling_seed": validation_sampling_seed,
        "bay_id": bay_id,
        "job_count": len(jobs),
        "machine_count": len(machines),
        "candidate_count": candidate_count,
        "best_source": best.source,
        "best_score_json": json.dumps(list(best.score_tuple), ensure_ascii=False),
        "best_heuristic_source": best_heuristic.source,
        "best_heuristic_score_json": json.dumps(
            list(best_heuristic.score_tuple),
            ensure_ascii=False,
        ),
        "proposed_vs_best_heuristic": _score_relation(
            proposed_best,
            best_heuristic,
        ),
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
    validation_sampling_seed: int,
    rank: int,
    candidate: Phase2BatchMachineCandidate,
    score_field_names: Sequence[str],
) -> Dict:
    row = _candidate_summary_row(validation_episode, rank, candidate, score_field_names)
    row["train_episode"] = train_episode
    row["validation_episode"] = validation_episode
    row["validation_sampling_seed"] = validation_sampling_seed
    row.pop("episode", None)
    return row


def _validation_problem_columns(
    problem: Phase2ValidationProblem,
) -> dict[str, int | str]:
    return {
        "validation_problem_id": problem.problem_id,
        "block_count": problem.block_count,
        "distribution_type": problem.distribution_type,
        "generation_seed": problem.generation_seed,
    }


def _validation_parent_candidates(
    candidate_banks_by_bay: Mapping[
        str,
        tuple[
            Mapping[str, object],
            Mapping[str, object],
            int,
            Sequence[Phase2BatchMachineCandidate],
        ],
    ],
    machines: Mapping[str, object],
    heuristic_algorithms: Sequence[str],
    max_wo_count: int,
    max_length_sum: float,
    score_mode: str,
) -> dict[str, Phase2BatchMachineCandidate]:
    """Bay별 결과를 같은 method끼리 합쳐 상위 validation 해를 만든다."""

    if not candidate_banks_by_bay:
        print("[ERROR][Phase2.merged._validation_parent_candidates] cause=no_bay_candidates")
        raise RuntimeError("Phase 2 validation problem produced no Bay candidate bank")
    result: dict[str, Phase2BatchMachineCandidate] = {}
    for source in (PHASE2_PROPOSED_BEST_OF_K_SOURCE, *heuristic_algorithms):
        bay_candidates: list[Phase2BatchMachineCandidate] = []
        for bay_id in sorted(candidate_banks_by_bay):
            ranked = candidate_banks_by_bay[bay_id][3]
            if source == PHASE2_PROPOSED_BEST_OF_K_SOURCE:
                _rank, selected = _agent_best_from_ranked(ranked)
                if selected is None:
                    print(
                        "[ERROR][Phase2.merged._validation_parent_candidates] "
                        f"cause=no_agent_candidate bay_id={bay_id}"
                    )
                    raise RuntimeError("Phase 2 validation requires an agent candidate in every non-empty Bay")
            else:
                matches = [candidate for candidate in ranked if candidate.source == source]
                if len(matches) != 1:
                    print(
                        "[ERROR][Phase2.merged._validation_parent_candidates] "
                        f"cause=heuristic_candidate_cardinality bay_id={bay_id} "
                        f"source={source} count={len(matches)}"
                    )
                    raise RuntimeError("Phase 2 validation heuristic candidate is missing or duplicated")
                selected = matches[0]
            bay_candidates.append(selected)
        combined = _combine_subproblem_bests(
            bay_bests=bay_candidates,
            machines=machines,
            max_wo_count=max_wo_count,
            max_length_sum=max_length_sum,
            score_mode=score_mode,
        )
        result[source] = replace(combined, source=source)
    return result


def _validation_parent_rows(
    train_episode: int,
    validation_episode: int,
    validation_problem: Phase2ValidationProblem,
    parent_candidates: Mapping[str, Phase2BatchMachineCandidate],
    score_field_names: Sequence[str],
) -> tuple[dict, list[dict]]:
    proposed = parent_candidates[PHASE2_PROPOSED_BEST_OF_K_SOURCE]
    heuristic_candidates = [
        candidate
        for source, candidate in parent_candidates.items()
        if source != PHASE2_PROPOSED_BEST_OF_K_SOURCE
    ]
    if not heuristic_candidates:
        print("[ERROR][Phase2.merged._validation_parent_rows] cause=no_heuristic_candidates")
        raise RuntimeError("Phase 2 parent validation requires heuristic candidates")
    best_heuristic = min(heuristic_candidates, key=_candidate_sort_key)
    ranked = sorted(parent_candidates.values(), key=_candidate_sort_key)
    best = ranked[0]
    proposed_rank = 1 + sum(
        candidate.score_tuple < proposed.score_tuple
        for candidate in ranked
    )
    parent_row = {
        "train_episode": train_episode,
        "validation_episode": validation_episode,
        **_validation_problem_columns(validation_problem),
        "job_count": len(validation_problem.jobs),
        "bay_subproblem_count": len(
            {
                proposed.machine_bay_ids[machine_id]
                for machine_id in proposed.machine_assignments.values()
            }
        ),
        "best_source": best.source,
        "best_score_json": json.dumps(list(best.score_tuple), ensure_ascii=False),
        "best_heuristic_source": best_heuristic.source,
        "best_heuristic_score_json": json.dumps(
            list(best_heuristic.score_tuple),
            ensure_ascii=False,
        ),
        "proposed_vs_best_heuristic": _score_relation(proposed, best_heuristic),
        "proposed_best_rank": proposed_rank,
        "proposed_best_source": proposed.source,
        "proposed_best_score_json": json.dumps(
            list(proposed.score_tuple),
            ensure_ascii=False,
        ),
    }
    parent_row.update(
        {
            field: proposed.score_tuple[index]
            for index, field in enumerate(score_field_names)
        }
    )
    method_rows: list[dict] = []
    for candidate in ranked:
        row = {
            "train_episode": train_episode,
            "validation_episode": validation_episode,
            **_validation_problem_columns(validation_problem),
            "source": candidate.source,
            "score_json": json.dumps(list(candidate.score_tuple), ensure_ascii=False),
        }
        row.update(
            {
                field: candidate.score_tuple[index]
                for index, field in enumerate(score_field_names)
            }
        )
        method_rows.append(row)
    return parent_row, method_rows


def _validation_bay_method_rows(
    train_episode: int,
    validation_episode: int,
    validation_problem: Phase2ValidationProblem,
    bay_id: str,
    ranked_candidates: Sequence[Phase2BatchMachineCandidate],
    heuristic_algorithms: Sequence[str],
    score_field_names: Sequence[str],
) -> list[dict]:
    """Bay별 그래프에 쓸 Proposed 1건과 휴리스틱별 1건을 고정한다."""

    _rank, proposed = _agent_best_from_ranked(ranked_candidates)
    if proposed is None:
        print(
            "[ERROR][Phase2.merged._validation_bay_method_rows] "
            f"cause=no_agent_candidate problem={validation_problem.problem_id} bay_id={bay_id}"
        )
        raise RuntimeError("Phase 2 Bay validation requires an agent candidate")
    selected = {PHASE2_PROPOSED_BEST_OF_K_SOURCE: proposed}
    for source in heuristic_algorithms:
        matches = [
            candidate
            for candidate in ranked_candidates
            if candidate.source == source
        ]
        if len(matches) != 1:
            print(
                "[ERROR][Phase2.merged._validation_bay_method_rows] "
                f"cause=heuristic_candidate_cardinality problem={validation_problem.problem_id} "
                f"bay_id={bay_id} source={source} count={len(matches)}"
            )
            raise RuntimeError("Phase 2 Bay validation heuristic candidate is missing or duplicated")
        selected[source] = matches[0]
    rows: list[dict] = []
    for source, candidate in selected.items():
        row = {
            "train_episode": train_episode,
            "validation_episode": validation_episode,
            **_validation_problem_columns(validation_problem),
            "bay_id": bay_id,
            "source": source,
            "score_json": json.dumps(list(candidate.score_tuple), ensure_ascii=False),
        }
        row.update(
            {
                field: candidate.score_tuple[index]
                for index, field in enumerate(score_field_names)
            }
        )
        rows.append(row)
    return rows


def _write_validation_problem_catalog(
    validation_root: Path,
    problems: Sequence[Phase2ValidationProblem],
    contract: Mapping[str, object],
) -> None:
    validation_root.mkdir(parents=True, exist_ok=True)
    contract_path = validation_root / "grid_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            print(
                "[ERROR][Phase2.merged._write_validation_problem_catalog] "
                f"cause=existing_contract_mismatch path={contract_path}"
            )
            raise RuntimeError("output directory contains a different Phase 2 validation grid")
    _write_json(contract_path, contract)
    for problem in problems:
        problem_root = _validation_problem_root(validation_root / "problems", problem)
        problem_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            problem_root / "problem_metadata.json",
            {
                **dict(problem.metadata),
                **_validation_problem_columns(problem),
            },
        )
        _write_json(
            problem_root / "distribution_profile.json",
            {
                "target": dict(problem.target_distribution),
                "normalized_actual": dict(problem.normalized_distribution),
                "raw_actual": dict(problem.actual_distribution),
            },
        )
        _write_rows(
            problem_root / "jobs.csv",
            _validation_job_rows(problem.jobs),
            _validation_job_fields(),
        )


def _write_validation_problem_evaluation(
    evaluation_root: Path,
    validation_problem: Phase2ValidationProblem,
    phase1_assignments: Mapping[str, str],
    machines: Mapping[str, object],
    candidate_banks_by_bay: Mapping[
        str,
        tuple[
            Mapping[str, object],
            Mapping[str, object],
            int,
            Sequence[Phase2BatchMachineCandidate],
        ],
    ],
    parent_candidates: Mapping[str, Phase2BatchMachineCandidate],
    score_field_names: Sequence[str],
) -> None:
    problem_root = _validation_problem_root(evaluation_root, validation_problem)
    problem_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        problem_root / "evaluation_metadata.json",
        {
            **_validation_problem_columns(validation_problem),
            "fixed_problem_path": str(
                _validation_problem_root(
                    evaluation_root.parents[1] / "problems",
                    validation_problem,
                )
            ),
            "evaluated_bays": sorted(candidate_banks_by_bay),
        },
    )
    _write_rows(
        problem_root / "phase1_assignments.csv",
        [
            {"block_set_id": block_set_id, "bay_id": bay_id}
            for block_set_id, bay_id in sorted(phase1_assignments.items())
        ],
        ["block_set_id", "bay_id"],
    )
    jobs_by_bay = _jobs_by_phase1_bay(validation_problem.jobs, phase1_assignments)
    machine_bay_ids = _machine_bay_ids(machines)
    machines_by_bay = _machines_by_bay(machines, machine_bay_ids)
    for bay_id in sorted(machines_by_bay):
        bay_root = problem_root / f"bay_{bay_id}"
        bay_root.mkdir(parents=True, exist_ok=True)
        bay_jobs = jobs_by_bay.get(bay_id, {})
        _write_json(
            bay_root / "problem_metadata.json",
            {
                **_validation_problem_columns(validation_problem),
                "bay_id": bay_id,
                "job_count": len(bay_jobs),
                "machine_count": len(machines_by_bay[bay_id]),
                "evaluated": bool(bay_jobs),
            },
        )
        if not bay_jobs:
            continue
        bank = candidate_banks_by_bay.get(bay_id)
        if bank is None:
            print(
                "[ERROR][Phase2.merged._write_validation_problem_evaluation] "
                f"cause=missing_candidate_bank problem={validation_problem.problem_id} bay_id={bay_id}"
            )
            raise RuntimeError("non-empty validation Bay has no candidate bank")
        ranked = bank[3]
        _write_rows(
            bay_root / "jobs.csv",
            _validation_job_rows(bay_jobs),
            _validation_job_fields(),
        )
        _write_rows(
            bay_root / "candidate_summary.csv",
            [
                _candidate_summary_row(0, rank, candidate, score_field_names)
                for rank, candidate in enumerate(ranked, start=1)
            ],
            _candidate_fields(score_field_names),
        )
        _rank, proposed = _agent_best_from_ranked(ranked)
        if proposed is None:
            print(
                "[ERROR][Phase2.merged._write_validation_problem_evaluation] "
                f"cause=no_proposed_candidate problem={validation_problem.problem_id} bay_id={bay_id}"
            )
            raise RuntimeError("validation Bay requires a Proposed candidate")
        _write_assignments(bay_root / "proposed_solution.csv", proposed)
    _write_rows(
        problem_root / "parent_method_summary.csv",
        [
            {
                "source": source,
                "score_json": json.dumps(list(candidate.score_tuple), ensure_ascii=False),
                "method_rank": 1
                + sum(
                    other.score_tuple < candidate.score_tuple
                    for other in parent_candidates.values()
                ),
                "is_rank1": int(
                    not any(
                        other.score_tuple < candidate.score_tuple
                        for other in parent_candidates.values()
                    )
                ),
                "rank1_tie_count": sum(
                    other.score_tuple
                    == min(
                        item.score_tuple
                        for item in parent_candidates.values()
                    )
                    for other in parent_candidates.values()
                ),
                **{
                    field: candidate.score_tuple[index]
                    for index, field in enumerate(score_field_names)
                },
            }
            for source, candidate in parent_candidates.items()
        ],
        [
            "source",
            "score_json",
            "method_rank",
            "is_rank1",
            "rank1_tie_count",
            *score_field_names,
        ],
    )


def _validation_problem_root(
    root: Path,
    problem: Phase2ValidationProblem,
) -> Path:
    return (
        root
        / f"blocks_{problem.block_count:03d}"
        / f"type_{problem.distribution_type:02d}"
    )


def _validation_job_rows(jobs: Mapping[str, object]) -> list[dict]:
    return [
        {
            "job_id": job_id,
            "block_set_id": str(getattr(job, "block_set_id")),
            "family": str(getattr(job, "family")),
            "plate_length": float(getattr(job, "plate_length")),
            "cut_length": float(getattr(job, "cut_length")),
            "bevel_quantity": float(getattr(job, "bevel_quantity")),
            "tact_time": float(_processing_time(job)),
        }
        for job_id, job in sorted(jobs.items())
    ]


def _validation_job_fields() -> list[str]:
    return [
        "job_id",
        "block_set_id",
        "family",
        "plate_length",
        "cut_length",
        "bevel_quantity",
        "tact_time",
    ]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _rank_validation_method_rows(
    rows: Sequence[Mapping],
) -> list[dict]:
    """상위 validation 문제마다 사전식 경쟁 순위를 계산한다."""

    if not rows:
        print("[ERROR][Phase2.merged._rank_validation_method_rows] cause=no_rows")
        raise RuntimeError("Phase 2 validation ranking requires method rows")
    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        problem_id = str(row.get("validation_problem_id", ""))
        source = str(row.get("source", ""))
        if not problem_id or not source:
            print(
                "[ERROR][Phase2.merged._rank_validation_method_rows] "
                f"cause=missing_identity problem_id={problem_id} source={source}"
            )
            raise RuntimeError("Phase 2 validation method row has no identity")
        grouped.setdefault(problem_id, []).append(row)

    ranked_rows: list[dict] = []
    for problem_id, problem_rows in grouped.items():
        sources = [str(row["source"]) for row in problem_rows]
        if len(set(sources)) != len(sources):
            print(
                "[ERROR][Phase2.merged._rank_validation_method_rows] "
                f"cause=duplicate_source problem_id={problem_id} sources={sources}"
            )
            raise RuntimeError("Phase 2 validation problem contains duplicate methods")
        scores = {
            str(row["source"]): _validation_score_tuple(row)
            for row in problem_rows
        }
        best_score = min(scores.values())
        rank1_tie_count = sum(score == best_score for score in scores.values())
        for row in problem_rows:
            source = str(row["source"])
            score = scores[source]
            ranked_rows.append(
                {
                    **dict(row),
                    "method_rank": 1
                    + sum(other_score < score for other_score in scores.values()),
                    "is_rank1": int(score == best_score),
                    "rank1_tie_count": rank1_tie_count,
                }
            )
    return sorted(
        ranked_rows,
        key=lambda row: (
            int(row["block_count"]),
            int(row["distribution_type"]),
            int(row["method_rank"]),
            str(row["source"]),
        ),
    )


def _validation_score_tuple(row: Mapping) -> tuple[float, ...]:
    raw = row.get("score_json")
    try:
        parsed = json.loads(str(raw))
        score = tuple(float(value) for value in parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        print(
            "[ERROR][Phase2.merged._validation_score_tuple] "
            f"cause=invalid_score_json value={raw}"
        )
        raise RuntimeError("invalid Phase 2 validation score JSON") from exc
    if not score or any(not math.isfinite(value) for value in score):
        print(
            "[ERROR][Phase2.merged._validation_score_tuple] "
            f"cause=invalid_score_values value={raw}"
        )
        raise RuntimeError("Phase 2 validation score must be finite and non-empty")
    return score


def _validation_rank_summary_rows(
    parent_problem_rows: Sequence[Mapping],
    ranked_method_rows: Sequence[Mapping],
) -> list[dict]:
    """전체·크기별·Type별 Proposed 관계와 method 1위 횟수를 집계한다."""

    if not parent_problem_rows or not ranked_method_rows:
        print(
            "[ERROR][Phase2.merged._validation_rank_summary_rows] "
            f"cause=missing_rows parent={len(parent_problem_rows)} methods={len(ranked_method_rows)}"
        )
        raise RuntimeError("Phase 2 validation rank summary requires parent and method rows")
    parent_by_problem = {
        str(row["validation_problem_id"]): row
        for row in parent_problem_rows
    }
    if len(parent_by_problem) != len(parent_problem_rows):
        print(
            "[ERROR][Phase2.merged._validation_rank_summary_rows] "
            "cause=duplicate_parent_problem"
        )
        raise RuntimeError("Phase 2 validation parent rows contain duplicate problems")
    method_problem_ids = {
        str(row["validation_problem_id"])
        for row in ranked_method_rows
    }
    if method_problem_ids != set(parent_by_problem):
        print(
            "[ERROR][Phase2.merged._validation_rank_summary_rows] "
            f"cause=problem_set_mismatch parent={sorted(parent_by_problem)} "
            f"methods={sorted(method_problem_ids)}"
        )
        raise RuntimeError("Phase 2 validation parent and method problem sets differ")

    scopes: list[tuple[str, int | str, int | str, set[str]]] = [
        ("overall", "", "", set(parent_by_problem)),
    ]
    for block_count in sorted(
        {int(row["block_count"]) for row in parent_problem_rows}
    ):
        scopes.append(
            (
                "block_size",
                block_count,
                "",
                {
                    str(row["validation_problem_id"])
                    for row in parent_problem_rows
                    if int(row["block_count"]) == block_count
                },
            )
        )
    for distribution_type in sorted(
        {int(row["distribution_type"]) for row in parent_problem_rows}
    ):
        scopes.append(
            (
                "distribution_type",
                "",
                distribution_type,
                {
                    str(row["validation_problem_id"])
                    for row in parent_problem_rows
                    if int(row["distribution_type"]) == distribution_type
                },
            )
        )

    train_episodes = {int(row["train_episode"]) for row in parent_problem_rows}
    if len(train_episodes) != 1:
        print(
            "[ERROR][Phase2.merged._validation_rank_summary_rows] "
            f"cause=mixed_train_episodes values={sorted(train_episodes)}"
        )
        raise RuntimeError("Phase 2 validation rank summary mixes checkpoints")
    train_episode = next(iter(train_episodes))
    sources = sorted({str(row["source"]) for row in ranked_method_rows})
    result: list[dict] = []
    for scope, block_count, distribution_type, problem_ids in scopes:
        parents = [parent_by_problem[problem_id] for problem_id in sorted(problem_ids)]
        methods = [
            row
            for row in ranked_method_rows
            if str(row["validation_problem_id"]) in problem_ids
        ]
        relations = [str(row["proposed_vs_best_heuristic"]) for row in parents]
        invalid_relations = sorted(set(relations) - {"win", "tie", "loss"})
        if invalid_relations:
            print(
                "[ERROR][Phase2.merged._validation_rank_summary_rows] "
                f"cause=invalid_relations scope={scope} values={invalid_relations}"
            )
            raise RuntimeError("Phase 2 validation rank summary has invalid relations")
        methods_by_source = {
            source: [
                row
                for row in methods
                if str(row["source"]) == source
            ]
            for source in sources
        }
        invalid_counts = {
            source: len(source_rows)
            for source, source_rows in methods_by_source.items()
            if len(source_rows) != len(problem_ids)
        }
        if invalid_counts:
            print(
                "[ERROR][Phase2.merged._validation_rank_summary_rows] "
                f"cause=method_problem_count_mismatch scope={scope} counts={invalid_counts} "
                f"expected={len(problem_ids)}"
            )
            raise RuntimeError("Phase 2 validation method grid is incomplete")
        winner_counts = {
            source: sum(
                int(row["method_rank"]) == 1
                for row in methods_by_source[source]
            )
            for source in sources
        }
        sole_winner_counts = {
            source: sum(
                int(row["method_rank"]) == 1
                and int(row["rank1_tie_count"]) == 1
                for row in methods_by_source[source]
            )
            for source in sources
        }
        tied_winner_counts = {
            source: winner_counts[source] - sole_winner_counts[source]
            for source in sources
        }
        mean_ranks = {
            source: round(
                sum(
                    int(row["method_rank"])
                    for row in methods_by_source[source]
                )
                / len(problem_ids),
                6,
            )
            for source in sources
        }
        proposed_ranks = [int(row["proposed_best_rank"]) for row in parents]
        result.append(
            {
                "train_episode": train_episode,
                "scope": scope,
                "block_count": block_count,
                "distribution_type": distribution_type,
                "problem_count": len(problem_ids),
                "proposed_win_count": relations.count("win"),
                "proposed_tie_count": relations.count("tie"),
                "proposed_loss_count": relations.count("loss"),
                "proposed_rank1_count": sum(rank == 1 for rank in proposed_ranks),
                "proposed_mean_rank": round(
                    sum(proposed_ranks) / len(proposed_ranks),
                    6,
                ),
                "winner_counts_json": json.dumps(
                    winner_counts,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "sole_winner_counts_json": json.dumps(
                    sole_winner_counts,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "tied_winner_counts_json": json.dumps(
                    tied_winner_counts,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "mean_rank_by_method_json": json.dumps(
                    mean_ranks,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return result


def _print_validation_rank_summary(rows: Sequence[Mapping]) -> None:
    for row in rows:
        scope_key = str(row["scope"])
        if row["block_count"] != "":
            scope_key += f":{row['block_count']}"
        if row["distribution_type"] != "":
            scope_key += f":{row['distribution_type']}"
        print(
            "[VALIDATION][Phase2.merged.validation_rank] "
            f"episode={row['train_episode']} scope={scope_key} "
            f"problems={row['problem_count']} "
            f"proposed_wtl={row['proposed_win_count']}/"
            f"{row['proposed_tie_count']}/{row['proposed_loss_count']} "
            f"proposed_rank1={row['proposed_rank1_count']} "
            f"proposed_mean_rank={row['proposed_mean_rank']} "
            f"winner_counts={row['winner_counts_json']}"
        )


def _validation_rank_summary_fields() -> list[str]:
    return [
        "train_episode",
        "scope",
        "block_count",
        "distribution_type",
        "problem_count",
        "proposed_win_count",
        "proposed_tie_count",
        "proposed_loss_count",
        "proposed_rank1_count",
        "proposed_mean_rank",
        "winner_counts_json",
        "sole_winner_counts_json",
        "tied_winner_counts_json",
        "mean_rank_by_method_json",
    ]


def _write_validation_rank_history(
    plt,
    validation_root: Path,
) -> dict[str, str]:
    """모든 checkpoint의 전체 winner count와 Proposed rank 추세를 기록한다."""

    evaluation_dirs = sorted(
        path
        for path in (validation_root / "evaluations").glob("ep_*")
        if path.is_dir()
    )
    if not evaluation_dirs:
        print(
            "[ERROR][Phase2.merged._write_validation_rank_history] "
            f"cause=no_evaluation_directories root={validation_root}"
        )
        raise RuntimeError("Phase 2 validation rank history requires evaluation directories")
    history_rows: list[dict] = []
    for evaluation_dir in evaluation_dirs:
        summary_path = evaluation_dir / "aggregate" / "validation_rank_summary.csv"
        if not summary_path.is_file():
            print(
                "[ERROR][Phase2.merged._write_validation_rank_history] "
                f"cause=missing_rank_summary path={summary_path}"
            )
            raise RuntimeError("Phase 2 validation checkpoint rank summary is missing")
        with summary_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = [dict(row) for row in csv.DictReader(file)]
        episodes = {int(row["train_episode"]) for row in rows}
        expected_episode = int(evaluation_dir.name.removeprefix("ep_"))
        if episodes != {expected_episode}:
            print(
                "[ERROR][Phase2.merged._write_validation_rank_history] "
                f"cause=episode_mismatch directory={evaluation_dir.name} rows={sorted(episodes)}"
            )
            raise RuntimeError("Phase 2 validation rank summary episode mismatch")
        history_rows.extend(rows)

    history_csv = validation_root / "validation_rank_history.csv"
    _write_rows(
        history_csv,
        history_rows,
        _validation_rank_summary_fields(),
    )
    overall_rows = sorted(
        (row for row in history_rows if row["scope"] == "overall"),
        key=lambda row: int(row["train_episode"]),
    )
    if len(overall_rows) != len(evaluation_dirs):
        print(
            "[ERROR][Phase2.merged._write_validation_rank_history] "
            f"cause=overall_row_count_mismatch expected={len(evaluation_dirs)} "
            f"actual={len(overall_rows)}"
        )
        raise RuntimeError("Phase 2 validation history requires one overall row per checkpoint")

    winner_counts_by_episode: list[dict[str, int]] = []
    for row in overall_rows:
        try:
            parsed = json.loads(str(row["winner_counts_json"]))
            winner_counts = {
                str(source): int(count)
                for source, count in parsed.items()
            }
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(
                "[ERROR][Phase2.merged._write_validation_rank_history] "
                f"cause=invalid_winner_counts value={row.get('winner_counts_json')}"
            )
            raise RuntimeError("invalid Phase 2 validation winner count JSON") from exc
        winner_counts_by_episode.append(winner_counts)
    method_sets = {tuple(sorted(counts)) for counts in winner_counts_by_episode}
    if len(method_sets) != 1:
        print(
            "[ERROR][Phase2.merged._write_validation_rank_history] "
            f"cause=method_set_mismatch values={sorted(method_sets)}"
        )
        raise RuntimeError("Phase 2 validation history method set changed")

    episodes = [int(row["train_episode"]) for row in overall_rows]
    methods = list(next(iter(method_sets)))
    winner_png = validation_root / "validation_overall_winner_count_history.png"
    plt.figure(figsize=(11, 6))
    for method in methods:
        plt.plot(
            episodes,
            [counts[method] for counts in winner_counts_by_episode],
            marker="o",
            linewidth=1.8,
            label=_display_source_name(method),
        )
    plt.plot(
        episodes,
        [int(row["problem_count"]) for row in overall_rows],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Total problems",
    )
    plt.title("Overall validation rank-1 count by checkpoint")
    plt.xlabel("Training episode")
    plt.ylabel("Rank-1 problem count")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(winner_png, dpi=160)
    plt.close()

    proposed_rank_png = validation_root / "validation_proposed_mean_rank_history.png"
    plt.figure(figsize=(11, 5))
    plt.plot(
        episodes,
        [float(row["proposed_mean_rank"]) for row in overall_rows],
        marker="D",
        linewidth=2.0,
        color="#1f77b4",
        label="Proposed mean rank",
    )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Rank 1")
    plt.gca().invert_yaxis()
    plt.title("Proposed overall validation mean rank by checkpoint")
    plt.xlabel("Training episode")
    plt.ylabel("Mean rank (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(proposed_rank_png, dpi=160)
    plt.close()
    return {
        "validation_rank_history_csv": str(history_csv),
        "validation_winner_count_history_png": str(winner_png),
        "validation_proposed_mean_rank_history_png": str(proposed_rank_png),
    }


def _write_validation_graph_hierarchy(
    evaluation_root: Path,
    parent_problem_rows: Sequence[Mapping],
    parent_method_rows: Sequence[Mapping],
    bay_method_rows: Sequence[Mapping],
    score_field_names: Sequence[str],
) -> dict[str, str]:
    """checkpoint 검증 결과를 전체 크기, parent Type, Bay Type 그래프로 기록한다."""

    if not parent_method_rows:
        print("[ERROR][Phase2.merged._write_validation_graph_hierarchy] cause=no_parent_method_rows")
        raise RuntimeError("Phase 2 validation graph hierarchy requires parent method rows")
    if not bay_method_rows:
        print("[ERROR][Phase2.merged._write_validation_graph_hierarchy] cause=no_bay_method_rows")
        raise RuntimeError("Phase 2 validation graph hierarchy requires Bay method rows")
    _validate_validation_method_grid(parent_method_rows, "parent")
    ranked_parent_method_rows = _rank_validation_method_rows(parent_method_rows)
    rank_summary_rows = _validation_rank_summary_rows(
        parent_problem_rows,
        ranked_parent_method_rows,
    )
    metric_specs = _validation_graph_metric_specs(score_field_names)
    aggregate_root = evaluation_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    summary_rows = _validation_size_method_summary(parent_method_rows, score_field_names)
    summary_fields = [
        "block_count",
        "source",
        "type_count",
        *[
            column
            for field in score_field_names
            for column in (f"{field}_mean", f"{field}_std")
        ],
    ]
    summary_csv = aggregate_root / "method_summary_by_block_size.csv"
    _write_rows(summary_csv, summary_rows, summary_fields)
    _write_rows(
        aggregate_root / "method_problem_scores.csv",
        ranked_parent_method_rows,
        [
            "train_episode",
            "validation_episode",
            "validation_problem_id",
            "block_count",
            "distribution_type",
            "generation_seed",
            "source",
            "score_json",
            "method_rank",
            "is_rank1",
            "rank1_tie_count",
            *score_field_names,
        ],
    )
    rank_summary_csv = aggregate_root / "validation_rank_summary.csv"
    _write_rows(
        rank_summary_csv,
        rank_summary_rows,
        _validation_rank_summary_fields(),
    )
    _print_validation_rank_summary(rank_summary_rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:
        print(
            "[ERROR][Phase2.merged._write_validation_graph_hierarchy] "
            f"cause=matplotlib_import_failed error={exc}"
        )
        raise RuntimeError("matplotlib is required to write Phase 2 validation plots") from exc

    paths = {
        "validation_method_summary_csv": str(summary_csv),
        "validation_rank_summary_csv": str(rank_summary_csv),
    }
    for output_key, field, stem, title, ylabel in metric_specs:
        path = aggregate_root / f"{stem}_by_block_size_boxplot.png"
        _plot_validation_block_size_boxplot(
            plt,
            Patch,
            path,
            parent_method_rows,
            field,
            title,
            ylabel,
        )
        paths[output_key] = str(path)

    block_counts = sorted(
        {int(row["block_count"]) for row in parent_method_rows}
    )
    for block_count in block_counts:
        parent_rows = [
            row
            for row in parent_method_rows
            if int(row["block_count"]) == block_count
        ]
        _write_validation_type_graphs(
            plt=plt,
            output_root=evaluation_root / f"blocks_{block_count:03d}" / "by_type" / "parent",
            method_rows=parent_rows,
            score_field_names=score_field_names,
            metric_specs=metric_specs,
            bay_id="parent",
        )
        bay_ids = sorted(
            {
                str(row["bay_id"])
                for row in bay_method_rows
                if int(row["block_count"]) == block_count
            }
        )
        for bay_id in bay_ids:
            rows = [
                row
                for row in bay_method_rows
                if int(row["block_count"]) == block_count
                and str(row["bay_id"]) == bay_id
            ]
            _write_validation_type_graphs(
                plt=plt,
                output_root=(
                    evaluation_root
                    / f"blocks_{block_count:03d}"
                    / "by_type"
                    / f"bay_{bay_id}"
                ),
                method_rows=rows,
                score_field_names=score_field_names,
                metric_specs=metric_specs,
                bay_id=bay_id,
            )
    paths.update(
        _write_validation_rank_history(
            plt,
            evaluation_root.parents[1],
        )
    )
    return paths


def _validation_graph_metric_specs(
    score_field_names: Sequence[str],
) -> list[tuple[str, str, str, str, str]]:
    """hard violation을 제외한 Phase 2 공개 그래프 지표를 확정한다."""

    fields = set(score_field_names)
    if "makespan" not in fields:
        print(
            "[ERROR][Phase2.merged._validation_graph_metric_specs] "
            "cause=missing_metric fragment=makespan"
        )
        raise RuntimeError("Phase 2 validation graph requires makespan")

    def field_with(fragment: str) -> str:
        matches = [field for field in score_field_names if fragment in field]
        if len(matches) != 1:
            print(
                "[ERROR][Phase2.merged._validation_graph_metric_specs] "
                f"cause=metric_cardinality fragment={fragment} count={len(matches)}"
            )
            raise RuntimeError(f"Phase 2 validation graph metric is missing or duplicated: {fragment}")
        return matches[0]

    cut_field = field_with("cut_length_gap")
    wo_field = field_with("wo_count_gap")
    bevel_field = field_with("bevel_quantity_gap")
    occupancy_field = field_with("occupancy_gap")
    normalized = any("normalized" in field for field in score_field_names)
    gap_prefix = "Normalized " if normalized else ""
    return [
        ("validation_makespan_png", "makespan", "makespan", "Makespan", "Minutes"),
        (
            "validation_cut_gap_png",
            cut_field,
            "cut_length_gap",
            "CUT_LTH load gap",
            f"{gap_prefix}CUT_LTH gap",
        ),
        (
            "validation_wo_gap_png",
            wo_field,
            "wo_count_gap",
            "W/O count load gap",
            f"{gap_prefix}W/O count gap",
        ),
        (
            "validation_bevel_gap_png",
            bevel_field,
            "bevel_quantity_gap",
            "BV_QTY load gap",
            f"{gap_prefix}BV_QTY gap",
        ),
        (
            "validation_occupancy_gap_png",
            occupancy_field,
            "occupancy_gap",
            "Machine occupancy gap",
            f"{gap_prefix}occupancy gap",
        ),
    ]


def _validate_validation_method_grid(
    method_rows: Sequence[Mapping],
    context: str,
) -> None:
    """모든 method가 동일한 실제 문제 key를 한 번씩 갖는지 검사한다."""

    if not method_rows:
        print(
            "[ERROR][Phase2.merged._validate_validation_method_grid] "
            f"cause=no_rows context={context}"
        )
        raise RuntimeError("Phase 2 validation method grid is empty")
    keys_by_source: dict[str, list[tuple[int, int]]] = {}
    for row in method_rows:
        source = str(row["source"])
        keys_by_source.setdefault(source, []).append(
            (int(row["block_count"]), int(row["distribution_type"]))
        )
    reference: set[tuple[int, int]] | None = None
    for source, keys in sorted(keys_by_source.items()):
        unique_keys = set(keys)
        if len(keys) != len(unique_keys):
            print(
                "[ERROR][Phase2.merged._validate_validation_method_grid] "
                f"cause=duplicate_problem context={context} source={source}"
            )
            raise RuntimeError("Phase 2 validation method grid contains duplicate problems")
        if reference is None:
            reference = unique_keys
        elif unique_keys != reference:
            print(
                "[ERROR][Phase2.merged._validate_validation_method_grid] "
                f"cause=problem_set_mismatch context={context} source={source}"
            )
            raise RuntimeError("Phase 2 validation methods do not cover the same problems")


def _write_validation_type_graphs(
    plt,
    output_root: Path,
    method_rows: Sequence[Mapping],
    score_field_names: Sequence[str],
    metric_specs: Sequence[tuple[str, str, str, str, str]],
    bay_id: str,
) -> None:
    """한 block size의 Type별 parent 또는 Bay 비교표와 PNG를 기록한다."""

    _validate_validation_method_grid(method_rows, f"by_type:{bay_id}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            **dict(row),
            "bay_id": bay_id,
        }
        for row in sorted(
            method_rows,
            key=lambda item: (
                int(item["distribution_type"]),
                str(item["source"]),
            ),
        )
    ]
    _write_rows(
        output_root / "method_summary_by_type.csv",
        rows,
        [
            "train_episode",
            "validation_episode",
            "validation_problem_id",
            "block_count",
            "distribution_type",
            "generation_seed",
            "bay_id",
            "source",
            "score_json",
            *score_field_names,
        ],
    )
    for _output_key, field, stem, title, ylabel in metric_specs:
        _plot_validation_type_metric(
            plt=plt,
            path=output_root / f"{stem}_by_type.png",
            rows=rows,
            field=field,
            title=title,
            ylabel=ylabel,
            bay_id=bay_id,
        )


def _validation_size_method_summary(
    method_rows: Sequence[Mapping],
    score_field_names: Sequence[str],
) -> list[dict]:
    groups: dict[tuple[int, str], list[Mapping]] = {}
    for row in method_rows:
        groups.setdefault(
            (int(row["block_count"]), str(row["source"])),
            [],
        ).append(row)
    result: list[dict] = []
    for (block_count, source), rows in sorted(groups.items()):
        summary = {
            "block_count": block_count,
            "source": source,
            "type_count": len(rows),
        }
        for field in score_field_names:
            values = [float(row[field]) for row in rows]
            mean = sum(values) / len(values)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_std"] = math.sqrt(
                sum((value - mean) ** 2 for value in values) / len(values)
            )
        result.append(summary)
    return result


def _plot_validation_block_size_boxplot(
    plt,
    patch_type,
    path: Path,
    rows: Sequence[Mapping],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    """Type 값들을 표본으로 사용해 method별 block-size boxplot을 그린다."""

    _validate_validation_method_grid(rows, "block_size_boxplot")
    block_counts = sorted({int(row["block_count"]) for row in rows})
    type_sets = {
        block_count: {
            int(row["distribution_type"])
            for row in rows
            if int(row["block_count"]) == block_count
        }
        for block_count in block_counts
    }
    if len({tuple(sorted(types)) for types in type_sets.values()}) != 1:
        print(
            "[ERROR][Phase2.merged._plot_validation_block_size_boxplot] "
            f"cause=type_set_mismatch values={type_sets}"
        )
        raise RuntimeError("Phase 2 validation block sizes must share the same Type set")
    sources = _ordered_sources(row["source"] for row in rows)
    base_positions = list(range(1, len(block_counts) + 1))
    group_width = 0.82
    source_width = group_width / len(sources)
    box_width = source_width * 0.82
    colors = [plt.get_cmap("tab20")(index % 20) for index in range(len(sources))]
    plt.figure(figsize=(max(12, len(block_counts) * 2.0), 6))
    for source_index, source in enumerate(sources):
        datasets = [
            [
                float(row[field])
                for row in rows
                if str(row["source"]) == source
                and int(row["block_count"]) == block_count
            ]
            for block_count in block_counts
        ]
        if any(not values for values in datasets):
            print(
                "[ERROR][Phase2.merged._plot_validation_block_size_boxplot] "
                f"cause=missing_method_size_values source={source} field={field}"
            )
            raise RuntimeError("Phase 2 validation boxplot has an empty method-size group")
        offset = (source_index - (len(sources) - 1) / 2.0) * source_width
        boxplot = plt.boxplot(
            datasets,
            positions=[position + offset for position in base_positions],
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 3,
            },
            medianprops={"color": "black", "linewidth": 1.0},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(colors[source_index])
            box.set_alpha(0.85)
    plt.xticks(base_positions, [str(block_count) for block_count in block_counts])
    plt.xlabel("Number of physical blocks")
    plt.ylabel(ylabel)
    plt.title(f"{title} by problem size (Type distribution)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(
        handles=[
            patch_type(
                facecolor=colors[index],
                edgecolor="black",
                label=_display_source_name(source),
            )
            for index, source in enumerate(sources)
        ],
        fontsize=8,
        ncol=min(5, len(sources)),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
    )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_validation_type_metric(
    plt,
    path: Path,
    rows: Sequence[Mapping],
    field: str,
    title: str,
    ylabel: str,
    bay_id: str,
) -> None:
    """고정 block size 안에서 Type별 method 값을 선·점으로 비교한다."""

    sources = _ordered_sources(row["source"] for row in rows)
    distribution_types = sorted(
        {int(row["distribution_type"]) for row in rows}
    )
    markers = ["D", "s", "^", "o", "v", "P", "X", "*", "h", "p", "<", ">"]
    plt.figure(figsize=(max(9, len(distribution_types) * 1.4), 5))
    for source_index, source in enumerate(sources):
        source_rows = sorted(
            (
                row
                for row in rows
                if str(row["source"]) == source
            ),
            key=lambda row: int(row["distribution_type"]),
        )
        plt.plot(
            [int(row["distribution_type"]) for row in source_rows],
            [float(row[field]) for row in source_rows],
            marker=markers[source_index % len(markers)],
            linewidth=1.3,
            markersize=6,
            label=_display_source_name(source),
        )
    plt.xticks(
        distribution_types,
        [f"Type {distribution_type}" for distribution_type in distribution_types],
    )
    plt.xlabel("Validation distribution Type")
    plt.ylabel(ylabel)
    scope = "Parent" if bay_id == "parent" else f"Bay {bay_id}"
    plt.title(f"{scope}: {title} by Type")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=min(4, len(sources)))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


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
        "validation_problem_id",
        "block_count",
        "distribution_type",
        "generation_seed",
        "validation_phase1_seed",
        "validation_sampling_seed",
        "bay_id",
        "job_count",
        "machine_count",
        "candidate_count",
        "best_source",
        "best_score_json",
        "best_heuristic_source",
        "best_heuristic_score_json",
        "proposed_vs_best_heuristic",
        "agent_best_rank",
        "agent_best_source",
        "agent_best_score_json",
        "proposed_best_rank",
        "proposed_best_source",
        "proposed_best_score_json",
        "greedy_rank",
        "greedy_source",
        "greedy_score_json",
        "checkpoint_selection_key_json",
        "best_checkpoint_saved",
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
        "validation_problem_id",
        "block_count",
        "distribution_type",
        "generation_seed",
        "validation_sampling_seed",
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


def _validation_parent_fields(score_field_names: Sequence[str]) -> list[str]:
    return [
        "train_episode",
        "validation_episode",
        "validation_problem_id",
        "block_count",
        "distribution_type",
        "generation_seed",
        "job_count",
        "bay_subproblem_count",
        "best_source",
        "best_score_json",
        "best_heuristic_source",
        "best_heuristic_score_json",
        "proposed_vs_best_heuristic",
        "proposed_best_rank",
        "proposed_best_source",
        "proposed_best_score_json",
        "checkpoint_selection_key_json",
        "best_checkpoint_saved",
        *score_field_names,
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
        key = f"{row['validation_episode']}::{row['bay_id']}"
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

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            int(row["validation_episode"]),
            str(row["bay_id"]),
            str(row["source"]),
        ),
    )
    validation_problems = sorted(
        {
            (int(row["validation_episode"]), str(row["bay_id"]))
            for row in ordered_rows
        }
    )
    problem_index = {
        problem: index + 1 for index, problem in enumerate(validation_problems)
    }
    sources = _ordered_sources(row["source"] for row in ordered_rows)
    markers = ["D", "s", "^", "o", "v", "P", "X", "*", "h", "p"]
    plt.figure(figsize=(11, 5))
    for source_index, source in enumerate(sources):
        source_rows = [row for row in ordered_rows if str(row["source"]) == source]
        x_values = [
            problem_index[(int(row["validation_episode"]), str(row["bay_id"]))]
            for row in source_rows
        ]
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
    plt.xticks(
        list(problem_index.values()),
        [f"VAL{episode:02d}-B{bay_id}" for episode, bay_id in validation_problems],
        rotation=45,
        ha="right",
    )
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

    ordered_rows = sorted(
        rows,
        key=lambda row: (int(row["validation_episode"]), str(row["bay_id"])),
    )
    x_values = list(range(1, len(ordered_rows) + 1))
    labels = [
        f"VAL{int(row['validation_episode']):02d}-B{row['bay_id']}"
        for row in ordered_rows
    ]
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
        "balanced_tact_load": "BalancedTact",
        "spt_batch": "SPTBatch",
        "lpt_batch": "LPTBatch",
        "long_cut_batch": "LongCutBatch",
        "short_cut_batch": "ShortCutBatch",
        "long_bevel_batch": "LongBevelBatch",
        "short_bevel_batch": "ShortBevelBatch",
        "worst_fit_lth": "WorstFitLTH",
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
