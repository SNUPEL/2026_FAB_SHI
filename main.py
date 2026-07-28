"""프로젝트 공통 실행 진입점.

초보자 기준 실행 예시:
  python3 main.py generate-phase1-blocks --n-blocks 40 --seed 0 --output-dir output/blocks
  python3 main.py phase1-plan-multi-series --block-xlsx input/260724_절단블록_데이터_None.xlsx --wo-xlsx input/260724_절단WO_데이터_None.xlsx --output-dir output/plan
  python3 main.py phase1-train-pair-self-labeling --config config_mixed.yaml --output-dir output/p1
  python3 main.py phase2-train-batch-machine-self-labeling --config config_mixed.yaml --output-dir output/p2
  python3 main.py phase2-run-full-workflow --config config_mixed.yaml --output-dir output/full
"""

# LINE-BY-LINE: Windows conda에서 pandas/numpy와 torch가 서로 다른 Intel OpenMP runtime을 초기화하면
# `libiomp5md.dll already initialized`로 학습이 중단될 수 있습니다.
# 사용: 반드시 torch/numpy/pandas import보다 먼저 설정해야 하며, 판넬라인 PPO eval runner와 같은 실행 보호 장치입니다.
import os

# LINE-BY-LINE: 사용자가 외부에서 명시한 값은 존중하고, 없는 경우에만 Windows OpenMP 중복 초기화 허용값을 설정합니다.
if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("[CHECK][main.openmp_guard] KMP_DUPLICATE_LIB_OK=TRUE")

# LINE-BY-LINE: Windows conda의 MKL은 numpy.linalg.lstsq(LAPACK gelsd)의 내부 멀티스레딩에서
# 간헐적 native crash(faulthandler C-stack dump)를 냅니다. 원인은 앱 코드가 아니라 MKL 스레딩 경합입니다.
# 사용: MKL을 단일 스레드/sequential로 고정해 이 경합 자체를 제거합니다. torch intra-op 병렬성(OMP_NUM_THREADS)은
# 건드리지 않아 학습 성능에 영향이 없습니다. 반드시 numpy/torch import보다 먼저 설정해야 합니다. 사용자 지정값은 존중합니다.
if os.name == "nt":
    _mkl_guarded = False
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "1"
        _mkl_guarded = True
    if "MKL_THREADING_LAYER" not in os.environ:
        os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        _mkl_guarded = True
    if _mkl_guarded:
        print("[CHECK][main.openmp_guard] MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL")

# LINE-BY-LINE: `argparse` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import argparse
# LINE-BY-LINE: `json` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import json
# LINE-BY-LINE: `collections` 모듈에서 `Counter`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from collections import Counter
from dataclasses import asdict
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from Environment.constraints.profiles import load_phase_constraint_profile
from Phase1.orchestrator import candidate_to_phase1_plan
from Phase2.orchestrator import run_phase2_full_graph_workflow, write_phase2_workflow_outputs
from Phase2.feedback import (
    build_frozen_phase2_schedule_feedback_scorer,
    build_phase2_feedback_contract,
)
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK,
    build_mixed_phase2_training_machines,
    create_phase2_candidate_executor,
    phase2_score_field_names,
    train_phase2_batch_machine_self_labeling,
)
from Phase2.run_spec import build_phase2_run_spec, require_matching_phase2_run_spec
from Phase1.pair_self_labeling import (
    run_phase1_pair_policy_resource_pool_best_of_k,
    train_phase1_pair_self_labeling,
)
from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    score_phase1_bay_loads,
    run_phase1_resource_pool_heuristic_candidate,
)
# LINE-BY-LINE: `Utils.config` 모듈에서 `load_config`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.config import load_config
from Utils.data.cutting_start_date import (
    audit_cutting_start_dates,
    write_cutting_start_date_audit,
)
from Utils.data.multi_series_cutting_data import (
    MIXED_PLANNING_MACHINE_IDS_BY_BAY,
    load_multi_series_cutting_data,
)
from Utils.learning.phase_agent_checkpoints import (
    load_phase1_feedback_contract,
    load_phase1_pair_pointer_checkpoint,
    load_phase2_checkpoint_run_spec,
    load_phase2_set_pointer_checkpoint,
)
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs
from Utils.phase1.phase1_bay_balancer import write_phase1_bay_plan
from Utils.phase1.multi_series_planner import (
    build_multi_series_phase1_daily_plans,
    write_multi_series_phase1_daily_plans,
)
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    PHASE1_OBJECTIVE_SCOPES,
    joint_phase1_bay_capacity_weights,
)
from Utils.data.report_formula_data_generator import (
    BTH_FORMULA_FEATURES,
    TACT_A_CUT,
    TACT_A_MARK,
    TACT_A_PTLST,
    TACT_A_THK,
    scenario_jobs_from_report_formula_jobs,
)
from Utils.data.multi_series_formula_data_generator import (
    generate_multi_series_formula_data,
    load_multi_series_generation_profile,
    load_physical_block_joint_profile,
)


# MIXED 합성 episode의 출처 식별자. CLI 로그와 scenario metadata가 같은 값을 써야
# 산출물에서 학습 입력을 역추적할 수 있다.
MIXED_SYNTHETIC_SOURCE = "mixed_physical_block_joint_distribution"

# episode seed 규칙. Phase 1과 Phase 2가 같은 값을 써야 두 phase의 train/validation
# 분리가 일치하고 gap 지표를 서로 비교할 수 있다. 한쪽만 바꾸면 실패가 아니라
# 조용히 잘못된 지표가 나온다.
VALIDATION_SEED_OFFSET = 10_000_000
EPISODE_SEED_STRIDE = 1_000_003


def parse_csv_argument(value: str | None, default: Sequence[str]) -> list[str]:
    """쉼표로 구분한 CLI 값을 공백 없는 문자열 목록으로 변환한다."""

    if value is None or str(value).strip() == "":
        return list(default)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        print(f"[ERROR][main.parse_csv_argument] cause=empty_csv_argument input={value!r}")
        raise ValueError("comma-separated argument produced no items")
    return items










def command_phase1_plan_multi_series(args: argparse.Namespace) -> None:
    """신규 다계열 Excel을 검증하고 날짜 audit와 Phase 1 계획을 저장한다."""

    print("[phase1-plan-multi-series]")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- wo_xlsx: {args.wo_xlsx}")
    print(f"- output_dir: {args.output_dir}")
    data = load_multi_series_cutting_data(args.block_xlsx, args.wo_xlsx)
    date_audit = audit_cutting_start_dates(data.blocks.to_dict("records"))
    audit_paths = write_cutting_start_date_audit(date_audit, args.output_dir)
    plan = build_multi_series_phase1_daily_plans(data)
    plan_paths = write_multi_series_phase1_daily_plans(
        plan,
        Path(args.output_dir) / "phase1",
    )
    print(f"- block_count: {len(data.blocks)}")
    print(f"- wo_count: {len(data.work_orders)}")
    print(f"- problem_count: {plan['problem_count']}")
    print(f"- matched_source_date_count: {int(date_audit['MATCHED_SOURCE_DATE'].sum())}")
    print(f"- date_audit_csv: {audit_paths['csv']}")
    print(f"- date_audit_summary_json: {audit_paths['summary_json']}")
    print(f"- plan_json: {plan_paths['json']}")
    print(f"- assignments_csv: {plan_paths['assignments_csv']}")
    print(f"- bay_loads_csv: {plan_paths['bay_loads_csv']}")


def command_phase1_train_pair_self_labeling(args: argparse.Namespace) -> None:
    """MIXED 물리 블록 episode로 Phase 1 pair-policy를 학습한다."""

    capacity_weights = joint_phase1_bay_capacity_weights()
    bay_ids = tuple(capacity_weights)
    validation_rollout_samples = (
        args.rollout_samples
        if args.validation_rollout_samples is None
        else args.validation_rollout_samples
    )
    heuristic_alias = str(args.heuristic_algorithms).strip().lower()
    if heuristic_alias in {"all", "mixed3"}:
        heuristic_algorithms = list(PHASE1_HEURISTIC_BANK)
    else:
        heuristic_algorithms = [
            item.strip()
            for item in str(args.heuristic_algorithms).split(",")
            if item.strip()
        ]
    unknown_heuristics = sorted(set(heuristic_algorithms) - set(PHASE1_HEURISTIC_BANK))
    if not heuristic_algorithms or unknown_heuristics:
        print(
            "[ERROR][main.command_phase1_train_pair_self_labeling] "
            f"cause=invalid_mixed_heuristics requested={heuristic_algorithms} "
            f"allowed={list(PHASE1_HEURISTIC_BANK)}"
        )
        raise RuntimeError("Phase 1 MIXED training received unsupported heuristics")
    feedback_requested = bool(args.enable_phase2_feedback_score)
    feedback_checkpoint = _optional_non_empty_cli_value(
        args.phase2_feedback_checkpoint,
        "phase2_feedback_checkpoint",
    )
    if feedback_requested != (feedback_checkpoint is not None):
        print(
            "[ERROR][main.command_phase1_train_pair_self_labeling] "
            f"cause=phase2_feedback_option_mismatch enabled={feedback_requested} "
            f"checkpoint={feedback_checkpoint}"
        )
        raise RuntimeError(
            "Phase 2 feedback requires both --enable-phase2-feedback-score and "
            "--phase2-feedback-checkpoint"
        )

    phase2_feedback_scorer = None
    phase2_feedback_contract = None
    if feedback_requested:
        phase2_context = "phase1_frozen_phase2_feedback"
        phase2_constraint_profile = load_phase_constraint_profile(load_config(args.config), "phase2")
        phase2_model = load_phase2_set_pointer_checkpoint(
            checkpoint_path=feedback_checkpoint,
            context=phase2_context,
        )
        phase2_run_spec = load_phase2_checkpoint_run_spec(
            feedback_checkpoint,
            context=phase2_context,
        )
        phase2_feedback_scorer = build_frozen_phase2_schedule_feedback_scorer(
            model=phase2_model,
            machines=build_mixed_phase2_training_machines(),
            run_spec=phase2_run_spec,
            constraint_profile=phase2_constraint_profile,
        )
        phase2_feedback_contract = build_phase2_feedback_contract(
            feedback_checkpoint,
            phase2_run_spec,
        )

    print("[phase1-train-pair-self-labeling-cli]")
    print(f"- config: {args.config}")
    print(f"- synthetic_source: {MIXED_SYNTHETIC_SOURCE}")
    print("- rule_profile: multi_series_260711")
    print("- score_mode: wo_first")
    print(f"- objective_scope: {args.objective_scope}")
    print(f"- episode_scope: {PHASE1_MULTI_SERIES_SCOPE_VERSION}")
    print(f"- bay_ids: {','.join(bay_ids)}")
    print(f"- bay_capacity_weights: {capacity_weights}")
    print(f"- min_physical_blocks: {args.min_blocks}")
    print(f"- max_physical_blocks: {args.max_blocks}")
    print(f"- episodes: {args.episodes}")
    print(f"- rollout_samples: {args.rollout_samples}")
    print(f"- rollout_samples_validation: {validation_rollout_samples}")
    print(f"- heuristic_algorithms: {','.join(heuristic_algorithms)}")
    print(f"- resume_checkpoint: {args.resume_checkpoint}")
    print(f"- phase2_feedback_enabled: {phase2_feedback_scorer is not None}")
    print(f"- phase2_feedback_checkpoint: {feedback_checkpoint or ''}")
    print(f"- write_candidate_summary: {args.write_candidate_summary}")
    print(f"- device: {args.device}")
    print(f"- output_dir: {args.output_dir}")

    def episode_payload(episode: int, *, validation: bool) -> dict:
        seed_offset = VALIDATION_SEED_OFFSET if validation else 0
        payload = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed + seed_offset + episode * EPISODE_SEED_STRIDE,
            verbose=False,
        )[0]
        episode_id = f"VAL{episode:05d}" if validation else f"EP{episode:05d}"
        metadata = dict(payload["metadata"])
        metadata.update(
            {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "validation_source": "synthetic" if validation else "training",
                "evaluation_input_type": "mixed_physical_block_joint_distribution_wo",
            }
        )
        return {"jobs": payload["jobs"], "metadata": metadata}

    summary = train_phase1_pair_self_labeling(
        episode_jobs=None,
        episode_metadata=None,
        bay_ids=bay_ids,
        output_dir=args.output_dir,
        episodes=args.episodes,
        rollout_samples=args.rollout_samples,
        heuristic_algorithms=heuristic_algorithms,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        seed=args.seed,
        episode_factory=lambda episode: episode_payload(episode, validation=False),
        checkpoint_every=args.checkpoint_every,
        validation_every=args.validation_every,
        validation_episodes=args.validation_episodes,
        validation_episode_factory=(
            (lambda episode: episode_payload(episode, validation=True))
            if args.validation_episodes > 0
            else None
        ),
        validation_rollout_samples=validation_rollout_samples,
        resume_checkpoint=args.resume_checkpoint,
        phase2_feedback_scorer=phase2_feedback_scorer,
        phase2_feedback_contract=phase2_feedback_contract,
        bay_capacity_weights=capacity_weights,
        device=args.device,
        objective_scope=args.objective_scope,
        write_candidate_summary=args.write_candidate_summary,
    )
    print(f"- checkpoint_path: {summary['checkpoint_path']}")
    print(f"- best_checkpoint_path: {summary['best_checkpoint_path']}")
    print(f"- metrics_csv: {summary['metrics_csv']}")
    print(f"- candidate_summary_csv: {summary['candidate_summary_csv']}")
    print(f"- best_action_table_jsonl: {summary['best_action_table_jsonl']}")
    print(f"- validation_summary_csv: {summary['validation_summary_csv']}")
    print(f"- validation_candidate_summary_csv: {summary['validation_candidate_summary_csv']}")
    print(f"- validation_wo_gap_png: {summary.get('validation_wo_gap_png', '')}")
    print(f"- validation_series_wo_gap_png: {summary.get('validation_series_wo_gap_png', '')}")
    print(f"- validation_cut_gap_png: {summary.get('validation_cut_gap_png', '')}")
    print(f"- validation_series_cut_gap_png: {summary.get('validation_series_cut_gap_png', '')}")
    print(f"- validation_bevel_gap_png: {summary.get('validation_bevel_gap_png', '')}")
    print(f"- validation_series_bevel_gap_png: {summary.get('validation_series_bevel_gap_png', '')}")
    print(f"- summary_json: {summary['summary_json']}")
    print(f"- best_source_counts: {summary['best_source_counts']}")


def command_phase2_train_batch_machine_self_labeling(args: argparse.Namespace) -> None:
    """MIXED 공동분포 episode로 merged Phase 2 policy를 학습한다."""

    print("[phase2-train-batch-machine-self-labeling-cli]")
    print(f"- config: {args.config}")
    print(f"- phase1_heuristic: {args.phase1_heuristic}")
    print(f"- phase1_checkpoint: {args.phase1_checkpoint}")
    print(f"- phase1_samples: {args.phase1_samples}")
    print(f"- phase1_temperature: {args.phase1_temperature}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- episodes: {args.episodes}")
    print(f"- hidden_dim: {args.hidden_dim}")
    print(f"- lr: {args.lr}")
    print(f"- seed: {args.seed}")
    print(f"- device: {args.device}")
    print(f"- heuristic_algorithms: {args.heuristic_algorithms}")
    print(f"- phase2_score_mode: {args.phase2_score_mode}")
    print(f"- rollout_samples: {args.rollout_samples}")
    validation_rollout_samples = (
        args.rollout_samples
        if args.validation_rollout_samples is None
        else args.validation_rollout_samples
    )
    print(f"- rollout_samples_validation: {validation_rollout_samples}")
    print(f"- validation_every: {args.validation_every}")
    print(f"- validation_episodes: {args.validation_episodes}")
    print(f"- checkpoint_every: {args.checkpoint_every}")
    print(f"- resume_checkpoint: {args.resume_checkpoint}")
    print(f"- action_pool_limit: {args.action_pool_limit}")
    print(f"- synthetic_source: {MIXED_SYNTHETIC_SOURCE}")
    print(f"- synthetic_physical_block_range: {args.min_blocks}..{args.max_blocks}")

    config = load_config(args.config)
    phase2_constraint_profile = load_phase_constraint_profile(config, "phase2")
    training_machines = build_mixed_phase2_training_machines()
    phase1_bay_capacity_weights = joint_phase1_bay_capacity_weights()
    phase1_bay_ids = tuple(phase1_bay_capacity_weights)
    requested_phase1_heuristic = _optional_non_empty_cli_value(args.phase1_heuristic, "phase1_heuristic")
    phase1_checkpoint = _optional_non_empty_cli_value(args.phase1_checkpoint, "phase1_checkpoint")
    if requested_phase1_heuristic is not None and phase1_checkpoint is not None:
        print(
            "[ERROR][main.command_phase2_train_batch_machine_self_labeling] "
            f"cause=conflicting_phase1_sources phase1_heuristic={requested_phase1_heuristic} "
            f"phase1_checkpoint={phase1_checkpoint}"
        )
        raise RuntimeError("use either --phase1-heuristic or --phase1-checkpoint, not both")
    phase1_heuristic = requested_phase1_heuristic if phase1_checkpoint is None else None
    if phase1_heuristic is None and phase1_checkpoint is None:
        phase1_heuristic = "bevel_first_balanced"
    if phase1_heuristic is not None and phase1_heuristic not in PHASE1_HEURISTIC_BANK:
        print(
            "[ERROR][main.command_phase2_train_batch_machine_self_labeling] "
            f"cause=unsupported_mixed_phase1_heuristic value={phase1_heuristic}"
        )
        raise RuntimeError(f"unsupported MIXED Phase 1 heuristic: {phase1_heuristic}")
    phase1_assignment_builder = None
    if phase1_checkpoint is not None:
        phase1_assignment_builder = _phase1_agent_assignment_builder(
            checkpoint=phase1_checkpoint,
            sample_count=args.phase1_samples,
            temperature=args.phase1_temperature,
            seed=args.seed,
        )
    print(f"- resolved_phase1_mode: {'checkpoint' if phase1_checkpoint is not None else 'heuristic'}")
    print(f"- resolved_phase1_heuristic: {phase1_heuristic or ''}")
    print(f"- resolved_phase1_bay_ids: {phase1_bay_ids}")
    print(f"- phase1_bay_capacity_weights: {phase1_bay_capacity_weights}")
    print(
        "- phase2_hard_constraints: "
        + ",".join(name for name, enabled in phase2_constraint_profile.hard_enabled.items() if enabled)
    )
    def mixed_jobs(seed_value: int) -> Mapping[str, object]:
        payload = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=seed_value,
            verbose=False,
        )[0]
        return payload["jobs"]

    training_jobs = mixed_jobs(args.seed)

    def episode_job_factory(episode: int) -> Mapping[str, object]:
        return mixed_jobs(args.seed + episode * EPISODE_SEED_STRIDE)

    def _validation_episode_jobs(validation_episode: int) -> Mapping[str, object]:
        return mixed_jobs(args.seed + VALIDATION_SEED_OFFSET + validation_episode * EPISODE_SEED_STRIDE)

    validation_episode_job_factory = (
        _validation_episode_jobs if args.validation_episodes > 0 else None
    )

    print(f"- candidate_workers: {args.candidate_workers}")
    candidate_executor = create_phase2_candidate_executor(args.candidate_workers)
    try:
        summary = train_phase2_batch_machine_self_labeling(
            jobs=training_jobs,
            machines=training_machines,
            phase1_assignments={},
            output_dir=args.output_dir,
            episodes=args.episodes,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
            heuristic_algorithms=tuple(parse_csv_argument(args.heuristic_algorithms, default=())),
            rollout_samples=args.rollout_samples,
            validation_rollout_samples=validation_rollout_samples,
            validation_every=args.validation_every,
            validation_episodes=args.validation_episodes,
            checkpoint_every=args.checkpoint_every,
            max_wo_count=args.max_wo_count,
            max_length_sum=args.max_length_sum,
            action_pool_limit=args.action_pool_limit,
            episode_jobs=None,
            episode_job_factory=episode_job_factory,
            validation_episode_jobs=None,
            validation_episode_job_factory=validation_episode_job_factory,
            phase1_heuristic=phase1_heuristic,
            phase1_bay_ids=phase1_bay_ids,
            phase1_assignment_builder=phase1_assignment_builder,
            phase1_bay_capacity_weights=phase1_bay_capacity_weights,
            device=args.device,
            write_candidate_summary=args.write_candidate_summary,
            score_mode=args.phase2_score_mode,
            resume_checkpoint=args.resume_checkpoint,
            constraint_profile=phase2_constraint_profile,
            candidate_workers=args.candidate_workers,
            candidate_executor=candidate_executor,
        )
    finally:
        if candidate_executor is not None:
            candidate_executor.shutdown(wait=True, cancel_futures=True)
    print(f"- job_count: {summary['job_count']}")
    print(f"- machine_count: {summary['machine_count']}")
    print(f"- feature_schema_version: {summary['feature_schema_version']}")
    print(f"- feature_group_dims: {summary['feature_group_dims']}")
    print(f"- score_fields: {summary['score_fields']}")
    print(f"- score_mode: {summary['score_mode']}")
    print(f"- start_episode: {summary['start_episode']}")
    print(f"- resumed_from_episode: {summary['resumed_from_episode']}")
    print(f"- resume_checkpoint: {summary['resume_checkpoint']}")
    print(f"- checkpoint_path: {summary['checkpoint_path']}")
    print(f"- checkpoint_dir: {summary['checkpoint_dir']}")
    print(f"- metrics_csv: {summary['metrics_csv']}")
    print(f"- subproblem_metrics_csv: {summary['subproblem_metrics_csv']}")
    print(f"- candidate_summary_csv: {summary['candidate_summary_csv']}")
    print(f"- write_candidate_summary: {summary['write_candidate_summary']}")
    print(f"- validation_summary_csv: {summary['validation_summary_csv']}")
    print(f"- validation_candidate_summary_csv: {summary['validation_candidate_summary_csv']}")
    print(f"- validation_hard_violation_png: {summary.get('validation_hard_violation_png', '')}")
    print(f"- validation_makespan_png: {summary.get('validation_makespan_png', '')}")
    print(f"- validation_wo_gap_png: {summary.get('validation_wo_gap_png', '')}")
    print(f"- validation_cut_gap_png: {summary.get('validation_cut_gap_png', '')}")
    print(f"- validation_bevel_gap_png: {summary.get('validation_bevel_gap_png', '')}")
    print(f"- validation_occupancy_gap_png: {summary.get('validation_occupancy_gap_png', '')}")
    print(f"- validation_best_source_counts_png: {summary.get('validation_best_source_counts_png', '')}")
    print(f"- validation_policy_rank_png: {summary.get('validation_policy_rank_png', '')}")
    print(f"- best_assignment_csv: {summary['best_assignment_csv']}")
    print(f"- best_batches_csv: {summary['best_batches_csv']}")
    print(f"- best_timeline_csv: {summary['best_timeline_csv']}")
    print(f"- summary_json: {summary['summary_json']}")


def command_phase2_run_full_workflow(args: argparse.Namespace) -> None:
    """MIXED 물리 블록 episode를 Phase 1과 merged Phase 2로 연속 실행한다."""

    print("[phase2-run-full-workflow-cli]")
    print(f"- config: {args.config}")
    print(f"- synthetic_source: {MIXED_SYNTHETIC_SOURCE}")
    print(f"- synthetic_physical_blocks: {args.synthetic_blocks}")
    print(f"- phase1_plan: {args.phase1_plan}")
    print(f"- phase1_checkpoint: {args.phase1_checkpoint}")
    print(f"- phase1_heuristic: {args.phase1_heuristic}")
    print(f"- phase1_samples: {args.phase1_samples}")
    print(f"- batch_machine_heuristic: {args.batch_machine_heuristic}")
    print(f"- batch_machine_checkpoint: {args.batch_machine_checkpoint}")
    print(f"- output_dir: {args.output_dir}")

    config = load_config(args.config)
    scenario = _build_mixed_full_flow_scenario(
        physical_block_count=args.synthetic_blocks,
        seed=args.seed,
    )
    phase2_constraint_profile = load_phase_constraint_profile(config, "phase2")
    batch_machine_heuristic = _optional_non_empty_cli_value(
        args.batch_machine_heuristic,
        "batch_machine_heuristic",
    )
    checkpoint_path = _optional_non_empty_cli_value(
        args.batch_machine_checkpoint,
        "batch_machine_checkpoint",
    )
    if checkpoint_path is not None and batch_machine_heuristic is not None:
        print(
            "[ERROR][main.command_phase2_run_full_workflow] "
            f"cause=checkpoint_heuristic_conflict checkpoint={checkpoint_path} "
            f"heuristic={batch_machine_heuristic}"
        )
        raise RuntimeError("Choose either a Phase 2 checkpoint or heuristic, not both")
    if checkpoint_path is None and batch_machine_heuristic is None:
        batch_machine_heuristic = "min_makespan"
    # checkpoint/heuristic 상호배타는 바로 위에서 이미 검사했다.
    batch_machine_model = (
        load_phase2_set_pointer_checkpoint(
            checkpoint_path=checkpoint_path,
            context="phase2_batch_machine",
        )
        if checkpoint_path is not None
        else None
    )
    checkpoint_run_spec = (
        load_phase2_checkpoint_run_spec(checkpoint_path, context="phase2_batch_machine")
        if checkpoint_path is not None
        else None
    )
    phase1_bay_capacity_weights = joint_phase1_bay_capacity_weights()
    requested_bay_ids = tuple(phase1_bay_capacity_weights)
    effective_run_spec = _resolve_phase2_full_flow_run_spec(
        args=args,
        checkpoint_spec=checkpoint_run_spec,
        constraint_profile=phase2_constraint_profile,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        heuristic=batch_machine_heuristic,
    )
    phase1_checkpoint_path = _optional_non_empty_cli_value(
        args.phase1_checkpoint,
        "phase1_checkpoint",
    )
    if phase1_checkpoint_path is not None:
        phase1_feedback_contract = load_phase1_feedback_contract(phase1_checkpoint_path)
        if phase1_feedback_contract is not None:
            if checkpoint_path is None or checkpoint_run_spec is None:
                print(
                    "[ERROR][main.command_phase2_run_full_workflow] "
                    "cause=feedback_trained_phase1_without_phase2_checkpoint"
                )
                raise RuntimeError(
                    "feedback-trained Phase 1 requires the exact Phase 2 checkpoint in full-flow"
                )
            selected_phase2_contract = build_phase2_feedback_contract(
                checkpoint_path,
                checkpoint_run_spec,
            )
            _require_matching_phase1_feedback_contract(
                phase1_feedback_contract,
                selected_phase2_contract,
            )
    phase1_plan = _load_or_build_phase1_plan_for_full_flow(
        args,
        scenario,
        bay_ids=requested_bay_ids,
        bay_capacity_weights=phase1_bay_capacity_weights,
    )
    print(f"- resolved_phase1_bay_ids: {requested_bay_ids}")
    print(f"- phase1_bay_capacity_weights: {phase1_bay_capacity_weights}")
    print(f"- phase2_score_mode: {effective_run_spec['score_mode']}")
    print(f"- action_pool_limit: {effective_run_spec['action_pool_limit']}")
    print(f"- max_wo_count: {effective_run_spec['max_wo_count']}")
    print(f"- max_length_sum: {effective_run_spec['max_length_sum']}")
    print(f"- rollout_samples: {effective_run_spec['validation_rollout_samples']}")
    result = run_phase2_full_graph_workflow(
        scenario=scenario,
        phase1_plan=phase1_plan,
        assignment_mode="allowed_bay_ids",
        model=batch_machine_model,
        max_wo_count=int(effective_run_spec["max_wo_count"]),
        max_length_sum=float(effective_run_spec["max_length_sum"]),
        batch_machine_heuristic=batch_machine_heuristic,
        score_mode=str(effective_run_spec["score_mode"]),
        action_pool_limit=effective_run_spec["action_pool_limit"],
        constraint_profile=phase2_constraint_profile,
        effective_run_spec=effective_run_spec,
        rollout_samples=(
            int(effective_run_spec["validation_rollout_samples"])
            if batch_machine_model is not None
            else 0
        ),
        seed=args.seed,
    )
    outputs = write_phase2_workflow_outputs(result, args.output_dir)
    print(f"- assignment_csv: {outputs['assignment_csv']}")
    print(f"- machine_load_csv: {outputs['machine_load_csv']}")
    print(f"- batch_csv: {outputs['batch_csv']}")
    print(f"- timeline_csv: {outputs['timeline_csv']}")
    print(f"- makespan_csv: {outputs['makespan_csv']}")
    print(f"- bay_metric_csv: {outputs['bay_metric_csv']}")
    print(f"- constraint_audit_csv: {outputs['constraint_audit_csv']}")
    print(f"- constraint_violation_count: {outputs['constraint_violation_count']}")
    print(f"- event_log_json: {outputs['event_log_json']}")
    print(f"- event_count: {outputs['event_count']}")
    print(f"- report_json: {outputs['report_json']}")




def _optional_non_empty_cli_value(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        print(f"[ERROR][main._optional_non_empty_cli_value] cause=empty_value name={name}")
        raise RuntimeError(f"{name} must not be empty")
    return stripped


def _parse_optional_positive_int(value: str) -> int | None:
    """Parse CLI positive int, allowing explicit None/all for unlimited mode."""

    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        print(f"[ERROR][main._parse_optional_positive_int] cause=invalid_integer value={value}")
        raise argparse.ArgumentTypeError(f"expected positive integer or None: {value}") from exc
    if parsed <= 0:
        print(f"[ERROR][main._parse_optional_positive_int] cause=non_positive_integer value={value}")
        raise argparse.ArgumentTypeError(f"expected positive integer or None: {value}")
    return parsed






def _phase1_checkpoint_path(value: str) -> Path:
    """Resolve a Phase 1 checkpoint file or output directory."""

    path = Path(value)
    if path.is_dir():
        path = path / "phase1_pair_pointer_best.pt"
    if not path.exists():
        print(f"[ERROR][main._phase1_checkpoint_path] cause=missing_phase1_checkpoint path={path}")
        raise RuntimeError(f"missing Phase 1 checkpoint: {path}")
    return path


def _phase1_agent_assignment_builder(
    checkpoint: str,
    sample_count: int,
    temperature: float,
    seed: int,
) -> Callable[[Mapping[str, object], int], Mapping[str, str]]:
    """MIXED Phase 1 frozen agent를 Phase 2 upstream assignment로 연결한다."""

    if sample_count <= 0:
        print(f"[ERROR][main._phase1_agent_assignment_builder] cause=non_positive_sample_count value={sample_count}")
        raise RuntimeError("--phase1-samples must be positive")
    if temperature <= 0:
        print(f"[ERROR][main._phase1_agent_assignment_builder] cause=non_positive_temperature value={temperature}")
        raise RuntimeError("--phase1-temperature must be positive")
    checkpoint_path = _phase1_checkpoint_path(checkpoint)
    model = load_phase1_pair_pointer_checkpoint(checkpoint_path)
    objective_scope = model.objective_scope
    capacity_weights = joint_phase1_bay_capacity_weights()
    bay_ids = tuple(capacity_weights)

    def build(jobs: Mapping[str, object], assignment_seed: int) -> Mapping[str, str]:
        best = run_phase1_pair_policy_resource_pool_best_of_k(
            jobs=jobs,
            bay_ids=bay_ids,
            model=model,
            sample_count=sample_count,
            temperature=temperature,
            seed=seed + assignment_seed * 10_000,
            bay_capacity_weights=capacity_weights,
            objective_scope=objective_scope,
        )
        print(
            "[CHECK][main._phase1_agent_assignment_builder] "
            f"assignment_seed={assignment_seed} source={best.source} samples={sample_count} "
            f"objective_scope={objective_scope} "
            f"score={score_phase1_bay_loads(best.bay_loads, objective_scope=objective_scope)}",
            flush=True,
        )
        return dict(best.assignments)

    return build


def _build_mixed_full_flow_scenario(
    physical_block_count: int,
    seed: int,
) -> dict[str, Any]:
    """Phase 1/2가 같은 MIXED episode와 확정 PLS/PLP 설비를 사용하는 scenario를 만든다."""

    if physical_block_count <= 0:
        print(
            "[ERROR][main._build_mixed_full_flow_scenario] "
            f"cause=invalid_physical_block_count value={physical_block_count}"
        )
        raise RuntimeError("--synthetic-blocks must be positive")
    episode = build_phase1_episode_jobs(
        episode_count=1,
        min_blocks=physical_block_count,
        max_blocks=physical_block_count,
        seed=seed,
        verbose=False,
    )[0]
    jobs = scenario_jobs_from_report_formula_jobs(episode["jobs"])
    machines = [
        asdict(machine)
        for machine in build_mixed_phase2_training_machines().values()
    ]
    scenario = {
        "metadata": {
            "job_source": MIXED_SYNTHETIC_SOURCE,
            "physical_block_count": int(episode["physical_block_count"]),
            "series_block_count": int(episode["block_count"]),
            "synthetic_job_count": len(jobs),
            "synthetic_seed": seed,
            "machine_source": "confirmed_eqp_pls_plp_mapping",
            "actual_eqp_mapping_available": True,
            "actual_only_eqp_ids": ["EQP_3"],
        },
        "jobs": jobs,
        "machines": machines,
    }
    if len(machines) != sum(len(ids) for ids in MIXED_PLANNING_MACHINE_IDS_BY_BAY.values()):
        print(
            "[ERROR][main._build_mixed_full_flow_scenario] "
            f"cause=machine_count_mismatch actual={len(machines)}"
        )
        raise RuntimeError("MIXED full-flow mapped machine count mismatch")
    print(
        "[VALIDATION][main._build_mixed_full_flow_scenario] "
        f"passed=true physical_blocks={physical_block_count} "
        f"series_blocks={episode['block_count']} jobs={len(jobs)} machines={len(machines)} seed={seed}"
    )
    return scenario


def _load_or_build_phase1_plan_for_full_flow(
    args: argparse.Namespace,
    scenario: Mapping[str, Any],
    *,
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float],
) -> dict:
    """Load Phase 1 plan or build it from a checkpoint/heuristic for full-flow."""

    phase1_plan = _optional_non_empty_cli_value(args.phase1_plan, "phase1_plan")
    phase1_checkpoint = _optional_non_empty_cli_value(args.phase1_checkpoint, "phase1_checkpoint")
    phase1_heuristic = _optional_non_empty_cli_value(args.phase1_heuristic, "phase1_heuristic")
    selected_source_count = sum(bool(value) for value in (phase1_plan, phase1_checkpoint, phase1_heuristic))
    if selected_source_count != 1:
        print(
            "[ERROR][main._load_or_build_phase1_plan_for_full_flow] "
            "cause=phase1_source_cardinality "
            f"phase1_plan={phase1_plan} phase1_checkpoint={phase1_checkpoint} "
            f"phase1_heuristic={phase1_heuristic}"
        )
        raise RuntimeError("Provide exactly one of --phase1-plan, --phase1-checkpoint, or --phase1-heuristic")

    if phase1_plan:
        phase1_plan_path = Path(phase1_plan)
        if not phase1_plan_path.exists():
            print(
                "[ERROR][main._load_or_build_phase1_plan_for_full_flow] "
                f"cause=missing_phase1_plan path={phase1_plan_path}"
            )
            raise FileNotFoundError(f"Phase 1 plan does not exist: {phase1_plan_path}")
        with phase1_plan_path.open("r", encoding="utf-8") as file:
            loaded_plan = json.load(file)
        _validate_phase1_plan_execution_contract(
            loaded_plan,
            bay_ids=bay_ids,
            bay_capacity_weights=bay_capacity_weights,
        )
        print(
            "[CHECK][main._load_or_build_phase1_plan_for_full_flow] "
            f"mode=plan_file path={phase1_plan_path}"
        )
        return loaded_plan

    jobs = _scenario_jobs_by_id(scenario)
    if not bay_ids:
        print("[ERROR][main._load_or_build_phase1_plan_for_full_flow] cause=no_phase1_bay_ids")
        raise RuntimeError("--phase1-bay-ids is required when building a Phase 1 plan")
    if phase1_heuristic:
        candidate = run_phase1_resource_pool_heuristic_candidate(
            jobs=jobs,
            bay_ids=bay_ids,
            algorithm=phase1_heuristic,
            bay_capacity_weights=bay_capacity_weights,
        )
        plan = candidate_to_phase1_plan(
            jobs=jobs,
            bay_ids=bay_ids,
            candidate=candidate,
        )
        plan_dir = Path(args.output_dir) / "phase1_heuristic_plan"
        plan_paths = write_phase1_bay_plan(plan, plan_dir)
        print(
            "[CHECK][main._load_or_build_phase1_plan_for_full_flow] "
            f"mode=heuristic source={phase1_heuristic} assignment_count={len(candidate.assignments)} "
            f"plan_json={plan_paths['json']}"
        )
        return plan
    if args.phase1_samples <= 0:
        print(
            "[ERROR][main._load_or_build_phase1_plan_for_full_flow] "
            f"cause=non_positive_phase1_samples value={args.phase1_samples}"
        )
        raise RuntimeError("--phase1-samples must be positive")
    if args.phase1_temperature <= 0:
        print(
            "[ERROR][main._load_or_build_phase1_plan_for_full_flow] "
            f"cause=non_positive_phase1_temperature value={args.phase1_temperature}"
        )
        raise RuntimeError("--phase1-temperature must be positive")

    model = load_phase1_pair_pointer_checkpoint(phase1_checkpoint)
    objective_scope = model.objective_scope
    best = run_phase1_pair_policy_resource_pool_best_of_k(
        jobs=jobs,
        bay_ids=bay_ids,
        model=model,
        sample_count=args.phase1_samples,
        temperature=args.phase1_temperature,
        seed=args.seed,
        bay_capacity_weights=bay_capacity_weights,
        objective_scope=objective_scope,
    )
    plan = candidate_to_phase1_plan(
        jobs=jobs,
        bay_ids=bay_ids,
        candidate=best,
        objective_scope=objective_scope,
    )
    plan_dir = Path(args.output_dir) / "phase1_agent_plan"
    plan_paths = write_phase1_bay_plan(plan, plan_dir)
    print(
        "[CHECK][main._load_or_build_phase1_plan_for_full_flow] "
        f"mode=checkpoint source={best.source} assignment_count={len(best.assignments)} "
        f"plan_json={plan_paths['json']}"
    )
    return plan


def _validate_phase1_plan_execution_contract(
    plan: Mapping[str, Any],
    *,
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float],
) -> None:
    """외부 Phase 1 plan이 MIXED joint 5-Bay 계약과 같은지 검증한다."""

    plan_weights = plan.get("bay_capacity_weights")
    expected_weights = {str(key): float(value) for key, value in bay_capacity_weights.items()}
    if not isinstance(plan_weights, Mapping):
        print("[ERROR][main._validate_phase1_plan_execution_contract] cause=missing_capacity_weights")
        raise RuntimeError("Phase 1 plan is missing bay_capacity_weights")
    normalized_plan_weights = {str(key): float(value) for key, value in plan_weights.items()}
    if set(str(value) for value in bay_ids) != set(expected_weights) or normalized_plan_weights != expected_weights:
        print(
            "[ERROR][main._validate_phase1_plan_execution_contract] "
            f"cause=capacity_weight_mismatch expected={expected_weights} actual={normalized_plan_weights}"
        )
        raise RuntimeError("Phase 1 plan capacity weights differ from full-flow")
    if "long_cut_hard_mask" in plan:
        print(
            "[ERROR][main._validate_phase1_plan_execution_contract] "
            "cause=legacy_long_cut_toggle_present"
        )
        raise RuntimeError("legacy Phase 1 plan is not compatible with MIXED full-flow")
    expected_contract = {
        "rule_profile": MULTI_SERIES_RULE_PROFILE,
        "scope_version": PHASE1_MULTI_SERIES_SCOPE_VERSION,
        "score_mode": "wo_first",
    }
    for field_name, expected_value in expected_contract.items():
        if plan.get(field_name) != expected_value:
            print(
                "[ERROR][main._validate_phase1_plan_execution_contract] "
                f"cause=phase1_contract_mismatch field={field_name} "
                f"expected={expected_value} actual={plan.get(field_name)}"
            )
            raise RuntimeError(f"Phase 1 plan {field_name} differs from MIXED full-flow")




def _resolve_phase2_full_flow_run_spec(
    *,
    args: argparse.Namespace,
    checkpoint_spec: Mapping[str, Any] | None,
    constraint_profile: object,
    phase1_bay_capacity_weights: Mapping[str, int | float],
    heuristic: str | None,
) -> dict[str, Any]:
    """Full-flow의 평가 조건을 checkpoint 계약과 동일하게 확정한다."""

    score_mode = str(
        getattr(
            args,
            "phase2_score_mode",
            checkpoint_spec["score_mode"] if checkpoint_spec is not None else "raw",
        )
    )
    action_pool_limit = getattr(
        args,
        "action_pool_limit",
        checkpoint_spec["action_pool_limit"] if checkpoint_spec is not None else None,
    )
    max_wo_count = int(
        getattr(
            args,
            "max_wo_count",
            checkpoint_spec["max_wo_count"] if checkpoint_spec is not None else 3,
        )
    )
    max_length_sum = float(
        getattr(
            args,
            "max_length_sum",
            checkpoint_spec["max_length_sum"] if checkpoint_spec is not None else 55_000.0,
        )
    )
    # score_mode -> 필드 목록 매핑은 Phase2.merged가 정본이다. 여기서 다시 쓰면
    # score 필드가 늘 때 두 곳이 어긋난다.
    score_fields = phase2_score_field_names(score_mode)
    if checkpoint_spec is None:
        if not heuristic:
            print("[ERROR][main._resolve_phase2_full_flow_run_spec] cause=no_checkpoint_or_heuristic")
            raise RuntimeError("full-flow requires a Phase 2 checkpoint or heuristic")
        heuristic_algorithms = (heuristic,)
        train_rollout_samples = 0
        validation_rollout_samples = 0
    else:
        heuristic_algorithms = tuple(checkpoint_spec["heuristic_algorithms"])
        train_rollout_samples = int(checkpoint_spec["train_rollout_samples"])
        validation_rollout_samples = int(checkpoint_spec["validation_rollout_samples"])
    requested = build_phase2_run_spec(
        score_mode=score_mode,
        score_fields=score_fields,
        action_pool_limit=action_pool_limit,
        max_wo_count=max_wo_count,
        max_length_sum=max_length_sum,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        constraint_profile=constraint_profile,
        heuristic_algorithms=heuristic_algorithms,
        train_rollout_samples=train_rollout_samples,
        validation_rollout_samples=validation_rollout_samples,
    )
    if checkpoint_spec is not None:
        require_matching_phase2_run_spec(
            checkpoint_spec,
            requested,
            context="full_flow",
        )
    return requested


def _require_matching_phase1_feedback_contract(
    phase1_feedback_contract: Mapping[str, Any],
    selected_phase2_contract: Mapping[str, Any],
) -> None:
    """Feedback로 학습한 Phase 1이 같은 frozen Phase 2와 연결되는지 검증한다."""

    if dict(phase1_feedback_contract) != dict(selected_phase2_contract):
        differing = sorted(
            key
            for key in set(phase1_feedback_contract) | set(selected_phase2_contract)
            if phase1_feedback_contract.get(key) != selected_phase2_contract.get(key)
        )
        print(
            "[ERROR][main._require_matching_phase1_feedback_contract] "
            f"cause=phase_checkpoint_contract_mismatch fields={differing}"
        )
        raise RuntimeError(
            f"feedback-trained Phase 1 requires its exact frozen Phase 2 checkpoint: {differing}"
        )


def _scenario_jobs_by_id(scenario: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs = scenario.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        print("[ERROR][main._scenario_jobs_by_id] cause=no_jobs")
        raise RuntimeError("scenario requires non-empty jobs")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(jobs):
        if not isinstance(row, Mapping):
            print(
                "[ERROR][main._scenario_jobs_by_id] "
                f"cause=invalid_job_row index={index} row_type={type(row).__name__}"
            )
            raise RuntimeError(f"invalid scenario job row at index={index}")
        job_id = str(row.get("job_id") or "").strip()
        if not job_id:
            print(f"[ERROR][main._scenario_jobs_by_id] cause=missing_job_id index={index}")
            raise RuntimeError(f"scenario job missing job_id at index={index}")
        if job_id in result:
            print(f"[ERROR][main._scenario_jobs_by_id] cause=duplicate_job_id job_id={job_id}")
            raise RuntimeError(f"duplicate scenario job_id: {job_id}")
        result[job_id] = row
    return result


def command_generate_phase1_blocks(args: argparse.Namespace) -> None:
    """물리 블록 공동분포를 보존한 MIXED block/W/O 파일을 만든다."""

    print("[generate-phase1-blocks]")
    print("- gyel: MIXED")
    print(f"- n_blocks: {args.n_blocks}")
    print(f"- seed: {args.seed}")
    synthetic_source = MIXED_SYNTHETIC_SOURCE
    print(f"- synthetic_source: {synthetic_source}")
    print(f"- output_dir: {args.output_dir}")
    generated = generate_multi_series_formula_data(
        n_physical_blocks=args.n_blocks,
        seed=args.seed,
    )
    physical_block_count = generated.physical_block_count
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_csv = output_dir / "phase1_synthetic_blocks.csv"
    wos_csv = output_dir / "phase1_synthetic_wos.csv"
    summary_json = output_dir / "summary.json"
    generated.block_df.to_csv(blocks_csv, index=False, encoding="utf-8-sig")
    generated.wo_df.to_csv(wos_csv, index=False, encoding="utf-8-sig")
    aggregation = {
        "max": ["LTH", "BTH", "THK", "TACT_TIME"],
        "sum": ["CUT_LTH", "MARK_LTH", "BVL_LTH", "STL_QTY", "BV_QTY", "PTLST_QTY"],
        "WO_QTY": "W/O row count",
    }
    formula_series = ("NP", "FN", "FL", "NC")
    fixed_profile = load_multi_series_generation_profile()
    fixed_bth_profiles = {
        "NP": fixed_profile.np_bth,
        **{
            series: fixed_profile.empirical_generators[series].conditional_bth_stl[
                "bth_formula"
            ]
            for series in formula_series
            if series != "NP"
        },
    }
    bth_profiles = {}
    for series in formula_series:
        profile = fixed_bth_profiles[series]
        bth_profiles[series] = {
            "coefficients": {
                "intercept": profile.coefficients[0],
                **dict(zip(BTH_FORMULA_FEATURES, profile.coefficients[1:])),
            },
            "residual_std": profile.residual_std,
            "r_squared": profile.r_squared,
            "observed_spec_count": len(profile.observed_specs),
            "observed_spec_min": min(profile.observed_specs),
            "observed_spec_max": max(profile.observed_specs),
        }
    summary = {
        "synthetic_source": synthetic_source,
        "gyel": "MIXED",
        "seed": args.seed,
        "physical_block_count": physical_block_count,
        "block_count": len(generated.block_df),
        "wo_count": len(generated.wo_df),
        "series_block_counts": {
            str(series): int(count)
            for series, count in generated.block_df["GYEL"].value_counts().sort_index().items()
        },
        "series_wo_counts": {
            str(series): int(count)
            for series, count in generated.wo_df["GYEL"].value_counts().sort_index().items()
        },
        "aggregation": aggregation,
        "bth_formula": {
            "equation": "ln(BTH)=b0+sum(bk*ln(1+xk))+epsilon",
            "features": list(BTH_FORMULA_FEATURES),
            "rounding": "nearest_observed_series_spec",
            "profiles": bth_profiles,
        },
        "tact_time_formula": {
            "equation": "0.3037*CUT_LTH+0.1325*MARK_LTH+0.4790*THK+0.3840*PTLST_QTY",
            "coefficients": {
                "CUT_LTH": TACT_A_CUT,
                "MARK_LTH": TACT_A_MARK,
                "THK": TACT_A_THK,
                "PTLST_QTY": TACT_A_PTLST,
            },
            "scope": "all_series_shared_np_ppt_case6",
        },
    }
    reference = load_physical_block_joint_profile()
    generated_combination_counts = Counter(
        "+".join(combination) for combination in generated.series_combinations
    )
    summary["actual_series_combination_distribution"] = {
        "+".join(combination): {
            "count": count,
            "probability": probability,
        }
        for combination, count, probability in zip(
            reference.combinations,
            reference.counts,
            reference.probabilities,
        )
    }
    summary["generated_series_combination_counts"] = dict(
        sorted(generated_combination_counts.items())
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"- blocks_csv: {blocks_csv}")
    print(f"- wos_csv: {wos_csv}")
    print(f"- summary_json: {summary_json}")
    print(
        "[VALIDATION][main.command_generate_phase1_blocks] "
        f"passed=true blocks={len(generated.block_df)} wos={len(generated.wo_df)}"
    )




# LINE-BY-LINE: `build_parser()` 함수를 정의합니다. 반환 타입: `argparse.ArgumentParser`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    # LINE-BY-LINE: `parser`에 `argparse.ArgumentParser(description="Cutting shop scheduling project")` 결과를 저장합니다. 의미/사용: `parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    parser = argparse.ArgumentParser(description="Cutting shop scheduling project")
    # LINE-BY-LINE: `common_parser`에 `argparse.ArgumentParser(add_help=False)` 결과를 저장합니다. 의미/사용: `common_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", default="config_mixed.yaml", help="YAML config path")
    # LINE-BY-LINE: `subparsers`에 `parser.add_subparsers(dest="command", required=True)` 결과를 저장합니다. 의미/사용: `subparsers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    subparsers = parser.add_subparsers(dest="command", required=True)


    phase1_multi_series_parser = subparsers.add_parser(
        "phase1-plan-multi-series",
        help="Validate multi-series Excel data and write daily Phase 1 Bay plans",
    )
    phase1_multi_series_parser.add_argument(
        "--block-xlsx",
        required=True,
        help="Multi-series block Excel/CSV path",
    )
    phase1_multi_series_parser.add_argument(
        "--wo-xlsx",
        required=True,
        help="Multi-series W/O Excel/CSV path",
    )
    phase1_multi_series_parser.add_argument(
        "--output-dir",
        default="output/generated/multi_series_260711",
        help="Directory for date audit and Phase 1 plan outputs",
    )
    phase1_multi_series_parser.set_defaults(func=command_phase1_plan_multi_series)

    phase1_train_pair_self_labeling_parser = subparsers.add_parser(
        "phase1-train-pair-self-labeling",
        parents=[common_parser],
        help="Train the MIXED physical-block SELECT_PAIR(block-series,bay) policy",
    )
    phase1_train_pair_self_labeling_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum physical blocks per episode")
    phase1_train_pair_self_labeling_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum physical blocks per episode")
    phase1_train_pair_self_labeling_parser.add_argument(
        "--output-dir",
        default="output/phase1_pair_self_labeling",
        help="Directory for checkpoint and metrics",
    )
    phase1_train_pair_self_labeling_parser.add_argument("--episodes", type=int, default=20, help="Self-labeling episodes")
    phase1_train_pair_self_labeling_parser.add_argument("--rollout-samples", type=int, default=4, help="Current-policy sampled candidates per episode")
    phase1_train_pair_self_labeling_parser.add_argument(
        "--rollout-samples_validation",
        "--rollout-samples-validation",
        dest="validation_rollout_samples",
        type=int,
        default=None,
        help="Current-policy sampled candidates per validation problem. Default follows --rollout-samples.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--heuristic-algorithms",
        default="all",
        help="all/mixed3 or a comma-separated subset of wo_first_balanced,cut_first_balanced,bevel_first_balanced",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--objective-scope",
        choices=PHASE1_OBJECTIVE_SCOPES,
        default="shared_and_series",
        help=(
            "Phase 1 lexicographic objective evaluated independently inside NP_NC and FN_FL: "
            "shared_and_series keeps resource-pool and per-series gaps; series_only uses only "
            "the per-series W/O, CUT_LTH, BV_QTY gap sums."
        ),
    )
    phase1_train_pair_self_labeling_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    phase1_train_pair_self_labeling_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    phase1_train_pair_self_labeling_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    phase1_train_pair_self_labeling_parser.add_argument("--seed", type=int, default=0, help="Torch random seed")
    phase1_train_pair_self_labeling_parser.add_argument("--device", default="cpu", help="Torch device for Phase 1 pair training: cpu, cuda, or cuda:0")
    phase1_train_pair_self_labeling_parser.add_argument("--checkpoint-every", type=int, default=100, help="Save periodic checkpoint every N episodes")
    phase1_train_pair_self_labeling_parser.add_argument("--validation-every", type=int, default=100, help="Run holdout validation every N episodes")
    phase1_train_pair_self_labeling_parser.add_argument("--validation-episodes", type=int, default=20, help="Holdout validation episodes per validation run")
    phase1_train_pair_self_labeling_parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Resume pair self-labeling from explicit checkpoint path or 'latest' in output-dir/checkpoints",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--write-candidate-summary",
        action="store_true",
        help="Write per-training-candidate audit rows. Disabled by default for long training speed.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--enable-phase2-feedback-score",
        action="store_true",
        help="Prepend a frozen mapped-factory Phase 2 schedule score to the Phase 1 teacher score.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--phase2-feedback-checkpoint",
        default=None,
        help="Frozen Phase 2 set-pointer checkpoint. Required with --enable-phase2-feedback-score.",
    )
    phase1_train_pair_self_labeling_parser.set_defaults(func=command_phase1_train_pair_self_labeling)

    phase2_train_graph_parser = subparsers.add_parser(
        "phase2-train-batch-machine-self-labeling",
        parents=[common_parser],
        help="Train MIXED merged Phase 2 batch-machine policy",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-heuristic",
        default=None,
        help="MIXED Phase 1 heuristic run for every Phase 2 episode. Default: bevel_first_balanced.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        help="Frozen Phase 1 pair-pointer checkpoint or output directory. Replaces --phase1-heuristic during Phase 2 training.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-samples",
        type=int,
        default=32,
        help="Frozen Phase 1 checkpoint best-of-N samples used to build each Phase 2 episode's block-to-Bay assignment.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-temperature",
        type=float,
        default=1.0,
        help="Frozen Phase 1 checkpoint sampling temperature for --phase1-samples > 1.",
    )
    phase2_train_graph_parser.add_argument(
        "--output-dir",
        default="output/phase2_batch_machine_self_labeling",
        help="Directory for Phase 2 checkpoint and metrics.",
    )
    phase2_train_graph_parser.add_argument("--episodes", type=int, default=10, help="Self-labeling training episodes")
    phase2_train_graph_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    phase2_train_graph_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    phase2_train_graph_parser.add_argument("--seed", type=int, default=0, help="Torch random seed")
    phase2_train_graph_parser.add_argument("--device", default="cpu", help="Torch device for merged Phase 2 training: cpu, cuda, or cuda:0")
    phase2_train_graph_parser.add_argument(
        "--heuristic-algorithms",
        default=",".join(PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK),
        help="Comma-separated merged Phase 2 candidate-bank heuristics.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase2-score-mode",
        choices=["raw", "normalized"],
        default="raw",
        help="Teacher score mode for merged Phase 2 candidate ranking.",
    )
    phase2_train_graph_parser.add_argument(
        "--rollout-samples",
        type=int,
        default=1,
        help="Number of stochastic agent assignment candidates per episode.",
    )
    phase2_train_graph_parser.add_argument(
        "--candidate-workers",
        type=int,
        default=1,
        help="Spawn processes for parallel Phase 2 candidate generation. 1 preserves the sequential baseline.",
    )
    phase2_train_graph_parser.add_argument(
        "--rollout-samples_validation",
        "--rollout-samples-validation",
        dest="validation_rollout_samples",
        type=int,
        default=None,
        help="Number of stochastic agent candidates per validation problem. Default follows --rollout-samples.",
    )
    phase2_train_graph_parser.add_argument("--validation-every", type=int, default=100, help="Run Phase 2 validation every N episodes")
    phase2_train_graph_parser.add_argument("--validation-episodes", type=int, default=20, help="Holdout Phase 2 validation episodes per validation run")
    phase2_train_graph_parser.add_argument("--checkpoint-every", type=int, default=0, help="Save periodic Phase 2 checkpoint every N episodes. 0 disables periodic checkpoints.")
    phase2_train_graph_parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Resume merged Phase 2 training from explicit checkpoint path or 'latest' in output-dir/checkpoints.",
    )
    phase2_train_graph_parser.add_argument(
        "--write-candidate-summary",
        action="store_true",
        help="Write per-training-candidate audit rows. Disabled by default for long training speed.",
    )
    phase2_train_graph_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum physical blocks per MIXED episode")
    phase2_train_graph_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum physical blocks per MIXED episode")
    phase2_train_graph_parser.add_argument("--max-wo-count", type=int, default=3, help="Maximum W/O count per machine batch")
    phase2_train_graph_parser.add_argument("--max-length-sum", type=float, default=55000.0, help="Maximum LTH sum per machine batch")
    phase2_train_graph_parser.add_argument(
        "--action-pool-limit",
        type=_parse_optional_positive_int,
        default=None,
        help="Top W/O count considered when selecting the next W/O for an open batch. Use None/all/full for all feasible W/O.",
    )
    phase2_train_graph_parser.set_defaults(func=command_phase2_train_batch_machine_self_labeling)

    phase2_run_full_parser = subparsers.add_parser(
        "phase2-run-full-workflow",
        parents=[common_parser],
        help="Run MIXED physical-block Phase 1 -> merged Phase 2 schedule export",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-plan",
        default=None,
        help="Phase 1 plan JSON path. Phase 2 candidates depend on block-to-Bay assignments.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        help="Optional Phase 1 pair-pointer checkpoint. Mutually exclusive with --phase1-plan.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-heuristic",
        default=None,
        choices=PHASE1_HEURISTIC_BANK,
        help="Optional fixed MIXED Phase 1 heuristic. Mutually exclusive with --phase1-plan/--phase1-checkpoint.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-samples",
        type=int,
        default=1,
        help="Phase 1 checkpoint inference samples. 1 uses greedy; values >1 choose best sampled plan.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-temperature",
        type=float,
        default=1.0,
        help="Phase 1 sampling temperature when --phase1-samples is greater than 1.",
    )
    phase2_run_full_parser.add_argument(
        "--batch-machine-heuristic",
        default=None,
        help="Optional fixed merged Phase 2 heuristic. Default is min_makespan only when no checkpoint is given.",
    )
    phase2_run_full_parser.add_argument(
        "--batch-machine-checkpoint",
        default=None,
        help="Optional merged Phase 2 batch-machine policy checkpoint. Mutually exclusive with --batch-machine-heuristic.",
    )
    phase2_run_full_parser.add_argument("--seed", type=int, default=0, help="Inference random seed")
    phase2_run_full_parser.add_argument(
        "--max-wo-count",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum W/O count per batch. A checkpoint run inherits this from RunSpec.",
    )
    phase2_run_full_parser.add_argument(
        "--max-length-sum",
        type=float,
        default=argparse.SUPPRESS,
        help="Maximum LTH sum per batch. A checkpoint run inherits this from RunSpec.",
    )
    phase2_run_full_parser.add_argument(
        "--action-pool-limit",
        type=_parse_optional_positive_int,
        default=argparse.SUPPRESS,
        help="Sequential W/O candidate limit. A checkpoint run inherits this from RunSpec.",
    )
    phase2_run_full_parser.add_argument(
        "--phase2-score-mode",
        choices=["raw", "normalized"],
        default=argparse.SUPPRESS,
        help="Phase 2 score mode. A checkpoint run inherits this from RunSpec.",
    )
    phase2_run_full_parser.add_argument(
        "--synthetic-blocks",
        type=int,
        default=30,
        help="Physical block count for the MIXED joint-distribution episode.",
    )
    phase2_run_full_parser.add_argument(
        "--output-dir",
        default="output/phase2_full_workflow",
        help="Directory for Phase 2 CSV/debug outputs.",
    )
    phase2_run_full_parser.set_defaults(func=command_phase2_run_full_workflow)


    generate_phase1_blocks_parser = subparsers.add_parser(
        "generate-phase1-blocks",
        help="Generate MIXED physical-block joint-distribution block/W/O data",
    )
    generate_phase1_blocks_parser.add_argument(
        "--n-blocks",
        type=int,
        default=787,
        help="MIXED physical-block count to generate",
    )
    generate_phase1_blocks_parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic generation",
    )
    generate_phase1_blocks_parser.add_argument(
        "--output-dir",
        default="output/generated/phase1_mixed_joint",
        help="Output directory for block/W/O CSV and generation summary",
    )
    generate_phase1_blocks_parser.set_defaults(func=command_generate_phase1_blocks)


    # LINE-BY-LINE: 호출자에게 `parser`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return parser


# LINE-BY-LINE: `main()` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def main() -> None:
    """프로그램 시작점."""

    # LINE-BY-LINE: `parser`에 `build_parser()` 결과를 저장합니다. 의미/사용: `parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    parser = build_parser()
    # LINE-BY-LINE: `args`에 `parser.parse_args()` 결과를 저장합니다. 의미/사용: `args` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    args = parser.parse_args()
    # LINE-BY-LINE: `args.func(args)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    args.func(args)


# LINE-BY-LINE: 조건 `__name__ == "__main__"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
if __name__ == "__main__":
    # LINE-BY-LINE: `main()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    main()
