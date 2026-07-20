"""프로젝트 공통 실행 진입점.

초보자 기준 실행 예시:
  python3 main.py show-config --config config_np_100.yaml
  python3 main.py build-scenario --config config_np_100.yaml
  python3 main.py simulate --config config_np_100.yaml
  python3 main.py playback --config config_np_100.yaml
  python3 main.py trace --config config_np_100.yaml
"""

# LINE-BY-LINE: Windows conda에서 pandas/numpy와 torch가 서로 다른 Intel OpenMP runtime을 초기화하면
# `libiomp5md.dll already initialized`로 학습이 중단될 수 있습니다.
# 사용: 반드시 torch/numpy/pandas import보다 먼저 설정해야 하며, 판넬라인 PPO eval runner와 같은 실행 보호 장치입니다.
import os

# LINE-BY-LINE: 사용자가 외부에서 명시한 값은 존중하고, 없는 경우에만 Windows OpenMP 중복 초기화 허용값을 설정합니다.
if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("[CHECK][main.openmp_guard] KMP_DUPLICATE_LIB_OK=TRUE")

# LINE-BY-LINE: `argparse` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import argparse
# LINE-BY-LINE: `csv` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import csv
# LINE-BY-LINE: `json` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import json
# LINE-BY-LINE: `collections` 모듈에서 `Counter, defaultdict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from collections import Counter, defaultdict
from dataclasses import asdict
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# LINE-BY-LINE: `Agent.heuristics` 모듈에서 `select_action_by_rule`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Agent.heuristics import select_action_by_rule
# LINE-BY-LINE: `Environment.environment` 모듈에서 `CuttingShopEnvironment`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.environment import CuttingShopEnvironment
from Environment.gym_wrapper import GYMNASIUM_AVAILABLE, run_wrapper_equivalence
from Environment.constraints.profiles import load_phase_constraint_profile
from Phase1.orchestrator import candidate_to_phase1_plan
from Phase2.orchestrator import run_phase2_full_graph_workflow, write_phase2_workflow_outputs
from Phase2.feedback import (
    build_frozen_phase2_schedule_feedback_scorer,
    build_phase2_feedback_contract,
)
from Phase2.merged import (
    PHASE2_BATCH_MACHINE_DEFAULT_HEURISTIC_BANK,
    PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES,
    PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES,
    build_mixed_phase2_training_machines,
    train_phase2_batch_machine_self_labeling,
)
from Phase2.run_spec import build_phase2_run_spec, require_matching_phase2_run_spec
from Phase1.pair_self_labeling import run_phase1_pair_policy_rollout, train_phase1_pair_self_labeling
from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    score_phase1_bay_loads,
    run_phase1_heuristic_candidate,
)
# LINE-BY-LINE: `Utils.config` 모듈에서 `load_config`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.config import load_config
# LINE-BY-LINE: `Utils.data.cutting_data_loader` 모듈에서 `load_and_clean_cutting_data, write_records_csv`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.cutting_data_loader import load_and_clean_cutting_data, write_records_csv
from Utils.data.cutting_start_date import (
    audit_cutting_start_dates,
    write_cutting_start_date_audit,
)
from Utils.data.multi_series_cutting_data import (
    MIXED_PLANNING_MACHINE_IDS_BY_BAY,
    load_multi_series_cutting_data,
)
# LINE-BY-LINE: `Utils.data.cutting_scenario_builder` 모듈에서 `build_scenario_from_cutting_records`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.cutting_scenario_builder import build_scenario_from_cutting_records
# LINE-BY-LINE: `Utils.data.factory_builder` 모듈에서 `build_factory_scenario_parts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.factory_builder import build_factory_scenario_parts
# LINE-BY-LINE: `Utils.data.io` 모듈에서 scenario loader를 가져옵니다. `load_scenario`는 명시 YAML용, `load_scenario_for_config`는 config 기반 원본 데이터 로딩용입니다.
from Utils.data.io import load_scenario, load_scenario_for_config
from Utils.learning.phase_agent_checkpoints import (
    load_phase1_feedback_contract,
    load_phase1_pair_pointer_checkpoint,
    load_phase2_checkpoint_run_spec,
    load_phase2_set_pointer_checkpoint,
)
from Utils.phase1.phase1_episode_dataset import build_phase1_episode_jobs
from Utils.phase1.phase1_bay_balancer import (
    apply_phase1_plan_to_scenario,
    write_phase1_bay_plan,
)
from Utils.phase1.multi_series_planner import (
    build_multi_series_phase1_daily_plans,
    write_multi_series_phase1_daily_plans,
)
from Utils.phase1.multi_series_rules import (
    MULTI_SERIES_RULE_PROFILE,
    PHASE1_MULTI_SERIES_SCOPE_VERSION,
    joint_phase1_bay_capacity_weights,
)
from Utils.data.report_formula_data_generator import (
    BTH_FORMULA_FEATURES,
    DEFAULT_MULTI_SERIES_WO_SOURCE,
    TACT_A_CUT,
    TACT_A_MARK,
    TACT_A_PTLST,
    TACT_A_THK,
    load_bth_formula_profile,
    scenario_jobs_from_report_formula_jobs,
)
from Utils.data.multi_series_formula_data_generator import (
    generate_multi_series_formula_data,
    load_physical_block_joint_profile,
)
# LINE-BY-LINE: `Utils.reporting.playback_builder` 모듈에서 `write_actual_replay_artifacts, write_playback_artifacts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.playback_builder import write_actual_replay_artifacts, write_playback_artifacts
# LINE-BY-LINE: `Utils.data.scenario_generator` 모듈에서 `generate_scenario_from_template, save_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.scenario_generator import generate_scenario_from_template, save_scenario
# LINE-BY-LINE: `Utils.reporting.tact_gap_analysis` 모듈에서 `build_tact_gap_analysis`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.tact_gap_analysis import build_tact_gap_analysis
# LINE-BY-LINE: `Utils.reporting.tact_time` 모듈에서 `build_tact_time_analysis_from_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.tact_time import build_tact_time_analysis_from_scenario
# LINE-BY-LINE: `Utils.data.test_data_selection` 모듈에서 `select_actual_day_test_data, select_actual_start_range_test_data`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.test_data_selection import select_actual_day_test_data, select_actual_start_range_test_data


def parse_csv_argument(value: str | None, default: Sequence[str]) -> list[str]:
    """쉼표로 구분한 CLI 값을 공백 없는 문자열 목록으로 변환한다."""

    if value is None or str(value).strip() == "":
        return list(default)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        print(f"[ERROR][main.parse_csv_argument] cause=empty_csv_argument input={value!r}")
        raise ValueError("comma-separated argument produced no items")
    return items


# LINE-BY-LINE: `build_environment(config_path: str, scenario_path: str | None = None)` 함수를 정의합니다. 반환 타입: `CuttingShopEnvironment`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def build_environment(config_path: str, scenario_path: str | None = None) -> CuttingShopEnvironment:
    """config와 scenario를 읽어서 환경 객체를 만듭니다.

    이 함수 하나만 호출하면
    - config 로딩
    - scenario YAML 로딩
    - environment 객체 생성
    가 한 번에 끝납니다.
    """

    # LINE-BY-LINE: `config`에 `load_config(config_path)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(config_path)
    # LINE-BY-LINE: 조건 `scenario_path is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if scenario_path is not None:
        # LINE-BY-LINE: `config["paths"]["scenario_path"]`에 `scenario_path` 결과를 저장합니다. 의미/사용: `config["paths"]["scenario_path"]` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        config["paths"]["scenario_path"] = scenario_path
    scenario = load_scenario_for_config(config, scenario_path_override=scenario_path)
    # LINE-BY-LINE: 호출자에게 `CuttingShopEnvironment(config=config, scenario=scenario)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return CuttingShopEnvironment(config=config, scenario=scenario)


def apply_action_space_overrides(env: CuttingShopEnvironment, args: argparse.Namespace) -> None:
    """CLI에서 명시한 action space override를 환경 config에 반영한다.

    `CuttingSimulation.reset()`은 매 실행 전에 현재 config로 state를 다시 만든다.
    따라서 command 함수는 run/reset 전에 이 helper를 호출해야 한다.
    """

    action_space = env.config.setdefault("action_space", {})
    if getattr(args, "action_mode", None) is not None:
        action_space["mode"] = args.action_mode
    if getattr(args, "allow_single_slot_baseline", False):
        action_space["allow_single_slot_baseline"] = True


# LINE-BY-LINE: `command_show_config(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_show_config(args: argparse.Namespace) -> None:
    """현재 활성 config의 핵심 내용만 요약해서 보여줍니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    # LINE-BY-LINE: `category_enabled`에 `[name for name, enabled in config["constraints"]["categories"].items() if enabled]` 결과를 저장합니다. 의미/사용: `category_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    category_enabled = [name for name, enabled in config["constraints"]["categories"].items() if enabled]
    # LINE-BY-LINE: `hard_enabled`에 `[name for name, enabled in config["constraints"]["hard_enabled"].items() if enabled]` 결과를 저장합니다. 의미/사용: `hard_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hard_enabled = [name for name, enabled in config["constraints"]["hard_enabled"].items() if enabled]
    # LINE-BY-LINE: `soft_enabled`에 `[name for name, enabled in config["constraints"]["soft_enabled"].items() if enabled]` 결과를 저장합니다. 의미/사용: `soft_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    soft_enabled = [name for name, enabled in config["constraints"]["soft_enabled"].items() if enabled]

    # LINE-BY-LINE: 콘솔에 `print("[config]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[config]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {config['paths']['scenario_path']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {config.get('paths', {}).get('scenario_path')}")
    print(f"- source_data_path: {config.get('paths', {}).get('source_data_path')}")
    print(f"- max_records: {config.get('data', {}).get('max_records')}")
    print(f"- process_time_source: {config.get('data', {}).get('process_time_source')}")
    # LINE-BY-LINE: 콘솔에 `print(f"- action_mode: {config['action_space']['mode']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- action_mode: {config['action_space']['mode']}")
    print(f"- allow_single_slot_baseline: {config.get('action_space', {}).get('allow_single_slot_baseline', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- default_heuristic: {config['agent']['default_heuristic']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- default_heuristic: {config['agent']['default_heuristic']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- enabled_categories: {', '.join(category_enabled)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- enabled_categories: {', '.join(category_enabled)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- hard_constraints: {', '.join(hard_enabled)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- hard_constraints: {', '.join(hard_enabled)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- soft_constraints: {', '.join(soft_enabled)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- soft_constraints: {', '.join(soft_enabled)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- holidays_off_enabled: {config['calendar'].get('enable_holidays_off', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- holidays_off_enabled: {config['calendar'].get('enable_holidays_off', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- half_day_off_enabled: {config['calendar'].get('enable_half_day_off', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- half_day_off_enabled: {config['calendar'].get('enable_half_day_off', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- lunch_break_enabled: {config['calendar'].get('enable_lunch_break', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- lunch_break_enabled: {config['calendar'].get('enable_lunch_break', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_operating_windows_enabled: {config['calendar'].get('enable_machine_operating_wi...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_operating_windows_enabled: {config['calendar'].get('enable_machine_operating_windows', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_shutdown_windows_enabled: {config['calendar'].get('enable_machine_shutdown_wind...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_shutdown_windows_enabled: {config['calendar'].get('enable_machine_shutdown_windows', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_breakdowns_enabled: {config['calendar'].get('enable_machine_breakdowns', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_breakdowns_enabled: {config['calendar'].get('enable_machine_breakdowns', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- operation_time_adjustment_enabled: {config['calendar'].get('enable_operation_time_adjus...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- operation_time_adjustment_enabled: {config['calendar'].get('enable_operation_time_adjustment', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- family_changeover_enabled: {config.get('setup', {}).get('enable_family_changeover', Fal...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- family_changeover_enabled: {config.get('setup', {}).get('enable_family_changeover', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- batch_processing_time_rule: {config.get('batch', {}).get('processing_time_rule', '')}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- batch_processing_time_rule: {config.get('batch', {}).get('processing_time_rule', '')}")


# LINE-BY-LINE: `command_factory_summary(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_factory_summary(args: argparse.Namespace) -> None:
    """Bay/설비 생성 결과와 설비별 처리 가능 조건을 출력합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    # LINE-BY-LINE: `scenario`에 `None` 결과를 저장합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario = None
    # LINE-BY-LINE: `scenario_path`에 `args.scenario_path or config.get("paths", {}).get("scenario_path")` 결과를 저장합니다. 의미/사용: `scenario_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    # LINE-BY-LINE: 조건 `scenario_path`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if scenario_path:
        # LINE-BY-LINE: `scenario`에 `load_scenario(scenario_path)` 결과를 저장합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
        scenario = load_scenario(scenario_path)

    # LINE-BY-LINE: 조건 `config.get("factory")`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if config.get("factory"):
        # LINE-BY-LINE: `machines, cut_bays` 여러 변수에 `build_factory_scenario_parts(config["factory"])` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        machines, cut_bays = build_factory_scenario_parts(config["factory"])
        # LINE-BY-LINE: `machine_source`에 `"config.factory"` 결과를 저장합니다. 의미/사용: `machine_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_source = "config.factory"
    # LINE-BY-LINE: 앞선 조건이 거짓일 때 `scenario is not None`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
    elif scenario is not None:
        # LINE-BY-LINE: `machines`에 `scenario.get("machines", [])` 결과를 저장합니다. 의미/사용: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machines = scenario.get("machines", [])
        # LINE-BY-LINE: `cut_bays`에 `scenario.get("cut_bays", [])` 결과를 저장합니다. 의미/사용: `cut_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cut_bays = scenario.get("cut_bays", [])
        # LINE-BY-LINE: `machine_source`에 `"scenario.machines"` 결과를 저장합니다. 의미/사용: `machine_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_source = "scenario.machines"
    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
    else:
        # LINE-BY-LINE: `machines`에 `[]` 결과를 저장합니다. 의미/사용: `machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machines = []
        # LINE-BY-LINE: `cut_bays`에 `[]` 결과를 저장합니다. 의미/사용: `cut_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        cut_bays = []
        # LINE-BY-LINE: `machine_source`에 `"none"` 결과를 저장합니다. 의미/사용: `machine_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        machine_source = "none"

    # LINE-BY-LINE: `machines_by_bay`에 `defaultdict(list)` 결과를 저장합니다. 의미/사용: `machines_by_bay` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    machines_by_bay = defaultdict(list)
    # LINE-BY-LINE: `machine in machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for machine in machines:
        # LINE-BY-LINE: `machines_by_bay[str(machine.get("bay_id", "unknown"))].append(machine)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        machines_by_bay[str(machine.get("bay_id", "unknown"))].append(machine)

    # LINE-BY-LINE: 콘솔에 `print("[factory summary]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[factory summary]")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_source: {machine_source}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_source: {machine_source}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- cut_bay_count: {len(cut_bays)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- cut_bay_count: {len(cut_bays)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_count: {len(machines)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_count: {len(machines)}")

    # LINE-BY-LINE: `bay_id in sorted(machines_by_bay)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for bay_id in sorted(machines_by_bay):
        # LINE-BY-LINE: `bay_machines`에 `sorted(machines_by_bay[bay_id], key=lambda item: str(item.get("machine_id", "")))` 결과를 저장합니다. 의미/사용: `bay_machines` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        bay_machines = sorted(machines_by_bay[bay_id], key=lambda item: str(item.get("machine_id", "")))
        # LINE-BY-LINE: `equipment_counts`에 `Counter(str(machine.get("equipment_type") or machine.get("machine_type")) for machine in bay_mach...` 결과를 저장합니다. 의미/사용: `equipment_counts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        equipment_counts = Counter(str(machine.get("equipment_type") or machine.get("machine_type")) for machine in bay_machines)
        # LINE-BY-LINE: `equipment_text`에 `", ".join(f"{name}={count}" for name, count in sorted(equipment_counts.items()))` 결과를 저장합니다. 의미/사용: `equipment_text` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        equipment_text = ", ".join(f"{name}={count}" for name, count in sorted(equipment_counts.items()))
        # LINE-BY-LINE: 콘솔에 `print(f"- Bay {bay_id}: {len(bay_machines)} machines ({equipment_text})")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- Bay {bay_id}: {len(bay_machines)} machines ({equipment_text})")
        # LINE-BY-LINE: `machine in bay_machines` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for machine in bay_machines:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `" "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "  "
                # LINE-BY-LINE: `f"{machine.get('machine_id')}: "`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                f"{machine.get('machine_id')}: "
                # LINE-BY-LINE: `f"type`에 `{machine.get('machine_type')}, "` 결과를 저장합니다. 의미/사용: `f"type` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"type={machine.get('machine_type')}, "
                # LINE-BY-LINE: `f"families`에 `{','.join(map(str, machine.get('eligible_families', [])))}, "` 결과를 저장합니다. 의미/사용: `f"families` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"families={','.join(map(str, machine.get('eligible_families', [])))}, "
                # LINE-BY-LINE: `f"thickness`에 `{machine.get('min_thickness')}..{machine.get('max_thickness')}, "` 결과를 저장합니다. 의미/사용: `f"thickness` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"thickness={machine.get('min_thickness')}..{machine.get('max_thickness')}, "
                # LINE-BY-LINE: `f"table_length_limit`에 `{machine.get('table_length_limit')}, "` 결과를 저장합니다. 의미/사용: `f"table_length_limit` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"table_length_limit={machine.get('table_length_limit')}, "
                # LINE-BY-LINE: `f"parallel_capacity`에 `{machine.get('parallel_capacity', 1)}, "` 결과를 저장합니다. 의미/사용: `f"parallel_capacity` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"parallel_capacity={machine.get('parallel_capacity', 1)}, "
                # LINE-BY-LINE: `f"enabled`에 `{machine.get('enabled')}"` 결과를 저장합니다. 의미/사용: `f"enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"enabled={machine.get('enabled')}"
            )

    # LINE-BY-LINE: `hard_enabled`에 `config.get("constraints", {}).get("hard_enabled", {})` 결과를 저장합니다. 의미/사용: `hard_enabled` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hard_enabled = config.get("constraints", {}).get("hard_enabled", {})
    # LINE-BY-LINE: 콘솔에 `print("[factory constraints]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[factory constraints]")
    # LINE-BY-LINE: 콘솔에 `print(f"- family_eligibility: {hard_enabled.get('family_eligibility', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- family_eligibility: {hard_enabled.get('family_eligibility', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- thickness_range: {hard_enabled.get('thickness_range', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- thickness_range: {hard_enabled.get('thickness_range', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- table_length_limit: {hard_enabled.get('table_length_limit', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- table_length_limit: {hard_enabled.get('table_length_limit', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_bay_consistency: {hard_enabled.get('machine_bay_consistency', False)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_bay_consistency: {hard_enabled.get('machine_bay_consistency', False)}")
    # LINE-BY-LINE: 콘솔에 batch W/O/길이 제약 rule on/off를 출력합니다. 사용: 현업 확인된 batch 제약이 켜졌는지 확인합니다.
    print(f"- batch_3wo_55000_rule_keys: batch_wo_count_limit={hard_enabled.get('batch_wo_count_limit', False)}, batch_length_sum_limit={hard_enabled.get('batch_length_sum_limit', False)}")
    # LINE-BY-LINE: 콘솔에 과거 동시 active audit rule on/off를 출력합니다. 사용: actual replay에서 timestamp 오류를 hard violation으로 오해하지 않게 확인합니다.
    print(f"- legacy_concurrent_rule_keys: machine_day_wo_count_limit={hard_enabled.get('machine_day_wo_count_limit', False)}, machine_day_length_sum_limit={hard_enabled.get('machine_day_length_sum_limit', False)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- batch_limits: {config.get('constraints', {}).get('machine_day_limits', {})}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- batch_limits: {config.get('constraints', {}).get('machine_day_limits', {})}")
    # LINE-BY-LINE: 콘솔에 `print(f"- preferred_machine_type_soft: {config.get('constraints', {}).get('soft_enabled', {}).get...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- preferred_machine_type_soft: {config.get('constraints', {}).get('soft_enabled', {}).get('preferred_machine_type', False)}")

    # LINE-BY-LINE: 조건 `scenario is not None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if scenario is not None:
        # LINE-BY-LINE: `jobs`에 `scenario.get("jobs", [])` 결과를 저장합니다. 의미/사용: `jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        jobs = scenario.get("jobs", [])
        # LINE-BY-LINE: `families`에 `Counter(str(job.get("family", "")) for job in jobs)` 결과를 저장합니다. 의미/사용: `families` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        families = Counter(str(job.get("family", "")) for job in jobs)
        # LINE-BY-LINE: `source_bays`에 `Counter(str(job.get("source_cut_bay", "")) for job in jobs)` 결과를 저장합니다. 의미/사용: `source_bays` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        source_bays = Counter(str(job.get("source_cut_bay", "")) for job in jobs)
        # LINE-BY-LINE: `thickness_values`에 `[float(job.get("thickness", 0.0) or 0.0) for job in jobs]` 결과를 저장합니다. 의미/사용: `thickness_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        thickness_values = [float(job.get("thickness", 0.0) or 0.0) for job in jobs]
        # LINE-BY-LINE: `length_values`에 `[float(job.get("plate_length", 0.0) or 0.0) for job in jobs]` 결과를 저장합니다. 의미/사용: `length_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        length_values = [float(job.get("plate_length", 0.0) or 0.0) for job in jobs]
        # LINE-BY-LINE: 콘솔에 `print("[job summary]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[job summary]")
        # LINE-BY-LINE: 콘솔에 `print(f"- job_count: {len(jobs)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- job_count: {len(jobs)}")
        # LINE-BY-LINE: 콘솔에 `print(f"- families: {dict(sorted(families.items()))}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- families: {dict(sorted(families.items()))}")
        # LINE-BY-LINE: 콘솔에 `print(f"- source_cut_bays: {dict(sorted(source_bays.items()))}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- source_cut_bays: {dict(sorted(source_bays.items()))}")
        # LINE-BY-LINE: 조건 `thickness_values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if thickness_values:
            # LINE-BY-LINE: 콘솔에 `print(f"- thickness_range_in_jobs: {min(thickness_values)}..{max(thickness_values)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- thickness_range_in_jobs: {min(thickness_values)}..{max(thickness_values)}")
        # LINE-BY-LINE: 조건 `length_values`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if length_values:
            # LINE-BY-LINE: 콘솔에 `print(f"- length_range_in_jobs: {min(length_values)}..{max(length_values)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- length_range_in_jobs: {min(length_values)}..{max(length_values)}")


def _phase1_bay_ids_from_env(env: CuttingShopEnvironment, requested_bay_ids: str | None) -> list[str]:
    """Return Phase 1 Bay IDs from CLI override or enabled machine layout."""

    enabled_bay_ids = sorted(
        {
            str(machine.bay_id)
            for machine in env.machines.values()
            if machine.enabled and machine.bay_id not in (None, "")
        }
    )
    if not enabled_bay_ids:
        print("[ERROR][main._phase1_bay_ids_from_env] cause=no_enabled_machine_bays")
        raise RuntimeError("Phase 1 requires at least one enabled machine Bay")

    if requested_bay_ids is None or str(requested_bay_ids).strip() == "":
        return enabled_bay_ids

    selected_bay_ids = parse_csv_argument(requested_bay_ids, default=enabled_bay_ids)
    unknown = sorted(set(selected_bay_ids) - set(enabled_bay_ids))
    if unknown:
        print(
            "[ERROR][main._phase1_bay_ids_from_env] "
            f"cause=requested_bay_not_in_enabled_factory requested={unknown} enabled={enabled_bay_ids}"
        )
        raise RuntimeError(f"requested Phase 1 Bay IDs are not enabled in factory: {unknown}")
    return selected_bay_ids


def _phase1_bay_capacity_weights_from_env(env: CuttingShopEnvironment, bay_ids: Sequence[str]) -> dict[str, float]:
    """Return enabled machine counts per selected Bay for Phase 1 load normalization."""

    selected = {str(bay_id) for bay_id in bay_ids}
    weights = {str(bay_id): 0.0 for bay_id in bay_ids}
    for machine in env.machines.values():
        bay_id = str(machine.bay_id)
        if not machine.enabled or bay_id not in selected:
            continue
        weights[bay_id] += 1.0
    missing = [bay_id for bay_id, weight in weights.items() if weight <= 0]
    if missing:
        print(
            "[ERROR][main._phase1_bay_capacity_weights_from_env] "
            f"cause=no_enabled_machine_in_selected_bay bay_ids={missing}"
        )
        raise RuntimeError(f"Phase 1 selected Bays have no enabled machines: {missing}")
    return weights


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
    print("- synthetic_source: mixed_physical_block_joint_distribution")
    print("- rule_profile: multi_series_260711")
    print("- score_mode: wo_first")
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
    print(f"- device: {args.device}")
    print(f"- output_dir: {args.output_dir}")

    def episode_payload(episode: int, *, validation: bool) -> dict:
        seed_offset = 10_000_000 if validation else 0
        payload = build_phase1_episode_jobs(
            episode_count=1,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed + seed_offset + episode * 1_000_003,
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
    print(f"- synthetic_source: mixed_physical_block_joint_distribution")
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
        return mixed_jobs(args.seed + episode * 1_000_003)

    validation_episode_job_factory = None
    if args.validation_episodes > 0:
        def validation_episode_job_factory(validation_episode: int) -> Mapping[str, object]:
            return mixed_jobs(args.seed + 10_000_000 + validation_episode * 1_000_003)

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
    )
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
    print("- synthetic_source: mixed_physical_block_joint_distribution")
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
    batch_machine_model = _load_phase2_full_flow_model(
        checkpoint=checkpoint_path,
        context="phase2_batch_machine",
        conflicting_heuristic=batch_machine_heuristic,
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


def _phase1_assignments_from_plan(plan: dict) -> dict[str, str]:
    assignments = plan.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        print(
            "[ERROR][main._phase1_assignments_from_plan] "
            f"cause=no_assignments rows_type={type(assignments).__name__}"
        )
        raise RuntimeError("Phase 1 plan has no assignments")
    result = {}
    for index, row in enumerate(assignments):
        block_set_id = str(row.get("block_set_id") or "").strip()
        assigned_bay = str(row.get("assigned_bay") or "").strip()
        if not block_set_id or not assigned_bay:
            print(
                "[ERROR][main._phase1_assignments_from_plan] "
                f"cause=invalid_assignment index={index} row={row}"
            )
            raise RuntimeError(f"invalid Phase 1 assignment at index={index}")
        if block_set_id in result and result[block_set_id] != assigned_bay:
            print(
                "[ERROR][main._phase1_assignments_from_plan] "
                f"cause=conflicting_assignment block_set_id={block_set_id} "
                f"old={result[block_set_id]} new={assigned_bay}"
            )
            raise RuntimeError(f"conflicting Phase 1 assignment: {block_set_id}")
        result[block_set_id] = assigned_bay
    return result


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


def _resolve_phase1_training_bay_ids(value: str | None, machines: Mapping[str, Any]) -> tuple[str, ...]:
    """Phase 2 학습 중 Phase 1 heuristic을 실행할 Bay 목록을 결정한다."""

    explicit = parse_csv_argument(value, default=())
    if explicit:
        return tuple(str(bay_id) for bay_id in explicit)
    bay_ids = []
    for machine_id, machine in sorted(machines.items()):
        bay_id = machine.get("bay_id") if isinstance(machine, Mapping) else getattr(machine, "bay_id", None)
        if bay_id in (None, ""):
            print(
                "[ERROR][main._resolve_phase1_training_bay_ids] "
                f"cause=missing_machine_bay_id machine_id={machine_id}"
            )
            raise RuntimeError(f"missing machine bay_id: {machine_id}")
        bay_ids.append(str(bay_id))
    result = tuple(dict.fromkeys(bay_ids))
    if not result:
        print("[ERROR][main._resolve_phase1_training_bay_ids] cause=no_bay_ids")
        raise RuntimeError("Phase 1 training bay IDs are required")
    return result


def _filter_machines_by_bay_ids(machines: Mapping[str, Any], bay_ids: Sequence[str]) -> dict[str, Any]:
    """Phase 학습에서 사용하지 않는 Bay의 설비를 score와 CSV에서 제외한다."""

    allowed = {str(bay_id) for bay_id in bay_ids}
    result = {}
    for machine_id, machine in machines.items():
        bay_id = machine.get("bay_id") if isinstance(machine, Mapping) else getattr(machine, "bay_id", None)
        if str(bay_id) in allowed:
            result[str(machine_id)] = machine
    if not result:
        print(
            "[ERROR][main._filter_machines_by_bay_ids] "
            f"cause=no_machine_after_filter bay_ids={sorted(allowed)}"
        )
        raise RuntimeError(f"no enabled machines for selected Bay IDs: {sorted(allowed)}")
    return result


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
    capacity_weights = joint_phase1_bay_capacity_weights()
    bay_ids = tuple(capacity_weights)

    def build(jobs: Mapping[str, object], assignment_seed: int) -> Mapping[str, str]:
        candidates = []
        if sample_count == 1:
            candidates.append(
                run_phase1_pair_policy_rollout(
                    jobs=jobs,
                    bay_ids=bay_ids,
                    model=model,
                    temperature=temperature,
                    seed=seed + assignment_seed,
                    source="phase1_agent_greedy",
                    selection="greedy",
                    bay_capacity_weights=capacity_weights,
                )
            )
        else:
            for sample_index in range(1, sample_count + 1):
                candidates.append(
                    run_phase1_pair_policy_rollout(
                        jobs=jobs,
                        bay_ids=bay_ids,
                        model=model,
                        temperature=temperature,
                        seed=seed + assignment_seed * 10_000 + sample_index,
                        source=f"phase1_agent_sample_{sample_index}",
                        selection="sample",
                        bay_capacity_weights=capacity_weights,
                    )
                )
        best = min(candidates, key=lambda candidate: score_phase1_bay_loads(candidate.bay_loads))
        print(
            "[CHECK][main._phase1_agent_assignment_builder] "
            f"assignment_seed={assignment_seed} source={best.source} samples={sample_count} "
            f"score={score_phase1_bay_loads(best.bay_loads)}",
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
            "job_source": "mixed_physical_block_joint_distribution",
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
        candidate = run_phase1_heuristic_candidate(
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
    candidates = []
    if args.phase1_samples == 1:
        candidates.append(
            run_phase1_pair_policy_rollout(
                jobs=jobs,
                bay_ids=bay_ids,
                model=model,
                temperature=args.phase1_temperature,
                seed=args.seed,
                source="phase1_agent_greedy",
                selection="greedy",
                bay_capacity_weights=bay_capacity_weights,
            )
        )
    else:
        for sample_index in range(args.phase1_samples):
            sample_number = sample_index + 1
            if sample_number == 1 or sample_number % 64 == 0 or sample_number == args.phase1_samples:
                print(
                    "[CHECK][main._load_or_build_phase1_plan_for_full_flow.sample_progress] "
                    f"sample={sample_number}/{args.phase1_samples}",
                    flush=True,
                )
            candidates.append(
                run_phase1_pair_policy_rollout(
                    jobs=jobs,
                    bay_ids=bay_ids,
                    model=model,
                    temperature=args.phase1_temperature,
                    seed=args.seed + sample_index,
                    source=f"phase1_agent_sample_{sample_number}",
                    selection="sample",
                    bay_capacity_weights=bay_capacity_weights,
                )
            )
    best = min(candidates, key=lambda candidate: score_phase1_bay_loads(candidate.bay_loads))
    plan = candidate_to_phase1_plan(
        jobs=jobs,
        bay_ids=bay_ids,
        candidate=best,
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


def _load_phase2_full_flow_model(
    checkpoint: str | None,
    context: str,
    conflicting_heuristic: str | None,
):
    checkpoint_path = _optional_non_empty_cli_value(checkpoint, f"{context}_checkpoint")
    if checkpoint_path is None:
        return None
    if conflicting_heuristic is not None:
        print(
            "[ERROR][main._load_phase2_full_flow_model] "
            f"cause=checkpoint_heuristic_conflict context={context} "
            f"checkpoint={checkpoint_path} heuristic={conflicting_heuristic}"
        )
        raise RuntimeError(f"Choose either {context} checkpoint or heuristic, not both")
    return load_phase2_set_pointer_checkpoint(
        checkpoint_path=checkpoint_path,
        context=context,
    )


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
    if score_mode == "raw":
        score_fields = PHASE2_BATCH_MACHINE_SCORE_FIELD_NAMES
    elif score_mode == "normalized":
        score_fields = PHASE2_BATCH_MACHINE_NORMALIZED_SCORE_FIELD_NAMES
    else:
        print(
            "[ERROR][main._resolve_phase2_full_flow_run_spec] "
            f"cause=invalid_score_mode value={score_mode}"
        )
        raise RuntimeError(f"invalid full-flow Phase 2 score mode: {score_mode}")
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


def command_apply_phase1_to_phase2(args: argparse.Namespace) -> None:
    """Write a Phase 2 scenario whose W/Os follow a Phase 1 block-Bay plan."""

    print("[apply-phase1-to-phase2]")
    print(f"- config: {args.config}")
    print(f"- scenario_path_override: {args.scenario_path}")
    print(f"- phase1_plan: {args.phase1_plan}")
    print(f"- assignment_mode: {args.assignment_mode}")
    print(f"- output_scenario: {args.output_scenario}")

    plan_path = Path(args.phase1_plan)
    if not plan_path.exists():
        print(
            "[ERROR][main.command_apply_phase1_to_phase2] "
            f"cause=missing_phase1_plan path={plan_path}"
        )
        raise FileNotFoundError(f"Phase 1 plan does not exist: {plan_path}")

    config = load_config(args.config)
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    with plan_path.open("r", encoding="utf-8") as file:
        plan = json.load(file)

    result = apply_phase1_plan_to_scenario(
        scenario=scenario,
        plan=plan,
        assignment_mode=args.assignment_mode,
    )
    save_scenario(result["scenario"], args.output_scenario)
    summary = result["summary"]

    print(f"- assigned_job_count: {summary['assigned_job_count']}")
    print(f"- assigned_block_count: {summary['assigned_block_count']}")
    print(f"- plan_assignment_count: {summary['plan_assignment_count']}")
    print(f"- output_scenario: {args.output_scenario}")
    print("[VALIDATION][main.command_apply_phase1_to_phase2] passed=true")


def command_generate_phase1_blocks(args: argparse.Namespace) -> None:
    """물리 블록 공동분포를 보존한 MIXED block/W/O 파일을 만든다."""

    print("[generate-phase1-blocks]")
    print("- gyel: MIXED")
    print(f"- n_blocks: {args.n_blocks}")
    print(f"- seed: {args.seed}")
    synthetic_source = "mixed_physical_block_joint_distribution"
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
        "max": ["LTH", "BTH", "THK", "MARK_LTH", "TACT_TIME"],
        "sum": ["CUT_LTH", "BVL_LTH", "STL_QTY", "BV_QTY", "PTLST_QTY"],
        "WO_QTY": "W/O row count",
    }
    formula_series = ("NP", "FN", "FL", "NC")
    bth_profiles = {}
    for series in formula_series:
        profile = load_bth_formula_profile(
            str(DEFAULT_MULTI_SERIES_WO_SOURCE.resolve()),
            series,
        )
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


# LINE-BY-LINE: `command_simulate(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_simulate(args: argparse.Namespace) -> None:
    """휴리스틱 1개로 에피소드를 끝까지 실행합니다."""

    # LINE-BY-LINE: `env`에 `build_environment(args.config, scenario_path=args.scenario_path)` 결과를 저장합니다. 의미/사용: `env`는 CuttingShopEnvironment 객체입니다. reset/step/run_with_policy를 제공합니다.
    env = build_environment(args.config, scenario_path=getattr(args, "scenario_path", None))
    apply_action_space_overrides(env, args)
    # LINE-BY-LINE: `heuristic_name`에 `args.heuristic or env.config["agent"]["default_heuristic"]` 결과를 저장합니다. 의미/사용: `heuristic_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    heuristic_name = args.heuristic or env.config["agent"]["default_heuristic"]
    # LINE-BY-LINE: `action_mode`에 `env.config.get("action_space", {}).get("mode", "job_machine_pair")` 결과를 저장합니다. 의미/사용: `action_mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    action_mode = env.config.get("action_space", {}).get("mode", "job_machine_pair")
    # LINE-BY-LINE: 조건 `action_mode == "job_machine_pair" and heuristic_name == "spt"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if action_mode == "job_machine_pair" and heuristic_name == "spt":
        # LINE-BY-LINE: `execution_mode`에 `"fast_spt"` 결과를 저장합니다. 의미/사용: `execution_mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        execution_mode = "fast_spt"
        # LINE-BY-LINE: `summary`에 `env.simulation.run_spt_fast()` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
        summary = env.simulation.run_spt_fast()
    # LINE-BY-LINE: 앞선 조건이 거짓일 때 `action_mode == "job_machine_pair" and heuristic_name == "load_balance"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
    elif action_mode == "job_machine_pair" and heuristic_name == "load_balance":
        # LINE-BY-LINE: `execution_mode`에 `"fast_load_balance"` 결과를 저장합니다. 의미/사용: `execution_mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        execution_mode = "fast_load_balance"
        # LINE-BY-LINE: `summary`에 `env.simulation.run_load_balance_fast()` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
        summary = env.simulation.run_load_balance_fast()
    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
    else:
        # LINE-BY-LINE: `execution_mode`에 `"generic_candidates"` 결과를 저장합니다. 의미/사용: `execution_mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        execution_mode = "generic_candidates"
        # LINE-BY-LINE: `summary`에 `env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, ...` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
        summary = env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, heuristic_name))

    # LINE-BY-LINE: 콘솔에 `print("[simulation]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[simulation]")
    # LINE-BY-LINE: 콘솔에 `print(f"- heuristic: {heuristic_name}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- heuristic: {heuristic_name}")
    # LINE-BY-LINE: 콘솔에 `print(f"- action_mode: {action_mode}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- action_mode: {action_mode}")
    # LINE-BY-LINE: 콘솔에 `print(f"- execution_mode: {execution_mode}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- execution_mode: {execution_mode}")
    # LINE-BY-LINE: 콘솔에 `print(f"- current_time: {summary['current_time']:.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- current_time: {summary['current_time']:.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- makespan: {summary['makespan']:.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- makespan: {summary['makespan']:.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {summary['scheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_log_count: {len(summary.get('event_log', []))}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_log_count: {len(summary.get('event_log', []))}")
    # LINE-BY-LINE: 콘솔에 `print(f"- downstream_loads: {summary['downstream_loads']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- downstream_loads: {summary['downstream_loads']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_loads: {summary['machine_loads']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_loads: {summary['machine_loads']}")


# LINE-BY-LINE: `run_heuristic_with_trace(env: CuttingShopEnvironment, heuristic_name: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def run_heuristic_with_trace(env: CuttingShopEnvironment, heuristic_name: str):
    """휴리스틱을 실행하면서 playback용 decision trace를 남깁니다."""

    # LINE-BY-LINE: `observation, _` 여러 변수에 `env.reset()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    observation, _ = env.reset()
    # LINE-BY-LINE: `action_mode`에 `env.config.get("action_space", {}).get("mode", "job_machine_pair")` 결과를 저장합니다. 의미/사용: `action_mode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    action_mode = env.config.get("action_space", {}).get("mode", "job_machine_pair")
    # LINE-BY-LINE: `trace_records`에 `[]` 결과를 저장합니다. 의미/사용: `trace_records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trace_records = []
    # LINE-BY-LINE: `step_index`에 `0` 결과를 저장합니다. 의미/사용: `step_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    step_index = 0
    # LINE-BY-LINE: `terminated`에 `False` 결과를 저장합니다. 의미/사용: `terminated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    terminated = False
    # LINE-BY-LINE: `truncated`에 `False` 결과를 저장합니다. 의미/사용: `truncated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    truncated = False

    # LINE-BY-LINE: 조건 `not (terminated or truncated)`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while not (terminated or truncated):
        # LINE-BY-LINE: 조건 `action_mode == "job_machine_pair" and heuristic_name == "spt"`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if action_mode == "job_machine_pair" and heuristic_name == "spt":
            # LINE-BY-LINE: `candidates`에 `[]` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
            candidates = []
            # LINE-BY-LINE: `selected`에 `env.simulation.select_spt_candidate_fast()` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
            selected = env.simulation.select_spt_candidate_fast()
        # LINE-BY-LINE: 앞선 조건이 거짓일 때 `action_mode == "job_machine_pair" and heuristic_name == "load_balance"`를 추가로 검사합니다. 사용: 여러 처리 모드 중 하나를 선택합니다.
        elif action_mode == "job_machine_pair" and heuristic_name == "load_balance":
            # LINE-BY-LINE: `candidates`에 `[]` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
            candidates = []
            # LINE-BY-LINE: `selected`에 `env.simulation.select_load_balance_candidate_fast()` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
            selected = env.simulation.select_load_balance_candidate_fast()
        # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
        else:
            # LINE-BY-LINE: `candidates`에 `env.get_action_candidates()` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
            candidates = env.get_action_candidates()
            # LINE-BY-LINE: `selected`에 `select_action_by_rule(candidates, env.simulation, heuristic_name) if candidates else None` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
            selected = select_action_by_rule(candidates, env.simulation, heuristic_name) if candidates else None
        # LINE-BY-LINE: 조건 `selected is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected is None:
            # LINE-BY-LINE: `before_time`에 `env.simulation.state.current_time` 결과를 저장합니다. 의미/사용: `before_time` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            before_time = env.simulation.state.current_time
            # LINE-BY-LINE: `env.simulation._advance_to_decision_epoch()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            env.simulation._advance_to_decision_epoch()
            # LINE-BY-LINE: 조건 `env.simulation.state.current_time <= before_time + 1e-9`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if env.simulation.state.current_time <= before_time + 1e-9:
                # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
                break
            # LINE-BY-LINE: 현재 반복의 남은 처리를 건너뛰고 다음 항목으로 넘어갑니다. 사용: 제외 row나 불가능 후보를 skip.
            continue

        # LINE-BY-LINE: `trace_record`에 `{` 결과를 저장합니다. 의미/사용: `trace_record` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        trace_record = {
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `step` 키에 `step_index` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "step": step_index,
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `decision_time_min` 키에 `round(float(env.simulation.state.current_time), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "decision_time_min": round(float(env.simulation.state.current_time), 6),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `candidate_count` 키에 `len(candidates) if candidates else -1` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "candidate_count": len(candidates) if candidates else -1,
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `remaining_jobs_before` 키에 `len(env.simulation.state.unscheduled_jobs)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "remaining_jobs_before": len(env.simulation.state.unscheduled_jobs),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `action_id` 키에 `selected.action_id` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_id": selected.action_id,
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `action_kind` 키에 `selected.action_kind` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "action_kind": selected.action_kind,
            # LINE-BY-LINE: 딕셔너리 키 `batch_id`에는 `selected.batch_id or ""` 값을 넣습니다. 의미: open/batch action을 식별하는 ID입니다. 예: `B000001`, 사용: open_batches와 event 연결.
            "batch_id": selected.batch_id or "",
            # LINE-BY-LINE: 딕셔너리 키 `batch_job_ids`에는 `"|".join(selected.batch_job_ids)` 값을 넣습니다. 의미: batch에 포함된 W/O ID 목록입니다. 사용: batch schedule/playback 표시.
            "batch_job_ids": "|".join(selected.batch_job_ids),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `batch_wo_count` 키에 `selected.batch_wo_count` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "batch_wo_count": selected.batch_wo_count,
            # LINE-BY-LINE: 딕셔너리 키 `batch_length_sum`에는 `round(float(selected.batch_length_sum), 6)` 값을 넣습니다. 의미: batch 또는 active W/O들의 길이합입니다. 예: `<=55000`, 사용: capacity 제약.
            "batch_length_sum": round(float(selected.batch_length_sum), 6),
            # LINE-BY-LINE: 딕셔너리 키 `job_id`에는 `selected.job.job_id` 값을 넣습니다. 의미: W/O를 환경 내부에서 식별하는 key입니다. 예: `job_001_WO123`, 사용: action_id, schedule row, event log 연결.
            "job_id": selected.job.job_id,
            # LINE-BY-LINE: 딕셔너리 키 `machine_id`에는 `selected.machine.machine_id` 값을 넣습니다. 의미: 절단 설비 ID입니다. 예: `PLS21`, 사용: machine dict 조회, action 선택, event log.
            "machine_id": selected.machine.machine_id,
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `machine_bay_id` 키에 `selected.machine.bay_id or ""` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "machine_bay_id": selected.machine.bay_id or "",
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `estimated_minutes` 키에 `round(float(selected.estimated_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "estimated_minutes": round(float(selected.estimated_minutes), 6),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `processing_minutes` 키에 `round(float(selected.processing_minutes), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "processing_minutes": round(float(selected.processing_minutes), 6),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `finish_time_min` 키에 `round(float(selected.finish_time), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "finish_time_min": round(float(selected.finish_time), 6),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `soft_penalty` 키에 `round(float(selected.soft_penalty), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "soft_penalty": round(float(selected.soft_penalty), 6),
            # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `soft_reasons` 키에 `"|".join(selected.soft_reasons)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
            "soft_reasons": "|".join(selected.soft_reasons),
        }

        # LINE-BY-LINE: `observation, reward, terminated, truncated, info` 여러 변수에 `env.simulation.step_candidate(` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        observation, reward, terminated, truncated, info = env.simulation.step_candidate(
            # LINE-BY-LINE: `step_candidate(...)` 호출에 `selected` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            selected,
            # LINE-BY-LINE: `build_observation_result`에 `not (action_mode == "job_machine_pair" and heuristic_name in {"spt", "load_balance"})` 결과를 저장합니다. 의미/사용: `build_observation_result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            build_observation_result=not (action_mode == "job_machine_pair" and heuristic_name in {"spt", "load_balance"}),
            # LINE-BY-LINE: `advance_decision_epoch`에 `not (action_mode == "job_machine_pair" and heuristic_name in {"spt", "load_balance"})` 결과를 저장합니다. 의미/사용: `advance_decision_epoch` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            advance_decision_epoch=not (action_mode == "job_machine_pair" and heuristic_name in {"spt", "load_balance"}),
        )
        # LINE-BY-LINE: `trace_record.update(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        trace_record.update(
            # LINE-BY-LINE: `{` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
            {
                # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `reward` 키에 `round(float(reward), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "reward": round(float(reward), 6),
                # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `current_time_after_min` 키에 `round(float(observation["current_time"]), 6)` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "current_time_after_min": round(float(observation["current_time"]), 6),
                # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `remaining_jobs_after` 키에 `int(observation["remaining_job_count"])` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "remaining_jobs_after": int(observation["remaining_job_count"]),
                # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `terminated` 키에 `terminated` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "terminated": terminated,
                # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `truncated` 키에 `truncated` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
                "truncated": truncated,
            }
        )
        # LINE-BY-LINE: `trace_records.append(trace_record)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        trace_records.append(trace_record)
        # LINE-BY-LINE: `step_index` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `step_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        step_index += 1

    # LINE-BY-LINE: 호출자에게 dict 반환을 시작합니다. 사용: 여러 결과 값을 key-value로 묶어 상위 로직에 전달합니다.
    return {
        # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `observation` 키에 `observation` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "observation": observation,
        # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `trace_records` 키에 `trace_records` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "trace_records": trace_records,
        # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `terminated` 키에 `terminated` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "terminated": terminated,
        # LINE-BY-LINE: `run_heuristic_with_trace`에서 반환/저장할 dict의 `truncated` 키에 `truncated` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "truncated": truncated,
    }


# LINE-BY-LINE: `command_build_scenario(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def command_build_scenario(args: argparse.Namespace) -> None:
    """실제 절단 Excel/CSV에서 NP 100건 scenario를 생성합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    # LINE-BY-LINE: `cleaned`에 `load_and_clean_cutting_data(` 결과를 저장합니다. 의미/사용: `cleaned` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    cleaned = load_and_clean_cutting_data(
        # LINE-BY-LINE: `load_and_clean_cutting_data(...)` 호출에 `args.input_path` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        args.input_path,
        # LINE-BY-LINE: `sheet_name`에 `args.sheet_name` 결과를 저장합니다. 의미/사용: `sheet_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        sheet_name=args.sheet_name,
        # LINE-BY-LINE: `target_series`에 `tuple(item.strip() for item in args.target_series.split(",") if item.strip())` 결과를 저장합니다. 의미/사용: `target_series` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        target_series=tuple(item.strip() for item in args.target_series.split(",") if item.strip()),
        # LINE-BY-LINE: `max_records`에 `args.sample_size` 결과를 저장합니다. 의미/사용: `max_records` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        max_records=args.sample_size,
    )
    # LINE-BY-LINE: `scenario`에 `build_scenario_from_cutting_records(` 결과를 저장합니다. 의미/사용: `scenario`는 scenario YAML에서 읽은 factory/jobs 데이터 dict입니다. 예: `input/np_100_scenario.yaml`.
    scenario = build_scenario_from_cutting_records(
        # LINE-BY-LINE: `build_scenario_from_cutting_records(...)` 호출에 `cleaned.records` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        cleaned.records,
        # LINE-BY-LINE: `factory_config`에 `config.get("factory")` 결과를 저장합니다. 의미/사용: `factory_config` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        factory_config=config.get("factory"),
        # LINE-BY-LINE: `source_file`에 `args.input_path` 결과를 저장합니다. 의미/사용: `source_file` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        source_file=args.input_path,
        # LINE-BY-LINE: `process_time_source`에 `args.process_time_source` 결과를 저장합니다. 의미/사용: `process_time_source` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        process_time_source=args.process_time_source,
    )

    # LINE-BY-LINE: `save_scenario(scenario, args.output_path)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    save_scenario(scenario, args.output_path)
    # LINE-BY-LINE: `write_records_csv(cleaned.records, args.csv_output)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    write_records_csv(cleaned.records, args.csv_output)
    # LINE-BY-LINE: `write_records_csv(cleaned.excluded_records, args.excluded_output)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    write_records_csv(cleaned.excluded_records, args.excluded_output)

    # LINE-BY-LINE: 콘솔에 `print("[build scenario]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[build scenario]")
    # LINE-BY-LINE: 콘솔에 `print(f"- input_path: {args.input_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- input_path: {args.input_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- selected_records: {len(cleaned.records)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- selected_records: {len(cleaned.records)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- excluded_records: {len(cleaned.excluded_records)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- excluded_records: {len(cleaned.excluded_records)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {args.output_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {args.output_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- process_time_source: {args.process_time_source}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- process_time_source: {args.process_time_source}")
    # LINE-BY-LINE: 콘솔에 `print(f"- csv_output: {args.csv_output}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- csv_output: {args.csv_output}")
    # LINE-BY-LINE: 콘솔에 `print(f"- excluded_output: {args.excluded_output}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- excluded_output: {args.excluded_output}")


# LINE-BY-LINE: `command_playback(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_playback(args: argparse.Namespace) -> None:
    """휴리스틱 실행 결과를 공장 시간흐름 playback 산출물로 저장합니다."""

    # LINE-BY-LINE: `env`에 `build_environment(args.config, scenario_path=args.scenario_path)` 결과를 저장합니다. 의미/사용: `env`는 CuttingShopEnvironment 객체입니다. reset/step/run_with_policy를 제공합니다.
    env = build_environment(args.config, scenario_path=args.scenario_path)
    apply_action_space_overrides(env, args)
    # LINE-BY-LINE: `heuristic_name`에 `args.heuristic or env.config["agent"]["default_heuristic"]` 결과를 저장합니다. 의미/사용: `heuristic_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    heuristic_name = args.heuristic or env.config["agent"]["default_heuristic"]
    # LINE-BY-LINE: `run_result`에 `run_heuristic_with_trace(env, heuristic_name)` 결과를 저장합니다. 의미/사용: `run_result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    run_result = run_heuristic_with_trace(env, heuristic_name)

    # LINE-BY-LINE: `output_dir`에 `args.output_dir` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = args.output_dir
    # LINE-BY-LINE: 조건 `output_dir is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if output_dir is None:
        # LINE-BY-LINE: `output_dir`에 `str(Path(env.config["paths"]["output_dir"]) / "playback_np_100")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        output_dir = str(Path(env.config["paths"]["output_dir"]) / "playback_np_100")

    # LINE-BY-LINE: `artifacts`에 `write_playback_artifacts(` 결과를 저장합니다. 의미/사용: `artifacts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    artifacts = write_playback_artifacts(
        # LINE-BY-LINE: `write_playback_artifacts(...)` 호출에 `env.simulation` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        env.simulation,
        # LINE-BY-LINE: `write_playback_artifacts(...)` 호출에 `env.config` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        env.config,
        # LINE-BY-LINE: `write_playback_artifacts(...)` 호출에 `run_result["trace_records"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        run_result["trace_records"],
        # LINE-BY-LINE: `write_playback_artifacts(...)` 호출에 `output_dir` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        output_dir,
    )

    # LINE-BY-LINE: 콘솔에 `print("[playback]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[playback]")
    # LINE-BY-LINE: 콘솔에 `print(f"- heuristic: {heuristic_name}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- heuristic: {heuristic_name}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {len(env.simulation.state.schedule)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scheduled_jobs: {len(env.simulation.state.schedule)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- unscheduled_jobs: {len(env.simulation.state.unscheduled_jobs)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- unscheduled_jobs: {len(env.simulation.state.unscheduled_jobs)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- makespan_minutes: {env.simulation.get_makespan():.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- makespan_minutes: {env.simulation.get_makespan():.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- hard_passed: {artifacts['hard_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- hard_passed: {artifacts['hard_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- hard_violation_count: {artifacts['hard_violation_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- hard_violation_count: {artifacts['hard_violation_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- hard_violation_affected_job_count: {artifacts['hard_violation_affected_job_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- hard_violation_affected_job_count: {artifacts['hard_violation_affected_job_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- hard_violation_excess_wo_count: {artifacts['hard_violation_excess_wo_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- hard_violation_excess_wo_count: {artifacts['hard_violation_excess_wo_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_log_count: {artifacts['event_log_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_log_count: {artifacts['event_log_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- playback_event_count: {artifacts['playback_event_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- playback_event_count: {artifacts['playback_event_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_identity_passed: {artifacts['event_identity_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_identity_passed: {artifacts['event_identity_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_identity_error_count: {artifacts['event_identity_error_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_identity_error_count: {artifacts['event_identity_error_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_constraint_passed: {artifacts['event_constraint_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_constraint_passed: {artifacts['event_constraint_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_constraint_violation_count: {artifacts['event_constraint_violation_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_constraint_violation_count: {artifacts['event_constraint_violation_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- duration_mae_minutes: {float(artifacts['duration_mae_minutes']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- duration_mae_minutes: {float(artifacts['duration_mae_minutes']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- duration_mape_percent: {float(artifacts['duration_mape_percent']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- duration_mape_percent: {float(artifacts['duration_mape_percent']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_mae_minutes: {float(artifacts['tact_mae_minutes']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_mae_minutes: {float(artifacts['tact_mae_minutes']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_mape_percent: {float(artifacts['tact_mape_percent']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_mape_percent: {float(artifacts['tact_mape_percent']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- playback_html: {artifacts['playback_html']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- playback_html: {artifacts['playback_html']}")


# LINE-BY-LINE: `command_actual_replay(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_actual_replay(args: argparse.Namespace) -> None:
    """실적 데이터 순서/장비/시간을 그대로 playback 산출물로 저장합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    replay_config = config
    if not scenario_path:
        replay_config = dict(config)
        replay_data_config = dict(config.get("data", {}))
        original_process_time_source = replay_data_config.get("process_time_source")
        replay_data_config["process_time_source"] = "actual_duration"
        replay_config["data"] = replay_data_config
        print(
            "[CHECK][main.command_actual_replay] "
            "process_time_source=actual_duration "
            f"original_process_time_source={original_process_time_source} "
            "reason=actual_replay_uses_actual_timestamps"
        )
    scenario = load_scenario_for_config(replay_config, scenario_path_override=args.scenario_path)

    # LINE-BY-LINE: `output_dir`에 `args.output_dir` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = args.output_dir
    # LINE-BY-LINE: 조건 `output_dir is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if output_dir is None:
        # LINE-BY-LINE: `output_dir`에 `str(Path(config["paths"]["output_dir"]) / "actual_replay_np_100")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        output_dir = str(Path(config["paths"]["output_dir"]) / "actual_replay_np_100")

    # LINE-BY-LINE: `artifacts`에 `write_actual_replay_artifacts(scenario, config, output_dir)` 결과를 저장합니다. 의미/사용: `artifacts` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    artifacts = write_actual_replay_artifacts(scenario, config, output_dir)

    # LINE-BY-LINE: 콘솔에 `print("[actual replay]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[actual replay]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print("- order: actual_start_datetime -> actual_end_datetime -> wk_seq/source_row_index")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("- order: actual_start_datetime -> actual_end_datetime -> wk_seq/source_row_index")
    # LINE-BY-LINE: 콘솔에 `print("- factory_replay_validation: reproduce RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM exact...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("- factory_replay_validation: reproduce RT_EQP_NM/CUT_BAY/RT_CUT_ST_DTM/RT_CUT_ED_DTM exactly")
    # LINE-BY-LINE: 콘솔에 `print("- constraint_audit: planning constraints are reported separately, not used as actual repla...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("- constraint_audit: planning constraints are reported separately, not used as actual replay failure")
    # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {artifacts['scheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scheduled_jobs: {artifacts['scheduled_jobs']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- factory_replay_passed: {artifacts['factory_replay_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- factory_replay_passed: {artifacts['factory_replay_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- identity_error_count: {artifacts['identity_error_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- identity_error_count: {artifacts['identity_error_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_log_count: {artifacts['event_log_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_log_count: {artifacts['event_log_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- playback_event_count: {artifacts['playback_event_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- playback_event_count: {artifacts['playback_event_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_identity_passed: {artifacts['event_identity_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_identity_passed: {artifacts['event_identity_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_identity_error_count: {artifacts['event_identity_error_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_identity_error_count: {artifacts['event_identity_error_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_constraint_passed: {artifacts['event_constraint_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_constraint_passed: {artifacts['event_constraint_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- event_constraint_violation_count: {artifacts['event_constraint_violation_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- event_constraint_violation_count: {artifacts['event_constraint_violation_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- constraint_audit_passed: {artifacts['constraint_audit_passed']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- constraint_audit_passed: {artifacts['constraint_audit_passed']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- constraint_audit_violation_count: {artifacts['constraint_audit_violation_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- constraint_audit_violation_count: {artifacts['constraint_audit_violation_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- constraint_audit_affected_job_count: {artifacts['constraint_audit_affected_job_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- constraint_audit_affected_job_count: {artifacts['constraint_audit_affected_job_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- constraint_audit_excess_wo_count: {artifacts['constraint_audit_excess_wo_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- constraint_audit_excess_wo_count: {artifacts['constraint_audit_excess_wo_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- constraint_audit_length_excess_total: {float(artifacts['constraint_audit_length_excess_...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- constraint_audit_length_excess_total: {float(artifacts['constraint_audit_length_excess_total']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- missing_event_count: {artifacts['missing_event_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- missing_event_count: {artifacts['missing_event_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- missing_events_are_inferred: {artifacts['missing_events_are_inferred']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- missing_events_are_inferred: {artifacts['missing_events_are_inferred']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- duration_mae_minutes: {float(artifacts['duration_mae_minutes']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- duration_mae_minutes: {float(artifacts['duration_mae_minutes']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- duration_mape_percent: {float(artifacts['duration_mape_percent']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- duration_mape_percent: {float(artifacts['duration_mape_percent']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_mae_minutes: {float(artifacts['tact_mae_minutes']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_mae_minutes: {float(artifacts['tact_mae_minutes']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_mape_percent: {float(artifacts['tact_mape_percent']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_mape_percent: {float(artifacts['tact_mape_percent']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- playback_html: {artifacts['playback_html']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- playback_html: {artifacts['playback_html']}")


def command_gym_equivalence(args: argparse.Namespace) -> None:
    """DES action mask wrapper가 기존 core 휴리스틱과 같은 결과를 내는지 검증한다."""

    heuristic_name = args.heuristic
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("output") / f"gym_equivalence_{Path(args.config).stem}")

    print("[gym equivalence]")
    print(f"- config: {args.config}")
    print(f"- heuristic: {heuristic_name}")
    print(f"- max_actions: {args.max_actions}")
    print(f"- output_dir: {output_dir}")
    print(f"- gymnasium_available: {GYMNASIUM_AVAILABLE}")
    direct_env = build_environment(args.config)
    wrapper_env = build_environment(args.config)
    result = run_wrapper_equivalence(
        direct_env=direct_env,
        wrapper_env=wrapper_env,
        heuristic_name=heuristic_name,
        max_actions=args.max_actions,
        output_dir=output_dir,
    )
    print(f"- passed: {result['passed']}")
    print(f"- direct_step_count: {result['direct_step_count']}")
    print(f"- wrapper_step_count: {result['wrapper_step_count']}")
    print(f"- scheduled_jobs: {result['direct_summary']['scheduled_jobs']}")
    print(f"- makespan: {result['direct_summary']['makespan']:.6f}")
    print(f"- trace_csv: {result['trace_csv']}")
    print(f"- summary_json: {result['summary_json']}")


# LINE-BY-LINE: `command_trace(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_trace(args: argparse.Namespace) -> None:
    """step 단위로 어떤 action이 선택되는지 상세히 출력합니다."""

    # LINE-BY-LINE: `env`에 `build_environment(args.config)` 결과를 저장합니다. 의미/사용: `env`는 CuttingShopEnvironment 객체입니다. reset/step/run_with_policy를 제공합니다.
    env = build_environment(args.config)
    apply_action_space_overrides(env, args)
    # LINE-BY-LINE: `heuristic_name`에 `args.heuristic or env.config["agent"]["default_heuristic"]` 결과를 저장합니다. 의미/사용: `heuristic_name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    heuristic_name = args.heuristic or env.config["agent"]["default_heuristic"]
    # LINE-BY-LINE: `observation, info` 여러 변수에 `env.reset()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    observation, info = env.reset()

    # LINE-BY-LINE: 콘솔에 `print("[trace]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[trace]")
    # LINE-BY-LINE: 콘솔에 `print(f"- heuristic: {heuristic_name}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- heuristic: {heuristic_name}")
    # LINE-BY-LINE: 콘솔에 `print(f"- current_time: {observation['current_time']:.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- current_time: {observation['current_time']:.2f}")

    # LINE-BY-LINE: `step_index`에 `0` 결과를 저장합니다. 의미/사용: `step_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    step_index = 0
    # LINE-BY-LINE: `terminated`에 `False` 결과를 저장합니다. 의미/사용: `terminated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    terminated = False
    # LINE-BY-LINE: `truncated`에 `False` 결과를 저장합니다. 의미/사용: `truncated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    truncated = False

    # LINE-BY-LINE: 조건 `not (terminated or truncated)`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while not (terminated or truncated):
        # LINE-BY-LINE: `candidates`에 `env.get_action_candidates()` 결과를 저장합니다. 의미/사용: `candidates`는 현재 decision epoch에서 선택 가능한 action 후보 목록입니다.
        candidates = env.get_action_candidates()
        # LINE-BY-LINE: 조건 `not candidates`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if not candidates:
            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            break

        # LINE-BY-LINE: `selected`에 `select_action_by_rule(candidates, env.simulation, heuristic_name)` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
        selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"step`에 `{step_index} action={selected.action_id} "` 결과를 저장합니다. 의미/사용: `f"step` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"step={step_index} action={selected.action_id} "
            # LINE-BY-LINE: `f"job`에 `{selected.job.job_id} machine={selected.machine.machine_id} "` 결과를 저장합니다. 의미/사용: `f"job` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"job={selected.job.job_id} machine={selected.machine.machine_id} "
            # LINE-BY-LINE: `f"minutes`에 `{selected.estimated_minutes:.2f} soft_penalty={selected.soft_penalty:.3f}"` 결과를 저장합니다. 의미/사용: `f"minutes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"minutes={selected.estimated_minutes:.2f} soft_penalty={selected.soft_penalty:.3f}"
        )
        # LINE-BY-LINE: 조건 `selected.soft_reasons`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected.soft_reasons:
            # LINE-BY-LINE: 콘솔에 `print(f" soft_reasons={selected.soft_reasons}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"  soft_reasons={selected.soft_reasons}")

        # LINE-BY-LINE: `observation, reward, terminated, truncated, info` 여러 변수에 `env.step(selected.action_id)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        observation, reward, terminated, truncated, info = env.step(selected.action_id)
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"  -> reward`에 `{reward:.3f} current_time={observation['current_time']:.2f} "` 결과를 저장합니다. 의미/사용: `f"  -> reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"  -> reward={reward:.3f} current_time={observation['current_time']:.2f} "
            # LINE-BY-LINE: `f"remaining_jobs`에 `{observation['remaining_job_count']} available_actions={len(observation['available_actions'])}"` 결과를 저장합니다. 의미/사용: `f"remaining_jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"remaining_jobs={observation['remaining_job_count']} available_actions={len(observation['available_actions'])}"
        )
        # LINE-BY-LINE: `step_index` 값을 `1` 기준으로 누적/증가합니다. 의미/사용: `step_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        step_index += 1

    # LINE-BY-LINE: 콘솔에 `print("[trace summary]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[trace summary]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {len(env.simulation.state.schedule)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scheduled_jobs: {len(env.simulation.state.schedule)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- current_time: {env.simulation.state.current_time:.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- current_time: {env.simulation.state.current_time:.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- makespan: {env.simulation.get_makespan():.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- makespan: {env.simulation.get_makespan():.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_loads: {env.simulation.state.machine_loads}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_loads: {env.simulation.state.machine_loads}")


# LINE-BY-LINE: `command_analyze_tact(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_analyze_tact(args: argparse.Namespace) -> None:
    """택트타임 분석을 실행합니다."""

    # LINE-BY-LINE: `_format_optional(value)` 함수를 정의합니다. 반환 타입: `str`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
    def _format_optional(value) -> str:
        """None 또는 숫자 값을 CLI 출력용 문자열로 변환한다."""

        # LINE-BY-LINE: 호출자에게 `"None" if value is None else f"{float(value):.2f}"`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return "None" if value is None else f"{float(value):.2f}"

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    # LINE-BY-LINE: `analysis`에 `build_tact_time_analysis_from_scenario(scenario)` 결과를 저장합니다. 의미/사용: `analysis` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    analysis = build_tact_time_analysis_from_scenario(scenario)

    # LINE-BY-LINE: `output_dir`에 `Path(args.output_dir or Path(config["paths"]["output_dir"]) / "tact_time_analysis")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = Path(args.output_dir or Path(config["paths"]["output_dir"]) / "tact_time_analysis")
    # LINE-BY-LINE: `output_dir.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `summary_path`에 `output_dir / "tact_time_analysis_summary.json"` 결과를 저장합니다. 의미/사용: `summary_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    summary_path = output_dir / "tact_time_analysis_summary.json"
    # LINE-BY-LINE: `rows_path`에 `output_dir / "tact_time_analysis_rows.csv"` 결과를 저장합니다. 의미/사용: `rows_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    rows_path = output_dir / "tact_time_analysis_rows.csv"
    # LINE-BY-LINE: `validation_rows_path`에 `output_dir / "tact_time_formula_validation_rows.csv"` 결과를 저장합니다. 의미/사용: `validation_rows_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_rows_path = output_dir / "tact_time_formula_validation_rows.csv"
    # LINE-BY-LINE: `segment_rows_path`에 `output_dir / "tact_time_segment_formulas.csv"` 결과를 저장합니다. 의미/사용: `segment_rows_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    segment_rows_path = output_dir / "tact_time_segment_formulas.csv"
    # LINE-BY-LINE: `summary_path.write_text(json.dumps(analysis["summary"], ensure_ascii` 여러 변수에 `False, indent=2), encoding="utf-8")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    summary_path.write_text(json.dumps(analysis["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    # LINE-BY-LINE: `rows`에 `analysis["rows"]` 결과를 저장합니다. 의미/사용: `rows`는 CSV/JSON으로 저장하거나 분석할 row dict 목록입니다.
    rows = analysis["rows"]
    # LINE-BY-LINE: `rows_path.open("w", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `fieldnames`에 `list(rows[0].keys()) if rows else []` 결과를 저장합니다. 의미/사용: `fieldnames` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        fieldnames = list(rows[0].keys()) if rows else []
        # LINE-BY-LINE: `writer`에 `csv.DictWriter(handle, fieldnames=fieldnames)` 결과를 저장합니다. 의미/사용: `writer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # LINE-BY-LINE: 조건 `fieldnames`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if fieldnames:
            # LINE-BY-LINE: `writer.writeheader()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writeheader()
            # LINE-BY-LINE: `writer.writerows(rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writerows(rows)
    # LINE-BY-LINE: `validation_rows`에 `analysis.get("validation_rows", [])` 결과를 저장합니다. 의미/사용: `validation_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    validation_rows = analysis.get("validation_rows", [])
    # LINE-BY-LINE: `validation_rows_path.open("w", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with validation_rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `fieldnames`에 `list(validation_rows[0].keys()) if validation_rows else []` 결과를 저장합니다. 의미/사용: `fieldnames` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        fieldnames = list(validation_rows[0].keys()) if validation_rows else []
        # LINE-BY-LINE: `writer`에 `csv.DictWriter(handle, fieldnames=fieldnames)` 결과를 저장합니다. 의미/사용: `writer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # LINE-BY-LINE: 조건 `fieldnames`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if fieldnames:
            # LINE-BY-LINE: `writer.writeheader()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writeheader()
            # LINE-BY-LINE: `writer.writerows(validation_rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writerows(validation_rows)
    # LINE-BY-LINE: `segment_rows`에 `analysis.get("segment_formula_rows", [])` 결과를 저장합니다. 의미/사용: `segment_rows` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    segment_rows = analysis.get("segment_formula_rows", [])
    # LINE-BY-LINE: `segment_rows_path.open("w", encoding="utf-8-sig", newline="") as handle` 자원을 안전하게 엽니다. 사용: 파일 handle 자동 close 보장.
    with segment_rows_path.open("w", encoding="utf-8-sig", newline="") as handle:
        # LINE-BY-LINE: `fieldnames`에 `list(segment_rows[0].keys()) if segment_rows else []` 결과를 저장합니다. 의미/사용: `fieldnames` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        fieldnames = list(segment_rows[0].keys()) if segment_rows else []
        # LINE-BY-LINE: `writer`에 `csv.DictWriter(handle, fieldnames=fieldnames)` 결과를 저장합니다. 의미/사용: `writer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # LINE-BY-LINE: 조건 `fieldnames`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if fieldnames:
            # LINE-BY-LINE: `writer.writeheader()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writeheader()
            # LINE-BY-LINE: `writer.writerows(segment_rows)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            writer.writerows(segment_rows)

    # LINE-BY-LINE: `tact_vs_actual`에 `analysis["summary"]["tact_vs_actual"]` 결과를 저장합니다. 의미/사용: `tact_vs_actual` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_vs_actual = analysis["summary"]["tact_vs_actual"]
    # LINE-BY-LINE: `tact_formula_metrics`에 `analysis["summary"]["linear_formula_fit_to_tact_time"]["metrics"]` 결과를 저장합니다. 의미/사용: `tact_formula_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_formula_metrics = analysis["summary"]["linear_formula_fit_to_tact_time"]["metrics"]
    # LINE-BY-LINE: `actual_formula`에 `analysis["summary"]["recommended_tact_time_formula"]` 결과를 저장합니다. 의미/사용: `actual_formula` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_formula = analysis["summary"]["recommended_tact_time_formula"]
    # LINE-BY-LINE: `actual_formula_metrics`에 `actual_formula["metrics"]` 결과를 저장합니다. 의미/사용: `actual_formula_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_formula_metrics = actual_formula["metrics"]
    # LINE-BY-LINE: `core_holdout_metrics`에 `analysis["summary"]["core_formula_holdout_validation"]["validation_metrics"]` 결과를 저장합니다. 의미/사용: `core_holdout_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    core_holdout_metrics = analysis["summary"]["core_formula_holdout_validation"]["validation_metrics"]
    # LINE-BY-LINE: 콘솔에 `print("[tact analysis]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[tact analysis]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- record_count: {analysis['summary']['record_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- record_count: {analysis['summary']['record_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_vs_actual_mae_minutes: {_format_optional(tact_vs_actual['mae_minutes'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_vs_actual_mae_minutes: {_format_optional(tact_vs_actual['mae_minutes'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_vs_actual_mape_percent: {_format_optional(tact_vs_actual['mape_percent'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_vs_actual_mape_percent: {_format_optional(tact_vs_actual['mape_percent'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- formula_fit_to_tact_mae_minutes: {_format_optional(tact_formula_metrics['mae_minutes'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- formula_fit_to_tact_mae_minutes: {_format_optional(tact_formula_metrics['mae_minutes'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- formula_fit_to_tact_mape_percent: {_format_optional(tact_formula_metrics['mape_percent'...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- formula_fit_to_tact_mape_percent: {_format_optional(tact_formula_metrics['mape_percent'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- core_formula_fit_to_actual_mae_minutes: {_format_optional(actual_formula_metrics['mae_m...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- core_formula_fit_to_actual_mae_minutes: {_format_optional(actual_formula_metrics['mae_minutes'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- core_formula_fit_to_actual_mape_percent: {_format_optional(actual_formula_metrics['mape...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- core_formula_fit_to_actual_mape_percent: {_format_optional(actual_formula_metrics['mape_percent'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- core_holdout_mae_minutes: {_format_optional(core_holdout_metrics['mae_minutes'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- core_holdout_mae_minutes: {_format_optional(core_holdout_metrics['mae_minutes'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- core_holdout_mape_percent: {_format_optional(core_holdout_metrics['mape_percent'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- core_holdout_mape_percent: {_format_optional(core_holdout_metrics['mape_percent'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- recommended_formula: {actual_formula['formula']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- recommended_formula: {actual_formula['formula']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- summary_path: {summary_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- summary_path: {summary_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- rows_path: {rows_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- rows_path: {rows_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- validation_rows_path: {validation_rows_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- validation_rows_path: {validation_rows_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- segment_rows_path: {segment_rows_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- segment_rows_path: {segment_rows_path}")


# LINE-BY-LINE: `command_analyze_tact_gap(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_analyze_tact_gap(args: argparse.Namespace) -> None:
    """TACT_TIME과 실적 elapsed 차이를 원인별로 분해합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    # LINE-BY-LINE: `output_dir`에 `args.output_dir or str(Path(config["paths"]["output_dir"]) / "tact_gap_analysis")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = args.output_dir or str(Path(config["paths"]["output_dir"]) / "tact_gap_analysis")
    # LINE-BY-LINE: `analysis`에 `build_tact_gap_analysis(` 결과를 저장합니다. 의미/사용: `analysis` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    analysis = build_tact_gap_analysis(
        # LINE-BY-LINE: `build_tact_gap_analysis(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario,
        # LINE-BY-LINE: `output_dir`에 `output_dir` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        output_dir=output_dir,
        # LINE-BY-LINE: `excluded_csv_path`에 `args.excluded_csv` 결과를 저장합니다. 의미/사용: `excluded_csv_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        excluded_csv_path=args.excluded_csv,
    )
    # LINE-BY-LINE: `summary`에 `analysis["summary"]` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = analysis["summary"]
    # LINE-BY-LINE: `overall`에 `summary["overall"]` 결과를 저장합니다. 의미/사용: `overall` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    overall = summary["overall"]
    # LINE-BY-LINE: `target_metrics`에 `{` 결과를 저장합니다. 의미/사용: `target_metrics` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    target_metrics = {
        # LINE-BY-LINE: `row["target_name"]: row` 로직을 실행합니다. 사용: 현재 함수의 상태 계산, 데이터 변환, 검증 흐름을 구성합니다.
        row["target_name"]: row
        # LINE-BY-LINE: `row in summary.get("target_candidate_metrics", [])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for row in summary.get("target_candidate_metrics", [])
    }
    # LINE-BY-LINE: `recommended_target`에 `target_metrics.get("recommended_learning_target_candidate", {})` 결과를 저장합니다. 의미/사용: `recommended_target` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    recommended_target = target_metrics.get("recommended_learning_target_candidate", {})

    # LINE-BY-LINE: 콘솔에 `print("[tact gap analysis]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[tact gap analysis]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- record_count: {summary['record_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- record_count: {summary['record_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- tact_actual_pearson_corr: {float(summary['tact_actual_pearson_corr']):.4f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- tact_actual_pearson_corr: {float(summary['tact_actual_pearson_corr']):.4f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- mae_minutes: {float(overall['mae_minutes']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- mae_minutes: {float(overall['mae_minutes']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- mape_percent: {float(overall['mape_percent']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- mape_percent: {float(overall['mape_percent']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- bias_actual_minus_tact: {float(overall['bias_actual_minus_tact']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- bias_actual_minus_tact: {float(overall['bias_actual_minus_tact']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- within_20min_rate: {float(overall['within_20min_rate']):.2f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- within_20min_rate: {float(overall['within_20min_rate']):.2f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- same_machine_time_group_count_min_2: {summary['same_machine_time_group_count_min_2']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- same_machine_time_group_count_min_2: {summary['same_machine_time_group_count_min_2']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- same_machine_time_group_rows_min_2: {summary['same_machine_time_group_rows_min_2']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- same_machine_time_group_rows_min_2: {summary['same_machine_time_group_rows_min_2']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- max_same_machine_time_group_size: {summary['max_same_machine_time_group_size']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- max_same_machine_time_group_size: {summary['max_same_machine_time_group_size']}")
    # LINE-BY-LINE: 조건 `recommended_target`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if recommended_target:
        # LINE-BY-LINE: 콘솔에 `print(f"- recommended_learning_target_mae_to_tact: {float(recommended_target['target_vs_tact_mae_...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- recommended_learning_target_mae_to_tact: {float(recommended_target['target_vs_tact_mae_minutes']):.2f}")
        # LINE-BY-LINE: 콘솔에 `print(f"- recommended_learning_target_corr_with_tact: {float(recommended_target['corr_target_with...` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- recommended_learning_target_corr_with_tact: {float(recommended_target['corr_target_with_tact_time']):.4f}")
    # LINE-BY-LINE: 콘솔에 `print(f"- batch_window_issue_count: {summary.get('batch_window_issue_count', 0)}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- batch_window_issue_count: {summary.get('batch_window_issue_count', 0)}")
    # LINE-BY-LINE: 콘솔에 `print(f"- output_dir: {output_dir}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- output_dir: {output_dir}")


# LINE-BY-LINE: `command_generate_scenario(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다. 예: `input/np_100_scenario.yaml` 구조 생성/읽기.
def command_generate_scenario(args: argparse.Namespace) -> None:
    """학습용 시나리오를 생성합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    template = load_scenario_for_config(config)
    # LINE-BY-LINE: `generated`에 `generate_scenario_from_template(template, duplicate_jobs=args.duplicate_jobs)` 결과를 저장합니다. 의미/사용: `generated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    generated = generate_scenario_from_template(template, duplicate_jobs=args.duplicate_jobs)
    # LINE-BY-LINE: `save_scenario(generated, args.output_path)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    save_scenario(generated, args.output_path)

    # LINE-BY-LINE: 콘솔에 `print("[scenario generation]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[scenario generation]")
    # LINE-BY-LINE: 콘솔에 `print(f"- template_jobs: {len(template['jobs'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- template_jobs: {len(template['jobs'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- generated_jobs: {len(generated['jobs'])}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- generated_jobs: {len(generated['jobs'])}")
    # LINE-BY-LINE: 콘솔에 `print(f"- output_path: {args.output_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- output_path: {args.output_path}")


# LINE-BY-LINE: `command_select_test_data(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_select_test_data(args: argparse.Namespace) -> None:
    """실적 착수일 분포 기반 테스트 scenario를 생성합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    # LINE-BY-LINE: `output_dir`에 `args.output_dir or str(Path(config["paths"]["output_dir"]) / "test_data_selection")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = args.output_dir or str(Path(config["paths"]["output_dir"]) / "test_data_selection")
    # LINE-BY-LINE: `result`에 `select_actual_day_test_data(` 결과를 저장합니다. 의미/사용: `result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    result = select_actual_day_test_data(
        # LINE-BY-LINE: `select_actual_day_test_data(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario,
        # LINE-BY-LINE: `output_dir`에 `output_dir` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        output_dir=output_dir,
        # LINE-BY-LINE: `scenario_output_dir`에 `args.scenario_output_dir` 결과를 저장합니다. 의미/사용: `scenario_output_dir` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        scenario_output_dir=args.scenario_output_dir,
        # LINE-BY-LINE: `stratified_count`에 `args.stratified_count` 결과를 저장합니다. 의미/사용: `stratified_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        stratified_count=args.stratified_count,
        # LINE-BY-LINE: `low_usable_min_count`에 `args.low_usable_min_count` 결과를 저장합니다. 의미/사용: `low_usable_min_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        low_usable_min_count=args.low_usable_min_count,
    )
    # LINE-BY-LINE: `summary`에 `result["summary"]` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = result["summary"]

    # LINE-BY-LINE: 콘솔에 `print("[test data selection]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[test data selection]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- source_job_count: {summary['source_job_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- source_job_count: {summary['source_job_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- actual_day_count: {summary['actual_day_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- actual_day_count: {summary['actual_day_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- start_job_count_min: {summary['start_job_count_min']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- start_job_count_min: {summary['start_job_count_min']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- start_job_count_median: {summary['start_job_count_median']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- start_job_count_median: {summary['start_job_count_median']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- start_job_count_max: {summary['start_job_count_max']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- start_job_count_max: {summary['start_job_count_max']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- stratified_count: {summary['stratified_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- stratified_count: {summary['stratified_count']}")
    # LINE-BY-LINE: `row in result["slice_rows"]` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for row in result["slice_rows"]:
        # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(
            # LINE-BY-LINE: `f"- slice`에 `{row['slice_name']} day={row['actual_day']} "` 결과를 저장합니다. 의미/사용: `f"- slice` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"- slice={row['slice_name']} day={row['actual_day']} "
            # LINE-BY-LINE: `f"jobs`에 `{row['job_count']} scenario={row['scenario_path']}"` 결과를 저장합니다. 의미/사용: `f"jobs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            f"jobs={row['job_count']} scenario={row['scenario_path']}"
        )
    # LINE-BY-LINE: 콘솔에 `print(f"- output_dir: {summary['output_dir']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- output_dir: {summary['output_dir']}")


# LINE-BY-LINE: `command_select_actual_range(args: argparse.Namespace)` 함수를 정의합니다. 반환 타입: `None`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def command_select_actual_range(args: argparse.Namespace) -> None:
    """실적 착수시간 range 기준 테스트 scenario를 생성합니다."""

    # LINE-BY-LINE: `config`에 `load_config(args.config)` 결과를 저장합니다. 의미/사용: `config`는 config YAML에서 읽은 전체 설정 dict입니다. 예: `config_np_100.yaml`, 사용: 제약/환경 옵션.
    config = load_config(args.config)
    scenario_path = args.scenario_path or config.get("paths", {}).get("scenario_path")
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    # LINE-BY-LINE: `output_dir`에 `args.output_dir or str(Path(config["paths"]["output_dir"]) / "test_data_selection_actual_range")` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = args.output_dir or str(Path(config["paths"]["output_dir"]) / "test_data_selection_actual_range")
    # LINE-BY-LINE: `result`에 `select_actual_start_range_test_data(` 결과를 저장합니다. 의미/사용: `result` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    result = select_actual_start_range_test_data(
        # LINE-BY-LINE: `select_actual_start_range_test_data(...)` 호출에 `scenario` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
        scenario,
        # LINE-BY-LINE: `output_dir`에 `output_dir` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
        output_dir=output_dir,
        # LINE-BY-LINE: `scenario_output_path`에 `args.output_scenario` 결과를 저장합니다. 의미/사용: `scenario_output_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        scenario_output_path=args.output_scenario,
        # LINE-BY-LINE: `start_value`에 `args.start` 결과를 저장합니다. 의미/사용: `start_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        start_value=args.start,
        # LINE-BY-LINE: `end_value`에 `args.end` 결과를 저장합니다. 의미/사용: `end_value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        end_value=args.end,
        # LINE-BY-LINE: `expected_count`에 `args.expected_count` 결과를 저장합니다. 의미/사용: `expected_count` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        expected_count=args.expected_count,
    )
    # LINE-BY-LINE: `summary`에 `result["summary"]` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = result["summary"]

    # LINE-BY-LINE: 콘솔에 `print("[actual range test data]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[actual range test data]")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_path: {scenario_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_path: {scenario_path}")
    # LINE-BY-LINE: 콘솔에 `print(f"- range_start: {summary['range_start']} ({summary['range_start_parse']})")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- range_start: {summary['range_start']} ({summary['range_start_parse']})")
    # LINE-BY-LINE: 콘솔에 `print(f"- range_end: {summary['range_end']} ({summary['range_end_parse']})")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- range_end: {summary['range_end']} ({summary['range_end_parse']})")
    # LINE-BY-LINE: 콘솔에 `print(f"- job_count: {summary['job_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- job_count: {summary['job_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- expected_count: {summary['expected_count']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- expected_count: {summary['expected_count']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- first_actual_start_datetime: {summary['first_actual_start_datetime']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- first_actual_start_datetime: {summary['first_actual_start_datetime']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- last_actual_start_datetime: {summary['last_actual_start_datetime']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- last_actual_start_datetime: {summary['last_actual_start_datetime']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scenario_output_path: {summary['scenario_output_path']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scenario_output_path: {summary['scenario_output_path']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- output_dir: {summary['output_dir']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- output_dir: {summary['output_dir']}")


# LINE-BY-LINE: `build_parser()` 함수를 정의합니다. 반환 타입: `argparse.ArgumentParser`. 사용: CLI 명령에서 사용자가 실행한 subcommand를 처리합니다.
def build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    # LINE-BY-LINE: `parser`에 `argparse.ArgumentParser(description="Cutting shop scheduling project")` 결과를 저장합니다. 의미/사용: `parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    parser = argparse.ArgumentParser(description="Cutting shop scheduling project")
    # LINE-BY-LINE: `common_parser`에 `argparse.ArgumentParser(add_help=False)` 결과를 저장합니다. 의미/사용: `common_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", default="config_np_100.yaml", help="YAML config path")
    # LINE-BY-LINE: `subparsers`에 `parser.add_subparsers(dest="command", required=True)` 결과를 저장합니다. 의미/사용: `subparsers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    subparsers = parser.add_subparsers(dest="command", required=True)

    # LINE-BY-LINE: `show_parser`에 `subparsers.add_parser("show-config", parents=[common_parser], help="Print active config summary")` 결과를 저장합니다. 의미/사용: `show_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    show_parser = subparsers.add_parser("show-config", parents=[common_parser], help="Print active config summary")
    # LINE-BY-LINE: `show_parser.set_defaults(func`에 `command_show_config)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    show_parser.set_defaults(func=command_show_config)

    # LINE-BY-LINE: `factory_summary_parser`에 `subparsers.add_parser("factory-summary", parents=[common_parser], help="Print expanded Bay/machin...` 결과를 저장합니다. 의미/사용: `factory_summary_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_summary_parser = subparsers.add_parser("factory-summary", parents=[common_parser], help="Print expanded Bay/machine/factory summary")
    # LINE-BY-LINE: `factory_summary_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    factory_summary_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `factory_summary_parser.set_defaults(func`에 `command_factory_summary)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_summary_parser.set_defaults(func=command_factory_summary)

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

    apply_phase1_parser = subparsers.add_parser(
        "apply-phase1-to-phase2",
        parents=[common_parser],
        help="Apply Phase 1 block-to-Bay result to a Phase 2 scenario",
    )
    apply_phase1_parser.add_argument(
        "--scenario-path",
        default=None,
        help="Optional source scenario override. Default follows config/data-source loader.",
    )
    apply_phase1_parser.add_argument(
        "--phase1-plan",
        required=True,
        help="Phase 1 plan JSON path, usually phase1_block_bay_plan.json",
    )
    apply_phase1_parser.add_argument(
        "--assignment-mode",
        default="cut_bay",
        choices=["cut_bay", "allowed_bay_ids"],
        help="How to write Phase 1 Bay into Phase 2 jobs",
    )
    apply_phase1_parser.add_argument(
        "--output-scenario",
        default="output/generated/phase2_from_phase1.yaml",
        help="Output Phase 2 scenario YAML path",
    )
    apply_phase1_parser.set_defaults(func=command_apply_phase1_to_phase2)

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

    # LINE-BY-LINE: `simulate_parser`에 `subparsers.add_parser("simulate", parents=[common_parser], help="Run one heuristic-based scheduli...` 결과를 저장합니다. 의미/사용: `simulate_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    simulate_parser = subparsers.add_parser("simulate", parents=[common_parser], help="Run one heuristic-based scheduling simulation")
    simulate_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `simulate_parser.add_argument("--heuristic", default` 여러 변수에 `None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    simulate_parser.add_argument("--heuristic", default=None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")
    # LINE-BY-LINE: `simulate_parser.add_argument("--action-mode", default` 여러 변수에 `None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    simulate_parser.add_argument("--action-mode", default=None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")
    simulate_parser.add_argument("--allow-single-slot-baseline", action="store_true", help="Allow job_machine_pair when batch capacity limits are active")
    # LINE-BY-LINE: `simulate_parser.set_defaults(func`에 `command_simulate)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    simulate_parser.set_defaults(func=command_simulate)

    # LINE-BY-LINE: `build_scenario_parser`에 `subparsers.add_parser("build-scenario", parents=[common_parser], help="Build NP 100 scenario from...` 결과를 저장합니다. 의미/사용: `build_scenario_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    build_scenario_parser = subparsers.add_parser("build-scenario", parents=[common_parser], help="Build NP 100 scenario from cutting Excel/CSV")
    build_scenario_parser.add_argument(
        "--input-path",
        default="input/절단03~04_NP물량_마스킹_WO_수정_260618.xlsx",
        help="Cutting Excel/CSV input path",
    )
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--sheet-name", default` 여러 변수에 `"Sheet", help="Excel sheet name")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--sheet-name", default="Sheet", help="Excel sheet name")
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--target-series", default` 여러 변수에 `"NP", help="Comma-separated target series")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--target-series", default="NP", help="Comma-separated target series")
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--sample-size", type` 여러 변수에 `int, default=100, help="Number of cleaned records to keep")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--sample-size", type=int, default=100, help="Number of cleaned records to keep")
    # LINE-BY-LINE: `build_scenario_parser.add_argument(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    build_scenario_parser.add_argument(
        # LINE-BY-LINE: 문자열 값 `"--process-time-source"`를 `add_argument(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
        "--process-time-source",
        # LINE-BY-LINE: `default`에 `"tact_time"` 결과를 저장합니다. 의미/사용: `default` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        default="tact_time",
        # LINE-BY-LINE: `choices`에 `["tact_time", "actual_duration", "estimated_actual_duration"]` 결과를 저장합니다. 의미/사용: `choices` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        choices=["tact_time", "actual_duration", "estimated_actual_duration"],
        # LINE-BY-LINE: `help`에 `"Processing time source for scenario jobs"` 결과를 저장합니다. 의미/사용: `help` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        help="Processing time source for scenario jobs",
    )
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--output-path", default` 여러 변수에 `"input/np_100_scenario.yaml", help="Generated scenario path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--output-path", default="output/generated/np_100_scenario.yaml", help="Generated scenario path")
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--csv-output", default` 여러 변수에 `"input/np_100_sample.csv", help="Cleaned sample CSV path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--csv-output", default="output/generated/np_100_sample.csv", help="Cleaned sample CSV path")
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--excluded-output", default` 여러 변수에 `"output/np_100_excluded_rows.csv", help="Excluded rows CSV path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--excluded-output", default="output/np_100_excluded_rows.csv", help="Excluded rows CSV path")
    # LINE-BY-LINE: `build_scenario_parser.set_defaults(func`에 `command_build_scenario)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    build_scenario_parser.set_defaults(func=command_build_scenario)

    # LINE-BY-LINE: `playback_parser`에 `subparsers.add_parser("playback", parents=[common_parser], help="Run heuristic and write factory ...` 결과를 저장합니다. 의미/사용: `playback_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    playback_parser = subparsers.add_parser("playback", parents=[common_parser], help="Run heuristic and write factory playback artifacts")
    # LINE-BY-LINE: `playback_parser.add_argument("--heuristic", default` 여러 변수에 `None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    playback_parser.add_argument("--heuristic", default=None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")
    # LINE-BY-LINE: `playback_parser.add_argument("--action-mode", default` 여러 변수에 `None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    playback_parser.add_argument("--action-mode", default=None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")
    playback_parser.add_argument("--allow-single-slot-baseline", action="store_true", help="Allow job_machine_pair when batch capacity limits are active")
    # LINE-BY-LINE: `playback_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    playback_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `playback_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Playback output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    playback_parser.add_argument("--output-dir", default=None, help="Playback output directory")
    # LINE-BY-LINE: `playback_parser.set_defaults(func`에 `command_playback)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    playback_parser.set_defaults(func=command_playback)

    # LINE-BY-LINE: `actual_replay_parser`에 `subparsers.add_parser("actual-replay", parents=[common_parser], help="Replay actual historical eq...` 결과를 저장합니다. 의미/사용: `actual_replay_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_replay_parser = subparsers.add_parser("actual-replay", parents=[common_parser], help="Replay actual historical equipment/Bay/start/end times")
    # LINE-BY-LINE: `actual_replay_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_replay_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `actual_replay_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Actual replay output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_replay_parser.add_argument("--output-dir", default=None, help="Actual replay output directory")
    # LINE-BY-LINE: `actual_replay_parser.set_defaults(func`에 `command_actual_replay)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_replay_parser.set_defaults(func=command_actual_replay)

    # LINE-BY-LINE: `factory_replay_parser`에 `subparsers.add_parser("factory-replay", parents=[common_parser], help="Alias for actual factory r...` 결과를 저장합니다. 의미/사용: `factory_replay_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_replay_parser = subparsers.add_parser("factory-replay", parents=[common_parser], help="Alias for actual factory replay from historical records")
    # LINE-BY-LINE: `factory_replay_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    factory_replay_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `factory_replay_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Factory replay output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    factory_replay_parser.add_argument("--output-dir", default=None, help="Factory replay output directory")
    # LINE-BY-LINE: `factory_replay_parser.set_defaults(func`에 `command_actual_replay)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    factory_replay_parser.set_defaults(func=command_actual_replay)

    gym_equivalence_parser = subparsers.add_parser(
        "gym-equivalence",
        parents=[common_parser],
        help="Validate DES action-mask wrapper against direct heuristic execution",
    )
    gym_equivalence_parser.add_argument(
        "--heuristic",
        default="batch_fill_spt",
        help="Heuristic to replay through wrapper indices",
    )
    gym_equivalence_parser.add_argument(
        "--max-actions",
        type=int,
        default=4096,
        help="Fixed Gym Discrete action space size; fails if candidates exceed this",
    )
    gym_equivalence_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for action_trace.csv and summary.json",
    )
    gym_equivalence_parser.set_defaults(func=command_gym_equivalence)

    # LINE-BY-LINE: `trace_parser`에 `subparsers.add_parser("trace", parents=[common_parser], help="Print step-by-step decision trace")` 결과를 저장합니다. 의미/사용: `trace_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trace_parser = subparsers.add_parser("trace", parents=[common_parser], help="Print step-by-step decision trace")
    # LINE-BY-LINE: `trace_parser.add_argument("--heuristic", default` 여러 변수에 `None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    trace_parser.add_argument("--heuristic", default=None, help="spt / priority / load_balance / batch_fill_spt / balanced_batch / balanced_batch_count")
    # LINE-BY-LINE: `trace_parser.add_argument("--action-mode", default` 여러 변수에 `None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    trace_parser.add_argument("--action-mode", default=None, help="Override action_space.mode: batch_open / job_machine_pair / rolling_capacity legacy")
    trace_parser.add_argument("--allow-single-slot-baseline", action="store_true", help="Allow job_machine_pair when batch capacity limits are active")
    # LINE-BY-LINE: `trace_parser.set_defaults(func`에 `command_trace)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trace_parser.set_defaults(func=command_trace)

    # LINE-BY-LINE: `tact_parser`에 `subparsers.add_parser("analyze-tact", parents=[common_parser], help="Run tact-time analysis")` 결과를 저장합니다. 의미/사용: `tact_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_parser = subparsers.add_parser("analyze-tact", parents=[common_parser], help="Run tact-time analysis")
    # LINE-BY-LINE: `tact_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    tact_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `tact_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Tact analysis output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    tact_parser.add_argument("--output-dir", default=None, help="Tact analysis output directory")
    # LINE-BY-LINE: `tact_parser.set_defaults(func`에 `command_analyze_tact)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_parser.set_defaults(func=command_analyze_tact)

    # LINE-BY-LINE: `tact_gap_parser`에 `subparsers.add_parser("analyze-tact-gap", parents=[common_parser], help="Analyze TACT_TIME vs act...` 결과를 저장합니다. 의미/사용: `tact_gap_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_gap_parser = subparsers.add_parser("analyze-tact-gap", parents=[common_parser], help="Analyze TACT_TIME vs actual elapsed gap")
    # LINE-BY-LINE: `tact_gap_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    tact_gap_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `tact_gap_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Tact gap analysis output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    tact_gap_parser.add_argument("--output-dir", default=None, help="Tact gap analysis output directory")
    # LINE-BY-LINE: `tact_gap_parser.add_argument("--excluded-csv", default` 여러 변수에 `None, help="Optional excluded rows CSV for raw NP count summary")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    tact_gap_parser.add_argument("--excluded-csv", default=None, help="Optional excluded rows CSV for raw NP count summary")
    # LINE-BY-LINE: `tact_gap_parser.set_defaults(func`에 `command_analyze_tact_gap)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tact_gap_parser.set_defaults(func=command_analyze_tact_gap)

    # LINE-BY-LINE: `generate_parser`에 `subparsers.add_parser("generate-scenario", parents=[common_parser], help="Generate training scena...` 결과를 저장합니다. 의미/사용: `generate_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    generate_parser = subparsers.add_parser("generate-scenario", parents=[common_parser], help="Generate training scenario")
    # LINE-BY-LINE: `generate_parser.add_argument("--duplicate-jobs", type` 여러 변수에 `int, default=2, help="How many copies of template jobs to make")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    generate_parser.add_argument("--duplicate-jobs", type=int, default=2, help="How many copies of template jobs to make")
    # LINE-BY-LINE: `generate_parser.add_argument("--output-path", default` 여러 변수에 `"output/generated_scenario.yaml", help="Generated yaml path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    generate_parser.add_argument("--output-path", default="output/generated_scenario.yaml", help="Generated yaml path")
    # LINE-BY-LINE: `generate_parser.set_defaults(func`에 `command_generate_scenario)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    generate_parser.set_defaults(func=command_generate_scenario)

    # LINE-BY-LINE: `select_test_parser`에 `subparsers.add_parser("select-test-data", parents=[common_parser], help="Select actual-day based ...` 결과를 저장합니다. 의미/사용: `select_test_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    select_test_parser = subparsers.add_parser("select-test-data", parents=[common_parser], help="Select actual-day based test scenarios")
    # LINE-BY-LINE: `select_test_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    select_test_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `select_test_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Distribution/report output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    select_test_parser.add_argument("--output-dir", default=None, help="Distribution/report output directory")
    # LINE-BY-LINE: `select_test_parser.add_argument("--scenario-output-dir", default` 여러 변수에 `"input/test_slices", help="Scenario slice output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    select_test_parser.add_argument("--scenario-output-dir", default="output/generated/test_slices", help="Scenario slice output directory")
    # LINE-BY-LINE: `select_test_parser.add_argument("--stratified-count", type` 여러 변수에 `int, default=100, help="Deterministic stratified sample size")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    select_test_parser.add_argument("--stratified-count", type=int, default=100, help="Deterministic stratified sample size")
    # LINE-BY-LINE: `select_test_parser.add_argument("--low-usable-min-count", type` 여러 변수에 `int, default=20, help="Minimum jobs for low-usable day slice")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    select_test_parser.add_argument("--low-usable-min-count", type=int, default=20, help="Minimum jobs for low-usable day slice")
    # LINE-BY-LINE: `select_test_parser.set_defaults(func`에 `command_select_test_data)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    select_test_parser.set_defaults(func=command_select_test_data)

    # LINE-BY-LINE: `actual_range_parser`에 `subparsers.add_parser("select-actual-range", parents=[common_parser], help="Select jobs by actual...` 결과를 저장합니다. 의미/사용: `actual_range_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_range_parser = subparsers.add_parser("select-actual-range", parents=[common_parser], help="Select jobs by actual start datetime range")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--scenario-path", default` 여러 변수에 `None, help="Override scenario path from config")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--scenario-path", default=None, help="Override scenario path from config")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--start", required` 여러 변수에 `True, help="Inclusive start: YYYYMMDD or YYYYMMDDHHMM")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--start", required=True, help="Inclusive start: YYYYMMDD or YYYYMMDDHHMM")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--end", required` 여러 변수에 `True, help="Inclusive end: YYYYMMDD or YYYYMMDDHHMM")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--end", required=True, help="Inclusive end: YYYYMMDD or YYYYMMDDHHMM")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--expected-count", type` 여러 변수에 `int, default=None, help="Fail if selected count differs")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--expected-count", type=int, default=None, help="Fail if selected count differs")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--output-dir", default` 여러 변수에 `None, help="Range selection report output directory")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--output-dir", default=None, help="Range selection report output directory")
    # LINE-BY-LINE: `actual_range_parser.add_argument("--output-scenario", default` 여러 변수에 `"input/test_slices/np_actual_start_range.yaml", help="Output scenario path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    actual_range_parser.add_argument("--output-scenario", default="output/generated/test_slices/np_actual_start_range.yaml", help="Output scenario path")
    # LINE-BY-LINE: `actual_range_parser.set_defaults(func`에 `command_select_actual_range)` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    actual_range_parser.set_defaults(func=command_select_actual_range)

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
