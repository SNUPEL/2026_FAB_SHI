"""프로젝트 공통 실행 진입점.

초보자 기준 실행 예시:
  python3 main.py show-config --config config.yaml
  python3 main.py build-scenario --config config_np_100.yaml
  python3 main.py simulate --config config.yaml
  python3 main.py playback --config config_np_100.yaml
  python3 main.py pygame-viewer --event-log output/share/html_viewer_package/clean_100/generated/balanced_batch/event_log.json --layout output/share/html_viewer_package/clean_100/generated/balanced_batch/factory_layout.json --schedule output/share/html_viewer_package/clean_100/generated/balanced_batch/job_schedule.csv --metrics output/share/html_viewer_package/clean_100/generated/balanced_batch/metrics.json
  python3 main.py trace --config config.yaml
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
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# LINE-BY-LINE: `Agent.heuristics` 모듈에서 `select_action_by_rule`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Agent.heuristics import select_action_by_rule
# LINE-BY-LINE: `Environment.environment` 모듈에서 `CuttingShopEnvironment`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.environment import CuttingShopEnvironment
from Environment.gym_wrapper import GYMNASIUM_AVAILABLE, run_hierarchical_trace_export, run_wrapper_equivalence
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
    train_phase2_batch_machine_self_labeling,
)
from Phase2.run_spec import build_phase2_run_spec, require_matching_phase2_run_spec
from Phase1.imitation import train_phase1_pointer_imitation
from Phase1.pair_self_labeling import run_phase1_pair_policy_rollout, train_phase1_pair_self_labeling
from Phase1.self_labeling import (
    PHASE1_SELF_LABEL_HEURISTIC_BANK,
    PPB_LCP6_HEURISTIC_BANK,
    _score_bay_loads,
    run_phase1_heuristic_candidate,
    train_phase1_pointer_self_labeling,
)
# LINE-BY-LINE: `Utils.config` 모듈에서 `load_config`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.config import load_config
# LINE-BY-LINE: `Utils.data.cutting_data_loader` 모듈에서 `load_and_clean_cutting_data, write_records_csv`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.cutting_data_loader import load_and_clean_cutting_data, write_records_csv
from Utils.data.cutting_start_date import (
    audit_cutting_start_dates,
    write_cutting_start_date_audit,
)
from Utils.data.multi_series_cutting_data import load_multi_series_cutting_data
# LINE-BY-LINE: `Utils.data.cutting_scenario_builder` 모듈에서 `build_scenario_from_cutting_records`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.cutting_scenario_builder import build_scenario_from_cutting_records
# LINE-BY-LINE: `Utils.data.factory_builder` 모듈에서 `build_factory_scenario_parts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.factory_builder import build_factory_scenario_parts
# LINE-BY-LINE: `Utils.data.io` 모듈에서 scenario loader를 가져옵니다. `load_scenario`는 명시 YAML용, `load_scenario_for_config`는 config 기반 원본 데이터 로딩용입니다.
from Utils.data.io import load_scenario, load_scenario_for_config
from Utils.learning.learning_data_builder import build_learning_data_package
from Utils.learning.phase_agent_checkpoints import (
    load_phase1_feedback_contract,
    load_phase1_pair_pointer_checkpoint,
    load_phase2_checkpoint_run_spec,
    load_phase2_set_pointer_checkpoint,
)
from Utils.phase1.phase1_block_data_generator import (
    load_phase1_actual_blocks,
    write_phase1_block_generation_package,
)
from Utils.phase1.phase1_episode_dataset import (
    build_phase1_actual_workday_jobs,
    build_phase1_candidate_workbook_jobs,
    build_phase1_episode_jobs,
    jobs_from_phase1_episode_blocks,
    write_phase1_episode_dataset,
)
from Utils.phase1.phase1_bay_balancer import (
    CANONICAL_PHASE1_HEURISTIC,
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    MULTI_OBJECTIVE_PHASE1_HEURISTIC,
    PRIORITY_GREEDY_PHASE1_HEURISTIC,
    PRIORITY_SWEEP_PHASE1_HEURISTIC,
    apply_phase1_plan_to_scenario,
    build_phase1_bay_plan,
    write_phase1_bay_plan,
)
from Utils.phase1.multi_series_planner import (
    build_multi_series_phase1_daily_plans,
    write_multi_series_phase1_daily_plans,
)
from Utils.learning.phase1_phase2_communication import (
    apply_phase1_messages_to_scenario,
    load_communication_jsonl,
    write_phase1_phase2_communication_package,
)
from Utils.data.phase2_candidate_workbook import load_phase2_candidate_workbook_problems
from Utils.phase1.phase1_mdp import write_phase1_mdp_trace_package
from Utils.data.report_formula_data_generator import (
    build_report_formula_episode_jobs,
    scenario_jobs_from_report_formula_jobs,
)
# LINE-BY-LINE: `Utils.reporting.playback_builder` 모듈에서 `write_actual_replay_artifacts, write_playback_artifacts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.playback_builder import write_actual_replay_artifacts, write_playback_artifacts
# LINE-BY-LINE: `Utils.data.scenario_generator` 모듈에서 `generate_scenario_from_template, save_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.scenario_generator import generate_scenario_from_template, save_scenario
# 회사 송부용 정적 HTML 패키지와 휴리스틱 baseline 비교표를 생성하는 helper입니다.
from Utils.reporting.share_report_builder import build_html_package, parse_csv_argument
# Pygame 로컬 공장 playback viewer입니다. `--dry-run`으로 GUI 없이 입력 검증도 가능합니다.
from Utils.reporting.pygame_factory_viewer import run_pygame_comparison_from_paths, run_pygame_viewer_from_paths
# LINE-BY-LINE: `Utils.reporting.tact_gap_analysis` 모듈에서 `build_tact_gap_analysis`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.tact_gap_analysis import build_tact_gap_analysis
# LINE-BY-LINE: `Utils.reporting.tact_time` 모듈에서 `build_tact_time_analysis_from_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.reporting.tact_time import build_tact_time_analysis_from_scenario
# LINE-BY-LINE: `Utils.data.test_data_selection` 모듈에서 `select_actual_day_test_data, select_actual_start_range_test_data`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.data.test_data_selection import select_actual_day_test_data, select_actual_start_range_test_data


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


def command_phase1(args: argparse.Namespace) -> None:
    """Run Phase 1 only: block-level cutting Bay assignment."""

    print("[phase1]")
    print(f"- config: {args.config}")
    print(f"- mode: {args.mode}")
    print(f"- algorithm: {args.algorithm}")
    print(f"- requested_bay_ids: {args.bay_ids}")
    if args.mode != "heuristic":
        print(
            "[ERROR][main.command_phase1] "
            f"cause=phase1_mode_not_implemented mode={args.mode} "
            "implemented_modes=['heuristic']"
        )
        raise RuntimeError(f"Phase 1 mode is not implemented yet: {args.mode}")

    env = build_environment(args.config)
    bay_ids = _phase1_bay_ids_from_env(env, args.bay_ids)
    bay_capacity_weights = _phase1_bay_capacity_weights_from_env(env, bay_ids)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("output") / f"phase1_{Path(args.config).stem}_{args.algorithm}")

    plan = build_phase1_bay_plan(
        jobs=env.jobs,
        bay_ids=bay_ids,
        algorithm=args.algorithm,
        bay_capacity_weights=bay_capacity_weights,
    )
    paths = write_phase1_bay_plan(plan, output_dir)
    summary = plan["summary"]

    print(f"- resolved_bay_ids: {bay_ids}")
    print(f"- bay_capacity_weights: {bay_capacity_weights}")
    print(f"- output_dir: {output_dir}")
    print(f"- job_count: {summary['job_count']}")
    print(f"- block_count: {summary['block_count']}")
    print(f"- assigned_block_count: {summary['assigned_block_count']}")
    print(f"- steel_quantity_total: {summary['steel_quantity_total']}")
    print(f"- steel_quantity_gap: {summary['steel_quantity_gap']}")
    if "cut_length_gap" in summary:
        print(f"- cut_length_total: {summary['cut_length_total']}")
        print(f"- cut_length_gap: {summary['cut_length_gap']}")
    if "bevel_quantity_gap" in summary:
        print(f"- bevel_quantity_total: {summary['bevel_quantity_total']}")
        print(f"- bevel_quantity_gap: {summary['bevel_quantity_gap']}")
    if "long_cut_bay24_count" in summary:
        print(f"- long_cut_bay24_count: {summary['long_cut_bay24_count']}")
    print(f"- wo_count_total: {summary['wo_count_total']}")
    print(f"- wo_count_gap: {summary['wo_count_gap']}")
    print(f"- block_count_gap: {summary['block_count_gap']}")
    print(f"- plan_json: {paths['json']}")
    print(f"- assignments_csv: {paths['assignments_csv']}")
    print(f"- bay_loads_csv: {paths['bay_loads_csv']}")


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


def command_phase1_mdp_trace(args: argparse.Namespace) -> None:
    """Build Phase 1 SELECT_BLOCK -> SELECT_BAY self-label trace."""

    print("[phase1-mdp-trace-cli]")
    print(f"- config: {args.config}")
    print(f"- algorithm: {args.algorithm}")
    print(f"- requested_bay_ids: {args.bay_ids}")
    env = build_environment(args.config)
    bay_ids = _phase1_bay_ids_from_env(env, args.bay_ids)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("output") / f"phase1_mdp_trace_{Path(args.config).stem}_{args.algorithm}")

    paths = write_phase1_mdp_trace_package(
        jobs=env.jobs,
        bay_ids=bay_ids,
        algorithm=args.algorithm,
        output_dir=output_dir,
    )

    print(f"- resolved_bay_ids: {bay_ids}")
    print(f"- output_dir: {output_dir}")
    print(f"- trace_csv: {paths['trace_csv']}")
    print(f"- action_table_jsonl: {paths['action_table_jsonl']}")
    print(f"- manifest_json: {paths['manifest_json']}")
    print(f"- plan_json: {paths['plan_json']}")


def command_phase1_train_imitation(args: argparse.Namespace) -> None:
    """Train Phase 1 pointer policy from a self-label action table."""

    print("[phase1-train-imitation]")
    print(f"- action_table: {args.action_table}")
    print(f"- eval_action_table: {args.eval_action_table}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- epochs: {args.epochs}")
    print(f"- lr: {args.lr}")
    print(f"- hidden_dim: {args.hidden_dim}")
    summary = train_phase1_pointer_imitation(
        action_table_path=args.action_table,
        output_dir=args.output_dir,
        eval_action_table_path=args.eval_action_table,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    print(f"- checkpoint_path: {summary['checkpoint_path']}")
    print(f"- metrics_csv: {summary['metrics_csv']}")
    print(f"- candidate_summary_csv: {summary['candidate_summary_csv']}")
    print(f"- best_assignment_csv: {summary['best_assignment_csv']}")
    print(f"- best_machine_load_csv: {summary['best_machine_load_csv']}")
    print(f"- summary_json: {summary['summary_json']}")
    print(f"- final_loss: {summary['final_loss']}")
    print(f"- final_accuracy: {summary['final_accuracy']}")
    if "eval_accuracy" in summary:
        print(f"- eval_loss: {summary['eval_loss']}")
        print(f"- eval_accuracy: {summary['eval_accuracy']}")


def command_phase1_train_self_labeling(args: argparse.Namespace) -> None:
    """Train Phase 1 pointer policy with best-of-K self-labeling."""

    print("[phase1-train-self-labeling-cli]")
    print(f"- config: {args.config}")
    print(f"- episode_mode: {args.episode_mode}")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- gyel: {args.gyel}")
    print(f"- min_blocks: {args.min_blocks}")
    print(f"- max_blocks: {args.max_blocks}")
    print(f"- noise_ratio: {args.noise_ratio}")
    print(f"- requested_bay_ids: {args.bay_ids}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- episodes: {args.episodes}")
    print(f"- rollout_samples: {args.rollout_samples}")
    print(f"- heuristic_algorithms: {args.heuristic_algorithms}")
    print(f"- score_mode: {args.score_mode}")
    env = build_environment(args.config)
    bay_ids = _phase1_bay_ids_from_env(env, args.bay_ids)
    heuristic_algorithms = [
        item.strip()
        for item in str(args.heuristic_algorithms).split(",")
        if item.strip()
    ]
    if str(args.heuristic_algorithms).strip().lower() in {"all", "all8"}:
        heuristic_algorithms = list(PHASE1_SELF_LABEL_HEURISTIC_BANK)
    if str(args.heuristic_algorithms).strip().lower() in {"business6", "all6", "ppb_lcp6"}:
        heuristic_algorithms = list(PPB_LCP6_HEURISTIC_BANK)
    fixed_jobs = None
    episode_jobs = None
    episode_metadata = None
    if args.episode_mode == "fixed":
        fixed_jobs = env.jobs
    else:
        actual_blocks = load_phase1_actual_blocks(args.block_xlsx, gyel=args.gyel)
        episode_specs = build_phase1_episode_jobs(
            actual_blocks=actual_blocks,
            episode_count=args.episodes,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed,
            noise_ratio=args.noise_ratio,
        )
        episode_jobs = [spec["jobs"] for spec in episode_specs]
        episode_metadata = [
            {
                "episode_id": spec["episode_id"],
                "problem_id": spec["problem_id"],
                "block_count": spec["block_count"],
                "seed": spec["seed"],
            }
            for spec in episode_specs
        ]
    summary = train_phase1_pointer_self_labeling(
        jobs=fixed_jobs,
        episode_jobs=episode_jobs,
        episode_metadata=episode_metadata,
        bay_ids=bay_ids,
        output_dir=args.output_dir,
        episodes=args.episodes,
        rollout_samples=args.rollout_samples,
        heuristic_algorithms=heuristic_algorithms,
        score_mode=args.score_mode,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"- resolved_bay_ids: {bay_ids}")
    print(f"- checkpoint_path: {summary['checkpoint_path']}")
    print(f"- metrics_csv: {summary['metrics_csv']}")
    print(f"- candidate_summary_csv: {summary['candidate_summary_csv']}")
    print(f"- best_action_table_jsonl: {summary['best_action_table_jsonl']}")
    print(f"- learning_data_manifest_json: {summary['learning_data_manifest_json']}")
    print(f"- loss_curve_png: {summary['loss_curve_png']}")
    print(f"- best_source_counts_png: {summary['best_source_counts_png']}")
    print(f"- best_score0_curve_png: {summary['best_score0_curve_png']}")
    print(f"- summary_json: {summary['summary_json']}")
    print(f"- best_source_counts: {summary['best_source_counts']}")


def command_phase1_train_pair_self_labeling(args: argparse.Namespace) -> None:
    """Train Phase 1 direct pair-action policy with best-of-K self-labeling."""

    print("[phase1-train-pair-self-labeling-cli]")
    print(f"- config: {args.config}")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- gyel: {args.gyel}")
    print(f"- min_blocks: {args.min_blocks}")
    print(f"- max_blocks: {args.max_blocks}")
    print(f"- noise_ratio: {args.noise_ratio}")
    print(f"- hard_case_ratio: {args.hard_case_ratio}")
    print(f"- hard_case_mode: {args.hard_case_mode}")
    print(f"- hard_case_target_corr: {args.hard_case_target_corr}")
    print(f"- hard_case_max_attempts: {args.hard_case_max_attempts}")
    print(f"- requested_bay_ids: {args.bay_ids}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- episodes: {args.episodes}")
    print(f"- rollout_samples: {args.rollout_samples}")
    validation_rollout_samples = (
        args.rollout_samples
        if args.validation_rollout_samples is None
        else args.validation_rollout_samples
    )
    print(f"- rollout_samples_validation: {validation_rollout_samples}")
    validation_hard_case_ratio = (
        args.hard_case_ratio
        if args.validation_hard_case_ratio is None
        else args.validation_hard_case_ratio
    )
    print(f"- validation_hard_case_ratio: {validation_hard_case_ratio}")
    print(f"- heuristic_algorithms: {args.heuristic_algorithms}")
    print(f"- score_mode: {args.score_mode}")
    print(f"- actual_validation_candidate_xlsx: {args.actual_validation_candidate_xlsx}")
    print(f"- actual_validation_workdays: {args.actual_validation_workdays}")
    print(f"- resume_checkpoint: {args.resume_checkpoint}")
    print(f"- device: {args.device}")
    print(f"- enable_phase2_feedback_score: {args.enable_phase2_feedback_score}")
    print(f"- phase2_feedback_checkpoint: {args.phase2_feedback_checkpoint}")
    env = build_environment(args.config)
    config = load_config(args.config)
    bay_ids = _phase1_bay_ids_from_env(env, args.bay_ids)
    bay_capacity_weights = _phase1_bay_capacity_weights_from_env(env, bay_ids)
    phase2_feedback_checkpoint = _optional_non_empty_cli_value(
        args.phase2_feedback_checkpoint,
        "phase2_feedback_checkpoint",
    )
    if args.enable_phase2_feedback_score != (phase2_feedback_checkpoint is not None):
        print(
            "[ERROR][main.command_phase1_train_pair_self_labeling] "
            f"cause=feedback_flag_checkpoint_mismatch enabled={args.enable_phase2_feedback_score} "
            f"checkpoint={phase2_feedback_checkpoint or ''}"
        )
        raise RuntimeError(
            "--enable-phase2-feedback-score and --phase2-feedback-checkpoint must be used together"
        )
    heuristic_algorithms = [
        item.strip()
        for item in str(args.heuristic_algorithms).split(",")
        if item.strip()
    ]
    if str(args.heuristic_algorithms).strip().lower() in {"all", "all8"}:
        heuristic_algorithms = list(PHASE1_SELF_LABEL_HEURISTIC_BANK)
    if str(args.heuristic_algorithms).strip().lower() in {"business6", "all6", "ppb_lcp6"}:
        heuristic_algorithms = list(PPB_LCP6_HEURISTIC_BANK)
    actual_blocks = (
        None
        if args.enable_phase2_feedback_score
        else load_phase1_actual_blocks(args.block_xlsx, gyel=args.gyel)
    )

    def episode_factory(episode: int):
        episode_id = f"EP{episode:05d}"
        if args.enable_phase2_feedback_score:
            spec = build_report_formula_episode_jobs(
                episode_count=1,
                min_blocks=args.min_blocks,
                max_blocks=args.max_blocks,
                seed=args.seed + episode * 1_000_003,
                gyel=args.gyel,
            )[0]
            return {
                "jobs": spec["jobs"],
                "metadata": {
                    "episode_id": episode_id,
                    "problem_id": episode_id,
                    "block_count": spec["block_count"],
                    "job_count": spec["job_count"],
                    "seed": spec["seed"],
                    "case_type": "report_formula_wo",
                    "hard_case_mode": "none",
                    "hard_case_ratio": 0.0,
                },
            }
        if actual_blocks is None:
            print("[ERROR][main.episode_factory] cause=missing_phase1_actual_blocks")
            raise RuntimeError("Phase 1 block bootstrap source is missing")
        specs = build_phase1_episode_jobs(
            actual_blocks=actual_blocks,
            episode_count=1,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed + episode - 1,
            noise_ratio=args.noise_ratio,
            hard_case_ratio=args.hard_case_ratio,
            hard_case_mode=args.hard_case_mode,
            hard_case_target_corr=args.hard_case_target_corr,
            hard_case_max_attempts=args.hard_case_max_attempts,
            verbose=False,
        )
        spec = specs[0]
        return {
            "jobs": jobs_from_phase1_episode_blocks(spec["blocks"], episode_id),
            "metadata": {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "block_count": spec["block_count"],
                "seed": spec["seed"],
                "case_type": spec.get("case_type", "normal"),
                "hard_case_mode": spec.get("hard_case_mode", "none"),
                "hard_case_ratio": spec.get("hard_case_ratio", args.hard_case_ratio),
                "hard_case_corr_steel_cut_before": spec.get("hard_case_corr_steel_cut_before"),
                "hard_case_corr_steel_cut_after": spec.get("hard_case_corr_steel_cut_after"),
            },
        }

    def synthetic_validation_episode_factory(validation_episode: int):
        episode_id = f"VAL{validation_episode:05d}"
        if args.enable_phase2_feedback_score:
            spec = build_report_formula_episode_jobs(
                episode_count=1,
                min_blocks=args.min_blocks,
                max_blocks=args.max_blocks,
                seed=args.seed + 10_000_000 + validation_episode * 1_000_003,
                gyel=args.gyel,
            )[0]
            return {
                "jobs": spec["jobs"],
                "metadata": {
                    "episode_id": episode_id,
                    "problem_id": episode_id,
                    "block_count": spec["block_count"],
                    "job_count": spec["job_count"],
                    "seed": spec["seed"],
                    "case_type": "report_formula_wo",
                    "hard_case_mode": "none",
                    "hard_case_ratio": 0.0,
                    "validation_source": "synthetic",
                    "evaluation_input_type": "report_formula_wo",
                },
            }
        if actual_blocks is None:
            print("[ERROR][main.synthetic_validation_episode_factory] cause=missing_phase1_actual_blocks")
            raise RuntimeError("Phase 1 validation block bootstrap source is missing")
        specs = build_phase1_episode_jobs(
            actual_blocks=actual_blocks,
            episode_count=1,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            seed=args.seed + 10_000_000 + validation_episode - 1,
            noise_ratio=args.noise_ratio,
            hard_case_ratio=validation_hard_case_ratio,
            hard_case_mode=args.hard_case_mode,
            hard_case_target_corr=args.hard_case_target_corr,
            hard_case_max_attempts=args.hard_case_max_attempts,
            verbose=False,
        )
        spec = specs[0]
        return {
            "jobs": jobs_from_phase1_episode_blocks(spec["blocks"], episode_id),
            "metadata": {
                "episode_id": episode_id,
                "problem_id": episode_id,
                "block_count": spec["block_count"],
                "seed": spec["seed"],
                "case_type": spec.get("case_type", "normal"),
                "hard_case_mode": spec.get("hard_case_mode", "none"),
                "hard_case_ratio": spec.get("hard_case_ratio", validation_hard_case_ratio),
                "hard_case_corr_steel_cut_before": spec.get("hard_case_corr_steel_cut_before"),
                "hard_case_corr_steel_cut_after": spec.get("hard_case_corr_steel_cut_after"),
                "validation_source": "synthetic",
                "evaluation_input_type": "synthetic_generated",
            },
        }

    actual_validation_workdays = [
        item.strip()
        for item in str(args.actual_validation_workdays).split(",")
        if item.strip()
    ]
    actual_validation_payloads = []
    if actual_validation_workdays:
        if args.enable_phase2_feedback_score:
            source_data_path = config.get("paths", {}).get("source_data_path")
            if not source_data_path:
                print(
                    "[ERROR][main.command_phase1_train_pair_self_labeling] "
                    "cause=missing_source_data_path_for_actual_feedback_validation"
                )
                raise RuntimeError("Phase 2 feedback actual validation requires paths.source_data_path")
            phase2_actual_payloads = load_phase2_candidate_workbook_problems(
                wo_path=source_data_path,
                candidate_path=args.actual_validation_candidate_xlsx,
                workdays=actual_validation_workdays,
                bay_ids=bay_ids,
                gyel=args.gyel,
                factory_config=config.get("factory"),
            )
            actual_validation_payloads = [
                {
                    "jobs": _scenario_jobs_by_id(payload["scenario"]),
                    "metadata": {
                        **dict(payload.get("phase1_metadata", {})),
                        "problem_id": payload["problem_id"],
                        "episode_id": payload["problem_id"],
                        "block_count": payload["candidate_block_count"],
                        "job_count": payload["wo_count"],
                        "validation_source": "actual_8days",
                        "evaluation_input_type": "candidate_workbook_wo_expanded",
                    },
                }
                for payload in phase2_actual_payloads
            ]
        else:
            actual_validation_payloads = build_phase1_candidate_workbook_jobs(
                candidate_path=args.actual_validation_candidate_xlsx,
                workdays=actual_validation_workdays,
                bay_ids=bay_ids,
            )
    total_validation_episodes = args.validation_episodes + len(actual_validation_payloads)

    def validation_episode_factory(validation_episode: int):
        if validation_episode <= args.validation_episodes:
            return synthetic_validation_episode_factory(validation_episode)
        actual_index = validation_episode - args.validation_episodes - 1
        payload = actual_validation_payloads[actual_index]
        return {"jobs": payload["jobs"], "metadata": payload["metadata"]}

    phase2_feedback_scorer = None
    phase2_feedback_contract = None
    if args.enable_phase2_feedback_score:
        enabled_machines = {
            machine_id: machine
            for machine_id, machine in env.machines.items()
            if machine.enabled and str(machine.bay_id) in set(bay_ids)
        }
        if not enabled_machines:
            print("[ERROR][main.command_phase1_train_pair_self_labeling] cause=no_enabled_machines_for_phase2_feedback")
            raise RuntimeError("Phase 2 feedback scoring requires at least one enabled machine")
        phase2_constraint_profile = load_phase_constraint_profile(config, "phase2")
        phase2_feedback_model = load_phase2_set_pointer_checkpoint(
            phase2_feedback_checkpoint,
            context="phase1_feedback",
        )
        phase2_feedback_run_spec = load_phase2_checkpoint_run_spec(
            phase2_feedback_checkpoint,
            context="phase1_feedback",
        )
        if not bool(phase2_feedback_run_spec["phase1_long_cut_hard_mask"]):
            print(
                "[ERROR][main.command_phase1_train_pair_self_labeling] "
                "cause=phase1_long_cut_mask_mismatch training=True checkpoint=False"
            )
            raise RuntimeError(
                "Phase 1 feedback training uses the long-cut hard mask, but the Phase 2 RunSpec does not"
            )
        phase2_feedback_scorer = build_frozen_phase2_schedule_feedback_scorer(
            model=phase2_feedback_model,
            machines=enabled_machines,
            run_spec=phase2_feedback_run_spec,
            constraint_profile=phase2_constraint_profile,
        )
        phase2_feedback_contract = build_phase2_feedback_contract(
            phase2_feedback_checkpoint,
            phase2_feedback_run_spec,
        )

    summary = train_phase1_pair_self_labeling(
        episode_jobs=None,
        episode_metadata=None,
        bay_ids=bay_ids,
        output_dir=args.output_dir,
        episodes=args.episodes,
        rollout_samples=args.rollout_samples,
        heuristic_algorithms=heuristic_algorithms,
        score_mode=args.score_mode,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
        seed=args.seed,
        episode_factory=episode_factory,
        checkpoint_every=args.checkpoint_every,
        validation_every=args.validation_every,
        validation_episodes=total_validation_episodes,
        validation_episode_factory=(
            validation_episode_factory if total_validation_episodes > 0 else None
        ),
        validation_rollout_samples=validation_rollout_samples,
        resume_checkpoint=args.resume_checkpoint,
        phase2_feedback_scorer=phase2_feedback_scorer,
        phase2_feedback_contract=phase2_feedback_contract,
        bay_capacity_weights=bay_capacity_weights,
        device=args.device,
    )
    print(f"- resolved_bay_ids: {bay_ids}")
    print(f"- bay_capacity_weights: {bay_capacity_weights}")
    print(f"- checkpoint_path: {summary['checkpoint_path']}")
    print(f"- best_checkpoint_path: {summary['best_checkpoint_path']}")
    print(f"- metrics_csv: {summary['metrics_csv']}")
    print(f"- candidate_summary_csv: {summary['candidate_summary_csv']}")
    print(f"- best_action_table_jsonl: {summary['best_action_table_jsonl']}")
    print(f"- validation_summary_csv: {summary['validation_summary_csv']}")
    print(f"- validation_candidate_summary_csv: {summary.get('validation_candidate_summary_csv', '')}")
    print(f"- validation_steel_gap_png: {summary.get('validation_steel_gap_png', '')}")
    print(f"- validation_cut_gap_png: {summary.get('validation_cut_gap_png', '')}")
    print(f"- validation_bevel_gap_png: {summary.get('validation_bevel_gap_png', '')}")
    print(f"- validation_best_source_counts_png: {summary.get('validation_best_source_counts_png', '')}")
    print(f"- validation_agent_rank_png: {summary.get('validation_agent_rank_png', '')}")
    print(f"- actual_validation_problem_count: {len(actual_validation_payloads)}")
    print(f"- summary_json: {summary['summary_json']}")
    print(f"- best_source_counts: {summary['best_source_counts']}")


def command_phase1_build_episode_dataset(args: argparse.Namespace) -> None:
    """Build variable-size Phase 1 train/test self-label episodes."""

    print("[phase1-build-episode-dataset]")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- gyel: {args.gyel}")
    print(f"- episode_count: {args.episode_count}")
    print(f"- min_blocks: {args.min_blocks}")
    print(f"- max_blocks: {args.max_blocks}")
    print(f"- train_ratio: {args.train_ratio}")
    print(f"- bay_ids: {args.bay_ids}")
    print(f"- algorithm: {args.algorithm}")
    print(f"- seed: {args.seed}")
    print(f"- noise_ratio: {args.noise_ratio}")
    print(f"- output_dir: {args.output_dir}")
    actual_blocks = load_phase1_actual_blocks(args.block_xlsx, gyel=args.gyel)
    bay_ids = [bay_id.strip() for bay_id in args.bay_ids.split(",") if bay_id.strip()]
    paths = write_phase1_episode_dataset(
        actual_blocks=actual_blocks,
        output_dir=args.output_dir,
        episode_count=args.episode_count,
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        train_ratio=args.train_ratio,
        bay_ids=bay_ids,
        algorithm=args.algorithm,
        seed=args.seed,
        noise_ratio=args.noise_ratio,
    )
    print(f"- train_action_table_jsonl: {paths['train_action_table_jsonl']}")
    print(f"- test_action_table_jsonl: {paths['test_action_table_jsonl']}")
    print(f"- manifest_json: {paths['manifest_json']}")
    print(f"- episode_summary_csv: {paths['episode_summary_csv']}")


def command_phase2_train_batch_machine_self_labeling(args: argparse.Namespace) -> None:
    """Train merged Phase 2 policy that selects W/O batch and machine together."""

    print("[phase2-train-batch-machine-self-labeling-cli]")
    print(f"- config: {args.config}")
    print(f"- scenario_path_override: {args.scenario_path}")
    print(f"- phase1_heuristic: {args.phase1_heuristic}")
    print(f"- phase1_checkpoint: {args.phase1_checkpoint}")
    print(f"- phase1_samples: {args.phase1_samples}")
    print(f"- phase1_temperature: {args.phase1_temperature}")
    print(f"- phase1_bay_ids: {args.phase1_bay_ids}")
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
    print(f"- synthetic_source: {args.synthetic_source}")
    print(f"- synthetic_block_range: {args.min_blocks}..{args.max_blocks}")

    env = build_environment(args.config, scenario_path=args.scenario_path)
    phase2_constraint_profile = load_phase_constraint_profile(load_config(args.config), "phase2")
    enabled_machines = {
        machine_id: machine
        for machine_id, machine in env.machines.items()
        if machine.enabled
    }
    if not enabled_machines:
        print("[ERROR][main.command_phase2_train_batch_machine_self_labeling] cause=no_enabled_machines")
        raise RuntimeError("Phase 2 training requires at least one enabled machine")
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
    phase1_bay_ids = _resolve_phase1_training_bay_ids(args.phase1_bay_ids, enabled_machines)
    training_machines = _filter_machines_by_bay_ids(enabled_machines, phase1_bay_ids)
    phase1_bay_capacity_weights = _phase1_bay_capacity_weights_from_env(env, phase1_bay_ids)
    phase1_assignment_builder = None
    if phase1_checkpoint is not None:
        phase1_assignment_builder = _phase1_agent_assignment_builder(
            checkpoint=phase1_checkpoint,
            bay_ids=phase1_bay_ids,
            sample_count=args.phase1_samples,
            temperature=args.phase1_temperature,
            seed=args.seed,
            score_mode=args.phase1_score_mode,
            long_cut_hard_mask=not args.phase1_no_long_cut_hard_mask,
            bay_capacity_weights=phase1_bay_capacity_weights,
        )
    print(f"- resolved_phase1_mode: {'checkpoint' if phase1_checkpoint is not None else 'heuristic'}")
    print(f"- resolved_phase1_heuristic: {phase1_heuristic or ''}")
    print(f"- resolved_phase1_bay_ids: {phase1_bay_ids}")
    print(f"- phase1_bay_capacity_weights: {phase1_bay_capacity_weights}")
    print(
        "- phase2_hard_constraints: "
        + ",".join(name for name, enabled in phase2_constraint_profile.hard_enabled.items() if enabled)
    )
    training_jobs = env.jobs
    episode_jobs = None
    episode_job_factory = None
    validation_episode_jobs = None
    validation_episode_job_factory = None
    if args.synthetic_source == "report_formula":
        def episode_job_factory(episode: int):
            spec = build_report_formula_episode_jobs(
                episode_count=1,
                min_blocks=args.min_blocks,
                max_blocks=args.max_blocks,
                seed=args.seed + episode * 1_000_003,
                gyel=args.gyel,
            )[0]
            return spec["jobs"]

        if args.validation_episodes > 0:
            def validation_episode_job_factory(validation_episode: int):
                spec = build_report_formula_episode_jobs(
                    episode_count=1,
                    min_blocks=args.min_blocks,
                    max_blocks=args.max_blocks,
                    seed=args.seed + 10_000_000 + validation_episode * 1_000_003,
                    gyel=args.gyel,
                )[0]
                return spec["jobs"]
    elif args.synthetic_source != "config":
        print(
            "[ERROR][main.command_phase2_train_batch_machine_self_labeling] "
            f"cause=unknown_synthetic_source value={args.synthetic_source}"
        )
        raise RuntimeError(f"unknown synthetic source: {args.synthetic_source}")

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
        episode_jobs=episode_jobs,
        episode_job_factory=episode_job_factory,
        validation_episode_jobs=validation_episode_jobs,
        validation_episode_job_factory=validation_episode_job_factory,
        phase1_heuristic=phase1_heuristic,
        phase1_bay_ids=phase1_bay_ids,
        phase1_assignment_builder=phase1_assignment_builder,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        phase1_long_cut_hard_mask=not args.phase1_no_long_cut_hard_mask,
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
    """Run merged Phase 2 batch-machine workflow, then export CSVs."""

    print("[phase2-run-full-workflow-cli]")
    print(f"- config: {args.config}")
    print(f"- scenario_path_override: {args.scenario_path}")
    print(f"- phase1_plan: {args.phase1_plan}")
    print(f"- phase1_checkpoint: {args.phase1_checkpoint}")
    print(f"- phase1_heuristic: {args.phase1_heuristic}")
    print(f"- phase1_bay_ids: {args.phase1_bay_ids}")
    print(f"- phase1_samples: {args.phase1_samples}")
    print(f"- assignment_mode: {args.assignment_mode}")
    print(f"- batch_machine_heuristic: {args.batch_machine_heuristic}")
    print(f"- batch_machine_checkpoint: {args.batch_machine_checkpoint}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- synthetic_source: {args.synthetic_source}")
    print(f"- synthetic_blocks: {args.synthetic_blocks}")

    config = load_config(args.config)
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    if args.synthetic_source == "report_formula":
        scenario = _replace_scenario_jobs_with_report_formula(
            scenario=scenario,
            block_count=args.synthetic_blocks,
            seed=args.seed,
            gyel=args.gyel,
        )
    elif args.synthetic_source != "config":
        print(
            "[ERROR][main.command_phase2_run_full_workflow] "
            f"cause=unknown_synthetic_source value={args.synthetic_source}"
        )
        raise RuntimeError(f"unknown synthetic source: {args.synthetic_source}")
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
    requested_bay_ids = (
        parse_csv_argument(args.phase1_bay_ids, default=())
        if hasattr(args, "phase1_bay_ids")
        else tuple(checkpoint_run_spec["phase1_bay_capacity_weights"])
        if checkpoint_run_spec is not None
        else ("22", "23", "24")
    )
    if not requested_bay_ids:
        print("[ERROR][main.command_phase2_run_full_workflow] cause=no_phase1_bay_ids")
        raise RuntimeError("full-flow requires Phase 1 Bay IDs")
    if checkpoint_run_spec is not None and set(requested_bay_ids) != set(
        checkpoint_run_spec["phase1_bay_capacity_weights"]
    ):
        print(
            "[ERROR][main.command_phase2_run_full_workflow] "
            f"cause=phase1_bay_id_mismatch checkpoint={sorted(checkpoint_run_spec['phase1_bay_capacity_weights'])} "
            f"requested={sorted(requested_bay_ids)}"
        )
        raise RuntimeError("Phase 1 Bay IDs differ from the Phase 2 checkpoint RunSpec")
    phase1_bay_capacity_weights = _phase1_bay_capacity_weights_from_scenario(
        scenario,
        requested_bay_ids,
    )
    phase1_long_cut_hard_mask = (
        bool(checkpoint_run_spec["phase1_long_cut_hard_mask"])
        if checkpoint_run_spec is not None and not args.phase1_no_long_cut_hard_mask
        else not args.phase1_no_long_cut_hard_mask
    )
    effective_run_spec = _resolve_phase2_full_flow_run_spec(
        args=args,
        checkpoint_spec=checkpoint_run_spec,
        constraint_profile=phase2_constraint_profile,
        phase1_bay_capacity_weights=phase1_bay_capacity_weights,
        phase1_long_cut_hard_mask=phase1_long_cut_hard_mask,
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
        long_cut_hard_mask=phase1_long_cut_hard_mask,
    )
    print(f"- resolved_phase1_bay_ids: {requested_bay_ids}")
    print(f"- phase1_bay_capacity_weights: {phase1_bay_capacity_weights}")
    print(f"- phase1_long_cut_hard_mask: {phase1_long_cut_hard_mask}")
    print(f"- phase2_score_mode: {effective_run_spec['score_mode']}")
    print(f"- action_pool_limit: {effective_run_spec['action_pool_limit']}")
    print(f"- max_wo_count: {effective_run_spec['max_wo_count']}")
    print(f"- max_length_sum: {effective_run_spec['max_length_sum']}")
    print(f"- rollout_samples: {effective_run_spec['validation_rollout_samples']}")
    result = run_phase2_full_graph_workflow(
        scenario=scenario,
        phase1_plan=phase1_plan,
        assignment_mode=args.assignment_mode,
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
    bay_ids: Sequence[str],
    sample_count: int,
    temperature: float,
    seed: int,
    score_mode: str,
    long_cut_hard_mask: bool,
    bay_capacity_weights: Mapping[str, int | float],
) -> Callable[[Mapping[str, object], int], Mapping[str, str]]:
    """Build frozen Phase 1 agent inference used as Phase 2 upstream input."""

    if sample_count <= 0:
        print(f"[ERROR][main._phase1_agent_assignment_builder] cause=non_positive_sample_count value={sample_count}")
        raise RuntimeError("--phase1-samples must be positive")
    if temperature <= 0:
        print(f"[ERROR][main._phase1_agent_assignment_builder] cause=non_positive_temperature value={temperature}")
        raise RuntimeError("--phase1-temperature must be positive")
    checkpoint_path = _phase1_checkpoint_path(checkpoint)
    model = load_phase1_pair_pointer_checkpoint(checkpoint_path)
    normalized_bay_ids = tuple(str(bay_id) for bay_id in bay_ids)

    def build(jobs: Mapping[str, object], assignment_seed: int) -> Mapping[str, str]:
        candidates = []
        if sample_count == 1:
            candidates.append(
                run_phase1_pair_policy_rollout(
                    jobs=jobs,
                    bay_ids=normalized_bay_ids,
                    model=model,
                    temperature=temperature,
                    seed=seed + assignment_seed,
                    source="phase1_agent_greedy",
                    selection="greedy",
                    long_cut_hard_mask=long_cut_hard_mask,
                    bay_capacity_weights=bay_capacity_weights,
                )
            )
        else:
            for sample_index in range(1, sample_count + 1):
                candidates.append(
                    run_phase1_pair_policy_rollout(
                        jobs=jobs,
                        bay_ids=normalized_bay_ids,
                        model=model,
                        temperature=temperature,
                        seed=seed + assignment_seed * 10_000 + sample_index,
                        source=f"phase1_agent_sample_{sample_index}",
                        selection="sample",
                        long_cut_hard_mask=long_cut_hard_mask,
                        bay_capacity_weights=bay_capacity_weights,
                    )
                )
        best = min(candidates, key=lambda candidate: _score_bay_loads(candidate.bay_loads, score_mode))
        print(
            "[CHECK][main._phase1_agent_assignment_builder] "
            f"assignment_seed={assignment_seed} source={best.source} samples={sample_count} "
            f"score={_score_bay_loads(best.bay_loads, score_mode)}",
            flush=True,
        )
        return dict(best.assignments)

    return build


def _replace_scenario_jobs_with_report_formula(
    scenario: Mapping[str, Any],
    block_count: int,
    seed: int,
    gyel: str,
) -> dict:
    """Return a scenario whose jobs come from the PDF fixed formulas.

    Full-flow still needs the real/configured machine layout.  Therefore this
    helper preserves every scenario field except `jobs`, which is replaced by
    generated W/O rows that follow the report formulas exactly.
    """

    if block_count <= 0:
        print(
            "[ERROR][main._replace_scenario_jobs_with_report_formula] "
            f"cause=invalid_block_count value={block_count}"
        )
        raise RuntimeError("--synthetic-blocks must be positive")
    if "machines" not in scenario:
        print("[ERROR][main._replace_scenario_jobs_with_report_formula] cause=missing_machines")
        raise RuntimeError("base scenario must contain machines for PDF synthetic full-flow")
    episode = build_report_formula_episode_jobs(
        episode_count=1,
        min_blocks=block_count,
        max_blocks=block_count,
        seed=seed,
        gyel=gyel,
    )[0]
    scenario_copy = dict(scenario)
    scenario_copy["jobs"] = scenario_jobs_from_report_formula_jobs(episode["jobs"])
    metadata = dict(scenario_copy.get("metadata") or {})
    metadata.update(
        {
            "job_source": "report_formula",
            "synthetic_block_count": block_count,
            "synthetic_job_count": len(scenario_copy["jobs"]),
            "synthetic_seed": seed,
            "synthetic_gyel": gyel,
        }
    )
    scenario_copy["metadata"] = metadata
    print(
        "[VALIDATION][main._replace_scenario_jobs_with_report_formula] "
        f"passed=true blocks={block_count} jobs={len(scenario_copy['jobs'])} seed={seed}"
    )
    return scenario_copy


def _load_or_build_phase1_plan_for_full_flow(
    args: argparse.Namespace,
    scenario: Mapping[str, Any],
    *,
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float],
    long_cut_hard_mask: bool,
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
            long_cut_hard_mask=long_cut_hard_mask,
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
            long_cut_hard_mask=long_cut_hard_mask,
            bay_capacity_weights=bay_capacity_weights,
        )
        plan = candidate_to_phase1_plan(
            jobs=jobs,
            bay_ids=bay_ids,
            candidate=candidate,
            score_mode=args.phase1_score_mode,
            long_cut_hard_mask=long_cut_hard_mask,
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
                    long_cut_hard_mask=long_cut_hard_mask,
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
                    long_cut_hard_mask=long_cut_hard_mask,
                    bay_capacity_weights=bay_capacity_weights,
                )
            )
    best = min(candidates, key=lambda candidate: _score_bay_loads(candidate.bay_loads, args.phase1_score_mode))
    plan = candidate_to_phase1_plan(
        jobs=jobs,
        bay_ids=bay_ids,
        candidate=best,
        score_mode=args.phase1_score_mode,
        long_cut_hard_mask=long_cut_hard_mask,
    )
    plan_dir = Path(args.output_dir) / "phase1_agent_plan"
    plan_paths = write_phase1_bay_plan(plan, plan_dir)
    print(
        "[CHECK][main._load_or_build_phase1_plan_for_full_flow] "
        f"mode=checkpoint source={best.source} assignment_count={len(best.assignments)} "
        f"plan_json={plan_paths['json']}"
    )
    return plan


def _phase1_bay_capacity_weights_from_scenario(
    scenario: Mapping[str, Any],
    bay_ids: Sequence[str],
) -> dict[str, float]:
    """Scenario의 enabled machine 수로 Phase 1 Bay 용량비를 계산한다."""

    machines = scenario.get("machines")
    if not isinstance(machines, list) or not machines:
        print("[ERROR][main._phase1_bay_capacity_weights_from_scenario] cause=no_machines")
        raise RuntimeError("full-flow scenario requires machines")
    normalized_bays = tuple(str(bay_id) for bay_id in bay_ids)
    weights = {bay_id: 0.0 for bay_id in normalized_bays}
    for index, machine in enumerate(machines):
        if not isinstance(machine, Mapping):
            print(
                "[ERROR][main._phase1_bay_capacity_weights_from_scenario] "
                f"cause=invalid_machine_row index={index} type={type(machine).__name__}"
            )
            raise RuntimeError(f"invalid full-flow machine row at index={index}")
        enabled = machine.get("enabled")
        if not isinstance(enabled, bool):
            print(
                "[ERROR][main._phase1_bay_capacity_weights_from_scenario] "
                f"cause=non_boolean_enabled index={index} value={enabled}"
            )
            raise RuntimeError(f"full-flow machine enabled must be boolean at index={index}")
        bay_id = str(machine.get("bay_id") or "").strip()
        if not bay_id:
            print(
                "[ERROR][main._phase1_bay_capacity_weights_from_scenario] "
                f"cause=missing_bay_id index={index}"
            )
            raise RuntimeError(f"full-flow machine bay_id is missing at index={index}")
        if enabled and bay_id in weights:
            weights[bay_id] += 1.0
    missing = sorted(bay_id for bay_id, value in weights.items() if value <= 0)
    if missing:
        print(
            "[ERROR][main._phase1_bay_capacity_weights_from_scenario] "
            f"cause=bay_without_enabled_machine bay_ids={missing}"
        )
        raise RuntimeError(f"full-flow selected Bays have no enabled machines: {missing}")
    return weights


def _validate_phase1_plan_execution_contract(
    plan: Mapping[str, Any],
    *,
    bay_ids: Sequence[str],
    bay_capacity_weights: Mapping[str, int | float],
    long_cut_hard_mask: bool,
) -> None:
    """외부 Phase 1 plan이 현재 full-flow 계약과 같은지 검증한다."""

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
    plan_mask = plan.get("long_cut_hard_mask")
    if not isinstance(plan_mask, bool):
        print("[ERROR][main._validate_phase1_plan_execution_contract] cause=missing_long_cut_mask")
        raise RuntimeError("Phase 1 plan is missing long_cut_hard_mask")
    if plan_mask is not bool(long_cut_hard_mask):
        print(
            "[ERROR][main._validate_phase1_plan_execution_contract] "
            f"cause=long_cut_mask_mismatch expected={bool(long_cut_hard_mask)} actual={plan_mask}"
        )
        raise RuntimeError("Phase 1 plan long-cut mask differs from full-flow")


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
    phase1_long_cut_hard_mask: bool,
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
        phase1_long_cut_hard_mask=phase1_long_cut_hard_mask,
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


def command_phase1_phase2_communicate(args: argparse.Namespace) -> None:
    """Write limited Phase 1 <-> Phase 2 communication messages and feedback."""

    print("[phase1-phase2-communicate]")
    print(f"- config: {args.config}")
    print(f"- scenario_path_override: {args.scenario_path}")
    print(f"- phase1_plan: {args.phase1_plan}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- wide_bth_threshold: {args.wide_bth_threshold}")
    print(f"- batch_max_wo_count: {args.batch_max_wo_count}")
    print(f"- batch_max_length_sum: {args.batch_max_length_sum}")

    plan_path = Path(args.phase1_plan)
    if not plan_path.exists():
        print(
            "[ERROR][main.command_phase1_phase2_communicate] "
            f"cause=missing_phase1_plan path={plan_path}"
        )
        raise FileNotFoundError(f"Phase 1 plan does not exist: {plan_path}")

    config = load_config(args.config)
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    with plan_path.open("r", encoding="utf-8") as file:
        plan = json.load(file)

    package = write_phase1_phase2_communication_package(
        scenario=scenario,
        plan=plan,
        output_dir=args.output_dir,
        wide_bth_threshold=args.wide_bth_threshold,
        batch_max_wo_count=args.batch_max_wo_count,
        batch_max_length_sum=args.batch_max_length_sum,
    )
    summary = package["summary"]
    print(f"- block_message_count: {summary['block_message_count']}")
    print(f"- feedback_message_count: {summary['feedback_message_count']}")
    print(f"- repair_required_count: {summary['repair_required_count']}")
    print(f"- infeasible_count: {summary['infeasible_count']}")
    print(f"- hard_violation_count: {summary['hard_violation_count']}")
    print(f"- phase1_to_phase2_jsonl: {package['phase1_to_phase2_jsonl']}")
    print(f"- phase2_to_phase1_jsonl: {package['phase2_to_phase1_jsonl']}")
    print(f"- feedback_csv: {package['feedback_csv']}")
    print(f"- manifest_json: {package['manifest_json']}")
    print("[VALIDATION][main.command_phase1_phase2_communicate] passed=true")


def command_apply_phase1_messages_to_phase2(args: argparse.Namespace) -> None:
    """Write a Phase 2 scenario using Phase 1-to-Phase 2 message JSONL."""

    print("[apply-phase1-messages-to-phase2]")
    print(f"- config: {args.config}")
    print(f"- scenario_path_override: {args.scenario_path}")
    print(f"- phase1_messages: {args.phase1_messages}")
    print(f"- assignment_mode: {args.assignment_mode}")
    print(f"- output_scenario: {args.output_scenario}")

    config = load_config(args.config)
    scenario = load_scenario_for_config(config, scenario_path_override=args.scenario_path)
    messages = load_communication_jsonl(args.phase1_messages)
    result = apply_phase1_messages_to_scenario(
        scenario=scenario,
        phase1_messages=messages,
        assignment_mode=args.assignment_mode,
    )
    save_scenario(result["scenario"], args.output_scenario)
    summary = result["summary"]

    print(f"- assigned_job_count: {summary['assigned_job_count']}")
    print(f"- assigned_block_count: {summary['assigned_block_count']}")
    print(f"- message_assignment_count: {summary['message_assignment_count']}")
    print(f"- output_scenario: {args.output_scenario}")
    print("[VALIDATION][main.command_apply_phase1_messages_to_phase2] passed=true")


def command_generate_phase1_blocks(args: argparse.Namespace) -> None:
    """Generate block-only synthetic data for Phase 1 training smoke tests."""

    print("[generate-phase1-blocks]")
    print(f"- block_xlsx: {args.block_xlsx}")
    print(f"- gyel: {args.gyel}")
    print(f"- n_blocks: {args.n_blocks}")
    print(f"- seed: {args.seed}")
    print(f"- noise_ratio: {args.noise_ratio}")
    print(f"- correlation_method: {args.correlation_method}")
    print(f"- output_dir: {args.output_dir}")
    actual_blocks = load_phase1_actual_blocks(args.block_xlsx, gyel=args.gyel)
    paths = write_phase1_block_generation_package(
        actual_blocks=actual_blocks,
        output_dir=args.output_dir,
        n_blocks=args.n_blocks,
        seed=args.seed,
        noise_ratio=args.noise_ratio,
        method=args.correlation_method,
    )
    print(f"- synthetic_csv: {paths['synthetic_csv']}")
    print(f"- summary_json: {paths['summary_json']}")
    print(f"- distribution_summary_csv: {paths['distribution_summary_csv']}")
    print(f"- actual_correlation_csv: {paths['actual_correlation_csv']}")
    print(f"- synthetic_correlation_csv: {paths['synthetic_correlation_csv']}")
    print(f"- correlation_delta_csv: {paths['correlation_delta_csv']}")


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


def command_build_html_package(args: argparse.Namespace) -> None:
    """clean 50/100/200 휴리스틱 baseline과 actual replay를 정적 HTML 패키지로 묶는다."""

    config_paths = parse_csv_argument(
        args.configs,
        default=("config_clean_50.yaml", "config_clean_100.yaml", "config_clean_200.yaml"),
    )
    heuristics = parse_csv_argument(
        args.heuristics,
        default=("batch_fill_spt", "balanced_batch", "balanced_batch_count", "spt", "load_balance", "priority"),
    )
    result = build_html_package(
        config_paths=config_paths,
        heuristics=heuristics,
        output_dir=args.output_dir,
        zip_path=args.zip_path,
    )

    print("[html package]")
    print(f"- configs: {len(config_paths)}")
    print(f"- heuristics: {len(heuristics)}")
    print(f"- baseline_rows: {result['baseline_rows']}")
    print(f"- actual_rows: {result['actual_rows']}")
    print(f"- package_dir: {result['package_dir']}")
    print(f"- index_html: {result['index_html']}")
    print(f"- zip_path: {result['zip_path']}")


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


def command_hierarchical_trace(args: argparse.Namespace) -> None:
    """계층형 Gym wrapper 기준 action trace를 CSV/JSONL로 저장한다."""

    heuristic_name = args.heuristic
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("output") / f"hierarchical_trace_{Path(args.config).stem}_{heuristic_name}")

    print("[hierarchical trace]")
    print(f"- config: {args.config}")
    print(f"- heuristic: {heuristic_name}")
    print(f"- max_actions: {args.max_actions}")
    print(f"- output_dir: {output_dir}")
    print(f"- gymnasium_available: {GYMNASIUM_AVAILABLE}")
    env = build_environment(args.config)
    result = run_hierarchical_trace_export(
        env=env,
        heuristic_name=heuristic_name,
        max_actions=args.max_actions,
        output_dir=output_dir,
    )
    print(f"- scheduled_jobs: {result['scheduled_jobs']}")
    print(f"- unscheduled_jobs: {result['unscheduled_jobs']}")
    print(f"- hierarchical_step_count: {result['hierarchical_step_count']}")
    print(f"- flat_decision_count: {result['flat_decision_count']}")
    print(f"- phase_counts: {result['phase_counts']}")
    print(f"- trace_csv: {result['trace_csv']}")
    print(f"- action_table_jsonl: {result['action_table_jsonl']}")
    print(f"- summary_json: {result['summary_json']}")


def command_build_learning_data(args: argparse.Namespace) -> None:
    """계층형 trace들을 학습 smoke dataset package로 묶는다."""

    config_paths = parse_csv_argument(args.configs, default=("config_np_100.yaml",))
    heuristics = parse_csv_argument(args.heuristics, default=("spt",))
    algorithm_plan = parse_csv_argument(
        args.algorithm_plan,
        default=("self_labeling", "ppo", "reinforce"),
    )
    print("[learning data]")
    print(f"- configs: {config_paths}")
    print(f"- heuristics: {heuristics}")
    print(f"- data_role: {args.data_role}")
    print(f"- algorithm_plan: {algorithm_plan}")
    print(f"- max_actions: {args.max_actions}")
    print(f"- output_dir: {args.output_dir}")
    result = build_learning_data_package(
        config_paths=config_paths,
        heuristics=heuristics,
        output_dir=args.output_dir,
        max_actions=args.max_actions,
        data_role=args.data_role,
        algorithm_plan=algorithm_plan,
        env_builder=build_environment,
    )
    print(f"- dataset_count: {result['dataset_count']}")
    print(f"- label_source: {result['label_source']}")
    print(f"- manifest_json: {result['manifest_json']}")


def command_pygame_viewer(args: argparse.Namespace) -> None:
    """event/schedule 산출물을 Pygame 로컬 공장 viewer로 연다.

    `--dry-run`을 주면 GUI 창을 열지 않고 입력 파일 4종의 구조와 count만 검증한다.
    """

    summary = run_pygame_viewer_from_paths(
        event_log_path=args.event_log,
        layout_path=args.layout,
        schedule_path=args.schedule,
        metrics_path=args.metrics,
        width=args.width,
        height=args.height,
        speed=args.speed,
        start_paused=args.paused,
        dry_run=args.dry_run,
        screenshot_path=args.screenshot,
        max_frames=args.max_frames,
    )
    print("[pygame viewer summary]")
    print(f"- operation_count: {summary['operation_count']}")
    print(f"- machine_count: {summary['machine_count']}")
    print(f"- batch_count: {summary['batch_count']}")
    print(f"- bay_ids: {summary['bay_ids']}")
    print(f"- event_count: {summary['event_count']}")
    print(f"- start_time_min: {float(summary['start_time_min']):.2f}")
    print(f"- end_time_min: {float(summary['end_time_min']):.2f}")


def command_pygame_compare(args: argparse.Namespace) -> None:
    """actual/generated 산출물을 한 Pygame 창에서 좌우 비교한다."""

    summary = run_pygame_comparison_from_paths(
        left_event_log_path=args.left_event_log,
        left_layout_path=args.left_layout,
        left_schedule_path=args.left_schedule,
        left_metrics_path=args.left_metrics,
        right_event_log_path=args.right_event_log,
        right_layout_path=args.right_layout,
        right_schedule_path=args.right_schedule,
        right_metrics_path=args.right_metrics,
        left_label=args.left_label,
        right_label=args.right_label,
        width=args.width,
        height=args.height,
        speed=args.speed,
        start_paused=args.paused,
        dry_run=args.dry_run,
        screenshot_path=args.screenshot,
        max_frames=args.max_frames,
        gif_path=args.gif,
        gif_frames=args.gif_frames,
        gif_duration_ms=args.gif_duration_ms,
    )
    print("[pygame compare summary]")
    print(f"- left_operation_count: {summary['left']['operation_count']}")
    print(f"- right_operation_count: {summary['right']['operation_count']}")
    print(f"- left_batch_count: {summary['left']['batch_count']}")
    print(f"- right_batch_count: {summary['right']['batch_count']}")
    print(f"- left_end_time_min: {float(summary['left']['end_time_min']):.2f}")
    print(f"- right_end_time_min: {float(summary['right']['end_time_min']):.2f}")
    if "gif_path" in summary:
        print(f"- gif_path: {summary['gif_path']}")


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
    # LINE-BY-LINE: `common_parser.add_argument("--config", default` 여러 변수에 `"config.yaml", help="YAML config path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    common_parser.add_argument("--config", default="config.yaml", help="YAML config path")
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

    phase1_parser = subparsers.add_parser(
        "phase1",
        parents=[common_parser],
        help="Run Phase 1 only: block-to-Bay workload balancing",
    )
    phase1_parser.add_argument(
        "--mode",
        default="heuristic",
        choices=["heuristic", "train", "infer"],
        help="Phase 1 execution mode. Only heuristic is implemented now.",
    )
    phase1_parser.add_argument(
        "--algorithm",
        default=CANONICAL_PHASE1_HEURISTIC,
        choices=[
            CANONICAL_PHASE1_HEURISTIC,
            LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
            MULTI_OBJECTIVE_PHASE1_HEURISTIC,
            PRIORITY_SWEEP_PHASE1_HEURISTIC,
            PRIORITY_GREEDY_PHASE1_HEURISTIC,
            "lpt_steel_quantity",
            "heuristic",
        ],
        help="Phase 1 heuristic algorithm",
    )
    phase1_parser.add_argument(
        "--bay-ids",
        default=None,
        help="Optional comma-separated Bay IDs. Default uses enabled machine Bays from config.",
    )
    phase1_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Phase 1 JSON/CSV outputs",
    )
    phase1_parser.set_defaults(func=command_phase1)

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

    phase1_mdp_trace_parser = subparsers.add_parser(
        "phase1-mdp-trace",
        parents=[common_parser],
        help="Build Phase 1 SELECT_BLOCK -> SELECT_BAY self-label trace",
    )
    phase1_mdp_trace_parser.add_argument(
        "--algorithm",
        default=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
        choices=[
            CANONICAL_PHASE1_HEURISTIC,
            LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
            MULTI_OBJECTIVE_PHASE1_HEURISTIC,
            PRIORITY_SWEEP_PHASE1_HEURISTIC,
            PRIORITY_GREEDY_PHASE1_HEURISTIC,
            "lpt_steel_quantity",
            "heuristic",
        ],
        help="Phase 1 heuristic used as self-label teacher",
    )
    phase1_mdp_trace_parser.add_argument(
        "--bay-ids",
        default=None,
        help="Optional comma-separated Bay IDs. Default uses enabled machine Bays from config.",
    )
    phase1_mdp_trace_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for Phase 1 MDP trace outputs",
    )
    phase1_mdp_trace_parser.set_defaults(func=command_phase1_mdp_trace)

    phase1_train_imitation_parser = subparsers.add_parser(
        "phase1-train-imitation",
        help="Train Phase 1 pointer policy from phase1_action_table.jsonl",
    )
    phase1_train_imitation_parser.add_argument(
        "--action-table",
        required=True,
        help="Path to phase1_action_table.jsonl from phase1-mdp-trace",
    )
    phase1_train_imitation_parser.add_argument(
        "--eval-action-table",
        default=None,
        help="Optional holdout phase1_action_table.jsonl for eval metrics",
    )
    phase1_train_imitation_parser.add_argument(
        "--output-dir",
        default="output/phase1_imitation",
        help="Directory for checkpoint and metrics",
    )
    phase1_train_imitation_parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    phase1_train_imitation_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    phase1_train_imitation_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    phase1_train_imitation_parser.add_argument("--seed", type=int, default=0, help="Torch random seed")
    phase1_train_imitation_parser.set_defaults(func=command_phase1_train_imitation)

    phase1_train_self_labeling_parser = subparsers.add_parser(
        "phase1-train-self-labeling",
        parents=[common_parser],
        help="Train Phase 1 pointer policy with best-of-K self-labeling",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--bay-ids",
        default="22,23,24",
        help="Optional comma-separated Bay IDs. Default uses 22,23,24 for Phase 1.",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--output-dir",
        default="output/phase1_self_labeling",
        help="Directory for checkpoint and metrics",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--episode-mode",
        choices=["sampled", "fixed"],
        default="sampled",
        help="sampled uses block-level synthetic episodes; fixed repeats config_np_100 for smoke/debug only",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--block-xlsx",
        default="input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx",
        help="Actual block Excel/CSV path used for sampled self-labeling episodes",
    )
    phase1_train_self_labeling_parser.add_argument("--gyel", default="NP", help="Series filter for sampled episodes")
    phase1_train_self_labeling_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum blocks per sampled episode")
    phase1_train_self_labeling_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum blocks per sampled episode")
    phase1_train_self_labeling_parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.03,
        help="Bootstrap jitter ratio for sampled episodes",
    )
    phase1_train_self_labeling_parser.add_argument("--episodes", type=int, default=20, help="Self-labeling episodes")
    phase1_train_self_labeling_parser.add_argument(
        "--rollout-samples",
        type=int,
        default=4,
        help="Current-policy sampled candidates per episode",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--heuristic-algorithms",
        default="all8",
        help="Comma-separated heuristic candidates that compete with agent samples",
    )
    phase1_train_self_labeling_parser.add_argument(
        "--score-mode",
        choices=["steel_first"],
        default="steel_first",
        help="Fixed objective: capacity-normalized steel gap, cut gap, bevel gap",
    )
    phase1_train_self_labeling_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    phase1_train_self_labeling_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    phase1_train_self_labeling_parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    phase1_train_self_labeling_parser.add_argument("--seed", type=int, default=0, help="Torch random seed")
    phase1_train_self_labeling_parser.set_defaults(func=command_phase1_train_self_labeling)

    phase1_train_pair_self_labeling_parser = subparsers.add_parser(
        "phase1-train-pair-self-labeling",
        parents=[common_parser],
        help="Train Phase 1 direct SELECT_PAIR(block,bay) policy with best-of-K self-labeling",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--bay-ids",
        default="22,23,24",
        help="Optional comma-separated Bay IDs. Default uses 22,23,24 for Phase 1.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--block-xlsx",
        default="input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx",
        help="Actual block Excel/CSV path used for sampled pair self-labeling episodes",
    )
    phase1_train_pair_self_labeling_parser.add_argument("--gyel", default="NP", help="Series filter for sampled episodes")
    phase1_train_pair_self_labeling_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum blocks per sampled episode")
    phase1_train_pair_self_labeling_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum blocks per sampled episode")
    phase1_train_pair_self_labeling_parser.add_argument("--noise-ratio", type=float, default=0.03, help="Bootstrap jitter ratio")
    phase1_train_pair_self_labeling_parser.add_argument(
        "--hard-case-ratio",
        type=float,
        default=0.0,
        help="Probability that a sampled synthetic episode becomes a hard steel-cut decorrelation case.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--hard-case-mode",
        choices=["none", "cut_shuffle"],
        default="cut_shuffle",
        help="Hard-case generator mode. Ratio 0 keeps the original bootstrap+jitter behavior.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--hard-case-target-corr",
        type=float,
        default=0.85,
        help="Target maximum Pearson corr(STL_QTY,CUT_LTH) for cut_shuffle hard cases.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--hard-case-max-attempts",
        type=int,
        default=20,
        help="Maximum cut-shuffle attempts before failing a hard-case episode.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--validation-hard-case-ratio",
        type=float,
        default=None,
        help="Synthetic validation hard-case ratio. Default follows --hard-case-ratio; actual_8days is unchanged.",
    )
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
        default="all8",
        help="Comma-separated heuristic candidates, or all8",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--score-mode",
        choices=["steel_first"],
        default="steel_first",
        help="Fixed objective: capacity-normalized steel gap, cut gap, bevel gap",
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
        "--actual-validation-candidate-xlsx",
        default="착수일 후보 블록.xlsx",
        help="Candidate block workbook used for fixed actual_8days Phase 1 validation.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--actual-validation-workdays",
        default="20260331,20260407,20260408,20260413,20260414,20260415,20260424,20260429",
        help="Comma-separated actual workdays for fixed validation. Empty string disables actual validation.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--enable-phase2-feedback-score",
        action="store_true",
        help="Prepend the frozen Phase 2 best-of-K schedule score to the Phase 1 objective.",
    )
    phase1_train_pair_self_labeling_parser.add_argument(
        "--phase2-feedback-checkpoint",
        default="",
        help="Frozen Phase 2 set-pointer checkpoint. Required with --enable-phase2-feedback-score.",
    )
    phase1_train_pair_self_labeling_parser.set_defaults(func=command_phase1_train_pair_self_labeling)

    phase1_episode_dataset_parser = subparsers.add_parser(
        "phase1-build-episode-dataset",
        help="Build variable-size Phase 1 train/test self-label episodes",
    )
    phase1_episode_dataset_parser.add_argument(
        "--block-xlsx",
        default="input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx",
        help="Actual block Excel/CSV path used as the block-only source distribution",
    )
    phase1_episode_dataset_parser.add_argument("--gyel", default="NP", help="Series filter")
    phase1_episode_dataset_parser.add_argument("--episode-count", type=int, default=40, help="Number of episodes")
    phase1_episode_dataset_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum blocks per episode")
    phase1_episode_dataset_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum blocks per episode")
    phase1_episode_dataset_parser.add_argument("--train-ratio", type=float, default=0.8, help="Train episode ratio")
    phase1_episode_dataset_parser.add_argument("--bay-ids", default="22,23,24", help="Comma-separated Phase 1 Bay IDs")
    phase1_episode_dataset_parser.add_argument(
        "--algorithm",
        default=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
        choices=[
            CANONICAL_PHASE1_HEURISTIC,
            LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
            MULTI_OBJECTIVE_PHASE1_HEURISTIC,
            PRIORITY_SWEEP_PHASE1_HEURISTIC,
            PRIORITY_GREEDY_PHASE1_HEURISTIC,
            "lpt_steel_quantity",
            "heuristic",
        ],
        help="Phase 1 teacher heuristic",
    )
    phase1_episode_dataset_parser.add_argument("--seed", type=int, default=2026, help="Dataset seed")
    phase1_episode_dataset_parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.03,
        help="Bootstrap jitter ratio. Set 0 for exact empirical bootstrap.",
    )
    phase1_episode_dataset_parser.add_argument(
        "--output-dir",
        default="output/phase1_episode_dataset",
        help="Output directory for episode dataset files",
    )
    phase1_episode_dataset_parser.set_defaults(func=command_phase1_build_episode_dataset)

    phase2_train_graph_parser = subparsers.add_parser(
        "phase2-train-batch-machine-self-labeling",
        parents=[common_parser],
        help="Train merged Phase 2 policy that selects W/O batch and machine together",
    )
    phase2_train_graph_parser.add_argument(
        "--scenario-path",
        default=None,
        help="Optional source scenario override. Default follows config/data-source loader.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-heuristic",
        default=None,
        help="Phase 1 heuristic run in memory for every merged Phase 2 training episode. Default is bevel_first_balanced when --phase1-checkpoint is absent.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-checkpoint",
        default=None,
        help="Frozen Phase 1 pair-pointer checkpoint or output directory. Replaces --phase1-heuristic during Phase 2 training.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-bay-ids",
        default=None,
        help="Comma-separated Bay IDs for Phase 1 heuristic. Default uses enabled machine Bays.",
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
        "--phase1-score-mode",
        choices=["steel_first"],
        default="steel_first",
        help="Phase 1 checkpoint candidate score mode.",
    )
    phase2_train_graph_parser.add_argument(
        "--phase1-no-long-cut-hard-mask",
        action="store_true",
        help="Disable Phase 1 long-cut Bay 24 hard mask during upstream heuristic generation.",
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
    phase2_train_graph_parser.add_argument(
        "--synthetic-source",
        choices=["report_formula", "config"],
        default="report_formula",
        help="Merged Phase 2 training episode source. report_formula uses the PDF fixed formulas; config reuses config jobs.",
    )
    phase2_train_graph_parser.add_argument("--min-blocks", type=int, default=12, help="Minimum synthetic blocks per report-formula episode")
    phase2_train_graph_parser.add_argument("--max-blocks", type=int, default=80, help="Maximum synthetic blocks per report-formula episode")
    phase2_train_graph_parser.add_argument("--gyel", default="NP", help="Synthetic series/family label")
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
        help="Run merged Phase 2 batch-machine schedule export",
    )
    phase2_run_full_parser.add_argument(
        "--scenario-path",
        default=None,
        help="Optional source scenario override. Default follows config/data-source loader.",
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
        help="Optional fixed Phase 1 heuristic, e.g. mbf_ppb. Mutually exclusive with --phase1-plan/--phase1-checkpoint.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-bay-ids",
        default="22,23,24",
        help="Comma-separated Bay IDs used when --phase1-checkpoint builds the upstream plan.",
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
        "--phase1-score-mode",
        choices=["steel_first"],
        default="steel_first",
        help="Phase 1 checkpoint candidate score mode.",
    )
    phase2_run_full_parser.add_argument(
        "--phase1-no-long-cut-hard-mask",
        action="store_true",
        help="Evaluation-only ablation: disable Phase 1 long-cut Bay 24 hard mask.",
    )
    phase2_run_full_parser.add_argument(
        "--assignment-mode",
        default="allowed_bay_ids",
        choices=["allowed_bay_ids", "cut_bay"],
        help="How to inject Phase 1 Bay assignments before Phase 2.",
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
        "--synthetic-source",
        choices=["config", "report_formula"],
        default="config",
        help="Full-flow job source. report_formula replaces config jobs with PDF fixed-formula W/O rows.",
    )
    phase2_run_full_parser.add_argument(
        "--synthetic-blocks",
        type=int,
        default=30,
        help="Block count for --synthetic-source report_formula.",
    )
    phase2_run_full_parser.add_argument(
        "--gyel",
        default="NP",
        help="Synthetic series label for --synthetic-source report_formula.",
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

    communicate_parser = subparsers.add_parser(
        "phase1-phase2-communicate",
        parents=[common_parser],
        help="Write limited Phase 1-to-Phase 2 messages and Phase 2 feedback",
    )
    communicate_parser.add_argument(
        "--scenario-path",
        default=None,
        help="Optional source scenario override. Default follows config/data-source loader.",
    )
    communicate_parser.add_argument(
        "--phase1-plan",
        required=True,
        help="Phase 1 plan JSON path, usually phase1_block_bay_plan.json",
    )
    communicate_parser.add_argument(
        "--output-dir",
        default="output/phase1_phase2_communication",
        help="Output directory for communication JSONL/CSV/manifest files",
    )
    communicate_parser.add_argument(
        "--wide-bth-threshold",
        type=float,
        default=4500.0,
        help="BTH/plate_width threshold for wide blocks. Width above this requires Bay 22/23.",
    )
    communicate_parser.add_argument(
        "--batch-max-wo-count",
        type=int,
        default=3,
        help="Phase 2 machine batch W/O capacity.",
    )
    communicate_parser.add_argument(
        "--batch-max-length-sum",
        type=float,
        default=55000.0,
        help="Phase 2 machine batch LTH sum capacity.",
    )
    communicate_parser.set_defaults(func=command_phase1_phase2_communicate)

    apply_messages_parser = subparsers.add_parser(
        "apply-phase1-messages-to-phase2",
        parents=[common_parser],
        help="Apply Phase 1-to-Phase 2 message JSONL to a Phase 2 scenario",
    )
    apply_messages_parser.add_argument(
        "--scenario-path",
        default=None,
        help="Optional source scenario override. Default follows config/data-source loader.",
    )
    apply_messages_parser.add_argument(
        "--phase1-messages",
        required=True,
        help="Phase 1-to-Phase 2 message JSONL path.",
    )
    apply_messages_parser.add_argument(
        "--assignment-mode",
        default="allowed_bay_ids",
        choices=["cut_bay", "allowed_bay_ids"],
        help="How to write Phase 1 message Bay into Phase 2 jobs.",
    )
    apply_messages_parser.add_argument(
        "--output-scenario",
        default="output/generated/phase2_from_phase1_messages.yaml",
        help="Output Phase 2 scenario YAML path.",
    )
    apply_messages_parser.set_defaults(func=command_apply_phase1_messages_to_phase2)

    generate_phase1_blocks_parser = subparsers.add_parser(
        "generate-phase1-blocks",
        help="Generate block-only synthetic data for Phase 1 training",
    )
    generate_phase1_blocks_parser.add_argument(
        "--block-xlsx",
        default="input/절단03~04_NP물량_마스킹_블록_수정_260618.xlsx",
        help="Actual block Excel/CSV path used as the block-only source distribution",
    )
    generate_phase1_blocks_parser.add_argument(
        "--gyel",
        default="NP",
        help="Series filter. Use NP for the current Phase 1 scope.",
    )
    generate_phase1_blocks_parser.add_argument(
        "--n-blocks",
        type=int,
        default=787,
        help="Number of synthetic block rows to generate",
    )
    generate_phase1_blocks_parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic generation",
    )
    generate_phase1_blocks_parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.03,
        help="Small bootstrap jitter ratio. Set 0 for exact empirical bootstrap.",
    )
    generate_phase1_blocks_parser.add_argument(
        "--correlation-method",
        choices=["pearson", "spearman"],
        default="pearson",
        help="Correlation method for actual vs synthetic validation",
    )
    generate_phase1_blocks_parser.add_argument(
        "--output-dir",
        default="output/generated/phase1_block_only_synthetic",
        help="Output directory for synthetic CSV and validation reports",
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
    # LINE-BY-LINE: `build_scenario_parser.add_argument("--input-path", default` 여러 변수에 `"메일내용/절단03~04_NP물량_마스킹.xlsx", help="Cutting Excel/CSV input path")` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    build_scenario_parser.add_argument("--input-path", default="메일내용/절단03~04_NP물량_마스킹.xlsx", help="Cutting Excel/CSV input path")
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

    html_package_parser = subparsers.add_parser("build-html-package", help="Build static heuristic baseline and playback HTML package")
    html_package_parser.add_argument(
        "--configs",
        default="config_clean_50.yaml,config_clean_100.yaml,config_clean_200.yaml",
        help="Comma-separated clean config paths",
    )
    html_package_parser.add_argument(
        "--heuristics",
        default="batch_fill_spt,balanced_batch,balanced_batch_count,spt,load_balance,priority",
        help="Comma-separated heuristic names",
    )
    html_package_parser.add_argument(
        "--output-dir",
        default="output/share/html_viewer_package",
        help="Static HTML package output directory",
    )
    html_package_parser.add_argument(
        "--zip-path",
        default="output/share/html_viewer_package.zip",
        help="Zip file path for static package",
    )
    html_package_parser.set_defaults(func=command_build_html_package)

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

    hierarchical_trace_parser = subparsers.add_parser(
        "hierarchical-trace",
        parents=[common_parser],
        help="Export SELECT_MACHINE -> SELECT_WO/COMMIT trace for hierarchical Gym wrapper",
    )
    hierarchical_trace_parser.add_argument(
        "--heuristic",
        default="spt",
        help="Heuristic used as pseudo-label source",
    )
    hierarchical_trace_parser.add_argument(
        "--max-actions",
        type=int,
        default=4096,
        help="Fixed Gym Discrete action space size; fails if candidates exceed this",
    )
    hierarchical_trace_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for hierarchical_trace.csv, hierarchical_action_table.jsonl, and summary.json",
    )
    hierarchical_trace_parser.set_defaults(func=command_hierarchical_trace)

    learning_data_parser = subparsers.add_parser(
        "build-learning-data",
        help="Build hierarchical trace package for self-labeling/imitation/RL smoke tests",
    )
    learning_data_parser.add_argument(
        "--configs",
        default="config_np_100.yaml",
        help="Comma-separated config paths used as dataset sources",
    )
    learning_data_parser.add_argument(
        "--heuristics",
        default="spt,balanced_batch_count",
        help="Comma-separated heuristic pseudo-label sources",
    )
    learning_data_parser.add_argument(
        "--algorithm-plan",
        default="self_labeling,ppo,reinforce",
        help="Comma-separated algorithms intended to consume this package",
    )
    learning_data_parser.add_argument(
        "--data-role",
        default="actual_source_smoke",
        help="Dataset role: actual_source_smoke / clean_slice_smoke / heuristic_baseline",
    )
    learning_data_parser.add_argument(
        "--max-actions",
        type=int,
        default=4096,
        help="Fixed hierarchical action space size",
    )
    learning_data_parser.add_argument(
        "--output-dir",
        default="output/learning_data_smoke_np_100",
        help="Learning data package output directory",
    )
    learning_data_parser.set_defaults(func=command_build_learning_data)

    pygame_viewer_parser = subparsers.add_parser(
        "pygame-viewer",
        help="Open local Pygame factory playback viewer from event/schedule artifacts",
    )
    pygame_viewer_parser.add_argument("--event-log", required=True, help="event_log.json path")
    pygame_viewer_parser.add_argument("--layout", required=True, help="factory_layout.json path")
    pygame_viewer_parser.add_argument("--schedule", required=True, help="job_schedule.csv path")
    pygame_viewer_parser.add_argument("--metrics", required=True, help="metrics.json path")
    pygame_viewer_parser.add_argument("--width", type=int, default=1280, help="Viewer window width")
    pygame_viewer_parser.add_argument("--height", type=int, default=760, help="Viewer window height")
    pygame_viewer_parser.add_argument("--speed", type=float, default=30.0, help="Simulation minutes per real second")
    pygame_viewer_parser.add_argument("--paused", action="store_true", help="Start paused")
    pygame_viewer_parser.add_argument("--dry-run", action="store_true", help="Validate inputs without opening a GUI window")
    pygame_viewer_parser.add_argument("--screenshot", default=None, help="Save first rendered frame to this PNG path")
    pygame_viewer_parser.add_argument("--max-frames", type=int, default=None, help="Render this many frames and exit")
    pygame_viewer_parser.set_defaults(func=command_pygame_viewer)

    pygame_compare_parser = subparsers.add_parser(
        "pygame-compare",
        help="Open side-by-side Pygame comparison viewer for two playback artifacts",
    )
    pygame_compare_parser.add_argument("--left-event-log", required=True, help="Left event_log.json path")
    pygame_compare_parser.add_argument("--left-layout", required=True, help="Left factory_layout.json path")
    pygame_compare_parser.add_argument("--left-schedule", required=True, help="Left schedule CSV path")
    pygame_compare_parser.add_argument("--left-metrics", required=True, help="Left metrics.json path")
    pygame_compare_parser.add_argument("--right-event-log", required=True, help="Right event_log.json path")
    pygame_compare_parser.add_argument("--right-layout", required=True, help="Right factory_layout.json path")
    pygame_compare_parser.add_argument("--right-schedule", required=True, help="Right schedule CSV path")
    pygame_compare_parser.add_argument("--right-metrics", required=True, help="Right metrics.json path")
    pygame_compare_parser.add_argument("--left-label", default="ACTUAL REPLAY", help="Left panel title")
    pygame_compare_parser.add_argument("--right-label", default="GENERATED PLAN", help="Right panel title")
    pygame_compare_parser.add_argument("--width", type=int, default=1920, help="Viewer window width")
    pygame_compare_parser.add_argument("--height", type=int, default=820, help="Viewer window height")
    pygame_compare_parser.add_argument("--speed", type=float, default=30.0, help="Reference minutes per real second")
    pygame_compare_parser.add_argument("--paused", action="store_true", help="Start paused")
    pygame_compare_parser.add_argument("--dry-run", action="store_true", help="Validate inputs without opening a GUI window")
    pygame_compare_parser.add_argument("--screenshot", default=None, help="Save first rendered frame to this PNG path")
    pygame_compare_parser.add_argument("--max-frames", type=int, default=None, help="Render this many frames and exit")
    pygame_compare_parser.add_argument("--gif", default=None, help="Save side-by-side playback to this GIF path")
    pygame_compare_parser.add_argument("--gif-frames", type=int, default=48, help="Number of frames for --gif export")
    pygame_compare_parser.add_argument("--gif-duration-ms", type=int, default=120, help="Frame duration in milliseconds for --gif export")
    pygame_compare_parser.set_defaults(func=command_pygame_compare)

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
