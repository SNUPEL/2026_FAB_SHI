"""프로젝트 공통 실행 진입점.

초보자 기준 실행 예시:
  python3 main.py show-config --config config.yaml
  python3 main.py build-scenario --config config_np_100.yaml
  python3 main.py simulate --config config.yaml
  python3 main.py playback --config config_np_100.yaml
  python3 main.py pygame-viewer --event-log output/share/html_viewer_package/clean_100/generated/balanced_batch/event_log.json --layout output/share/html_viewer_package/clean_100/generated/balanced_batch/factory_layout.json --schedule output/share/html_viewer_package/clean_100/generated/balanced_batch/job_schedule.csv --metrics output/share/html_viewer_package/clean_100/generated/balanced_batch/metrics.json
  python3 main.py trace --config config.yaml
  python3 main.py train --config config.yaml
  python3 main.py eval --config config.yaml
"""

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

# LINE-BY-LINE: `Agent.heuristics` 모듈에서 `select_action_by_rule`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Agent.heuristics import select_action_by_rule
# LINE-BY-LINE: `Environment.environment` 모듈에서 `CuttingShopEnvironment`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.environment import CuttingShopEnvironment
from Environment.gym_wrapper import GYMNASIUM_AVAILABLE, run_hierarchical_trace_export, run_wrapper_equivalence
# LINE-BY-LINE: `Train.runner` 모듈에서 `run_eval, run_train`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Train.runner import run_eval, run_train
# LINE-BY-LINE: `Utils.config` 모듈에서 `load_config`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.config import load_config
# LINE-BY-LINE: `Utils.cutting_data_loader` 모듈에서 `load_and_clean_cutting_data, write_records_csv`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.cutting_data_loader import load_and_clean_cutting_data, write_records_csv
# LINE-BY-LINE: `Utils.cutting_scenario_builder` 모듈에서 `build_scenario_from_cutting_records`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.cutting_scenario_builder import build_scenario_from_cutting_records
# LINE-BY-LINE: `Utils.factory_builder` 모듈에서 `build_factory_scenario_parts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.factory_builder import build_factory_scenario_parts
# LINE-BY-LINE: `Utils.io` 모듈에서 scenario loader를 가져옵니다. `load_scenario`는 명시 YAML용, `load_scenario_for_config`는 config 기반 원본 데이터 로딩용입니다.
from Utils.io import load_scenario, load_scenario_for_config
from Utils.learning_data_builder import build_learning_data_package
from Utils.phase1_bay_balancer import (
    CANONICAL_PHASE1_HEURISTIC,
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    MULTI_OBJECTIVE_PHASE1_HEURISTIC,
    PRIORITY_GREEDY_PHASE1_HEURISTIC,
    PRIORITY_SWEEP_PHASE1_HEURISTIC,
    apply_phase1_plan_to_scenario,
    build_phase1_bay_plan,
    write_phase1_bay_plan,
)
# LINE-BY-LINE: `Utils.playback_builder` 모듈에서 `write_actual_replay_artifacts, write_playback_artifacts`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.playback_builder import write_actual_replay_artifacts, write_playback_artifacts
# LINE-BY-LINE: `Utils.scenario_generator` 모듈에서 `generate_scenario_from_template, save_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.scenario_generator import generate_scenario_from_template, save_scenario
# 회사 송부용 정적 HTML 패키지와 휴리스틱 baseline 비교표를 생성하는 helper입니다.
from Utils.share_report_builder import build_html_package, parse_csv_argument
# Pygame 로컬 공장 playback viewer입니다. `--dry-run`으로 GUI 없이 입력 검증도 가능합니다.
from Utils.pygame_factory_viewer import run_pygame_comparison_from_paths, run_pygame_viewer_from_paths
# LINE-BY-LINE: `Utils.tact_gap_analysis` 모듈에서 `build_tact_gap_analysis`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.tact_gap_analysis import build_tact_gap_analysis
# LINE-BY-LINE: `Utils.tact_time` 모듈에서 `build_tact_time_analysis_from_scenario`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.tact_time import build_tact_time_analysis_from_scenario
# LINE-BY-LINE: `Utils.test_data_selection` 모듈에서 `select_actual_day_test_data, select_actual_start_range_test_data`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Utils.test_data_selection import select_actual_day_test_data, select_actual_start_range_test_data


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
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path("output") / f"phase1_{Path(args.config).stem}_{args.algorithm}")

    plan = build_phase1_bay_plan(
        jobs=env.jobs,
        bay_ids=bay_ids,
        algorithm=args.algorithm,
    )
    paths = write_phase1_bay_plan(plan, output_dir)
    summary = plan["summary"]

    print(f"- resolved_bay_ids: {bay_ids}")
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

    # LINE-BY-LINE: `train_parser`에 `subparsers.add_parser("train", parents=[common_parser], help="Run training")` 결과를 저장합니다. 의미/사용: `train_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Run training")
    # LINE-BY-LINE: `train_parser.set_defaults(func`에 `lambda args: run_train(load_config(args.config)))` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    train_parser.set_defaults(func=lambda args: run_train(load_config(args.config)))

    # LINE-BY-LINE: `eval_parser`에 `subparsers.add_parser("eval", parents=[common_parser], help="Run evaluation")` 결과를 저장합니다. 의미/사용: `eval_parser` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    eval_parser = subparsers.add_parser("eval", parents=[common_parser], help="Run evaluation")
    # LINE-BY-LINE: `eval_parser.set_defaults(func`에 `lambda args: run_eval(load_config(args.config)))` 결과를 저장합니다. 의미/사용: `set_defaults(func` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    eval_parser.set_defaults(func=lambda args: run_eval(load_config(args.config)))

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
