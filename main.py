"""프로젝트 공통 실행 진입점.

초보자 기준 실행 예시:
  python3 main.py show-config --config config.yaml
  python3 main.py simulate --config config.yaml
  python3 main.py trace --config config.yaml
  python3 main.py train --config config.yaml
  python3 main.py eval --config config.yaml
"""

import argparse

from Agent.heuristics import select_action_by_rule
from Environment.environment import CuttingShopEnvironment
from Train.runner import run_eval, run_train
from Utils.config import load_config
from Utils.io import load_scenario
from Utils.scenario_generator import generate_scenario_from_template, save_scenario
from Utils.tact_time import TactTimeEstimator, TactTimeRecord


def build_environment(config_path: str) -> CuttingShopEnvironment:
    """config와 scenario를 읽어서 환경 객체를 만듭니다.

    이 함수 하나만 호출하면
    - config 로딩
    - scenario YAML 로딩
    - environment 객체 생성
    가 한 번에 끝납니다.
    """

    config = load_config(config_path)
    scenario = load_scenario(config["paths"]["scenario_path"])
    return CuttingShopEnvironment(config=config, scenario=scenario)


def command_show_config(args: argparse.Namespace) -> None:
    """현재 활성 config의 핵심 내용만 요약해서 보여줍니다."""

    config = load_config(args.config)
    category_enabled = [name for name, enabled in config["constraints"]["categories"].items() if enabled]
    hard_enabled = [name for name, enabled in config["constraints"]["hard_enabled"].items() if enabled]
    soft_enabled = [name for name, enabled in config["constraints"]["soft_enabled"].items() if enabled]

    print("[config]")
    print(f"- scenario_path: {config['paths']['scenario_path']}")
    print(f"- action_mode: {config['action_space']['mode']}")
    print(f"- default_heuristic: {config['agent']['default_heuristic']}")
    print(f"- enabled_categories: {', '.join(category_enabled)}")
    print(f"- hard_constraints: {', '.join(hard_enabled)}")
    print(f"- soft_constraints: {', '.join(soft_enabled)}")
    print(f"- holidays_off_enabled: {config['calendar'].get('enable_holidays_off', False)}")
    print(f"- half_day_off_enabled: {config['calendar'].get('enable_half_day_off', False)}")
    print(f"- lunch_break_enabled: {config['calendar'].get('enable_lunch_break', False)}")
    print(f"- machine_operating_windows_enabled: {config['calendar'].get('enable_machine_operating_windows', False)}")
    print(f"- machine_shutdown_windows_enabled: {config['calendar'].get('enable_machine_shutdown_windows', False)}")
    print(f"- machine_breakdowns_enabled: {config['calendar'].get('enable_machine_breakdowns', False)}")
    print(f"- operation_time_adjustment_enabled: {config['calendar'].get('enable_operation_time_adjustment', False)}")
    print(f"- family_changeover_enabled: {config.get('setup', {}).get('enable_family_changeover', False)}")


def command_simulate(args: argparse.Namespace) -> None:
    """휴리스틱 1개로 에피소드를 끝까지 실행합니다."""

    env = build_environment(args.config)
    heuristic_name = args.heuristic or env.config["agent"]["default_heuristic"]
    summary = env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, heuristic_name))

    print("[simulation]")
    print(f"- heuristic: {heuristic_name}")
    print(f"- current_time: {summary['current_time']:.2f}")
    print(f"- makespan: {summary['makespan']:.2f}")
    print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
    print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")
    print(f"- downstream_loads: {summary['downstream_loads']}")
    print(f"- machine_loads: {summary['machine_loads']}")


def command_trace(args: argparse.Namespace) -> None:
    """step 단위로 어떤 action이 선택되는지 상세히 출력합니다."""

    env = build_environment(args.config)
    heuristic_name = args.heuristic or env.config["agent"]["default_heuristic"]
    observation, info = env.reset()

    print("[trace]")
    print(f"- heuristic: {heuristic_name}")
    print(f"- current_time: {observation['current_time']:.2f}")

    step_index = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        candidates = env.get_action_candidates()
        if not candidates:
            break

        selected = select_action_by_rule(candidates, env.simulation, heuristic_name)
        print(
            f"step={step_index} action={selected.action_id} "
            f"job={selected.job.job_id} machine={selected.machine.machine_id} "
            f"minutes={selected.estimated_minutes:.2f} soft_penalty={selected.soft_penalty:.3f}"
        )
        if selected.soft_reasons:
            print(f"  soft_reasons={selected.soft_reasons}")

        observation, reward, terminated, truncated, info = env.step(selected.action_id)
        print(
            f"  -> reward={reward:.3f} current_time={observation['current_time']:.2f} "
            f"remaining_jobs={observation['remaining_job_count']} available_actions={len(observation['available_actions'])}"
        )
        step_index += 1

    print("[trace summary]")
    print(f"- scheduled_jobs: {len(env.simulation.state.schedule)}")
    print(f"- current_time: {env.simulation.state.current_time:.2f}")
    print(f"- makespan: {env.simulation.get_makespan():.2f}")
    print(f"- machine_loads: {env.simulation.state.machine_loads}")


def command_analyze_tact(args: argparse.Namespace) -> None:
    """택트타임 분석을 실행합니다."""

    scenario = load_scenario(load_config(args.config)["paths"]["scenario_path"])
    estimator = TactTimeEstimator()

    # 지금은 실적 데이터가 없으므로 샘플 job을 가짜 record처럼 변환해서 구조만 확인합니다.
    demo_records = []
    for job in scenario["jobs"]:
        demo_records.append(
            TactTimeRecord(
                machine_type="plasma",
                family=job["family"],
                thickness=job["thickness"],
                plate_length=job["plate_length"],
                actual_minutes=float(job["base_stage_minutes"]["setup"] + job["base_stage_minutes"]["cut"] + job["base_stage_minutes"]["finish"]),
            )
        )

    estimator.fit(demo_records)
    print("[tact analysis]")
    print(f"- record_count: {estimator.summary.get('count', 0)}")
    print(f"- average_actual_minutes: {estimator.summary.get('average_actual_minutes', 0.0):.2f}")


def command_generate_scenario(args: argparse.Namespace) -> None:
    """학습용 시나리오를 생성합니다."""

    config = load_config(args.config)
    template = load_scenario(config["paths"]["scenario_path"])
    generated = generate_scenario_from_template(template, duplicate_jobs=args.duplicate_jobs)
    save_scenario(generated, args.output_path)

    print("[scenario generation]")
    print(f"- template_jobs: {len(template['jobs'])}")
    print(f"- generated_jobs: {len(generated['jobs'])}")
    print(f"- output_path: {args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 구성합니다."""

    parser = argparse.ArgumentParser(description="Cutting shop scheduling project")
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--config", default="config.yaml", help="YAML config path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show-config", parents=[common_parser], help="Print active config summary")
    show_parser.set_defaults(func=command_show_config)

    simulate_parser = subparsers.add_parser("simulate", parents=[common_parser], help="Run one heuristic-based scheduling simulation")
    simulate_parser.add_argument("--heuristic", default=None, help="spt / priority / load_balance")
    simulate_parser.set_defaults(func=command_simulate)

    trace_parser = subparsers.add_parser("trace", parents=[common_parser], help="Print step-by-step decision trace")
    trace_parser.add_argument("--heuristic", default=None, help="spt / priority / load_balance")
    trace_parser.set_defaults(func=command_trace)

    tact_parser = subparsers.add_parser("analyze-tact", parents=[common_parser], help="Run tact-time analysis")
    tact_parser.set_defaults(func=command_analyze_tact)

    generate_parser = subparsers.add_parser("generate-scenario", parents=[common_parser], help="Generate training scenario")
    generate_parser.add_argument("--duplicate-jobs", type=int, default=2, help="How many copies of template jobs to make")
    generate_parser.add_argument("--output-path", default="output/generated_scenario.yaml", help="Generated yaml path")
    generate_parser.set_defaults(func=command_generate_scenario)

    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Run training")
    train_parser.set_defaults(func=lambda args: run_train(load_config(args.config)))

    eval_parser = subparsers.add_parser("eval", parents=[common_parser], help="Run evaluation")
    eval_parser.set_defaults(func=lambda args: run_eval(load_config(args.config)))

    return parser


def main() -> None:
    """프로그램 시작점."""

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
