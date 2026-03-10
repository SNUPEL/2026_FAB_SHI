"""학습/평가 러너.

현재 역할:
- config -> 환경 생성
- 알고리즘 선택 및 학습 실행
- 저장된 checkpoint가 있으면 deterministic 평가
- checkpoint가 없으면 휴리스틱 fallback 평가
"""

from Agent.heuristics import select_action_by_rule
from Environment.environment import CuttingShopEnvironment
from Utils.io import load_scenario
from .algorithm import build_trainer
from .algorithm.common import build_model, collect_episode, get_device


def _build_env_from_config(config):
    """config에서 scenario를 읽어 환경 객체를 만듭니다.

    train / eval 모두 동일한 환경 생성 함수를 사용합니다.
    그래서 나중에 환경 입력 형식이 바뀌더라도
    여기만 고치면 train/eval 전체가 같이 따라갑니다.
    """

    scenario = load_scenario(config["paths"]["scenario_path"])
    return CuttingShopEnvironment(config=config, scenario=scenario)


def run_train(config):
    """선택된 알고리즘으로 학습을 실행합니다."""

    env = _build_env_from_config(config)
    print("[train]")
    print(f"- algorithm: {config['train']['algorithm']}")
    print(f"- episodes: {config['train']['episodes']}")

    # 알고리즘 이름에 따라 PPO / REINFORCE / Self-labeling trainer를 고릅니다.
    trainer = build_trainer(config["train"]["algorithm"])
    trainer.train(env, config)

    # 학습 후에는 기본 휴리스틱도 한 번 돌려서
    # 환경이 정상적으로 끝까지 실행되는지 같이 확인합니다.
    summary = env.run_with_policy(
        lambda candidates, simulation: select_action_by_rule(
            candidates,
            simulation,
            config["agent"]["default_heuristic"],
        )
    )
    print(f"- warmup_scheduled_jobs: {summary['scheduled_jobs']}")


def run_eval(config):
    """학습된 정책 또는 휴리스틱을 평가합니다.

    우선순위:
    1. output/{algorithm}_latest.pt checkpoint가 있으면 정책 평가
    2. 없으면 기본 휴리스틱 평가
    """

    env = _build_env_from_config(config)
    algorithm = config["train"]["algorithm"]
    checkpoint_path = f"{config['paths']['output_dir']}/{algorithm}_latest.pt"

    try:
        import os
        import torch

        if os.path.exists(checkpoint_path):
            device = get_device(config)
            model = build_model(env, config, device)
            try:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            # eval은 deterministic=True로 두어
            # 같은 checkpoint면 같은 행동을 내도록 합니다.
            _, summary = collect_episode(env, model, device, deterministic=True, temperature=1.0)
            print("[eval]")
            print(f"- mode: trained_policy")
            print(f"- algorithm: {algorithm}")
            print(f"- checkpoint: {checkpoint_path}")
            print(f"- makespan: {summary['makespan']:.3f}")
            print(f"- load_imbalance: {summary['load_imbalance']:.3f}")
            print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
            return
    except Exception as exc:
        print(f"[eval] checkpoint evaluation failed: {exc}")

    # checkpoint가 없으면 휴리스틱으로 fallback합니다.
    heuristic = config["agent"]["default_heuristic"]
    summary = env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, heuristic))

    print("[eval]")
    print(f"- mode: heuristic_fallback")
    print(f"- heuristic: {heuristic}")
    print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
    print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")
    print(f"- machine_loads: {summary['machine_loads']}")
