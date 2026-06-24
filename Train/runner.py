"""학습/평가 러너.

현재 역할:
- config -> 환경 생성
- 알고리즘 선택 및 학습 실행
- 저장된 checkpoint가 있으면 deterministic 평가
- checkpoint가 없으면 명시적 heuristic baseline 평가

중요:
- checkpoint가 있는데 로딩/평가가 실패하면 휴리스틱으로 조용히 넘어가지 않는다.
- 오류 원인을 출력하고 예외를 다시 발생시킨다.
"""

# LINE-BY-LINE: `Agent.heuristics` 모듈에서 `select_action_by_rule`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Agent.heuristics import select_action_by_rule
# LINE-BY-LINE: `Environment.environment` 모듈에서 `CuttingShopEnvironment`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.environment import CuttingShopEnvironment
# LINE-BY-LINE: `Utils.io` 모듈에서 config 기반 scenario loader를 가져옵니다. YAML 또는 원본 Excel/CSV 입력을 같은 환경 dict로 바꿉니다.
from Utils.io import load_scenario_for_config
# LINE-BY-LINE: `.algorithm` 모듈에서 `build_trainer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .algorithm import build_trainer
# LINE-BY-LINE: `.algorithm.common` 모듈에서 `build_model, collect_episode, get_device`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .algorithm.common import build_model, collect_episode, get_device


# LINE-BY-LINE: `_build_env_from_config(config)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def _build_env_from_config(config):
    """config에서 scenario를 읽어 환경 객체를 만듭니다.

    train / eval 모두 동일한 환경 생성 함수를 사용합니다.
    그래서 나중에 환경 입력 형식이 바뀌더라도
    여기만 고치면 train/eval 전체가 같이 따라갑니다.
    """

    scenario = load_scenario_for_config(config)
    # LINE-BY-LINE: 호출자에게 `CuttingShopEnvironment(config=config, scenario=scenario)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return CuttingShopEnvironment(config=config, scenario=scenario)


# LINE-BY-LINE: `run_train(config)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def run_train(config):
    """선택된 알고리즘으로 학습을 실행합니다."""

    # LINE-BY-LINE: `env`에 `_build_env_from_config(config)` 결과를 저장합니다. 의미/사용: `env`는 CuttingShopEnvironment 객체입니다. reset/step/run_with_policy를 제공합니다.
    env = _build_env_from_config(config)
    # LINE-BY-LINE: 콘솔에 `print("[train]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[train]")
    # LINE-BY-LINE: 콘솔에 `print(f"- algorithm: {config['train']['algorithm']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- algorithm: {config['train']['algorithm']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- episodes: {config['train']['episodes']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- episodes: {config['train']['episodes']}")

    # 알고리즘 이름에 따라 PPO / REINFORCE / Self-labeling trainer를 고릅니다.
    # LINE-BY-LINE: `trainer`에 `build_trainer(config["train"]["algorithm"])` 결과를 저장합니다. 의미/사용: `trainer` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trainer = build_trainer(config["train"]["algorithm"])
    # LINE-BY-LINE: `trainer.train(env, config)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    train_result = trainer.train(env, config)
    if isinstance(train_result, dict) and train_result.get("report_paths"):
        print(f"- training_metrics_csv: {train_result['report_paths']['metrics_csv']}")
        print(f"- training_summary_json: {train_result['report_paths']['summary_json']}")
        print(f"- training_report_html: {train_result['report_paths']['report_html']}")

    # 학습 후에는 기본 휴리스틱도 한 번 돌려서
    # 환경이 정상적으로 끝까지 실행되는지 같이 확인합니다.
    # LINE-BY-LINE: `summary`에 `env.run_with_policy(` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = env.run_with_policy(
        # LINE-BY-LINE: `lambda candidates, simulation: select_action_by_rule(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        lambda candidates, simulation: select_action_by_rule(
            # LINE-BY-LINE: `select_action_by_rule(...)` 호출에 `candidates` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            candidates,
            # LINE-BY-LINE: `select_action_by_rule(...)` 호출에 `simulation` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            simulation,
            # LINE-BY-LINE: `select_action_by_rule(...)` 호출에 `config["agent"]["default_heuristic"]` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
            config["agent"]["default_heuristic"],
        )
    )
    # LINE-BY-LINE: 콘솔에 `print(f"- warmup_scheduled_jobs: {summary['scheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- warmup_scheduled_jobs: {summary['scheduled_jobs']}")


# LINE-BY-LINE: `run_eval(config)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def run_eval(config):
    """학습된 정책 또는 휴리스틱을 평가합니다.

    우선순위:
    1. output/{algorithm}_latest.pt checkpoint가 있으면 정책 평가
    2. checkpoint가 없으면 기본 휴리스틱 baseline 평가

    실패 원칙:
    - checkpoint 파일이 있는데 torch load/model eval이 실패하면 즉시 실패한다.
    - 실패한 학습 정책을 휴리스틱 결과로 대체하지 않는다.
    """

    # LINE-BY-LINE: `env`에 `_build_env_from_config(config)` 결과를 저장합니다. 의미/사용: `env`는 CuttingShopEnvironment 객체입니다. reset/step/run_with_policy를 제공합니다.
    env = _build_env_from_config(config)
    # LINE-BY-LINE: `algorithm`에 `config["train"]["algorithm"]` 결과를 저장합니다. 의미/사용: `algorithm` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    algorithm = config["train"]["algorithm"]
    # LINE-BY-LINE: `checkpoint_path`에 `f"{config['paths']['output_dir']}/{algorithm}_latest.pt"` 결과를 저장합니다. 의미/사용: `checkpoint_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    checkpoint_path = f"{config['paths']['output_dir']}/{algorithm}_latest.pt"

    # LINE-BY-LINE: `os` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
    import os
    # LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
    import torch

    # LINE-BY-LINE: 조건 `os.path.exists(checkpoint_path)`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if os.path.exists(checkpoint_path):
        # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
        try:
            # LINE-BY-LINE: `device`에 `get_device(config)` 결과를 저장합니다. 의미/사용: `device`는 torch 연산 장치입니다. 예: cpu 또는 cuda.
            device = get_device(config)
            # LINE-BY-LINE: `model`에 `build_model(env, config, device)` 결과를 저장합니다. 의미/사용: `model`는 policy-value neural network입니다. observation을 logits/value로 바꿉니다.
            model = build_model(env, config, device)
            # LINE-BY-LINE: 실패 가능성이 있는 작업을 실행합니다. 사용: 실패 시 except에서 원인을 출력하고 raise하기 위한 보호 구간입니다.
            try:
                # LINE-BY-LINE: `state_dict`에 `torch.load(checkpoint_path, map_location=device, weights_only=True)` 결과를 저장합니다. 의미/사용: `state_dict` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            # LINE-BY-LINE: `except TypeError:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
            except TypeError:
                # LINE-BY-LINE: `state_dict`에 `torch.load(checkpoint_path, map_location=device)` 결과를 저장합니다. 의미/사용: `state_dict` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                state_dict = torch.load(checkpoint_path, map_location=device)
            # LINE-BY-LINE: `model.load_state_dict(state_dict)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            model.load_state_dict(state_dict)
            # LINE-BY-LINE: `model.eval()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            model.eval()

            # eval은 deterministic=True로 두어
            # 같은 checkpoint면 같은 행동을 내도록 합니다.
            # LINE-BY-LINE: `_, summary` 여러 변수에 `collect_episode(env, model, device, deterministic=True, temperature=1.0)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            _, summary = collect_episode(env, model, device, deterministic=True, temperature=1.0)
            # LINE-BY-LINE: 콘솔에 `print("[eval]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print("[eval]")
            # LINE-BY-LINE: 콘솔에 `print(f"- mode: trained_policy")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- mode: trained_policy")
            # LINE-BY-LINE: 콘솔에 `print(f"- algorithm: {algorithm}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- algorithm: {algorithm}")
            # LINE-BY-LINE: 콘솔에 `print(f"- checkpoint: {checkpoint_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- checkpoint: {checkpoint_path}")
            # LINE-BY-LINE: 콘솔에 `print(f"- makespan: {summary['makespan']:.3f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- makespan: {summary['makespan']:.3f}")
            # LINE-BY-LINE: 콘솔에 `print(f"- load_imbalance: {summary['load_imbalance']:.3f}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- load_imbalance: {summary['load_imbalance']:.3f}")
            # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {summary['scheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
            # LINE-BY-LINE: 현재 함수를 여기서 종료합니다. 사용: 더 이상 처리할 필요가 없을 때 빠져나갑니다.
            return
        # LINE-BY-LINE: `except Exception as exc:` 예외를 잡습니다. 사용: 오류 원인을 print한 뒤 RuntimeError 등으로 연결합니다.
        except Exception as exc:
            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: 문자열 값 `"[ERROR][runner.run_eval] "`를 `print(...)` 호출/컬렉션에 넣습니다. 사용: option, event type, 컬럼명 또는 출력 문구로 쓰입니다.
                "[ERROR][runner.run_eval] "
                # LINE-BY-LINE: `f"cause`에 `checkpoint_evaluation_failed checkpoint={checkpoint_path} error={exc}"` 결과를 저장합니다. 의미/사용: `f"cause` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"cause=checkpoint_evaluation_failed checkpoint={checkpoint_path} error={exc}"
            )
            # LINE-BY-LINE: `RuntimeError("checkpoint evaluation failed") from exc` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
            raise RuntimeError("checkpoint evaluation failed") from exc

    # checkpoint가 없을 때만 명시적으로 휴리스틱 baseline을 실행한다.
    # 이것은 오류 대체가 아니라 비교 기준 산출이다.
    # LINE-BY-LINE: `heuristic`에 `config["agent"]["default_heuristic"]` 결과를 저장합니다. 의미/사용: `heuristic` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    heuristic = config["agent"]["default_heuristic"]
    # LINE-BY-LINE: `summary`에 `env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, ...` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = env.run_with_policy(lambda candidates, simulation: select_action_by_rule(candidates, simulation, heuristic))

    # LINE-BY-LINE: 콘솔에 `print("[eval]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print("[eval]")
    # LINE-BY-LINE: 콘솔에 `print(f"- mode: heuristic_baseline_no_checkpoint")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- mode: heuristic_baseline_no_checkpoint")
    # LINE-BY-LINE: 콘솔에 `print(f"- heuristic: {heuristic}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- heuristic: {heuristic}")
    # LINE-BY-LINE: 콘솔에 `print(f"- scheduled_jobs: {summary['scheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- scheduled_jobs: {summary['scheduled_jobs']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- unscheduled_jobs: {summary['unscheduled_jobs']}")
    # LINE-BY-LINE: 콘솔에 `print(f"- machine_loads: {summary['machine_loads']}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
    print(f"- machine_loads: {summary['machine_loads']}")
