"""알고리즘 공통 유틸.

이 파일은 PPO / REINFORCE / Self-labeling이 공통으로 쓰는 함수를 모아둔 곳입니다.
예를 들어:
- 모델 생성
- 한 episode rollout 수집
- action sampling
- checkpoint 저장
"""

# LINE-BY-LINE: `copy` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import copy
# LINE-BY-LINE: `dataclasses` 모듈에서 `dataclass`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from dataclasses import dataclass
# LINE-BY-LINE: `pathlib` 모듈에서 `Path`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from pathlib import Path
# LINE-BY-LINE: `typing` 모듈에서 `Dict, List, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List, Tuple

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn as nn` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn as nn
# LINE-BY-LINE: `torch.distributions` 모듈에서 `Categorical`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from torch.distributions import Categorical

# LINE-BY-LINE: `Environment.reward` 모듈에서 `compute_load_imbalance`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Environment.reward import compute_load_imbalance
# LINE-BY-LINE: `Train.network` 모듈에서 `CandidateGraphPolicyValueNet, build_tensor_observation`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from Train.network import CandidateGraphPolicyValueNet, build_tensor_observation


# LINE-BY-LINE: `@dataclass`는 아래 class에 자동 `__init__`, 비교/표현 메서드를 만들어 줍니다. 예: `Job(...)`를 바로 생성 가능.
@dataclass
# LINE-BY-LINE: `Transition` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class Transition:
    """학습에 필요한 step 1개 기록."""
    # LINE-BY-LINE: `observation`를 `Dict` 타입으로 선언합니다. 의미/사용: `Transition.observation` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    observation: Dict
    # LINE-BY-LINE: `action_index`를 `int` 타입으로 선언합니다. 의미/사용: `Transition.action_index` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    action_index: int
    # LINE-BY-LINE: `action_id`를 `str` 타입으로 선언합니다. 의미/사용: `Transition.action_id` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    action_id: str
    # LINE-BY-LINE: `reward`를 `float` 타입으로 선언합니다. 의미/사용: `Transition.reward` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    reward: float
    # LINE-BY-LINE: `done`를 `bool` 타입으로 선언합니다. 의미/사용: `Transition.done` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    done: bool
    # LINE-BY-LINE: `log_prob`를 `float` 타입으로 선언합니다. 의미/사용: `Transition.log_prob` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    log_prob: float
    # LINE-BY-LINE: `value`를 `float` 타입으로 선언합니다. 의미/사용: `Transition.value` 필드/속성입니다. 사용: Transition 객체를 만들거나 이후 로직에서 참조합니다.
    value: float


# LINE-BY-LINE: `get_device(config: Dict)` 함수를 정의합니다. 반환 타입: `torch.device`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def get_device(config: Dict) -> torch.device:
    """config의 device 설정을 보고 실제 torch device를 선택한다."""

    # LINE-BY-LINE: `requested`에 `config.get("device", "cpu")` 결과를 저장합니다. 의미/사용: `requested` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    requested = config.get("device", "cpu")
    # LINE-BY-LINE: 조건 `requested == "cuda" and torch.cuda.is_available()`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if requested == "cuda" and torch.cuda.is_available():
        # LINE-BY-LINE: 호출자에게 `torch.device("cuda")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return torch.device("cuda")
    # LINE-BY-LINE: 호출자에게 `torch.device("cpu")`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return torch.device("cpu")


# LINE-BY-LINE: `infer_input_dims(env, device: torch.device)` 함수를 정의합니다. 반환 타입: `Tuple[int, int]`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def infer_input_dims(env, device: torch.device) -> Tuple[int, int]:
    """환경 observation을 한 번 읽어 네트워크 입력 차원을 추론합니다."""

    # LINE-BY-LINE: `observation, _` 여러 변수에 `env.reset()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    observation, _ = env.reset()
    # LINE-BY-LINE: `tensor_obs`에 `build_tensor_observation(observation, device)` 결과를 저장합니다. 의미/사용: `tensor_obs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    tensor_obs = build_tensor_observation(observation, device)
    # LINE-BY-LINE: `candidate_input_dim`에 `int(tensor_obs.candidate_features.size(-1))` 결과를 저장합니다. 의미/사용: `candidate_input_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    candidate_input_dim = int(tensor_obs.candidate_features.size(-1))
    # LINE-BY-LINE: `env_input_dim`에 `int(tensor_obs.env_features.size(-1))` 결과를 저장합니다. 의미/사용: `env_input_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    env_input_dim = int(tensor_obs.env_features.size(-1))
    # LINE-BY-LINE: 호출자에게 `candidate_input_dim, env_input_dim`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return candidate_input_dim, env_input_dim


# LINE-BY-LINE: `build_model(env, config: Dict, device: torch.device)` 함수를 정의합니다. 반환 타입: `CandidateGraphPolicyValueNet`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def build_model(env, config: Dict, device: torch.device) -> CandidateGraphPolicyValueNet:
    """환경 observation 차원을 추론한 뒤 policy-value network를 생성한다."""

    # LINE-BY-LINE: `candidate_input_dim, env_input_dim` 여러 변수에 `infer_input_dims(env, device)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    candidate_input_dim, env_input_dim = infer_input_dims(env, device)
    # LINE-BY-LINE: `hidden_dim`에 `int(config["network"]["hidden_dim"])` 결과를 저장합니다. 의미/사용: `hidden_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    hidden_dim = int(config["network"]["hidden_dim"])
    # LINE-BY-LINE: `num_gnn_layers`에 `int(config["network"]["num_gnn_layers"])` 결과를 저장합니다. 의미/사용: `num_gnn_layers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    num_gnn_layers = int(config["network"]["num_gnn_layers"])
    # LINE-BY-LINE: `model`에 `CandidateGraphPolicyValueNet(` 결과를 저장합니다. 의미/사용: `model`는 policy-value neural network입니다. observation을 logits/value로 바꿉니다.
    model = CandidateGraphPolicyValueNet(
        # LINE-BY-LINE: `candidate_input_dim`에 `candidate_input_dim` 결과를 저장합니다. 의미/사용: `candidate_input_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        candidate_input_dim=candidate_input_dim,
        # LINE-BY-LINE: `env_input_dim`에 `env_input_dim` 결과를 저장합니다. 의미/사용: `env_input_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        env_input_dim=env_input_dim,
        # LINE-BY-LINE: `hidden_dim`에 `hidden_dim` 결과를 저장합니다. 의미/사용: `hidden_dim` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        hidden_dim=hidden_dim,
        # LINE-BY-LINE: `num_gnn_layers`에 `num_gnn_layers` 결과를 저장합니다. 의미/사용: `num_gnn_layers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        num_gnn_layers=num_gnn_layers,
    )
    # LINE-BY-LINE: 호출자에게 `model.to(device)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return model.to(device)


# LINE-BY-LINE: `select_action(model: nn.Module, observation: Dict, device: torch.device, deterministic: boo...)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다. 예: action_id는 `job_001@PLS21` 또는 `open_batch:...` 형태.
def select_action(model: nn.Module, observation: Dict, device: torch.device, deterministic: bool = False, temperature: float = 1.0):
    """현재 observation에서 action 1개를 고릅니다."""

    # LINE-BY-LINE: `output`에 `model(observation, device)` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output = model(observation, device)
    # LINE-BY-LINE: `logits`에 `output["logits"]` 결과를 저장합니다. 의미/사용: `logits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    logits = output["logits"]
    # LINE-BY-LINE: 조건 `logits.numel() == 0`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if logits.numel() == 0:
        # LINE-BY-LINE: 호출자에게 `None`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return None

    # LINE-BY-LINE: `scaled_logits`에 `logits / max(temperature, 1e-6)` 결과를 저장합니다. 의미/사용: `scaled_logits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    scaled_logits = logits / max(temperature, 1e-6)
    # logits -> categorical distribution으로 바꿉니다.
    # LINE-BY-LINE: `distribution`에 `Categorical(logits=scaled_logits)` 결과를 저장합니다. 의미/사용: `distribution` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    distribution = Categorical(logits=scaled_logits)
    # LINE-BY-LINE: 조건 `deterministic`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if deterministic:
        # LINE-BY-LINE: `action_index`에 `int(torch.argmax(scaled_logits).item())` 결과를 저장합니다. 의미/사용: `action_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_index = int(torch.argmax(scaled_logits).item())
    # LINE-BY-LINE: 앞 조건들이 모두 거짓일 때 실행되는 기본 분기입니다. 사용: 대체 처리 경로를 명시적으로 분리합니다.
    else:
        # LINE-BY-LINE: `action_index`에 `int(distribution.sample().item())` 결과를 저장합니다. 의미/사용: `action_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        action_index = int(distribution.sample().item())

    # LINE-BY-LINE: `log_prob`에 `float(distribution.log_prob(torch.tensor(action_index, device=device)).item())` 결과를 저장합니다. 의미/사용: `log_prob` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    log_prob = float(distribution.log_prob(torch.tensor(action_index, device=device)).item())
    # LINE-BY-LINE: `value`에 `float(output["value"].item())` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    value = float(output["value"].item())
    # LINE-BY-LINE: `action_id`에 `output["action_ids"][action_index]` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    action_id = output["action_ids"][action_index]
    # LINE-BY-LINE: 호출자에게 `action_index, action_id, log_prob, value`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return action_index, action_id, log_prob, value


# LINE-BY-LINE: `collect_episode(env, model: nn.Module, device: torch.device, deterministic: bool = False, tem...)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def collect_episode(env, model: nn.Module, device: torch.device, deterministic: bool = False, temperature: float = 1.0):
    """환경을 끝까지 실행하며 transition 목록을 수집합니다."""

    # LINE-BY-LINE: `observation, _` 여러 변수에 `env.reset()` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
    observation, _ = env.reset()
    # LINE-BY-LINE: `transitions` 변수에 `[]` 결과를 저장합니다. 의미: `transitions` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    transitions: List[Transition] = []
    # LINE-BY-LINE: `total_reward`에 `0.0` 결과를 저장합니다. 의미/사용: `total_reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    total_reward = 0.0

    # LINE-BY-LINE: `terminated`에 `False` 결과를 저장합니다. 의미/사용: `terminated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    terminated = False
    # LINE-BY-LINE: `truncated`에 `False` 결과를 저장합니다. 의미/사용: `truncated` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    truncated = False
    # LINE-BY-LINE: 조건 `not (terminated or truncated)`가 유지되는 동안 반복합니다. 사용: episode 진행 또는 시간 전진 루프에 씁니다.
    while not (terminated or truncated):
        # LINE-BY-LINE: `selected`에 `select_action(model, observation, device, deterministic=deterministic, temperature=temperature)` 결과를 저장합니다. 의미/사용: `selected`는 휴리스틱/정책이 실제로 선택한 action 후보입니다.
        selected = select_action(model, observation, device, deterministic=deterministic, temperature=temperature)
        # LINE-BY-LINE: 조건 `selected is None`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
        if selected is None:
            # LINE-BY-LINE: 현재 반복문을 종료합니다. 사용: 더 이상 탐색/진행할 필요가 없을 때 중단.
            break

        # LINE-BY-LINE: `action_index, action_id, log_prob, value` 여러 변수에 `selected` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        action_index, action_id, log_prob, value = selected
        # 여기서 실제로 환경이 1 step 전진합니다.
        # LINE-BY-LINE: `next_observation, reward, terminated, truncated, _` 여러 변수에 `env.step(action_id)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
        next_observation, reward, terminated, truncated, _ = env.step(action_id)
        # LINE-BY-LINE: `transitions.append(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        transitions.append(
            # LINE-BY-LINE: `Transition(`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
            Transition(
                # LINE-BY-LINE: `observation`에 `copy.deepcopy(observation)` 결과를 저장합니다. 의미/사용: `observation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                observation=copy.deepcopy(observation),
                # LINE-BY-LINE: `action_index`에 `action_index` 결과를 저장합니다. 의미/사용: `action_index` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                action_index=action_index,
                # LINE-BY-LINE: `action_id`에 `action_id` 결과를 저장합니다. 의미/사용: `action_id` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                action_id=action_id,
                # LINE-BY-LINE: `reward`에 `float(reward)` 결과를 저장합니다. 의미/사용: `reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                reward=float(reward),
                # LINE-BY-LINE: `done`에 `bool(terminated or truncated)` 결과를 저장합니다. 의미/사용: `done` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                done=bool(terminated or truncated),
                # LINE-BY-LINE: `log_prob`에 `log_prob` 결과를 저장합니다. 의미/사용: `log_prob` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                log_prob=log_prob,
                # LINE-BY-LINE: `value`에 `value` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                value=value,
            )
        )
        # LINE-BY-LINE: `total_reward` 값을 `float(reward)` 기준으로 누적/증가합니다. 의미/사용: `total_reward` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        total_reward += float(reward)
        # LINE-BY-LINE: `observation`에 `next_observation` 결과를 저장합니다. 의미/사용: `observation` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        observation = next_observation

    # LINE-BY-LINE: `summary`에 `{` 결과를 저장합니다. 의미/사용: `summary`는 실행 결과 요약 dict입니다. 예: scheduled_jobs, makespan, event_log_count.
    summary = {
        # LINE-BY-LINE: `collect_episode`에서 반환/저장할 dict의 `total_reward` 키에 `total_reward` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "total_reward": total_reward,
        # LINE-BY-LINE: 실행 summary의 `makespan` 항목에 `env.simulation.get_makespan()`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "makespan": env.simulation.get_makespan(),
        # LINE-BY-LINE: `collect_episode`에서 반환/저장할 dict의 `load_imbalance` 키에 `float(compute_load_imbalance(env.simulation.state.machine_loads))` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "load_imbalance": float(compute_load_imbalance(env.simulation.state.machine_loads)),
        # LINE-BY-LINE: 실행 summary의 `scheduled_jobs` 항목에 `len(env.simulation.state.schedule)`를 저장합니다. 사용: CLI 출력과 검증 결과 비교.
        "scheduled_jobs": len(env.simulation.state.schedule),
    }
    # LINE-BY-LINE: 호출자에게 `transitions, summary`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return transitions, summary


# LINE-BY-LINE: `compute_discounted_returns(rewards: List[float], gamma: float)` 함수를 정의합니다. 반환 타입: `List[float]`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """reward 리스트를 뒤에서부터 누적해 discounted return을 계산합니다."""

    # LINE-BY-LINE: `returns` 변수에 `[]` 결과를 저장합니다. 의미: `returns` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    returns: List[float] = []
    # LINE-BY-LINE: `running_return`에 `0.0` 결과를 저장합니다. 의미/사용: `running_return` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    running_return = 0.0
    # LINE-BY-LINE: `reward in reversed(rewards)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for reward in reversed(rewards):
        # LINE-BY-LINE: `running_return`에 `reward + gamma * running_return` 결과를 저장합니다. 의미/사용: `running_return` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        running_return = reward + gamma * running_return
        # LINE-BY-LINE: `returns.append(running_return)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        returns.append(running_return)
    # LINE-BY-LINE: `returns.reverse()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    returns.reverse()
    # LINE-BY-LINE: 호출자에게 `returns`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return returns


# LINE-BY-LINE: `save_checkpoint(model: nn.Module, config: Dict, algorithm_name: str)` 함수를 정의합니다. 반환 타입: `str`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def save_checkpoint(model: nn.Module, config: Dict, algorithm_name: str) -> str:
    """학습된 모델 가중치를 output_dir에 저장하고 경로를 반환한다."""

    # LINE-BY-LINE: `output_dir`에 `Path(config["paths"]["output_dir"])` 결과를 저장합니다. 의미/사용: `output_dir`는 결과 파일을 저장할 디렉터리입니다. 예: `output/playback_np_100`.
    output_dir = Path(config["paths"]["output_dir"])
    # LINE-BY-LINE: `output_dir.mkdir(parents`에 `True, exist_ok=True)` 결과를 저장합니다. 의미/사용: `mkdir(parents` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # LINE-BY-LINE: `checkpoint_path`에 `output_dir / f"{algorithm_name}_latest.pt"` 결과를 저장합니다. 의미/사용: `checkpoint_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    checkpoint_path = output_dir / f"{algorithm_name}_latest.pt"
    # LINE-BY-LINE: `torch.save(model.state_dict(), checkpoint_path)`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    torch.save(model.state_dict(), checkpoint_path)
    # LINE-BY-LINE: 호출자에게 `str(checkpoint_path)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return str(checkpoint_path)
