"""알고리즘 공통 유틸.

이 파일은 PPO / REINFORCE / Self-labeling이 공통으로 쓰는 함수를 모아둔 곳입니다.
예를 들어:
- 모델 생성
- 한 episode rollout 수집
- action sampling
- checkpoint 저장
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from Environment.reward import compute_load_imbalance
from Train.network import CandidateGraphPolicyValueNet, build_tensor_observation


@dataclass
class Transition:
    """학습에 필요한 step 1개 기록."""
    observation: Dict
    action_index: int
    action_id: str
    reward: float
    done: bool
    log_prob: float
    value: float


def get_device(config: Dict) -> torch.device:
    requested = config.get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_input_dims(env, device: torch.device) -> Tuple[int, int]:
    """환경 observation을 한 번 읽어 네트워크 입력 차원을 추론합니다."""

    observation, _ = env.reset()
    tensor_obs = build_tensor_observation(observation, device)
    candidate_input_dim = int(tensor_obs.candidate_features.size(-1))
    env_input_dim = int(tensor_obs.env_features.size(-1))
    return candidate_input_dim, env_input_dim


def build_model(env, config: Dict, device: torch.device) -> CandidateGraphPolicyValueNet:
    candidate_input_dim, env_input_dim = infer_input_dims(env, device)
    hidden_dim = int(config["network"]["hidden_dim"])
    num_gnn_layers = int(config["network"]["num_gnn_layers"])
    model = CandidateGraphPolicyValueNet(
        candidate_input_dim=candidate_input_dim,
        env_input_dim=env_input_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
    )
    return model.to(device)


def select_action(model: nn.Module, observation: Dict, device: torch.device, deterministic: bool = False, temperature: float = 1.0):
    """현재 observation에서 action 1개를 고릅니다."""

    output = model(observation, device)
    logits = output["logits"]
    if logits.numel() == 0:
        return None

    scaled_logits = logits / max(temperature, 1e-6)
    # logits -> categorical distribution으로 바꿉니다.
    distribution = Categorical(logits=scaled_logits)
    if deterministic:
        action_index = int(torch.argmax(scaled_logits).item())
    else:
        action_index = int(distribution.sample().item())

    log_prob = float(distribution.log_prob(torch.tensor(action_index, device=device)).item())
    value = float(output["value"].item())
    action_id = output["action_ids"][action_index]
    return action_index, action_id, log_prob, value


def collect_episode(env, model: nn.Module, device: torch.device, deterministic: bool = False, temperature: float = 1.0):
    """환경을 끝까지 실행하며 transition 목록을 수집합니다."""

    observation, _ = env.reset()
    transitions: List[Transition] = []
    total_reward = 0.0

    terminated = False
    truncated = False
    while not (terminated or truncated):
        selected = select_action(model, observation, device, deterministic=deterministic, temperature=temperature)
        if selected is None:
            break

        action_index, action_id, log_prob, value = selected
        # 여기서 실제로 환경이 1 step 전진합니다.
        next_observation, reward, terminated, truncated, _ = env.step(action_id)
        transitions.append(
            Transition(
                observation=copy.deepcopy(observation),
                action_index=action_index,
                action_id=action_id,
                reward=float(reward),
                done=bool(terminated or truncated),
                log_prob=log_prob,
                value=value,
            )
        )
        total_reward += float(reward)
        observation = next_observation

    summary = {
        "total_reward": total_reward,
        "makespan": env.simulation.get_makespan(),
        "load_imbalance": float(compute_load_imbalance(env.simulation.state.machine_loads)),
        "scheduled_jobs": len(env.simulation.state.schedule),
    }
    return transitions, summary


def compute_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    """reward 리스트를 뒤에서부터 누적해 discounted return을 계산합니다."""

    returns: List[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.append(running_return)
    returns.reverse()
    return returns


def save_checkpoint(model: nn.Module, config: Dict, algorithm_name: str) -> str:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{algorithm_name}_latest.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return str(checkpoint_path)
