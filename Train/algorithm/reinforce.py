"""REINFORCE 구현.

가장 기본적인 policy gradient 방법입니다.

아이디어:
- rollout을 하나 수집
- 각 step의 log_prob * return으로 loss 계산
- return이 큰 행동의 확률을 높이도록 업데이트
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .common import build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint


class ReinforceTrainer:
    name = "reinforce"

    def train(self, env, config: Dict):
        device = get_device(config)
        model = build_model(env, config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["reinforce"]["lr"]))
        gamma = float(config["algorithm"]["reinforce"]["gamma"])
        entropy_coef = float(config["algorithm"]["reinforce"]["entropy_coef"])
        episodes = int(config["train"]["episodes"])

        print("[reinforce]")
        print(f"- device: {device}")
        print(f"- episodes: {episodes}")

        for episode in range(episodes):
            # 현재 정책으로 에피소드 하나를 끝까지 실행합니다.
            transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=1.0)
            if not transitions:
                continue

            # reward sequence -> discounted return
            returns = compute_discounted_returns([transition.reward for transition in transitions], gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            if returns_tensor.numel() > 1:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)

            loss = torch.zeros((), dtype=torch.float32, device=device)
            entropy_bonus = torch.zeros((), dtype=torch.float32, device=device)

            for step_idx, transition in enumerate(transitions):
                # transition.observation을 다시 모델에 넣어
                # 현재 정책 기준 log_prob를 계산합니다.
                output = model(transition.observation, device)
                distribution = torch.distributions.Categorical(logits=output["logits"])
                action_tensor = torch.tensor(transition.action_index, dtype=torch.long, device=device)
                log_prob = distribution.log_prob(action_tensor)
                entropy_bonus = entropy_bonus + distribution.entropy()
                loss = loss - log_prob * returns_tensor[step_idx]

            loss = loss / len(transitions) - entropy_coef * entropy_bonus / len(transitions)

            # gradient 계산 -> parameter update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print(
                f"episode={episode + 1} reward={summary['total_reward']:.3f} "
                f"makespan={summary['makespan']:.3f} loss={float(loss.item()):.4f}"
            )

        checkpoint_path = save_checkpoint(model, config, self.name)
        print(f"- checkpoint: {checkpoint_path}")
