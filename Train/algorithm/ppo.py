"""PPO 구현.

현재 버전은 episode-level PPO의 최소 구현입니다.

구성:
- rollout 수집
- return 계산
- advantage = return - old_value
- clipped policy objective
- value loss
- entropy bonus
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .common import build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint


class PPOTrainer:
    name = "ppo"

    def train(self, env, config: Dict):
        device = get_device(config)
        model = build_model(env, config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["ppo"]["lr"]))

        gamma = float(config["algorithm"]["ppo"]["gamma"])
        clip_eps = float(config["algorithm"]["ppo"]["clip_eps"])
        update_epochs = int(config["algorithm"]["ppo"]["update_epochs"])
        value_coef = float(config["algorithm"]["ppo"]["value_coef"])
        entropy_coef = float(config["algorithm"]["ppo"]["entropy_coef"])
        episodes = int(config["train"]["episodes"])

        print("[ppo]")
        print(f"- device: {device}")
        print(f"- episodes: {episodes}")

        for episode in range(episodes):
            # 현재 정책으로 episode rollout을 수집합니다.
            transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=1.0)
            if not transitions:
                continue

            # 각 step의 누적 return을 계산합니다.
            returns = compute_discounted_returns([transition.reward for transition in transitions], gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            old_log_probs = torch.tensor([transition.log_prob for transition in transitions], dtype=torch.float32, device=device)
            old_values = torch.tensor([transition.value for transition in transitions], dtype=torch.float32, device=device)
            advantages = returns_tensor - old_values
            if advantages.numel() > 1:
                # PPO에서 흔히 쓰는 advantage 정규화입니다.
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            for _ in range(update_epochs):
                total_actor_loss = torch.zeros((), dtype=torch.float32, device=device)
                total_value_loss = torch.zeros((), dtype=torch.float32, device=device)
                total_entropy = torch.zeros((), dtype=torch.float32, device=device)

                for step_idx, transition in enumerate(transitions):
                    # old rollout을 다시 current policy로 평가합니다.
                    output = model(transition.observation, device)
                    distribution = Categorical(logits=output["logits"])
                    action_tensor = torch.tensor(transition.action_index, dtype=torch.long, device=device)
                    new_log_prob = distribution.log_prob(action_tensor)
                    entropy = distribution.entropy()
                    value = output["value"]

                    # ratio = pi_new / pi_old
                    ratio = torch.exp(new_log_prob - old_log_probs[step_idx])
                    unclipped = ratio * advantages[step_idx]
                    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[step_idx]
                    actor_loss = -torch.min(unclipped, clipped)
                    value_loss = F.mse_loss(value, returns_tensor[step_idx])

                    total_actor_loss = total_actor_loss + actor_loss
                    total_value_loss = total_value_loss + value_loss
                    total_entropy = total_entropy + entropy

                total_actor_loss = total_actor_loss / len(transitions)
                total_value_loss = total_value_loss / len(transitions)
                total_entropy = total_entropy / len(transitions)
                loss = total_actor_loss + value_coef * total_value_loss - entropy_coef * total_entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            print(
                f"episode={episode + 1} reward={summary['total_reward']:.3f} "
                f"makespan={summary['makespan']:.3f} actor_loss={float(total_actor_loss.item()):.4f} "
                f"value_loss={float(total_value_loss.item()):.4f}"
            )

        checkpoint_path = save_checkpoint(model, config, self.name)
        print(f"- checkpoint: {checkpoint_path}")
