"""Self-labeling 구현.

아이디어:
- 현재 정책으로 여러 개 rollout을 샘플링
- 그중 objective가 가장 좋은 rollout을 선택
- 그 rollout의 action sequence를 pseudo-label로 사용
- 지도학습처럼 cross-entropy로 정책을 업데이트
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .common import build_model, collect_episode, get_device, save_checkpoint


class SelfLabelingTrainer:
    name = "self_labeling"

    def _objective_score(self, summary: Dict, config: Dict) -> float:
        """낮을수록 좋은 objective를 계산합니다."""

        return summary["makespan"] + config["reward"]["load_balance_weight"] * summary["load_imbalance"]

    def train(self, env, config: Dict):
        device = get_device(config)
        model = build_model(env, config, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["self_labeling"]["lr"]))

        rollout_samples = int(config["algorithm"]["self_labeling"]["rollout_samples"])
        imitation_epochs = int(config["algorithm"]["self_labeling"]["imitation_epochs"])
        sample_temperature = float(config["algorithm"]["self_labeling"]["sample_temperature"])
        episodes = int(config["train"]["episodes"])

        print("[self_labeling]")
        print(f"- device: {device}")
        print(f"- episodes: {episodes}")
        print(f"- rollout_samples: {rollout_samples}")

        for episode in range(episodes):
            rollout_bank: List[Tuple[List, Dict]] = []

            for _ in range(rollout_samples):
                # 같은 문제를 여러 번 rollout해서
                # "어떤 action sequence가 가장 좋았는지"를 찾습니다.
                transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=sample_temperature)
                if transitions:
                    rollout_bank.append((transitions, summary))

            if not rollout_bank:
                continue

            best_transitions, best_summary = min(
                rollout_bank,
                key=lambda item: self._objective_score(item[1], config),
            )

            for _ in range(imitation_epochs):
                total_loss = torch.zeros((), dtype=torch.float32, device=device)
                for transition in best_transitions:
                    # best rollout에서 실제로 선택했던 action index를
                    # 정답 label처럼 사용합니다.
                    output = model(transition.observation, device)
                    logits = output["logits"].unsqueeze(0)
                    target = torch.tensor([transition.action_index], dtype=torch.long, device=device)
                    total_loss = total_loss + F.cross_entropy(logits, target)

                total_loss = total_loss / len(best_transitions)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            print(
                f"episode={episode + 1} best_makespan={best_summary['makespan']:.3f} "
                f"objective={self._objective_score(best_summary, config):.3f} "
                f"imitation_loss={float(total_loss.item()):.4f}"
            )

        checkpoint_path = save_checkpoint(model, config, self.name)
        print(f"- checkpoint: {checkpoint_path}")
