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

# LINE-BY-LINE: `typing` 모듈에서 `Dict, List`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn.functional as F` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn.functional as F
# LINE-BY-LINE: `torch.distributions` 모듈에서 `Categorical`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from torch.distributions import Categorical

from Utils.training_report import write_training_report
# LINE-BY-LINE: `.common` 모듈에서 `build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .common import build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint


# LINE-BY-LINE: `PPOTrainer` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class PPOTrainer:
    """PPO 학습 루프를 담당하는 trainer."""

    # LINE-BY-LINE: `name`에 `"ppo"` 결과를 저장합니다. 의미/사용: `name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    name = "ppo"

    # LINE-BY-LINE: `train(self, env, config: Dict)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def train(self, env, config: Dict):
        """환경과 config를 받아 PPO episode 학습을 수행한다."""

        # LINE-BY-LINE: `device`에 `get_device(config)` 결과를 저장합니다. 의미/사용: `device`는 torch 연산 장치입니다. 예: cpu 또는 cuda.
        device = get_device(config)
        # LINE-BY-LINE: `model`에 `build_model(env, config, device)` 결과를 저장합니다. 의미/사용: `model`는 policy-value neural network입니다. observation을 logits/value로 바꿉니다.
        model = build_model(env, config, device)
        # LINE-BY-LINE: `optimizer`에 `torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["ppo"]["lr"]))` 결과를 저장합니다. 의미/사용: `optimizer`는 PyTorch optimizer입니다. loss.backward 이후 parameter를 갱신합니다.
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["ppo"]["lr"]))

        # LINE-BY-LINE: `gamma`에 `float(config["algorithm"]["ppo"]["gamma"])` 결과를 저장합니다. 의미/사용: `gamma` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        gamma = float(config["algorithm"]["ppo"]["gamma"])
        # LINE-BY-LINE: `clip_eps`에 `float(config["algorithm"]["ppo"]["clip_eps"])` 결과를 저장합니다. 의미/사용: `clip_eps` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        clip_eps = float(config["algorithm"]["ppo"]["clip_eps"])
        # LINE-BY-LINE: `update_epochs`에 `int(config["algorithm"]["ppo"]["update_epochs"])` 결과를 저장합니다. 의미/사용: `update_epochs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        update_epochs = int(config["algorithm"]["ppo"]["update_epochs"])
        # LINE-BY-LINE: `value_coef`에 `float(config["algorithm"]["ppo"]["value_coef"])` 결과를 저장합니다. 의미/사용: `value_coef` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        value_coef = float(config["algorithm"]["ppo"]["value_coef"])
        # LINE-BY-LINE: `entropy_coef`에 `float(config["algorithm"]["ppo"]["entropy_coef"])` 결과를 저장합니다. 의미/사용: `entropy_coef` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        entropy_coef = float(config["algorithm"]["ppo"]["entropy_coef"])
        # LINE-BY-LINE: `episodes`에 `int(config["train"]["episodes"])` 결과를 저장합니다. 의미/사용: `episodes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        episodes = int(config["train"]["episodes"])
        if episodes <= 0 or update_epochs <= 0:
            print(
                "[ERROR][PPOTrainer.train] "
                f"cause=invalid_training_loop_config episodes={episodes} update_epochs={update_epochs}"
            )
            raise ValueError("episodes and update_epochs must be positive")

        # LINE-BY-LINE: 콘솔에 `print("[ppo]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[ppo]")
        # LINE-BY-LINE: 콘솔에 `print(f"- device: {device}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- device: {device}")
        # LINE-BY-LINE: 콘솔에 `print(f"- episodes: {episodes}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- episodes: {episodes}")
        episode_metrics: List[Dict] = []

        # LINE-BY-LINE: `episode in range(episodes)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for episode in range(episodes):
            # 현재 정책으로 episode rollout을 수집합니다.
            # LINE-BY-LINE: `transitions, summary` 여러 변수에 `collect_episode(env, model, device, deterministic=False, temperature=1.0)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=1.0)
            # LINE-BY-LINE: 조건 `not transitions`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not transitions:
                print(
                    "[ERROR][PPOTrainer.train] "
                    f"cause=empty_rollout episode={episode + 1}"
                )
                raise RuntimeError("PPO rollout produced no transitions")

            # 각 step의 누적 return을 계산합니다.
            # LINE-BY-LINE: `returns`에 `compute_discounted_returns([transition.reward for transition in transitions], gamma)` 결과를 저장합니다. 의미/사용: `returns` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            returns = compute_discounted_returns([transition.reward for transition in transitions], gamma)
            # LINE-BY-LINE: `returns_tensor`에 `torch.tensor(returns, dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `returns_tensor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            # LINE-BY-LINE: `old_log_probs`에 `torch.tensor([transition.log_prob for transition in transitions], dtype=torch.float32, device=dev...` 결과를 저장합니다. 의미/사용: `old_log_probs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            old_log_probs = torch.tensor([transition.log_prob for transition in transitions], dtype=torch.float32, device=device)
            # LINE-BY-LINE: `old_values`에 `torch.tensor([transition.value for transition in transitions], dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `old_values` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            old_values = torch.tensor([transition.value for transition in transitions], dtype=torch.float32, device=device)
            # LINE-BY-LINE: `advantages`에 `returns_tensor - old_values` 결과를 저장합니다. 의미/사용: `advantages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            advantages = returns_tensor - old_values
            # LINE-BY-LINE: 조건 `advantages.numel() > 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if advantages.numel() > 1:
                # PPO에서 흔히 쓰는 advantage 정규화입니다.
                # LINE-BY-LINE: `advantages`에 `(advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)` 결과를 저장합니다. 의미/사용: `advantages` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # LINE-BY-LINE: `_ in range(update_epochs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for _ in range(update_epochs):
                # LINE-BY-LINE: `total_actor_loss`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `total_actor_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_actor_loss = torch.zeros((), dtype=torch.float32, device=device)
                # LINE-BY-LINE: `total_value_loss`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `total_value_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_value_loss = torch.zeros((), dtype=torch.float32, device=device)
                # LINE-BY-LINE: `total_entropy`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `total_entropy` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_entropy = torch.zeros((), dtype=torch.float32, device=device)

                # LINE-BY-LINE: `step_idx, transition in enumerate(transitions)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for step_idx, transition in enumerate(transitions):
                    # old rollout을 다시 current policy로 평가합니다.
                    # LINE-BY-LINE: `output`에 `model(transition.observation, device)` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    output = model(transition.observation, device)
                    # LINE-BY-LINE: `distribution`에 `Categorical(logits=output["logits"])` 결과를 저장합니다. 의미/사용: `distribution` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    distribution = Categorical(logits=output["logits"])
                    # LINE-BY-LINE: `action_tensor`에 `torch.tensor(transition.action_index, dtype=torch.long, device=device)` 결과를 저장합니다. 의미/사용: `action_tensor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    action_tensor = torch.tensor(transition.action_index, dtype=torch.long, device=device)
                    # LINE-BY-LINE: `new_log_prob`에 `distribution.log_prob(action_tensor)` 결과를 저장합니다. 의미/사용: `new_log_prob` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    new_log_prob = distribution.log_prob(action_tensor)
                    # LINE-BY-LINE: `entropy`에 `distribution.entropy()` 결과를 저장합니다. 의미/사용: `entropy` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    entropy = distribution.entropy()
                    # LINE-BY-LINE: `value`에 `output["value"]` 결과를 저장합니다. 의미/사용: `value` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    value = output["value"]

                    # ratio = pi_new / pi_old
                    # LINE-BY-LINE: `ratio`에 `torch.exp(new_log_prob - old_log_probs[step_idx])` 결과를 저장합니다. 의미/사용: `ratio` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    ratio = torch.exp(new_log_prob - old_log_probs[step_idx])
                    # LINE-BY-LINE: `unclipped`에 `ratio * advantages[step_idx]` 결과를 저장합니다. 의미/사용: `unclipped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    unclipped = ratio * advantages[step_idx]
                    # LINE-BY-LINE: `clipped`에 `torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[step_idx]` 결과를 저장합니다. 의미/사용: `clipped` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[step_idx]
                    # LINE-BY-LINE: `actor_loss`에 `-torch.min(unclipped, clipped)` 결과를 저장합니다. 의미/사용: `actor_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    actor_loss = -torch.min(unclipped, clipped)
                    # LINE-BY-LINE: `value_loss`에 `F.mse_loss(value, returns_tensor[step_idx])` 결과를 저장합니다. 의미/사용: `value_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    value_loss = F.mse_loss(value, returns_tensor[step_idx])

                    # LINE-BY-LINE: `total_actor_loss`에 `total_actor_loss + actor_loss` 결과를 저장합니다. 의미/사용: `total_actor_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    total_actor_loss = total_actor_loss + actor_loss
                    # LINE-BY-LINE: `total_value_loss`에 `total_value_loss + value_loss` 결과를 저장합니다. 의미/사용: `total_value_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    total_value_loss = total_value_loss + value_loss
                    # LINE-BY-LINE: `total_entropy`에 `total_entropy + entropy` 결과를 저장합니다. 의미/사용: `total_entropy` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    total_entropy = total_entropy + entropy

                # LINE-BY-LINE: `total_actor_loss`에 `total_actor_loss / len(transitions)` 결과를 저장합니다. 의미/사용: `total_actor_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_actor_loss = total_actor_loss / len(transitions)
                # LINE-BY-LINE: `total_value_loss`에 `total_value_loss / len(transitions)` 결과를 저장합니다. 의미/사용: `total_value_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_value_loss = total_value_loss / len(transitions)
                # LINE-BY-LINE: `total_entropy`에 `total_entropy / len(transitions)` 결과를 저장합니다. 의미/사용: `total_entropy` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_entropy = total_entropy / len(transitions)
                # LINE-BY-LINE: `loss`에 `total_actor_loss + value_coef * total_value_loss - entropy_coef * total_entropy` 결과를 저장합니다. 의미/사용: `loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                loss = total_actor_loss + value_coef * total_value_loss - entropy_coef * total_entropy

                # LINE-BY-LINE: `optimizer.zero_grad()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                optimizer.zero_grad()
                # LINE-BY-LINE: `loss.backward()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                loss.backward()
                # LINE-BY-LINE: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm` 여러 변수에 `1.0)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # LINE-BY-LINE: `optimizer.step()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                optimizer.step()

            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: `f"episode`에 `{episode + 1} reward={summary['total_reward']:.3f} "` 결과를 저장합니다. 의미/사용: `f"episode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"episode={episode + 1} reward={summary['total_reward']:.3f} "
                # LINE-BY-LINE: `f"makespan`에 `{summary['makespan']:.3f} actor_loss={float(total_actor_loss.item()):.4f} "` 결과를 저장합니다. 의미/사용: `f"makespan` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"makespan={summary['makespan']:.3f} actor_loss={float(total_actor_loss.item()):.4f} "
                # LINE-BY-LINE: `f"value_loss`에 `{float(total_value_loss.item()):.4f}"` 결과를 저장합니다. 의미/사용: `f"value_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"value_loss={float(total_value_loss.item()):.4f}"
            )
            episode_metrics.append(
                {
                    "episode": episode + 1,
                    "total_reward": round(float(summary["total_reward"]), 6),
                    "makespan": round(float(summary["makespan"]), 6),
                    "load_imbalance": round(float(summary["load_imbalance"]), 6),
                    "scheduled_jobs": int(summary["scheduled_jobs"]),
                    "transition_count": len(transitions),
                    "actor_loss": round(float(total_actor_loss.item()), 6),
                    "value_loss": round(float(total_value_loss.item()), 6),
                    "entropy": round(float(total_entropy.item()), 6),
                }
            )

        # LINE-BY-LINE: `checkpoint_path`에 `save_checkpoint(model, config, self.name)` 결과를 저장합니다. 의미/사용: `checkpoint_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        checkpoint_path = save_checkpoint(model, config, self.name)
        # LINE-BY-LINE: 콘솔에 `print(f"- checkpoint: {checkpoint_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- checkpoint: {checkpoint_path}")
        report_paths = write_training_report(
            metrics=episode_metrics,
            output_dir=config["paths"]["output_dir"],
            algorithm=self.name,
            checkpoint_path=checkpoint_path,
        )
        print(f"- training_report_html: {report_paths['report_html']}")
        return {
            "checkpoint_path": checkpoint_path,
            "metrics": episode_metrics,
            "report_paths": report_paths,
        }
