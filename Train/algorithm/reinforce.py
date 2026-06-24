"""REINFORCE 구현.

가장 기본적인 policy gradient 방법입니다.

아이디어:
- rollout을 하나 수집
- 각 step의 log_prob * return으로 loss 계산
- return이 큰 행동의 확률을 높이도록 업데이트
"""

# LINE-BY-LINE: `typing` 모듈에서 `Dict`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn.functional as F` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn.functional as F

# LINE-BY-LINE: `.common` 모듈에서 `build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .common import build_model, collect_episode, compute_discounted_returns, get_device, save_checkpoint


# LINE-BY-LINE: `ReinforceTrainer` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class ReinforceTrainer:
    """REINFORCE 학습 루프를 담당하는 trainer."""

    # LINE-BY-LINE: `name`에 `"reinforce"` 결과를 저장합니다. 의미/사용: `name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    name = "reinforce"

    # LINE-BY-LINE: `train(self, env, config: Dict)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def train(self, env, config: Dict):
        """환경과 config를 받아 REINFORCE episode 학습을 수행한다."""

        # LINE-BY-LINE: `device`에 `get_device(config)` 결과를 저장합니다. 의미/사용: `device`는 torch 연산 장치입니다. 예: cpu 또는 cuda.
        device = get_device(config)
        # LINE-BY-LINE: `model`에 `build_model(env, config, device)` 결과를 저장합니다. 의미/사용: `model`는 policy-value neural network입니다. observation을 logits/value로 바꿉니다.
        model = build_model(env, config, device)
        # LINE-BY-LINE: `optimizer`에 `torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["reinforce"]["lr"]))` 결과를 저장합니다. 의미/사용: `optimizer`는 PyTorch optimizer입니다. loss.backward 이후 parameter를 갱신합니다.
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["reinforce"]["lr"]))
        # LINE-BY-LINE: `gamma`에 `float(config["algorithm"]["reinforce"]["gamma"])` 결과를 저장합니다. 의미/사용: `gamma` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        gamma = float(config["algorithm"]["reinforce"]["gamma"])
        # LINE-BY-LINE: `entropy_coef`에 `float(config["algorithm"]["reinforce"]["entropy_coef"])` 결과를 저장합니다. 의미/사용: `entropy_coef` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        entropy_coef = float(config["algorithm"]["reinforce"]["entropy_coef"])
        # LINE-BY-LINE: `episodes`에 `int(config["train"]["episodes"])` 결과를 저장합니다. 의미/사용: `episodes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        episodes = int(config["train"]["episodes"])

        # LINE-BY-LINE: 콘솔에 `print("[reinforce]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[reinforce]")
        # LINE-BY-LINE: 콘솔에 `print(f"- device: {device}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- device: {device}")
        # LINE-BY-LINE: 콘솔에 `print(f"- episodes: {episodes}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- episodes: {episodes}")

        # LINE-BY-LINE: `episode in range(episodes)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for episode in range(episodes):
            # 현재 정책으로 에피소드 하나를 끝까지 실행합니다.
            # LINE-BY-LINE: `transitions, summary` 여러 변수에 `collect_episode(env, model, device, deterministic=False, temperature=1.0)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=1.0)
            # LINE-BY-LINE: 조건 `not transitions`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not transitions:
                print(
                    "[ERROR][ReinforceTrainer.train] "
                    f"cause=empty_rollout episode={episode + 1}"
                )
                raise RuntimeError("REINFORCE rollout produced no transitions")

            # reward sequence -> discounted return
            # LINE-BY-LINE: `returns`에 `compute_discounted_returns([transition.reward for transition in transitions], gamma)` 결과를 저장합니다. 의미/사용: `returns` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            returns = compute_discounted_returns([transition.reward for transition in transitions], gamma)
            # LINE-BY-LINE: `returns_tensor`에 `torch.tensor(returns, dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `returns_tensor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            # LINE-BY-LINE: 조건 `returns_tensor.numel() > 1`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if returns_tensor.numel() > 1:
                # LINE-BY-LINE: `returns_tensor`에 `(returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)` 결과를 저장합니다. 의미/사용: `returns_tensor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)

            # LINE-BY-LINE: `loss`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            loss = torch.zeros((), dtype=torch.float32, device=device)
            # LINE-BY-LINE: `entropy_bonus`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `entropy_bonus` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            entropy_bonus = torch.zeros((), dtype=torch.float32, device=device)

            # LINE-BY-LINE: `step_idx, transition in enumerate(transitions)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for step_idx, transition in enumerate(transitions):
                # transition.observation을 다시 모델에 넣어
                # 현재 정책 기준 log_prob를 계산합니다.
                # LINE-BY-LINE: `output`에 `model(transition.observation, device)` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                output = model(transition.observation, device)
                # LINE-BY-LINE: `distribution`에 `torch.distributions.Categorical(logits=output["logits"])` 결과를 저장합니다. 의미/사용: `distribution` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                distribution = torch.distributions.Categorical(logits=output["logits"])
                # LINE-BY-LINE: `action_tensor`에 `torch.tensor(transition.action_index, dtype=torch.long, device=device)` 결과를 저장합니다. 의미/사용: `action_tensor` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                action_tensor = torch.tensor(transition.action_index, dtype=torch.long, device=device)
                # LINE-BY-LINE: `log_prob`에 `distribution.log_prob(action_tensor)` 결과를 저장합니다. 의미/사용: `log_prob` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                log_prob = distribution.log_prob(action_tensor)
                # LINE-BY-LINE: `entropy_bonus`에 `entropy_bonus + distribution.entropy()` 결과를 저장합니다. 의미/사용: `entropy_bonus` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                entropy_bonus = entropy_bonus + distribution.entropy()
                # LINE-BY-LINE: `loss`에 `loss - log_prob * returns_tensor[step_idx]` 결과를 저장합니다. 의미/사용: `loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                loss = loss - log_prob * returns_tensor[step_idx]

            # LINE-BY-LINE: `loss`에 `loss / len(transitions) - entropy_coef * entropy_bonus / len(transitions)` 결과를 저장합니다. 의미/사용: `loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            loss = loss / len(transitions) - entropy_coef * entropy_bonus / len(transitions)

            # gradient 계산 -> parameter update
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
                # LINE-BY-LINE: `f"makespan`에 `{summary['makespan']:.3f} loss={float(loss.item()):.4f}"` 결과를 저장합니다. 의미/사용: `f"makespan` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"makespan={summary['makespan']:.3f} loss={float(loss.item()):.4f}"
            )

        # LINE-BY-LINE: `checkpoint_path`에 `save_checkpoint(model, config, self.name)` 결과를 저장합니다. 의미/사용: `checkpoint_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        checkpoint_path = save_checkpoint(model, config, self.name)
        # LINE-BY-LINE: 콘솔에 `print(f"- checkpoint: {checkpoint_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- checkpoint: {checkpoint_path}")
