"""Self-labeling 구현.

아이디어:
- 현재 정책으로 여러 개 rollout을 샘플링
- 그중 objective가 가장 좋은 rollout을 선택
- 그 rollout의 action sequence를 pseudo-label로 사용
- 지도학습처럼 cross-entropy로 정책을 업데이트
"""

# LINE-BY-LINE: `typing` 모듈에서 `Dict, List, Tuple`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Dict, List, Tuple

# LINE-BY-LINE: `torch` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch
# LINE-BY-LINE: `torch.nn.functional as F` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn.functional as F

# LINE-BY-LINE: `.common` 모듈에서 `build_model, collect_episode, get_device, save_checkpoint`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .common import build_model, collect_episode, get_device, save_checkpoint


# LINE-BY-LINE: `SelfLabelingTrainer` class를 정의합니다. 사용: 관련 데이터와 동작을 한 객체로 묶어 다른 모듈에서 import합니다. 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
class SelfLabelingTrainer:
    """self-labeling 방식의 imitation 학습 trainer."""

    # LINE-BY-LINE: `name`에 `"self_labeling"` 결과를 저장합니다. 의미/사용: `name` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    name = "self_labeling"

    # LINE-BY-LINE: `_objective_score(self, summary: Dict, config: Dict)` 함수를 정의합니다. 반환 타입: `float`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def _objective_score(self, summary: Dict, config: Dict) -> float:
        """낮을수록 좋은 objective를 계산합니다."""

        # LINE-BY-LINE: 호출자에게 `summary["makespan"] + config["reward"]["load_balance_weight"] * summary["load_imbalance"]`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
        return summary["makespan"] + config["reward"]["load_balance_weight"] * summary["load_imbalance"]

    # LINE-BY-LINE: `train(self, env, config: Dict)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
    def train(self, env, config: Dict):
        """여러 rollout 중 가장 좋은 action sequence를 pseudo-label로 학습한다."""

        # LINE-BY-LINE: `device`에 `get_device(config)` 결과를 저장합니다. 의미/사용: `device`는 torch 연산 장치입니다. 예: cpu 또는 cuda.
        device = get_device(config)
        # LINE-BY-LINE: `model`에 `build_model(env, config, device)` 결과를 저장합니다. 의미/사용: `model`는 policy-value neural network입니다. observation을 logits/value로 바꿉니다.
        model = build_model(env, config, device)
        # LINE-BY-LINE: `optimizer`에 `torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["self_labeling"]["lr"]))` 결과를 저장합니다. 의미/사용: `optimizer`는 PyTorch optimizer입니다. loss.backward 이후 parameter를 갱신합니다.
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["algorithm"]["self_labeling"]["lr"]))

        # LINE-BY-LINE: `rollout_samples`에 `int(config["algorithm"]["self_labeling"]["rollout_samples"])` 결과를 저장합니다. 의미/사용: `rollout_samples` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        rollout_samples = int(config["algorithm"]["self_labeling"]["rollout_samples"])
        # LINE-BY-LINE: `imitation_epochs`에 `int(config["algorithm"]["self_labeling"]["imitation_epochs"])` 결과를 저장합니다. 의미/사용: `imitation_epochs` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        imitation_epochs = int(config["algorithm"]["self_labeling"]["imitation_epochs"])
        # LINE-BY-LINE: `sample_temperature`에 `float(config["algorithm"]["self_labeling"]["sample_temperature"])` 결과를 저장합니다. 의미/사용: `sample_temperature` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        sample_temperature = float(config["algorithm"]["self_labeling"]["sample_temperature"])
        # LINE-BY-LINE: `episodes`에 `int(config["train"]["episodes"])` 결과를 저장합니다. 의미/사용: `episodes` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        episodes = int(config["train"]["episodes"])

        # LINE-BY-LINE: 콘솔에 `print("[self_labeling]")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print("[self_labeling]")
        # LINE-BY-LINE: 콘솔에 `print(f"- device: {device}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- device: {device}")
        # LINE-BY-LINE: 콘솔에 `print(f"- episodes: {episodes}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- episodes: {episodes}")
        # LINE-BY-LINE: 콘솔에 `print(f"- rollout_samples: {rollout_samples}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- rollout_samples: {rollout_samples}")

        # LINE-BY-LINE: `episode in range(episodes)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
        for episode in range(episodes):
            # LINE-BY-LINE: `rollout_bank` 변수에 `[]` 결과를 저장합니다. 의미: `rollout_bank` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
            rollout_bank: List[Tuple[List, Dict]] = []

            # LINE-BY-LINE: `_ in range(rollout_samples)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for _ in range(rollout_samples):
                # 같은 문제를 여러 번 rollout해서
                # "어떤 action sequence가 가장 좋았는지"를 찾습니다.
                # LINE-BY-LINE: `transitions, summary` 여러 변수에 `collect_episode(env, model, device, deterministic=False, temperature=sample_temperature)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                transitions, summary = collect_episode(env, model, device, deterministic=False, temperature=sample_temperature)
                # LINE-BY-LINE: 조건 `transitions`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
                if transitions:
                    # LINE-BY-LINE: `rollout_bank.append((transitions, summary))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                    rollout_bank.append((transitions, summary))

            # LINE-BY-LINE: 조건 `not rollout_bank`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
            if not rollout_bank:
                print(
                    "[ERROR][SelfLabelingTrainer.train] "
                    f"cause=empty_rollout_bank episode={episode + 1} "
                    f"rollout_samples={rollout_samples}"
                )
                raise RuntimeError("self-labeling rollout bank is empty")

            # LINE-BY-LINE: `best_transitions, best_summary` 여러 변수에 `min(` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
            best_transitions, best_summary = min(
                # LINE-BY-LINE: `min(...)` 호출에 `rollout_bank` 값을 인자로 전달합니다. 사용: 호출 함수가 이 값을 받아 계산/생성/검증합니다.
                rollout_bank,
                # LINE-BY-LINE: `key`에 `lambda item: self._objective_score(item[1], config)` 결과를 저장합니다. 의미/사용: `key` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                key=lambda item: self._objective_score(item[1], config),
            )

            # LINE-BY-LINE: `_ in range(imitation_epochs)` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
            for _ in range(imitation_epochs):
                # LINE-BY-LINE: `total_loss`에 `torch.zeros((), dtype=torch.float32, device=device)` 결과를 저장합니다. 의미/사용: `total_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_loss = torch.zeros((), dtype=torch.float32, device=device)
                # LINE-BY-LINE: `transition in best_transitions` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
                for transition in best_transitions:
                    # best rollout에서 실제로 선택했던 action index를
                    # 정답 label처럼 사용합니다.
                    # LINE-BY-LINE: `output`에 `model(transition.observation, device)` 결과를 저장합니다. 의미/사용: `output` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    output = model(transition.observation, device)
                    # LINE-BY-LINE: `logits`에 `output["logits"].unsqueeze(0)` 결과를 저장합니다. 의미/사용: `logits` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    logits = output["logits"].unsqueeze(0)
                    # LINE-BY-LINE: `target`에 `torch.tensor([transition.action_index], dtype=torch.long, device=device)` 결과를 저장합니다. 의미/사용: `target` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    target = torch.tensor([transition.action_index], dtype=torch.long, device=device)
                    # LINE-BY-LINE: `total_loss`에 `total_loss + F.cross_entropy(logits, target)` 결과를 저장합니다. 의미/사용: `total_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                    total_loss = total_loss + F.cross_entropy(logits, target)

                # LINE-BY-LINE: `total_loss`에 `total_loss / len(best_transitions)` 결과를 저장합니다. 의미/사용: `total_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                total_loss = total_loss / len(best_transitions)

                # LINE-BY-LINE: `optimizer.zero_grad()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                optimizer.zero_grad()
                # LINE-BY-LINE: `total_loss.backward()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                total_loss.backward()
                # LINE-BY-LINE: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm` 여러 변수에 `1.0)` 결과를 풀어 저장합니다. 사용: 반환 tuple을 각각의 의미 있는 값으로 나눕니다.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # LINE-BY-LINE: `optimizer.step()`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
                optimizer.step()

            # LINE-BY-LINE: 콘솔에 `print(` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
            print(
                # LINE-BY-LINE: `f"episode`에 `{episode + 1} best_makespan={best_summary['makespan']:.3f} "` 결과를 저장합니다. 의미/사용: `f"episode` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"episode={episode + 1} best_makespan={best_summary['makespan']:.3f} "
                # LINE-BY-LINE: `f"objective`에 `{self._objective_score(best_summary, config):.3f} "` 결과를 저장합니다. 의미/사용: `f"objective` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"objective={self._objective_score(best_summary, config):.3f} "
                # LINE-BY-LINE: `f"imitation_loss`에 `{float(total_loss.item()):.4f}"` 결과를 저장합니다. 의미/사용: `f"imitation_loss` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
                f"imitation_loss={float(total_loss.item()):.4f}"
            )

        # LINE-BY-LINE: `checkpoint_path`에 `save_checkpoint(model, config, self.name)` 결과를 저장합니다. 의미/사용: `checkpoint_path` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
        checkpoint_path = save_checkpoint(model, config, self.name)
        # LINE-BY-LINE: 콘솔에 `print(f"- checkpoint: {checkpoint_path}")` 내용을 출력합니다. 사용: 실행 config, count, 오류 원인을 사용자가 확인합니다.
        print(f"- checkpoint: {checkpoint_path}")
