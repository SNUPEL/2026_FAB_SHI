"""학습 알고리즘 선택 팩토리."""

# LINE-BY-LINE: `.ppo` 모듈에서 `PPOTrainer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .ppo import PPOTrainer
# LINE-BY-LINE: `.reinforce` 모듈에서 `ReinforceTrainer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .reinforce import ReinforceTrainer
# LINE-BY-LINE: `.self_labeling` 모듈에서 `SelfLabelingTrainer`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from .self_labeling import SelfLabelingTrainer


# LINE-BY-LINE: `build_trainer(algorithm_name: str)` 함수를 정의합니다. 반환 타입: `명시 없음`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def build_trainer(algorithm_name: str):
    """알고리즘 이름에 해당하는 trainer instance를 만든다."""

    # LINE-BY-LINE: `trainers`에 `{` 결과를 저장합니다. 의미/사용: `trainers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    trainers = {
        # LINE-BY-LINE: `build_trainer`에서 반환/저장할 dict의 `reinforce` 키에 `ReinforceTrainer` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "reinforce": ReinforceTrainer,
        # LINE-BY-LINE: `build_trainer`에서 반환/저장할 dict의 `ppo` 키에 `PPOTrainer` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "ppo": PPOTrainer,
        # LINE-BY-LINE: `build_trainer`에서 반환/저장할 dict의 `self_labeling` 키에 `SelfLabelingTrainer` 값을 넣습니다. 사용: 호출자가 이 key로 값을 읽습니다.
        "self_labeling": SelfLabelingTrainer,
    }
    # LINE-BY-LINE: 조건 `algorithm_name not in trainers`를 검사합니다. 참이면 아래 블록을 실행합니다. 사용: 데이터 오류, 제약 통과 여부, 분기 처리를 결정합니다.
    if algorithm_name not in trainers:
        # LINE-BY-LINE: `ValueError(f"unknown algorithm: {algorithm_name}")` 예외를 발생시킵니다. 사용: 오류를 숨기지 않고 호출자/CLI에 실패를 전달합니다.
        raise ValueError(f"unknown algorithm: {algorithm_name}")
    # LINE-BY-LINE: 호출자에게 `trainers[algorithm_name]()`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return trainers[algorithm_name]()
