"""작은 MLP 유틸.

프로젝트 전체에서 fully connected network를 여러 군데 쓰므로,
같은 패턴을 반복하지 않기 위해 공용 함수로 분리했습니다.
"""

# LINE-BY-LINE: `typing` 모듈에서 `Iterable, List`를 가져옵니다. 사용: 이 파일의 타입 생성/함수 호출에 직접 씁니다.
from typing import Iterable, List

# LINE-BY-LINE: `torch.nn as nn` 모듈을 가져옵니다. 사용: 이 파일 안에서 해당 라이브러리 기능을 호출합니다.
import torch.nn as nn


# LINE-BY-LINE: `build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation=nn.ReLU)` 함수를 정의합니다. 반환 타입: `nn.Sequential`. 사용: 학습/평가 루프에서 observation, model, reward를 연결할 때 사용됩니다.
def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """간단한 MLP를 생성합니다.

    예:
    - input_dim=16
    - hidden_dims=[64, 64]
    - output_dim=32

    이면
    16 -> 64 -> 64 -> 32 구조가 만들어집니다.
    """

    # LINE-BY-LINE: `layers` 변수에 `[]` 결과를 저장합니다. 의미: `layers` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    layers: List[nn.Module] = []
    # LINE-BY-LINE: `dims`에 `[input_dim, *hidden_dims, output_dim]` 결과를 저장합니다. 의미/사용: `dims` 값입니다. 사용: 이후 같은 함수/블록에서 계산, 검증, 출력에 참조됩니다.
    dims = [input_dim, *hidden_dims, output_dim]
    # LINE-BY-LINE: `in_dim, out_dim in zip(dims[:-2], dims[1:-1])` 반복을 시작합니다. 사용: jobs, machines, rows, events 같은 목록을 하나씩 처리합니다.
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        # LINE-BY-LINE: `layers.append(nn.Linear(in_dim, out_dim))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        layers.append(nn.Linear(in_dim, out_dim))
        # LINE-BY-LINE: `layers.append(activation())`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
        layers.append(activation())
    # LINE-BY-LINE: `layers.append(nn.Linear(dims[-2], dims[-1]))`를 실행합니다. 의미/사용: 해당 함수/메서드를 호출해 필요한 계산이나 상태 변경을 수행합니다.
    layers.append(nn.Linear(dims[-2], dims[-1]))
    # LINE-BY-LINE: 호출자에게 `nn.Sequential(*layers)`를 반환합니다. 사용: 상위 함수가 이 결과로 다음 계산/검증/출력을 진행합니다.
    return nn.Sequential(*layers)
