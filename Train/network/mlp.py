"""작은 MLP 유틸.

프로젝트 전체에서 fully connected network를 여러 군데 쓰므로,
같은 패턴을 반복하지 않기 위해 공용 함수로 분리했습니다.
"""

from typing import Iterable, List

import torch.nn as nn


def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """간단한 MLP를 생성합니다.

    예:
    - input_dim=16
    - hidden_dims=[64, 64]
    - output_dim=32

    이면
    16 -> 64 -> 64 -> 32 구조가 만들어집니다.
    """

    layers: List[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)
