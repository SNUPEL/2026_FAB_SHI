"""학습 알고리즘 선택 팩토리."""

from .ppo import PPOTrainer
from .reinforce import ReinforceTrainer
from .self_labeling import SelfLabelingTrainer


def build_trainer(algorithm_name: str):
    trainers = {
        "reinforce": ReinforceTrainer,
        "ppo": PPOTrainer,
        "self_labeling": SelfLabelingTrainer,
    }
    if algorithm_name not in trainers:
        raise ValueError(f"unknown algorithm: {algorithm_name}")
    return trainers[algorithm_name]()
