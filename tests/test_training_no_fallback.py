"""학습 알고리즘의 silent fallback 금지 검증."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from Train.algorithm.reinforce import ReinforceTrainer
from Train.algorithm.self_labeling import SelfLabelingTrainer


def _dummy_model():
    """optimizer 생성에 필요한 parameter를 가진 최소 PyTorch module."""

    return torch.nn.Linear(1, 1)


class TrainingNoFallbackTest(unittest.TestCase):
    """빈 rollout을 조용히 건너뛰지 않고 실패시키는지 확인한다."""

    def test_reinforce_empty_rollout_raises(self) -> None:
        """REINFORCE는 transitions가 비면 continue가 아니라 예외를 내야 한다."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "train": {"episodes": 1},
                "paths": {"output_dir": str(Path(temp_dir) / "reinforce")},
                "algorithm": {
                    "reinforce": {"lr": 0.001, "gamma": 0.99, "entropy_coef": 0.001}
                },
            }
            with patch("Train.algorithm.reinforce.get_device", return_value=torch.device("cpu")):
                with patch("Train.algorithm.reinforce.build_model", return_value=_dummy_model()):
                    with patch(
                        "Train.algorithm.reinforce.collect_episode",
                        return_value=([], {"total_reward": 0.0, "makespan": 0.0}),
                    ):
                        with self.assertRaises(RuntimeError):
                            ReinforceTrainer().train(object(), config)

    def test_self_labeling_empty_rollout_bank_raises(self) -> None:
        """Self-labeling은 rollout_bank가 비면 continue가 아니라 예외를 내야 한다."""

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "train": {"episodes": 1},
                "paths": {"output_dir": str(Path(temp_dir) / "self_labeling")},
                "reward": {"load_balance_weight": 0.1},
                "algorithm": {
                    "self_labeling": {
                        "lr": 0.001,
                        "rollout_samples": 1,
                        "imitation_epochs": 1,
                        "sample_temperature": 1.0,
                    }
                },
            }
            with patch("Train.algorithm.self_labeling.get_device", return_value=torch.device("cpu")):
                with patch("Train.algorithm.self_labeling.build_model", return_value=_dummy_model()):
                    with patch(
                        "Train.algorithm.self_labeling.collect_episode",
                        return_value=([], {"makespan": 0.0, "load_imbalance": 0.0}),
                    ):
                        with self.assertRaises(RuntimeError):
                            SelfLabelingTrainer().train(object(), config)


if __name__ == "__main__":
    unittest.main()
