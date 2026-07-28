"""할당 W/O 개수 기준 balanced batch 휴리스틱 검증."""

from types import SimpleNamespace
import unittest

from Agent.heuristics import select_action_by_rule


class BalancedBatchCountHeuristicTest(unittest.TestCase):
    """`balanced_batch_count`가 처리시간 부하가 아니라 작업 개수 균형을 우선하는지 확인한다."""

    def test_prefers_machine_and_bay_with_lower_assigned_job_count(self) -> None:
        """Bay 22에 이미 2개가 있으면 같은 Bay보다 빈 Bay 23 후보를 고른다."""

        simulation = SimpleNamespace(
            machines={
                "PLS21": SimpleNamespace(machine_id="PLS21", bay_id="22"),
                "PLS22": SimpleNamespace(machine_id="PLS22", bay_id="22"),
                "PLS31": SimpleNamespace(machine_id="PLS31", bay_id="23"),
            },
            state=SimpleNamespace(
                schedule=[
                    SimpleNamespace(machine_id="PLS21"),
                    SimpleNamespace(machine_id="PLS21"),
                ],
                open_batches={},
                machine_loads={"PLS21": 10.0, "PLS22": 0.0, "PLS31": 0.0},
                machine_slot_available_at={"PLS21": [10.0], "PLS22": [0.0], "PLS31": [0.0]},
                current_time=0.0,
            ),
        )

        candidates = [
            self._candidate("candidate_same_bay", "PLS22", "22"),
            self._candidate("candidate_empty_bay", "PLS31", "23"),
        ]

        selected = select_action_by_rule(candidates, simulation, "balanced_batch_count")

        self.assertEqual(selected.action_id, "candidate_empty_bay")

    def test_prefers_filled_close_batch_when_only_count_imbalance_would_pick_single(self) -> None:
        """1개 close와 3개 close가 모두 가능하면 불필요한 1개 batch를 선호하지 않는다."""

        simulation = SimpleNamespace(
            machines={
                "PLS21": SimpleNamespace(machine_id="PLS21", bay_id="22", max_batch_wo_count=3),
                "PLS31": SimpleNamespace(machine_id="PLS31", bay_id="23", max_batch_wo_count=3),
            },
            state=SimpleNamespace(
                schedule=[],
                open_batches={},
                machine_loads={"PLS21": 0.0, "PLS31": 0.0},
                machine_slot_available_at={"PLS21": [0.0], "PLS31": [0.0]},
                current_time=0.0,
            ),
        )

        candidates = [
            self._close_candidate("close_single", "PLS21", "22", 1),
            self._close_candidate("close_filled", "PLS21", "22", 3),
        ]

        selected = select_action_by_rule(candidates, simulation, "balanced_batch_count")

        self.assertEqual(selected.action_id, "close_filled")

    @staticmethod
    def _candidate(action_id: str, machine_id: str, bay_id: str) -> SimpleNamespace:
        """테스트용 후보 action 객체를 만든다."""

        return SimpleNamespace(
            action_id=action_id,
            action_kind="open_batch",
            machine=SimpleNamespace(machine_id=machine_id, bay_id=bay_id),
            job=SimpleNamespace(job_id=f"job_{machine_id}"),
            batch_id="",
            batch_wo_count=1,
            batch_job_ids=(f"job_{machine_id}",),
            processing_minutes=5.0,
            finish_time=5.0,
            estimated_minutes=5.0,
            soft_penalty=0.0,
        )

    @staticmethod
    def _close_candidate(action_id: str, machine_id: str, bay_id: str, batch_wo_count: int) -> SimpleNamespace:
        """테스트용 close_batch 후보 action 객체를 만든다."""

        job_ids = tuple(f"job_{index}" for index in range(batch_wo_count))
        return SimpleNamespace(
            action_id=action_id,
            action_kind="close_batch",
            machine=SimpleNamespace(machine_id=machine_id, bay_id=bay_id, max_batch_wo_count=3),
            job=SimpleNamespace(job_id=job_ids[0]),
            batch_id=f"batch_{batch_wo_count}",
            batch_wo_count=batch_wo_count,
            batch_job_ids=job_ids,
            processing_minutes=10.0,
            finish_time=10.0,
            estimated_minutes=10.0,
            soft_penalty=0.0,
        )


if __name__ == "__main__":
    unittest.main()
