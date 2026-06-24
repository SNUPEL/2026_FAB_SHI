"""계층형 DES action adapter 검증.

이 테스트는 실제 Excel을 읽지 않고 작은 scenario만 사용한다.
목적은 RL wrapper가 볼 action을 `SELECT_MACHINE -> SELECT_WO/COMMIT` 단계로
잘 나누면서도, 실제 hard-feasible 후보는 기존 DES core 후보만 사용하는지 확인하는 것이다.
"""

import csv
import json
import tempfile
import unittest

from Environment.environment import CuttingShopEnvironment
from Environment.gym_wrapper import (
    GYMNASIUM_AVAILABLE,
    HierarchicalDESActionAdapter,
    PMSPHierarchicalGymnasiumWrapper,
    run_hierarchical_trace_export,
)


def _minimal_config() -> dict:
    """batch_open adapter 테스트에 필요한 최소 config를 만든다."""

    return {
        "simulation": {
            "start_date": "2026-03-19",
            "minutes_per_day": 1440,
            "horizon_minutes": 1440,
            "auto_advance_time": True,
        },
        "calendar": {
            "enable_holidays_off": False,
            "enable_half_day_off": False,
            "enable_lunch_break": False,
            "enable_global_shutdown_windows": False,
            "enable_machine_operating_windows": False,
            "enable_machine_shutdown_windows": False,
            "enable_machine_breakdowns": False,
            "enable_operation_time_adjustment": False,
        },
        "setup": {
            "enable_family_changeover": False,
            "default_family_changeover_minutes": 0,
            "machine_type_changeover_minutes": {},
        },
        "action_space": {
            "mode": "batch_open",
            "allow_single_slot_baseline": False,
        },
        "batch": {
            "processing_time_rule": "max_individual",
        },
        "constraints": {
            "categories": {
                "machine": True,
                "capacity": True,
                "downstream": False,
                "priority": False,
                "preference": False,
                "layout": True,
                "calendar": True,
            },
            "hard_enabled": {
                "calendar_open": True,
                "machine_calendar_open": True,
                "machine_breakdown": True,
                "machine_enabled": True,
                "family_eligibility": True,
                "thickness_range": True,
                "table_length_limit": True,
                "machine_single_processing": True,
                "machine_bay_consistency": True,
                "block_set_same_bay": True,
                "auxiliary_resources_available": True,
                "batch_wo_count_limit": True,
                "batch_length_sum_limit": True,
            },
            "soft_enabled": {},
            "soft_penalty_weights": {},
            "overrides": {},
            "machine_day_limits": {
                "max_wo_count": 3,
                "max_length_sum": 55000,
            },
        },
        "reward": {
            "makespan_weight": 1.0,
            "load_balance_weight": 0.1,
            "priority_bonus_weight": 0.02,
            "soft_penalty_weight": 1.0,
        },
    }


def _minimal_scenario() -> dict:
    """두 설비와 W/O 3개짜리 작은 batch_open scenario를 만든다."""

    machine_template = {
        "machine_type": "plasma",
        "enabled": True,
        "eligible_families": ["NP"],
        "min_thickness": 0,
        "max_thickness": 100,
        "table_length_limit": 55000,
        "cut_speed_factor": 1.0,
        "daily_capacity_minutes": 1440,
        "parallel_capacity": 1,
        "required_resource_ids": [],
        "max_batch_wo_count": 3,
        "max_batch_length_sum": 55000,
    }
    jobs = []
    for index in range(1, 4):
        jobs.append(
            {
                "job_id": f"J{index}",
                "family": "NP",
                "thickness": 12.0,
                "plate_length": 20000.0,
                "downstream_bay": "D1",
                "priority_weight": 1,
                "base_stage_minutes": {"setup": 0.0, "cut": float(index * 10), "finish": 0.0},
                "block_set_id": f"BLOCK_{index}",
            }
        )
    return {
        "metadata": {"job_count": len(jobs)},
        "machines": [
            {"machine_id": "PLS21", "bay_id": "22", **machine_template},
            {"machine_id": "PLS31", "bay_id": "23", **machine_template},
        ],
        "cut_bays": [],
        "resources": [],
        "bays": [
            {
                "bay_id": "D1",
                "priority_rank": 1,
                "capacity_limit": 999,
                "transfer_time_minutes": 0.0,
            }
        ],
        "jobs": jobs,
    }


class HierarchicalActionAdapterTest(unittest.TestCase):
    """계층형 adapter가 DES 후보와 state를 보존하는지 확인한다."""

    def setUp(self) -> None:
        self.env = CuttingShopEnvironment(config=_minimal_config(), scenario=_minimal_scenario())
        self.adapter = HierarchicalDESActionAdapter(self.env, max_actions=64)
        self.adapter.refresh()

    def test_select_machine_does_not_mutate_des_state(self) -> None:
        """설비 선택 단계는 RL용 context 선택일 뿐 DES state를 바꾸면 안 된다."""

        snapshot = self.adapter.snapshot
        self.assertEqual(snapshot.phase, "SELECT_MACHINE")
        self.assertEqual(
            [row["machine_id"] for row in snapshot.action_id_table],
            ["PLS21", "PLS31"],
        )

        selected_index = self._row_index(action_kind="select_machine", machine_id="PLS21")
        _, _, _, info = self.adapter.step(selected_index)

        self.assertFalse(info["des_state_mutated"])
        self.assertEqual(self.adapter.snapshot.phase, "SELECT_WO")
        self.assertEqual(self.adapter.snapshot.selected_machine_id, "PLS21")
        self.assertEqual(len(self.env.simulation.state.unscheduled_jobs), 3)
        self.assertEqual(len(self.env.simulation.state.open_batches), 0)

    def test_select_wo_then_commit_uses_existing_des_batch_candidates(self) -> None:
        """W/O 선택과 commit은 기존 hard-feasible DES candidate만 실행해야 한다."""

        self.adapter.step(self._row_index(action_kind="select_machine", machine_id="PLS21"))

        self.adapter.step(self._row_index(action_kind="select_wo", job_id="J1"))
        self.assertEqual(len(self.env.simulation.state.open_batches), 1)
        self.assertEqual(len(self.env.simulation.state.unscheduled_jobs), 2)
        self.assertEqual(self.adapter.snapshot.phase, "SELECT_WO")

        self.adapter.step(self._row_index(action_kind="select_wo", job_id="J2"))
        rows_after_two_jobs = self.adapter.current_action_table()
        select_wo_job_ids = [row["job_id"] for row in rows_after_two_jobs if row["action_kind"] == "select_wo"]
        self.assertEqual(select_wo_job_ids, [])
        self.assertEqual([row["action_kind"] for row in rows_after_two_jobs], ["commit_batch"])

        self.adapter.step(self._row_index(action_kind="commit_batch"))
        self.assertEqual(len(self.env.simulation.state.open_batches), 0)
        self.assertEqual(len(self.env.simulation.state.schedule), 2)
        self.assertEqual(self.adapter.snapshot.phase, "SELECT_MACHINE")

    def _row_index(self, *, action_kind: str, machine_id: str | None = None, job_id: str | None = None) -> int:
        """현재 action table에서 조건에 맞는 row index를 찾는다."""

        for row in self.adapter.current_action_table():
            if row["action_kind"] != action_kind:
                continue
            if machine_id is not None and row.get("machine_id") != machine_id:
                continue
            if job_id is not None and row.get("job_id") != job_id:
                continue
            return int(row["index"])
        raise AssertionError(f"row not found: action_kind={action_kind} machine_id={machine_id} job_id={job_id}")


@unittest.skipUnless(GYMNASIUM_AVAILABLE, "gymnasium is required for PMSPHierarchicalGymnasiumWrapper")
class HierarchicalGymnasiumWrapperTest(unittest.TestCase):
    """계층형 adapter가 실제 Gymnasium Env 형태로도 노출되는지 확인한다."""

    def test_reset_and_select_machine_step_expose_hierarchical_observation(self) -> None:
        """reset은 설비 선택 단계이고, 첫 step은 DES state를 변경하지 않는 machine 선택이어야 한다."""

        env = CuttingShopEnvironment(config=_minimal_config(), scenario=_minimal_scenario())
        wrapper = PMSPHierarchicalGymnasiumWrapper(env, max_actions=64)

        obs, info = wrapper.reset()
        self.assertEqual(info["phase"], "SELECT_MACHINE")
        self.assertEqual(int(obs["action_mask"].sum()), 2)
        self.assertEqual(float(obs["phase_code"][0]), 0.0)

        obs, reward, terminated, truncated, step_info = wrapper.step(0)

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(step_info["des_state_mutated"])
        self.assertEqual(step_info["phase"], "SELECT_WO")
        self.assertEqual(float(obs["phase_code"][0]), 1.0)
        self.assertEqual(len(env.simulation.state.open_batches), 0)
        self.assertEqual(len(env.simulation.state.schedule), 0)


@unittest.skipUnless(GYMNASIUM_AVAILABLE, "gymnasium is required for hierarchical trace export")
class HierarchicalTraceExportTest(unittest.TestCase):
    """계층형 wrapper가 학습용 trace 파일을 만들 수 있는지 확인한다."""

    def test_export_records_machine_and_wo_phases(self) -> None:
        """exporter는 선택 phase, action mask count, 선택 row를 CSV/JSON으로 남겨야 한다."""

        env = CuttingShopEnvironment(config=_minimal_config(), scenario=_minimal_scenario())
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_hierarchical_trace_export(
                env=env,
                heuristic_name="spt",
                max_actions=64,
                output_dir=temp_dir,
            )

            self.assertEqual(summary["scheduled_jobs"], 3)
            self.assertEqual(summary["unscheduled_jobs"], [])
            self.assertGreater(summary["hierarchical_step_count"], summary["flat_decision_count"])

            with open(summary["trace_csv"], encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            phases = {row["phase"] for row in rows}
            self.assertIn("SELECT_MACHINE", phases)
            self.assertIn("SELECT_WO", phases)
            self.assertTrue(all(int(row["mask_true_count"]) > 0 for row in rows))

            with open(summary["action_table_jsonl"], encoding="utf-8") as handle:
                table_records = [json.loads(line) for line in handle]
            self.assertEqual(len(table_records), len(rows))
            self.assertTrue(all(record["action_id_table"] for record in table_records))


if __name__ == "__main__":
    unittest.main()
