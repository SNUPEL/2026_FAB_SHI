"""Phase 1/2가 공유할 계층 상태와 batch DES transition 검증."""

from types import SimpleNamespace
import unittest

from Environment.constraints.profiles import default_phase2_constraint_profile
from Environment.hierarchical import (
    CommonHierarchicalEnvironment,
    apply_phase1_block_assignment,
    create_phase1_planning_state,
)


class CommonHierarchicalEnvironmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.jobs = {
            "WO_A": self._job("WO_A", "P1::A", 30.0, 10_000.0, 500.0, 3),
            "WO_B": self._job("WO_B", "P1::A", 10.0, 12_000.0, 200.0, 1),
            "WO_C": self._job("WO_C", "P1::B", 20.0, 11_000.0, 300.0, 2),
        }
        self.machines = {
            "PLS21": self._machine("PLS21", "22"),
            "PLS22": self._machine("PLS22", "22"),
            "PLS31": self._machine("PLS31", "23"),
        }
        self.env = CommonHierarchicalEnvironment(
            jobs=self.jobs,
            machines=self.machines,
            bay_capacity_weights={"22": 2.0, "23": 1.0},
            constraint_profile=default_phase2_constraint_profile(),
            max_batch_wo_count=3,
            max_batch_length_sum=55_000.0,
        )
        self.env.commit_phase1(
            block_to_bay={"P1::A": "22", "P1::B": "23"},
            bay_loads={
                "22": self._bay_load(1, 2, 700.0, 4, 2.0),
                "23": self._bay_load(1, 1, 300.0, 2, 1.0),
            },
        )

    def test_commit_phase1_rejects_inconsistent_authoritative_bay_loads(self) -> None:
        env = CommonHierarchicalEnvironment(
            jobs=self.jobs,
            machines=self.machines,
            bay_capacity_weights={"22": 2.0, "23": 1.0},
            constraint_profile=default_phase2_constraint_profile(),
            max_batch_wo_count=3,
            max_batch_length_sum=55_000.0,
        )

        with self.assertRaises(RuntimeError):
            env.commit_phase1(
                block_to_bay={"P1::A": "22", "P1::B": "23"},
                bay_loads={
                    "22": self._bay_load(2, 2, 700.0, 4, 2.0),
                    "23": self._bay_load(1, 1, 300.0, 2, 1.0),
                },
            )

    def test_phase1_planning_rejects_non_numeric_capacity_weight(self) -> None:
        with self.assertRaises(RuntimeError):
            create_phase1_planning_state({"22": "invalid"})

    def test_phase1_assignment_rejects_fractional_discrete_load(self) -> None:
        state = create_phase1_planning_state({"22": 1.0})

        with self.assertRaises(RuntimeError):
            apply_phase1_block_assignment(
                state,
                block_set_id="P1::A",
                bay_id="22",
                steel_quantity_sum=1,
                cut_length_sum=100.0,
                bevel_quantity_sum=0,
                wo_count=1.5,
                long_cut_over_1000=0,
            )

    def test_snapshot_fork_shares_static_data_but_isolates_mutable_state(self) -> None:
        snapshot = self.env.snapshot()
        fork = self.env.fork(snapshot)

        self.assertIs(fork.jobs, self.env.jobs)
        self.assertIs(fork.machines, self.env.machines)
        batch_id = fork.open_batch("PLS21", target_batch_size=2)
        fork.add_wo(batch_id, "WO_A")

        self.assertFalse(self.env.state.runtime.open_batches)
        self.assertIn(batch_id, fork.state.runtime.open_batches)

    def test_phase2_bay_view_exposes_only_assigned_jobs_and_local_machines(self) -> None:
        view = self.env.phase2_bay_view("22")

        self.assertEqual(set(view.jobs), {"WO_A", "WO_B"})
        self.assertEqual(set(view.machines), {"PLS21", "PLS22"})
        with self.assertRaises(RuntimeError):
            self.env.phase2_bay_view("99")

    def test_open_add_close_commits_max_tact_batch_once_and_emits_events(self) -> None:
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)
        self.env.add_wo(batch_id, "WO_A")
        self.assertEqual(self.env.state.runtime.machine_loads["PLS21"]["wo_count"], 0)
        self.env.add_wo(batch_id, "WO_B")

        closed = self.env.close_batch(batch_id)

        self.assertEqual(closed["batch_duration"], 30.0)
        self.assertEqual(closed["start_time"], 0.0)
        self.assertEqual(closed["finish_time"], 30.0)
        self.assertEqual(self.env.state.runtime.machine_available_at["PLS21"], 30.0)
        self.assertEqual(self.env.state.events.current_time, 30.0)
        load = self.env.state.runtime.machine_loads["PLS21"]
        self.assertEqual(load["wo_count"], 2)
        self.assertEqual(load["processing_time_sum"], 40.0)
        self.assertEqual(load["occupancy_time_sum"], 30.0)
        self.assertEqual(load["cut_length_sum"], 700.0)
        self.assertEqual(load["bevel_quantity_sum"], 4)
        self.assertEqual(load["batch_count"], 1)

        rows = [event.to_dict() for event in self.env.state.events.event_log]
        starts = [row for row in rows if row["event_type"] == "PROCESS_START"]
        finishes = [row for row in rows if row["event_type"] == "PROCESS_FINISH"]
        self.assertEqual({row["time_min"] for row in starts}, {0.0})
        self.assertEqual({row["time_min"] for row in finishes}, {30.0})
        self.assertEqual({row["job_id"] for row in starts}, {"WO_A", "WO_B"})

    def test_same_machine_batches_never_overlap(self) -> None:
        first = self.env.open_batch("PLS21", target_batch_size=1)
        self.env.add_wo(first, "WO_A")
        first_row = self.env.close_batch(first)
        second = self.env.open_batch("PLS21", target_batch_size=1)
        self.env.add_wo(second, "WO_B")
        second_row = self.env.close_batch(second)

        self.assertEqual(first_row["finish_time"], second_row["start_time"])
        self.assertEqual(self.env.state.events.current_time, second_row["finish_time"])

    def test_add_wo_rejects_non_finite_job_values(self) -> None:
        self.jobs["WO_A"].plate_length = float("nan")
        batch_id = self.env.open_batch("PLS21", target_batch_size=1)

        with self.assertRaises(RuntimeError):
            self.env.add_wo(batch_id, "WO_A")

    def test_add_wo_rejects_non_finite_processing_time(self) -> None:
        self.jobs["WO_A"].base_stage_minutes = {"cut": float("inf")}
        batch_id = self.env.open_batch("PLS21", target_batch_size=1)

        with self.assertRaises(RuntimeError):
            self.env.add_wo(batch_id, "WO_A")

    @staticmethod
    def _job(job_id, block_set_id, tact, length, cut, bevel):
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            plate_length=length,
            thickness=13.0,
            cut_length=cut,
            bevel_quantity=bevel,
            cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": tact},
        )

    @staticmethod
    def _machine(machine_id, bay_id):
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id=bay_id,
            enabled=True,
            eligible_families=("NP",),
            min_thickness=0.0,
            max_thickness=100.0,
            table_length_limit=55_000.0,
        )

    @staticmethod
    def _bay_load(block_count, steel, cut, bevel, capacity_weight):
        return {
            "block_count": block_count,
            "wo_count": steel,
            "steel_quantity_sum": steel,
            "cut_length_sum": cut,
            "bevel_quantity_sum": bevel,
            "long_cut_bay24_count": 0,
            "capacity_weight": capacity_weight,
        }


if __name__ == "__main__":
    unittest.main()
