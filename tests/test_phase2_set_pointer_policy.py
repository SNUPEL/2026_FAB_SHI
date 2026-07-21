"""Phase 2 heterogeneous set state와 pointer scorer 검증."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from Environment.constraints.profiles import default_phase2_constraint_profile
from Environment.hierarchical import CommonHierarchicalEnvironment
from Phase2.set_pointer_policy import Phase2SetPointerPolicy
from Phase2.state import (
    PHASE2_BAY_CONTEXT_FEATURE_NAMES,
    PHASE2_MACHINE_NODE_FEATURE_NAMES,
    PHASE2_OPEN_BATCH_FEATURE_NAMES,
    PHASE2_PROJECTED_FEATURE_NAMES,
    PHASE2_WO_NODE_FEATURE_NAMES,
    build_phase2_policy_state,
)


class Phase2SetPointerPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", 30.0, 500.0, 3),
            "WO_B": self._job("WO_B", 10.0, 200.0, 1),
            "WO_C": self._job("WO_C", 20.0, 300.0, 2),
        }
        machines = {
            "PLS21": self._machine("PLS21"),
            "PLS22": self._machine("PLS22"),
        }
        self.env = CommonHierarchicalEnvironment(
            jobs=jobs,
            machines=machines,
            bay_capacity_weights={"22": 2.0},
            constraint_profile=default_phase2_constraint_profile(),
            max_batch_wo_count=3,
            max_batch_length_sum=55_000.0,
        )
        self.env.commit_phase1(
            block_to_bay={"P1::A": "22"},
            bay_loads={
                "22": {
                    "block_count": 1,
                    "wo_count": 3,
                    "steel_quantity_sum": 3,
                    "cut_length_sum": 1000.0,
                    "bevel_quantity_sum": 6,
                    "long_cut_bay24_count": 0,
                    "capacity_weight": 2.0,
                }
            },
        )

    def test_feature_schema_contains_variable_sets_and_projected_loads(self) -> None:
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)
        actions = [
            self._wo_action("WO_A", duration=30.0, cut=500.0, bevel=3.0, wo_count=1),
            self._wo_action("WO_B", duration=10.0, cut=200.0, bevel=1.0, wo_count=1),
        ]
        state = build_phase2_policy_state(
            environment=self.env,
            bay_id="22",
            actions=actions,
            selected_machine_id="PLS21",
            open_batch_id=batch_id,
        )

        self.assertEqual(len(state.bay_context_features), len(PHASE2_BAY_CONTEXT_FEATURE_NAMES))
        self.assertEqual(len(state.machine_node_features), 2)
        self.assertEqual(len(state.machine_node_features[0]), len(PHASE2_MACHINE_NODE_FEATURE_NAMES))
        self.assertEqual(len(state.wo_node_features), 3)
        self.assertEqual(len(state.wo_node_features[0]), len(PHASE2_WO_NODE_FEATURE_NAMES))
        self.assertEqual(len(state.open_batch_features), len(PHASE2_OPEN_BATCH_FEATURE_NAMES))
        self.assertEqual(len(state.action_projected_features[0]), len(PHASE2_PROJECTED_FEATURE_NAMES))
        self.assertEqual(state.action_candidate_node_indices, [0, 1])
        self.assertIn("family_np", PHASE2_WO_NODE_FEATURE_NAMES)
        self.assertIn("eligible_family_np", PHASE2_MACHINE_NODE_FEATURE_NAMES)
        self.assertEqual(
            state.wo_node_features[0][PHASE2_WO_NODE_FEATURE_NAMES.index("family_np")],
            1.0,
        )
        self.assertEqual(
            state.machine_node_features[0][PHASE2_MACHINE_NODE_FEATURE_NAMES.index("eligible_family_np")],
            1.0,
        )

    def test_machine_feasible_ratio_uses_family_eligibility(self) -> None:
        self.env.jobs["WO_C"].family = "FL"
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)

        state = build_phase2_policy_state(
            environment=self.env,
            bay_id="22",
            actions=[self._wo_action("WO_A", duration=30.0, cut=500.0, bevel=3.0, wo_count=1)],
            selected_machine_id="PLS21",
            open_batch_id=batch_id,
        )

        ratio_index = PHASE2_MACHINE_NODE_FEATURE_NAMES.index("feasible_remaining_wo_ratio")
        self.assertEqual(state.machine_node_features[0][ratio_index], 2.0 / 3.0)

    def test_pointer_scores_are_equivariant_to_wo_and_action_permutation(self) -> None:
        torch.manual_seed(7)
        model = Phase2SetPointerPolicy(hidden_dim=16)
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)
        state = build_phase2_policy_state(
            environment=self.env,
            bay_id="22",
            actions=[
                self._wo_action("WO_A", duration=30.0, cut=500.0, bevel=3.0, wo_count=1),
                self._wo_action("WO_B", duration=10.0, cut=200.0, bevel=1.0, wo_count=1),
            ],
            selected_machine_id="PLS21",
            open_batch_id=batch_id,
        )
        original = model(state)

        permuted = replace(
            state,
            wo_ids=list(reversed(state.wo_ids)),
            wo_node_features=list(reversed(state.wo_node_features)),
            action_ids=list(reversed(state.action_ids)),
            action_projected_features=list(reversed(state.action_projected_features)),
            action_candidate_node_indices=[1, 2],
        )
        reordered = model(permuted)

        self.assertTrue(torch.allclose(original, torch.flip(reordered, dims=[0]), atol=1e-6))

    def test_wo_state_reflects_open_batch_and_projected_gap(self) -> None:
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)
        self.env.add_wo(batch_id, "WO_A")
        actions = [
            self._wo_action("WO_B", duration=30.0, cut=700.0, bevel=4.0, wo_count=2),
            self._wo_action("WO_C", duration=40.0, cut=800.0, bevel=5.0, wo_count=2),
        ]
        state = build_phase2_policy_state(
            environment=self.env,
            bay_id="22",
            actions=actions,
            selected_machine_id="PLS21",
            open_batch_id=batch_id,
        )
        model = Phase2SetPointerPolicy(hidden_dim=16)

        self.assertEqual(len(state.open_batch_features), len(PHASE2_OPEN_BATCH_FEATURE_NAMES))
        self.assertGreater(state.open_batch_features[0], 0.0)
        self.assertEqual(len(model(state)), 2)
        self.assertNotEqual(state.action_projected_features[0], state.action_projected_features[1])

    def test_state_rejects_missing_machine_load_field(self) -> None:
        del self.env.state.runtime.machine_loads["PLS21"]["batch_count"]
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)

        with self.assertRaises(RuntimeError):
            build_phase2_policy_state(
                environment=self.env,
                bay_id="22",
                actions=[self._wo_action("WO_A", duration=30.0, cut=500.0, bevel=3.0, wo_count=1)],
                selected_machine_id="PLS21",
                open_batch_id=batch_id,
            )

    def test_state_rejects_non_finite_processing_time(self) -> None:
        self.env.jobs["WO_A"].base_stage_minutes = {"cut": float("nan")}
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)

        with self.assertRaises(RuntimeError):
            build_phase2_policy_state(
                environment=self.env,
                bay_id="22",
                actions=[self._wo_action("WO_A", duration=30.0, cut=500.0, bevel=3.0, wo_count=1)],
                selected_machine_id="PLS21",
                open_batch_id=batch_id,
            )

    def test_state_rejects_missing_projected_action_field(self) -> None:
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)
        self.env.add_wo(batch_id, "WO_A")
        actions = [self._wo_action("WO_B", duration=30.0, cut=700.0, bevel=4.0, wo_count=2)]
        del actions[0]["duration_increment"]

        with self.assertRaises(RuntimeError):
            build_phase2_policy_state(
                environment=self.env,
                bay_id="22",
                actions=actions,
                selected_machine_id="PLS21",
                open_batch_id=batch_id,
            )

    def test_state_rejects_machine_selection_action(self) -> None:
        batch_id = self.env.open_batch("PLS21", target_batch_size=2)

        with self.assertRaises(RuntimeError):
            build_phase2_policy_state(
                environment=self.env,
                bay_id="22",
                actions=[
                    {
                        "action_type": "select_machine",
                        "machine_id": "PLS21",
                        "job_ids": (),
                    }
                ],
                selected_machine_id="PLS21",
                open_batch_id=batch_id,
            )

    @staticmethod
    def _wo_action(job_id, duration, cut, bevel, wo_count):
        return {
            "action_id": f"wo:{job_id}",
            "action_type": "select_wo",
            "machine_id": "PLS21",
            "job_ids": (job_id,),
            "target_batch_size": 2,
            "projected_finish_time": duration,
            "projected_makespan": duration,
            "duration_increment": 0.0,
            "wo_count": wo_count,
            "cut_length_sum": cut,
            "bevel_quantity_sum": bevel,
            "batch_duration": duration,
        }

    @staticmethod
    def _job(job_id, tact, cut, bevel):
        return SimpleNamespace(
            job_id=job_id,
            block_set_id="P1::A",
            family="NP",
            plate_length=10_000.0,
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
    def _machine(machine_id):
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id="22",
            enabled=True,
            eligible_families=("NP",),
            min_thickness=0.0,
            max_thickness=100.0,
            table_length_limit=55_000.0,
        )


if __name__ == "__main__":
    unittest.main()
