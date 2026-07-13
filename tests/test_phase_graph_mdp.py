"""Tests for variable-size graph MDP state builders.

These tests protect the next architecture change:
- Phase 1 should not be hard-coded to Bay 22/23/24 feature slots.
- Phase 2 should work when each Bay has a different number of machines.
"""

from types import SimpleNamespace
import unittest

from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph, build_phase2_wo_machine_graph
from Utils.phase1.multi_series_rules import MULTI_SERIES_RULE_PROFILE


class PhaseGraphMdpTest(unittest.TestCase):
    """Graph state should stay valid when Bay or machine counts change."""

    def test_phase1_graph_supports_bay25_without_fixed_bay_feature_names(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }

        graph = build_phase1_block_bay_graph(jobs=jobs, bay_ids=["22", "23", "24", "25"])

        self.assertEqual(graph["metadata"]["bay_ids"], ["22", "23", "24", "25"])
        self.assertEqual(graph["metadata"]["block_count"], 2)
        self.assertEqual(graph["metadata"]["candidate_edge_count"], 8)
        self.assertIn("25", {node["bay_id"] for node in graph["bay_nodes"]})
        self.assertNotIn("bay_22_flag", graph["feature_names"]["edge"])
        self.assertNotIn("bay_24_flag", graph["feature_names"]["edge"])
        self.assertIn("bay_capacity_weight_ratio", graph["feature_names"]["edge"])
        self.assertNotIn("projected_long_cut_bay24_count_ratio", graph["feature_names"]["edge"])

    def test_phase1_graph_long_cut_mask_removes_bay24_not_all_extra_bays(self) -> None:
        jobs = {
            "WO_LONG": self._job("WO_LONG", "P1::LONG", steel=12, cut=1200.0, bevel=2),
        }

        graph = build_phase1_block_bay_graph(jobs=jobs, bay_ids=["22", "23", "24", "25"])
        target_bays = {edge["target_bay_id"] for edge in graph["candidate_edges"]}

        self.assertEqual(target_bays, {"22", "23", "25"})
        self.assertEqual(graph["metadata"]["candidate_edge_count"], 3)

    def test_phase1_graph_rejects_partial_existing_bay_loads(self) -> None:
        jobs = {"WO_A": self._job("WO_A", "P1::A", steel=1, cut=100.0, bevel=0)}

        with self.assertRaises(RuntimeError):
            build_phase1_block_bay_graph(
                jobs=jobs,
                bay_ids=["22", "23"],
                bay_loads={"22": self._bay_load(wo_count=0, cut=0.0, bevel=0, capacity=1.0)},
            )

    def test_phase1_multi_series_graph_uses_wo_first_capacity_normalized_features(self) -> None:
        jobs = {
            "WO_A": self._multi_series_job("WO_A", "P1::NP::A", cut=1200.0, bevel=3, width=4200.0),
            "WO_B": self._multi_series_job("WO_B", "P1::NP::A", cut=300.0, bevel=1, width=4200.0),
            "WO_C": self._multi_series_job("WO_C", "P1::NP::B", cut=200.0, bevel=2, width=4600.0),
        }
        bay_loads = {
            "22": self._bay_load(wo_count=4, cut=800.0, bevel=4, capacity=4.0),
            "23": self._bay_load(wo_count=2, cut=400.0, bevel=2, capacity=4.0),
            "24": self._bay_load(wo_count=0, cut=0.0, bevel=0, capacity=3.0),
        }

        graph = build_phase1_block_bay_graph(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            bay_loads=bay_loads,
            bay_capacity_weights={"22": 4.0, "23": 4.0, "24": 3.0},
            score_mode="wo_first",
            rule_profile=MULTI_SERIES_RULE_PROFILE,
        )

        self.assertEqual(graph["metadata"]["score_mode"], "wo_first")
        self.assertEqual(graph["metadata"]["rule_profile"], MULTI_SERIES_RULE_PROFILE)
        self.assertIn("wo_count_ratio", graph["feature_names"]["block"])
        self.assertIn("balancing_group_np", graph["feature_names"]["block"])
        self.assertIn("wide_plate_over_4500", graph["feature_names"]["block"])
        self.assertIn("current_wo_per_capacity_ratio", graph["feature_names"]["bay"])
        self.assertIn("projected_wo_gap_ratio", graph["feature_names"]["edge"])
        self.assertNotIn("block_steel_quantity_ratio", graph["feature_names"]["edge"])

        edges_a = [row for row in graph["candidate_edges"] if row["source_block_id"] == "P1::NP::A"]
        edges_b = [row for row in graph["candidate_edges"] if row["source_block_id"] == "P1::NP::B"]
        self.assertEqual({row["target_bay_id"] for row in edges_a}, {"22", "23"})
        self.assertEqual({row["target_bay_id"] for row in edges_b}, {"22", "23"})
        self.assertTrue(all(len(row["features"]) == len(graph["feature_names"]["edge"]) for row in edges_a + edges_b))

    def test_phase2_graph_uses_phase1_bay_assignment_with_variable_machine_counts(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::A", steel=10, cut=900.0, bevel=3),
            "WO_B": self._job("WO_B", "P1::B", steel=5, cut=300.0, bevel=1),
        }
        machines = {
            "PLS21": self._machine("PLS21", "22"),
            "PLS22": self._machine("PLS22", "22"),
            "PLS51": self._machine("PLS51", "25"),
        }

        graph = build_phase2_wo_machine_graph(
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::A": "22", "P1::B": "25"},
        )

        edges_by_job = {}
        for edge in graph["candidate_edges"]:
            edges_by_job.setdefault(edge["source_job_id"], set()).add(edge["target_machine_id"])

        self.assertEqual(edges_by_job["WO_A"], {"PLS21", "PLS22"})
        self.assertEqual(edges_by_job["WO_B"], {"PLS51"})
        self.assertEqual(graph["metadata"]["machine_count"], 3)
        self.assertEqual(graph["metadata"]["candidate_edge_count"], 3)

    def test_phase2_graph_masks_disabled_and_family_ineligible_machines(self) -> None:
        jobs = {
            "WO_NP": self._job("WO_NP", "P1::NP", steel=1, cut=100.0, bevel=0),
            "WO_FL": self._job("WO_FL", "P1::FL", steel=1, cut=100.0, bevel=0),
        }
        jobs["WO_FL"].family = "FL"
        machines = {
            "NP_ENABLED": self._machine("NP_ENABLED", "22", eligible_families=("NP",)),
            "FL_ENABLED": self._machine("FL_ENABLED", "22", eligible_families=("FL",)),
            "BOTH_DISABLED": self._machine(
                "BOTH_DISABLED",
                "22",
                enabled=False,
                eligible_families=("NP", "FL"),
            ),
        }

        graph = build_phase2_wo_machine_graph(
            jobs=jobs,
            machines=machines,
            phase1_assignments={"P1::NP": "22", "P1::FL": "22"},
        )

        edges_by_job = {}
        for edge in graph["candidate_edges"]:
            edges_by_job.setdefault(edge["source_job_id"], set()).add(edge["target_machine_id"])

        self.assertEqual(edges_by_job, {"WO_NP": {"NP_ENABLED"}, "WO_FL": {"FL_ENABLED"}})
        self.assertIn("family_np", graph["feature_names"]["wo"])
        self.assertIn("eligible_family_fl", graph["feature_names"]["machine"])
        self.assertTrue(all(edge["family_eligible"] for edge in graph["candidate_edges"]))

    @staticmethod
    def _job(job_id: str, block_set_id: str, steel: int, cut: float, bevel: int) -> SimpleNamespace:
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family="NP",
            steel_quantity=steel,
            cut_length=cut,
            bevel_quantity=bevel,
            plate_length=10000.0,
            thickness=13.0,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": 10.0},
            extra={"source_project_no": block_set_id.split("::")[0], "source_block_no": block_set_id.split("::")[1]},
        )

    @staticmethod
    def _multi_series_job(
        job_id: str,
        block_set_id: str,
        cut: float,
        bevel: int,
        width: float,
    ) -> SimpleNamespace:
        project_no, family, block_no = block_set_id.split("::")
        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            family=family,
            steel_quantity=1,
            cut_length=cut,
            bevel_quantity=bevel,
            plate_length=10000.0,
            plate_width=width,
            thickness=13.0,
            cut_bay=None,
            source_cut_bay="22",
            allowed_bay_ids=(),
            allowed_machine_ids=(),
            prohibited_machine_ids=(),
            base_stage_minutes={"cut": 10.0},
            extra={"source_project_no": project_no, "source_block_no": block_no},
        )

    @staticmethod
    def _bay_load(wo_count: int, cut: float, bevel: int, capacity: float) -> dict:
        return {
            "steel_quantity_sum": 0,
            "cut_length_sum": cut,
            "bevel_quantity_sum": bevel,
            "long_cut_bay24_count": 0,
            "wo_count": wo_count,
            "block_count": 0,
            "capacity_weight": capacity,
        }

    @staticmethod
    def _machine(
        machine_id: str,
        bay_id: str,
        *,
        enabled: bool = True,
        eligible_families: tuple[str, ...] = ("NP",),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id=bay_id,
            machine_type="PLS",
            enabled=enabled,
            eligible_families=eligible_families,
            min_thickness=0.0,
            max_thickness=100.0,
            table_length_limit=55000.0,
        )


if __name__ == "__main__":
    unittest.main()
