"""Tests for variable-size graph MDP state builders.

These tests protect the next architecture change:
- Phase 1 should not be hard-coded to Bay 22/23/24 feature slots.
- Phase 2 should work when each Bay has a different number of machines.
"""

from types import SimpleNamespace
import unittest

from Utils.learning.phase_graph_mdp import build_phase1_block_bay_graph, build_phase2_wo_machine_graph


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
    def _machine(machine_id: str, bay_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            machine_id=machine_id,
            bay_id=bay_id,
            machine_type="PLS",
            enabled=True,
            eligible_families=("NP",),
            min_thickness=0.0,
            max_thickness=100.0,
            table_length_limit=55000.0,
        )


if __name__ == "__main__":
    unittest.main()
