"""MIXED Phase 1/2 가변 크기 graph state 회귀시험."""

from __future__ import annotations

import unittest

from Phase1.heuristics import run_phase1_heuristic_candidate
from Phase2.merged import build_mixed_phase2_training_machines
from Utils.learning.phase_graph_mdp import (
    PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES,
    build_phase1_block_bay_graph,
    build_phase2_wo_machine_graph,
)
from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights


class PhaseGraphMdpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = joint_phase1_bay_capacity_weights()
        self.jobs = {
            "WO_NP": self._job("WO_NP", "P1::NP::BLK_1", "NP", 1_200.0),
            "WO_NC": self._job("WO_NC", "P2::NC::BLK_2", "NC", 300.0),
            "WO_FN": self._job("WO_FN", "P3::FN::BLK_3", "FN", 200.0),
            "WO_FL": self._job("WO_FL", "P4::FL::BLK_4", "FL", 100.0),
        }

    def test_phase1_graph_has_five_bays_and_only_feasible_edges(self) -> None:
        graph = build_phase1_block_bay_graph(
            jobs=self.jobs,
            bay_ids=tuple(self.weights),
            bay_capacity_weights=self.weights,
        )

        self.assertEqual(graph["metadata"]["rule_profile"], "multi_series_260711")
        self.assertEqual(graph["metadata"]["score_mode"], "wo_first")
        self.assertEqual(graph["metadata"]["bay_ids"], list(self.weights))
        self.assertEqual(len(graph["bay_nodes"]), 5)
        self.assertEqual(
            graph["feature_names"]["edge"],
            PHASE1_MULTI_SERIES_BLOCK_BAY_EDGE_FEATURES,
        )
        by_block: dict[str, set[str]] = {}
        for edge in graph["candidate_edges"]:
            by_block.setdefault(edge["source_block_id"], set()).add(edge["target_bay_id"])
        self.assertEqual(by_block["P1::NP::BLK_1"], {"22", "23"})
        self.assertEqual(by_block["P2::NC::BLK_2"], {"22", "23", "24"})
        self.assertEqual(by_block["P3::FN::BLK_3"], {"25", "trans"})
        self.assertEqual(by_block["P4::FL::BLK_4"], {"25", "trans"})

    def test_phase1_graph_removes_assigned_block_without_changing_schema(self) -> None:
        graph = build_phase1_block_bay_graph(
            jobs=self.jobs,
            bay_ids=tuple(self.weights),
            assigned_block_ids={"P1::NP::BLK_1"},
            bay_capacity_weights=self.weights,
        )

        self.assertEqual(graph["metadata"]["block_count"], 4)
        self.assertEqual(graph["metadata"]["unassigned_block_count"], 3)
        self.assertNotIn(
            "P1::NP::BLK_1",
            {edge["source_block_id"] for edge in graph["candidate_edges"]},
        )

    def test_phase1_graph_requires_capacity_contract(self) -> None:
        with self.assertRaises(RuntimeError):
            build_phase1_block_bay_graph(self.jobs, tuple(self.weights))

    def test_phase2_graph_connects_each_wo_only_to_its_phase1_bay_and_family(self) -> None:
        phase1 = run_phase1_heuristic_candidate(
            jobs=self.jobs,
            bay_ids=tuple(self.weights),
            algorithm="wo_first_balanced",
            bay_capacity_weights=self.weights,
        )
        machines = build_mixed_phase2_training_machines()
        graph = build_phase2_wo_machine_graph(
            jobs=self.jobs,
            machines=machines,
            phase1_assignments=phase1.assignments,
        )

        self.assertEqual(len(graph["machine_nodes"]), 15)
        self.assertGreater(len(graph["candidate_edges"]), 0)
        for edge in graph["candidate_edges"]:
            job = self.jobs[edge["source_job_id"]]
            machine = machines[edge["target_machine_id"]]
            self.assertEqual(edge["target_bay_id"], phase1.assignments[job["block_set_id"]])
            self.assertIn(job["family"], machine.eligible_families)
            self.assertTrue(edge["family_eligible"])

    @staticmethod
    def _job(job_id: str, block_set_id: str, family: str, cut_length: float) -> dict:
        return {
            "job_id": job_id,
            "block_set_id": block_set_id,
            "family": family,
            "steel_quantity": 1,
            "plate_length": 10_000.0,
            "plate_width": 3_000.0,
            "thickness": 20.0,
            "cut_length": cut_length,
            "bevel_quantity": 2,
            "processing_time": 30.0,
            "base_stage_minutes": {"cut": 30.0},
            "cut_bay": None,
            "source_cut_bay": None,
            "allowed_bay_ids": (),
            "allowed_machine_ids": (),
            "prohibited_machine_ids": (),
            "extra": {"source_wk_ord_no": job_id},
        }


if __name__ == "__main__":
    unittest.main()
