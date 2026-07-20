"""MIXED Phase 1 block-series 집계와 Bay 배정 회귀시험."""

from __future__ import annotations

import unittest

from Phase1.heuristics import (
    PHASE1_HEURISTIC_BANK,
    run_phase1_heuristic_candidate,
    score_phase1_bay_loads,
)
from Phase1.orchestrator import candidate_to_phase1_plan
from Utils.phase1.multi_series_rules import joint_phase1_bay_capacity_weights
from Utils.phase1.phase1_bay_balancer import apply_phase1_plan_to_scenario


class Phase1BayBalancerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.weights = joint_phase1_bay_capacity_weights()
        self.bay_ids = tuple(self.weights)
        self.jobs = {
            "WO_NP_1": self._job("WO_NP_1", "P1::NP::BLK_1", "NP", cut=700.0),
            "WO_NP_2": self._job("WO_NP_2", "P1::NP::BLK_1", "NP", cut=350.0),
            "WO_NC_1": self._job("WO_NC_1", "P2::NC::BLK_2", "NC", cut=200.0),
            "WO_FN_1": self._job("WO_FN_1", "P3::FN::BLK_3", "FN", cut=120.0),
            "WO_FL_1": self._job("WO_FL_1", "P4::FL::BLK_4", "FL", cut=180.0),
        }

    def test_all_public_heuristics_assign_every_block_with_same_hard_mask(self) -> None:
        for algorithm in PHASE1_HEURISTIC_BANK:
            candidate = run_phase1_heuristic_candidate(
                jobs=self.jobs,
                bay_ids=self.bay_ids,
                algorithm=algorithm,
                bay_capacity_weights=self.weights,
            )

            self.assertEqual(len(candidate.assignments), 4)
            self.assertIn(candidate.assignments["P1::NP::BLK_1"], {"22", "23"})
            self.assertIn(candidate.assignments["P2::NC::BLK_2"], {"22", "23", "24"})
            self.assertIn(candidate.assignments["P3::FN::BLK_3"], {"25", "trans"})
            self.assertIn(candidate.assignments["P4::FL::BLK_4"], {"25", "trans"})
            self.assertEqual(set(candidate.bay_loads), set(self.bay_ids))

    def test_np_long_cut_wide_plate_and_cnt_masks_exclude_bay24(self) -> None:
        jobs = {
            "LONG": self._job("LONG", "P1::NP::BLK_LONG", "NP", cut=1_000.0),
            "WIDE": self._job(
                "WIDE", "P2::NP::BLK_WIDE", "NP", cut=100.0, width=4_501.0
            ),
            "CNT": self._job("CNT", "P3::NP::CNT_BLK_1", "NP", cut=100.0),
        }
        candidate = run_phase1_heuristic_candidate(
            jobs=jobs,
            bay_ids=self.bay_ids,
            algorithm="wo_first_balanced",
            bay_capacity_weights=self.weights,
        )

        self.assertTrue(set(candidate.assignments.values()) <= {"22", "23"})

    def test_same_block_series_work_orders_receive_one_committed_bay(self) -> None:
        candidate = run_phase1_heuristic_candidate(
            jobs=self.jobs,
            bay_ids=self.bay_ids,
            algorithm="wo_first_balanced",
            bay_capacity_weights=self.weights,
        )
        plan = candidate_to_phase1_plan(self.jobs, self.bay_ids, candidate)
        scenario = {"jobs": [dict(job) for job in self.jobs.values()]}

        applied = apply_phase1_plan_to_scenario(scenario, plan)["scenario"]
        np_bays = {
            tuple(row["allowed_bay_ids"])
            for row in applied["jobs"]
            if row["block_set_id"] == "P1::NP::BLK_1"
        }

        self.assertEqual(len(np_bays), 1)
        self.assertEqual(len(next(iter(np_bays))), 1)
        self.assertEqual(plan["rule_profile"], "multi_series_260711")
        self.assertEqual(plan["scope_version"], "joint_five_bay_v3_shared_pool")
        self.assertEqual(plan["score_mode"], "wo_first")
        self.assertEqual(len(plan["score"]), 6)

    def test_shared_pool_score_lets_nc_compensate_np_bay24_underload(self) -> None:
        compensated = self._bay_loads(
            np_wo={"22": 40, "23": 30, "24": 18},
            nc_wo={"22": 0, "23": 0, "24": 22},
        )
        series_balanced = self._bay_loads(
            np_wo={"22": 40, "23": 30, "24": 18},
            nc_wo={"22": 8, "23": 6, "24": 8},
        )

        compensated_score = score_phase1_bay_loads(compensated)
        series_balanced_score = score_phase1_bay_loads(series_balanced)

        self.assertEqual(compensated_score, (0.0, 11.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(series_balanced_score, (5.5, 5.5, 0.0, 0.0, 0.0, 0.0))
        self.assertLess(compensated_score, series_balanced_score)

    def test_series_only_score_excludes_shared_pool_components(self) -> None:
        compensated = self._bay_loads(
            np_wo={"22": 40, "23": 30, "24": 18},
            nc_wo={"22": 0, "23": 0, "24": 22},
        )
        series_balanced = self._bay_loads(
            np_wo={"22": 40, "23": 30, "24": 18},
            nc_wo={"22": 8, "23": 6, "24": 8},
        )

        compensated_score = score_phase1_bay_loads(
            compensated,
            objective_scope="series_only",
        )
        series_balanced_score = score_phase1_bay_loads(
            series_balanced,
            objective_scope="series_only",
        )

        self.assertEqual(compensated_score, (11.0, 0.0, 0.0))
        self.assertEqual(series_balanced_score, (5.5, 0.0, 0.0))
        self.assertLess(series_balanced_score, compensated_score)

    def test_invalid_objective_scope_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            score_phase1_bay_loads(
                self._bay_loads(
                    np_wo={"22": 1, "23": 1, "24": 1},
                    nc_wo={"22": 0, "23": 0, "24": 0},
                ),
                objective_scope="unknown",
            )

    def test_mixed_family_inside_one_block_series_fails(self) -> None:
        jobs = {
            "WO_A": self._job("WO_A", "P1::NP::BLK_1", "NP", cut=100.0),
            "WO_B": self._job("WO_B", "P1::NP::BLK_1", "FL", cut=100.0),
        }

        with self.assertRaises(RuntimeError):
            run_phase1_heuristic_candidate(
                jobs=jobs,
                bay_ids=self.bay_ids,
                algorithm="wo_first_balanced",
                bay_capacity_weights=self.weights,
            )

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        family: str,
        *,
        cut: float,
        width: float = 3_000.0,
    ) -> dict:
        return {
            "job_id": job_id,
            "block_set_id": block_set_id,
            "family": family,
            "steel_quantity": 1,
            "cut_length": cut,
            "bevel_quantity": 2,
            "plate_length": 10_000.0,
            "plate_width": width,
            "thickness": 20.0,
            "processing_time": 30.0,
            "base_stage_minutes": {"cut": 30.0},
            "cut_bay": None,
            "source_cut_bay": None,
            "allowed_bay_ids": (),
            "allowed_machine_ids": (),
            "prohibited_machine_ids": (),
            "extra": {"source_wk_ord_no": job_id},
        }

    def _bay_loads(self, *, np_wo: dict[str, int], nc_wo: dict[str, int]) -> dict:
        loads = {}
        for bay_id, capacity_weight in self.weights.items():
            row = {
                "capacity_weight": capacity_weight,
                "wo_count": np_wo.get(bay_id, 0) + nc_wo.get(bay_id, 0),
                "cut_length_sum": 0.0,
                "bevel_quantity_sum": 0,
            }
            for group in ("np", "fn_fl", "nc"):
                row[f"group_{group}_wo_count"] = 0
                row[f"group_{group}_cut_length_sum"] = 0.0
                row[f"group_{group}_bevel_quantity_sum"] = 0
            row["group_np_wo_count"] = np_wo.get(bay_id, 0)
            row["group_nc_wo_count"] = nc_wo.get(bay_id, 0)
            loads[bay_id] = row
        return loads


if __name__ == "__main__":
    unittest.main()
