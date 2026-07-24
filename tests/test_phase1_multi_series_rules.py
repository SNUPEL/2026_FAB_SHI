from __future__ import annotations

import unittest

from Utils.data.multi_series_cutting_data import (
    MIXED_PLANNING_MACHINE_IDS_BY_BAY,
    resolve_actual_equipment,
)
from Utils.phase1.multi_series_rules import (
    apply_phase1_series_bay_mask,
    balancing_group_for_series,
    bay_capacity_weights_for_group,
    phase1_resource_pool_for_series,
)


class Phase1MultiSeriesRulesTest(unittest.TestCase):
    def test_balancing_groups_and_latest_capacity_contract(self) -> None:
        self.assertEqual(balancing_group_for_series("NP"), "NP")
        self.assertEqual(balancing_group_for_series("FN"), "FN")
        self.assertEqual(balancing_group_for_series("FL"), "FL")
        self.assertEqual(balancing_group_for_series("NC"), "NC")
        self.assertEqual(
            bay_capacity_weights_for_group("NP"),
            {"22": 4.0, "23": 3.0, "24": 4.0},
        )
        self.assertEqual(
            bay_capacity_weights_for_group("FN"),
            {"25": 2.0, "trans": 2.0},
        )
        self.assertEqual(
            bay_capacity_weights_for_group("FL"),
            {"25": 2.0, "trans": 2.0},
        )

    def test_series_are_partitioned_into_two_independent_resource_pools(self) -> None:
        self.assertEqual(phase1_resource_pool_for_series("NP"), "NP_NC")
        self.assertEqual(phase1_resource_pool_for_series("NC"), "NP_NC")
        self.assertEqual(phase1_resource_pool_for_series("FN"), "FN_FL")
        self.assertEqual(phase1_resource_pool_for_series("FL"), "FN_FL")

    def test_confirmed_eqp_mapping_and_planning_machine_scope(self) -> None:
        self.assertEqual(
            MIXED_PLANNING_MACHINE_IDS_BY_BAY,
            {
                "22": ("PLS21", "PLS22", "PLS23", "PLS24"),
                "23": ("PLS31", "PLS32", "PLS33"),
                "24": ("PLS41", "PLS42", "PLS43", "PLS44"),
                "25": ("PLS51", "PLS52"),
                "trans": ("PLP01", "PLP02"),
            },
        )
        expected = {
            "EQP_1": "PLS51",
            "EQP_2": "PLS52",
            "EQP_4": "PLS21",
            "EQP_5": "PLS22",
            "EQP_6": "PLS23",
            "EQP_7": "PLS24",
            "EQP_8": "PLS31",
            "EQP_9": "PLS32",
            "EQP_10": "PLS33",
            "EQP_11": "PLS41",
            "EQP_12": "PLS42",
            "EQP_13": "PLS43",
            "EQP_14": "PLS44",
            "EQP_15": "PLP01",
            "EQP_16": "PLP02",
        }
        for eqp_id, machine_id in expected.items():
            resolution = resolve_actual_equipment(eqp_id, series="NP", cut_bay="22")
            self.assertEqual(resolution.machine_id, machine_id)
            self.assertTrue(resolution.planning_candidate)
            self.assertEqual(resolution.mapping_status, "mapped_pls_plp")

    def test_eqp3_is_nc_trans_actual_only_and_not_a_planning_machine(self) -> None:
        resolution = resolve_actual_equipment("EQP_3", series="NC", cut_bay="trans")

        self.assertEqual(resolution.machine_id, "EQP_3")
        self.assertEqual(resolution.home_bay, "trans")
        self.assertFalse(resolution.planning_candidate)
        self.assertEqual(resolution.mapping_status, "actual_nc_trans_only")
        with self.assertRaisesRegex(RuntimeError, "EQP_3 actual contract mismatch"):
            resolve_actual_equipment("EQP_3", series="NP", cut_bay="trans")
        with self.assertRaisesRegex(RuntimeError, "EQP_3 actual contract mismatch"):
            resolve_actual_equipment("EQP_3", series="NC", cut_bay="22")

    def test_unknown_eqp_fails_without_mapping_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unknown actual EQP id"):
            resolve_actual_equipment("EQP_99", series="NC", cut_bay="trans")

    def test_np_hard_masks_use_confirmed_boundaries(self) -> None:
        ordinary = apply_phase1_series_bay_mask(
            series="NP",
            block_no="BLK_1",
            width_max=4500,
            cut_length_sum=999.999,
            requested_bays=("22", "23", "24", "25"),
        )
        wide = apply_phase1_series_bay_mask(
            series="NP",
            block_no="BLK_2",
            width_max=4500.001,
            cut_length_sum=100,
            requested_bays=("22", "23", "24"),
        )
        long_cut = apply_phase1_series_bay_mask(
            series="NP",
            block_no="BLK_3",
            width_max=4000,
            cut_length_sum=1000,
            requested_bays=("22", "23", "24"),
        )
        cnt = apply_phase1_series_bay_mask(
            series="NP",
            block_no="CNT_BLK_9",
            width_max=4000,
            cut_length_sum=100,
            requested_bays=("22", "23", "24"),
        )

        self.assertEqual(ordinary.allowed_bay_ids, ("22", "23", "24"))
        self.assertEqual(wide.allowed_bay_ids, ("22", "23"))
        self.assertEqual(long_cut.allowed_bay_ids, ("22", "23"))
        self.assertEqual(cnt.allowed_bay_ids, ("22", "23"))
        self.assertIn("np_wide_bth_gt_4500", wide.reason_codes)
        self.assertIn("np_cut_lth_ge_1000", long_cut.reason_codes)
        self.assertIn("np_cnt_block", cnt.reason_codes)

    def test_series_eligibility_is_not_derived_from_actual_cut_bay(self) -> None:
        fn = apply_phase1_series_bay_mask(
            series="FN",
            block_no="BLK_1",
            width_max=3000,
            cut_length_sum=100,
            requested_bays=("22", "23", "24", "25", "trans"),
        )
        nc = apply_phase1_series_bay_mask(
            series="NC",
            block_no="BLK_2",
            width_max=3000,
            cut_length_sum=100,
            requested_bays=("22", "23", "24", "25", "trans"),
        )

        self.assertEqual(fn.allowed_bay_ids, ("25", "trans"))
        self.assertEqual(nc.allowed_bay_ids, ("22", "23", "24"))

    def test_empty_candidate_intersection_fails_without_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no feasible Phase 1 Bay"):
            apply_phase1_series_bay_mask(
                series="FL",
                block_no="BLK_1",
                width_max=3000,
                cut_length_sum=100,
                requested_bays=("22", "23", "24"),
            )


if __name__ == "__main__":
    unittest.main()
