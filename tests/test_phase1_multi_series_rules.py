from __future__ import annotations

import unittest

from Utils.phase1.multi_series_rules import (
    apply_phase1_series_bay_mask,
    balancing_group_for_series,
    bay_capacity_weights_for_group,
)


class Phase1MultiSeriesRulesTest(unittest.TestCase):
    def test_balancing_groups_and_latest_capacity_contract(self) -> None:
        self.assertEqual(balancing_group_for_series("NP"), "NP")
        self.assertEqual(balancing_group_for_series("FN"), "FN_FL")
        self.assertEqual(balancing_group_for_series("FL"), "FN_FL")
        self.assertEqual(balancing_group_for_series("NC"), "NC")
        self.assertEqual(
            bay_capacity_weights_for_group("NP"),
            {"22": 4.0, "23": 4.0, "24": 3.0},
        )

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
