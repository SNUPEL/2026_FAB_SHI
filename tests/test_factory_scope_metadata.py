import unittest

from Utils.data.cutting_scenario_builder import build_scenario_from_cutting_records


class FactoryScopeMetadataTest(unittest.TestCase):
    def test_factory_scope_uses_configured_bays_and_machine_ids(self):
        records = [
            {
                "work_order_no": "WO_A",
                "project_no": "PROJ_1",
                "block_no": "BLK_1",
                "source_row_index": 1,
                "block_set_id": "PROJ_1::BLK_1",
                "series": "NP",
                "thickness": 12.0,
                "length": 8000.0,
                "cut_length": 80.0,
                "mark_length": 0.0,
                "bevel_length": 0.0,
                "steel_qty": 1,
                "part_qty": 10,
                "tact_time": 30.0,
                "actual_duration_minutes": 35.0,
                "actual_start_minutes": 0.0,
                "actual_finish_minutes": 35.0,
                "planned_start_date": "20260319",
                "planned_end_date": "20260319",
                "actual_start_datetime": "202603190830",
                "actual_end_datetime": "202603190905",
                "source_machine_id": "PLS41",
                "source_cut_bay": "24",
            }
        ]
        factory_config = {
            "bays": [
                {
                    "bay_id": "22",
                    "equipment": {"PLS": {"machine_ids": ["PLS21", "PLS22"]}},
                },
                {
                    "bay_id": "24",
                    "equipment": {"PLS": {"machine_ids": ["PLS41"]}},
                },
            ]
        }

        scenario = build_scenario_from_cutting_records(records, factory_config=factory_config)

        self.assertEqual(scenario["metadata"]["factory_scope"], "Bay 22: PLS21,PLS22; Bay 24: PLS41")


if __name__ == "__main__":
    unittest.main()
