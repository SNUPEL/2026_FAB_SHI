import csv
import tempfile
import unittest
from pathlib import Path

from Utils.data.cutting_data_loader import load_and_clean_cutting_data
from Utils.data.io import load_scenario_for_config


class SourceDataProcessTimeFilterTest(unittest.TestCase):
    def test_tact_time_source_excludes_non_positive_tact_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "sample.csv"
            self._write_rows(source_path)
            config = self._config(source_path, process_time_source="tact_time")

            scenario = load_scenario_for_config(config)

            self.assertEqual(scenario["metadata"]["selected_records"], 1)
            self.assertEqual(scenario["metadata"]["excluded_records"], 1)
            self.assertEqual(scenario["jobs"][0]["source_wk_ord_no"], "WO_POSITIVE_TACT")

    def test_actual_duration_source_keeps_zero_tact_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "sample.csv"
            self._write_rows(source_path)
            config = self._config(source_path, process_time_source="actual_duration")

            scenario = load_scenario_for_config(config)

            self.assertEqual(scenario["metadata"]["selected_records"], 2)
            self.assertEqual(scenario["metadata"]["excluded_records"], 0)

    def test_missing_steel_quantity_is_excluded_without_default_one(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = Path(tmp_dir) / "missing_steel_qty.csv"
            self._write_rows_without_steel_quantity(source_path)

            cleaned = load_and_clean_cutting_data(str(source_path), target_series=("NP",))

            self.assertEqual(len(cleaned.records), 0)
            self.assertEqual(len(cleaned.excluded_records), 1)
            self.assertEqual(cleaned.excluded_records[0]["exclude_reason"], "missing_steel_qty")

    def _write_rows(self, path: Path) -> None:
        rows = [
            {
                "PROJ_NO": "PROJ_1",
                "BLK_NO": "BLK_1",
                "WK_ORD_NO": "WO_ZERO_TACT",
                "GYEL_ACT_STDT": "20260319",
                "GYEL_ACT_EDDT": "20260319",
                "RT_CUT_ST_DTM": "202603190830",
                "RT_CUT_ED_DTM": "202603190900",
                "GYEL": "NP",
                "LTH": "7000",
                "THK": "12",
                "CUT_LTH": "70",
                "MARK_LTH": "0",
                "BVL_LTH": "0",
                "STL_QTY": "1",
                "RT_EQP_NM": "PLS21",
                "CUT_BAY": "22",
                "TACT_TIME": "0",
                "ASS_ST_DT": "20260320",
                "PTLST_QTY": "10",
            },
            {
                "PROJ_NO": "PROJ_1",
                "BLK_NO": "BLK_2",
                "WK_ORD_NO": "WO_POSITIVE_TACT",
                "GYEL_ACT_STDT": "20260319",
                "GYEL_ACT_EDDT": "20260319",
                "RT_CUT_ST_DTM": "202603190910",
                "RT_CUT_ED_DTM": "202603190950",
                "GYEL": "NP",
                "LTH": "8000",
                "THK": "13",
                "CUT_LTH": "80",
                "MARK_LTH": "0",
                "BVL_LTH": "0",
                "STL_QTY": "1",
                "RT_EQP_NM": "PLS21",
                "CUT_BAY": "22",
                "TACT_TIME": "30",
                "ASS_ST_DT": "20260320",
                "PTLST_QTY": "12",
            },
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_rows_without_steel_quantity(self, path: Path) -> None:
        rows = [
            {
                "PROJ_NO": "PROJ_1",
                "BLK_NO": "BLK_1",
                "WK_ORD_NO": "WO_MISSING_STEEL",
                "GYEL_ACT_STDT": "20260319",
                "GYEL_ACT_EDDT": "20260319",
                "RT_CUT_ST_DTM": "202603190830",
                "RT_CUT_ED_DTM": "202603190900",
                "GYEL": "NP",
                "LTH": "7000",
                "THK": "12",
                "CUT_LTH": "70",
                "MARK_LTH": "0",
                "BVL_LTH": "0",
                "RT_EQP_NM": "PLS21",
                "CUT_BAY": "22",
                "TACT_TIME": "10",
                "ASS_ST_DT": "20260320",
                "PTLST_QTY": "10",
            },
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _config(self, source_path: Path, process_time_source: str) -> dict:
        return {
            "paths": {
                "scenario_path": None,
                "source_data_path": str(source_path),
                "source_sheet_name": "Sheet1",
                "output_dir": str(source_path.parent),
            },
            "data": {
                "target_series": ["NP"],
                "max_records": None,
                "process_time_source": process_time_source,
            },
            "factory": {
                "bays": [
                    {
                        "bay_id": "22",
                        "equipment": {"PLS": {"machine_ids": ["PLS21"]}},
                    }
                ]
            },
        }


if __name__ == "__main__":
    unittest.main()
