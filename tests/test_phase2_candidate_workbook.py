import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Utils.data.phase2_candidate_workbook import load_phase2_candidate_workbook_problems


class Phase2CandidateWorkbookTest(unittest.TestCase):
    def test_candidate_workbook_blocks_expand_to_all_matching_wo_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_path = base / "candidate.xlsx"
            wo_path = base / "wo.xlsx"
            self._write_candidate(candidate_path, [("PROJ_A", "BLK_1", 22), ("PROJ_A", "BLK_2", 23)])
            self._write_wo(
                wo_path,
                [
                    self._wo("PROJ_A", "BLK_1", "WO_1", 22),
                    self._wo("PROJ_A", "BLK_1", "WO_2", 22),
                    self._wo("PROJ_A", "BLK_2", "WO_3", 23),
                    self._wo("PROJ_Z", "BLK_X", "WO_4", 24),
                ],
            )

            problems = load_phase2_candidate_workbook_problems(
                wo_path=wo_path,
                candidate_path=candidate_path,
                workdays=("20260331",),
                bay_ids=("22", "23", "24"),
                gyel="NP",
            )

        problem = problems[0]
        self.assertEqual(problem["workday"], "20260331")
        self.assertEqual(problem["candidate_block_count"], 2)
        self.assertEqual(problem["wo_count"], 3)
        self.assertEqual({job["source_wk_ord_no"] for job in problem["scenario"]["jobs"]}, {"WO_1", "WO_2", "WO_3"})
        self.assertEqual(
            {job["block_set_id"] for job in problem["scenario"]["jobs"]},
            {"PROJ_A::NP::BLK_1", "PROJ_A::NP::BLK_2"},
        )

    def test_missing_candidate_block_in_wo_source_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_path = base / "candidate.xlsx"
            wo_path = base / "wo.xlsx"
            self._write_candidate(candidate_path, [("PROJ_A", "BLK_1", 22), ("PROJ_MISSING", "BLK_9", 23)])
            self._write_wo(wo_path, [self._wo("PROJ_A", "BLK_1", "WO_1", 22)])

            with self.assertRaises(RuntimeError):
                load_phase2_candidate_workbook_problems(
                    wo_path=wo_path,
                    candidate_path=candidate_path,
                    workdays=("20260331",),
                    bay_ids=("22", "23", "24"),
                    gyel="NP",
                )

    def test_blank_actual_timestamp_is_allowed_for_tact_time_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_path = base / "candidate.xlsx"
            wo_path = base / "wo.xlsx"
            self._write_candidate(candidate_path, [("PROJ_A", "BLK_1", 22)])
            row = self._wo("PROJ_A", "BLK_1", "WO_1", 22)
            row["RT_CUT_ST_DTM"] = None
            row["RT_CUT_ED_DTM"] = None
            self._write_wo(wo_path, [row])

            problems = load_phase2_candidate_workbook_problems(
                wo_path=wo_path,
                candidate_path=candidate_path,
                workdays=("20260331",),
                bay_ids=("22", "23", "24"),
                gyel="NP",
            )

        self.assertEqual(problems[0]["wo_count"], 1)
        self.assertEqual(problems[0]["scenario"]["jobs"][0]["actual_start_datetime"], "")

    def test_blank_source_machine_is_allowed_for_generated_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            candidate_path = base / "candidate.xlsx"
            wo_path = base / "wo.xlsx"
            self._write_candidate(candidate_path, [("PROJ_A", "BLK_1", 22)])
            row = self._wo("PROJ_A", "BLK_1", "WO_1", 22)
            row["RT_EQP_NM"] = None
            self._write_wo(wo_path, [row])

            problems = load_phase2_candidate_workbook_problems(
                wo_path=wo_path,
                candidate_path=candidate_path,
                workdays=("20260331",),
                bay_ids=("22", "23", "24"),
                gyel="NP",
            )

        self.assertEqual(problems[0]["wo_count"], 1)
        self.assertEqual(problems[0]["scenario"]["jobs"][0]["source_machine_id"], "")

    @staticmethod
    def _write_candidate(path: Path, blocks: list[tuple[str, str, int]]) -> None:
        frame = pd.DataFrame(
            [
                {
                    "PROJ_NO": project,
                    "BLK_NO": block,
                    "GYEL": "NP",
                    "LTH": 10000,
                    "THK": 13,
                    "CUT_LTH": 100.0,
                    "STL_QTY": 5,
                    "BV_QTY": 2,
                    "CUT_BAY": bay,
                }
                for project, block, bay in blocks
            ]
        )
        with pd.ExcelWriter(path) as writer:
            frame.to_excel(writer, sheet_name="20260331_BLK", index=False)

    @staticmethod
    def _write_wo(path: Path, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_excel(path, index=False)

    @staticmethod
    def _wo(project: str, block: str, wo: str, bay: int) -> dict:
        return {
            "PROJ_NO": project,
            "BLK_NO": block,
            "WK_ORD_NO": wo,
            "GYEL_ACT_STDT": 20260330,
            "GYEL_ACT_EDDT": 20260331,
            "RT_CUT_ST_DTM": 202603310800,
            "RT_CUT_ED_DTM": 202603310830,
            "GYEL": "NP",
            "LTH": 10000,
            "THK": 13,
            "CUT_LTH": 100.0,
            "MARK_LTH": 10.0,
            "BVL_LTH": 0.0,
            "STL_QTY": 5,
            "BV_QTY": 2,
            "RT_EQP_NM": "PLS21",
            "CUT_BAY": bay,
            "TACT_TIME": 30.0,
            "ASS_ST_DT": 20260401,
            "PTLST_QTY": 10,
        }


if __name__ == "__main__":
    unittest.main()
