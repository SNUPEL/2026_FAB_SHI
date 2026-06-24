"""Phase 1 block-to-Bay workload balancing tests.

These tests define the first-stage planning problem:
- one action assigns one block set to one cutting Bay;
- every W/O in the same block set follows that Bay assignment;
- the primary balancing load is the sum of steel quantity, not block count.
"""

from types import SimpleNamespace
import unittest

from Utils.phase1_bay_balancer import (
    LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
    MULTI_OBJECTIVE_PHASE1_HEURISTIC,
    PRIORITY_GREEDY_PHASE1_HEURISTIC,
    PRIORITY_SWEEP_PHASE1_HEURISTIC,
    Phase1Block,
    apply_phase1_plan_to_scenario,
    build_phase1_bay_plan,
    _improve_multi_objective_assignment,
    _multi_objective_assignment_score,
    _priority_sweep_load_score,
)
from Utils.cutting_scenario_builder import build_scenario_from_cutting_records


class Phase1BayBalancerTest(unittest.TestCase):
    """Block-level Bay assignment must stay independent from machine scheduling."""

    def test_assigns_each_block_to_one_bay_using_steel_quantity_load(self) -> None:
        """The heaviest block should be placed first and steel quantity should drive loads."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 5),
            "WO_A2": self._job("WO_A2", "P1::B1", 4),
            "WO_B1": self._job("WO_B1", "P1::B2", 6),
            "WO_C1": self._job("WO_C1", "P1::B3", 3),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23"],
            algorithm="lpt_steel_quantity",
        )

        assignments = {row["block_set_id"]: row["assigned_bay"] for row in result["assignments"]}

        self.assertEqual(result["summary"]["load_metric"], "steel_quantity_sum")
        self.assertEqual(result["summary"]["block_count"], 3)
        self.assertEqual(result["summary"]["job_count"], 4)
        self.assertEqual(assignments["P1::B1"], "22")
        self.assertEqual(assignments["P1::B2"], "23")
        self.assertEqual(assignments["P1::B3"], "23")
        self.assertEqual(result["bay_loads"]["22"]["steel_quantity_sum"], 9)
        self.assertEqual(result["bay_loads"]["23"]["steel_quantity_sum"], 9)
        self.assertEqual(result["bay_loads"]["22"]["wo_count"], 2)
        self.assertEqual(result["bay_loads"]["23"]["wo_count"], 2)

    def test_respects_allowed_bay_ids_for_a_block(self) -> None:
        """If a block is restricted to one Bay, Phase 1 must not assign another Bay."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 5, allowed_bay_ids=("23",)),
            "WO_A2": self._job("WO_A2", "P1::B1", 4, allowed_bay_ids=("23",)),
            "WO_B1": self._job("WO_B1", "P1::B2", 7),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23"],
            algorithm="lpt_steel_quantity",
        )

        assignments = {row["block_set_id"]: row["assigned_bay"] for row in result["assignments"]}

        self.assertEqual(assignments["P1::B1"], "23")

    def test_steel_lpt_greedy_insertion_is_canonical_heuristic_name(self) -> None:
        """The visible heuristic name should describe sorting and insertion behavior."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 5),
            "WO_B1": self._job("WO_B1", "P1::B2", 3),
        }

        canonical = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23"],
            algorithm="steel_lpt_greedy_insertion",
        )
        legacy_alias = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23"],
            algorithm="lpt_steel_quantity",
        )

        self.assertEqual(canonical["algorithm"], "steel_lpt_greedy_insertion")
        self.assertEqual(legacy_alias["algorithm"], "steel_lpt_greedy_insertion")
        self.assertEqual(canonical["summary"]["algorithm"], "steel_lpt_greedy_insertion")

    def test_multi_objective_balanced_reports_three_phase1_objectives(self) -> None:
        """The new Phase 1 heuristic balances steel, cut length, and bevel quantity."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 10, cut_length=1500, bevel_quantity=0),
            "WO_B1": self._job("WO_B1", "P1::B2", 10, cut_length=10, bevel_quantity=10),
            "WO_C1": self._job("WO_C1", "P1::B3", 10, cut_length=10, bevel_quantity=0),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm=MULTI_OBJECTIVE_PHASE1_HEURISTIC,
        )
        assignments = {row["block_set_id"]: row["assigned_bay"] for row in result["assignments"]}

        self.assertEqual(result["algorithm"], "multi_objective_balanced")
        self.assertEqual(result["summary"]["load_metric"], "multi_objective_lexicographic")
        self.assertEqual(result["summary"]["cut_length_total"], 1520.0)
        self.assertEqual(result["summary"]["bevel_quantity_total"], 10)
        self.assertEqual(assignments["P1::B1"], "22")
        self.assertEqual(result["assignments"][0]["long_cut_over_1000"], 1)

    def test_multi_objective_requires_bevel_quantity_without_fallback(self) -> None:
        """BV_QTY is a required objective input for the multi-objective heuristic."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 10, cut_length=100, bevel_quantity=None),
        }

        with self.assertRaisesRegex(RuntimeError, "missing_bevel_quantity"):
            build_phase1_bay_plan(
                jobs=jobs,
                bay_ids=["22", "23", "24"],
                algorithm=MULTI_OBJECTIVE_PHASE1_HEURISTIC,
            )

    def test_multi_objective_local_search_can_improve_by_swapping_two_blocks(self) -> None:
        """Some balanced plans need a two-block swap, not a one-block move."""

        blocks = (
            self._block("P1::A", steel=5, cut_length=100),
            self._block("P1::B", steel=5, cut_length=900),
            self._block("P1::C", steel=3, cut_length=100),
            self._block("P1::D", steel=3, cut_length=900),
        )
        bay_ids = ("22", "23")
        initial = {
            "P1::A": "22",
            "P1::B": "23",
            "P1::C": "22",
            "P1::D": "23",
        }

        improved = _improve_multi_objective_assignment(blocks, initial, bay_ids)

        self.assertLess(
            _multi_objective_assignment_score(blocks, improved, bay_ids),
            _multi_objective_assignment_score(blocks, initial, bay_ids),
        )

    def test_priority_sweep_score_prioritizes_long_cut_bay24_before_balance(self) -> None:
        """The priority-sweep mode must treat long-cut Bay 24 assignment as the first objective."""

        no_long_cut_bay24_but_uneven = {
            "22": self._load(steel=1, cut_length=10.0, bevel_quantity=1, long_cut_bay24_count=0),
            "23": self._load(steel=1, cut_length=10.0, bevel_quantity=1, long_cut_bay24_count=0),
            "24": self._load(steel=98, cut_length=980.0, bevel_quantity=98, long_cut_bay24_count=0),
        }
        one_long_cut_bay24_but_balanced = {
            "22": self._load(steel=33, cut_length=330.0, bevel_quantity=33, long_cut_bay24_count=0),
            "23": self._load(steel=33, cut_length=330.0, bevel_quantity=33, long_cut_bay24_count=0),
            "24": self._load(steel=34, cut_length=340.0, bevel_quantity=34, long_cut_bay24_count=1),
        }

        self.assertLess(
            _priority_sweep_load_score(no_long_cut_bay24_but_uneven),
            _priority_sweep_load_score(one_long_cut_bay24_but_balanced),
        )

    def test_priority_sweep_balanced_reports_long_cut_first_priority(self) -> None:
        """The new heuristic name should expose the 4th objective as first priority."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", 10, cut_length=1500, bevel_quantity=0),
            "WO_B1": self._job("WO_B1", "P1::B2", 10, cut_length=10, bevel_quantity=10),
            "WO_C1": self._job("WO_C1", "P1::B3", 10, cut_length=10, bevel_quantity=0),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm=PRIORITY_SWEEP_PHASE1_HEURISTIC,
        )

        self.assertEqual(result["algorithm"], "priority_sweep_balanced")
        self.assertEqual(result["summary"]["load_metric"], "priority_sweep_lexicographic")
        self.assertEqual(
            result["summary"]["objective_priority"],
            [
                "long_cut_over_1000_prefer_bay22_23",
                "steel_quantity_sum",
                "cut_length_sum",
                "bevel_quantity_sum",
            ],
        )

    def test_long_cut_preferred_balanced_exposes_field_preference_name(self) -> None:
        """The official field heuristic should keep long-cut Bay 24 preference visible."""

        jobs = {
            "WO_LONG": self._job("WO_LONG", "P1::LONG", 10, cut_length=1500, bevel_quantity=3),
            "WO_BEVEL": self._job("WO_BEVEL", "P1::BEVEL", 9, cut_length=100, bevel_quantity=20),
            "WO_A": self._job("WO_A", "P1::A", 8, cut_length=100, bevel_quantity=1),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm=LONG_CUT_PREFERRED_PHASE1_HEURISTIC,
        )

        priority_sweep = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm=PRIORITY_SWEEP_PHASE1_HEURISTIC,
        )

        self.assertEqual(result["algorithm"], "long_cut_preferred_balanced")
        self.assertEqual(result["summary"]["load_metric"], "long_cut_preferred_lexicographic")
        self.assertEqual(
            result["summary"]["objective_priority"],
            [
                "long_cut_over_1000_prefer_bay22_23",
                "steel_quantity_sum",
                "cut_length_sum",
                "bevel_quantity_sum",
            ],
        )
        self.assertEqual(result["summary"]["long_cut_bay24_count"], 0)
        self.assertEqual(result["bay_loads"], priority_sweep["bay_loads"])

    def test_priority_greedy_insertion_assigns_one_bay_per_block_step_by_step(self) -> None:
        """The presentation heuristic should use one deterministic greedy insertion pass."""

        jobs = {
            "WO_LONG": self._job("WO_LONG", "P1::LONG", 10, cut_length=1500, bevel_quantity=3),
            "WO_A": self._job("WO_A", "P1::A", 9, cut_length=100, bevel_quantity=1),
            "WO_B": self._job("WO_B", "P1::B", 8, cut_length=100, bevel_quantity=1),
        }

        result = build_phase1_bay_plan(
            jobs=jobs,
            bay_ids=["22", "23", "24"],
            algorithm=PRIORITY_GREEDY_PHASE1_HEURISTIC,
        )

        assignments = {row["block_set_id"]: row["assigned_bay"] for row in result["assignments"]}

        self.assertEqual(result["algorithm"], "priority_greedy_insertion")
        self.assertEqual(result["summary"]["load_metric"], "priority_greedy_lexicographic")
        self.assertEqual(
            result["summary"]["block_selection_priority"],
            [
                "long_cut_over_1000_first",
                "steel_quantity_sum_desc",
                "cut_length_sum_desc",
                "bevel_quantity_sum_desc",
            ],
        )
        self.assertNotEqual(assignments["P1::LONG"], "24")
        self.assertEqual(result["summary"]["long_cut_bay24_count"], 0)

    def test_applies_phase1_plan_to_phase2_scenario_as_cut_bay(self) -> None:
        """Phase 2 scenario injection should fix every W/O in a block to the assigned Bay."""

        scenario = {
            "metadata": {"scenario_name": "unit"},
            "jobs": [
                {"job_id": "WO_A1", "block_set_id": "P1::B1", "cut_bay": None},
                {"job_id": "WO_A2", "block_set_id": "P1::B1", "cut_bay": None},
                {"job_id": "WO_B1", "block_set_id": "P1::B2", "cut_bay": None},
            ],
        }
        plan = {
            "algorithm": "steel_lpt_greedy_insertion",
            "assignments": [
                {"block_set_id": "P1::B1", "assigned_bay": "22"},
                {"block_set_id": "P1::B2", "assigned_bay": "23"},
            ],
        }

        result = apply_phase1_plan_to_scenario(
            scenario=scenario,
            plan=plan,
            assignment_mode="cut_bay",
        )
        phase2_jobs = {job["job_id"]: job for job in result["scenario"]["jobs"]}

        self.assertEqual(phase2_jobs["WO_A1"]["cut_bay"], "22")
        self.assertEqual(phase2_jobs["WO_A2"]["cut_bay"], "22")
        self.assertEqual(phase2_jobs["WO_B1"]["cut_bay"], "23")
        self.assertTrue(result["scenario"]["metadata"]["phase1_applied"])
        self.assertEqual(result["summary"]["assigned_job_count"], 3)
        self.assertEqual(result["summary"]["assigned_block_count"], 2)

    def test_apply_phase1_plan_fails_when_a_job_block_is_unassigned(self) -> None:
        """Phase 2 injection must not silently leave an unassigned block free."""

        scenario = {
            "jobs": [
                {"job_id": "WO_A1", "block_set_id": "P1::B1"},
                {"job_id": "WO_MISSING", "block_set_id": "P1::B9"},
            ],
        }
        plan = {
            "algorithm": "steel_lpt_greedy_insertion",
            "assignments": [
                {"block_set_id": "P1::B1", "assigned_bay": "22"},
            ],
        }

        with self.assertRaisesRegex(RuntimeError, "missing_phase1_assignment"):
            apply_phase1_plan_to_scenario(
                scenario=scenario,
                plan=plan,
                assignment_mode="cut_bay",
            )

    def test_missing_steel_quantity_fails_without_wo_count_fallback(self) -> None:
        """Phase 1 must not silently replace missing steel quantity with W/O count."""

        jobs = {
            "WO_A1": self._job("WO_A1", "P1::B1", None),
        }

        with self.assertRaisesRegex(RuntimeError, "missing_steel_quantity"):
            build_phase1_bay_plan(
                jobs=jobs,
                bay_ids=["22", "23"],
                algorithm="lpt_steel_quantity",
            )

    def test_scenario_builder_exposes_steel_quantity_for_environment_jobs(self) -> None:
        """Scenario jobs must use the dataclass field name `steel_quantity`."""

        scenario = build_scenario_from_cutting_records(
            records=[
                {
                    "work_order_no": "WO_A1",
                    "project_no": "P1",
                    "block_no": "B1",
                    "block_set_id": "P1::B1",
                    "series": "NP",
                    "length": 1000,
                    "thickness": 10,
                    "cut_length": 100,
                    "mark_length": 0,
                    "bevel_length": 0,
                    "steel_qty": 4,
                    "part_qty": 2,
                    "tact_time": 3,
                    "source_cut_bay": "22",
                    "source_machine_id": "PLS21",
                }
            ],
            process_time_source="tact_time",
        )

        self.assertEqual(scenario["jobs"][0]["steel_quantity"], 4)

    @staticmethod
    def _job(
        job_id: str,
        block_set_id: str,
        steel_quantity: int | None,
        allowed_bay_ids: tuple[str, ...] = (),
        cut_length: float | None = 100.0,
        bevel_quantity: int | None = 0,
    ) -> SimpleNamespace:
        """Create the smallest Job-like object needed by Phase 1 logic."""

        return SimpleNamespace(
            job_id=job_id,
            block_set_id=block_set_id,
            steel_quantity=steel_quantity,
            cut_length=cut_length,
            bevel_quantity=bevel_quantity,
            cut_bay=None,
            source_cut_bay=None,
            allowed_bay_ids=allowed_bay_ids,
            extra={
                "source_project_no": block_set_id.split("::")[0],
                "source_block_no": block_set_id.split("::")[1],
                "source_wk_ord_no": job_id,
            },
        )

    @staticmethod
    def _block(block_set_id: str, steel: int, cut_length: float) -> Phase1Block:
        """Create a small Phase 1 block for local-search tests."""

        return Phase1Block(
            block_set_id=block_set_id,
            project_no=block_set_id.split("::")[0],
            block_no=block_set_id.split("::")[1],
            job_ids=(f"{block_set_id}_WO",),
            wo_count=1,
            steel_quantity_sum=steel,
            cut_length_sum=cut_length,
            bevel_quantity_sum=0,
            long_cut_over_1000=0,
            allowed_bay_ids=("22", "23"),
        )

    @staticmethod
    def _load(
        steel: int,
        cut_length: float,
        bevel_quantity: int,
        long_cut_bay24_count: int,
    ) -> dict:
        """Create the Bay load shape used by Phase 1 score functions."""

        return {
            "steel_quantity_sum": steel,
            "cut_length_sum": cut_length,
            "bevel_quantity_sum": bevel_quantity,
            "long_cut_bay24_count": long_cut_bay24_count,
            "wo_count": steel,
            "block_count": steel,
        }


if __name__ == "__main__":
    unittest.main()
