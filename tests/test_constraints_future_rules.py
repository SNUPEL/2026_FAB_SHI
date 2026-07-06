"""Future constraint registry tests for width, Bay, and priority rules."""

from types import SimpleNamespace
import unittest

from Environment.constraints.base import ConstraintContext
from Environment.constraints.registry import RULES, RULE_CATEGORIES


class FutureConstraintRulesTest(unittest.TestCase):
    """Future constraints should fail loudly and be registry-toggleable."""

    def test_registry_contains_width_bay_and_priority_rules(self) -> None:
        expected = {
            "plate_width_range": "machine",
            "allowed_bay_ids": "layout",
            "machine_priority_tier_policy": "priority",
            "bay_priority_tier_policy": "priority",
        }

        for rule_name, category in expected.items():
            self.assertIn(rule_name, RULES)
            self.assertEqual(RULE_CATEGORIES[rule_name], category)

    def test_plate_width_range_blocks_wide_plate_on_narrow_machine(self) -> None:
        context = self._context(
            plate_width=4700,
            machine_max_plate_width=4500,
        )

        result = RULES["plate_width_range"](context)

        self.assertFalse(result.passed)
        self.assertIn("exceeds max", result.reason)

    def test_plate_width_range_does_not_pass_when_width_is_missing(self) -> None:
        context = self._context(plate_width=None, machine_max_plate_width=4500)

        result = RULES["plate_width_range"](context)

        self.assertFalse(result.passed)
        self.assertEqual(result.metadata["missing_field"], "job.plate_width")

    def test_allowed_bay_ids_blocks_machine_outside_phase1_mask(self) -> None:
        context = self._context(
            job=SimpleNamespace(allowed_bay_ids=("22", "23")),
            machine_bay_id="24",
        )

        result = RULES["allowed_bay_ids"](context)

        self.assertFalse(result.passed)
        self.assertIn("Bay 24", result.reason)

    def test_priority_tier_policy_blocks_low_tier_when_primary_exists(self) -> None:
        machine_result = RULES["machine_priority_tier_policy"](
            self._context(machine_priority_tier=3, has_primary_machine_tier_candidate=True)
        )
        bay_result = RULES["bay_priority_tier_policy"](
            self._context(bay_priority_tier=4, has_primary_bay_tier_candidate=True)
        )

        self.assertFalse(machine_result.passed)
        self.assertFalse(bay_result.passed)

    @staticmethod
    def _context(**overrides) -> ConstraintContext:
        values = {
            "job": SimpleNamespace(allowed_bay_ids=("22", "23", "24")),
            "machine": SimpleNamespace(machine_id="PLS41"),
            "state": SimpleNamespace(),
            "jobs": {},
            "machines": {},
            "bays": {},
            "layout": {},
            "config": {},
        }
        values.update(overrides)
        return ConstraintContext(**values)


if __name__ == "__main__":
    unittest.main()
