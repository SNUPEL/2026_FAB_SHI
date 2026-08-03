"""Phase 1 sampling temperature annealing 계약 테스트.

핵심은 두 가지다.
1. 스케줄이 지수 감쇠이고 `t_min`에서 clamp된다.
2. **미설정 시 기존과 동일**하게 상수로 동작한다(하위호환).

Phase 2(`Phase2/merged.annealed_temperature`)와 같은 식이어야 하므로 두 구현이
같은 값을 내는지도 함께 고정한다.
"""

import unittest

from Phase1.pair_self_labeling import annealed_temperature
from Phase2.merged import annealed_temperature as phase2_annealed_temperature


class AnnealedTemperatureScheduleTestCase(unittest.TestCase):
    def test_first_episode_is_the_initial_temperature(self):
        self.assertAlmostEqual(annealed_temperature(0, 1.0, 0.3, 100), 1.0)

    def test_decays_to_the_minimum_at_the_anneal_horizon(self):
        self.assertAlmostEqual(annealed_temperature(100, 1.0, 0.3, 100), 0.3)

    def test_clamps_after_the_anneal_horizon(self):
        for episode in (101, 500, 20_000):
            self.assertAlmostEqual(
                annealed_temperature(episode, 1.0, 0.3, 100),
                0.3,
                msg=f"episode={episode} should stay clamped at t_min",
            )

    def test_is_monotonically_non_increasing(self):
        values = [annealed_temperature(episode, 1.5, 0.5, 50) for episode in range(0, 60)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(later, earlier + 1e-12)
        self.assertAlmostEqual(values[0], 1.5)
        self.assertAlmostEqual(values[50], 0.5)

    def test_follows_the_documented_exponential_form(self):
        # T(ep) = t0 * (t_min/t0)^(ep/anneal)
        t0, t_min, anneal = 1.0, 0.3, 20_000
        for episode in (1, 5_000, 10_000, 19_999):
            expected = t0 * (t_min / t0) ** (episode / anneal)
            self.assertAlmostEqual(annealed_temperature(episode, t0, t_min, anneal), expected, places=9)


class AnnealedTemperatureBackwardCompatibilityTestCase(unittest.TestCase):
    """미설정 시 상수. 해석 규칙은 train 함수에서 t_min=t0, anneal=episodes로 넘어온다."""

    def test_t_min_equal_to_t0_keeps_it_constant(self):
        for episode in (0, 1, 10, 10_000):
            self.assertAlmostEqual(annealed_temperature(episode, 1.0, 1.0, 20_000), 1.0)

    def test_t_min_above_t0_keeps_it_constant(self):
        self.assertAlmostEqual(annealed_temperature(5_000, 1.0, 2.0, 20_000), 1.0)

    def test_single_episode_horizon_keeps_it_constant(self):
        self.assertAlmostEqual(annealed_temperature(1, 1.0, 0.3, 1), 1.0)

    def test_rejects_non_positive_temperature(self):
        with self.assertRaises(ValueError):
            annealed_temperature(1, 0.0, 0.3, 100)
        with self.assertRaises(ValueError):
            annealed_temperature(1, 1.0, 0.0, 100)


class PhaseParityTestCase(unittest.TestCase):
    def test_phase1_and_phase2_schedules_agree(self):
        for episode in (0, 1, 37, 1_000, 19_999, 20_000, 25_000):
            self.assertAlmostEqual(
                annealed_temperature(episode, 1.0, 0.3, 20_000),
                phase2_annealed_temperature(episode, 1.0, 0.3, 20_000),
                places=12,
                msg=f"Phase 1/2 temperature schedules diverge at episode={episode}",
            )


if __name__ == "__main__":
    unittest.main()