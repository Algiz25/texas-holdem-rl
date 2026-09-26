"""Kontrola najważniejszych założeń pierwszego treningu DQN."""

import unittest

import config
from training.train_dqn import phase_one_epsilon


class DQNPhaseOneConfigurationTests(unittest.TestCase):
    def test_phase_contains_comparable_number_of_learner_decisions(self) -> None:
        self.assertEqual(
            config.DQN_MAX_EPOCHS * config.DQN_STEPS_PER_EPOCH,
            250_000,
        )
        self.assertEqual(
            config.DQN_STEPS_PER_EPOCH % config.DQN_COLLECTION_STEPS,
            0,
        )

    def test_opponent_mixture_matches_phase_one_plan(self) -> None:
        self.assertEqual(
            config.PHASE1_OPPONENT_WEIGHTS,
            {"random": 0.50, "passive": 0.40, "mixed": 0.10},
        )
        self.assertAlmostEqual(sum(config.PHASE1_OPPONENT_WEIGHTS.values()), 1.0)

    def test_epsilon_reaches_point_one_after_180k_learner_decisions(self) -> None:
        self.assertAlmostEqual(phase_one_epsilon(0), 1.0)
        self.assertAlmostEqual(phase_one_epsilon(90_000), 0.55)
        self.assertAlmostEqual(phase_one_epsilon(180_000), 0.1)
        self.assertAlmostEqual(phase_one_epsilon(250_000), 0.1)

    def test_overnight_run_can_use_a_longer_epsilon_schedule(self) -> None:
        self.assertAlmostEqual(phase_one_epsilon(0, 700_000), 1.0)
        self.assertAlmostEqual(phase_one_epsilon(350_000, 700_000), 0.55)
        self.assertAlmostEqual(phase_one_epsilon(700_000, 700_000), 0.1)

    def test_macbook_uses_benchmarked_parallelism(self) -> None:
        self.assertEqual(config.DQN_NUM_TRAIN_ENVS, 8)
        self.assertEqual(config.DQN_NUM_TEST_ENVS, 1)
        self.assertEqual(config.TORCH_NUM_THREADS, 1)
        self.assertEqual(config.TORCH_NUM_INTEROP_THREADS, 1)
        self.assertEqual(config.DQN_BUFFER_WARMUP, 25_000)
        self.assertEqual(config.EVAL_NUM_WORKERS, 4)
        self.assertEqual(config.EVAL_SMOKE_TOURNAMENTS, 1)
        self.assertEqual(config.DQN_FULL_STATE_INTERVAL_DECISIONS, 100_000)


if __name__ == "__main__":
    unittest.main()
