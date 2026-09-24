"""Testy obliczeń ewaluatora bez uruchamiania długiego treningu."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import config
from evaluation.evaluator import (
    BasePokerEvaluator,
    EvaluationAccumulator,
)
from observation import schema


class EvaluationAccumulatorTests(unittest.TestCase):
    def test_check_and_call_are_reported_separately(self) -> None:
        accumulator = EvaluationAccumulator()
        observation = np.zeros(schema.OBSERVATION_SIZE, dtype=np.float32)
        observation[schema.STREET.start + schema.STREET_PREFLOP] = 1.0

        accumulator.record_action(
            observation,
            schema.ACTION_CHECK_CALL,
            legal=True,
        )
        observation[schema.TO_CALL_INDEX] = 0.25
        accumulator.record_action(
            observation,
            schema.ACTION_CHECK_CALL,
            legal=True,
        )

        self.assertEqual(accumulator.actions["check"], 1)
        self.assertEqual(accumulator.actions["call"], 1)

    def test_bb_per_100_and_poker_statistics_use_completed_hands(self) -> None:
        accumulator = EvaluationAccumulator()
        player_stats = SimpleNamespace(
            observed_hands=10,
            vpip_hands=4,
            pfr_hands=2,
            showdowns=3,
            showdown_wins=2,
        )
        env = SimpleNamespace(
            completed_hands=10,
            hand_wins={"player_0": 6},
            tournament_chips={"player_0": 220, "player_1": 180},
            starting_chips=200,
            finishing_positions={"player_0": 1.0},
            opponent_stats=SimpleNamespace(players={"player_0": player_stats}),
        )
        accumulator.record_tournament(env, "player_0", completed=True)
        result = accumulator.finish(
            stage="validation",
            step=100_000,
            checkpoint=Path("checkpoint.pth"),
            opponent_suite="phase1_mix",
            seed_base=123,
            tournaments=1,
        )

        expected_bb_per_100 = 100 * 20 / (10 * config.BIG_BLIND)
        self.assertEqual(result.bb_per_100, expected_bb_per_100)
        self.assertEqual(result.vpip, 0.4)
        self.assertEqual(result.pfr, 0.2)
        self.assertAlmostEqual(result.showdown_win_rate, 2 / 3)


class EvaluationOpponentTests(unittest.TestCase):
    def test_phase_one_lineup_is_repeatable_for_the_same_seed(self) -> None:
        first = BasePokerEvaluator._opponents_for_tournament(
            "phase1_mix",
            "player_2",
            np.random.default_rng(42),
        )
        second = BasePokerEvaluator._opponents_for_tournament(
            "phase1_mix",
            "player_2",
            np.random.default_rng(42),
        )

        self.assertEqual(first, second)
        self.assertNotIn("player_2", first)
        self.assertEqual(len(first), 3)

    def test_passive_bot_prefers_check_call(self) -> None:
        action = BasePokerEvaluator._opponent_action(
            "passive",
            np.array([0, 1, 3]),
            np.random.default_rng(1),
        )
        self.assertEqual(action, schema.ACTION_CHECK_CALL)

    def test_report_contains_csv_and_detailed_json(self) -> None:
        accumulator = EvaluationAccumulator()
        result = accumulator.finish(
            stage="validation",
            step=100_000,
            checkpoint=Path("checkpoint.pth"),
            opponent_suite="random",
            seed_base=123,
            tournaments=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            evaluator = object.__new__(BasePokerEvaluator)
            evaluator.report_dir = Path(directory)
            evaluator._save_report(
                {"random": result},
                stage="validation",
                step=100_000,
            )

            self.assertTrue((Path(directory) / "evaluations.csv").exists())
            self.assertTrue(
                (Path(directory) / "validation_step_000100000.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
