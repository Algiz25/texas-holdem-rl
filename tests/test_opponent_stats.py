import unittest

import numpy as np

from observation import schema
from observation.opponent_stats import OBSERVED_HANDS_SCALE, OpponentStatsTracker


class OpponentStatsTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seats = ["player_0", "player_1", "player_2", "player_3"]
        self.tracker = OpponentStatsTracker(self.seats)

    def test_empty_statistics_are_zero(self) -> None:
        np.testing.assert_array_equal(
            self.tracker.encode("player_0"),
            np.zeros(18, dtype=np.float32),
        )

    def test_vpip_and_pfr_are_counted_once_per_hand(self) -> None:
        self.tracker.start_hand(self.seats)
        self.tracker.record_action(
            "player_1",
            schema.ACTION_RAISE_HALF_POT,
            schema.STREET_PREFLOP,
        )
        self.tracker.record_action(
            "player_1",
            schema.ACTION_RAISE_POT,
            schema.STREET_PREFLOP,
        )
        self.tracker.record_action(
            "player_2",
            schema.ACTION_CHECK_CALL,
            schema.STREET_PREFLOP,
            to_call=10,
        )
        self.tracker.finish_hand()

        encoded = self.tracker.encode("player_0")
        self.assertEqual(encoded[schema.STAT_VPIP], 1)
        self.assertEqual(encoded[schema.STAT_PFR], 1)
        self.assertEqual(encoded[6 + schema.STAT_VPIP], 1)
        self.assertEqual(encoded[6 + schema.STAT_PFR], 0)

    def test_blind_or_check_does_not_count_as_vpip(self) -> None:
        self.tracker.start_hand(self.seats)
        self.tracker.record_action(
            "player_1",
            schema.ACTION_CHECK_CALL,
            schema.STREET_PREFLOP,
            to_call=0,
        )
        self.tracker.finish_hand()
        self.assertEqual(self.tracker.encode("player_0")[schema.STAT_VPIP], 0)

    def test_aggression_is_postflop_aggressive_action_frequency(self) -> None:
        self.tracker.start_hand(self.seats)
        self.tracker.record_action(
            "player_1",
            schema.ACTION_RAISE_POT,
            schema.STREET_FLOP,
        )
        self.tracker.record_action(
            "player_1",
            schema.ACTION_CHECK_CALL,
            schema.STREET_TURN,
            to_call=20,
        )
        self.tracker.finish_hand()
        self.assertEqual(
            self.tracker.encode("player_0")[schema.STAT_AGGRESSION],
            0.5,
        )

    def test_fold_to_raise_uses_only_real_opportunities(self) -> None:
        self.tracker.start_hand(self.seats)
        self.tracker.record_action(
            "player_1",
            schema.ACTION_FOLD,
            schema.STREET_FLOP,
            facing_raise=True,
        )
        self.tracker.record_action(
            "player_2",
            schema.ACTION_FOLD,
            schema.STREET_FLOP,
            facing_raise=False,
        )
        self.tracker.finish_hand()

        encoded = self.tracker.encode("player_0")
        self.assertEqual(encoded[schema.STAT_FOLD_TO_RAISE], 1)
        self.assertEqual(encoded[6 + schema.STAT_FOLD_TO_RAISE], 0)

    def test_showdown_win_rate(self) -> None:
        for winners in (("player_1",), ("player_2",)):
            self.tracker.start_hand(self.seats)
            self.tracker.finish_hand(
                showdown_players=("player_1", "player_2"),
                winners=winners,
            )

        encoded = self.tracker.encode("player_0")
        self.assertEqual(encoded[schema.STAT_SHOWDOWN_WIN_RATE], 0.5)
        self.assertEqual(encoded[6 + schema.STAT_SHOWDOWN_WIN_RATE], 0.5)

    def test_observed_hands_are_capped_after_normalization(self) -> None:
        for _ in range(OBSERVED_HANDS_SCALE + 1):
            self.tracker.start_hand(self.seats)
            self.tracker.finish_hand()
        self.assertEqual(
            self.tracker.encode("player_0")[schema.STAT_OBSERVED_HANDS],
            1,
        )

    def test_opponents_are_relative_to_observer(self) -> None:
        self.tracker.start_hand(self.seats)
        self.tracker.record_action(
            "player_0",
            schema.ACTION_RAISE_POT,
            schema.STREET_PREFLOP,
        )
        self.tracker.finish_hand()

        encoded = self.tracker.encode("player_2")
        self.assertEqual(encoded[6 + schema.STAT_VPIP], 1)

    def test_invalid_lifecycle_and_results_are_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.tracker.record_action("player_0", 0, 0)
        with self.assertRaises(RuntimeError):
            self.tracker.finish_hand()

        self.tracker.start_hand(self.seats)
        with self.assertRaises(RuntimeError):
            self.tracker.start_hand(self.seats)
        with self.assertRaises(ValueError):
            self.tracker.finish_hand(
                showdown_players=("player_0",),
                winners=("player_1",),
            )


if __name__ == "__main__":
    unittest.main()
