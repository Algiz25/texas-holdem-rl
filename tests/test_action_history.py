import unittest

import numpy as np

from observation import schema
from observation.actions import ActionHistory, normalize_count


class ActionHistoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seats = ["player_0", "player_1", "player_2", "player_3"]
        self.history = ActionHistory(self.seats)

    def test_empty_history_has_no_actions_and_no_aggressors(self) -> None:
        encoded = self.history.encode("player_0")

        np.testing.assert_array_equal(encoded[:20], np.zeros(20))
        for street in range(schema.NUM_STREETS):
            start = 20 + street * schema.STREET_SUMMARY_SIZE
            np.testing.assert_array_equal(
                encoded[start : start + 5],
                [1, 0, 0, 0, 0],
            )

    def test_last_action_is_one_hot_and_overwritten(self) -> None:
        self.history.record("player_1", schema.ACTION_CHECK_CALL, 0)
        self.history.record("player_1", schema.ACTION_FOLD, 0)

        encoded = self.history.encode("player_0")
        np.testing.assert_array_equal(encoded[5:10], [1, 0, 0, 0, 0])

    def test_actions_are_relative_to_observer(self) -> None:
        self.history.record("player_0", schema.ACTION_RAISE_POT, 0)

        encoded = self.history.encode("player_2")
        np.testing.assert_array_equal(encoded[10:15], [0, 0, 0, 1, 0])

    def test_check_is_not_counted_as_call(self) -> None:
        self.history.record("player_0", schema.ACTION_CHECK_CALL, 1, to_call=0)
        self.assertEqual(self.history.call_counts[1], 0)

    def test_call_and_calling_all_in_are_counted(self) -> None:
        self.history.record("player_0", schema.ACTION_CHECK_CALL, 1, to_call=10)
        self.history.record("player_1", schema.ACTION_ALL_IN, 1, to_call=10)
        self.assertEqual(self.history.call_counts[1], 2)

    def test_raise_and_aggressive_all_in_update_last_aggressor(self) -> None:
        self.history.record("player_0", schema.ACTION_RAISE_HALF_POT, 2)
        self.history.record(
            "player_3",
            schema.ACTION_ALL_IN,
            2,
            to_call=20,
            all_in_increases_bet=True,
        )

        encoded = self.history.encode("player_2")
        summary_start = 20 + 2 * schema.STREET_SUMMARY_SIZE
        np.testing.assert_array_equal(
            encoded[summary_start : summary_start + 5],
            [0, 0, 1, 0, 0],
        )
        self.assertEqual(self.history.raise_counts[2], 2)

    def test_counts_use_bounded_normalization(self) -> None:
        self.assertEqual(normalize_count(0), 0)
        self.assertEqual(normalize_count(1), 0.5)
        self.assertAlmostEqual(normalize_count(2), 2 / 3)
        self.assertLess(normalize_count(100), 1)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ActionHistory(["player_0"])
        with self.assertRaises(ValueError):
            self.history.record("unknown", 0, 0)
        with self.assertRaises(ValueError):
            self.history.record("player_0", 5, 0)
        with self.assertRaises(ValueError):
            self.history.record("player_0", 0, 4)
        with self.assertRaises(ValueError):
            normalize_count(-1)


if __name__ == "__main__":
    unittest.main()
