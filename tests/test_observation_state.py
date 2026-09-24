import unittest

import numpy as np

from observation import schema
from observation.actions import ActionHistory
from observation.cards import card_index
from observation.opponent_stats import OpponentStatsTracker
from observation.state import (
    PlayerSnapshot,
    calculate_to_call,
    encode_base_observation,
    normalize_chips,
    relative_seat_order,
)


class ObservationStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seats = ["player_0", "player_1", "player_2", "player_3"]
        self.player_states = {
            "player_0": PlayerSnapshot(160, 10, 40, True),
            "player_1": PlayerSnapshot(100, 20, 100, True, folded=True),
            "player_2": PlayerSnapshot(0, 30, 150, True, all_in=True),
            "player_3": PlayerSnapshot(0, 0, 0, False),
        }

    def build_observation(self) -> np.ndarray:
        return encode_base_observation(
            observer="player_2",
            seats=self.seats,
            player_states=self.player_states,
            button="player_0",
            street=schema.STREET_FLOP,
            pot=300,
            to_call=20,
            chip_scale=800,
            own_cards=["SA", "HK"],
            board_cards=["D2", "C3", "S4"],
        )

    def test_relative_seat_order_starts_with_observer(self) -> None:
        self.assertEqual(
            relative_seat_order("player_2", self.seats),
            ("player_2", "player_3", "player_0", "player_1"),
        )

    def test_cards_are_written_to_separate_blocks(self) -> None:
        observation = self.build_observation()

        self.assertEqual(float(observation[schema.OWN_CARDS].sum()), 2.0)
        self.assertEqual(float(observation[schema.BOARD_CARDS].sum()), 3.0)
        self.assertEqual(
            observation[schema.OWN_CARDS.start + card_index("SA")],
            1.0,
        )
        self.assertEqual(
            observation[schema.BOARD_CARDS.start + card_index("D2")],
            1.0,
        )

    def test_player_values_follow_relative_seat_order(self) -> None:
        observation = self.build_observation()

        np.testing.assert_allclose(
            observation[schema.PLAYER_STACKS],
            np.array([0, 0, 160, 100], dtype=np.float32) / 800,
        )
        np.testing.assert_allclose(
            observation[schema.STREET_CONTRIBUTIONS],
            np.array([30, 0, 10, 20], dtype=np.float32) / 800,
        )
        np.testing.assert_allclose(
            observation[schema.HAND_CONTRIBUTIONS],
            np.array([150, 0, 40, 100], dtype=np.float32) / 800,
        )

    def test_statuses_follow_relative_seat_order(self) -> None:
        observation = self.build_observation()

        np.testing.assert_array_equal(
            observation[schema.PLAYER_ACTIVE],
            [1, 0, 1, 1],
        )
        np.testing.assert_array_equal(
            observation[schema.PLAYER_FOLDED],
            [0, 0, 0, 1],
        )
        np.testing.assert_array_equal(
            observation[schema.PLAYER_ALL_IN],
            [1, 0, 0, 0],
        )

    def test_button_street_pot_and_to_call(self) -> None:
        observation = self.build_observation()

        np.testing.assert_array_equal(
            observation[schema.BUTTON_POSITION],
            [0, 0, 1, 0],
        )
        np.testing.assert_array_equal(
            observation[schema.STREET],
            [0, 1, 0, 0],
        )
        self.assertAlmostEqual(observation[schema.POT_INDEX], 300 / 800)
        self.assertAlmostEqual(observation[schema.TO_CALL_INDEX], 20 / 800)

    def test_optional_history_and_statistics_blocks_remain_zero(self) -> None:
        observation = self.build_observation()

        self.assertEqual(observation.shape, (schema.OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(
            float(observation[schema.LAST_ACTIONS.start : schema.OPPONENT_STATS.stop].sum()),
            0.0,
        )
        self.assertEqual(float(observation[schema.LAST_ACTIONS].sum()), 0.0)
        self.assertEqual(float(observation[schema.OPPONENT_STATS].sum()), 0.0)
        self.assertGreaterEqual(float(observation.min()), 0.0)
        self.assertLessEqual(float(observation.max()), 1.0)

    def test_action_history_can_be_added_to_base_observation(self) -> None:
        history = ActionHistory(self.seats)
        history.record("player_0", schema.ACTION_RAISE_POT, schema.STREET_FLOP)

        observation = encode_base_observation(
            observer="player_2",
            seats=self.seats,
            player_states=self.player_states,
            button="player_0",
            street=schema.STREET_FLOP,
            pot=300,
            to_call=20,
            chip_scale=800,
            own_cards=["SA", "HK"],
            board_cards=["D2", "C3", "S4"],
            action_history=history,
        )

        np.testing.assert_array_equal(
            observation[schema.last_action_slice(2)],
            [0, 0, 0, 1, 0],
        )
        self.assertEqual(
            observation[schema.street_raise_count_index(schema.STREET_FLOP)],
            0.5,
        )

    def test_opponent_statistics_can_be_added_to_base_observation(self) -> None:
        tracker = OpponentStatsTracker(self.seats)
        tracker.start_hand(self.seats)
        tracker.record_action(
            "player_3",
            schema.ACTION_RAISE_POT,
            schema.STREET_PREFLOP,
        )
        tracker.finish_hand()

        observation = encode_base_observation(
            observer="player_2",
            seats=self.seats,
            player_states=self.player_states,
            button="player_0",
            street=schema.STREET_FLOP,
            pot=300,
            to_call=20,
            chip_scale=800,
            own_cards=["SA", "HK"],
            board_cards=["D2", "C3", "S4"],
            opponent_stats=tracker,
        )

        first_opponent = schema.opponent_stats_slice(0)
        self.assertEqual(
            observation[first_opponent.start + schema.STAT_VPIP],
            1,
        )

    def test_calculate_to_call_handles_check_call_and_short_stack(self) -> None:
        self.assertEqual(calculate_to_call(20, 20, 100), 0)
        self.assertEqual(calculate_to_call(30, 10, 100), 20)
        self.assertEqual(calculate_to_call(100, 10, 25), 25)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_chips(-1, 800)

        with self.assertRaises(ValueError):
            normalize_chips(801, 800)

        with self.assertRaises(ValueError):
            PlayerSnapshot(100, 0, 0, True, folded=True, all_in=True)

        with self.assertRaises(ValueError):
            PlayerSnapshot(10, 10, 20, True, all_in=True)

        with self.assertRaises(ValueError):
            PlayerSnapshot(100, 20, 10, True)

        with self.assertRaises(ValueError):
            PlayerSnapshot(1, 0, 0, False)

        with self.assertRaises(ValueError):
            relative_seat_order("player_4", self.seats)


if __name__ == "__main__":
    unittest.main()
