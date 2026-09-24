import unittest

import numpy as np

from observation import schema
from observation.derived import (
    calculate_pot_odds,
    encode_board_texture,
    encode_derived_features,
    encode_draws,
    hand_category,
)


class DerivedObservationTest(unittest.TestCase):
    def test_pot_odds(self) -> None:
        self.assertAlmostEqual(calculate_pot_odds(100, 20), 20 / 120)
        self.assertEqual(calculate_pot_odds(100, 0), 0)
        with self.assertRaises(ValueError):
            calculate_pot_odds(-1, 10)

    def test_all_nine_hand_categories(self) -> None:
        examples = (
            (["SA", "HK"], ["D2", "C5", "S9"], schema.CATEGORY_HIGH_CARD),
            (["SA", "HA"], ["D2", "C5", "S9"], schema.CATEGORY_PAIR),
            (["SA", "HA"], ["D2", "C2", "S9"], schema.CATEGORY_TWO_PAIR),
            (["SA", "HA"], ["DA", "C5", "S9"], schema.CATEGORY_THREE_OF_A_KIND),
            (["S2", "H3"], ["D4", "C5", "S6"], schema.CATEGORY_STRAIGHT),
            (["SA", "S3"], ["S5", "S7", "S9"], schema.CATEGORY_FLUSH),
            (["SA", "HA"], ["DA", "C5", "H5"], schema.CATEGORY_FULL_HOUSE),
            (["SA", "HA"], ["DA", "CA", "S9"], schema.CATEGORY_FOUR_OF_A_KIND),
            (["S2", "S3"], ["S4", "S5", "S6"], schema.CATEGORY_STRAIGHT_FLUSH),
        )

        for own_cards, board_cards, expected in examples:
            with self.subTest(expected=expected):
                self.assertEqual(hand_category(own_cards, board_cards), expected)

    def test_best_five_cards_are_selected_on_turn_and_river(self) -> None:
        self.assertEqual(
            hand_category(["SA", "HA"], ["DA", "C5", "H5", "S5"]),
            schema.CATEGORY_FULL_HOUSE,
        )
        self.assertEqual(
            hand_category(["SA", "S2"], ["S3", "S4", "S5", "HK", "DQ"]),
            schema.CATEGORY_STRAIGHT_FLUSH,
        )

    def test_preflop_category_is_high_card_or_pair(self) -> None:
        self.assertEqual(hand_category(["SA", "HK"], []), schema.CATEGORY_HIGH_CARD)
        self.assertEqual(hand_category(["SA", "HA"], []), schema.CATEGORY_PAIR)

    def test_flush_and_backdoor_flush_draws(self) -> None:
        flush_draw = encode_draws(["SA", "HK"], ["S2", "S7", "S9"])
        backdoor = encode_draws(["SA", "HK"], ["S2", "D7", "S9"])

        self.assertEqual(flush_draw[schema.DRAW_FLUSH], 1)
        self.assertEqual(flush_draw[schema.DRAW_BACKDOOR_FLUSH], 0)
        self.assertEqual(backdoor[schema.DRAW_FLUSH], 0)
        self.assertEqual(backdoor[schema.DRAW_BACKDOOR_FLUSH], 1)

    def test_open_ended_and_gutshot_draws(self) -> None:
        open_ended = encode_draws(["S5", "H6"], ["D7", "C8", "SA"])
        gutshot = encode_draws(["S5", "H6"], ["D7", "C9", "SA"])

        self.assertEqual(open_ended[schema.DRAW_OPEN_ENDED], 1)
        self.assertEqual(open_ended[schema.DRAW_GUTSHOT], 0)
        self.assertEqual(gutshot[schema.DRAW_OPEN_ENDED], 0)
        self.assertEqual(gutshot[schema.DRAW_GUTSHOT], 1)

    def test_draws_are_zero_after_river(self) -> None:
        draws = encode_draws(["SA", "HK"], ["S2", "S7", "D9", "C4", "H5"])
        np.testing.assert_array_equal(draws, np.zeros(4))

    def test_board_texture(self) -> None:
        texture = encode_board_texture(["S2", "H2", "S4"])
        self.assertEqual(texture[schema.BOARD_PAIRED], 1)
        self.assertEqual(texture[schema.BOARD_CONNECTED], 0)
        self.assertEqual(texture[schema.BOARD_TWO_TONE], 1)

        connected = encode_board_texture(["S2", "H3", "D4"])
        self.assertEqual(connected[schema.BOARD_CONNECTED], 1)

        monotone = encode_board_texture(["S2", "S7", "S9"])
        self.assertEqual(monotone[schema.BOARD_MONOTONE], 1)
        self.assertEqual(monotone[schema.BOARD_TWO_TONE], 0)

    def test_complete_derived_block_layout(self) -> None:
        encoded = encode_derived_features(
            ["SA", "HK"],
            ["D2", "C3", "S4"],
            pot=100,
            to_call=20,
        )
        self.assertEqual(encoded.shape, (18,))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded[1:10].sum(), 1)
        self.assertAlmostEqual(encoded[0], 20 / 120)


if __name__ == "__main__":
    unittest.main()
