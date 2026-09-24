import unittest

import numpy as np

from observation.cards import card_index, encode_card_observation, encode_cards


class CardEncodingTest(unittest.TestCase):
    def test_card_indices_match_rlcard_order(self) -> None:
        self.assertEqual(card_index("SA"), 0)
        self.assertEqual(card_index("SK"), 12)
        self.assertEqual(card_index("HA"), 13)
        self.assertEqual(card_index("DA"), 26)
        self.assertEqual(card_index("CA"), 39)
        self.assertEqual(card_index("CK"), 51)

    def test_encode_cards_returns_float32_one_hot_vector(self) -> None:
        encoded = encode_cards(["SA", "H7"])

        self.assertEqual(encoded.shape, (52,))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(float(encoded.sum()), 2.0)
        self.assertEqual(encoded[card_index("SA")], 1.0)
        self.assertEqual(encoded[card_index("H7")], 1.0)

    def test_private_cards_and_board_are_separate(self) -> None:
        own, board = encode_card_observation(
            own_cards=["SA", "H7"],
            board_cards=["D2", "C3", "ST"],
        )

        self.assertEqual(float(own.sum()), 2.0)
        self.assertEqual(float(board.sum()), 3.0)
        self.assertEqual(own[card_index("D2")], 0.0)
        self.assertEqual(board[card_index("SA")], 0.0)

    def test_board_sizes_for_all_streets(self) -> None:
        boards = (
            [],
            ["S2", "H3", "D4"],
            ["S2", "H3", "D4", "C5"],
            ["S2", "H3", "D4", "C5", "S6"],
        )

        for board_cards in boards:
            with self.subTest(board_size=len(board_cards)):
                _, board = encode_card_observation(["SA", "HK"], board_cards)
                self.assertEqual(float(board.sum()), float(len(board_cards)))

    def test_duplicate_card_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_cards(["SA", "SA"])

        with self.assertRaises(ValueError):
            encode_card_observation(["SA", "HK"], ["SA", "D2", "C3"])

    def test_invalid_card_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            card_index("X1")

    def test_invalid_private_or_board_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_card_observation(["SA"], [])

        with self.assertRaises(ValueError):
            encode_card_observation(["SA", "HK"], ["D2"])


if __name__ == "__main__":
    unittest.main()
