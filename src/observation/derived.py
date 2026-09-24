"""Pochodne cechy pokerowe obliczane wyłącznie z widocznych informacji."""

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

import numpy as np

from observation import schema
from observation.cards import encode_card_observation


RANK_VALUES = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

STRAIGHT_WINDOWS = (
    {14, 2, 3, 4, 5},
    *({start + offset for offset in range(5)} for start in range(2, 11)),
)


def calculate_pot_odds(pot: float, to_call: float) -> np.float32:
    """Zwróć część końcowej puli, którą trzeba dopłacić, aby sprawdzić."""
    if pot < 0 or to_call < 0:
        raise ValueError("Pula i to_call nie mogą być ujemne")
    if to_call == 0:
        return np.float32(0.0)
    return np.float32(to_call / (pot + to_call))


def _rank_values(cards: Sequence[str]) -> set[int]:
    return {RANK_VALUES[card[1]] for card in cards}


def _has_straight(ranks: set[int]) -> bool:
    return any(window <= ranks for window in STRAIGHT_WINDOWS)


def _five_card_category(cards: Sequence[str]) -> int:
    ranks = [RANK_VALUES[card[1]] for card in cards]
    counts = sorted(Counter(ranks).values(), reverse=True)
    flush = len({card[0] for card in cards}) == 1
    straight = _has_straight(set(ranks))

    if straight and flush:
        return schema.CATEGORY_STRAIGHT_FLUSH
    if counts == [4, 1]:
        return schema.CATEGORY_FOUR_OF_A_KIND
    if counts == [3, 2]:
        return schema.CATEGORY_FULL_HOUSE
    if flush:
        return schema.CATEGORY_FLUSH
    if straight:
        return schema.CATEGORY_STRAIGHT
    if counts == [3, 1, 1]:
        return schema.CATEGORY_THREE_OF_A_KIND
    if counts == [2, 2, 1]:
        return schema.CATEGORY_TWO_PAIR
    if counts == [2, 1, 1, 1]:
        return schema.CATEGORY_PAIR
    return schema.CATEGORY_HIGH_CARD


def hand_category(own_cards: Sequence[str], board_cards: Sequence[str]) -> int:
    """Zwróć indeks najlepszej aktualnie utworzonej kategorii układu."""
    encode_card_observation(own_cards, board_cards)
    cards = [*(card.upper() for card in own_cards), *(card.upper() for card in board_cards)]

    if len(cards) < 5:
        return (
            schema.CATEGORY_PAIR
            if cards[0][1] == cards[1][1]
            else schema.CATEGORY_HIGH_CARD
        )

    return max(_five_card_category(group) for group in combinations(cards, 5))


def encode_draws(
    own_cards: Sequence[str],
    board_cards: Sequence[str],
) -> np.ndarray:
    """Zakoduj draw do koloru, OESD, gutshot i backdoor flush draw."""
    encode_card_observation(own_cards, board_cards)
    encoded = np.zeros(4, dtype=np.float32)

    if len(board_cards) not in (3, 4):
        return encoded

    own = [card.upper() for card in own_cards]
    visible = [*own, *(card.upper() for card in board_cards)]
    suit_counts = Counter(card[0] for card in visible)
    own_suits = {card[0] for card in own}

    encoded[schema.DRAW_FLUSH] = float(
        any(count == 4 and suit in own_suits for suit, count in suit_counts.items())
    )
    encoded[schema.DRAW_BACKDOOR_FLUSH] = float(
        len(board_cards) == 3
        and any(count == 3 and suit in own_suits for suit, count in suit_counts.items())
    )

    ranks = _rank_values(visible)
    if _has_straight(ranks):
        return encoded

    ranks_with_low_ace = ranks | ({1} if 14 in ranks else set())
    open_ended = any(
        set(range(start, start + 4)) <= ranks_with_low_ace
        for start in range(2, 11)
    )
    one_card_straight_out = any(
        len(window - ranks) == 1 for window in STRAIGHT_WINDOWS
    )

    encoded[schema.DRAW_OPEN_ENDED] = float(open_ended)
    encoded[schema.DRAW_GUTSHOT] = float(one_card_straight_out and not open_ended)
    return encoded


def encode_board_texture(board_cards: Sequence[str]) -> np.ndarray:
    """Zakoduj sparowanie, połączenie i liczbę kolorów publicznego boardu."""
    encoded = np.zeros(4, dtype=np.float32)
    if not board_cards:
        return encoded

    board = [card.upper() for card in board_cards]
    ranks = _rank_values(board)
    suits = {card[0] for card in board}

    encoded[schema.BOARD_PAIRED] = float(len(ranks) < len(board))
    encoded[schema.BOARD_CONNECTED] = float(
        len(board) >= 3
        and any(len(window & ranks) >= 3 for window in STRAIGHT_WINDOWS)
    )
    encoded[schema.BOARD_TWO_TONE] = float(len(board) >= 3 and len(suits) == 2)
    encoded[schema.BOARD_MONOTONE] = float(len(board) >= 3 and len(suits) == 1)
    return encoded


def encode_derived_features(
    own_cards: Sequence[str],
    board_cards: Sequence[str],
    pot: float,
    to_call: float,
) -> np.ndarray:
    """Zbuduj kompletny blok indeksów 204-221."""
    encoded = np.zeros(
        schema.OBSERVATION_SIZE - schema.POT_ODDS_INDEX,
        dtype=np.float32,
    )
    encoded[0] = calculate_pot_odds(pot, to_call)
    encoded[1 + hand_category(own_cards, board_cards)] = 1.0
    encoded[10:14] = encode_draws(own_cards, board_cards)
    encoded[14:18] = encode_board_texture(board_cards)
    return encoded
