"""Kodowanie widocznych kart do bloków one-hot obserwacji."""

from collections.abc import Sequence

import numpy as np

from observation.schema import NUM_CARDS


# Kolejność musi pozostać zgodna z mapowaniem używanym przez RLCard:
# SA, S2, ..., SK, HA, ..., HK, DA, ..., DK, CA, ..., CK.
SUITS = "SHDC"
RANKS = "A23456789TJQK"

CARD_TO_INDEX = {
    f"{suit}{rank}": suit_index * len(RANKS) + rank_index
    for suit_index, suit in enumerate(SUITS)
    for rank_index, rank in enumerate(RANKS)
}

VALID_BOARD_CARD_COUNTS = {0, 3, 4, 5}


def card_index(card: str) -> int:
    """Zwróć indeks 0-51 dla karty zapisanej w formacie RLCard, np. ``SA``."""
    if not isinstance(card, str):
        raise TypeError(f"Karta musi być napisem, otrzymano: {type(card).__name__}")

    normalized_card = card.upper()

    try:
        return CARD_TO_INDEX[normalized_card]
    except KeyError as error:
        raise ValueError(f"Nieprawidłowa karta: {card!r}") from error


def encode_cards(cards: Sequence[str]) -> np.ndarray:
    """Zakoduj kolekcję kart jako 52-elementowy wektor one-hot."""
    normalized_cards = [card.upper() if isinstance(card, str) else card for card in cards]

    if len(normalized_cards) != len(set(normalized_cards)):
        raise ValueError(f"Lista zawiera powtórzoną kartę: {cards!r}")

    encoded = np.zeros(NUM_CARDS, dtype=np.float32)

    for card in normalized_cards:
        encoded[card_index(card)] = 1.0

    return encoded


def encode_card_observation(
    own_cards: Sequence[str],
    board_cards: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Zakoduj osobno karty agenta oraz publiczny board.

    Funkcja celowo przyjmuje wyłącznie dwie prywatne karty obserwującego
    agenta. Karty przeciwników nigdy nie powinny trafiać do tego interfejsu.
    """
    if len(own_cards) != 2:
        raise ValueError(
            f"Agent musi mieć dokładnie 2 własne karty, otrzymano: {len(own_cards)}"
        )

    if len(board_cards) not in VALID_BOARD_CARD_COUNTS:
        allowed = ", ".join(str(count) for count in sorted(VALID_BOARD_CARD_COUNTS))
        raise ValueError(
            f"Board musi zawierać {allowed} kart, otrzymano: {len(board_cards)}"
        )

    all_visible_cards = [
        *(card.upper() if isinstance(card, str) else card for card in own_cards),
        *(card.upper() if isinstance(card, str) else card for card in board_cards),
    ]

    if len(all_visible_cards) != len(set(all_visible_cards)):
        raise ValueError("Ta sama karta występuje jednocześnie w widocznych kartach")

    return encode_cards(own_cards), encode_cards(board_cards)
