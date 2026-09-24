"""Kodowanie publicznego stanu stołu z perspektywy obserwującego gracza."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from observation import schema
from observation.actions import ActionHistory
from observation.cards import encode_card_observation
from observation.derived import encode_derived_features
from observation.opponent_stats import OpponentStatsTracker


@dataclass(frozen=True, slots=True)
class PlayerSnapshot:
    """Widoczny stan jednego gracza w aktualnym rozdaniu."""

    stack: float
    street_contribution: float
    hand_contribution: float
    active: bool
    folded: bool = False
    all_in: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("stack", self.stack),
            ("street_contribution", self.street_contribution),
            ("hand_contribution", self.hand_contribution),
        ):
            if value < 0:
                raise ValueError(f"{name} nie może być ujemne: {value}")

        if self.folded and self.all_in:
            raise ValueError("Gracz nie może być jednocześnie po foldzie i all-in")

        if self.street_contribution > self.hand_contribution:
            raise ValueError(
                "Wkład na ulicy nie może przekraczać wkładu w całym rozdaniu"
            )

        if self.all_in and self.stack != 0:
            raise ValueError("Gracz all-in musi mieć zerowy pozostały stack")

        if not self.active and (self.folded or self.all_in):
            raise ValueError(
                "Wyeliminowany gracz nie może mieć statusu fold ani all-in"
            )

        if not self.active and any(
            (self.stack, self.street_contribution, self.hand_contribution)
        ):
            raise ValueError("Wyeliminowany gracz musi mieć zerowe wartości żetonów")


def relative_seat_order(observer: str, seats: Sequence[str]) -> tuple[str, ...]:
    """Ułóż miejsca jako: obserwator, następny, kolejny, ostatni."""
    if len(seats) != schema.NUM_PLAYERS:
        raise ValueError(
            f"Oczekiwano {schema.NUM_PLAYERS} miejsc, otrzymano: {len(seats)}"
        )

    if len(set(seats)) != len(seats):
        raise ValueError("Nazwy miejsc graczy muszą być unikalne")

    try:
        observer_index = seats.index(observer)
    except ValueError as error:
        raise ValueError(f"Obserwator {observer!r} nie występuje na liście miejsc") from error

    return tuple(
        seats[(observer_index + offset) % len(seats)]
        for offset in range(len(seats))
    )


def normalize_chips(chips: float, chip_scale: float) -> np.float32:
    """Znormalizuj liczbę żetonów do zakresu 0-1."""
    if chip_scale <= 0:
        raise ValueError(f"chip_scale musi być dodatni, otrzymano: {chip_scale}")

    if chips < 0:
        raise ValueError(f"Liczba żetonów nie może być ujemna: {chips}")

    if chips > chip_scale:
        raise ValueError(
            f"Liczba żetonów {chips} przekracza skalę turnieju {chip_scale}"
        )

    return np.float32(chips / chip_scale)


def calculate_to_call(
    highest_street_contribution: float,
    own_street_contribution: float,
    own_stack: float,
) -> float:
    """Oblicz rzeczywisty koszt sprawdzenia z uwzględnieniem krótkiego stacka."""
    for name, value in (
        ("highest_street_contribution", highest_street_contribution),
        ("own_street_contribution", own_street_contribution),
        ("own_stack", own_stack),
    ):
        if value < 0:
            raise ValueError(f"{name} nie może być ujemne: {value}")

    missing_chips = max(0.0, highest_street_contribution - own_street_contribution)
    return min(missing_chips, own_stack)


def encode_base_observation(
    *,
    observer: str,
    seats: Sequence[str],
    player_states: Mapping[str, PlayerSnapshot],
    button: str,
    street: int,
    pot: float,
    to_call: float,
    chip_scale: float,
    own_cards: Sequence[str],
    board_cards: Sequence[str],
    action_history: ActionHistory | None = None,
    opponent_stats: OpponentStatsTracker | None = None,
) -> np.ndarray:
    """Zbuduj podstawowy stan oraz opcjonalną historię akcji."""
    if not 0 <= street < schema.NUM_STREETS:
        raise ValueError(
            f"street musi być w zakresie 0-{schema.NUM_STREETS - 1}, "
            f"otrzymano: {street}"
        )

    relative_seats = relative_seat_order(observer, seats)

    missing_states = set(seats) - set(player_states)
    unexpected_states = set(player_states) - set(seats)
    if missing_states or unexpected_states:
        raise ValueError(
            "player_states musi zawierać dokładnie wszystkich graczy; "
            f"brakujące={sorted(missing_states)}, "
            f"nadmiarowe={sorted(unexpected_states)}"
        )

    if button not in seats:
        raise ValueError(f"Button {button!r} nie występuje na liście miejsc")

    own_cards_encoded, board_cards_encoded = encode_card_observation(
        own_cards,
        board_cards,
    )

    observation = np.zeros(schema.OBSERVATION_SIZE, dtype=np.float32)
    observation[schema.OWN_CARDS] = own_cards_encoded
    observation[schema.BOARD_CARDS] = board_cards_encoded

    for relative_seat, player_name in enumerate(relative_seats):
        player = player_states[player_name]

        observation[schema.PLAYER_STACKS.start + relative_seat] = normalize_chips(
            player.stack,
            chip_scale,
        )
        observation[
            schema.STREET_CONTRIBUTIONS.start + relative_seat
        ] = normalize_chips(player.street_contribution, chip_scale)
        observation[
            schema.HAND_CONTRIBUTIONS.start + relative_seat
        ] = normalize_chips(player.hand_contribution, chip_scale)
        observation[schema.PLAYER_ACTIVE.start + relative_seat] = float(
            player.active
        )
        observation[schema.PLAYER_FOLDED.start + relative_seat] = float(
            player.folded
        )
        observation[schema.PLAYER_ALL_IN.start + relative_seat] = float(
            player.all_in
        )

    button_relative_seat = relative_seats.index(button)
    observation[schema.BUTTON_POSITION.start + button_relative_seat] = 1.0
    observation[schema.STREET.start + street] = 1.0
    observation[schema.POT_INDEX] = normalize_chips(pot, chip_scale)
    observation[schema.TO_CALL_INDEX] = normalize_chips(to_call, chip_scale)

    if action_history is not None:
        observation[
            schema.LAST_ACTIONS.start : schema.STREET_SUMMARIES.stop
        ] = action_history.encode(observer)

    if opponent_stats is not None:
        observation[schema.OPPONENT_STATS] = opponent_stats.encode(observer)

    observation[schema.POT_ODDS_INDEX :] = encode_derived_features(
        own_cards,
        board_cards,
        pot,
        to_call,
    )

    return observation
