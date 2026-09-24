"""Historia akcji widoczna dla wszystkich graczy w bieżącym rozdaniu."""

from collections.abc import Sequence

import numpy as np

from observation import schema


def normalize_count(count: int) -> np.float32:
    """Zamień nieograniczony licznik na wartość z zakresu 0-1."""
    if count < 0:
        raise ValueError(f"Licznik nie może być ujemny: {count}")
    return np.float32(count / (count + 1)) if count else np.float32(0.0)


class ActionHistory:
    """Śledź ostatnie ruchy oraz krótkie podsumowania czterech ulic."""

    def __init__(self, seats: Sequence[str]) -> None:
        if len(seats) != schema.NUM_PLAYERS or len(set(seats)) != len(seats):
            raise ValueError("Historia wymaga dokładnie czterech unikalnych miejsc")

        self.seats = tuple(seats)
        self.last_actions: dict[str, int | None] = dict.fromkeys(self.seats)
        self.last_aggressors: list[str | None] = [None] * schema.NUM_STREETS
        self.call_counts = [0] * schema.NUM_STREETS
        self.raise_counts = [0] * schema.NUM_STREETS

    def record(
        self,
        player: str,
        action: int,
        street: int,
        *,
        to_call: float = 0.0,
        all_in_increases_bet: bool = False,
    ) -> None:
        """Zapisz ruch wraz z kontekstem potrzebnym do klasyfikacji calla/all-ina."""
        if player not in self.last_actions:
            raise ValueError(f"Nieznany gracz: {player!r}")
        if not 0 <= action < schema.NUM_ACTIONS:
            raise ValueError(f"Nieznana akcja: {action}")
        if not 0 <= street < schema.NUM_STREETS:
            raise ValueError(f"Nieznana ulica: {street}")
        if to_call < 0:
            raise ValueError(f"to_call nie może być ujemne: {to_call}")

        self.last_actions[player] = action

        is_regular_raise = action in (
            schema.ACTION_RAISE_HALF_POT,
            schema.ACTION_RAISE_POT,
        )
        is_aggressive_all_in = (
            action == schema.ACTION_ALL_IN and all_in_increases_bet
        )

        if is_regular_raise or is_aggressive_all_in:
            self.raise_counts[street] += 1
            self.last_aggressors[street] = player
        elif action in (schema.ACTION_CHECK_CALL, schema.ACTION_ALL_IN) and to_call > 0:
            self.call_counts[street] += 1

    def encode(self, observer: str) -> np.ndarray:
        """Zakoduj historię z perspektywy wskazanego gracza."""
        if observer not in self.seats:
            raise ValueError(f"Nieznany obserwator: {observer!r}")

        observer_index = self.seats.index(observer)
        relative_seats = tuple(
            self.seats[(observer_index + offset) % schema.NUM_PLAYERS]
            for offset in range(schema.NUM_PLAYERS)
        )

        size = schema.STREET_SUMMARIES.stop - schema.LAST_ACTIONS.start
        encoded = np.zeros(size, dtype=np.float32)

        for relative_seat, player in enumerate(relative_seats):
            action = self.last_actions[player]
            if action is not None:
                encoded[relative_seat * schema.NUM_ACTIONS + action] = 1.0

        summaries_start = schema.LAST_ACTIONS.stop - schema.LAST_ACTIONS.start
        for street in range(schema.NUM_STREETS):
            block_start = summaries_start + street * schema.STREET_SUMMARY_SIZE
            aggressor = self.last_aggressors[street]
            aggressor_index = 0 if aggressor is None else 1 + relative_seats.index(aggressor)

            encoded[block_start + aggressor_index] = 1.0
            encoded[block_start + 5] = normalize_count(self.call_counts[street])
            encoded[block_start + 6] = normalize_count(self.raise_counts[street])

        return encoded
