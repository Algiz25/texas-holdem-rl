"""Statystyki graczy aktualizowane pomiędzy kolejnymi rozdaniami."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from observation import schema


OBSERVED_HANDS_SCALE = 100


@dataclass(slots=True)
class PlayerStatistics:
    observed_hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    aggressive_actions: int = 0
    call_actions: int = 0
    faced_raise: int = 0
    folded_to_raise: int = 0
    showdowns: int = 0
    showdown_wins: int = 0


def _ratio(numerator: int, denominator: int) -> np.float32:
    return np.float32(numerator / denominator) if denominator else np.float32(0.0)


class OpponentStatsTracker:
    """Gromadź publicznie dostępne statystyki wszystkich miejsc przy stole."""

    def __init__(self, seats: Sequence[str]) -> None:
        if len(seats) != schema.NUM_PLAYERS or len(set(seats)) != len(seats):
            raise ValueError("Statystyki wymagają dokładnie czterech unikalnych miejsc")

        self.seats = tuple(seats)
        self.players = {seat: PlayerStatistics() for seat in self.seats}
        self._active_players: set[str] | None = None
        self._vpip_players: set[str] = set()
        self._pfr_players: set[str] = set()

    def start_hand(self, active_players: Iterable[str]) -> None:
        """Rozpocznij zbieranie zdarzeń nowego rozdania."""
        if self._active_players is not None:
            raise RuntimeError("Poprzednie rozdanie nie zostało zakończone")

        active = set(active_players)
        if not active or not active <= set(self.seats):
            raise ValueError("Lista aktywnych graczy zawiera nieznane miejsca lub jest pusta")

        self._active_players = active
        self._vpip_players.clear()
        self._pfr_players.clear()

    def record_action(
        self,
        player: str,
        action: int,
        street: int,
        *,
        to_call: float = 0.0,
        facing_raise: bool = False,
        all_in_increases_bet: bool = False,
    ) -> None:
        """Zapisz decyzję potrzebną do obliczenia statystyk gracza."""
        if self._active_players is None:
            raise RuntimeError("Najpierw trzeba rozpocząć rozdanie")
        if player not in self._active_players:
            raise ValueError(f"Gracz {player!r} nie uczestniczy w tym rozdaniu")
        if not 0 <= action < schema.NUM_ACTIONS:
            raise ValueError(f"Nieznana akcja: {action}")
        if not 0 <= street < schema.NUM_STREETS:
            raise ValueError(f"Nieznana ulica: {street}")
        if to_call < 0:
            raise ValueError(f"to_call nie może być ujemne: {to_call}")

        stats = self.players[player]
        regular_raise = action in (
            schema.ACTION_RAISE_HALF_POT,
            schema.ACTION_RAISE_POT,
        )
        aggressive_all_in = (
            action == schema.ACTION_ALL_IN and all_in_increases_bet
        )
        aggressive_action = regular_raise or aggressive_all_in
        calling_action = (
            action in (schema.ACTION_CHECK_CALL, schema.ACTION_ALL_IN)
            and to_call > 0
            and not aggressive_all_in
        )

        if street == schema.STREET_PREFLOP:
            if aggressive_action or calling_action:
                self._vpip_players.add(player)
            if aggressive_action:
                self._pfr_players.add(player)
        else:
            if aggressive_action:
                stats.aggressive_actions += 1
            elif calling_action:
                stats.call_actions += 1

        if facing_raise:
            stats.faced_raise += 1
            if action == schema.ACTION_FOLD:
                stats.folded_to_raise += 1

    def finish_hand(
        self,
        *,
        showdown_players: Iterable[str] = (),
        winners: Iterable[str] = (),
    ) -> None:
        """Zatwierdź statystyki rozdania i opcjonalny wynik showdownu."""
        if self._active_players is None:
            raise RuntimeError("Brak rozpoczętego rozdania")

        showdown = set(showdown_players)
        winning_players = set(winners)
        if not showdown <= self._active_players:
            raise ValueError("Showdown zawiera gracza spoza rozdania")
        if not winning_players <= showdown:
            raise ValueError("Zwycięzca showdownu musi uczestniczyć w showdownie")

        for player in self._active_players:
            stats = self.players[player]
            stats.observed_hands += 1
            stats.vpip_hands += player in self._vpip_players
            stats.pfr_hands += player in self._pfr_players
            stats.showdowns += player in showdown
            stats.showdown_wins += player in winning_players

        self._active_players = None
        self._vpip_players.clear()
        self._pfr_players.clear()

    def encode(self, observer: str) -> np.ndarray:
        """Zwróć 18 wartości dla trzech przeciwników względem obserwatora."""
        if observer not in self.seats:
            raise ValueError(f"Nieznany obserwator: {observer!r}")

        observer_index = self.seats.index(observer)
        opponents = tuple(
            self.seats[(observer_index + offset) % schema.NUM_PLAYERS]
            for offset in range(1, schema.NUM_PLAYERS)
        )
        encoded = np.zeros(
            schema.NUM_OPPONENTS * schema.OPPONENT_STATS_SIZE,
            dtype=np.float32,
        )

        for opponent_index, player in enumerate(opponents):
            stats = self.players[player]
            start = opponent_index * schema.OPPONENT_STATS_SIZE
            decisions = stats.aggressive_actions + stats.call_actions

            encoded[start + schema.STAT_VPIP] = _ratio(
                stats.vpip_hands,
                stats.observed_hands,
            )
            encoded[start + schema.STAT_PFR] = _ratio(
                stats.pfr_hands,
                stats.observed_hands,
            )
            encoded[start + schema.STAT_AGGRESSION] = _ratio(
                stats.aggressive_actions,
                decisions,
            )
            encoded[start + schema.STAT_FOLD_TO_RAISE] = _ratio(
                stats.folded_to_raise,
                stats.faced_raise,
            )
            encoded[start + schema.STAT_SHOWDOWN_WIN_RATE] = _ratio(
                stats.showdown_wins,
                stats.showdowns,
            )
            encoded[start + schema.STAT_OBSERVED_HANDS] = np.float32(
                min(stats.observed_hands / OBSERVED_HANDS_SCALE, 1.0)
            )

        return encoded
