"""Jednoagentowy widok turnieju używany wyłącznie do treningu DQN i PPO.

Podstawowe środowisko jest środowiskiem AEC: jeden krok oznacza jeden ruch
przy stole, niezależnie od tego, który gracz go wykonuje. To jest właściwy
interfejs do rozgrywania i ewaluacji pokera, ale nie do uczenia pojedynczego
DQN. Replay buffer DQN musi łączyć decyzję ucznia z jego kolejną decyzją, a
nie z obserwacją przeciwnika, który wykonuje następny ruch.

Ta nakładka pozostawia całą mechanikę pokera w ``TexasHoldemTournament``.
Wewnątrz wykonuje ruch ucznia oraz wszystkie następujące po nim ruchy botów.
Na zewnątrz zwraca sterowanie dopiero wtedy, gdy ponownie rusza się uczeń lub
gdy jego epizod się zakończy. Dzięki temu każda próbka ma poprawną postać:

    obserwacja ucznia -> akcja ucznia -> jego nagroda -> obserwacja ucznia
"""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np

import config
from environment import TexasHoldemTournament


# Korzystamy z tego samego rozkładu co ewaluator. Przeciwnik Mixed nie może
# zmieniać charakteru tylko dlatego, że został uruchomiony inną ścieżką kodu.
MIXED_ACTION_WEIGHTS = np.asarray(
    config.MIXED_ACTION_WEIGHTS,
    dtype=np.float64,
)


class PokerLearnerEnv(gym.Env):
    """Pokaż kolektorowi wyłącznie kolejne decyzje jednego ucznia.

    Przeciwnicy są częścią środowiska, a nie osobnymi algorytmami uczonymi
    przez ten sam replay buffer. Ich osobowości losujemy niezależnie na
    początku każdego turnieju i zachowujemy aż do końca tego turnieju.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        learner: str = "player_0",
        opponent_weights: Mapping[str, float] | None = None,
        initial_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.learner = learner
        self.poker_env = TexasHoldemTournament(
            num_players=config.NUM_PLAYERS,
            starting_chips=config.STARTING_CHIPS,
        )

        if learner not in self.poker_env.possible_agents:
            raise ValueError(f"Nieznany uczący się gracz: {learner!r}")

        weights = dict(opponent_weights or config.PHASE1_OPPONENT_WEIGHTS)
        supported = {"random", "passive", "mixed"}
        if set(weights) != supported:
            raise ValueError(
                "Faza 1 wymaga dokładnie wag: random, passive i mixed"
            )
        if any(value < 0 for value in weights.values()) or not np.isclose(
            sum(weights.values()),
            1.0,
        ):
            raise ValueError("Wagi przeciwników muszą być nieujemne i sumować się do 1")

        self._opponent_names = tuple(weights)
        self._opponent_probabilities = np.asarray(
            tuple(weights.values()),
            dtype=np.float64,
        )
        self._rng = np.random.default_rng()
        self._initial_seed = initial_seed
        self._episode_number = 0
        self.opponent_styles: dict[str, str] = {}
        self.total_table_actions = 0

        # Tianshou rozpoznaje maskowanie akcji po nazwach ``obs`` i ``mask``.
        # Wektor pozostaje dokładnie tym samym 222-elementowym wektorem, który
        # zwraca środowisko pokerowe; zmienia się tylko nazwa pól adaptera.
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.poker_env.observation_space(learner),
                "mask": gym.spaces.MultiBinary(config.ACTION_SPACE),
            }
        )
        self.action_space = gym.spaces.Discrete(config.ACTION_SPACE)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Rozpocznij turniej i dojdź do pierwszej decyzji ucznia."""
        # Kolektor zwykle wywołuje reset bez seedu. Dla nazwanego eksperymentu
        # wyprowadzamy wtedy kolejny, deterministyczny seed dla każdego turnieju.
        # Osobne fabryki nadają różne zakresy seedów poszczególnym workerom.
        effective_seed = seed
        if effective_seed is None and self._initial_seed is not None:
            effective_seed = self._initial_seed + self._episode_number
        self._episode_number += 1

        super().reset(seed=effective_seed)
        if effective_seed is not None:
            # Jeden generator steruje doborem osobowości i losowymi decyzjami
            # botów. Jawny seed daje powtarzalny test; trening bez seedu nadal
            # korzysta z niezależnej entropii w każdym procesie środowiska.
            self._rng = np.random.default_rng(effective_seed)

        self.poker_env.reset(seed=effective_seed, options=options)
        self.total_table_actions = 0
        self.opponent_styles = {
            agent: str(
                self._rng.choice(
                    self._opponent_names,
                    p=self._opponent_probabilities,
                )
            )
            for agent in self.poker_env.possible_agents
            if agent != self.learner
        }

        # Button może sprawić, że pierwszy ruch turnieju należy do bota.
        # Te ruchy nie są osobnymi próbkami DQN, dlatego rozgrywamy je przed
        # zwróceniem pierwszej obserwacji kolektorowi.
        _, opening_actions = self._play_opponents_until_learner()
        return self._learner_observation(), self._info(
            table_actions_this_step=opening_actions
        )

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Wykonaj jedną decyzję DQN i całą odpowiedź pozostałych graczy."""
        if self._learner_finished():
            raise RuntimeError("Wywołano step() po zakończeniu epizodu ucznia")
        if self.poker_env.agent_selection != self.learner:
            raise RuntimeError(
                "Nakładka DQN oddała sterowanie, mimo że nie rusza się uczeń"
            )

        current = self.poker_env.observe(self.learner)
        action = int(action)
        if not 0 <= action < config.ACTION_SPACE:
            raise ValueError(f"Akcja DQN jest poza zakresem: {action}")
        if not current["action_mask"][action]:
            # Nie zamieniamy błędnej akcji po cichu na fold. Taki fallback
            # fałszowałby replay buffer: zapisana akcja różniłaby się od tej,
            # którą naprawdę wykonało środowisko.
            raise ValueError(f"DQN wybrał niedozwoloną akcję: {action}")

        reward = self._step_underlying(action)
        opponent_reward, opponent_actions = self._play_opponents_until_learner()
        reward += opponent_reward
        table_actions_this_step = 1 + opponent_actions

        terminated = bool(self.poker_env.terminations.get(self.learner, False))
        truncated = bool(self.poker_env.truncations.get(self.learner, False))
        return (
            self._learner_observation(),
            float(reward),
            terminated,
            truncated,
            self._info(table_actions_this_step=table_actions_this_step),
        )

    def _play_opponents_until_learner(self) -> tuple[float, int]:
        """Rozgrywaj stół do następnej decyzji ucznia i sumuj jego nagrody."""
        learner_reward = 0.0
        action_count = 0

        while not self._learner_finished():
            selected = self.poker_env.agent_selection
            _, _, terminated, truncated, _ = self.poker_env.last()

            if selected == self.learner and not terminated and not truncated:
                break

            if terminated or truncated:
                # PettingZoo wymaga osobnego martwego kroku dla wyeliminowanego
                # gracza. Nie jest to akcja pokerowa, więc nie zwiększa licznika.
                self.poker_env.step(None)
                learner_reward += float(self.poker_env.rewards.get(self.learner, 0.0))
                continue

            observation = self.poker_env.observe(selected)
            opponent_action = self._choose_opponent_action(
                selected,
                observation["action_mask"],
            )
            learner_reward += self._step_underlying(opponent_action)
            action_count += 1

        return learner_reward, action_count

    def _step_underlying(self, action: int) -> float:
        """Wykonaj ruch stołu i pobierz natychmiastową nagrodę ucznia."""
        self.poker_env.step(action)
        self.total_table_actions += 1
        # Nagroda może pojawić się po ruchu dowolnego gracza kończącym rozdanie.
        # Odczytujemy ją bezpośrednio po każdym ruchu, zanim kolejny krok AEC ją
        # wyczyści, i przypisujemy do decyzji ucznia rozpoczynającej ten odcinek.
        return float(self.poker_env.rewards.get(self.learner, 0.0))

    def _choose_opponent_action(self, agent: str, mask: np.ndarray) -> int:
        """Wybierz legalny ruch zgodnie ze stałą osobowością przeciwnika."""
        legal_actions = np.flatnonzero(mask)
        if not len(legal_actions):
            raise RuntimeError(f"Aktywny przeciwnik {agent} nie ma legalnej akcji")

        style = self.opponent_styles[agent]
        if style == "passive":
            # Check/call zachowuje pasywny charakter i jest preferowany zawsze,
            # gdy silnik pokera dopuszcza tę akcję.
            if 1 in legal_actions:
                return 1
            if 0 in legal_actions:
                return 0
            return int(legal_actions[0])

        if style == "mixed":
            legal_weights = MIXED_ACTION_WEIGHTS * np.asarray(mask, dtype=np.float64)
            probabilities = legal_weights / legal_weights.sum()
            return int(self._rng.choice(config.ACTION_SPACE, p=probabilities))

        if style == "random":
            return int(self._rng.choice(legal_actions))

        raise RuntimeError(f"Nieobsługiwana osobowość przeciwnika: {style!r}")

    def _learner_finished(self) -> bool:
        return bool(
            self.poker_env.terminations.get(self.learner, False)
            or self.poker_env.truncations.get(self.learner, False)
        )

    def _learner_observation(self) -> dict[str, np.ndarray]:
        raw = self.poker_env.observe(self.learner)
        return {
            "obs": raw["observation"],
            "mask": raw["action_mask"].astype(np.int8, copy=False),
        }

    def _info(self, *, table_actions_this_step: int) -> dict:
        """Dołącz liczniki pozwalające odróżnić decyzje od ruchów stołu."""
        return {
            "learner_id": self.learner,
            "table_actions_this_step": table_actions_this_step,
            "total_table_actions": self.total_table_actions,
            "completed_hands": self.poker_env.completed_hands,
        }

    def close(self) -> None:
        self.poker_env.close()


def make_learner_env(initial_seed: int | None = None) -> PokerLearnerEnv:
    """Fabryka na poziomie modułu, którą można bezpiecznie wysłać do procesu."""
    return PokerLearnerEnv(initial_seed=initial_seed)
