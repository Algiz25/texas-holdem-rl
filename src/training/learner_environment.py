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
import torch

import config
from environment import TexasHoldemTournament
from models import MaskedActor, PokerFeatureExtractor
from tianshou.utils.net.discrete import ImplicitQuantileNetwork


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
        placement_reward_weight: float = 1.0,
        training_phase: int | None = None,
        self_play_policy_kind: str = "dqn",
        algo_name: str | None = None,
        self_play_opponent_epsilon: float = 0.0,
        max_historical_models: int | None = None,
    ) -> None:
        super().__init__()
        self.learner = learner
        self.placement_reward_weight = float(placement_reward_weight)
        # Bez parametrów adapter pozostaje prostym środowiskiem fazy 1, co
        # jest bezpieczne w testach i narzędziach. Trenerzy z maina przekazują
        # ``algo_name`` i zachowują globalnie wybraną fazę.
        effective_phase = (
            config.TRAINING_PHASE
            if training_phase is None and algo_name is not None
            else (1 if training_phase is None else training_phase)
        )
        self.training_phase = int(effective_phase)
        # ``algo_name`` zachowuje zgodność z trenerami PPO/SAC/IQN z maina.
        # DQN przekazuje bardziej opisową nazwę ``self_play_policy_kind``.
        policy_kind = algo_name or self_play_policy_kind
        if policy_kind not in {"dqn", "ppo", "sac", "iqn"}:
            raise ValueError("Nieobsługiwany rodzaj polityki self-play")
        self.self_play_policy_kind = policy_kind
        self.self_play_opponent_epsilon = float(self_play_opponent_epsilon)
        if not 0.0 <= self.self_play_opponent_epsilon <= 1.0:
            raise ValueError("Epsilon przeciwnika self-play musi należeć do [0, 1]")
        if max_historical_models is None:
            max_historical_models = (
                config.DQN_PHASE2_MAX_HISTORICAL_MODELS
                if policy_kind == "dqn"
                else config.PHASE2_MAX_HISTORICAL_MODELS
            )
        if max_historical_models <= 0:
            raise ValueError("Liga self-play musi mieścić co najmniej jeden model")
        self.poker_env = TexasHoldemTournament(
            num_players=config.NUM_PLAYERS,
            starting_chips=config.STARTING_CHIPS,
            placement_reward_weight=placement_reward_weight,
        )

        if learner not in self.poker_env.possible_agents:
            raise ValueError(f"Nieznany uczący się gracz: {learner!r}")

        if self.training_phase == 1:
            default_weights = (
                config.DQN_PHASE1_OPPONENT_WEIGHTS
                if policy_kind == "dqn"
                else config.PHASE1_OPPONENT_WEIGHTS
            )
            weights = dict(opponent_weights or default_weights)
            supported = {"random", "passive", "mixed"}
        elif self.training_phase == 2:
            default_weights = (
                config.DQN_PHASE2_OPPONENT_WEIGHTS
                if policy_kind == "dqn"
                else config.PHASE2_OPPONENT_WEIGHTS
            )
            weights = dict(opponent_weights or default_weights)
            supported = {
                "historical_self",
                "latest_self",
                "mixed",
                "passive",
            }
            if policy_kind == "dqn":
                supported.add("random")
        else:
            raise ValueError(f"Nieobsługiwana faza: {self.training_phase}")

        if set(weights) != supported:
            raise ValueError(
                f"Faza {self.training_phase} wymaga dokładnie wag: {supported}"
            )
        
        if any(value < 0 for value in weights.values()) or not np.isclose(
            sum(weights.values()), 1.0
        ):
            raise ValueError(
                "Wagi przeciwników muszą być nieujemne i sumować się do 1"
            )

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

        # Modele są prywatne dla procesu środowiska. Konkretna kopia zostaje
        # przypisana do miejsca przy stole w ``reset`` i nie zmienia się do
        # końca turnieju, nawet jeśli trener w międzyczasie rozsyła nowe wagi.
        self.latest_model: torch.nn.Module | None = None
        self.historical_models: list[torch.nn.Module] = []
        self.opponent_models: dict[str, torch.nn.Module] = {}
        self.max_historical = int(max_historical_models)

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

    # --- METODY DLA SELF-PLAY ---
    @property
    def latest_model_weights(self) -> dict | None:
        return None

    @latest_model_weights.setter
    def latest_model_weights(self, actor_state_dict: dict) -> None:
        """Zastąp najnowszego przeciwnika bez zmiany trwających turniejów."""
        model = self._new_self_play_model()
        model.load_state_dict(actor_state_dict)
        model.eval()
        self.latest_model = model

    @property
    def new_historical_model_weights(self) -> dict | None:
        return None

    @new_historical_model_weights.setter
    def new_historical_model_weights(self, actor_state_dict: dict) -> None:
        """Dodaj zamrożony snapshot do małej, ograniczonej pamięci ligi."""
        model = self._new_self_play_model()
        model.load_state_dict(actor_state_dict)
        model.eval()

        self.historical_models.append(model)
        if len(self.historical_models) > self.max_historical:
            # FIFO zachowuje deterministyczność i nie usuwa losowo innego
            # modelu w każdym workerze.
            self.historical_models.pop(0)

    @property
    def historical_model_replacement(self) -> None:
        return None

    @historical_model_replacement.setter
    def historical_model_replacement(self, replacement: tuple[int, dict]) -> None:
        """Zastąp wskazany model bez usuwania stałej kotwicy fazy 1."""
        index, actor_state_dict = replacement
        if not 0 <= int(index) < len(self.historical_models):
            raise IndexError("Indeks zastępowanego modelu ligi jest niepoprawny")
        model = self._new_self_play_model()
        model.load_state_dict(actor_state_dict)
        model.eval()
        self.historical_models[int(index)] = model

    def _get_nn_action(
        self,
        model: torch.nn.Module,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        """Wybierz legalną akcję zgodnie z semantyką użytego algorytmu."""
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)

            if self.self_play_policy_kind == "iqn":
                (quantiles, _), _ = model(
                    obs_tensor,
                    sample_size=config.IQN_SAMPLE_SIZE,
                )
                values = quantiles.mean(dim=2)
            else:
                values, _ = model(obs_tensor)
            values = values.masked_fill(~mask_tensor, -torch.inf)
            if self.self_play_policy_kind in {"ppo", "sac"}:
                # PPO i SAC zwracają logity polityki, więc zachowujemy ich
                # naturalną stochastyczność podczas self-play.
                action = torch.distributions.Categorical(logits=values).sample().item()
            elif self.self_play_policy_kind == "iqn":
                # Zachowujemy dotychczasową, prawie zachłanną semantykę IQN.
                action = torch.distributions.Categorical(
                    logits=values / 0.01
                ).sample().item()
            else:
                # DQN zwraca wartości Q, a nie logity prawdopodobieństwa.
                # Niewielki epsilon daje zamrożonym przeciwnikom różnorodność
                # podobną do stochastycznej polityki PPO, ale nadal respektuje
                # maskę legalnych akcji i semantykę wartości Q.
                legal_actions = np.flatnonzero(mask)
                if self._rng.random() < self.self_play_opponent_epsilon:
                    action = int(self._rng.choice(legal_actions))
                else:
                    action = int(torch.argmax(values, dim=-1).item())

        return int(action)

    def _new_self_play_model(self) -> torch.nn.Module:
        """Utwórz CPU-ową kopię sieci zgodną z algorytmem danego trenera."""
        if self.self_play_policy_kind == "iqn":
            feature_net = PokerFeatureExtractor(
                state_shape=config.OBSERVATION_SIZE
            ).to("cpu")
            return ImplicitQuantileNetwork(
                preprocess_net=feature_net,
                action_shape=config.ACTION_SPACE,
                num_cosines=config.IQN_NUM_COSINES,
            ).to("cpu")
        return MaskedActor(
            state_shape=config.OBSERVATION_SIZE,
            action_shape=config.ACTION_SPACE,
        ).to("cpu")

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
        self.opponent_models = {}
        for agent, style in self.opponent_styles.items():
            if style == "latest_self":
                if self.latest_model is None:
                    raise RuntimeError(
                        "Faza self-play nie otrzymała modelu latest_self przed resetem"
                    )
                self.opponent_models[agent] = self.latest_model
            elif style == "historical_self":
                if not self.historical_models:
                    raise RuntimeError(
                        "Faza self-play nie otrzymała żadnego modelu historycznego"
                    )
                index = int(self._rng.integers(len(self.historical_models)))
                self.opponent_models[agent] = self.historical_models[index]

        # Button może sprawić, że pierwszy ruch turnieju należy do bota.
        # Te ruchy nie są osobnymi próbkami DQN, dlatego rozgrywamy je przed
        # zwróceniem pierwszej obserwacji kolektorowi.
        _, opening_actions = self._play_opponents_until_learner()
        return self._learner_observation(), self._info(
            table_actions_this_step=opening_actions,
            chip_reward=0.0,
            placement_reward=0.0,
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
        # Środowisko zwraca jedną łączną nagrodę. Do diagnostyki rozdzielamy ją
        # na część żetonową oraz terminalną premię za miejsce. Suma obu części
        # pozostaje dokładnie nagrodą zapisywaną w replay bufferze.
        placement_reward = 0.0
        if terminated and self.learner in self.poker_env.finishing_positions:
            finishing_position = self.poker_env.finishing_positions[self.learner]
            placement_reward = (
                self.placement_reward_weight
                * (2.5 - finishing_position)
                * 2.0
            )
        chip_reward = float(reward) - placement_reward
        return (
            self._learner_observation(),
            float(reward),
            terminated,
            truncated,
            self._info(
                table_actions_this_step=table_actions_this_step,
                chip_reward=chip_reward,
                placement_reward=placement_reward,
            ),
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
                observation["observation"],
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

    def _choose_opponent_action(
        self,
        agent: str,
        obs: np.ndarray,
        mask: np.ndarray,
    ) -> int:
        """Wybierz legalny ruch zgodnie ze stałą osobowością przeciwnika."""
        legal_actions = np.flatnonzero(mask)
        if not len(legal_actions):
            raise RuntimeError(f"Aktywny przeciwnik {agent} nie ma legalnej akcji")

        style = self.opponent_styles[agent]

        # --- Obsługa sieci neuronowych ---
        if style in {"latest_self", "historical_self"}:
            try:
                model = self.opponent_models[agent]
            except KeyError as error:
                raise RuntimeError(
                    f"Gracz {agent} nie ma modelu przypisanego na ten turniej"
                ) from error
            return self._get_nn_action(model, obs, mask)
        # -----------------------------------------

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

    def _info(
        self,
        *,
        table_actions_this_step: int,
        chip_reward: float,
        placement_reward: float,
    ) -> dict:
        """Dołącz lekką diagnostykę bez zapisywania pełnego przebiegu gry."""
        return {
            "learner_id": self.learner,
            "table_actions_this_step": table_actions_this_step,
            "total_table_actions": self.total_table_actions,
            "completed_hands": self.poker_env.completed_hands,
            "chip_reward": float(chip_reward),
            "placement_reward": float(placement_reward),
        }

    def close(self) -> None:
        self.poker_env.close()


def make_learner_env(
    initial_seed: int | None = None,
    placement_reward_weight: float = 1.0,
    training_phase: int | None = None,
    self_play_policy_kind: str = "dqn",
    algo_name: str | None = None,
    opponent_weights: Mapping[str, float] | None = None,
    self_play_opponent_epsilon: float = 0.0,
    max_historical_models: int | None = None,
) -> PokerLearnerEnv:
    """Fabryka na poziomie modułu, którą można bezpiecznie wysłać do procesu."""
    return PokerLearnerEnv(
        initial_seed=initial_seed,
        placement_reward_weight=placement_reward_weight,
        training_phase=training_phase,
        self_play_policy_kind=self_play_policy_kind,
        algo_name=algo_name,
        opponent_weights=opponent_weights,
        self_play_opponent_epsilon=self_play_opponent_epsilon,
        max_historical_models=max_historical_models,
    )
