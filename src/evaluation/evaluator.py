"""Powtarzalna ewaluacja polityk pokerowych i zapis wyników na dysku.

Ewaluator celowo nie korzysta z kolektora treningowego. Dzięki temu dokładnie
kontroluje przeciwników, seedy, pozycję ucznia oraz sposób liczenia statystyk.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tianshou.data import Batch

import config
from environment import TexasHoldemTournament
from observation import schema
from paths import evaluation_dir


ACTION_NAMES = ("fold", "check_call", "raise_half", "raise_pot", "all_in")
STREET_NAMES = ("preflop", "flop", "turn", "river")
HAND_CATEGORY_NAMES = (
    "high_card",
    "pair",
    "two_pair",
    "three_of_a_kind",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",
    "straight_flush",
)

# Każdy zestaw odpowiada na inne pytanie. Osobne wyniki pokazują, czy model
# naprawdę się rozwija, czy nauczył się wykorzystywać tylko jeden typ bota.
EVALUATION_SUITES = ("random", "passive", "mixed", "phase1_mix")

MIXED_ACTION_WEIGHTS = np.array([0.20, 0.45, 0.18, 0.12, 0.05])
PHASE1_OPPONENTS = tuple(config.PHASE1_OPPONENT_WEIGHTS)
PHASE1_WEIGHTS = np.array(tuple(config.PHASE1_OPPONENT_WEIGHTS.values()))


@dataclass
class EvaluationResult:
    """Podsumowanie jednego checkpointu przeciwko jednemu zestawowi botów."""

    stage: str
    step: int
    checkpoint: str
    opponent_suite: str
    seed_base: int
    tournaments: int
    completed_tournaments: int
    hand_limited_matches: int
    action_limited_matches: int
    truncated_tournaments: int
    hands: int
    decisions: int
    tournament_wins: int
    hand_wins: int
    total_chip_delta: float
    mean_chip_delta_per_tournament: float
    mean_chip_delta_per_hand: float
    bb_per_100: float
    tournament_win_rate: float
    average_finish: float
    hand_win_rate: float
    chip_delta_ci95: float
    vpip: float
    pfr: float
    showdown_rate: float
    showdown_win_rate: float
    illegal_action_rate: float
    action_rates: dict[str, float] = field(default_factory=dict)
    actions_by_street: dict[str, dict[str, int]] = field(default_factory=dict)
    actions_by_hand_category: dict[str, dict[str, int]] = field(default_factory=dict)

    def csv_row(self) -> dict[str, Any]:
        """Spłaszcz najważniejsze metryki do formatu wygodnego dla arkusza."""
        row = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"action_rates", "actions_by_street", "actions_by_hand_category"}
        }
        for action_name in ("fold", "check", "call", "raise_half", "raise_pot", "all_in"):
            row[f"{action_name}_rate"] = self.action_rates.get(action_name, 0.0)
        return row


class EvaluationAccumulator:
    """Zbiera surowe zdarzenia, a po serii wylicza stabilne metryki."""

    def __init__(self) -> None:
        self.completed_tournaments = 0
        self.hand_limited_matches = 0
        self.action_limited_matches = 0
        self.truncated_tournaments = 0
        self.hands = 0
        self.decisions = 0
        self.tournament_wins = 0
        self.hand_wins = 0
        self.illegal_actions = 0
        self.chip_deltas: list[float] = []
        self.finishing_positions: list[float] = []
        self.actions: Counter[str] = Counter()
        self.actions_by_street: dict[str, Counter[str]] = defaultdict(Counter)
        self.actions_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        self.observed_hands = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.showdowns = 0
        self.showdown_wins = 0

    def record_action(self, observation: np.ndarray, action: int, legal: bool) -> None:
        self.decisions += 1
        self.illegal_actions += not legal

        # CHECK/CALL jest jedną akcją modelu, ale raport rozdziela ją na check
        # i call na podstawie kosztu pozostania w rozdaniu.
        if not 0 <= action < schema.NUM_ACTIONS:
            action_name = "invalid"
        elif action == schema.ACTION_CHECK_CALL:
            action_name = "call" if observation[schema.TO_CALL_INDEX] > 0.0 else "check"
        else:
            action_name = ACTION_NAMES[action]
        self.actions[action_name] += 1

        street_block = observation[schema.STREET]
        street = STREET_NAMES[int(np.argmax(street_block))]
        self.actions_by_street[street][action_name] += 1

        category_block = observation[schema.HAND_CATEGORY]
        category = (
            HAND_CATEGORY_NAMES[int(np.argmax(category_block))]
            if np.any(category_block)
            else "preflop_unknown"
        )
        self.actions_by_category[category][action_name] += 1

    def record_tournament(
        self,
        env: TexasHoldemTournament,
        learner: str,
        *,
        stop_reason: str,
    ) -> None:
        if stop_reason == "completed":
            self.completed_tournaments += 1
        elif stop_reason == "hand_limit":
            self.hand_limited_matches += 1
        elif stop_reason == "action_limit":
            self.action_limited_matches += 1
            self.truncated_tournaments += 1
        else:
            raise ValueError(f"Nieznany powód zakończenia ewaluacji: {stop_reason}")

        completed = stop_reason == "completed"
        player_stats = env.opponent_stats.players[learner]
        # Po odpadnięciu ucznia turniej może trwać dalej. Do bb/100 liczymy
        # wyłącznie rozdania, w których badany gracz faktycznie uczestniczył.
        self.hands += player_stats.observed_hands
        self.hand_wins += env.hand_wins[learner]

        chip_delta = float(env.tournament_chips[learner] - env.starting_chips)
        self.chip_deltas.append(chip_delta)
        if completed and env.tournament_chips[learner] == max(env.tournament_chips.values()):
            self.tournament_wins += 1
        if completed:
            self.finishing_positions.append(env.finishing_positions[learner])

        # Tracker środowiska liczy VPIP poprawnie: blind i darmowy check nie są
        # dobrowolnym wejściem do puli.
        self.observed_hands += player_stats.observed_hands
        self.vpip_hands += player_stats.vpip_hands
        self.pfr_hands += player_stats.pfr_hands
        self.showdowns += player_stats.showdowns
        self.showdown_wins += player_stats.showdown_wins

    def finish(
        self,
        *,
        stage: str,
        step: int,
        checkpoint: Path,
        opponent_suite: str,
        seed_base: int,
        tournaments: int,
    ) -> EvaluationResult:
        total_chip_delta = float(sum(self.chip_deltas))
        chip_std = float(np.std(self.chip_deltas, ddof=1)) if tournaments > 1 else 0.0
        ci95 = 1.96 * chip_std / math.sqrt(tournaments) if tournaments else 0.0

        action_rates = {
            action: _safe_ratio(count, self.decisions)
            for action, count in sorted(self.actions.items())
        }
        # Wszystkie znane akcje występują w raporcie nawet wtedy, gdy model nie
        # wybrał ich ani razu. Ułatwia to porównywanie kolejnych wierszy CSV.
        for action in ("fold", "check", "call", "raise_half", "raise_pot", "all_in"):
            action_rates.setdefault(action, 0.0)

        return EvaluationResult(
            stage=stage,
            step=step,
            checkpoint=str(checkpoint),
            opponent_suite=opponent_suite,
            seed_base=seed_base,
            tournaments=tournaments,
            completed_tournaments=self.completed_tournaments,
            hand_limited_matches=self.hand_limited_matches,
            action_limited_matches=self.action_limited_matches,
            truncated_tournaments=self.truncated_tournaments,
            hands=self.hands,
            decisions=self.decisions,
            tournament_wins=self.tournament_wins,
            hand_wins=self.hand_wins,
            total_chip_delta=total_chip_delta,
            mean_chip_delta_per_tournament=_safe_ratio(total_chip_delta, tournaments),
            mean_chip_delta_per_hand=_safe_ratio(total_chip_delta, self.hands),
            bb_per_100=100.0 * _safe_ratio(
                total_chip_delta,
                self.hands * config.BIG_BLIND,
            ),
            tournament_win_rate=_safe_ratio(
                self.tournament_wins,
                self.completed_tournaments,
            ),
            average_finish=(
                float(np.mean(self.finishing_positions))
                if self.finishing_positions
                else 0.0
            ),
            hand_win_rate=_safe_ratio(self.hand_wins, self.hands),
            chip_delta_ci95=ci95,
            vpip=_safe_ratio(self.vpip_hands, self.observed_hands),
            pfr=_safe_ratio(self.pfr_hands, self.observed_hands),
            showdown_rate=_safe_ratio(self.showdowns, self.observed_hands),
            showdown_win_rate=_safe_ratio(self.showdown_wins, self.showdowns),
            illegal_action_rate=_safe_ratio(self.illegal_actions, self.decisions),
            action_rates=action_rates,
            actions_by_street=_nested_counters_to_dict(self.actions_by_street),
            actions_by_hand_category=_nested_counters_to_dict(self.actions_by_category),
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _nested_counters_to_dict(
    counters: dict[str, Counter[str]],
) -> dict[str, dict[str, int]]:
    return {name: dict(sorted(counter.items())) for name, counter in sorted(counters.items())}


class BasePokerEvaluator:
    """Wspólna logika ewaluacji modeli DQN i PPO."""

    algorithm_name = "base"

    def __init__(
        self,
        num_tournaments: int = config.EVAL_TOURNAMENTS_PER_SUITE,
        model_path: str | Path = "model.pth",
        training_phase: int | str = config.TRAINING_PHASE,
        report_dir: str | Path | None = None,
    ) -> None:
        self.num_tournaments = num_tournaments
        self.model_path = Path(model_path)
        self.training_phase = training_phase
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.report_dir = Path(report_dir) if report_dir else evaluation_dir(self.algorithm_name)
        self.env = TexasHoldemTournament(
            num_players=config.NUM_PLAYERS,
            starting_chips=config.STARTING_CHIPS,
            debug=False,
        )

    def load_policy(self):
        raise NotImplementedError("Podklasa musi zaimplementować ładowanie polityki")

    def evaluate(
        self,
        *,
        suites: tuple[str, ...] = EVALUATION_SUITES,
        stage: str = "manual",
        step: int = 0,
        seed_base: int = config.EVAL_VALIDATION_SEED,
        save_report: bool = True,
        learner_mode: str = "model",
    ) -> dict[str, EvaluationResult]:
        """Uruchom wszystkie wskazane zestawy i zwróć wyniki w pamięci."""
        unknown = set(suites) - set(EVALUATION_SUITES)
        if unknown:
            raise ValueError(f"Nieznane zestawy ewaluacyjne: {sorted(unknown)}")

        if learner_mode not in {"model", "random"}:
            raise ValueError(f"Nieznany tryb badanego gracza: {learner_mode}")

        policy = self.load_policy() if learner_mode == "model" else None
        if learner_mode == "model" and policy is None:
            return {}

        print(
            f"\nEwaluacja {self.algorithm_name.upper()} ({stage}, krok {step:,}) "
            f"na urządzeniu {self.device}."
        )
        results = {
            suite: self._evaluate_suite(
                policy,
                suite=suite,
                stage=stage,
                step=step,
                # Zestawy dostają rozłączne seedy, a kolejne checkpointy zawsze
                # używają dokładnie tych samych zakresów.
                seed_base=seed_base + suite_index * 10_000,
                learner_mode=learner_mode,
            )
            for suite_index, suite in enumerate(suites)
        }

        if save_report:
            self._save_report(results, stage=stage, step=step)
        self._print_results(results)
        if save_report:
            print(f"Raporty: {self.report_dir / 'evaluations_v2.csv'}\n")
        return results

    def _evaluate_suite(
        self,
        policy,
        *,
        suite: str,
        stage: str,
        step: int,
        seed_base: int,
        learner_mode: str,
    ) -> EvaluationResult:
        accumulator = EvaluationAccumulator()

        for tournament_index in range(self.num_tournaments):
            # Cztery kolejne turnieje używają tego samego rozdania początkowego,
            # lecz uczeń siedzi kolejno na każdym miejscu przy stole.
            learner = f"player_{tournament_index % config.NUM_PLAYERS}"
            tournament_seed = seed_base + tournament_index // config.NUM_PLAYERS
            rng = np.random.default_rng(tournament_seed + 1_000_000)
            opponents = self._opponents_for_tournament(suite, learner, rng)
            self.env.reset(seed=tournament_seed)

            step_count = 0
            stop_reason = "completed"
            for agent in self.env.agent_iter():
                observation, _, termination, truncation, _ = self.env.last()
                if termination or truncation:
                    self.env.step(None)
                    continue

                obs = observation["observation"]
                mask = observation["action_mask"]
                legal_actions = np.flatnonzero(mask)

                if agent == learner:
                    action = (
                        int(rng.choice(legal_actions))
                        if learner_mode == "random"
                        else self._policy_action(policy, obs, mask)
                    )
                    legal = 0 <= action < len(mask) and bool(mask[action])
                    accumulator.record_action(obs, action, legal)
                    if not legal:
                        # Błędna akcja jest raportowana, ale turniej trwa dalej.
                        action = int(rng.choice(legal_actions))
                else:
                    action = self._opponent_action(
                        opponents[agent],
                        legal_actions,
                        rng,
                    )

                self.env.step(action)
                step_count += 1

                # Kończymy planowo po stałej liczbie rozdań. Wynik żetonowy i
                # wszystkie statystyki z tych rozdań pozostają ważne; jedynie
                # win-rate całego turnieju nie jest wtedy dostępny.
                if self.env.completed_hands >= config.EVAL_MAX_HANDS_PER_MATCH:
                    stop_reason = "hand_limit"
                    break
                if step_count >= config.EVAL_MAX_ACTIONS_PER_MATCH:
                    stop_reason = "action_limit"
                    break

            accumulator.record_tournament(
                self.env,
                learner,
                stop_reason=stop_reason,
            )

        return accumulator.finish(
            stage=stage,
            step=step,
            checkpoint=self.model_path,
            opponent_suite=suite,
            seed_base=seed_base,
            tournaments=self.num_tournaments,
        )

    @staticmethod
    def _policy_action(policy, observation: np.ndarray, mask: np.ndarray) -> int:
        batch = Batch(
            obs=Batch(
                observation=np.expand_dims(observation, axis=0),
                action_mask=np.expand_dims(mask, axis=0),
            ),
            info={},
        )
        with torch.inference_mode():
            result = policy(batch)
        action = result.act[0]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().item()
        return int(action)

    @staticmethod
    def _opponents_for_tournament(
        suite: str,
        learner: str,
        rng: np.random.Generator,
    ) -> dict[str, str]:
        opponents: dict[str, str] = {}
        for player_index in range(config.NUM_PLAYERS):
            player = f"player_{player_index}"
            if player == learner:
                continue
            opponents[player] = (
                str(rng.choice(PHASE1_OPPONENTS, p=PHASE1_WEIGHTS))
                if suite == "phase1_mix"
                else suite
            )
        return opponents

    @staticmethod
    def _opponent_action(
        opponent_type: str,
        legal_actions: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        if opponent_type == "passive":
            if schema.ACTION_CHECK_CALL in legal_actions:
                return schema.ACTION_CHECK_CALL
            if schema.ACTION_FOLD in legal_actions:
                return schema.ACTION_FOLD
            return int(legal_actions[0])

        if opponent_type == "mixed":
            legal_weights = MIXED_ACTION_WEIGHTS[legal_actions]
            probabilities = legal_weights / legal_weights.sum()
            return int(rng.choice(legal_actions, p=probabilities))

        if opponent_type == "random":
            return int(rng.choice(legal_actions))

        raise ValueError(f"Nieznany typ przeciwnika: {opponent_type}")

    def _save_report(
        self,
        results: dict[str, EvaluationResult],
        *,
        stage: str,
        step: int,
    ) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        # Druga wersja raportu rozróżnia prawidłowe zakończenie po limicie
        # rozdań od awaryjnego limitu akcji. Nie dopisujemy nowych kolumn do
        # starego CSV, bo powstałby plik z niezgodnym nagłówkiem.
        csv_path = self.report_dir / "evaluations_v2.csv"
        rows = [result.csv_row() for result in results.values()]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        # JSON zachowuje szczegółowe rozkłady ulic i kategorii układu, których
        # nie warto rozpychać na dziesiątki kolumn w głównym pliku CSV.
        json_path = self.report_dir / f"{stage}_step_{step:09d}.json"
        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                {suite: asdict(result) for suite, result in results.items()},
                json_file,
                ensure_ascii=False,
                indent=2,
            )

    @staticmethod
    def _print_results(results: dict[str, EvaluationResult]) -> None:
        print("\n=== WYNIKI EWALUACJI ===")
        for suite, result in results.items():
            # Win-rate turniejowy nie istnieje, jeśli wszystkie mecze zostały
            # planowo zakończone po 100 rozdaniach. Pokazanie 0% sugerowałoby
            # przegraną, dlatego w terminalu wyświetlamy wtedy "n/d".
            tournament_win_rate = (
                f"{result.tournament_win_rate:6.1%}"
                if result.completed_tournaments
                else "   n/d"
            )
            print(
                f"{suite:12s} | {result.bb_per_100:8.2f} bb/100 | "
                f"turnieje {tournament_win_rate} | "
                f"ręce {result.hands:5d} | decyzje {result.decisions:6d} | "
                f"pełne/100-rąk/awarie "
                f"{result.completed_tournaments}/"
                f"{result.hand_limited_matches}/"
                f"{result.action_limited_matches}"
            )
        print()
