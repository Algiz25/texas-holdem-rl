"""Powtarzalna ewaluacja polityk pokerowych i zapis wyników na dysku.

Ewaluator celowo nie korzysta z kolektora treningowego. Dzięki temu dokładnie
kontroluje przeciwników, seedy, pozycję ucznia oraz sposób liczenia statystyk.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
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
EVALUATION_SUITES = ("random", "passive", "mixed", "phase1_mix", "baseline")

MIXED_ACTION_WEIGHTS = np.asarray(config.MIXED_ACTION_WEIGHTS)
PHASE1_OPPONENTS = tuple(config.PHASE1_OPPONENT_WEIGHTS)
PHASE1_WEIGHTS = np.array(tuple(config.PHASE1_OPPONENT_WEIGHTS.values()))


def _evaluate_suite_in_worker(
    evaluator_class,
    num_tournaments: int,
    model_path: Path,
    baseline_model_path: Path | None,
    training_phase: int | str,
    report_dir: Path,
    suite: str,
    stage: str,
    step: int,
    seed_base: int,
    learner_mode: str,
) -> tuple[str, "EvaluationResult"]:
    """Policz jeden niezależny zestaw w osobnym procesie CPU.

    Proces tworzy własne środowisko i sam wczytuje politykę. Dzięki temu nie
    współdzielimy mutowalnego stanu turnieju, a seedy pozostają identyczne jak
    w wersji sekwencyjnej.
    """
    torch.set_num_threads(config.EVAL_WORKER_TORCH_THREADS)
    evaluator = evaluator_class(
        num_tournaments=num_tournaments,
        model_path=model_path,
        baseline_model_path=baseline_model_path,
        training_phase=training_phase,
        report_dir=report_dir,
    )
    policy = evaluator.load_policy() if learner_mode == "model" else None
    baseline_policy = evaluator.load_policy(baseline_model_path) if baseline_model_path else None

    if learner_mode == "model" and policy is None:
        raise RuntimeError(f"Nie udało się wczytać polityki z {model_path}")
    result = evaluator._evaluate_suite(
        policy,
        baseline_policy=baseline_policy,
        suite=suite,
        stage=stage,
        step=step,
        seed_base=seed_base,
        learner_mode=learner_mode,
    )
    return suite, result


def _evaluate_suite_worker_unpack(arguments) -> tuple[str, "EvaluationResult"]:
    """Adapter wymagany przez `ProcessPoolExecutor.map` dla wielu argumentów."""
    return _evaluate_suite_in_worker(*arguments)


@dataclass
class TournamentSample:
    """Surowy wynik jednego meczu potrzebny do późniejszej analizy statystycznej.

    Zapisujemy mały rekord zamiast całego przebiegu gry. Dzięki temu można
    policzyć bootstrapowe przedziały ufności i porównywać checkpointy na tych
    samych seedach, nie tworząc wielkich logów każdej pojedynczej akcji.
    """

    seed: int
    learner_seat: int
    hands: int
    chip_delta: float
    won: bool
    resolved: bool
    learner_eliminated: bool
    finish: float | None
    stop_reason: str


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
    resolved_tournaments: int
    learner_eliminations: int
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
    tournament_samples: list[TournamentSample] = field(default_factory=list)

    def csv_row(self) -> dict[str, Any]:
        """Spłaszcz najważniejsze metryki do formatu wygodnego dla arkusza."""
        row = {
            key: value
            for key, value in asdict(self).items()
            if key
            not in {
                "action_rates",
                "actions_by_street",
                "actions_by_hand_category",
                "tournament_samples",
            }
        }
        for action_name in ("fold", "check", "call", "raise_half", "raise_pot", "all_in"):
            row[f"{action_name}_rate"] = self.action_rates.get(action_name, 0.0)
        return row


class EvaluationAccumulator:
    """Zbiera surowe zdarzenia, a po serii wylicza stabilne metryki."""

    def __init__(self) -> None:
        self.completed_tournaments = 0
        self.resolved_tournaments = 0
        self.learner_eliminations = 0
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
        self.tournament_samples: list[TournamentSample] = []

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
        tournament_seed: int = 0,
        learner_seat: int = 0,
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
        hands = int(player_stats.observed_hands)
        self.hands += hands
        self.hand_wins += env.hand_wins[learner]

        chip_delta = float(env.tournament_chips[learner] - env.starting_chips)
        # Eliminacja ucznia jest rozstrzygnięciem nawet wtedy, gdy limit stu
        # rozdań zatrzymuje dalszą część turnieju. Poprzednia wersja wyrzucała
        # takie porażki z mianownika win rate, przez co np. 87 zwycięstw i 135
        # eliminacji mogło wyglądać jak 53%, zamiast uczciwego 29%.
        learner_eliminated = bool(
            getattr(env, "terminations", {}).get(learner, False)
            or env.tournament_chips[learner] <= 0
        )
        resolved = completed or learner_eliminated
        if resolved:
            self.resolved_tournaments += 1
        if learner_eliminated:
            self.learner_eliminations += 1

        # Wartości żetonów w środowisku mogą być skalarami NumPy. Ich
        # porównanie zwraca wtedy ``numpy.bool_``, którego standardowy encoder
        # JSON nie obsługuje, dlatego zapisujemy jawny, wbudowany ``bool``.
        won = bool(
            completed
            and not learner_eliminated
            and env.tournament_chips[learner] == max(env.tournament_chips.values())
        )
        finish = (
            float(env.finishing_positions[learner])
            if resolved and learner in env.finishing_positions
            else None
        )
        self.chip_deltas.append(chip_delta)
        if won:
            self.tournament_wins += 1
        if finish is not None:
            self.finishing_positions.append(finish)

        # Ten kompaktowy rekord wystarcza do bootstrapu, błędu standardowego
        # i testów parowanych. Nie zapisujemy kart ani akcji, więc raport rośnie
        # o zaledwie kilkaset bajtów na turniej, a nie o rozmiar pełnego replaya.
        self.tournament_samples.append(
            TournamentSample(
                seed=tournament_seed,
                learner_seat=learner_seat,
                hands=hands,
                chip_delta=chip_delta,
                won=won,
                resolved=resolved,
                learner_eliminated=learner_eliminated,
                finish=finish,
                stop_reason=stop_reason,
            )
        )

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
            resolved_tournaments=self.resolved_tournaments,
            learner_eliminations=self.learner_eliminations,
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
                self.resolved_tournaments,
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
            tournament_samples=list(self.tournament_samples),
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
        baseline_model_path: str | Path | None = None,
        training_phase: int | str = config.TRAINING_PHASE,
        report_dir: str | Path | None = None,
    ) -> None:
        self.num_tournaments = num_tournaments
        self.model_path = Path(model_path)
        self.baseline_model_path = Path(baseline_model_path) if baseline_model_path else None
        self.training_phase = training_phase
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.report_dir = Path(report_dir) if report_dir else evaluation_dir(self.algorithm_name)
        self.env = TexasHoldemTournament(
            num_players=config.NUM_PLAYERS,
            starting_chips=config.STARTING_CHIPS,
            debug=False,
        )

    def load_policy(self, path: Path | None = None):
        raise NotImplementedError("Podklasa musi zaimplementować ładowanie polityki")

    def evaluate(
        self,
        *,
        suites: tuple[str, ...] | None = None,
        stage: str = "manual",
        step: int = 0,
        seed_base: int = config.EVAL_VALIDATION_SEED,
        save_report: bool = True,
        learner_mode: str = "model",
    ) -> dict[str, EvaluationResult]:
        """Uruchom wszystkie wskazane zestawy i zwróć wyniki w pamięci."""
        if suites is None:
            # Domyślnie uruchamiamy test 'baseline' tylko jeśli podano ścieżkę do modelu bazowego
            suites = tuple(s for s in EVALUATION_SUITES if s != "baseline" or self.baseline_model_path is not None)

        unknown = set(suites) - set(EVALUATION_SUITES)
        if unknown:
            raise ValueError(f"Nieznane zestawy ewaluacyjne: {sorted(unknown)}")

        if learner_mode not in {"model", "random"}:
            raise ValueError(f"Nieznany tryb badanego gracza: {learner_mode}")

        print(
            f"\nEwaluacja {self.algorithm_name.upper()} ({stage}, krok {step:,}) "
            f"na urządzeniu {self.device}."
        )

        suite_arguments = [
            (
                self.__class__,
                self.num_tournaments,
                self.model_path,
                self.baseline_model_path,
                self.training_phase,
                self.report_dir,
                suite,
                stage,
                step,
                # Zestawy dostają rozłączne seedy, a kolejne checkpointy zawsze
                # używają dokładnie tych samych zakresów.
                seed_base + suite_index * 10_000,
                learner_mode,
            )
            for suite_index, suite in enumerate(suites)
        ]

        if len(suites) > 1 and config.EVAL_NUM_WORKERS > 1:
            worker_count = min(config.EVAL_NUM_WORKERS, len(suites))
            # `spawn` jest najbezpieczniejszy na macOS i nie dziedziczy stanu
            # PyTorch ani środowisk treningowych po procesie głównym.
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                evaluated = executor.map(
                    _evaluate_suite_worker_unpack,
                    suite_arguments,
                )
                results = dict(evaluated)
        else:
            # W trybie sekwencyjnym wczytujemy model tylko raz. Jest to
            # szybsze dla pojedynczego zestawu i stanowi lekki fallback dla
            # środowisk, w których procesy potomne są niedostępne.
            policy = self.load_policy() if learner_mode == "model" else None
            baseline_policy = self.load_policy(self.baseline_model_path) if self.baseline_model_path else None
            if learner_mode == "model" and policy is None:
                return {}
            results = {
                suite: self._evaluate_suite(
                    policy,
                    baseline_policy=baseline_policy,
                    suite=suite,
                    stage=stage,
                    step=step,
                    seed_base=seed_base + suite_index * 10_000,
                    learner_mode=learner_mode,
                )
                for suite_index, suite in enumerate(suites)
            }

        if save_report:
            self._save_report(results, stage=stage, step=step)
        self._print_results(results)
        if save_report:
            print(f"Raporty: {self.report_dir / 'evaluations_v3.csv'}\n")
        return results

    def _evaluate_suite(
        self,
        policy,
        *,
        baseline_policy=None,
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
                        baseline_policy=baseline_policy,
                        observation=obs,
                        mask=mask,  
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
                tournament_seed=tournament_seed,
                learner_seat=tournament_index % config.NUM_PLAYERS,
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


    @classmethod
    def _opponent_action(
        cls,
        opponent_type: str,
        legal_actions: np.ndarray,
        rng: np.random.Generator,
        *,
        baseline_policy=None,
        observation: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> int:
        if opponent_type == "baseline":
            if baseline_policy is None or observation is None or mask is None:
                raise ValueError("Brak polityki bazowej lub obserwacji do wykonania ruchu baseline.")
            return cls._policy_action(baseline_policy, observation, mask)
        
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
        # JSON zachowuje szczegółowe rozkłady oraz kompaktową próbkę każdego
        # turnieju. Najpierw zapisujemy plik tymczasowy, a dopiero kompletny
        # raport podmieniamy atomowo. Awaria serializacji nie pozostawi już
        # uciętego pliku wyglądającego jak prawidłowy wynik ewaluacji.
        json_path = self.report_dir / f"{stage}_step_{step:09d}.json"
        temporary_json_path = json_path.with_suffix(json_path.suffix + ".tmp")
        with temporary_json_path.open("w", encoding="utf-8") as json_file:
            json.dump(
                {suite: asdict(result) for suite, result in results.items()},
                json_file,
                ensure_ascii=False,
                indent=2,
            )
        temporary_json_path.replace(json_path)

        # CSV zapisujemy dopiero po udanym JSON-ie. Dzięki temu oba raporty
        # opisują ten sam kompletny pomiar. Nie umieszczamy w nim listy próbek,
        # więc pozostaje mały i zawiera jeden wiersz na zestaw przeciwników.
        # Nowy plik ma inną definicję win rate niż `evaluations_v2.csv`. Osobna
        # wersja nie miesza starych, zawyżonych wyników z poprawionymi.
        csv_path = self.report_dir / "evaluations_v3.csv"
        rows = [result.csv_row() for result in results.values()]
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        # Zachowujemy nazwę używaną przez istniejącą infrastrukturę maina.
        # Nowe analizy powinny korzystać z v3, ponieważ ma poprawiony mianownik
        # win rate; plik v2 jest wyłącznie kompatybilnym indeksem tych samych
        # nowych wierszy dla dotychczasowych skryptów i testów.
        legacy_csv_path = self.report_dir / "evaluations_v2.csv"
        legacy_write_header = not legacy_csv_path.exists()
        with legacy_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            if legacy_write_header:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _print_results(results: dict[str, EvaluationResult]) -> None:
        print("\n=== WYNIKI EWALUACJI ===")
        for suite, result in results.items():
            # Win rate istnieje dla ukończonych turniejów oraz dla każdej
            # eliminacji ucznia. Gdy limit zatrzymał wyłącznie nierozstrzygnięte
            # stoły, uczciwie raportujemy "n/d" zamiast sztucznego zera.
            tournament_win_rate = (
                f"{result.tournament_win_rate:6.1%}"
                if result.resolved_tournaments
                else "   n/d"
            )
            print(
                f"{suite:12s} | {result.bb_per_100:8.2f} bb/100 | "
                f"turnieje {tournament_win_rate} | "
                f"ręce {result.hands:5d} | decyzje {result.decisions:6d} | "
                f"rozstrzygnięte/pełne/100-rąk/awarie "
                f"{result.resolved_tournaments}/"
                f"{result.completed_tournaments}/"
                f"{result.hand_limited_matches}/"
                f"{result.action_limited_matches}"
            )
        print()
