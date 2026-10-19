import argparse
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger

import config
from training.base_trainer import BasePokerTrainer
from training.diagnostic_dqn import DiagnosticDQN
from training.learner_environment import make_learner_env
from evaluation.evaluator_dqn import DQNEvaluator
from models import MaskedActor
from paths import dqn_run_dir, tensorboard_run_dir


RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def save_training_state(
    dqn: DQN,
    env_step: int,
    path: Path,
    *,
    best_validation_score: float,
    epsilon_tau_steps: int,
    placement_reward_weight: float,
    training_phase: int = 1,
    self_play_pool_paths: tuple[str, ...] = (),
    runtime_config: dict | None = None,
) -> None:
    """Zapisz stan DQN potrzebny do bezpiecznej kontynuacji treningu.

    Stan algorytmu zawiera sieć ucznia, sieć docelową i optymalizator. Celowo
    nie zapisujemy replay buffera: przy milionie obserwacji zajmowałby setki
    megabajtów. Po wznowieniu bufor jest ponownie rozgrzewany zgodnie z fazą
    (25 tys. w fazie 1, 50 tys. w fazie 2), ale wagi i momentum optymalizatora
    pozostają zachowane.
    """
    phase_two = int(training_phase) == 2
    epsilon = (
        phase_two_epsilon(env_step, epsilon_tau_steps)
        if phase_two
        else phase_one_epsilon(env_step, epsilon_tau_steps)
    )
    runtime_config = dict(runtime_config or {})
    phase_two_learning_rate_value = phase_two_learning_rate(
        env_step,
        initial=runtime_config.get(
            "learning_rate_initial", config.DQN_PHASE2_LEARNING_RATE
        ),
        final=runtime_config.get(
            "learning_rate_final", config.DQN_PHASE2_LEARNING_RATE_AFTER_1M
        ),
        boundary=runtime_config.get(
            "learning_rate_boundary", config.DQN_PHASE2_LR_BOUNDARY
        ),
    )
    payload = {
        "format_version": 8,
        # Od wersji 2 jeden krok oznacza decyzję ucznia, a nie dowolną akcję
        # przy stole. Jawna jednostka zapobiega cichej kontynuacji starego,
        # wieloagentowego treningu z nieporównywalnym licznikiem kroków.
        "step_unit": "learner_decisions",
        "algorithm_state": dqn.state_dict(),
        "algorithm_iteration": dqn._iter,
        "completed_env_steps": env_step,
        # Nazwa harmonogramu i tau są częścią stanu. Wznowienie z inną
        # funkcją eksploracji zmieniałoby eksperyment w połowie przebiegu.
        "epsilon_schedule": "exponential",
        "training_phase": int(training_phase),
        "epsilon": epsilon,
        "epsilon_tau_steps": epsilon_tau_steps,
        "epsilon_min": (
            config.DQN_PHASE2_EPS_MIN
            if phase_two
            else config.DQN_RAND_PHASE_EPS_MIN
        ),
        # Harmonogramy są częścią eksperymentu tak samo jak nagroda. Zapisane
        # wartości pozwalają odrzucić przypadkowe wznowienie ze starszymi
        # ustawieniami, które zmieniłoby znaczenie drugiej części treningu.
        "learning_rate_schedule": (
            runtime_config.get("learning_rate_schedule", "phase2_piecewise_v1")
            if phase_two
            else "phase1_piecewise_v1"
        ),
        "learning_rate": (
            phase_two_learning_rate_value
            if phase_two
            else phase_one_learning_rate(env_step)
        ),
        "update_ratio_schedule": (
            runtime_config.get("update_ratio_schedule", "phase2_constant_v1")
            if phase_two
            else "phase1_piecewise_v1"
        ),
        "update_ratio": (
            runtime_config.get("update_ratio", config.DQN_PHASE2_UPDATE_RATIO)
            if phase_two
            else phase_one_update_ratio(env_step)
        ),
        "replay_buffer_size": runtime_config.get(
            "replay_buffer_size", config.DQN_BUFFER_SIZE
        ),
        # Sygnał nagrody zmienia dane w replay bufferze. Wznowienie pod inną
        # wagą mieszałoby dwie różne funkcje celu w jednym eksperymencie.
        "placement_reward_weight": placement_reward_weight,
        "huber_loss_delta": config.DQN_HUBER_LOSS_DELTA,
        # Wynik jest częścią stanu sterującego treningiem. Bez niego wznowiony
        # run mógłby uznać pierwszy, nawet słabszy checkpoint za „najlepszy” i
        # nadpisać rzeczywiście najlepszy model sprzed przerwania.
        "best_validation_score": best_validation_score,
        "observation_size": config.OBSERVATION_SIZE,
        "action_space": config.ACTION_SPACE,
        # Ścieżki są lekkim, trwałym opisem ligi. Same modele pozostają
        # osobnymi checkpointami, więc stan treningu nie rośnie o dziesiątki MB.
        "self_play_pool_paths": list(self_play_pool_paths),
        # Pełny opis profilu chroni wznowienie przed cichą zmianą ligi,
        # learning rate, bufora albo sposobu doboru przeciwników.
        "runtime_config": runtime_config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


@contextmanager
def single_training_process(checkpoint_dir: Path):
    """Nie pozwól uruchomić dwóch procesów zapisujących ten sam run."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lock_path = checkpoint_dir / "training.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            raise RuntimeError(
                "Inny trening DQN już działa. Nie uruchamiaj drugiego procesu."
            ) from error

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file, fcntl.LOCK_UN)

def phase_one_epsilon(env_step: int, tau_steps: int | None = None) -> float:
    """Wykładniczo zmniejsz eksplorację według liczby decyzji ucznia.

    Epsilon zbliża się płynnie do minimum, ale nigdy go nie przekracza. Stała
    czasowa ``tau`` określa tempo: po ``tau`` pozostaje 1/e początkowej
    różnicy między epsilonem maksymalnym i minimalnym.
    """
    tau_steps = tau_steps or config.DQN_PHASE1_EPS_TAU_STEPS
    if tau_steps <= 0:
        raise ValueError("Stała czasowa epsilon musi być dodatnia")
    elapsed_steps = max(env_step, 0)
    return (
        config.DQN_RAND_PHASE_EPS_MIN
        + (config.DQN_EPS_MAX - config.DQN_RAND_PHASE_EPS_MIN)
        * math.exp(-elapsed_steps / tau_steps)
    )


def phase_two_epsilon(env_step: int, tau_steps: int | None = None) -> float:
    """Łagodna eksploracja podczas dalszego uczenia już silnego modelu."""
    tau_steps = tau_steps or config.DQN_PHASE2_EPS_TAU_STEPS
    if tau_steps <= 0:
        raise ValueError("Stała czasowa epsilon self-play musi być dodatnia")
    elapsed_steps = max(env_step, 0)
    return (
        config.DQN_PHASE2_EPS_MIN
        + (config.DQN_PHASE2_EPS_MAX - config.DQN_PHASE2_EPS_MIN)
        * math.exp(-elapsed_steps / tau_steps)
    )


def phase_one_learning_rate(env_step: int) -> float:
    """Zmniejszaj krok optymalizatora po nauczeniu się podstaw strategii."""
    if env_step < config.DQN_PHASE1_LR_FIRST_BOUNDARY:
        return config.DQN_LEARNING_RATE
    if env_step < config.DQN_PHASE1_LR_SECOND_BOUNDARY:
        return config.DQN_PHASE1_LR_AFTER_500K
    return config.DQN_PHASE1_LR_AFTER_2M


def phase_two_learning_rate(
    env_step: int,
    *,
    initial: float = config.DQN_PHASE2_LEARNING_RATE,
    final: float = config.DQN_PHASE2_LEARNING_RATE_AFTER_1M,
    boundary: int = config.DQN_PHASE2_LR_BOUNDARY,
) -> float:
    """Zmniejsz LR po pierwszym milionie nowych decyzji self-play."""
    if env_step < boundary:
        return initial
    return final


def phase_two_profile(name: str) -> dict:
    """Zwróć kompletny, niemutowalny w trakcie runu profil self-play."""
    if name == "standard":
        return {
            "name": name,
            "opponent_weights": dict(config.DQN_PHASE2_OPPONENT_WEIGHTS),
            "max_historical_models": config.DQN_PHASE2_MAX_HISTORICAL_MODELS,
            "history_interval_decisions": config.DQN_PHASE2_HISTORY_INTERVAL_DECISIONS,
            "league_retention": "fifo",
            "opponent_epsilon": 0.0,
            "replay_buffer_size": config.DQN_BUFFER_SIZE,
            "learning_rate_initial": config.DQN_PHASE2_LEARNING_RATE,
            "learning_rate_final": config.DQN_PHASE2_LEARNING_RATE_AFTER_1M,
            "learning_rate_boundary": config.DQN_PHASE2_LR_BOUNDARY,
            "learning_rate_schedule": "phase2_piecewise_v1",
            "update_ratio": config.DQN_PHASE2_UPDATE_RATIO,
            "update_ratio_schedule": "phase2_constant_v1",
        }
    if name == "ppo_style":
        return {
            "name": name,
            "opponent_weights": dict(
                config.DQN_PHASE2_PPO_STYLE_OPPONENT_WEIGHTS
            ),
            "max_historical_models": (
                config.DQN_PHASE2_PPO_STYLE_MAX_HISTORICAL_MODELS
            ),
            "history_interval_decisions": (
                config.DQN_PHASE2_PPO_STYLE_HISTORY_INTERVAL_DECISIONS
            ),
            "league_retention": "anchored_rotation",
            "opponent_epsilon": config.DQN_PHASE2_PPO_STYLE_OPPONENT_EPSILON,
            "replay_buffer_size": config.DQN_PHASE2_PPO_STYLE_BUFFER_SIZE,
            "learning_rate_initial": config.DQN_PHASE2_PPO_STYLE_LEARNING_RATE,
            "learning_rate_final": (
                config.DQN_PHASE2_PPO_STYLE_LEARNING_RATE_AFTER_BOUNDARY
            ),
            "learning_rate_boundary": config.DQN_PHASE2_PPO_STYLE_LR_BOUNDARY,
            "learning_rate_schedule": "phase2_piecewise_ppo_style_v1",
            "update_ratio": config.DQN_PHASE2_PPO_STYLE_UPDATE_RATIO,
            "update_ratio_schedule": "phase2_constant_ppo_style_v1",
        }
    raise ValueError(f"Nieznany profil self-play: {name!r}")


def phase_one_update_ratio(env_step: int) -> float:
    """Trenuj intensywnie na początku, a później ograniczaj nadpisywanie Q."""
    if env_step < config.DQN_PHASE1_UPDATE_FIRST_BOUNDARY:
        return config.DQN_UPDATE_RATIO
    if env_step < config.DQN_PHASE1_UPDATE_SECOND_BOUNDARY:
        return config.DQN_PHASE1_UPDATE_RATIO_AFTER_500K
    return config.DQN_PHASE1_UPDATE_RATIO_AFTER_2M


def extract_dqn_model_weights(policy_state_dict: dict) -> dict:
    """Wyjmij samą sieć Q z checkpointu polityki Tianshou.

    Środowiska przeciwników nie potrzebują target network ani optymalizatora.
    Klucze checkpointu DQN zaczynają się od ``model.``; usuwamy ten prefiks,
    aby można je było wczytać bezpośrednio do ``MaskedActor``.
    """
    weights = {
        key.removeprefix("model."): value.detach().cpu()
        for key, value in policy_state_dict.items()
        if key.startswith("model.")
    }
    if not weights:
        raise ValueError("Checkpoint DQN nie zawiera wag sieci pod prefiksem 'model.'")
    return weights


def set_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    """Ustaw learning rate w optymalizatorze używanym przez Tianshou.

    Tianshou 2.x nie wystawia bezpośrednio optymalizatora PyTorch. Opakowuje go
    we własny obiekt ``Algorithm.Optimizer`` i przechowuje właściwy obiekt w
    polu ``_optim``. Obsługujemy również surowy optymalizator PyTorch, dzięki
    czemu ta funkcja pozostaje zgodna z prostszymi testami i starszym kodem.

    Jawny błąd jest celowy: jeśli przyszła wersja biblioteki ponownie zmieni
    interfejs, trening ma zatrzymać się przed pierwszą aktualizacją zamiast
    cicho zignorować zaplanowany harmonogram learning rate.
    """
    torch_optimizer = getattr(optimizer, "_optim", optimizer)
    parameter_groups = getattr(torch_optimizer, "param_groups", None)
    if parameter_groups is None:
        raise TypeError(
            "Nie można odnaleźć grup parametrów optymalizatora Tianshou/PyTorch"
        )
    for parameter_group in parameter_groups:
        parameter_group["lr"] = learning_rate


def configure_cpu_runtime() -> None:
    """Zapobiega przeciążeniu procesora przy wielu środowiskach.
    
    Gdy używamy SubprocVectorEnv (np. 8 procesów), domyślne zachowanie PyTorch
    (używanie wszystkich rdzeni przez każdy proces) prowadzi do drastycznego
    spadku wydajności. Ograniczenie do 1 wątku na proces jest optymalne.
    """
    torch.set_num_threads(config.TORCH_NUM_THREADS)
    torch.set_num_interop_threads(config.TORCH_NUM_INTEROP_THREADS)


def ensure_finite_model(model: torch.nn.Module, env_step: int) -> None:
    """Przerwij trening od razu, gdy wagi zawierają NaN albo nieskończoność."""
    invalid_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all()
    ]
    if invalid_parameters:
        names = ", ".join(invalid_parameters)
        raise FloatingPointError(
            f"Niestabilny DQN po {env_step:,} decyzjach ucznia; "
            f"niepoprawne parametry: {names}"
        )


def validate_phase_one_configuration() -> None:
    """Wykryj literówki w konfiguracji przed kosztownym uruchomieniem."""
    if not np.isclose(sum(config.DQN_PHASE1_OPPONENT_WEIGHTS.values()), 1.0):
        raise ValueError("Wagi przeciwników fazy 1 muszą sumować się do 1.0")
    if config.DQN_PHASE1_EPS_TAU_STEPS <= 0:
        raise ValueError("Stała czasowa epsilon musi być dodatnia")
    if config.DQN_PHASE1_PLACEMENT_REWARD_WEIGHT < 0.0:
        raise ValueError("Waga nagrody za miejsce nie może być ujemna")
    if not (
        0
        < config.DQN_PHASE1_LR_FIRST_BOUNDARY
        < config.DQN_PHASE1_LR_SECOND_BOUNDARY
    ):
        raise ValueError("Progi harmonogramu learning rate muszą rosnąć")
    if not (
        config.DQN_LEARNING_RATE
        >= config.DQN_PHASE1_LR_AFTER_500K
        >= config.DQN_PHASE1_LR_AFTER_2M
        > 0
    ):
        raise ValueError("Learning rate fazy 1 musi być dodatni i malejący")
    if not (
        0
        < config.DQN_PHASE1_UPDATE_FIRST_BOUNDARY
        < config.DQN_PHASE1_UPDATE_SECOND_BOUNDARY
    ):
        raise ValueError("Progi harmonogramu update ratio muszą rosnąć")
    if not (
        config.DQN_UPDATE_RATIO
        >= config.DQN_PHASE1_UPDATE_RATIO_AFTER_500K
        >= config.DQN_PHASE1_UPDATE_RATIO_AFTER_2M
        > 0
    ):
        raise ValueError("Update ratio fazy 1 musi być dodatnie i malejące")


class DQNPokerTrainer(BasePokerTrainer):
    def __init__(
        self,
        *args,
        resume_path: Path | None = None,
        start_step: int | None = None,
        run_name: str = "dqn",
        epsilon_tau_steps: int = config.DQN_PHASE1_EPS_TAU_STEPS,
        placement_reward_weight: float = config.DQN_PHASE1_PLACEMENT_REWARD_WEIGHT,
        base_model_path: Path | None = None,
        historical_model_paths: tuple[Path, ...] = (),
        self_play_runtime_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.requested_start_step = start_step
        self.starting_step = 0
        self.run_name = run_name
        self.epsilon_tau_steps = epsilon_tau_steps
        self.placement_reward_weight = placement_reward_weight
        self.base_model_path = base_model_path
        self.historical_model_paths = list(historical_model_paths)
        self.self_play_runtime_config = dict(
            self_play_runtime_config or phase_two_profile("standard")
        )
        self.max_historical_models = int(
            self.self_play_runtime_config["max_historical_models"]
        )
        self.history_interval_decisions = int(
            self.self_play_runtime_config["history_interval_decisions"]
        )
        self.replay_buffer_size = int(
            self.self_play_runtime_config["replay_buffer_size"]
        )
        self.self_play_pool_paths: list[Path] = []
        self._last_history_step = -1

    @staticmethod
    def _checkpoint_policy_state(path: Path) -> dict:
        """Wczytaj lekki checkpoint polityki używany przez przeciwnika."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "algorithm_state" in payload:
            raise ValueError(
                f"Model ligi musi być plikiem wag step_*.pth, nie pełnym stanem: {path}"
            )
        return payload

    def _broadcast_latest_opponent(self, policy) -> None:
        weights = extract_dqn_model_weights(policy.state_dict())
        self.train_envs.set_env_attr("latest_model_weights", weights)
        self.test_envs.set_env_attr("latest_model_weights", weights)

    def _broadcast_historical_checkpoint(self, path: Path) -> None:
        policy_state = self._checkpoint_policy_state(path)
        weights = extract_dqn_model_weights(policy_state)
        self.train_envs.set_env_attr("new_historical_model_weights", weights)
        self.test_envs.set_env_attr("new_historical_model_weights", weights)

    def _replace_historical_checkpoint(self, index: int, path: Path) -> None:
        """Zastąp ten sam slot we wszystkich workerach ligi."""
        policy_state = self._checkpoint_policy_state(path)
        weights = extract_dqn_model_weights(policy_state)
        replacement = (index, weights)
        self.train_envs.set_env_attr("historical_model_replacement", replacement)
        self.test_envs.set_env_attr("historical_model_replacement", replacement)

    def _write_league_manifest(self) -> None:
        """Zapisz skład ligi, aby Ctrl+C i wznowienie były deterministyczne."""
        manifest = self.checkpoint_dir / "self_play_pool.json"
        payload = {
            "format_version": 2,
            "profile": self.self_play_runtime_config["name"],
            "retention": self.self_play_runtime_config["league_retention"],
            "max_models": self.max_historical_models,
            "models": [str(path.resolve()) for path in self.self_play_pool_paths],
        }
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest)

    def _initialize_self_play_league(self, policy) -> None:
        """Rozprowadź mistrza i odtwórz trwałą pulę przed pierwszym resetem."""
        self._broadcast_latest_opponent(policy)

        requested: list[Path] = []
        manifest = self.checkpoint_dir / "self_play_pool.json"
        if manifest.exists():
            try:
                stored = json.loads(manifest.read_text(encoding="utf-8"))
                stored_profile = stored.get("profile", "standard")
                if stored_profile != self.self_play_runtime_config["name"]:
                    raise ValueError(
                        "Manifest ligi pochodzi z innego profilu self-play: "
                        f"{stored_profile!r}."
                    )
                # Manifest jest autorytatywny przy wznowieniu. Ponowne
                # dodawanie modeli przekazanych w CLI wypierałoby najnowsze
                # snapshoty i po cichu cofało skład ligi po każdym Ctrl+C.
                requested.extend(Path(path) for path in stored.get("models", []))
            except (json.JSONDecodeError, OSError):
                pass
        if not requested:
            requested.extend(self.historical_model_paths)
            if self.base_model_path is not None:
                requested.insert(0, self.base_model_path)

        unique: list[Path] = []
        seen: set[Path] = set()
        for path in requested:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            if not resolved.exists():
                raise FileNotFoundError(f"Brakuje modelu ligi self-play: {resolved}")
            seen.add(resolved)
            unique.append(resolved)
        if not unique:
            raise ValueError("Faza 2 wymaga co najmniej jednego modelu historycznego")

        if self.self_play_runtime_config["league_retention"] == "anchored_rotation":
            # Pierwszy model jest stałą kotwicą fazy 1. Gdy manifest jest
            # większy od limitu, zachowujemy kotwicę i najnowszą resztę ligi.
            self.self_play_pool_paths = [
                unique[0],
                *unique[-(self.max_historical_models - 1) :],
            ]
            self.self_play_pool_paths = list(
                dict.fromkeys(self.self_play_pool_paths)
            )
        else:
            self.self_play_pool_paths = unique[-self.max_historical_models :]
        for path in self.self_play_pool_paths:
            self._broadcast_historical_checkpoint(path)
        self._write_league_manifest()
        print(
            "Liga self-play: latest + "
            f"{len(self.self_play_pool_paths)} zamrożonych checkpointów."
        )

    def _add_self_play_snapshot(self, path: Path, policy, env_step: int) -> None:
        """Dodaj oceniony checkpoint do ligi i odśwież latest_self."""
        if self.training_phase != 2:
            return
        self._broadcast_latest_opponent(policy)
        if not path.exists():
            raise FileNotFoundError(f"Nie zapisano checkpointu ligi: {path}")
        resolved = path.resolve()
        if resolved not in self.self_play_pool_paths:
            if len(self.self_play_pool_paths) < self.max_historical_models:
                self._broadcast_historical_checkpoint(resolved)
                self.self_play_pool_paths.append(resolved)
            elif self.self_play_runtime_config["league_retention"] == "anchored_rotation":
                # Slot 0 nigdy nie jest zastępowany. Pozostałe miejsca rotują
                # deterministycznie, dzięki czemu wszystkie procesy mają
                # identyczny skład ligi, a model bazowy zapobiega zapominaniu.
                replaceable = self.max_historical_models - 1
                replacement_index = 1 + (
                    env_step // self.history_interval_decisions
                ) % replaceable
                self._replace_historical_checkpoint(replacement_index, resolved)
                self.self_play_pool_paths[replacement_index] = resolved
            else:
                self._broadcast_historical_checkpoint(resolved)
                self.self_play_pool_paths.append(resolved)
                self.self_play_pool_paths = self.self_play_pool_paths[
                    -self.max_historical_models :
                ]
            self._write_league_manifest()

    def setup_and_train(self):
        # Konfiguracja Ucznia
        net_learner = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)

        policy_learner = DiscreteQLearningPolicy(
            model=net_learner,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=config.DQN_EPS_MAX,
            eps_inference=0.0
        )

        if self.training_phase == 2 and self.resume_path is None:
            if self.base_model_path is None:
                raise ValueError("Nowa faza 2 wymaga checkpointu --base-model")
            base_state = torch.load(
                self.base_model_path,
                map_location=self.device,
                weights_only=True,
            )
            if "algorithm_state" in base_state:
                raise ValueError(
                    "--base-model ma wskazywać lekki checkpoint wag step_*.pth, "
                    "a nie pełny training_state"
                )
            policy_learner.load_state_dict(base_state)
            print(f"Wczytano model fazy 1: {self.base_model_path}")
        # Checkpoint zawierający pełny stan może bezpiecznie wznowić tylko nowy
        # trening jednoagentowy. Stary stan algorytmu powstał z przejść między
        # różnymi graczami, więc nie wolno kontynuować go pod nowym licznikiem.
        # Same wagi nadal można jawnie wczytać przez --resume i --start-step,
        # choć do czystego eksperymentu zalecany jest start od zera.
        resume_payload = None
        if self.resume_path is not None:
            resume_payload = torch.load(
                self.resume_path,
                map_location=self.device,
                weights_only=True,
            )
            if "algorithm_state" not in resume_payload:
                if self.requested_start_step is None:
                    raise ValueError(
                        "Przy wznowieniu ze starego pliku wag podaj --start-step."
                    )
                policy_learner.load_state_dict(resume_payload)

        dqn_learner = DiagnosticDQN(
            policy=policy_learner,
            optim=AdamOptimizerFactory(
                lr=(
                    self.self_play_runtime_config["learning_rate_initial"]
                    if self.training_phase == 2
                    else config.DQN_LEARNING_RATE
                )
            ),
            gamma=config.DQN_GAMMA, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE,
            huber_loss_delta=config.DQN_HUBER_LOSS_DELTA,
        )

        if resume_payload is not None:
            if "algorithm_state" in resume_payload:
                if resume_payload.get("step_unit") != "learner_decisions":
                    raise ValueError(
                        "Ten stan treningu pochodzi ze starego kolektora "
                        "wieloagentowego. Rozpocznij poprawiony trening od zera."
                    )
                if resume_payload["observation_size"] != config.OBSERVATION_SIZE:
                    raise ValueError("Checkpoint ma niezgodny rozmiar obserwacji.")
                if resume_payload["action_space"] != config.ACTION_SPACE:
                    raise ValueError("Checkpoint ma niezgodny rozmiar akcji.")
                dqn_learner.load_state_dict(resume_payload["algorithm_state"])
                dqn_learner._iter = resume_payload["algorithm_iteration"]
                self.starting_step = int(resume_payload["completed_env_steps"])
                self.best_validation_score = float(
                    resume_payload.get("best_validation_score", float("-inf"))
                )
                saved_phase = int(resume_payload.get("training_phase", 1))
                if saved_phase != int(self.training_phase):
                    raise ValueError(
                        f"Stan pochodzi z fazy {saved_phase}, a uruchomiono fazę "
                        f"{self.training_phase}."
                    )
                if resume_payload.get("epsilon_schedule") != "exponential":
                    raise ValueError(
                        "Checkpoint korzystał ze starego harmonogramu epsilon. "
                        "Nowy trening wykładniczy rozpocznij od zera."
                    )
                saved_tau_steps = int(resume_payload["epsilon_tau_steps"])
                if saved_tau_steps != self.epsilon_tau_steps:
                    raise ValueError(
                        "Checkpoint korzystał z innego tau epsilon: "
                        f"{saved_tau_steps:,}, obecnie "
                        f"{self.epsilon_tau_steps:,}."
                    )
                expected_epsilon_min = (
                    config.DQN_PHASE2_EPS_MIN
                    if self.training_phase == 2
                    else config.DQN_RAND_PHASE_EPS_MIN
                )
                if float(resume_payload.get("epsilon_min", -1.0)) != float(
                    expected_epsilon_min
                ):
                    raise ValueError(
                        "Checkpoint korzystał z innego minimalnego epsilona. "
                        "Nowy harmonogram rozpocznij od zera."
                    )
                expected_lr_schedule = (
                    self.self_play_runtime_config["learning_rate_schedule"]
                    if self.training_phase == 2
                    else "phase1_piecewise_v1"
                )
                if resume_payload.get("learning_rate_schedule") != expected_lr_schedule:
                    raise ValueError(
                        "Checkpoint korzystał z innego harmonogramu learning rate."
                    )
                expected_update_schedule = (
                    self.self_play_runtime_config["update_ratio_schedule"]
                    if self.training_phase == 2
                    else "phase1_piecewise_v1"
                )
                if resume_payload.get("update_ratio_schedule") != expected_update_schedule:
                    raise ValueError(
                        "Checkpoint korzystał z innego harmonogramu update ratio."
                    )
                if int(resume_payload.get("replay_buffer_size", -1)) != int(
                    self.replay_buffer_size
                ):
                    raise ValueError(
                        "Checkpoint korzystał z replay buffera o innym rozmiarze."
                    )
                saved_reward_weight = float(
                    resume_payload.get("placement_reward_weight", 1.0)
                )
                if saved_reward_weight != self.placement_reward_weight:
                    raise ValueError(
                        "Checkpoint korzystał z innej wagi nagrody za miejsce: "
                        f"{saved_reward_weight}, obecnie "
                        f"{self.placement_reward_weight}."
                    )
                saved_huber_delta = resume_payload.get("huber_loss_delta")
                if saved_huber_delta != config.DQN_HUBER_LOSS_DELTA:
                    raise ValueError(
                        "Checkpoint korzystał z innej funkcji straty Huber. "
                        "Nowy eksperyment rozpocznij od zera zamiast mieszać "
                        "stan optymalizatora z innym celem."
                    )
                if self.training_phase == 2:
                    saved_runtime_config = resume_payload.get("runtime_config")
                    if saved_runtime_config is None:
                        # Stany zapisane przed wprowadzeniem profili należą do
                        # starego wariantu standardowego. Nie wolno wznowić ich
                        # jako eksperymentu ppo_style.
                        if self.self_play_runtime_config["name"] != "standard":
                            raise ValueError(
                                "Stary checkpoint nie zawiera profilu self-play. "
                                "Eksperyment ppo_style rozpocznij od modelu fazy 1."
                            )
                    elif saved_runtime_config != self.self_play_runtime_config:
                        raise ValueError(
                            "Checkpoint korzystał z innego profilu self-play."
                        )
                    self.historical_model_paths.extend(
                        Path(path)
                        for path in resume_payload.get("self_play_pool_paths", [])
                    )
            else:
                self.starting_step = int(self.requested_start_step)

            # Następna ewaluacja przypada na pierwszy pełny próg po kroku, z
            # którego kontynuujemy; nie powtarzamy raportu dla starego modelu.
            interval = self.evaluation_interval_steps
            self.next_evaluation_step = (
                self.starting_step // interval + 1
            ) * interval
            print(
                f"Wznowiono DQN od kroku {self.starting_step:,} z "
                f"'{self.resume_path}'."
            )

        # Przeciwnicy są teraz częścią DQNLearnerEnv. Replay buffer widzi tylko
        # decyzje player_0, więc nie potrzebujemy MultiAgentOffPolicyAlgorithm
        # ani sztucznych algorytmów reprezentujących boty.
        if self.training_phase not in {1, 2}:
            raise NotImplementedError(f"Nieobsługiwana faza DQN: {self.training_phase}")

        if self.training_phase == 2:
            self._initialize_self_play_league(dqn_learner.policy)

        # Kolektory
        buffer = VectorReplayBuffer(self.replay_buffer_size, len(self.train_envs))
        train_collector = Collector(
            dqn_learner,
            self.train_envs, 
            buffer, 
            exploration_noise=True,
        )

        test_collector = Collector(
            dqn_learner,
            self.test_envs, 
            exploration_noise=False,
        )

        print("Zapełnianie bufora pierwszymi legalnymi decyzjami ucznia...")
        # Replay buffer nie jest częścią checkpointu, aby pojedynczy zapis nie
        # ważył setek MB. Świeży warm-up po wznowieniu zapewnia różnorodne dane
        # przed pierwszą kolejną aktualizacją sieci.
        # ``random=True`` w Collectorze losuje z całej przestrzeni Discrete i
        # ignoruje maskę pokera. Zamiast tego ustawiamy epsilon=1: polityka DQN
        # losuje wtedy w 100%, ale wyłącznie spośród legalnych ruchów.
        warmup_epsilon = (
            config.DQN_PHASE2_EPS_MAX if self.training_phase == 2 else 1.0
        )
        warmup_steps = (
            config.DQN_PHASE2_BUFFER_WARMUP
            if self.training_phase == 2
            else config.DQN_BUFFER_WARMUP
        )
        dqn_learner.policy.set_eps_training(warmup_epsilon)
        # Warm-up odbywa się jeszcze przed wejściem trenera w kontekst
        # ``training_step`` Tianshou. Dlatego na czas tej jednej kolekcji
        # ustawiamy również epsilon inferencyjny, po czym natychmiast wracamy
        # do deterministycznej ewaluacji.
        dqn_learner.policy.set_eps_inference(warmup_epsilon)
        train_collector.collect(
            n_step=warmup_steps,
            random=False,
            reset_before_collect=True,
        )
        dqn_learner.policy.set_eps_inference(0.0)

        # Każdy nazwany eksperyment ma osobny katalog zdarzeń. TensorBoard
        # zapisuje loss i statystyki Tianshou, a poniżej dokładamy epsilon,
        # rozmiar bufora oraz pokerowe wyniki walidacji.
        tensorboard_dir = tensorboard_run_dir("dqn", self.run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        self.tensorboard_writer = writer
        logger = TensorboardLogger(
            writer,
            # Kolektor zwiększa env_step o 1000, więc zapisujemy każdą serię.
            training_interval=config.DQN_COLLECTION_STEPS,
            # W tej wersji Tianshou update_step oznacza serię około 250
            # gradientów, a nie pojedynczy gradient. Interwał 1 daje jeden
            # czytelny punkt diagnostyczny na każde 1000 nowych decyzji.
            update_interval=config.DQN_TENSORBOARD_UPDATE_INTERVAL,
        )

        # Funkcje trenujące z logiką DQN (Epsilon Decay)
        # eps definiuje jak często podejmowane są losowe decyzje
        # TODO: można tu coś pokombinować, ale raczej jest git
        def train_fn(epoch, env_step):
            global_step = self.starting_step + env_step
            if self.training_phase == 1:
                eps = phase_one_epsilon(global_step, self.epsilon_tau_steps)
                learning_rate = phase_one_learning_rate(global_step)
                update_ratio = phase_one_update_ratio(global_step)
            else:
                eps = phase_two_epsilon(global_step, self.epsilon_tau_steps)
                learning_rate = phase_two_learning_rate(
                    global_step,
                    initial=self.self_play_runtime_config["learning_rate_initial"],
                    final=self.self_play_runtime_config["learning_rate_final"],
                    boundary=self.self_play_runtime_config["learning_rate_boundary"],
                )
                update_ratio = self.self_play_runtime_config["update_ratio"]
            dqn_learner.policy.set_eps_training(eps)
            set_optimizer_learning_rate(dqn_learner.optim, learning_rate)
            trainer_params.update_step_num_gradient_steps_per_sample = update_ratio
            writer.add_scalar("training/epsilon", eps, global_step=global_step)
            writer.add_scalar(
                "training/learning_rate",
                learning_rate,
                global_step=global_step,
            )
            writer.add_scalar(
                "training/update_ratio",
                update_ratio,
                global_step=global_step,
            )
            writer.add_scalar(
                "training/replay_buffer_size",
                len(buffer),
                global_step=global_step,
            )

            # Callback jest wykonywany na granicy epok. Kontrola po poprzedniej
            # serii aktualizacji zatrzyma proces, zanim NaN uszkodzi kolejne
            # checkpointy albo cały replay buffer.
            ensure_finite_model(net_learner, global_step)

            # Stan do wznowienia zapisujemy przed czasochłonną ewaluacją. Jeśli
            # komputer zostanie wyłączony w jej trakcie, nie tracimy ostatnich
            # pełnego interwału decyzji treningowych.
            evaluation_due = global_step >= self.next_evaluation_step
            if evaluation_due:
                save_training_state(
                    dqn_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau_steps=self.epsilon_tau_steps,
                    placement_reward_weight=self.placement_reward_weight,
                    training_phase=self.training_phase,
                    self_play_pool_paths=tuple(
                        str(path) for path in self.self_play_pool_paths
                    ),
                    runtime_config=(
                        self.self_play_runtime_config
                        if self.training_phase == 2
                        else None
                    ),
                )

            self.run_periodic_evaluation(
                env_step=global_step,
                learner_policy=dqn_learner.policy,
            )

            if self.training_phase == 2:
                # `latest_self` zmieniamy na granicy ewaluacji. Jest to wolny,
                # przewidywalny przeciwnik, a nie kopia aktualizowana po każdym
                # minibatchu, która czyniłaby środowisko skrajnie niestacjonarnym.
                if evaluation_due:
                    self._broadcast_latest_opponent(dqn_learner.policy)

                # Historia ligi ma własny interwał, niezależny od ewaluacji.
                # W profilu ppo_style 100 tys. i 250 tys. mają wspólny próg
                # dopiero co 500 tys.; wiązanie snapshotu z evaluation_due
                # nieświadomie pięciokrotnie zmniejszałoby różnorodność ligi.
                if (
                    global_step > 0
                    and global_step % self.history_interval_decisions == 0
                    and global_step != self._last_history_step
                ):
                    snapshot_path = (
                        self.checkpoint_dir / f"step_{global_step:09d}.pth"
                    )
                    if not snapshot_path.exists():
                        torch.save(dqn_learner.policy.state_dict(), snapshot_path)
                    self._add_self_play_snapshot(
                        snapshot_path,
                        dqn_learner.policy,
                        global_step,
                    )
                    self._last_history_step = global_step

            # Po udanej ewaluacji zapisujemy stan ponownie, aby zawierał także
            # nowy najlepszy wynik. Pierwszy zapis powyżej pozostaje awaryjną
            # kopią na wypadek przerwania samej ewaluacji.
            if evaluation_due:
                save_training_state(
                    dqn_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau_steps=self.epsilon_tau_steps,
                    placement_reward_weight=self.placement_reward_weight,
                    training_phase=self.training_phase,
                    self_play_pool_paths=tuple(
                        str(path) for path in self.self_play_pool_paths
                    ),
                    runtime_config=(
                        self.self_play_runtime_config
                        if self.training_phase == 2
                        else None
                    ),
                )

            # Pełny, nienadpisywany stan zapisujemy razem z walidacją. Pozwala
            # to wznowić wybrany punkt wraz z target network i optymalizatorem,
            # bez tworzenia dziesiątek nieocenionych, ciężkich plików.
            if (
                global_step > 0
                and global_step % config.DQN_FULL_STATE_INTERVAL_DECISIONS == 0
            ):
                save_training_state(
                    dqn_learner,
                    global_step,
                    self.checkpoint_dir
                    / f"training_state_step_{global_step:09d}.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau_steps=self.epsilon_tau_steps,
                    placement_reward_weight=self.placement_reward_weight,
                    training_phase=self.training_phase,
                    self_play_pool_paths=tuple(
                        str(path) for path in self.self_play_pool_paths
                    ),
                    runtime_config=(
                        self.self_play_runtime_config
                        if self.training_phase == 2
                        else None
                    ),
                )

        def test_fn(epoch, env_step):
            dqn_learner.policy.set_eps_inference(0.0)

        # Trener
        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=config.DQN_COLLECTION_STEPS,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            batch_size=config.DQN_BATCH_SIZE,
            # Tianshou mnoży tę wartość przez liczbę właśnie zebranych
            # przejść. Dla kolekcji 1000 decyzji ratio 0.25 daje około 250
            # aktualizacji, każdą na losowym batchu z replay buffera.
            update_step_num_gradient_steps_per_sample=(
                self.self_play_runtime_config["update_ratio"]
                if self.training_phase == 2
                else phase_one_update_ratio(self.starting_step)
            ),
            training_fn=train_fn,
            test_fn=test_fn,
            logger=logger,
            # Pasek postępu dla każdej serii aktualizacji tworzyłby podczas
            # wielogodzinnego runu ogromny log znaków sterujących terminala.
            # Metryki nadal trafiają do TensorBoard, a podsumowania epok są
            # nadal wypisywane w terminalu.
            show_progress=False,
        )

        # Pomiar przed treningiem daje uczciwy punkt odniesienia dla wszystkich
        # późniejszych checkpointów tej samej, losowo zainicjalizowanej sieci.
        if self.starting_step == 0:
            self.run_initial_evaluation(learner_policy=dqn_learner.policy)
        else:
            # Pliki pod stałymi nazwami (`final.pth`, `best.pth`) zostaną pod
            # koniec nadpisane. Najpierw zachowujemy komplet poprzedniego runu
            # w archiwum, tak samo jak przy rozpoczęciu nowego treningu.
            self._preserve_existing_checkpoints()
        print("Rozpoczęcie treningu DQN...")
        runtime_trainer = OffPolicyTrainer(
            algorithm=dqn_learner,
            params=trainer_params,
        )
        try:
            result = runtime_trainer.run()
        except KeyboardInterrupt:
            # Ctrl+C ma być bezpiecznym „przyciskiem Stop”. Nie uruchamiamy tu
            # długiej ewaluacji końcowej: zapisujemy aktualne wagi i pełny stan,
            # zamykamy logi, po czym użytkownik od razu odzyskuje komputer.
            interrupted_step = self.starting_step + runtime_trainer._env_step
            ensure_finite_model(net_learner, interrupted_step)
            interrupted_path = self.checkpoint_dir / "interrupted.pth"
            torch.save(dqn_learner.policy.state_dict(), interrupted_path)
            torch.save(
                dqn_learner.policy.state_dict(),
                self.checkpoint_dir / "latest.pth",
            )
            save_training_state(
                dqn_learner,
                interrupted_step,
                self.checkpoint_dir / "training_state_latest.pth",
                best_validation_score=self.best_validation_score,
                epsilon_tau_steps=self.epsilon_tau_steps,
                placement_reward_weight=self.placement_reward_weight,
                training_phase=self.training_phase,
                self_play_pool_paths=tuple(
                    str(path) for path in self.self_play_pool_paths
                ),
                runtime_config=(
                    self.self_play_runtime_config
                    if self.training_phase == 2
                    else None
                ),
            )
            writer.flush()
            writer.close()
            print(
                "\n[STOP] Trening zatrzymany bezpiecznie po "
                f"{interrupted_step:,} decyzjach DQN."
            )
            print(f"[ZAPIS] Stan do wznowienia: {self.checkpoint_dir / 'training_state_latest.pth'}")
            return
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        # Callback treningowy działa przed kolekcją, dlatego ostatni próg
        # planowanej liczby decyzji obsługujemy jawnie po zwróceniu statystyk.
        # Zapewnia to checkpoint i walidację również na końcu fazy.
        final_step = self.starting_step + result.train_step
        ensure_finite_model(net_learner, final_step)
        self.run_periodic_evaluation(
            env_step=final_step,
            learner_policy=dqn_learner.policy,
        )
        self.run_final_evaluation(
            learner_policy=dqn_learner.policy,
            env_step=final_step,
        )
        save_training_state(
            dqn_learner,
            final_step,
            self.checkpoint_dir / "training_state_final.pth",
            best_validation_score=self.best_validation_score,
            epsilon_tau_steps=self.epsilon_tau_steps,
            placement_reward_weight=self.placement_reward_weight,
            training_phase=self.training_phase,
            self_play_pool_paths=tuple(
                str(path) for path in self.self_play_pool_paths
            ),
            runtime_config=(
                self.self_play_runtime_config
                if self.training_phase == 2
                else None
            ),
        )
        writer.flush()
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening DQN: faza 1 lub self-play")
    parser.add_argument(
        "--phase",
        type=int,
        choices=(1, 2),
        default=1,
        help="1 = boty bazowe, 2 = liga self-play.",
    )
    parser.add_argument(
        "--self-play-profile",
        choices=("standard", "ppo_style"),
        default="standard",
        help=(
            "Profil fazy 2. ppo_style używa 70%% self-play, większej ligi, "
            "częstszych snapshotów i parametrów DQN dopasowanych do tego trybu."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Stan treningu albo starszy plik wag .pth, od którego kontynuować.",
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Nazwa izolowanego katalogu checkpointów i TensorBoard. "
            "Domyślnie tworzona z daty i czasu."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11_001,
        help="Seed inicjalizacji sieci, eksploracji i środowisk.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        help=(
            "Liczba wykonanych decyzji ucznia; wymagana tylko dla pliku "
            "zawierającego same wagi."
        ),
    )
    parser.add_argument(
        "--actions",
        "--decisions",
        dest="actions",
        type=int,
        default=config.DQN_MAX_EPOCHS * config.DQN_STEPS_PER_EPOCH,
        help="Liczba nowych decyzji DQN (domyślnie 250 000).",
    )
    parser.add_argument(
        "--epsilon-tau-decisions",
        type=int,
        default=None,
        help=(
            "Stała czasowa tau wykładniczego wygaszania epsilon. "
            "Domyślna wartość zależy od fazy."
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=config.DQN_EVAL_INTERVAL_DECISIONS,
        help="Odstęp pomiędzy pełnymi walidacjami, liczony w decyzjach DQN.",
    )
    parser.add_argument(
        "--placement-reward-weight",
        type=float,
        default=None,
        help=(
            "Waga terminalnej nagrody za miejsce. 0.0 oznacza czystą nagrodę "
            "żetonową; 1.0 odtwarza wcześniejszy sygnał −3/−1/+1/+3."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Checkpoint wag fazy 1, od którego rozpocznie się faza 2.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Zamrożony checkpoint dodany do początkowej ligi. Opcję można "
            "powtórzyć; bez niej liga zaczyna od --base-model."
        ),
    )
    args = parser.parse_args()
    if args.actions <= 0 or args.actions % config.DQN_STEPS_PER_EPOCH != 0:
        parser.error(
            f"--decisions musi być dodatnią wielokrotnością "
            f"{config.DQN_STEPS_PER_EPOCH:,}."
        )
    if args.start_step is not None and args.resume is None:
        parser.error("--start-step ma sens tylko razem z --resume.")
    if args.epsilon_tau_decisions is None:
        args.epsilon_tau_decisions = (
            config.DQN_PHASE2_EPS_TAU_STEPS
            if args.phase == 2
            else config.DQN_PHASE1_EPS_TAU_STEPS
        )
    if args.placement_reward_weight is None:
        args.placement_reward_weight = (
            config.DQN_PHASE2_PLACEMENT_REWARD_WEIGHT
            if args.phase == 2
            else config.DQN_PHASE1_PLACEMENT_REWARD_WEIGHT
        )
    if args.phase == 2 and args.resume is None and args.base_model is None:
        parser.error("Nowa faza 2 wymaga --base-model.")
    if args.phase == 1 and (args.base_model or args.opponent_checkpoint):
        parser.error("--base-model i --opponent-checkpoint są przeznaczone dla fazy 2.")
    if args.phase == 1 and args.self_play_profile != "standard":
        parser.error("--self-play-profile jest przeznaczony wyłącznie dla fazy 2.")
    if args.epsilon_tau_decisions <= 0:
        parser.error("--epsilon-tau-decisions musi być dodatnie.")
    if args.evaluation_interval <= 0:
        parser.error("--evaluation-interval musi być dodatni.")
    if args.placement_reward_weight < 0.0:
        parser.error("--placement-reward-weight nie może być ujemne.")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error(
            "--run-name może zawierać tylko litery, cyfry, '-' i '_' "
            "(maksymalnie 64 znaki)."
        )
    return args

if __name__ == "__main__":
    args = parse_args()
    self_play_config = phase_two_profile(args.self_play_profile)
    if args.phase == 1:
        validate_phase_one_configuration()
    elif not np.isclose(sum(self_play_config["opponent_weights"].values()), 1.0):
        raise ValueError("Wagi przeciwników fazy 2 muszą sumować się do 1.0")
    configure_cpu_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = dqn_run_dir(args.run_name)
    train_env_factories = [
        partial(
            make_learner_env,
            initial_seed=args.seed + worker_index * 1_000_000,
            placement_reward_weight=args.placement_reward_weight,
            training_phase=args.phase,
            self_play_policy_kind="dqn",
            opponent_weights=(
                self_play_config["opponent_weights"]
                if args.phase == 2
                else config.DQN_PHASE1_OPPONENT_WEIGHTS
            ),
            self_play_opponent_epsilon=(
                self_play_config["opponent_epsilon"] if args.phase == 2 else 0.0
            ),
            max_historical_models=self_play_config["max_historical_models"],
        )
        for worker_index in range(config.DQN_NUM_TRAIN_ENVS)
    ]
    test_env_factories = [
        partial(
            make_learner_env,
            initial_seed=args.seed + 100_000_000 + worker_index * 1_000_000,
            placement_reward_weight=args.placement_reward_weight,
            training_phase=args.phase,
            self_play_policy_kind="dqn",
            opponent_weights=(
                self_play_config["opponent_weights"]
                if args.phase == 2
                else config.DQN_PHASE1_OPPONENT_WEIGHTS
            ),
            self_play_opponent_epsilon=(
                self_play_config["opponent_epsilon"] if args.phase == 2 else 0.0
            ),
            max_historical_models=self_play_config["max_historical_models"],
        )
        for worker_index in range(config.DQN_NUM_TEST_ENVS)
    ]
    active_buffer_size = (
        self_play_config["replay_buffer_size"]
        if args.phase == 2
        else config.DQN_BUFFER_SIZE
    )
    active_learning_rate = (
        f"{self_play_config['learning_rate_initial']:g}→"
        f"{self_play_config['learning_rate_final']:g}"
        if args.phase == 2
        else "1e-4→5e-5→2e-5"
    )
    active_update_ratio = (
        f"{self_play_config['update_ratio']:g}"
        if args.phase == 2
        else "0.25→0.15→0.10"
    )
    print(
        f"Konfiguracja fazy {args.phase}: "
        f"run '{args.run_name}', seed {args.seed}, "
        f"{args.actions:,} decyzji DQN, "
        f"profil {self_play_config['name'] if args.phase == 2 else 'phase1'}, "
        f"bufor {active_buffer_size:,}, warm-up "
        f"{config.DQN_PHASE2_BUFFER_WARMUP if args.phase == 2 else config.DQN_BUFFER_WARMUP:,}, "
        f"epsilon wykładniczy "
        f"{config.DQN_PHASE2_EPS_MAX if args.phase == 2 else config.DQN_EPS_MAX:g}→"
        f"{config.DQN_PHASE2_EPS_MIN if args.phase == 2 else config.DQN_RAND_PHASE_EPS_MIN:g}, tau "
        f"{args.epsilon_tau_decisions:,} decyzji, "
        f"ewaluacja co {args.evaluation_interval:,}, "
        f"waga nagrody za miejsce {args.placement_reward_weight:g}, "
        f"Huber δ={config.DQN_HUBER_LOSS_DELTA:g}, "
        f"learning rate {active_learning_rate}, "
        f"update ratio {active_update_ratio}, "
        f"{config.DQN_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    with single_training_process(checkpoint_dir):
        trainer = DQNPokerTrainer(
            algo_name="dqn",
            training_phase=args.phase,
            evaluator_class=DQNEvaluator,
            num_train_envs=config.DQN_NUM_TRAIN_ENVS,
            num_test_envs=config.DQN_NUM_TEST_ENVS,
            max_epochs=args.actions // config.DQN_STEPS_PER_EPOCH,
            steps_per_epoch=config.DQN_STEPS_PER_EPOCH,
            env_factory=partial(
                make_learner_env,
                placement_reward_weight=args.placement_reward_weight,
                training_phase=args.phase,
                self_play_policy_kind="dqn",
                opponent_weights=(
                    self_play_config["opponent_weights"]
                    if args.phase == 2
                    else config.DQN_PHASE1_OPPONENT_WEIGHTS
                ),
                self_play_opponent_epsilon=(
                    self_play_config["opponent_epsilon"]
                    if args.phase == 2
                    else 0.0
                ),
                max_historical_models=self_play_config["max_historical_models"],
            ),
            evaluation_interval_steps=args.evaluation_interval,
            checkpoint_dir=checkpoint_dir,
            train_env_factories=train_env_factories,
            test_env_factories=test_env_factories,
            resume_path=args.resume,
            start_step=args.start_step,
            run_name=args.run_name,
            epsilon_tau_steps=args.epsilon_tau_decisions,
            placement_reward_weight=args.placement_reward_weight,
            base_model_path=args.base_model,
            historical_model_paths=tuple(args.opponent_checkpoint),
            self_play_runtime_config=self_play_config,
            baseline_model_path=args.base_model if args.phase == 2 else None,
        )
        trainer.setup_and_train()
