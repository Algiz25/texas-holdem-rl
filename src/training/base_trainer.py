"""Wspólna infrastruktura treningu oraz harmonogram rzetelnej ewaluacji."""
from __future__ import annotations
import os
from contextlib import contextmanager
from pathlib import Path

import shutil
from datetime import datetime

import torch
from tianshou.env import PettingZooEnv, SubprocVectorEnv

import config
from environment import TexasHoldemTournament
from paths import DQN_CHECKPOINT_DIR, PPO_CHECKPOINT_DIR, ensure_output_directories

def configure_cpu_runtime() -> None:
    """Zapobiega przeciążeniu procesora przy wielu środowiskach."""
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
            f"Niestabilny model po {env_step:,} decyzjach ucznia; "
            f"niepoprawne parametry: {names}"
        )

@contextmanager
def single_training_process(checkpoint_dir: Path):
    """Nie pozwól przypadkowo uruchomić dwóch treningów w tym samym katalogu."""
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
                "Inny trening już działa w tym katalogu. Nie uruchamiaj drugiego procesu."
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

def save_training_state(
    algorithm,
    env_step: int,
    path: Path,
    *,
    best_validation_score: float,
    **kwargs
) -> None:
    """Zapisz stan algorytmu potrzebny do bezpiecznej kontynuacji treningu."""
    payload = {
        "format_version": 3,
        "step_unit": "learner_decisions",
        "algorithm_state": algorithm.state_dict(),
        "completed_env_steps": env_step,
        "best_validation_score": best_validation_score,
        "observation_size": config.OBSERVATION_SIZE,
        "action_space": config.ACTION_SPACE,
    }
    
    # Zapisz iterację algorytmu, jeśli jest dostępna (np. w OffPolicyAlgorithm)
    if hasattr(algorithm, "_iter"):
        payload["algorithm_iteration"] = algorithm._iter
        
    # Dodaj wszelkie specyficzne dla algorytmu parametry (np. epsilon)
    payload.update(kwargs)
    
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def make_poker_env():
    return PettingZooEnv(
        TexasHoldemTournament(
            num_players=config.NUM_PLAYERS,
            starting_chips=config.STARTING_CHIPS,
        )
    )


class BasePokerTrainer:
    def __init__(
        self,
        algo_name,
        training_phase,
        evaluator_class,
        num_train_envs,
        num_test_envs,
        max_epochs,
        steps_per_epoch,
        env_factory=make_poker_env,
        evaluation_interval_steps=config.EVAL_INTERVAL_STEPS,
        checkpoint_dir=None,
        train_env_factories=None,
        test_env_factories=None,
    ):
        self.algo_name = algo_name
        self.training_phase = training_phase
        self.evaluator_class = evaluator_class
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = max_epochs * steps_per_epoch
        self.observation_size = config.OBSERVATION_SIZE

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        default_checkpoint_dir = (
            DQN_CHECKPOINT_DIR if algo_name == "dqn" else PPO_CHECKPOINT_DIR
        )
        self.checkpoint_dir = checkpoint_dir or default_checkpoint_dir
        self.evaluation_report_dir = self.checkpoint_dir / "evaluations"
        # PPO nadal korzysta z wieloagentowego środowiska PettingZoo. DQN może
        # przekazać jednoagentową fabrykę i własny interwał liczony w decyzjach
        # ucznia, bez duplikowania wspólnej obsługi checkpointów i ewaluacji.
        self.evaluation_interval_steps = evaluation_interval_steps
        self.next_evaluation_step = evaluation_interval_steps
        self.best_validation_score = float("-inf")
        self._existing_checkpoints_preserved = False
        ensure_output_directories()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_report_dir.mkdir(parents=True, exist_ok=True)
        # Trener DQN może podłączyć writer po utworzeniu nazwanego runu.
        # Klasa bazowa pozostaje niezależna od TensorBoard i PPO działa bez zmian.
        self.tensorboard_writer = None

        print(f"Inicjalizacja środowisk dla {self.algo_name.upper()}...")
        self.env = env_factory()
        train_factories = train_env_factories or [
            env_factory for _ in range(num_train_envs)
        ]
        test_factories = test_env_factories or [
            env_factory for _ in range(num_test_envs)
        ]
        if len(train_factories) != num_train_envs:
            raise ValueError("Liczba fabryk treningowych nie zgadza się z konfiguracją")
        if len(test_factories) != num_test_envs:
            raise ValueError("Liczba fabryk testowych nie zgadza się z konfiguracją")
        self.train_envs = SubprocVectorEnv(train_factories)
        self.test_envs = SubprocVectorEnv(test_factories)

    @property
    def phase_name(self) -> str:
        """Jednolita nazwa działa zarówno dla fazy `1`, jak i późniejszych nazw."""
        return str(self.training_phase).lower()

    def run_periodic_evaluation(
        self,
        *,
        env_step: int,
        learner_policy,
        opponent_policy=None,
    ) -> None:
        """Po skonfigurowanym interwale oceń model na stałych rozdaniach."""
        if env_step < self.next_evaluation_step:
            return

        checkpoint_path = self.checkpoint_dir / f"step_{env_step:09d}.pth"
        torch.save(learner_policy.state_dict(), checkpoint_path)
        shutil.copy2(checkpoint_path, self.checkpoint_dir / "latest.pth")

        evaluator = self.evaluator_class(
            num_tournaments=config.EVAL_TOURNAMENTS_PER_SUITE,
            model_path=checkpoint_path,
            training_phase=self.training_phase,
            report_dir=self.evaluation_report_dir,
        )
        results = evaluator.evaluate(
            stage="validation",
            step=env_step,
            seed_base=config.EVAL_VALIDATION_SEED,
        )
        # Mieszanka fazy 1 jest głównym środowiskiem walidacyjnym. bb/100
        # wykorzystuje wszystkie rozdania, więc jest stabilniejsze od samego
        # procentu wygranych turniejów.
        phase_result = results.get("phase1_mix")
        if phase_result and phase_result.bb_per_100 > self.best_validation_score:
            self.best_validation_score = phase_result.bb_per_100
            shutil.copy2(checkpoint_path, self.checkpoint_dir / "best.pth")
            print(
                f"[ZAPIS] Nowy najlepszy model: "
                f"{phase_result.bb_per_100:.2f} bb/100."
            )

        self._log_evaluation_to_tensorboard(results, "validation", env_step)

        # Mechanizm jest gotowy na późniejsze fazy self-play. Faza liczbowa nie
        # wywołuje już błędu `.lower()`, który wcześniej zatrzymywał trening.
        if self.phase_name in {"self", "advanced"} and opponent_policy is not None:
            opponent_policy.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )

        # Jeżeli callback został wywołany po przekroczeniu progu, przechodzimy
        # do pierwszego przyszłego punktu zamiast powtarzać tę samą ewaluację.
        while self.next_evaluation_step <= env_step:
            self.next_evaluation_step += self.evaluation_interval_steps

    def run_initial_evaluation(self, *, learner_policy) -> None:
        """Zapisz punkt odniesienia przed wykonaniem pierwszej aktualizacji sieci."""
        self._preserve_existing_checkpoints()
        checkpoint_path = self.checkpoint_dir / "step_000000000.pth"
        torch.save(learner_policy.state_dict(), checkpoint_path)
        evaluator = self.evaluator_class(
            num_tournaments=config.EVAL_TOURNAMENTS_PER_SUITE,
            model_path=checkpoint_path,
            training_phase=self.training_phase,
            report_dir=self.evaluation_report_dir,
        )
        results = evaluator.evaluate(
            stage="baseline_untrained",
            step=0,
            seed_base=config.EVAL_VALIDATION_SEED,
        )
        # Losowy uczeń jest stałym punktem odniesienia niezależnym od
        # inicjalizacji sieci. Liczymy go raz, na identycznych rozdaniach.
        evaluator.evaluate(
            stage="baseline_random",
            step=0,
            seed_base=config.EVAL_VALIDATION_SEED,
            learner_mode="random",
        )
        self._log_evaluation_to_tensorboard(results, "baseline_untrained", 0)
        phase_result = results.get("phase1_mix")
        if phase_result:
            self.best_validation_score = phase_result.bb_per_100
            shutil.copy2(checkpoint_path, self.checkpoint_dir / "best.pth")
            shutil.copy2(checkpoint_path, self.checkpoint_dir / "latest.pth")

    def _preserve_existing_checkpoints(self) -> None:
        """Skopiuj modele z wcześniejszego treningu przed użyciem stałych nazw.

        `best.pth`, `latest.pth` i checkpointy krokowe są wygodne dla skryptów,
        ale kolejny trening używa tych samych nazw. Jednorazowa kopia do
        katalogu `archive` chroni poprzednie wagi bez wpływu na nowy przebieg.
        """
        if self._existing_checkpoints_preserved:
            return

        existing = sorted(self.checkpoint_dir.glob("*.pth"))
        if existing:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            archive_dir = self.checkpoint_dir / "archive" / timestamp
            archive_dir.mkdir(parents=True, exist_ok=False)
            for checkpoint in existing:
                shutil.copy2(checkpoint, archive_dir / checkpoint.name)
            print(
                f"[ARCHIWUM] Zachowano {len(existing)} poprzednich checkpointów "
                f"w '{archive_dir}'."
            )

        self._existing_checkpoints_preserved = True

    def run_final_evaluation(self, *, learner_policy, env_step: int) -> None:
        """Zapisz ostatni model i wykonaj duży test na nieużywanych seedach."""
        final_path = self.checkpoint_dir / "final.pth"
        torch.save(learner_policy.state_dict(), final_path)
        shutil.copy2(final_path, self.checkpoint_dir / "latest.pth")
        evaluator = self.evaluator_class(
            num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
            model_path=final_path,
            training_phase=self.training_phase,
            report_dir=self.evaluation_report_dir,
        )
        final_results = evaluator.evaluate(
            stage="final_last",
            step=env_step,
            seed_base=config.EVAL_FINAL_SEED,
        )
        self._log_evaluation_to_tensorboard(final_results, "final_last", env_step)

        # Najlepszy checkpoint walidacyjny może pochodzić ze środka treningu.
        # Oceniamy go na tych samych, nowych seedach co model końcowy, aby ich
        # porównanie nie zależało od szczęścia w rozdaniach.
        best_path = self.checkpoint_dir / "best.pth"
        if best_path.exists() and best_path != final_path:
            best_evaluator = self.evaluator_class(
                num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
                model_path=best_path,
                training_phase=self.training_phase,
                report_dir=self.evaluation_report_dir,
            )
            best_results = best_evaluator.evaluate(
                stage="final_best",
                step=env_step,
                seed_base=config.EVAL_FINAL_SEED,
            )
            self._log_evaluation_to_tensorboard(best_results, "final_best", env_step)

    def _log_evaluation_to_tensorboard(self, results, stage: str, step: int) -> None:
        """Zapisz najważniejsze metryki pokera obok lossu trenera."""
        if self.tensorboard_writer is None:
            return
        for suite, result in results.items():
            prefix = f"evaluation/{stage}/{suite}"
            self.tensorboard_writer.add_scalar(
                f"{prefix}/bb_per_100", result.bb_per_100, step
            )
            self.tensorboard_writer.add_scalar(
                f"{prefix}/mean_chip_delta_per_hand",
                result.mean_chip_delta_per_hand,
                step,
            )
            self.tensorboard_writer.add_scalar(
                f"{prefix}/vpip", result.vpip, step
            )
            self.tensorboard_writer.add_scalar(
                f"{prefix}/pfr", result.pfr, step
            )
        self.tensorboard_writer.flush()

    def setup_and_train(self):
        """Metoda abstrakcyjna implementowana przez trener DQN albo PPO."""
        raise NotImplementedError
