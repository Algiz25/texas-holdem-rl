"""Wspólna infrastruktura treningu oraz harmonogram rzetelnej ewaluacji."""

from __future__ import annotations

import shutil

import torch
from tianshou.env import PettingZooEnv, SubprocVectorEnv

import config
from environment import TexasHoldemTournament
from paths import DQN_CHECKPOINT_DIR, PPO_CHECKPOINT_DIR, ensure_output_directories


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
    ):
        self.algo_name = algo_name
        self.training_phase = training_phase
        self.evaluator_class = evaluator_class
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = max_epochs * steps_per_epoch
        self.observation_size = config.OBSERVATION_SIZE

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = (
            DQN_CHECKPOINT_DIR if algo_name == "dqn" else PPO_CHECKPOINT_DIR
        )
        self.next_evaluation_step = config.EVAL_INTERVAL_STEPS
        self.best_validation_score = float("-inf")
        ensure_output_directories()

        print(f"Inicjalizacja środowisk PettingZoo dla {self.algo_name.upper()}...")
        self.env = make_poker_env()
        self.train_envs = SubprocVectorEnv([make_poker_env for _ in range(num_train_envs)])
        self.test_envs = SubprocVectorEnv([make_poker_env for _ in range(num_test_envs)])

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
        """Co 100 tys. akcji oceń checkpoint na stałych zestawach rozdań."""
        if env_step < self.next_evaluation_step:
            return

        checkpoint_path = self.checkpoint_dir / f"step_{env_step:09d}.pth"
        torch.save(learner_policy.state_dict(), checkpoint_path)
        shutil.copy2(checkpoint_path, self.checkpoint_dir / "latest.pth")

        evaluator = self.evaluator_class(
            num_tournaments=config.EVAL_TOURNAMENTS_PER_SUITE,
            model_path=checkpoint_path,
            training_phase=self.training_phase,
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
            self.next_evaluation_step += config.EVAL_INTERVAL_STEPS

    def run_initial_evaluation(self, *, learner_policy) -> None:
        """Zapisz punkt odniesienia przed wykonaniem pierwszej aktualizacji sieci."""
        checkpoint_path = self.checkpoint_dir / "step_000000000.pth"
        torch.save(learner_policy.state_dict(), checkpoint_path)
        evaluator = self.evaluator_class(
            num_tournaments=config.EVAL_TOURNAMENTS_PER_SUITE,
            model_path=checkpoint_path,
            training_phase=self.training_phase,
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
        phase_result = results.get("phase1_mix")
        if phase_result:
            self.best_validation_score = phase_result.bb_per_100
            shutil.copy2(checkpoint_path, self.checkpoint_dir / "best.pth")
            shutil.copy2(checkpoint_path, self.checkpoint_dir / "latest.pth")

    def run_final_evaluation(self, *, learner_policy, env_step: int) -> None:
        """Zapisz ostatni model i wykonaj duży test na nieużywanych seedach."""
        final_path = self.checkpoint_dir / "final.pth"
        torch.save(learner_policy.state_dict(), final_path)
        shutil.copy2(final_path, self.checkpoint_dir / "latest.pth")
        evaluator = self.evaluator_class(
            num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
            model_path=final_path,
            training_phase=self.training_phase,
        )
        evaluator.evaluate(
            stage="final_last",
            step=env_step,
            seed_base=config.EVAL_FINAL_SEED,
        )

        # Najlepszy checkpoint walidacyjny może pochodzić ze środka treningu.
        # Oceniamy go na tych samych, nowych seedach co model końcowy, aby ich
        # porównanie nie zależało od szczęścia w rozdaniach.
        best_path = self.checkpoint_dir / "best.pth"
        if best_path.exists() and best_path != final_path:
            best_evaluator = self.evaluator_class(
                num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
                model_path=best_path,
                training_phase=self.training_phase,
            )
            best_evaluator.evaluate(
                stage="final_best",
                step=env_step,
                seed_base=config.EVAL_FINAL_SEED,
            )

    def setup_and_train(self):
        """Metoda abstrakcyjna implementowana przez trener DQN albo PPO."""
        raise NotImplementedError
