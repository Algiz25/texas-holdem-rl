import argparse
import os
import re
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
import math

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger

import config
from training.learner_environment import make_learner_env
from evaluation.evaluator_dqn import DQNEvaluator
from models import MaskedActor
from paths import DQN_CHECKPOINT_DIR, dqn_run_dir, tensorboard_run_dir

from training.base_trainer import (
    BasePokerTrainer,
    save_training_state,
    single_training_process,
    configure_cpu_runtime,
    ensure_finite_model,
)


RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def phase_one_epsilon(env_step: int, tau: int | None = None) -> float:
    """Wykładniczo zmniejsz eksplorację według liczby decyzji ucznia."""
    tau = tau or config.DQN_PHASE1_EPS_TAU
    if tau <= 0:
        raise ValueError("Stała czasowa tau musi być dodatnia")
    
    # Wzór: eps_min + (eps_max - eps_min) * exp(-t / tau)
    decay = math.exp(-max(env_step, 0) / tau)
    return config.DQN_PHASE1_EPS_MIN + (config.DQN_EPS_MAX - config.DQN_PHASE1_EPS_MIN) * decay

class DQNPokerTrainer(BasePokerTrainer):
    def __init__(
        self,
        *args,
        resume_path: Path | None = None,
        start_step: int | None = None,
        run_name: str = "dqn",
        epsilon_tau: int = config.DQN_PHASE1_EPS_TAU,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.requested_start_step = start_step
        self.starting_step = 0
        self.run_name = run_name
        self.epsilon_tau = epsilon_tau

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

        if self.training_phase in ["SELF", "ADVANCED"]:
            try:
                #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do ucznia
                policy_learner.load_state_dict(torch.load(self.checkpoint_dir / 'final.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak końcowego modelu DQN dla ucznia, start od zera.")
        
        dqn_learner = DQN(
            policy=policy_learner,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA,
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
                saved_tau = int(resume_payload.get("epsilon_tau", config.DQN_PHASE1_EPS_TAU))
                if saved_tau != self.epsilon_tau:
                    raise ValueError(f"Checkpoint korzystał z innego tau: {saved_tau:,}, obecnie {self.epsilon_tau:,}.")
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
        if self.training_phase != 1:
            raise NotImplementedError(f"Nieobsługiwana faza DQN: {self.training_phase}")

        # Kolektory
        buffer = VectorReplayBuffer(config.DQN_BUFFER_SIZE, len(self.train_envs))
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
        dqn_learner.policy.set_eps_training(1.0)
        # Warm-up odbywa się jeszcze przed wejściem trenera w kontekst
        # ``training_step`` Tianshou. Dlatego na czas tej jednej kolekcji
        # ustawiamy również epsilon inferencyjny, po czym natychmiast wracamy
        # do deterministycznej ewaluacji.
        dqn_learner.policy.set_eps_inference(1.0)
        train_collector.collect(
            n_step=config.DQN_BUFFER_WARMUP,
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
            training_interval=1_000,
            update_interval=1_000,
        )

        # Funkcje trenujące z logiką DQN
        def train_fn(epoch, env_step):
            global_step = self.starting_step + env_step
            eps = phase_one_epsilon(global_step, self.epsilon_tau)
            dqn_learner.policy.set_eps_training(eps)

            writer.add_scalar("training/epsilon", eps, global_step=global_step)
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
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
                )

            self.run_periodic_evaluation(
                env_step=global_step,
                learner_policy=dqn_learner.policy,
            )

            # Po udanej ewaluacji zapisujemy stan ponownie, aby zawierał także
            # nowy najlepszy wynik. Pierwszy zapis powyżej pozostaje awaryjną
            # kopią na wypadek przerwania samej ewaluacji.
            if evaluation_due:
                save_training_state(
                    dqn_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
                )

            # Pełny, nienadpisywany stan co 10 epok pozwala wybrać konkretny
            # punkt startowy kolejnej fazy. Zapis następuje po ewaluacji, więc
            # zawiera również aktualny najlepszy wynik walidacyjny.
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
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
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
            update_step_num_gradient_steps_per_sample=config.DQN_UPDATE_RATIO,
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
            interrupted_step = self.starting_step + runtime_trainer._env_step
            ensure_finite_model(net_learner, interrupted_step)
            interrupted_path = self.checkpoint_dir / "interrupted.pth"
            torch.save(dqn_learner.policy.state_dict(), interrupted_path)
            torch.save(
                dqn_learner.policy.state_dict(),
                self.checkpoint_dir / "latest.pth",
            )
            
            eps = phase_one_epsilon(interrupted_step, self.epsilon_tau)
                
            save_training_state(
                dqn_learner,
                interrupted_step,
                self.checkpoint_dir / "training_state_latest.pth",
                best_validation_score=self.best_validation_score,
                epsilon_tau=self.epsilon_tau,
                epsilon=eps,
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
        
        eps = phase_one_epsilon(final_step, self.epsilon_tau)
            
        save_training_state(
            dqn_learner,
            final_step,
            self.checkpoint_dir / "training_state_final.pth",
            best_validation_score=self.best_validation_score,
            epsilon_tau=self.epsilon_tau,
            epsilon=eps,
        )
        writer.flush()
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening pierwszej fazy DQN")
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
        "--epsilon-tau", # <--- ZMIENIONE
        type=int,
        default=config.DQN_PHASE1_EPS_TAU,
        help=(
            "Stała czasowa tau dla wykładniczego zaniku epsilona "
            "(domyślnie 50 000)."
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=config.DQN_EVAL_INTERVAL_DECISIONS,
        help="Odstęp pomiędzy pełnymi walidacjami, liczony w decyzjach DQN.",
    )
    args = parser.parse_args()
    if args.actions <= 0 or args.actions % config.DQN_STEPS_PER_EPOCH != 0:
        parser.error(
            f"--decisions musi być dodatnią wielokrotnością "
            f"{config.DQN_STEPS_PER_EPOCH:,}."
        )
    if args.start_step is not None and args.resume is None:
        parser.error("--start-step ma sens tylko razem z --resume.")
    if args.epsilon_tau <= 0:
        parser.error("--epsilon-tau musi być dodatnie.")
    if args.evaluation_interval <= 0:
        parser.error("--evaluation-interval musi być dodatni.")
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
    configure_cpu_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = dqn_run_dir(args.run_name)
    train_env_factories = [
        partial(make_learner_env, args.seed + worker_index * 1_000_000, algo_name="dqn")
        for worker_index in range(config.DQN_NUM_TRAIN_ENVS)
    ]
    test_env_factories = [
        partial(make_learner_env, args.seed + 100_000_000 + worker_index * 1_000_000, algo_name="dqn")
        for worker_index in range(config.DQN_NUM_TEST_ENVS)
    ]
    print(
        "Konfiguracja fazy 1: "
        f"run '{args.run_name}', seed {args.seed}, "
        f"{args.actions:,} decyzji DQN, "
        f"warm-up {config.DQN_BUFFER_WARMUP:,}, "
        f"epsilon tau {args.epsilon_tau:,}, "
        f"ewaluacja co {args.evaluation_interval:,}, "
        f"update ratio {config.DQN_UPDATE_RATIO}, "
        f"{config.DQN_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    with single_training_process(checkpoint_dir):
        trainer = DQNPokerTrainer(
            algo_name="dqn",
            training_phase=config.TRAINING_PHASE,
            evaluator_class=DQNEvaluator,
            num_train_envs=config.DQN_NUM_TRAIN_ENVS,
            num_test_envs=config.DQN_NUM_TEST_ENVS,
            max_epochs=args.actions // config.DQN_STEPS_PER_EPOCH,
            steps_per_epoch=config.DQN_STEPS_PER_EPOCH,
            env_factory=make_learner_env,
            evaluation_interval_steps=args.evaluation_interval,
            checkpoint_dir=checkpoint_dir,
            train_env_factories=train_env_factories,
            test_env_factories=test_env_factories,
            resume_path=args.resume,
            start_step=args.start_step,
            run_name=args.run_name,
            epsilon_tau=args.epsilon_tau,
        )
        trainer.setup_and_train()
