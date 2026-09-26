import argparse
import os
import re
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger

import config
from training.base_trainer import BasePokerTrainer
from training.learner_environment import make_learner_env
from evaluation.evaluator_ppo import PPOEvaluator
from models import MaskedActor, Critic
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from paths import PPO_CHECKPOINT_DIR, tensorboard_run_dir

RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

def save_training_state(
    ppo: PPO,
    env_step: int,
    path: Path,
    *,
    best_validation_score: float,
) -> None:
    """Zapisz stan PPO potrzebny do bezpiecznej kontynuacji treningu."""
    payload = {
        "format_version": 3,
        "step_unit": "learner_decisions",
        "algorithm_state": ppo.state_dict(),
        "completed_env_steps": env_step,
        "best_validation_score": best_validation_score,
        "observation_size": config.OBSERVATION_SIZE,
        "action_space": config.ACTION_SPACE,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)

@contextmanager
def single_training_process():
    """Nie pozwól przypadkowo uruchomić dwóch treningów w tym samym katalogu."""
    PPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = PPO_CHECKPOINT_DIR / "training.lock"
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
                "Inny trening PPO już działa. Nie uruchamiaj drugiego procesu."
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

def configure_cpu_runtime() -> None:
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
            f"Niestabilny PPO po {env_step:,} decyzjach ucznia; "
            f"niepoprawne parametry: {names}"
        )

class PPOPokerTrainer(BasePokerTrainer):
    def __init__(
        self,
        *args,
        resume_path: Path | None = None,
        start_step: int | None = None,
        run_name: str = "ppo",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.requested_start_step = start_step
        self.starting_step = 0
        self.run_name = run_name

    def setup_and_train(self):
        # Konfiguracja Ucznia
        actor_learner = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        critic_learner = Critic(state_shape=self.observation_size).to(self.device)

        def dist_fn(logits):
            return Categorical(logits=logits)

        policy_learner = ProbabilisticActorPolicy(
            actor=actor_learner,
            dist_fn=dist_fn,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            action_scaling=False
        )

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
                policy_learner.load_state_dict(torch.load(self.checkpoint_dir / 'final.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak końcowego modelu PPO dla ucznia, start od zera.")

        ppo_learner = PPO(
            policy=policy_learner,
            critic=critic_learner,
            optim=AdamOptimizerFactory(lr=config.PPO_LEARNING_RATE),
            gamma=config.PPO_GAMMA,
            gae_lambda=config.PPO_GAE_LAMBDA,
            vf_coef=config.PPO_VF_COEF,
            ent_coef=config.PPO_ENT_COEF,
            eps_clip=config.PPO_EPS_CLIP,
            advantage_normalization=True
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
                ppo_learner.load_state_dict(resume_payload["algorithm_state"])
                self.starting_step = int(resume_payload["completed_env_steps"])
                self.best_validation_score = float(
                    resume_payload.get("best_validation_score", float("-inf"))
                )
            else:
                self.starting_step = int(self.requested_start_step)

            interval = self.evaluation_interval_steps
            self.next_evaluation_step = (
                self.starting_step // interval + 1
            ) * interval
            print(
                f"Wznowiono PPO od kroku {self.starting_step:,} z "
                f"'{self.resume_path}'."
            )

        if self.training_phase != 1:
            raise NotImplementedError(f"Nieobsługiwana faza PPO: {self.training_phase}")

        # Kolektory
        buffer = VectorReplayBuffer(config.PPO_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(
            ppo_learner,
            self.train_envs,
            buffer,
            exploration_noise=True,
        )

        test_collector = Collector(
            ppo_learner,
            self.test_envs,
            exploration_noise=False,
        )

        tensorboard_dir = tensorboard_run_dir("ppo", self.run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        self.tensorboard_writer = writer
        logger = TensorboardLogger(
            writer,
            training_interval=1,
            update_interval=1,
        )

        def train_fn(epoch, env_step):
            global_step = self.starting_step + env_step
            writer.add_scalar(
                "training/replay_buffer_size",
                len(buffer),
                global_step=global_step,
            )

            ensure_finite_model(actor_learner, global_step)
            ensure_finite_model(critic_learner, global_step)

            evaluation_due = global_step >= self.next_evaluation_step
            if evaluation_due:
                save_training_state(
                    ppo_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                )

            self.run_periodic_evaluation(
                env_step=global_step,
                learner_policy=ppo_learner.policy,
            )

            if evaluation_due:
                save_training_state(
                    ppo_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                )

            if (
                global_step > 0
                and global_step % config.PPO_FULL_STATE_INTERVAL_DECISIONS == 0
            ):
                save_training_state(
                    ppo_learner,
                    global_step,
                    self.checkpoint_dir
                    / f"training_state_step_{global_step:09d}.pth",
                    best_validation_score=self.best_validation_score,
                )

        def test_fn(epoch, env_step):
            pass

        # Trener
        trainer_params = OnPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=self.steps_per_epoch,
            update_step_num_repetitions=config.PPO_REPEAT_PER_COLLECT,
            batch_size=config.PPO_BATCH_SIZE,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            training_fn=train_fn,
            test_fn=test_fn,
            logger=logger,
            show_progress=False,
        )

        if self.starting_step == 0:
            self.run_initial_evaluation(learner_policy=ppo_learner.policy)
        else:
            self._preserve_existing_checkpoints()
        
        print("Rozpoczęcie treningu PPO...")
        runtime_trainer = OnPolicyTrainer(
            algorithm=ppo_learner,
            params=trainer_params,
        )
        
        try:
            result = runtime_trainer.run()
        except KeyboardInterrupt:
            interrupted_step = self.starting_step + runtime_trainer._env_step
            ensure_finite_model(actor_learner, interrupted_step)
            interrupted_path = self.checkpoint_dir / "interrupted.pth"
            torch.save(ppo_learner.policy.state_dict(), interrupted_path)
            torch.save(
                ppo_learner.policy.state_dict(),
                self.checkpoint_dir / "latest.pth",
            )
            save_training_state(
                ppo_learner,
                interrupted_step,
                self.checkpoint_dir / "training_state_latest.pth",
                best_validation_score=self.best_validation_score,
            )
            writer.flush()
            writer.close()
            print(
                "\n[STOP] Trening zatrzymany bezpiecznie po "
                f"{interrupted_step:,} decyzjach PPO."
            )
            print(f"[ZAPIS] Stan do wznowienia: {self.checkpoint_dir / 'training_state_latest.pth'}")
            return
        
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        final_step = self.starting_step + result.train_step
        ensure_finite_model(actor_learner, final_step)
        self.run_periodic_evaluation(
            env_step=final_step,
            learner_policy=ppo_learner.policy,
        )
        self.run_final_evaluation(
            learner_policy=ppo_learner.policy,
            env_step=final_step,
        )
        save_training_state(
            ppo_learner,
            final_step,
            self.checkpoint_dir / "training_state_final.pth",
            best_validation_score=self.best_validation_score,
        )
        writer.flush()
        writer.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening pierwszej fazy PPO")
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
        default=config.PPO_MAX_EPOCHS * config.PPO_STEPS_PER_EPOCH,
        help="Liczba nowych decyzji PPO (domyślnie ~250 000).",
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=config.PPO_EVAL_INTERVAL_DECISIONS,
        help="Odstęp pomiędzy pełnymi walidacjami, liczony w decyzjach PPO.",
    )
    args = parser.parse_args()
    if args.actions <= 0 or args.actions % config.PPO_STEPS_PER_EPOCH != 0:
        parser.error(
            f"--decisions musi być dodatnią wielokrotnością "
            f"{config.PPO_STEPS_PER_EPOCH:,}."
        )
    if args.start_step is not None and args.resume is None:
        parser.error("--start-step ma sens tylko razem z --resume.")
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
    if config.TRAINING_PHASE != 1:
        raise ValueError("Ten skrypt fazy rozgrzewkowej wymaga TRAINING_PHASE = 1")
    
    configure_cpu_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = PPO_CHECKPOINT_DIR / args.run_name
    
    train_env_factories = [
        partial(make_learner_env, args.seed + worker_index * 1_000_000)
        for worker_index in range(config.PPO_NUM_TRAIN_ENVS)
    ]
    test_env_factories = [
        partial(make_learner_env, args.seed + 100_000_000 + worker_index * 1_000_000)
        for worker_index in range(config.PPO_NUM_TEST_ENVS)
    ]
    
    print(
        "Konfiguracja fazy 1 PPO: "
        f"run '{args.run_name}', seed {args.seed}, "
        f"{args.actions:,} decyzji PPO, "
        f"ewaluacja co {args.evaluation_interval:,}, "
        f"{config.PPO_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    
    with single_training_process():
        trainer = PPOPokerTrainer(
            algo_name="ppo",
            training_phase=config.TRAINING_PHASE,
            evaluator_class=PPOEvaluator,
            num_train_envs=config.PPO_NUM_TRAIN_ENVS,
            num_test_envs=config.PPO_NUM_TEST_ENVS,
            max_epochs=args.actions // config.PPO_STEPS_PER_EPOCH,
            steps_per_epoch=config.PPO_STEPS_PER_EPOCH,
            env_factory=make_learner_env,
            evaluation_interval_steps=args.evaluation_interval,
            checkpoint_dir=checkpoint_dir,
            train_env_factories=train_env_factories,
            test_env_factories=test_env_factories,
            resume_path=args.resume,
            start_step=args.start_step,
            run_name=args.run_name,
        )
        trainer.setup_and_train()