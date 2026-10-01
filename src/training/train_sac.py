# --- UTWÓRZ NOWY PLIK: src/training/train_sac.py ---
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
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm import DiscreteSAC
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSACPolicy
from tianshou.algorithm.modelfree.sac import AutoAlpha
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger

import config
from training.learner_environment import make_learner_env
from evaluation.evaluator_sac import SACEvaluator
from models import MaskedActor, DiscreteActionCritic
from paths import SAC_CHECKPOINT_DIR, sac_run_dir, tensorboard_run_dir

from training.base_trainer import (
    BasePokerTrainer,
    save_training_state,
    single_training_process,
    configure_cpu_runtime,
    ensure_finite_model,
)

RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

class SACPokerTrainer(BasePokerTrainer):
    def __init__(
        self,
        *args,
        resume_path: Path | None = None,
        start_step: int | None = None,
        run_name: str = "sac",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.requested_start_step = start_step
        self.starting_step = 0
        self.run_name = run_name

    def setup_and_train(self):
        # Konfiguracja sieci
        actor = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        actor_optim = AdamOptimizerFactory(lr=config.SAC_ACTOR_LR)
        
        critic1 = DiscreteActionCritic(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        critic1_optim = AdamOptimizerFactory(lr=config.SAC_CRITIC_LR)
        
        critic2 = DiscreteActionCritic(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        critic2_optim = AdamOptimizerFactory(lr=config.SAC_CRITIC_LR)

        # Konfiguracja parametru Alpha (entropia)
        if config.SAC_AUTO_ALPHA:
            max_entropy = np.log(config.ACTION_SPACE)
            # Ustawiamy cel na ułamek maksymalnej entropii (np. 15%)
            target_entropy = config.SAC_TARGET_ENTROPY_RATIO * max_entropy
            
            # Inicjalizacja log_alpha na podstawie wartości startowej
            log_alpha = np.log(config.SAC_ALPHA) if config.SAC_ALPHA > 0 else 0.0
            alpha_optim = AdamOptimizerFactory(lr=config.SAC_ALPHA_LR)
            
            alpha_param = AutoAlpha(target_entropy, log_alpha, alpha_optim).to(self.device)
        else:
            alpha_param = config.SAC_ALPHA

        policy_learner = DiscreteSACPolicy(
            actor=actor,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            deterministic_eval=(config.TRAINING_PHASE == 1)
        )

        resume_payload = None
        if self.resume_path is not None:
            resume_payload = torch.load(self.resume_path, map_location=self.device, weights_only=True)
            if "algorithm_state" not in resume_payload:
                if self.requested_start_step is None:
                    raise ValueError("Przy wznowieniu ze starego pliku wag podaj --start-step.")
                policy_learner.load_state_dict(resume_payload)

        sac_learner = DiscreteSAC(
            policy=policy_learner,
            policy_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            tau=config.SAC_TAU,
            gamma=config.SAC_GAMMA,
            alpha=alpha_param,
            n_step_return_horizon=3,
        )

        if resume_payload is not None:
            if "algorithm_state" in resume_payload:
                if resume_payload.get("step_unit") != "learner_decisions":
                    raise ValueError("Ten stan treningu pochodzi ze starego kolektora wieloagentowego.")
                sac_learner.load_state_dict(resume_payload["algorithm_state"])
                self.starting_step = int(resume_payload["completed_env_steps"])
                self.best_validation_score = float(resume_payload.get("best_validation_score", float("-inf")))
            else:
                self.starting_step = int(self.requested_start_step)

            interval = self.evaluation_interval_steps
            self.next_evaluation_step = (self.starting_step // interval + 1) * interval
            print(f"Wznowiono SAC od kroku {self.starting_step:,} z '{self.resume_path}'.")

        if self.training_phase != 1:
            raise NotImplementedError(f"Na razie obsługiwana jest tylko faza 1 dla SAC. Otrzymano: {self.training_phase}")

        # Kolektory
        buffer = VectorReplayBuffer(config.SAC_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(
            sac_learner,
            self.train_envs, 
            buffer, 
            exploration_noise=True,
        )

        test_collector = Collector(
            sac_learner,
            self.test_envs, 
            exploration_noise=False,
        )

        print("Zapełnianie bufora pierwszymi decyzjami ucznia (warm-up)...")
        # W SAC nie ma epsilon-greedy. Nieustrenowana sieć zwraca w miarę jednorodne logity,
        # a maskowanie w MaskedActor dba o to, by wybierać tylko legalne akcje.
        train_collector.collect(
            n_step=config.SAC_BUFFER_WARMUP,
            random=False,
            reset_before_collect=True,
        )

        tensorboard_dir = tensorboard_run_dir("sac", self.run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        self.tensorboard_writer = writer
        logger = TensorboardLogger(writer, training_interval=1_000, update_interval=1_000)

        def train_fn(epoch, env_step):
            global_step = self.starting_step + env_step
            writer.add_scalar("training/replay_buffer_size", len(buffer), global_step=global_step)

            ensure_finite_model(actor, global_step)
            ensure_finite_model(critic1, global_step)
            ensure_finite_model(critic2, global_step)

            evaluation_due = global_step >= self.next_evaluation_step
            if evaluation_due:
                save_training_state(
                    sac_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                )

            self.run_periodic_evaluation(
                env_step=global_step,
                learner_policy=sac_learner.policy,
            )

            if evaluation_due:
                save_training_state(
                    sac_learner,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                )

            if global_step > 0 and global_step % config.SAC_FULL_STATE_INTERVAL_DECISIONS == 0:
                save_training_state(
                    sac_learner,
                    global_step,
                    self.checkpoint_dir / f"training_state_step_{global_step:09d}.pth",
                    best_validation_score=self.best_validation_score,
                )

        def test_fn(epoch, env_step):
            pass

        # Trener
        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=config.SAC_COLLECTION_STEPS,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            batch_size=config.SAC_BATCH_SIZE,
            update_step_num_gradient_steps_per_sample=config.SAC_UPDATE_RATIO,
            training_fn=train_fn,
            test_fn=test_fn,
            logger=logger,
            show_progress=False,
        )

        if self.starting_step == 0:
            self.run_initial_evaluation(learner_policy=sac_learner.policy)
        else:
            self._preserve_existing_checkpoints()
            
        print("Rozpoczęcie treningu SAC...")
        runtime_trainer = OffPolicyTrainer(
            algorithm=sac_learner,
            params=trainer_params,
        )
        
        try:
            result = runtime_trainer.run()
        except KeyboardInterrupt:
            interrupted_step = self.starting_step + runtime_trainer._env_step
            ensure_finite_model(actor, interrupted_step)
            interrupted_path = self.checkpoint_dir / "interrupted.pth"
            torch.save(sac_learner.policy.state_dict(), interrupted_path)
            torch.save(sac_learner.policy.state_dict(), self.checkpoint_dir / "latest.pth")
            save_training_state(
                sac_learner,
                interrupted_step,
                self.checkpoint_dir / "training_state_latest.pth",
                best_validation_score=self.best_validation_score,
            )
            writer.flush()
            writer.close()
            print(f"\n[STOP] Trening zatrzymany bezpiecznie po {interrupted_step:,} decyzjach SAC.")
            print(f"[ZAPIS] Stan do wznowienia: {self.checkpoint_dir / 'training_state_latest.pth'}")
            return
            
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        final_step = self.starting_step + result.train_step
        ensure_finite_model(actor, final_step)
        self.run_periodic_evaluation(env_step=final_step, learner_policy=sac_learner.policy)
        self.run_final_evaluation(learner_policy=sac_learner.policy, env_step=final_step)
        save_training_state(
            sac_learner,
            final_step,
            self.checkpoint_dir / "training_state_final.pth",
            best_validation_score=self.best_validation_score,
        )
        writer.flush()
        writer.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening pierwszej fazy SAC")
    parser.add_argument("--resume", type=Path, help="Stan treningu albo starszy plik wag .pth")
    parser.add_argument("--run-name", help="Nazwa izolowanego katalogu checkpointów i TensorBoard.")
    parser.add_argument("--seed", type=int, default=11_001, help="Seed inicjalizacji sieci.")
    parser.add_argument("--start-step", type=int, help="Liczba wykonanych decyzji ucznia.")
    parser.add_argument(
        "--actions", "--decisions", dest="actions", type=int,
        default=config.SAC_MAX_EPOCHS * config.SAC_STEPS_PER_EPOCH,
        help="Liczba nowych decyzji SAC (domyślnie 250 000)."
    )
    parser.add_argument(
        "--evaluation-interval", type=int, default=config.SAC_EVAL_INTERVAL_DECISIONS,
        help="Odstęp pomiędzy pełnymi walidacjami, liczony w decyzjach SAC."
    )
    args = parser.parse_args()
    
    if args.actions <= 0 or args.actions % config.SAC_STEPS_PER_EPOCH != 0:
        parser.error(f"--decisions musi być dodatnią wielokrotnością {config.SAC_STEPS_PER_EPOCH:,}.")
    if args.start_step is not None and args.resume is None:
        parser.error("--start-step ma sens tylko razem z --resume.")
    if args.evaluation_interval <= 0:
        parser.error("--evaluation-interval musi być dodatni.")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name może zawierać tylko litery, cyfry, '-' i '_' (maksymalnie 64 znaki).")
    return args

if __name__ == "__main__":
    args = parse_args()
    if config.TRAINING_PHASE != 1:
        raise ValueError("Ten skrypt wymaga TRAINING_PHASE = 1")
        
    configure_cpu_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = sac_run_dir(args.run_name)
    
    train_env_factories = [
        partial(make_learner_env, args.seed + worker_index * 1_000_000)
        for worker_index in range(config.SAC_NUM_TRAIN_ENVS)
    ]
    test_env_factories = [
        partial(make_learner_env, args.seed + 100_000_000 + worker_index * 1_000_000)
        for worker_index in range(config.SAC_NUM_TEST_ENVS)
    ]
    
    print(
        "Konfiguracja fazy 1 SAC: "
        f"run '{args.run_name}', seed {args.seed}, "
        f"{args.actions:,} decyzji SAC, "
        f"warm-up {config.SAC_BUFFER_WARMUP:,}, "
        f"ewaluacja co {args.evaluation_interval:,}, "
        f"update ratio {config.SAC_UPDATE_RATIO}, "
        f"{config.SAC_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    
    with single_training_process(checkpoint_dir):
        trainer = SACPokerTrainer(
            algo_name="sac",
            training_phase=config.TRAINING_PHASE,
            evaluator_class=SACEvaluator,
            num_train_envs=config.SAC_NUM_TRAIN_ENVS,
            num_test_envs=config.SAC_NUM_TEST_ENVS,
            max_epochs=args.actions // config.SAC_STEPS_PER_EPOCH,
            steps_per_epoch=config.SAC_STEPS_PER_EPOCH,
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