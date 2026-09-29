import argparse
import os
import re
import math
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, CollectStats
from tianshou.algorithm import IQN
from tianshou.algorithm.modelfree.iqn import IQNPolicy
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import ImplicitQuantileNetwork

import config
from training.learner_environment import make_learner_env
from evaluation.evaluator_iqn import IQNEvaluator
from models import PokerFeatureExtractor
from paths import iqn_run_dir, tensorboard_run_dir

def extract_model_weights(policy_state_dict: dict) -> dict:
    """Wyciąga same wagi modelu z polityki IQN i przenosi je na CPU dla workerów."""
    return {
        k.removeprefix("model."): v.cpu() 
        for k, v in policy_state_dict.items() 
        if k.startswith("model.")
    }

from training.base_trainer import (
    BasePokerTrainer,
    save_training_state,
    single_training_process,
    configure_cpu_runtime,
    ensure_finite_model,
)

RUN_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

def get_current_epsilon(env_step: int, tau: int | None = None) -> float:
    """Wykładniczo zmniejsz eksplorację według liczby decyzji ucznia."""
    tau = tau or config.IQN_EPS_TAU
    if tau <= 0:
        raise ValueError("Stała czasowa tau musi być dodatnia")
    
    decay = math.exp(-max(env_step, 0) / tau)
    return config.IQN_EPS_MIN + (config.IQN_EPS_MAX - config.IQN_EPS_MIN) * decay

class IQNPokerTrainer(BasePokerTrainer):
    def __init__(
        self,
        *args,
        resume_path: Path | None = None,
        start_step: int | None = None,
        run_name: str = "iqn",
        epsilon_tau: int = config.IQN_EPS_TAU,
        base_model_path: Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, baseline_model_path=base_model_path, **kwargs) 
        self.resume_path = resume_path
        self.requested_start_step = start_step
        self.starting_step = 0
        self.run_name = run_name
        self.epsilon_tau = epsilon_tau
        self.base_model_path = base_model_path

    def setup_and_train(self):
        # Konfiguracja sieci
        feature_net = PokerFeatureExtractor(state_shape=self.observation_size).to(self.device)
        net_learner = ImplicitQuantileNetwork(
            preprocess_net=feature_net,
            action_shape=config.ACTION_SPACE,
            num_cosines=config.IQN_NUM_COSINES,
        ).to(self.device)

        optim = AdamOptimizerFactory(lr=config.IQN_LEARNING_RATE)

        policy_learner = IQNPolicy(
            model=net_learner,
            action_space=self.env.action_space,
            sample_size=config.IQN_SAMPLE_SIZE,
            online_sample_size=config.IQN_ONLINE_SAMPLE_SIZE,
            target_sample_size=config.IQN_TARGET_SAMPLE_SIZE,
            eps_training=config.IQN_EPS_MAX,
            eps_inference=0.0
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
                    raise ValueError("Przy wznowieniu ze starego pliku wag podaj --start-step.")
                policy_learner.load_state_dict(resume_payload)

        algorithm = IQN(
            policy=policy_learner,
            optim=optim,
            gamma=config.IQN_GAMMA,
            n_step_return_horizon=config.IQN_N_STEP,
            target_update_freq=config.IQN_TARGET_NET_UPDATE,
        ).to(self.device)

        if self.training_phase == 2:
            if not self.base_model_path and not self.resume_path:
                raise ValueError("Faza 2 wymaga podania --base-model (np. best.pth z Fazy 1).")
            
            if self.base_model_path:
                print(f"Ładowanie modelu bazowego z {self.base_model_path}...")
                base_state = torch.load(self.base_model_path, map_location=self.device, weights_only=True)
                
                if "algorithm_state" not in base_state:
                    policy_learner.load_state_dict(base_state)
                else:
                    algo_state = base_state["algorithm_state"]
                    if "_optimizers" in algo_state:
                        del algo_state["_optimizers"]
                    policy_learner.load_state_dict(self.algo_state, strict=False)

                # Rozsyłamy wagi do środowisk (Latest Self i pierwszy Historical Self)
                model_weights = extract_model_weights(policy_learner.state_dict())
                self.train_envs.set_env_attr("latest_model_weights", model_weights)
                self.train_envs.set_env_attr("new_historical_model_weights", model_weights)
                self.test_envs.set_env_attr("latest_model_weights", model_weights)
                self.test_envs.set_env_attr("new_historical_model_weights", model_weights)
                print("Rozesłano model bazowy do przeciwników w środowiskach.")

        if resume_payload is not None:
            if "algorithm_state" in resume_payload:
                if resume_payload.get("step_unit") != "learner_decisions":
                    raise ValueError("Ten stan treningu pochodzi ze starego kolektora wieloagentowego.")
                algorithm.load_state_dict(resume_payload["algorithm_state"])
                self.starting_step = int(resume_payload["completed_env_steps"])
                self.best_validation_score = float(resume_payload.get("best_validation_score", float("-inf")))
                saved_tau = int(resume_payload.get("epsilon_tau", config.IQN_EPS_TAU))
                if saved_tau != self.epsilon_tau:
                    raise ValueError(f"Checkpoint korzystał z innego tau: {saved_tau:,}, obecnie {self.epsilon_tau:,}.")
                
                # Jeśli wznawiamy Fazę 2, musimy też rozesłać wagi wznowionego modelu do środowisk
                if self.training_phase == 2:
                    model_weights = extract_model_weights(algorithm.policy.state_dict())
                    self.train_envs.set_env_attr("latest_model_weights", model_weights)
                    self.test_envs.set_env_attr("latest_model_weights", model_weights)
            else:
                self.starting_step = int(self.requested_start_step)

            interval = self.evaluation_interval_steps
            self.next_evaluation_step = (self.starting_step // interval + 1) * interval
            print(f"Wznowiono IQN od kroku {self.starting_step:,} z '{self.resume_path}'.")

        # Kolektory
        buffer = VectorReplayBuffer(config.IQN_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector[CollectStats](
            algorithm,
            self.train_envs, 
            buffer, 
            exploration_noise=True,
        )

        test_collector = Collector[CollectStats](
            algorithm,
            self.test_envs, 
            exploration_noise=False,
        )

        print("Zapełnianie bufora pierwszymi legalnymi decyzjami ucznia...")
        algorithm.policy.set_eps_training(1.0)
        algorithm.policy.set_eps_inference(1.0)
        train_collector.collect(
            n_step=config.IQN_BUFFER_WARMUP,
            random=False,
            reset_before_collect=True,
        )
        algorithm.policy.set_eps_inference(0.0)

        tensorboard_dir = tensorboard_run_dir("iqn", self.run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        self.tensorboard_writer = writer
        logger = TensorboardLogger(
            writer,
            training_interval=1_000,
            update_interval=1_000,
        )

        def train_fn(epoch, env_step):
            global_step = self.starting_step + env_step
            eps = get_current_epsilon(global_step, self.epsilon_tau)
            algorithm.policy.set_eps_training(eps)

            writer.add_scalar("training/epsilon", eps, global_step=global_step)
            writer.add_scalar("training/replay_buffer_size", len(buffer), global_step=global_step)

            ensure_finite_model(net_learner, global_step)

            if self.training_phase == 2:
                # Co ewaluację aktualizujemy "Latest Self"
                if evaluation_due:
                    model_weights = extract_model_weights(algorithm.policy.state_dict())
                    self.train_envs.set_env_attr("latest_model_weights", model_weights)
                    self.test_envs.set_env_attr("latest_model_weights", model_weights)
                
                # Co pełny zapis stanu dodajemy model do "Historical Self"
                if global_step > 0 and global_step % config.IQN_FULL_STATE_INTERVAL_DECISIONS == 0:
                    model_weights = extract_model_weights(algorithm.policy.state_dict())
                    self.train_envs.set_env_attr("new_historical_model_weights", model_weights)
                    self.test_envs.set_env_attr("new_historical_model_weights", model_weights)

            evaluation_due = global_step >= self.next_evaluation_step
            if evaluation_due:
                save_training_state(
                    algorithm,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
                )

            self.run_periodic_evaluation(
                env_step=global_step,
                learner_policy=algorithm.policy,
            )

            if evaluation_due:
                save_training_state(
                    algorithm,
                    global_step,
                    self.checkpoint_dir / "training_state_latest.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
                )

            if global_step > 0 and global_step % config.IQN_FULL_STATE_INTERVAL_DECISIONS == 0:
                save_training_state(
                    algorithm,
                    global_step,
                    self.checkpoint_dir / f"training_state_step_{global_step:09d}.pth",
                    best_validation_score=self.best_validation_score,
                    epsilon_tau=self.epsilon_tau,
                    epsilon=eps,
                )

        def test_fn(epoch, env_step):
            algorithm.policy.set_eps_inference(0.0)

        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=config.IQN_COLLECTION_STEPS,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            batch_size=config.IQN_BATCH_SIZE,
            update_step_num_gradient_steps_per_sample=config.IQN_UPDATE_RATIO,
            training_fn=train_fn,
            test_fn=test_fn,
            logger=logger,
            show_progress=False,
        )

        if self.starting_step == 0:
            self.run_initial_evaluation(learner_policy=algorithm.policy)
        else:
            self._preserve_existing_checkpoints()
            
        print("Rozpoczęcie treningu IQN...")
        runtime_trainer = OffPolicyTrainer(
            algorithm=algorithm,
            params=trainer_params,
        )
        try:
            result = runtime_trainer.run()
        except KeyboardInterrupt:
            interrupted_step = self.starting_step + runtime_trainer._env_step
            ensure_finite_model(net_learner, interrupted_step)
            interrupted_path = self.checkpoint_dir / "interrupted.pth"
            torch.save(algorithm.policy.state_dict(), interrupted_path)
            torch.save(algorithm.policy.state_dict(), self.checkpoint_dir / "latest.pth")
            
            eps = get_current_epsilon(interrupted_step, self.epsilon_tau)
            save_training_state(
                algorithm,
                interrupted_step,
                self.checkpoint_dir / "training_state_latest.pth",
                best_validation_score=self.best_validation_score,
                epsilon_tau=self.epsilon_tau,
                epsilon=eps,
            )
            writer.flush()
            writer.close()
            print(f"\n[STOP] Trening zatrzymany bezpiecznie po {interrupted_step:,} decyzjach IQN.")
            print(f"[ZAPIS] Stan do wznowienia: {self.checkpoint_dir / 'training_state_latest.pth'}")
            return
            
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        final_step = self.starting_step + result.train_step
        ensure_finite_model(net_learner, final_step)
        self.run_periodic_evaluation(env_step=final_step, learner_policy=algorithm.policy)
        self.run_final_evaluation(learner_policy=algorithm.policy, env_step=final_step)
        
        eps = get_current_epsilon(final_step, self.epsilon_tau)
        save_training_state(
            algorithm,
            final_step,
            self.checkpoint_dir / "training_state_final.pth",
            best_validation_score=self.best_validation_score,
            epsilon_tau=self.epsilon_tau,
            epsilon=eps,
        )
        writer.flush()
        writer.close()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening pierwszej fazy IQN")
    parser.add_argument("--resume", type=Path, help="Stan treningu albo starszy plik wag .pth")
    parser.add_argument("--run-name", help="Nazwa izolowanego katalogu checkpointów i TensorBoard.")
    parser.add_argument("--seed", type=int, default=11_001, help="Seed inicjalizacji sieci.")
    parser.add_argument("--start-step", type=int, help="Liczba wykonanych decyzji ucznia.")
    parser.add_argument(
        "--actions", "--decisions", dest="actions", type=int,
        default=config.IQN_MAX_EPOCHS * config.IQN_STEPS_PER_EPOCH,
        help="Liczba nowych decyzji IQN (domyślnie 250 000)."
    )
    parser.add_argument(
        "--epsilon-tau", type=int, default=config.IQN_EPS_TAU,
        help="Stała czasowa tau dla wykładniczego zaniku epsilona."
    )
    parser.add_argument(
        "--evaluation-interval", type=int, default=config.IQN_EVAL_INTERVAL_DECISIONS,
        help="Odstęp pomiędzy pełnymi walidacjami, liczony w decyzjach IQN."
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        help="Ścieżka do najlepszego modelu z Fazy 1 (wymagane w Fazie 2)."
    )

    args = parser.parse_args()
    
    if args.actions <= 0 or args.actions % config.IQN_STEPS_PER_EPOCH != 0:
        parser.error(f"--decisions musi być dodatnią wielokrotnością {config.IQN_STEPS_PER_EPOCH:,}.")
    if args.start_step is not None and args.resume is None:
        parser.error("--start-step ma sens tylko razem z --resume.")
    if args.epsilon_tau <= 0:
        parser.error("--epsilon-tau musi być dodatnie.")
    if args.evaluation_interval <= 0:
        parser.error("--evaluation-interval musi być dodatni.")
    if args.run_name is None:
        args.run_name = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    if not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name może zawierać tylko litery, cyfry, '-' i '_' (maksymalnie 64 znaki).")
    return args

if __name__ == "__main__":
    args = parse_args()
    configure_cpu_runtime()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_dir = iqn_run_dir(args.run_name)
    
    train_env_factories = [
        partial(make_learner_env, args.seed + worker_index * 1_000_000, algo_name="iqn")
        for worker_index in range(config.IQN_NUM_TRAIN_ENVS)
    ]
    test_env_factories = [
        partial(make_learner_env, args.seed + 100_000_000 + worker_index * 1_000_000, algo_name="iqn")
        for worker_index in range(config.IQN_NUM_TEST_ENVS)
    ]
    
    print(
        "Konfiguracja fazy 1 IQN: "
        f"run '{args.run_name}', seed {args.seed}, "
        f"{args.actions:,} decyzji IQN, "
        f"warm-up {config.IQN_BUFFER_WARMUP:,}, "
        f"epsilon tau {args.epsilon_tau:,}, "
        f"ewaluacja co {args.evaluation_interval:,}, "
        f"update ratio {config.IQN_UPDATE_RATIO}, "
        f"{config.IQN_NUM_TRAIN_ENVS} środowisk, "
        f"{config.TORCH_NUM_THREADS} wątek PyTorch."
    )
    
    with single_training_process(checkpoint_dir):
        trainer = IQNPokerTrainer(
            algo_name="iqn",
            training_phase=config.TRAINING_PHASE,
            evaluator_class=IQNEvaluator,
            num_train_envs=config.IQN_NUM_TRAIN_ENVS,
            num_test_envs=config.IQN_NUM_TEST_ENVS,
            max_epochs=args.actions // config.IQN_STEPS_PER_EPOCH,
            steps_per_epoch=config.IQN_STEPS_PER_EPOCH,
            env_factory=make_learner_env,
            evaluation_interval_steps=args.evaluation_interval,
            checkpoint_dir=checkpoint_dir,
            train_env_factories=train_env_factories,
            test_env_factories=test_env_factories,
            resume_path=args.resume,
            start_step=args.start_step,
            run_name=args.run_name,
            epsilon_tau=args.epsilon_tau,
            base_model_path=args.base_model
        )
        trainer.setup_and_train()