import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

import config
from training.base_trainer import BasePokerTrainer
from evaluation.evaluator_dqn import DQNEvaluator
from models import MaskedActor
from opponents import FrozenDQN, PassiveAlgorithm, AggressiveAlgorithm, SeededMixedAlgorithm
from phases import DynamicOpponentAlgorithm, ShuffleOpponentsHook
from paths import DQN_CHECKPOINT_DIR

class DQNPokerTrainer(BasePokerTrainer):
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

        if self.training_phase in ["SELF", "ADVANCED"]:
            try:
                #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do ucznia
                policy_learner.load_state_dict(torch.load(DQN_CHECKPOINT_DIR / 'final.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak końcowego modelu DQN dla ucznia, start od zera.")
        
        dqn_learner = DQN(
            policy=policy_learner,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE
        )

        # Konfiguracja Przeciwników
        net_opponent = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        policy_opponent = DiscreteQLearningPolicy(
            model=net_opponent,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=config.DQN_OPONENT_EPS,  # mała losowość, żeby nie był bardzo przewidywalny
            eps_inference=0.0
        )

        policy_opponent.eval()

        for param in policy_opponent.parameters():
            param.requires_grad = False

        frozen_opponent = FrozenDQN(
            policy=policy_opponent,
            optim=AdamOptimizerFactory(lr=0.0),
            gamma=config.DQN_GAMMA,
            n_step_return_horizon=3,
            target_update_freq=0
        )

        # W fazie 1 zamrożony model nie jest przeciwnikiem, dlatego nie
        # wczytujemy starego checkpointu. Dawny model 68-wejściowy nie pasuje
        # do obecnej sieci z 222 obserwacjami.
        if self.phase_name in {"self", "advanced"}:
            try:
                frozen_opponent.policy.load_state_dict(
                    torch.load(
                        DQN_CHECKPOINT_DIR / "best.pth",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            except FileNotFoundError:
                pass

        # Inne agenty
        random_agent = MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=self.env.action_space)
        passive_agent = PassiveAlgorithm(action_space=self.env.action_space)
        aggressive_agent = AggressiveAlgorithm(action_space=self.env.action_space)
        mixed_agent = SeededMixedAlgorithm(action_space=self.env.action_space, seed=12345)


        # Złożenie środowiska MARL
        if self.training_phase == 1:
            available_opponents = {
                "random": random_agent.policy,
                "passive": passive_agent.policy,
                "mixed": mixed_agent.policy
            }

            # 50% random, 40% passive, 10% mixed
            opponent_weights = {
                "random": 0.50,
                "passive": 0.40,
                "mixed": 0.10
            }

            opponent_1 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights)
            opponent_2 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights)
            opponent_3 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights)

            agents = [dqn_learner, opponent_1, opponent_2, opponent_3]
        # TODO: trzeba zrobić inne fazy treningu
        else:
            print("Nie ma takiej fazy")
            
        marl_algo = MultiAgentOffPolicyAlgorithm(algorithms=agents, env=self.env)

        shuffle_hook = ShuffleOpponentsHook(opponent_1, opponent_2, opponent_3)

        # Kolektory
        buffer = VectorReplayBuffer(config.DQN_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(
            marl_algo, 
            self.train_envs, 
            buffer, 
            exploration_noise=True, 
            on_episode_done_hook=shuffle_hook
        )

        test_collector = Collector(
            marl_algo, 
            self.test_envs, 
            exploration_noise=False, 
            on_episode_done_hook=shuffle_hook
        )

        print("Zapełnianie bufora pierwszymi losowymi danymi...")
        train_collector.collect(n_step=config.DQN_BUFFER_WARMUP, random=True, reset_before_collect=True)

        # Funkcje trenujące z logiką DQN (Epsilon Decay)
        # eps definiuje jak często podejmowane są losowe decyzje
        # TODO: można tu coś pokombinować, ale raczej jest git
        latest_step = {"value": 0}

        def train_fn(epoch, env_step):
            latest_step["value"] = env_step
            if self.phase_name in {"1", "random"}:
                eps = max(config.DQN_RAND_PHASE_EPS_MIN, config.DQN_EPS_MAX - env_step / (config.DQN_RAND_PHASE_EPS_DECAY * self.total_steps))
            else:
                eps = max(config.DQN_OTHER_PHASE_EPS_MIN, config.DQN_OTHER_PHASE_EPS_MAX - env_step / (config.DQN_OTHER_PHASE_EPS_DECAY * self.total_steps))
            dqn_learner.policy.set_eps_training(eps)

            self.run_periodic_evaluation(
                env_step=env_step,
                learner_policy=dqn_learner.policy,
                opponent_policy=policy_opponent,
            )

        def test_fn(epoch, env_step):
            dqn_learner.policy.set_eps_inference(0.0)

        # Trener
        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            batch_size=config.DQN_BATCH_SIZE,
            training_fn=train_fn,
            test_fn=test_fn,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        # Pomiar przed treningiem daje uczciwy punkt odniesienia dla wszystkich
        # późniejszych checkpointów tej samej, losowo zainicjalizowanej sieci.
        self.run_initial_evaluation(learner_policy=dqn_learner.policy)
        print("Rozpoczęcie treningu DQN...")
        result = OffPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        self.run_final_evaluation(
            learner_policy=dqn_learner.policy,
            env_step=latest_step["value"] or self.total_steps,
        )

if __name__ == "__main__":
    trainer = DQNPokerTrainer(
        algo_name="dqn",
        training_phase=config.TRAINING_PHASE,
        evaluator_class=DQNEvaluator,
        num_train_envs=config.DQN_NUM_TRAIN_ENVS,
        num_test_envs=config.DQN_NUM_TEST_ENVS,
        max_epochs=config.DQN_MAX_EPOCHS,
        steps_per_epoch=config.DQN_STEPS_PER_EPOCH
    )
    trainer.setup_and_train()
