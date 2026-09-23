import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

import config
from base_trainer import BasePokerTrainer
from evaluator_dqn import DQNEvaluator
from models import MaskedActor
from opponents import FrozenDQN
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

        try:
            #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do przeciwnika
            frozen_opponent.policy.load_state_dict(torch.load(DQN_CHECKPOINT_DIR / 'best.pth', map_location=self.device, weights_only=True))
        except FileNotFoundError:
            pass

        random_agent = MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=self.env.action_space)

        # Złożenie środowiska MARL
        if self.training_phase == "RANDOM":
            agents = [dqn_learner, random_agent, random_agent, random_agent]
        elif self.training_phase == "SELF":
            agents = [dqn_learner, frozen_opponent, frozen_opponent, frozen_opponent]
        elif self.training_phase == "ADVANCED":
            agents = [dqn_learner, frozen_opponent, random_agent, frozen_opponent]
            
        marl_algo = MultiAgentOffPolicyAlgorithm(algorithms=agents, env=self.env)

        # Kolektory
        buffer = VectorReplayBuffer(config.DQN_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(marl_algo, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(marl_algo, self.test_envs, exploration_noise=False)

        print("Zapełnianie bufora pierwszymi losowymi danymi...")
        train_collector.collect(n_step=config.DQN_BUFFER_WARMUP, random=True, reset_before_collect=True)

        # Funkcje trenujące z logiką DQN (Epsilon Decay)
        # eps definiuje jak często podejmowane są losowe decyzje
        # TODO: można tu coś pokombinować, ale raczej jest git
        def train_fn(epoch, env_step):
            if self.training_phase == "RANDOM":
                eps = max(config.DQN_RAND_PHASE_EPS_MIN, config.DQN_EPS_MAX - env_step / (config.DQN_RAND_PHASE_EPS_DECAY * self.total_steps))
            else:
                eps = max(config.DQN_OTHER_PHASE_EPS_MIN, config.DQN_OTHER_PHASE_EPS_MAX - env_step / (config.DQN_OTHER_PHASE_EPS_DECAY * self.total_steps))
            dqn_learner.policy.set_eps_training(eps)
            
            self.run_periodic_opponent_update(epoch, dqn_learner.policy, policy_opponent)

        def test_fn(epoch, env_step):
            dqn_learner.policy.set_eps_inference(0.0)

        # Trener
        trainer_params = OffPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.OPPONENT_UPDATE_INTERVAL,
            batch_size=config.DQN_BATCH_SIZE,
            training_fn=train_fn,
            test_fn=test_fn,
            save_best_fn=self.save_best_model,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        print("Rozpoczęcie treningu DQN...")
        result = OffPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        torch.save(dqn_learner.policy.state_dict(), DQN_CHECKPOINT_DIR / 'final.pth')

if __name__ == "__main__":
    trainer = DQNPokerTrainer(
        algo_name="dqn",
        training_phase="RANDOM",
        evaluator_class=DQNEvaluator,
        num_train_envs=config.DQN_NUM_TRAIN_ENVS, # dla colaba 8
        num_test_envs=config.DQN_NUM_TEST_ENVS, # dla colaba 4
        max_epochs=config.DQN_MAX_EPOCHS,
        steps_per_epoch=config.DQN_STEPS_PER_EPOCH
    )
    trainer.setup_and_train()
