import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

from base_trainer import BasePokerTrainer
from masked_actor import MaskedActor
from utils import FrozenDQN
from evaluator_dqn import DQNEvaluator

class DQNPokerTrainer(BasePokerTrainer):
    def setup_and_train(self):
        # Konfiguracja Ucznia
        net_learner = MaskedActor(state_shape=self.observation_size, action_shape=5).to(self.device)

        policy_learner = DiscreteQLearningPolicy(
            model=net_learner,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=1.0,
            eps_inference=0.0
        )

        if self.training_phase in ["SELF", "ADVANCED"]:
            try:
                #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do ucznia
                policy_learner.load_state_dict(torch.load('final_RANDOM_dqn.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak pliku 'final_RANDOM_dqn.pth' dla ucznia, start od zera.")
        
        dqn_learner = DQN(
            policy=policy_learner,
            optim=AdamOptimizerFactory(lr=1e-4),
            gamma=0.99, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            n_step_return_horizon=3,
            target_update_freq=5000
        )

        # Konfiguracja Przeciwników
        net_opponent = MaskedActor(state_shape=self.observation_size, action_shape=5).to(self.device)
        policy_opponent = DiscreteQLearningPolicy(
            model=net_opponent,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            eps_training=0.05,  # mała losowość, żeby nie był bardzo przewidywalny
            eps_inference=0.0
        )

        policy_opponent.eval()

        for param in policy_opponent.parameters():
            param.requires_grad = False

        frozen_opponent = FrozenDQN(
            policy=policy_opponent,
            optim=AdamOptimizerFactory(lr=0.0),
            gamma=0.99,
            n_step_return_horizon=3,
            target_update_freq=0
        )

        try:
            #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do przeciwnika
            frozen_opponent.policy.load_state_dict(torch.load(f'best_{self.training_phase}_ppo.pth', map_location=self.device, weights_only=True))
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
        buffer = VectorReplayBuffer(500_000, len(self.train_envs))
        train_collector = Collector(marl_algo, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(marl_algo, self.test_envs, exploration_noise=False)

        print("Zapełnianie bufora pierwszymi losowymi danymi...")
        train_collector.collect(n_step=10_000, random=True, reset_before_collect=True)

        # Funkcje trenujące z logiką DQN (Epsilon Decay)
        # eps definiuje jak często podejmowane są losowe decyzje
        # TODO: można tu coś pokombinować, ale raczej jest git
        def train_fn(epoch, env_step):
            if self.training_phase == "RANDOM":
                eps = max(0.1, 1.0 - env_step / (0.8 * self.total_steps))
            else:
                eps = max(0.02, 0.2 - env_step / (0.5 * self.total_steps))
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
            test_step_num_episodes=10,
            batch_size=64,
            training_fn=train_fn,
            test_fn=test_fn,
            save_best_fn=self.save_best_model,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        print("Rozpoczęcie treningu DQN...")
        result = OffPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        torch.save(dqn_learner.policy.state_dict(), 'dqn_agent_1.pth')

if __name__ == "__main__":
    trainer = DQNPokerTrainer(
        algo_name="dqn",
        training_phase="RANDOM",
        evaluator_class=DQNEvaluator,
        num_train_envs=2, # dla colaba 8
        num_test_envs=1, # dla colaba 4
        max_epochs=100,
        steps_per_epoch=10_000
    )
    trainer.setup_and_train()