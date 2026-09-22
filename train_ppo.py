import torch
from torch.distributions import Categorical
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

from base_trainer import BasePokerTrainer
from masked_actor import MaskedActor, Critic, CPUActionActorPolicy
from utils import RandomOnPolicyAgent, FrozenPPO
from evaluator_ppo import PPOEvaluator

class PPOPokerTrainer(BasePokerTrainer):
    def setup_and_train(self):
        # Konfiguracja Ucznia
        actor_learner = MaskedActor(state_shape=self.observation_size, action_shape=5).to(self.device)
        critic_learner = Critic(state_shape=self.observation_size).to(self.device)
        
        def dist_fn(logits):
            return Categorical(logits=logits)

        policy_learner = CPUActionActorPolicy(
            actor=actor_learner,
            dist_fn=dist_fn,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            action_scaling=False
        )

        if self.training_phase in ["SELF", "ADVANCED"]:
            try:
                #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do ucznia
                policy_learner.load_state_dict(torch.load(f'final_RANDOM_ppo.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak pliku 'final_RANDOM_ppo.pth' dla ucznia, start od zera.")
        
        ppo_learner = PPO(
            policy=policy_learner,
            critic=critic_learner,
            optim=AdamOptimizerFactory(lr=3e-4), # TODO: przemyśleć tą wartość
            gamma=0.99, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            gae_lambda=0.95,
            vf_coef=0.5,
            ent_coef=0.01,
            eps_clip=0.2,
            advantage_normalization=True
        )

        # Konfiguracja Przeciwników
        actor_opponent = MaskedActor(state_shape=self.observation_size, action_shape=5).to(self.device)
        critic_opponent = Critic(state_shape=self.observation_size).to(self.device)
        
        policy_opponent = CPUActionActorPolicy(
            actor=actor_opponent,
            dist_fn=dist_fn,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            action_scaling=False
        )

        # Oszczędza zasoby
        policy_opponent.eval()
        critic_opponent.eval()

        for param in policy_opponent.parameters():
            param.requires_grad = False
            
        for param in critic_opponent.parameters():
            param.requires_grad = False

        frozen_opponent = FrozenPPO(
            policy=policy_opponent,
            critic=critic_opponent,
            optim=AdamOptimizerFactory(0),
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=0.0,
            vf_coef=0.0,
            ent_coef=0.0,
            eps_clip=0.2
        )

        try:
            #TODO: trzeba zrobić jakiś lepszy system wczytywania modelu do przeciwnika
            frozen_opponent.policy.load_state_dict(torch.load(f'best_{self.training_phase}_ppo.pth', map_location=self.device, weights_only=True))
        except FileNotFoundError:
            pass

        random_agent = RandomOnPolicyAgent(action_space=self.env.action_space)

        # 3. Złożenie algorytmu MARL
        if self.training_phase == "RANDOM":
            agents = [ppo_learner, random_agent, random_agent, random_agent]
        elif self.training_phase == "SELF":
            agents = [ppo_learner, frozen_opponent, frozen_opponent, frozen_opponent]
        elif self.training_phase == "ADVANCED":
            agents = [ppo_learner, frozen_opponent, random_agent, frozen_opponent]

        marl_algo = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=self.env)

        # Kolektory
        buffer = VectorReplayBuffer(2048, len(self.train_envs))
        train_collector = Collector(marl_algo, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(marl_algo, self.test_envs, exploration_noise=False)

        # Funkcje trenujące z logiką PPO
        def train_fn(epoch, env_step):
            self.run_periodic_opponent_update(epoch, ppo_learner.policy, policy_opponent)

        def test_fn(epoch, env_step):
            pass

        # 6. Trener
        trainer_params = OnPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=self.steps_per_epoch,
            update_step_num_repetitions=4,
            batch_size=256,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=10,
            training_fn=train_fn,
            test_fn=test_fn,
            save_best_fn=self.save_best_model,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        print("Rozpoczęcie treningu PPO...")
        result = OnPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening PPO Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        torch.save(ppo_learner.policy.state_dict(), f'final_{self.training_phase}_ppo.pth')

if __name__ == "__main__":
    trainer = PPOPokerTrainer(
        algo_name="ppo",
        training_phase="RANDOM",
        evaluator_class=PPOEvaluator,
        num_train_envs=2, # dla colaba 8
        num_test_envs=1, # dla colaba 4
        max_epochs=100,
        steps_per_epoch=4096 # TODO: raczej trzeba zwiększyć
    )
    trainer.setup_and_train()