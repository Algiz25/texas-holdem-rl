import torch
from torch.distributions import Categorical
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory

import config
from training.base_trainer import BasePokerTrainer
from evaluation.evaluator_ppo import PPOEvaluator
from models import MaskedActor, Critic, CPUActionActorPolicy
from phases import DynamicOpponentAlgorithm
from opponents import RandomOnPolicyAgent, FrozenPPO, PassiveAlgorithm, AggressiveAlgorithm, SeededMixedAlgorithm
from paths import PPO_CHECKPOINT_DIR

class PPOPokerTrainer(BasePokerTrainer):
    def setup_and_train(self):
        # Konfiguracja Ucznia
        actor_learner = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
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
                policy_learner.load_state_dict(torch.load(PPO_CHECKPOINT_DIR / 'final.pth', map_location=self.device, weights_only=True))
                print("Wczytano wagi ucznia z poprzedniej fazy!")
            except FileNotFoundError:
                print("Brak końcowego modelu PPO dla ucznia, start od zera.")
        
        ppo_learner = PPO(
            policy=policy_learner,
            critic=critic_learner,
            optim=AdamOptimizerFactory(lr=config.PPO_LEARNING_RATE), # TODO: przemyśleć tą wartość
            gamma=config.PPO_GAMMA, # TODO: zobaczyć czy lepiej nie ustawić 0.95
            gae_lambda=0.95,
            vf_coef=0.5,
            ent_coef=0.01,
            eps_clip=0.2,
            advantage_normalization=True
        )

        # Konfiguracja Przeciwników
        actor_opponent = MaskedActor(state_shape=self.observation_size, action_shape=config.ACTION_SPACE).to(self.device)
        critic_opponent = Critic(state_shape=self.observation_size).to(self.device)
        
        policy_opponent = CPUActionActorPolicy(
            actor=actor_opponent,
            dist_fn=dist_fn,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            action_scaling=False
        )

        # Oszczędza zasoby
        # TODO: trzeba to dodać dla dqn jeśli działa
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
            gamma=config.PPO_GAMMA,
            gae_lambda=0.95,
            max_grad_norm=0.0,
            vf_coef=0.0,
            ent_coef=0.0,
            eps_clip=0.2
        )

        if self.phase_name in {"self", "advanced"}:
            try:
                frozen_opponent.policy.load_state_dict(
                    torch.load(
                        PPO_CHECKPOINT_DIR / "best.pth",
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            except FileNotFoundError:
                pass

        # Inne agenty
        random_agent = RandomOnPolicyAgent(action_space=self.env.action_space)
        passive_agent = PassiveAlgorithm(action_space=self.env.action_space)
        aggressive_agent = AggressiveAlgorithm(action_space=self.env.action_space)
        mixed_agent = SeededMixedAlgorithm(action_space=self.env.action_space, seed=12345)

        # Złożenie algorytmu MARL

        if self.training_phase == 1:
            available_opponents = {
                "random": random_agent.policy,
                "passive": passive_agent.policy,
                "mixed": mixed_agent.policy
            }

            # Wspólna konfiguracja nie pozwala treningowi PPO i ewaluatorowi
            # nieświadomie używać różnych proporcji przeciwników fazy 1.
            opponent_weights = config.PHASE1_OPPONENT_WEIGHTS

            opponent_1 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=20_001)
            opponent_2 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=20_002)
            opponent_3 = DynamicOpponentAlgorithm(self.env.action_space, available_opponents, opponent_weights, seed=20_003)

            agents = [ppo_learner, opponent_1, opponent_2, opponent_3]
        # TODO: trzeba zdefiniować inne fazy
        else:
            print("Nie ma takiej fazy")

        marl_algo = MultiAgentOnPolicyAlgorithm(algorithms=agents, env=self.env)

        # Kolektory
        buffer = VectorReplayBuffer(config.PPO_BUFFER_SIZE, len(self.train_envs))
        train_collector = Collector(
            marl_algo, 
            self.train_envs, 
            buffer, 
            exploration_noise=True,
        )

        test_collector = Collector(
            marl_algo, 
            self.test_envs, 
            exploration_noise=False,
        )

        # Funkcje trenujące z logiką PPO
        latest_step = {"value": 0}

        def train_fn(epoch, env_step):
            latest_step["value"] = env_step
            self.run_periodic_evaluation(
                env_step=env_step,
                learner_policy=ppo_learner.policy,
                opponent_policy=policy_opponent,
            )

        def test_fn(epoch, env_step):
            pass

        # Trener
        trainer_params = OnPolicyTrainerParams(
            max_epochs=self.max_epochs,
            epoch_num_steps=self.steps_per_epoch,
            collection_step_num_env_steps=self.steps_per_epoch,
            update_step_num_repetitions=4,
            batch_size=config.PPO_BATCH_SIZE,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=config.EVAL_SMOKE_TOURNAMENTS,
            training_fn=train_fn,
            test_fn=test_fn,
            multi_agent_return_reduction=lambda ret: ret[:, 0]
        )

        self.run_initial_evaluation(learner_policy=ppo_learner.policy)
        print("Rozpoczęcie treningu PPO...")
        result = OnPolicyTrainer(algorithm=marl_algo, params=trainer_params).run()
        print(f"\n=== Trening PPO Zakończony ===\nNajlepsza nagroda: {result.best_reward}")
        self.run_final_evaluation(
            learner_policy=ppo_learner.policy,
            env_step=latest_step["value"] or self.total_steps,
        )

if __name__ == "__main__":
    trainer = PPOPokerTrainer(
        algo_name="ppo",
        training_phase=config.TRAINING_PHASE,
        evaluator_class=PPOEvaluator,
        num_train_envs=config.PPO_NUM_TRAIN_ENVS,
        num_test_envs=config.PPO_NUM_TEST_ENVS,
        max_epochs=config.PPO_MAX_EPOCHS,
        steps_per_epoch=config.PPO_STEPS_PER_EPOCH # TODO: raczej trzeba zwiększyć
    )
    trainer.setup_and_train()
