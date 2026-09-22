import torch
from evaluator import BasePokerEvaluator
from masked_actor import MaskedActor, Critic, CPUActionActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from torch.distributions import Categorical
from tianshou.algorithm.modelfree.ppo import PPO
import config

class PPOEvaluator(BasePokerEvaluator):
    def load_policy(self):
        actor = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(self.device)
        critic = Critic(state_shape=config.OBSERVATION_SIZE).to(self.device)
        optim_factory = AdamOptimizerFactory(lr=config.PPO_LEARNING_RATE)

        def dist_fn(logits):
            return Categorical(logits=logits)

        policy = CPUActionActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_space=self.env.action_space("player_0"),
            observation_space=self.env.observation_space("player_0"),
            action_scaling=False
        )
        
        ppo_algo = PPO(
            policy=policy,
            critic=critic,
            optim=optim_factory,
            gamma=config.PPO_GAMMA,
            gae_lambda=0.95,
            vf_coef=0.5,
            ent_coef=0.05,
            eps_clip=0.2,
            advantage_normalization=True
        )
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
            ppo_algo.policy.load_state_dict(state_dict)
            print(f"Załadowano model PPO: {self.model_path}")
            ppo_algo.eval()
            return ppo_algo.policy
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {self.model_path}.")
            return None

if __name__ == "__main__":
    ppo_eval = PPOEvaluator(num_tournaments=100, model_path='best_ppo_poker.pth', training_phase="RANDOM")
    ppo_eval.evaluate()