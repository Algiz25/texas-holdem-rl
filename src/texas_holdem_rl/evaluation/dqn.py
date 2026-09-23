import torch
from .base import BasePokerEvaluator
from ..models import MaskedActor
from ..paths import DQN_CHECKPOINT_DIR
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from .. import config

class DQNEvaluator(BasePokerEvaluator):
    def load_policy(self):
        net = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(self.device)
        policy = DiscreteQLearningPolicy(
            model=net,
            action_space=self.env.action_space("player_0"),
            observation_space=self.env.observation_space("player_0"),
            eps_inference=0.0 
        )
        
        try:
            policy.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
            print(f"Załadowano model DQN: {self.model_path}")
            policy.eval()
            return policy
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {self.model_path}.")
            return None

if __name__ == "__main__":
    dqn_eval = DQNEvaluator(num_tournaments=100, model_path=DQN_CHECKPOINT_DIR / 'best.pth', training_phase="RANDOM")
    dqn_eval.evaluate()
