import torch
from evaluation.evaluator import BasePokerEvaluator
from models import MaskedActor, Critic
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from paths import PPO_CHECKPOINT_DIR
from tianshou.algorithm.optim import AdamOptimizerFactory
from torch.distributions import Categorical
from tianshou.algorithm.modelfree.ppo import PPO
import config

class PPOEvaluator(BasePokerEvaluator):
    algorithm_name = "ppo"

    def load_policy(self, path=None):
        if path is None:
            path = self.model_path

        actor = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(self.device)

        def dist_fn(logits):
            return Categorical(logits=logits)

        policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=dist_fn,
            action_space=self.env.action_space("player_0"),
            observation_space=self.env.observation_space("player_0"),
            action_scaling=False,
            # Podczas ewaluacji wybieramy próbkę z
            # rozkładu PPO, BO INACZEJ NIE BLEFUJE. SZKODA, ŻE DOPIERO PO 20 GODZINACH TRENINGU TO ZNALAZLEM. GG
            # tylko w pierwszej fazie nie musi blefować
            deterministic_eval=(config.TRAINING_PHASE == 1),
        )
        
        try:
            policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"Załadowano model PPO: {self.model_path}")
            policy.eval()
            return policy
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {self.model_path}.")
            return None
        except RuntimeError as error:
            print(
                f"BŁĄD: Checkpoint {self.model_path} nie pasuje do aktualnej "
                f"architektury ({config.OBSERVATION_SIZE} obserwacje): {error}"
            )
            return None

if __name__ == "__main__":
    ppo_eval = PPOEvaluator(
        num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
        model_path=PPO_CHECKPOINT_DIR / "phase2_selfplay_seed_11001" / "final.pth",
    )
    ppo_eval.evaluate()
