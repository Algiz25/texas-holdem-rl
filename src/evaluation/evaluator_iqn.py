import torch
from evaluation.evaluator import BasePokerEvaluator
from models import PokerFeatureExtractor
from paths import IQN_CHECKPOINT_DIR
from tianshou.algorithm.modelfree.iqn import IQNPolicy
from tianshou.utils.net.discrete import ImplicitQuantileNetwork
import config

class IQNEvaluator(BasePokerEvaluator):
    algorithm_name = "iqn"

    def load_policy(self, path=None):
        if path is None:
            path = self.model_path

        feature_net = PokerFeatureExtractor(state_shape=config.OBSERVATION_SIZE).to(self.device)
        net = ImplicitQuantileNetwork(
            preprocess_net=feature_net,
            action_shape=config.ACTION_SPACE,
            num_cosines=config.IQN_NUM_COSINES,
        ).to(self.device)

        policy = IQNPolicy(
            model=net,
            action_space=self.env.action_space("player_0"),
            sample_size=config.IQN_SAMPLE_SIZE,
            online_sample_size=config.IQN_ONLINE_SAMPLE_SIZE,
            target_sample_size=config.IQN_TARGET_SAMPLE_SIZE,
            eps_training=0.0,
            eps_inference=0.0 
        )
        
        try:
            policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"Załadowano model IQN: {path}")
            policy.eval()
            return policy
        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku {path}.")
            return None
        except RuntimeError as error:
            print(
                f"BŁĄD: Checkpoint {path} nie pasuje do aktualnej "
                f"architektury ({config.OBSERVATION_SIZE} obserwacje): {error}"
            )
            return None

if __name__ == "__main__":
    iqn_eval = IQNEvaluator(
        num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
        model_path=IQN_CHECKPOINT_DIR / "best.pth",
    )
    iqn_eval.evaluate()