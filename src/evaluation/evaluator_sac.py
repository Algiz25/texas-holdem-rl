import torch
from evaluation.evaluator import BasePokerEvaluator
from models import MaskedActor
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSACPolicy
from paths import SAC_CHECKPOINT_DIR
import config

class SACEvaluator(BasePokerEvaluator):
    algorithm_name = "sac"

    def load_policy(self, path=None):
        if path is None:
            path = self.model_path

        actor = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(self.device)

        policy = DiscreteSACPolicy(
            actor=actor,
            action_space=self.env.action_space("player_0"),
            observation_space=self.env.observation_space("player_0"),
            deterministic_eval=(config.TRAINING_PHASE == 1), # Podczas ewaluacji wybieramy też z rozkładu bo inaczej by nie blefował (w pierwszej fazie niech tego nie robi)
        )
        
        try:
            policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            print(f"Załadowano model SAC: {path}")
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
    sac_eval = SACEvaluator(
        num_tournaments=config.FINAL_EVAL_TOURNAMENTS_PER_SUITE,
        model_path=SAC_CHECKPOINT_DIR / "best.pth",
    )
    sac_eval.evaluate()