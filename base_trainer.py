import torch
from tianshou.env import PettingZooEnv, SubprocVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from pettingzoo_tournament import TexasHoldemTournament

# Wyciągamy funkcję poza klasę - dzięki temu procesy Windowsa
# mogą ją bez problemu zserializować (spicklować).
def make_poker_env():
    return PettingZooEnv(TexasHoldemTournament(num_players=4, starting_chips=200))

class BasePokerTrainer:
    def __init__(self, algo_name, training_phase, evaluator_class, 
                 num_train_envs, num_test_envs, max_epochs, steps_per_epoch):
        self.algo_name = algo_name
        self.training_phase = training_phase
        self.evaluator_class = evaluator_class
        
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = max_epochs * steps_per_epoch
        self.observation_size = 68
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_opponent_update = 0

        print(f"Inicjalizacja środowisk PettingZoo dla algorytmu {self.algo_name.upper()}...")
        self.env = make_poker_env()
        self.train_envs = SubprocVectorEnv([make_poker_env for _ in range(num_train_envs)])
        self.test_envs = SubprocVectorEnv([make_poker_env for _ in range(num_test_envs)])

    def save_best_model(self, algo):
        """Zapisuje model ucznia, gdy testy wykażą najwyższą średnią nagrodę."""
        learner = algo.get_algorithm("player_0")
        model_name = f"best_{self.training_phase}_{self.algo_name}.pth"
        torch.save(learner.policy.state_dict(), model_name)
        print(f"\n[ZAPIS] Zapisano nowy najlepszy model do '{model_name}'")

    def run_periodic_opponent_update(self, epoch, learner_policy, opponent_policy):
        """Cykliczna ewaluacja i nadpisywanie wag przeciwników co 10 epok."""
        if epoch > 0 and epoch % 10 == 0 and epoch != self.last_opponent_update:
            model_name = f'{self.training_phase}_{self.algo_name}_{epoch}.pth'
            torch.save(learner_policy.state_dict(), model_name)
            
            # Ewaluacja
            evaluator = self.evaluator_class(num_tournaments=10, model_path=model_name)
            evaluator.evaluate()

            # Aktualizacja przeciwników w trybach zaawansowanych
            if self.training_phase in ["SELF", "ADVANCED"]:
                try:
                    opponent_policy.load_state_dict(
                        torch.load(model_name, map_location=self.device, weights_only=True)
                    )
                    print(f"\n---> [EPOKA {epoch}] Przeciwnicy zaktualizowali wagi! <---")
                except FileNotFoundError:
                    print(f"\n---> [EPOKA {epoch}] Brak pliku, wrogowie grają dalej starymi wagami. <---")
            
            self.last_opponent_update = epoch

    def setup_and_train(self):
        """Metoda abstrakcyjna - musi być zaimplementowana przez podklasy."""
        raise NotImplementedError