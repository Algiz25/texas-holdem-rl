import numpy as np
import torch
import torch.nn as nn
import tianshou as ts
from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import PettingZooEnv, DummyVectorEnv, SubprocVectorEnv

from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer
from tianshou.trainer import OffPolicyTrainerParams

from pettingzoo_tournament import TexasHoldemTournament
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
from masked_actor import MaskedActor
from utils import FrozenDQN
from evaluator_dqn import DQNEvaluator
from utils import TightHeuristicAlgorithm


def save_best_model(algo):
    """Zapisuje model, gdy testy wykażą najwyższą średnią nagrodę."""
    learner = algo.get_algorithm("player_0")
    torch.save(learner.policy.state_dict(), "best_dqn_poker.pth")
    print("\n[ZAPIS] Zapisano nowy najlepszy model do 'best_dqn_poker.pth'")

if __name__ == "__main__":
    # ETAP NAUKI
    # dostępne: "RANDOM", "SELF", "ADVANCED"
    training_phase = "RANDOM"
    observation_size = 68

    # dla colaba 8-4
    num_train_envs = 2
    num_test_envs = 1

    # bardzo ważne
    MAX_EPOCHS = 100
    STEPS_PER_EPOCH = 10_000
    TOTAL_STEPS = MAX_EPOCHS * STEPS_PER_EPOCH

    #learning rate
    lr = 1e-4

    print("Inicjalizacja środowisk PettingZoo w Tianshou v2.0.1...")
    def get_env():
        return PettingZooEnv(TexasHoldemTournament(num_players=4, starting_chips=200))

    env = get_env()

    train_envs = SubprocVectorEnv([get_env for _ in range(num_train_envs)])
    test_envs = SubprocVectorEnv([get_env for _ in range(num_test_envs)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Trening na urządzeniu: {device}")
    
    net_learner = MaskedActor(state_shape=observation_size, action_shape=5).to(device) # rozmiar obserwacji
    
    optim_factory_learner = AdamOptimizerFactory(lr=lr)

    policy_learner = DiscreteQLearningPolicy(
        model=net_learner,
        action_space=env.action_space,  # output sieci
        observation_space=env.observation_space,    #input sieci
        eps_training=1.0,   # losowość podczas treningu
        eps_inference=0.0
    )

    if training_phase in ["SELF", "ADVANCED"]:
        try:
            policy_learner.load_state_dict(torch.load('dqn_agent_1.pth', map_location=device, weights_only=True))
            print("Wczytano wagi ucznia z poprzedniej fazy!")
        except FileNotFoundError:
            print("Brak pliku dqn_agent_1.pth dla ucznia, start od zera.")
    
    dqn_learner = DQN(
        policy=policy_learner,
        optim=optim_factory_learner,
        gamma=0.99, # woli długoterminowe zagrania - jeśli zbyt pasywny to obniżyć
        n_step_return_horizon=3, # ile ruchów do przodu patrzy
        target_update_freq=5000 # co ile update drugiej tablicy
    )

    # --- 2. KONFIGURACJA ZAMROŻONYCH PRZECIWNIKÓW (Player 1, 2, 3) ---
    net_opponent = MaskedActor(state_shape=observation_size, action_shape=5).to(device)
    policy_opponent = DiscreteQLearningPolicy(
        model=net_opponent,
        action_space=env.action_space,
        observation_space=env.observation_space,
        eps_training=0.05, # Zostawiamy 5% losowości, żeby wróg nie był zawsze w 100% przewidywalny
        eps_inference=0.0
    )

    # Wczytywanie modelu
    try:
        policy_opponent.load_state_dict(torch.load('dqn_agent_1.pth', map_location=device, weights_only=True))
        print("Pomyślnie załadowano dqn_agent_1.pth jako strategię dla przeciwników!")
    except FileNotFoundError:
        print("UWAGA: Nie znaleziono dqn_agent_1.pth. Przeciwnicy zagrają losowymi wagami z inicjalizacji.")

    frozen_opponent = FrozenDQN(
        policy=policy_opponent,
        optim=AdamOptimizerFactory(lr=0.0), # Nie będzie używane
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=0
    )

    #TODO fix heuristic agent
    random_agent = MARLRandomDiscreteMaskedOffPolicyAlgorithm(action_space=env.action_space)
    # heuristic_agent = TightHeuristicAlgorithm(action_space=env.action_space)

    if training_phase == "RANDOM":
        marl_algo = MultiAgentOffPolicyAlgorithm(
            algorithms=[
                dqn_learner,
                random_agent,
                random_agent,
                random_agent
            ],
            env=env
        )
    elif training_phase == "SELF":
        marl_algo = MultiAgentOffPolicyAlgorithm(
            algorithms=[
                dqn_learner,
                frozen_opponent,
                frozen_opponent,
                frozen_opponent
            ],
            env=env
        )
    elif training_phase == "ADVANCED":
        marl_algo = MultiAgentOffPolicyAlgorithm(
            algorithms=[
                dqn_learner,
                frozen_opponent,
                random_agent,
                frozen_opponent
            ],
            env=env
        )

    buffer = VectorReplayBuffer(
        100_000,  # ilość wspomnień (docelowo 500_000)
        len(train_envs) # dla ilu env
    )
    train_collector = Collector(marl_algo, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(marl_algo, test_envs, exploration_noise=False)

    print("Zapełnianie bufora pierwszymi losowymi danymi...")
    train_collector.collect(n_step=10_000, random=True, reset_before_collect=True)

    last_opponent_update = 0

    def train_fn(epoch, env_step):
        if training_phase == "RANDOM":
            # Od 1.0 do 0.1 przez 80% czasu
            eps = max(0.1, 1.0 - env_step / (0.8 * TOTAL_STEPS))
        else:
            # W fazach SELF/ADVANCED agent już umie grać, zaczynamy od niskiego epsilona (np. 0.2)
            eps = max(0.02, 0.2 - env_step / (0.5 * TOTAL_STEPS))
            
        dqn_learner.policy.set_eps_training(eps)

        global last_opponent_update
        if epoch > 0 and epoch % 10 == 0 and epoch != last_opponent_update:
            torch.save(dqn_learner.policy.state_dict(), f'{training_phase}_dqn_{epoch}.pth')
            dqn_eval = DQNEvaluator(num_tournaments=10, model_path=f'{training_phase}_dqn_{epoch}.pth')
            dqn_eval.evaluate()

            if (training_phase == "SELF" or training_phase == "ADVANCED"):
                try:
                    policy_opponent.load_state_dict(
                        torch.load(f'{training_phase}_dqn_{epoch}.pth', map_location=device, weights_only=True)    # BEST albo ten zapisany
                    )
                    print(f"\n---> [EPOKA {epoch}] Przeciwnicy zaktualizowali wagi do modelu z aktualnej epoki! <---")
                except FileNotFoundError:
                    print(f"\n---> [EPOKA {epoch}] Brak pliku best_dqn_poker.pth, wrogowie grają dalej starymi wagami. <---")
                last_opponent_update = epoch

    def test_fn(epoch, env_step):
        dqn_learner.policy.set_eps_inference(0.0)

    print("Rozpoczęcie treningu...")
    
    trainer_params = OffPolicyTrainerParams(
        max_epochs=MAX_EPOCHS,
        epoch_num_steps=STEPS_PER_EPOCH,
        training_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=10,
        batch_size=64,
        training_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_model,
        multi_agent_return_reduction=lambda ret: ret[:, 0]
    )

    trainer = OffPolicyTrainer(
        algorithm=marl_algo,
        params=trainer_params
    )
    
    result = trainer.run()

    print("\n=== Trening Zakończony ===")
    print(f"Najlepsza nagroda z testów (dla wszystkich agentów): {result.best_reward}")

    torch.save(dqn_learner.policy.state_dict(), 'dqn_agent_1.pth')
    print("Wytrenowany model DQN został zapisany jako 'dqn_agent_1.pth'")