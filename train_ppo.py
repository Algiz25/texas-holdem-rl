import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tianshou.data import Batch, Collector, VectorReplayBuffer
from tianshou.env import PettingZooEnv, SubprocVectorEnv
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.data import SequenceSummaryStats

from pettingzoo_tournament import TexasHoldemTournament
from masked_actor import MaskedActor, Critic, CPUActionActorPolicy
from utils import RandomOnPolicyAgent, FrozenPPO
from evaluator_ppo import PPOEvaluator
    
def save_best_model(algo):
    """Zapisuje wagi całego algorytmu (Actor + Critic)."""
    learner = algo.get_algorithm("player_0")
    torch.save(learner.policy.state_dict(), "best_ppo_poker.pth")
    print("\n[ZAPIS] Zapisano nowy najlepszy model PPO do 'best_ppo_poker.pth'")

if __name__ == "__main__":
    # ETAP NAUKI
    # dostępne: "RANDOM", "SELF", "ADVANCED"
    training_phase = "RANDOM"
    observation_size = 68

    # PPO jest bardziej stabilne przy większej liczbie zebranych trajektorii z różnych środowisk
    num_train_envs = 8
    num_test_envs = 4

    # learning rate
    lr = 3e-4 # Niższe LR dla PPO jest bezpieczniejsze

    print("Inicjalizacja środowisk dla PPO (Self-Play)...")
    def get_env():
        return PettingZooEnv(TexasHoldemTournament(num_players=4, starting_chips=200))

    env = get_env()

    train_envs = SubprocVectorEnv([get_env for _ in range(num_train_envs)])
    test_envs = SubprocVectorEnv([get_env for _ in range(num_test_envs)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Trening na urządzeniu: {device}")
    
    # Inicjalizacja sieci
    actor_learner = MaskedActor(state_shape=observation_size, action_shape=5).to(device)
    critic_learner = Critic(state_shape=observation_size).to(device)
    
    optim_factory_learner = AdamOptimizerFactory(lr=lr)

    # Definiujemy politykę probabilistyczną, w której z logitów tworzony jest rozkład Categorical
    def dist_fn(logits):
        return Categorical(logits=logits)

    policy_learner = CPUActionActorPolicy(
        actor=actor_learner,
        dist_fn=dist_fn,
        action_space=env.action_space,
        observation_space=env.observation_space,
        action_scaling=False
    )
    
    # Tworzenie algorytmu PPO
    ppo_learner = PPO(
        policy=policy_learner,
        critic=critic_learner,
        optim=optim_factory_learner,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        eps_clip=0.2, # Kluczowy parametr PPO
        advantage_normalization=True
    )

    random_agent = RandomOnPolicyAgent(action_space=env.action_space)

    actor_opponent = MaskedActor(state_shape=observation_size, action_shape=5).to(device)
    critic_opponent = Critic(state_shape=observation_size).to(device)

    policy_opponent = CPUActionActorPolicy(
        actor=actor_opponent,
        dist_fn=dist_fn,
        action_space=env.action_space,
        observation_space=env.observation_space,
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
        state_dict = torch.load('best_ppo_poker.pth', map_location=device, weights_only=True)
        frozen_opponent.policy.load_state_dict(state_dict)
        print("Pomyślnie załadowano 'best_ppo_poker.pth' jako początkową strategię przeciwników.")
    except FileNotFoundError:
        print("UWAGA: Nie znaleziono 'best_ppo_poker.pth'. Przeciwnicy użyją wag losowych z inicjalizacji.")

    if training_phase == "RANDOM":
        marl_algo = MultiAgentOnPolicyAlgorithm(
            algorithms=[
                ppo_learner, 
                random_agent, 
                random_agent, 
                random_agent
            ],
            env=env
        )
    elif training_phase == "SELF":
        marl_algo = MultiAgentOnPolicyAlgorithm(
            algorithms=[
                ppo_learner, 
                frozen_opponent,
                frozen_opponent, 
                frozen_opponent
            ],
            env=env
        )
    elif training_phase == "ADVANCED":
        marl_algo = MultiAgentOnPolicyAlgorithm(
            algorithms=[
                ppo_learner, 
                frozen_opponent,
                random_agent, 
                frozen_opponent
            ],
            env=env
        )

    # Dla PPO bufor powinien zazwyczaj mieścić kroki tylko z jednej iteracji zbierania.
    buffer_size = 2048
    buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    train_collector = Collector(marl_algo, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(marl_algo, test_envs, exploration_noise=False)

    last_opponent_update = 0

    def train_fn(epoch, env_step):
        global last_opponent_update
        if epoch > 0 and epoch % 10 == 0 and epoch != last_opponent_update:
            torch.save(ppo_learner.policy.state_dict(), f'{training_phase}_ppo_{epoch}.pth')
            ppo_eval = PPOEvaluator(num_tournaments=10, model_path=f'{training_phase}_ppo_{epoch}.pth')
            ppo_eval.evaluate()

            if (training_phase == "SELF" or training_phase == "ADVANCED"):
                try:
                    state_dict = torch.load(f'{training_phase}_ppo_{epoch}.pth', map_location=device, weights_only=True)
                    policy_opponent.load_state_dict(state_dict)
                    print(f"\n---> [EPOKA {epoch}] Przeciwnicy zaktualizowali wagi do najlepszego modelu PPO! <---")
                except FileNotFoundError:
                    print(f"\n---> [EPOKA {epoch}] Brak pliku best_dqn_poker.pth, wrogowie grają dalej starymi wagami. <---")
                last_opponent_update = epoch

    def test_fn(epoch, env_step):
        pass

    print("Rozpoczęcie treningu PPO...")
    
    trainer_params = OnPolicyTrainerParams(
        max_epochs=100,
        epoch_num_steps=4096,
        collection_step_num_env_steps=2048, # Ile kroków zbiera przed aktualizacją wag
        update_step_num_repetitions=4,      # Ile razy przetwarza zebrany bufor (mini-epoki w PPO)
        batch_size=256,
        training_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=10,
        training_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_model,
        multi_agent_return_reduction=lambda returns: returns[:, 0]
    )

    trainer = OnPolicyTrainer(
        algorithm=marl_algo,
        params=trainer_params
    )
    
    result = trainer.run()

    print("\n=== Trening PPO Zakończony ===")
    print(f"Najlepsza nagroda: {result.best_reward}")
    torch.save(ppo_learner.policy.state_dict(), 'final_ppo_agent.pth')