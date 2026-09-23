from tianshou.algorithm.random import MARLRandomDiscreteMaskedOffPolicyAlgorithm
from tianshou.algorithm.modelfree.ppo import PPO 
from tianshou.algorithm.modelfree.dqn import DQN 
from tianshou.data import SequenceSummaryStats
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats 
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats 
import numpy as np
from tianshou.algorithm.algorithm_base import Policy
from tianshou.data import Batch
import torch

class RandomOnPolicyAgent(MARLRandomDiscreteMaskedOffPolicyAlgorithm):
    """
    Adapter pozwalający użyć losowego agenta off-policy w środowisku on-policy.
    Przyjmuje dodatkowe argumenty i zwraca pustą statystykę.
    """
    def _update_with_batch(self, batch, batch_size=None, repeat=1):
        empty_stat = SequenceSummaryStats.from_sequence([0.0])
        return A2CTrainingStats(
            loss=empty_stat,
            actor_loss=empty_stat,
            vf_loss=empty_stat,
            ent_loss=empty_stat,
            gradient_steps=0
        )

import numpy as np
from tianshou.data import Batch
# Zaimportuj z właściwej lokalizacji w Twoim projekcie Tianshou (zazwyczaj tianshou.algorithm_base lub tianshou.algorithm.base)
from tianshou.algorithm.algorithm_base import Policy, OffPolicyAlgorithm, TrainingStats 

class TightHeuristicPolicy(Policy):
    """
    Polityka bota heurystycznego (Tight).
    Oczekuje przestrzeni akcji w konstruktorze.
    """
    def __init__(self, action_space):
        super().__init__(action_space=action_space)

    def forward(self, batch, state=None, **kwargs):
        actions = []
        
        # Bezpieczne wyciągnięcie tablic dla całej paczki naraz
        if isinstance(batch.obs, dict):
            observations = batch.obs["observation"]
            action_masks = batch.obs["action_mask"]
        else: # Tianshou Batch
            observations = batch.obs.observation
            action_masks = batch.obs.action_mask
            
        # Iterujemy po klasycznych tablicach NumPy/Torch
        for i in range(len(observations)):
            observation = observations[i]
            action_mask = action_masks[i]
            
            my_chips = observation[52]
            max_chips = observation[53]
            to_call = max_chips - my_chips
            
            legal_actions = np.where(action_mask == 1)[0]
            
            if to_call == 0:
                if 2 in legal_actions and np.random.rand() < 0.10:
                    act = 2
                elif 1 in legal_actions:
                    act = 1
                else:
                    act = np.random.choice(legal_actions)
            else:
                r = np.random.rand()
                if r < 0.75 and 0 in legal_actions:
                    act = 0
                elif r < 0.95 and 1 in legal_actions:
                    act = 1
                elif 2 in legal_actions:
                    act = 2
                elif 1 in legal_actions:
                    act = 1
                else:
                    act = np.random.choice(legal_actions)
                    
            if act not in legal_actions:
                act = np.random.choice(legal_actions)
                
            actions.append(act)
            
        from tianshou.data import Batch
        import numpy as np
        return Batch(act=np.array(actions), state=state)


class TightHeuristicAlgorithm(OffPolicyAlgorithm):
    """
    Opakowanie algorytmu do Tianshou. Przekazuje politykę i całkowicie wyłącza uczenie.
    """
    def __init__(self, action_space):
        policy = TightHeuristicPolicy(action_space=action_space)
        # Konstruktor bazowy Algorithm oczekuje argumentu policy[cite: 9]
        super().__init__(policy=policy)

    def _update_with_batch(self, batch):
        # Ta funkcja jest wywoływana podczas próby aktualizacji wag. 
        # Zwracamy pusty obiekt TrainingStats, więc bot niczego się nie uczy[cite: 9].
        return TrainingStats()

class FrozenPPO(PPO):
    """
    Zamrożona wersja algorytmu PPO.
    Całkowicie pomija krok aktualizacji wag (uczenia), co oszczędza zasoby.
    """
    def _update_with_batch(self, batch, batch_size, repeat):
        # Generujemy pustą statystykę dla sztucznej pojedynczej "wartości" zera
        empty_stat = SequenceSummaryStats.from_sequence([0.0])
        
        return A2CTrainingStats(
            loss=empty_stat,
            actor_loss=empty_stat,
            vf_loss=empty_stat,
            ent_loss=empty_stat,
            gradient_steps=0
        )

class FrozenDQN(DQN):
    """
    Zamrożona wersja algorytmu DQN. Pomija liczenie gradientów i aktualizację wag,
    co drastycznie oszczędza moc obliczeniową dla agentów-przeciwników.
    """
    def _update_with_batch(self, batch):
        return SimpleLossTrainingStats(loss=0.0)
