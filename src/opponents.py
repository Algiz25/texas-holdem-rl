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
import gymnasium as gym
import config
from typing import cast
from tianshou.algorithm.algorithm_base import OffPolicyAlgorithm, TrainingStats
from tianshou.algorithm.algorithm_base import Policy as BasePolicy
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol

class PassiveTrainingStats(TrainingStats):
    pass

class PassivePolicy(BasePolicy):
    """
    Polityka bota pasywnego. Zawsze wybiera akcje w kolejności:
    1. Check/Call (indeks 1)
    2. Fold (indeks 0)
    3. Pierwsza dostępna legalna akcja
    """
    def __init__(self, action_space: gym.spaces.Space) -> None:
        super().__init__(action_space=action_space)

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: dict,
    ) -> ActBatchProtocol:
        
        # Bezpieczne wyciągnięcie maski
        if hasattr(batch.obs, "action_mask"):
            action_masks = batch.obs.action_mask
        elif hasattr(batch.obs, "mask"):
            action_masks = batch.obs.mask
        elif isinstance(batch.obs, dict) and "action_mask" in batch.obs:
            action_masks = batch.obs["action_mask"]
        else:
            raise ValueError("Nie znaleziono maski akcji w batch.obs")

        # Konwersja na NumPy w przypadku otrzymania tensora z PyTorcha
        if isinstance(action_masks, torch.Tensor):
            action_masks = action_masks.cpu().numpy()

        actions = []
        # Iteracja po wszystkich obserwacjach w batchu
        for mask in action_masks:
            legal_actions = np.where(mask)[0]
            
            if 1 in legal_actions:
                act = 1
            elif 0 in legal_actions:
                act = 0
            else:
                act = legal_actions[0] if len(legal_actions) > 0 else 0

            actions.append(act)

        # Pakowanie wyników z powrotem do formatu oczekiwanego przez Tianshou
        result = Batch(act=np.array(actions, dtype=np.int64), state=state)
        return cast(ActBatchProtocol, result)

class PassiveAlgorithm(OffPolicyAlgorithm):
    """
    Opakowanie polityki pasywnej na algorytm Off-Policy.
    """
    def __init__(self, action_space: gym.spaces.Space) -> None:
        super().__init__(policy=PassivePolicy(action_space))

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> PassiveTrainingStats: 
        return PassiveTrainingStats()

class AggressiveTrainingStats(TrainingStats):
    pass

class AggressivePolicy(BasePolicy):
    """
    Polityka bota agresywnego. Losuje akcje według wag:
    Fold (0): 5%
    Check/Call (1): 15%
    Small Raise (2): 30%
    Big Raise (3): 40%
    All-in (4): 10%
    """
    def __init__(self, action_space: gym.spaces.Space) -> None:
        super().__init__(action_space=action_space)
        self.base_weights = np.array([0.05, 0.15, 0.30, 0.40, 0.10], dtype=np.float32)

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: dict,
    ) -> ActBatchProtocol:
        
        # Ekstrakcja maski akcji
        if hasattr(batch.obs, "action_mask"):
            action_masks = batch.obs.action_mask
        elif hasattr(batch.obs, "mask"):
            action_masks = batch.obs.mask
        elif isinstance(batch.obs, dict) and "action_mask" in batch.obs:
            action_masks = batch.obs["action_mask"]
        else:
            raise ValueError("Nie znaleziono maski akcji w batch.obs")

        if isinstance(action_masks, torch.Tensor):
            action_masks = action_masks.cpu().numpy()

        actions = []
        for mask in action_masks:
            # Zerowanie prawdopodobieństw dla niedozwolonych akcji
            valid_weights = self.base_weights * mask
            weight_sum = np.sum(valid_weights)
            
            if weight_sum > 0:
                # Normalizacja wag, aby sumowały się do 1.0 (wymóg np.random.choice)
                normalized_probs = valid_weights / weight_sum
                act = np.random.choice(5, p=normalized_probs)
            else:
                # Fallback awaryjny - pierwsza legalna akcja (nie powinien nigdy wystąpić w pokerze)
                legal_actions = np.where(mask)[0]
                act = legal_actions[0] if len(legal_actions) > 0 else 0
                
            actions.append(act)

        result = Batch(act=np.array(actions, dtype=np.int64), state=state)
        return cast(ActBatchProtocol, result)

class AggressiveAlgorithm(OffPolicyAlgorithm):
    def __init__(self, action_space: gym.spaces.Space) -> None:
        super().__init__(policy=AggressivePolicy(action_space))

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> AggressiveTrainingStats: 
        return AggressiveTrainingStats()

class SeededMixedTrainingStats(TrainingStats):
    pass

class SeededMixedPolicy(BasePolicy):
    """
    Polityka bota zrównoważonego z powtarzalnym ziarnem losowości (seed).
    Losuje akcje według bazowych wag:
    Fold (0): 20%
    Check/Call (1): 45%
    Small Raise (2): 18%
    Big Raise (3): 12%
    All-in (4): 5%
    """
    def __init__(self, action_space: gym.spaces.Space, seed: int = 42) -> None:
        super().__init__(action_space=action_space)
        # Generator dla powtarzalności zachowań
        self.rng = np.random.default_rng(seed)
        self.base_weights = np.asarray(
            config.MIXED_ACTION_WEIGHTS,
            dtype=np.float32,
        )

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: dict,
    ) -> ActBatchProtocol:
        
        # Ekstrakcja maski akcji
        if hasattr(batch.obs, "action_mask"):
            action_masks = batch.obs.action_mask
        elif hasattr(batch.obs, "mask"):
            action_masks = batch.obs.mask
        elif isinstance(batch.obs, dict) and "action_mask" in batch.obs:
            action_masks = batch.obs["action_mask"]
        else:
            raise ValueError("Nie znaleziono maski akcji w batch.obs")

        if isinstance(action_masks, torch.Tensor):
            action_masks = action_masks.cpu().numpy()

        actions = []
        for mask in action_masks:
            # Zerowanie prawdopodobieństw dla niedozwolonych akcji
            valid_weights = self.base_weights * mask
            weight_sum = np.sum(valid_weights)
            
            if weight_sum > 0:
                # Normalizacja wag, aby sumowały się do 1.0 dla akcji legalnych
                normalized_probs = valid_weights / weight_sum
                act = self.rng.choice(5, p=normalized_probs)
            else:
                # Fallback awaryjny - pierwsza legalna akcja
                legal_actions = np.where(mask)[0]
                act = legal_actions[0] if len(legal_actions) > 0 else 0
                
            actions.append(act)

        result = Batch(act=np.array(actions, dtype=np.int64), state=state)
        return cast(ActBatchProtocol, result)

class SeededMixedAlgorithm(OffPolicyAlgorithm):
    def __init__(self, action_space: gym.spaces.Space, seed: int = 42) -> None:
        super().__init__(policy=SeededMixedPolicy(action_space, seed=seed))

    def _update_with_batch(self, batch: RolloutBatchProtocol) -> SeededMixedTrainingStats: 
        return SeededMixedTrainingStats()

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
