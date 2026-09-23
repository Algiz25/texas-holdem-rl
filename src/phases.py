import numpy as np
import gymnasium as gym
from typing import cast

from tianshou.algorithm.algorithm_base import OffPolicyAlgorithm, TrainingStats
from tianshou.algorithm.algorithm_base import Policy as BasePolicy
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.data.collector import EpisodeRolloutHook, EpisodeBatchProtocol
from tianshou.data import SequenceSummaryStats
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats

class DynamicOpponentTrainingStats(TrainingStats):
    pass

class DynamicOpponentPolicy(BasePolicy):
    def __init__(self, action_space: gym.spaces.Space, opponents: dict[str, BasePolicy], weights: dict[str, float]) -> None:
        super().__init__(action_space=action_space)
        self.opponents = opponents
        self.names = list(weights.keys())
        self.probs = list(weights.values())
        
        assert np.isclose(sum(self.probs), 1.0), "Wagi przeciwników muszą sumować się do 1.0"
        
        # losowanie
        self.current_name = np.random.choice(self.names, p=self.probs)
        self.current_policy = self.opponents[self.current_name]

    def new_tournament_reset(self):
        """Losuje nową osobowość bota i zwraca jej nazwę."""
        self.current_name = np.random.choice(self.names, p=self.probs)
        self.current_policy = self.opponents[self.current_name]
        return self.current_name

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: dict,
    ) -> ActBatchProtocol:
        return self.current_policy.forward(batch, state, **kwargs)

class DynamicOpponentAlgorithm(OffPolicyAlgorithm):
    def __init__(self, action_space: gym.spaces.Space, opponents: dict[str, BasePolicy], weights: dict[str, float]) -> None:
        dynamic_policy = DynamicOpponentPolicy(action_space, opponents, weights)
        super().__init__(policy=dynamic_policy)
        self.dynamic_policy = dynamic_policy

    def _update_with_batch(self, batch, *args, **kwargs): 
        # Zachowanie dla PPO
        if args or "repeat" in kwargs or "batch_size" in kwargs: 
            empty_stat = SequenceSummaryStats.from_sequence([0.0])
            return A2CTrainingStats(
                loss=empty_stat,
                actor_loss=empty_stat,
                vf_loss=empty_stat,
                ent_loss=empty_stat,
                gradient_steps=0
            )
        else:
            # Zachowanie dla DQN
            return DynamicOpponentTrainingStats()
        
    def reset_personality(self):
        return self.dynamic_policy.new_tournament_reset()

class ShuffleOpponentsHook(EpisodeRolloutHook):
    def __init__(self, opp1, opp2, opp3):
        self.opps = [opp1, opp2, opp3]
        self.tournament_count = 0

    def __call__(self, episode_batch: EpisodeBatchProtocol) -> dict | None:
        names = []
        for opp in self.opps:
            # Sprawdzamy czy agent ma funkcję reset_personality
            if hasattr(opp, 'reset_personality'):
                names.append(opp.reset_personality())
            else:
                names.append("STATIC")
                
        self.tournament_count += 1
        # debug
        # print(f"\n[Turniej {self.tournament_count}] P1: {names[0]} | P2: {names[1]} | P3: {names[2]}")
        
        return None