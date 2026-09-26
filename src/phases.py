import numpy as np
import gymnasium as gym
import torch
from collections import OrderedDict
from typing import cast

from tianshou.algorithm.algorithm_base import OffPolicyAlgorithm, TrainingStats
from tianshou.algorithm.algorithm_base import Policy as BasePolicy
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.data import SequenceSummaryStats
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats

class DynamicOpponentTrainingStats(TrainingStats):
    pass

class DynamicOpponentPolicy(BasePolicy):
    def __init__(
        self,
        action_space: gym.spaces.Space,
        opponents: dict[str, BasePolicy],
        weights: dict[str, float],
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(action_space=action_space)
        self.opponents = opponents
        self.names = list(weights.keys())
        self.probs = list(weights.values())

        assert np.isclose(sum(self.probs), 1.0), "Wagi przeciwników muszą sumować się do 1.0"

        self.rng = np.random.default_rng(seed)
        # Osobowość jest przypisana do identyfikatora turnieju, a nie globalnie
        # do polityki. To konieczne przy wielu środowiskach: zakończenie gry w
        # procesie A nie może zmienić bota w trwającym turnieju procesu B.
        self._personality_by_tournament: OrderedDict[str, str] = OrderedDict()
        self._max_remembered_tournaments = 10_000

    def personality_for(self, tournament_id: str) -> str:
        """Zwróć stałą osobowość bota dla jednego turnieju."""
        if tournament_id not in self._personality_by_tournament:
            self._personality_by_tournament[tournament_id] = str(
                self.rng.choice(self.names, p=self.probs)
            )
            # Stare turnieje nie wracają. Limit zapobiega niepotrzebnemu
            # wzrostowi pamięci podczas wielomilionowych treningów.
            if len(self._personality_by_tournament) > self._max_remembered_tournaments:
                self._personality_by_tournament.popitem(last=False)
        return self._personality_by_tournament[tournament_id]

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: dict,
    ) -> ActBatchProtocol:
        tournament_ids = self._extract_tournament_ids(batch)
        actions = np.empty(len(tournament_ids), dtype=np.int64)

        # Grupowanie pozwala każdej statycznej polityce obsłużyć naraz tylko te
        # środowiska, w których została wylosowana na początku turnieju.
        personalities = np.array(
            [self.personality_for(tournament_id) for tournament_id in tournament_ids]
        )
        for name, policy in self.opponents.items():
            indices = np.flatnonzero(personalities == name)
            if not len(indices):
                continue
            output = policy.forward(batch[indices], state=None, **kwargs)
            policy_actions = output.act
            if isinstance(policy_actions, torch.Tensor):
                policy_actions = policy_actions.detach().cpu().numpy()
            actions[indices] = np.asarray(policy_actions, dtype=np.int64)

        result = Batch(act=actions, state=state)
        return cast(ActBatchProtocol, result)

    @staticmethod
    def _extract_tournament_ids(batch: ObsBatchProtocol) -> list[str]:
        """Odczytaj identyfikatory przekazywane w `info` przez środowisko."""
        if not hasattr(batch, "info") or not hasattr(batch.info, "tournament_id"):
            raise ValueError(
                "Dynamiczny przeciwnik wymaga tournament_id w informacji środowiska"
            )
        values = np.atleast_1d(batch.info.tournament_id)
        return [str(value) for value in values]

class DynamicOpponentAlgorithm(OffPolicyAlgorithm):
    def __init__(
        self,
        action_space: gym.spaces.Space,
        opponents: dict[str, BasePolicy],
        weights: dict[str, float],
        *,
        seed: int | None = None,
    ) -> None:
        dynamic_policy = DynamicOpponentPolicy(
            action_space,
            opponents,
            weights,
            seed=seed,
        )
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
