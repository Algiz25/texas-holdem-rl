"""DQN z lekkimi metrykami diagnostycznymi dla TensorBoard.

Standardowa implementacja Tianshou zwraca tylko loss. To nie wystarcza do
odróżnienia zdrowego uczenia od eksplozji wartości Q, dużych błędów TD lub
zapadnięcia polityki do jednej akcji. Klasa poniżej zachowuje identyczną regułę
aktualizacji DQN i jedynie zwraca dodatkowe, skalarne podsumowania batcha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import config
from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
from tianshou.data import to_torch_as
from tianshou.data.types import RolloutBatchProtocol


@dataclass(kw_only=True)
class DQNDiagnosticTrainingStats(SimpleLossTrainingStats):
    """Skalary wystarczające do diagnozy stabilności aktualizacji."""

    td_error_abs_mean: float
    td_error_abs_max: float
    q_selected_mean: float
    q_selected_min: float
    q_selected_max: float
    q_greedy_mean: float
    target_mean: float
    target_min: float
    target_max: float
    gradient_norm: float
    reward_mean: float
    reward_abs_max: float
    chip_reward_mean: float
    placement_reward_mean: float
    action_fold_fraction: float
    action_check_call_fraction: float
    action_raise_half_fraction: float
    action_raise_pot_fraction: float
    action_all_in_fraction: float


def _batch_info_mean(batch: RolloutBatchProtocol, field: str) -> float:
    """Odczytaj liczbową kolumnę ``info``; brak pola oznacza starszy replay."""
    info = getattr(batch, "info", None)
    if info is None or not hasattr(info, field):
        return 0.0
    values = np.asarray(getattr(info, field), dtype=np.float64)
    return float(values.mean()) if values.size else 0.0


class DiagnosticDQN(DQN):
    """Standardowy Double DQN wzbogacony wyłącznie o pomiary aktualizacji."""

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> DQNDiagnosticTrainingStats:
        # Kod aktualizacji odpowiada implementacji Tianshou. Diagnostyka nie
        # zmienia celu, kolejności target update ani sposobu liczenia gradientu.
        self._periodically_update_lagged_network_weights()
        weight = batch.pop("weight", 1.0)
        all_q_values = self.policy(batch).logits
        selected_q = all_q_values[np.arange(len(all_q_values)), batch.act]
        targets = to_torch_as(batch.returns.flatten(), selected_q)
        td_error = targets - selected_q

        if self.huber_loss_delta is not None:
            loss = torch.nn.functional.huber_loss(
                selected_q.reshape(-1, 1),
                targets.reshape(-1, 1),
                delta=self.huber_loss_delta,
                reduction="mean",
            )
        else:
            loss = (td_error.pow(2) * weight).mean()

        # Błąd TD staje się priorytetem po ewentualnym przejściu na PER.
        # Dla zwykłego replay buffera zachowuje zgodność ze standardowym DQN.
        batch.weight = td_error
        self.optim.step(loss)

        with torch.no_grad():
            td_abs = td_error.detach().abs()
            selected_q_detached = selected_q.detach()
            targets_detached = targets.detach()
            # Niedozwolone akcje mają logit -1e9, więc maksimum nadal wybiera
            # jedną z legalnych akcji obecnych w każdej obserwacji.
            greedy_q = all_q_values.detach().max(dim=1).values

            squared_gradient_norm = torch.zeros((), device=selected_q.device)
            for parameter in self.policy.parameters():
                if parameter.grad is not None:
                    squared_gradient_norm += parameter.grad.detach().pow(2).sum()
            gradient_norm = squared_gradient_norm.sqrt()

            actions = np.asarray(batch.act, dtype=np.int64)
            action_fractions = [
                float(np.mean(actions == action))
                for action in range(config.ACTION_SPACE)
            ]
            rewards = np.asarray(batch.rew, dtype=np.float64)

        return DQNDiagnosticTrainingStats(
            loss=float(loss.item()),
            td_error_abs_mean=float(td_abs.mean().item()),
            td_error_abs_max=float(td_abs.max().item()),
            q_selected_mean=float(selected_q_detached.mean().item()),
            q_selected_min=float(selected_q_detached.min().item()),
            q_selected_max=float(selected_q_detached.max().item()),
            q_greedy_mean=float(greedy_q.mean().item()),
            target_mean=float(targets_detached.mean().item()),
            target_min=float(targets_detached.min().item()),
            target_max=float(targets_detached.max().item()),
            gradient_norm=float(gradient_norm.item()),
            reward_mean=float(rewards.mean()),
            reward_abs_max=float(np.abs(rewards).max()),
            chip_reward_mean=_batch_info_mean(batch, "chip_reward"),
            placement_reward_mean=_batch_info_mean(batch, "placement_reward"),
            action_fold_fraction=action_fractions[0],
            action_check_call_fraction=action_fractions[1],
            action_raise_half_fraction=action_fractions[2],
            action_raise_pot_fraction=action_fractions[3],
            action_all_in_fraction=action_fractions[4],
        )
