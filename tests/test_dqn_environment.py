"""Regresje dla jednoagentowej ścieżki zbierania doświadczeń DQN."""

import json
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.torch_utils import policy_within_training_step

import config
from models import MaskedActor
from training.diagnostic_dqn import DiagnosticDQN
from training.learner_environment import PokerLearnerEnv, make_learner_env
from training.train_dqn import (
    DQNPokerTrainer,
    phase_two_profile,
    set_optimizer_learning_rate,
)


class DQNLearnerEnvironmentTests(unittest.TestCase):
    def test_self_play_league_manifest_is_valid_json(self) -> None:
        """Start fazy 2 musi móc zapisać ligę przed pierwszym resetem.

        Ten mały test celowo przechodzi przez rzeczywistą metodę zapisu. Dzięki
        temu brak importu ``json`` albo uszkodzony zapis atomowy zostaną wykryte
        przed uruchomieniem wielogodzinnego treningu.
        """
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            model_path = checkpoint_dir / "step_000000000.pth"
            model_path.touch()

            trainer = DQNPokerTrainer.__new__(DQNPokerTrainer)
            trainer.checkpoint_dir = checkpoint_dir
            trainer.self_play_pool_paths = [model_path]
            trainer.self_play_runtime_config = phase_two_profile("ppo_style")
            trainer.max_historical_models = 20
            trainer._write_league_manifest()

            manifest = json.loads(
                (checkpoint_dir / "self_play_pool.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(manifest["profile"], "ppo_style")
            self.assertEqual(manifest["retention"], "anchored_rotation")
            self.assertEqual(manifest["models"], [str(model_path.resolve())])

    def test_self_play_models_reach_subprocess_workers_before_reset(self) -> None:
        """Prawdziwa komunikacja między trenerem i workerami musi działać."""
        weights = MaskedActor().state_dict()
        phase_two_weights = {
            "historical_self": 0.40,
            "latest_self": 0.10,
            "mixed": 0.25,
            "random": 0.15,
            "passive": 0.10,
        }
        environments = SubprocVectorEnv(
            [
                partial(
                    make_learner_env,
                    initial_seed=91_000 + worker,
                    training_phase=2,
                    self_play_policy_kind="dqn",
                    opponent_weights=phase_two_weights,
                )
                for worker in range(2)
            ]
        )
        try:
            environments.set_env_attr("latest_model_weights", weights)
            environments.set_env_attr("new_historical_model_weights", weights)
            observations, infos = environments.reset()
            self.assertEqual(len(observations), 2)
            for observation in observations:
                self.assertEqual(
                    observation["obs"].shape,
                    (config.OBSERVATION_SIZE,),
                )
                self.assertEqual(
                    observation["mask"].shape,
                    (config.ACTION_SPACE,),
                )
            self.assertEqual(len(infos), 2)
        finally:
            environments.close()

    def test_self_play_dqn_uses_masked_q_argmax(self) -> None:
        """DQN zwraca wartości Q, więc nie wolno próbkować ich jak PPO."""

        class FixedValues(torch.nn.Module):
            def forward(self, observation):
                values = torch.tensor(
                    [[1.0, 9.0, 4.0, 3.0, 2.0]],
                    dtype=torch.float32,
                )
                return values, None

        env = PokerLearnerEnv()
        # Akcja 1 ma najwyższą wartość, ale jest niedozwolona. Adapter
        # powinien wybrać legalną akcję 2 o następnej najwyższej Q.
        action = env._get_nn_action(
            FixedValues(),
            np.zeros(config.OBSERVATION_SIZE, dtype=np.float32),
            np.asarray([1, 0, 1, 1, 1], dtype=np.int8),
        )
        self.assertEqual(action, 2)
        env.close()

    def test_self_play_dqn_opponent_epsilon_samples_only_legal_actions(self) -> None:
        """Eksplorujący rywal DQN nie może ominąć maski legalnych ruchów."""

        class FixedValues(torch.nn.Module):
            def forward(self, observation):
                return torch.zeros((1, config.ACTION_SPACE)), None

        env = PokerLearnerEnv(self_play_opponent_epsilon=1.0)
        env._rng = np.random.default_rng(123)
        mask = np.asarray([1, 0, 1, 0, 1], dtype=np.int8)
        actions = {
            env._get_nn_action(
                FixedValues(),
                np.zeros(config.OBSERVATION_SIZE, dtype=np.float32),
                mask,
            )
            for _ in range(50)
        }
        self.assertTrue(actions.issubset({0, 2, 4}))
        self.assertGreater(len(actions), 1)
        env.close()

    def test_self_play_opponent_is_frozen_for_the_whole_tournament(self) -> None:
        """Nowy snapshot nie może zmienić rywala w trwającym rozdaniu."""
        env = PokerLearnerEnv(
            training_phase=2,
            opponent_weights={
                "historical_self": 1.0,
                "latest_self": 0.0,
                "mixed": 0.0,
                "random": 0.0,
                "passive": 0.0,
            },
        )
        first = MaskedActor().state_dict()
        second = MaskedActor().state_dict()
        env.new_historical_model_weights = first
        env.reset(seed=17)

        assigned_before = {
            player: id(model) for player, model in env.opponent_models.items()
        }
        self.assertEqual(len(assigned_before), config.NUM_PLAYERS - 1)

        env.new_historical_model_weights = second
        assigned_after = {
            player: id(model) for player, model in env.opponent_models.items()
        }
        self.assertEqual(assigned_before, assigned_after)
        env.close()

    def test_self_play_history_pool_is_bounded_fifo(self) -> None:
        """Każdy worker przechowuje tylko ostatnie zamrożone modele ligi."""
        env = PokerLearnerEnv(training_phase=2)
        env.max_historical = 2
        for _ in range(3):
            env.new_historical_model_weights = MaskedActor().state_dict()
        self.assertEqual(len(env.historical_models), 2)
        env.close()

    def test_historical_slot_can_be_replaced_without_touching_anchor(self) -> None:
        """Profil PPO-style musi zachować model bazowy w slocie zero."""
        env = PokerLearnerEnv(training_phase=2, max_historical_models=3)
        for _ in range(3):
            env.new_historical_model_weights = MaskedActor().state_dict()
        anchor_id = id(env.historical_models[0])
        old_second_id = id(env.historical_models[1])

        env.historical_model_replacement = (1, MaskedActor().state_dict())

        self.assertEqual(id(env.historical_models[0]), anchor_id)
        self.assertNotEqual(id(env.historical_models[1]), old_second_id)
        self.assertEqual(len(env.historical_models), 3)
        env.close()

    def test_ppo_style_profile_matches_the_planned_league(self) -> None:
        profile = phase_two_profile("ppo_style")
        self.assertEqual(profile["opponent_weights"]["historical_self"], 0.40)
        self.assertEqual(profile["opponent_weights"]["latest_self"], 0.30)
        self.assertAlmostEqual(sum(profile["opponent_weights"].values()), 1.0)
        self.assertEqual(profile["max_historical_models"], 20)
        self.assertEqual(profile["history_interval_decisions"], 100_000)
        self.assertEqual(profile["league_retention"], "anchored_rotation")
        self.assertEqual(profile["opponent_epsilon"], 0.05)

    def test_named_run_seed_reproduces_the_opening_state(self) -> None:
        """Ten sam seed runu ma odtwarzać stół i osobowości botów."""
        first_env = PokerLearnerEnv(initial_seed=11_001)
        second_env = PokerLearnerEnv(initial_seed=11_001)

        first_observation, first_info = first_env.reset()
        second_observation, second_info = second_env.reset()

        np.testing.assert_array_equal(
            first_observation["obs"],
            second_observation["obs"],
        )
        np.testing.assert_array_equal(
            first_observation["mask"],
            second_observation["mask"],
        )
        self.assertEqual(first_env.opponent_styles, second_env.opponent_styles)
        self.assertEqual(first_info, second_info)

        first_env.close()
        second_env.close()

    def test_every_nonterminal_boundary_is_a_learner_decision(self) -> None:
        """Adapter nie może oddać kolektorowi obserwacji przeciwnika."""
        env = PokerLearnerEnv()
        observation, _ = env.reset(seed=123)

        for _ in range(300):
            self.assertEqual(env.poker_env.agent_selection, env.learner)
            legal_actions = np.flatnonzero(observation["mask"])
            observation, _, terminated, truncated, _ = env.step(
                int(legal_actions[0])
            )
            if terminated or truncated:
                break
        else:
            self.fail("Testowy turniej nie zakończył się w limicie decyzji")

        self.assertEqual(int(observation["mask"].sum()), 0)
        env.close()

    def test_episode_reward_matches_the_selected_reward_function(self) -> None:
        """Adapter ma przekazać pełną nagrodę, także po ruchach botów.

        Ten test obejmuje oba sygnały porównywane w fazie 1. Dzięki temu
        zmiana flagi treningowej nie może po cichu dodać albo zgubić nagrody
        podczas kończenia turnieju.
        """
        for placement_weight in (0.0, 1.0):
            env = PokerLearnerEnv(placement_reward_weight=placement_weight)
            observation, _ = env.reset(seed=123)
            total_reward = 0.0

            for _ in range(5_000):
                legal_actions = np.flatnonzero(observation["mask"])
                observation, reward, terminated, truncated, info = env.step(
                    int(legal_actions[0])
                )
                total_reward += reward
                if terminated or truncated:
                    break
            else:
                self.fail("Testowy turniej nie zakończył się w limicie decyzji")

            chip_reward = (
                env.poker_env.tournament_chips[env.learner]
                - env.poker_env.starting_chips
            ) / env.poker_env.starting_chips
            finishing_position = env.poker_env.finishing_positions[env.learner]
            placement_reward = placement_weight * (2.5 - finishing_position) * 2.0
            self.assertAlmostEqual(
                total_reward,
                chip_reward + placement_reward,
            )
            # Informacja diagnostyczna rozdziela dokładnie tę samą nagrodę,
            # którą kolektor zapisał w ostatnim przejściu replay buffera.
            self.assertAlmostEqual(
                info["chip_reward"] + info["placement_reward"],
                reward,
            )
            env.close()

    def test_illegal_learner_action_is_not_silently_changed_to_fold(self) -> None:
        """Replay buffer musi zapisywać tę samą akcję, którą wykonano."""
        env = PokerLearnerEnv()
        observation, _ = env.reset(seed=7)
        illegal_actions = np.flatnonzero(observation["mask"] == 0)
        self.assertGreater(len(illegal_actions), 0)

        with self.assertRaisesRegex(ValueError, "niedozwoloną akcję"):
            env.step(int(illegal_actions[0]))
        env.close()

    def test_collector_stores_only_legal_learner_transitions(self) -> None:
        """Batch 64 ma być batchem 64 decyzji DQN, bez próbek botów."""
        reference_env = PokerLearnerEnv()
        model = MaskedActor()
        policy = DiscreteQLearningPolicy(
            model=model,
            action_space=reference_env.action_space,
            observation_space=reference_env.observation_space,
            eps_training=1.0,
            # Kolekcja testowa, podobnie jak warm-up, odbywa się poza
            # kontekstem treningowym Tianshou. Epsilon inferencyjny zapewnia
            # więc rzeczywiście losowe, ale nadal zamaskowane akcje.
            eps_inference=1.0,
        )
        algorithm = DQN(
            policy=policy,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA,
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE,
        )
        vector_env = DummyVectorEnv([PokerLearnerEnv])
        buffer = VectorReplayBuffer(256, 1)
        collector = Collector(
            algorithm,
            vector_env,
            buffer,
            exploration_noise=True,
        )

        collector.collect(n_step=128, random=False, reset_before_collect=True)
        indices = buffer.sample_indices(0)
        transitions = buffer[indices]

        self.assertEqual(len(indices), 128)
        self.assertEqual(transitions.obs.obs.shape, (128, config.OBSERVATION_SIZE))
        self.assertEqual(
            transitions.obs_next.obs.shape,
            (128, config.OBSERVATION_SIZE),
        )
        chosen_action_is_legal = transitions.obs.mask[
            np.arange(len(indices)),
            transitions.act,
        ]
        self.assertTrue(np.all(chosen_action_is_legal))
        # Wieloagentowy bufor posiadał pole agent_id i mieszał cztery polityki.
        # Jego brak potwierdza, że wszystkie rekordy należą do jednego ucznia.
        self.assertNotIn("agent_id", transitions.obs.get_keys())

        collector.close()
        reference_env.close()

    def test_diagnostic_dqn_reports_finite_learning_signals(self) -> None:
        """Dodatkowe wykresy nie mogą zmieniać ani psuć aktualizacji DQN."""
        reference_env = PokerLearnerEnv(placement_reward_weight=1.0)
        policy = DiscreteQLearningPolicy(
            model=MaskedActor(),
            action_space=reference_env.action_space,
            observation_space=reference_env.observation_space,
            eps_training=1.0,
            eps_inference=1.0,
        )
        algorithm = DiagnosticDQN(
            policy=policy,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA,
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE,
            huber_loss_delta=config.DQN_HUBER_LOSS_DELTA,
        )
        vector_env = DummyVectorEnv(
            [lambda: PokerLearnerEnv(placement_reward_weight=1.0)]
        )
        buffer = VectorReplayBuffer(256, 1)
        collector = Collector(
            algorithm,
            vector_env,
            buffer,
            exploration_noise=True,
        )

        collector.collect(n_step=128, random=False, reset_before_collect=True)
        with policy_within_training_step(algorithm.policy):
            stats = algorithm.update(sample_size=64, buffer=buffer)

        for value in stats.get_loss_stats_dict().values():
            self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(stats.td_error_abs_mean, 0.0)
        self.assertGreaterEqual(stats.gradient_norm, 0.0)
        self.assertAlmostEqual(
            stats.action_fold_fraction
            + stats.action_check_call_fraction
            + stats.action_raise_half_fraction
            + stats.action_raise_pot_fraction
            + stats.action_all_in_fraction,
            1.0,
        )

        collector.close()
        reference_env.close()

    def test_learning_rate_schedule_reaches_tianshou_optimizer(self) -> None:
        """Harmonogram musi sterować prawdziwym optymalizatorem PyTorch.

        Tianshou 2.x opakowuje Adam we własny ``Algorithm.Optimizer``. Ten test
        używa prawdziwego DQN, a nie atrapy, więc wykryje zmianę interfejsu,
        która wcześniej powodowała awarię dopiero po kosztownej ewaluacji.
        """
        reference_env = PokerLearnerEnv()
        policy = DiscreteQLearningPolicy(
            model=MaskedActor(),
            action_space=reference_env.action_space,
            observation_space=reference_env.observation_space,
            eps_training=1.0,
            eps_inference=0.0,
        )
        algorithm = DQN(
            policy=policy,
            optim=AdamOptimizerFactory(lr=config.DQN_LEARNING_RATE),
            gamma=config.DQN_GAMMA,
            n_step_return_horizon=3,
            target_update_freq=config.DQN_TARGET_NET_UPDATE,
        )

        set_optimizer_learning_rate(algorithm.optim, 5e-5)

        for parameter_group in algorithm.optim._optim.param_groups:
            self.assertEqual(parameter_group["lr"], 5e-5)
        reference_env.close()


if __name__ == "__main__":
    unittest.main()
