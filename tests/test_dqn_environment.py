"""Regresje dla jednoagentowej ścieżki zbierania doświadczeń DQN."""

import unittest

import numpy as np
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

import config
from models import MaskedActor
from training.learner_environment import PokerLearnerEnv


class DQNLearnerEnvironmentTests(unittest.TestCase):
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

    # def test_rewards_between_decisions_are_accumulated_for_the_learner(self) -> None:
    #     """Suma nagród epizodu musi odpowiadać zmianie stacka ucznia."""
    #     env = PokerLearnerEnv()
    #     observation, _ = env.reset(seed=123)
    #     total_reward = 0.0

    #     for _ in range(300):
    #         legal_actions = np.flatnonzero(observation["mask"])
    #         observation, reward, terminated, truncated, _ = env.step(
    #             int(legal_actions[0])
    #         )
    #         total_reward += reward
    #         if terminated or truncated:
    #             break

    #     expected_reward = (
    #         env.poker_env.tournament_chips[env.learner]
    #         - env.poker_env.starting_chips
    #     ) / env.poker_env.starting_chips
    #     self.assertAlmostEqual(total_reward, expected_reward)
    #     env.close()

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


if __name__ == "__main__":
    unittest.main()
