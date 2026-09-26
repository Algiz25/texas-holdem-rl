import unittest

import numpy as np
from rlcard.games.limitholdem import PlayerStatus

from environment import TexasHoldemTournament
from observation import schema


class TexasHoldemTournamentTest(unittest.TestCase):
    @staticmethod
    def _deterministic_trace(seed: int, actions: int = 120):
        env = TexasHoldemTournament(num_players=4, starting_chips=200)
        env.reset(seed=seed)
        trace = []
        completed_actions = 0
        while completed_actions < actions:
            observation, reward, terminated, truncated, _ = env.last()
            if terminated or truncated:
                env.step(None)
                if not env.agents:
                    break
                continue
            legal_actions = np.flatnonzero(observation["action_mask"])
            action = int(legal_actions[0])
            trace.append(
                (
                    env.agent_selection,
                    observation["observation"].tobytes(),
                    action,
                    float(reward),
                )
            )
            env.step(action)
            completed_actions += 1
        return trace

    def test_each_reset_exposes_a_new_tournament_identifier(self) -> None:
        env = TexasHoldemTournament()
        env.reset(seed=10)
        first_id = env.infos["player_0"]["tournament_id"]

        env.reset(seed=10)
        second_id = env.infos["player_0"]["tournament_id"]

        self.assertNotEqual(first_id, second_id)
        self.assertTrue(
            all(
                info["tournament_id"] == second_id
                for info in env.infos.values()
            )
        )

    def test_reused_rlcard_environment_remains_seed_reproducible(self) -> None:
        first = self._deterministic_trace(seed=123)
        second = self._deterministic_trace(seed=123)
        different = self._deterministic_trace(seed=124)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_rlcard_environment_is_reused_between_four_player_hands(self) -> None:
        original_environment = self.env.rlcard_env
        starting_completed_hands = self.env.completed_hands

        for agent in self.env.agent_iter(max_iter=20):
            observation, _, terminated, truncated, _ = self.env.last()
            if terminated or truncated:
                action = None
            else:
                legal_actions = np.flatnonzero(observation["action_mask"])
                action = (
                    schema.ACTION_FOLD
                    if schema.ACTION_FOLD in legal_actions
                    else int(legal_actions[0])
                )
            self.env.step(action)
            if self.env.completed_hands > starting_completed_hands:
                break

        self.assertGreater(self.env.completed_hands, starting_completed_hands)
        self.assertIs(self.env.rlcard_env, original_environment)
        self.assertEqual(self.env.rlcard_env.game.dealer_id, self.env.dealer_idx)

    def test_rlcard_environment_is_recreated_when_player_count_changes(self) -> None:
        four_player_environment = self.env.rlcard_env

        self.env._prepare_rlcard_env(num_active=3)

        self.assertIsNot(self.env.rlcard_env, four_player_environment)
        self.assertEqual(self.env._rlcard_player_count, 3)
        self.assertEqual(self.env.rlcard_env.num_players, 3)

    def setUp(self) -> None:
        self.env = TexasHoldemTournament(num_players=4, starting_chips=200)
        self.env.reset(seed=7)

    def test_observation_and_action_mask_shapes(self) -> None:
        observation = self.env.observe(self.env.agent_selection)

        self.assertEqual(
            observation["observation"].shape,
            (schema.OBSERVATION_SIZE,),
        )
        self.assertEqual(observation["action_mask"].shape, (5,))
        self.assertEqual(observation["observation"].dtype, np.float32)
        self.assertGreater(int(observation["action_mask"].sum()), 0)

        vector = observation["observation"]
        self.assertEqual(float(vector[schema.OWN_CARDS].sum()), 2.0)
        self.assertEqual(float(vector[schema.BOARD_CARDS].sum()), 0.0)
        np.testing.assert_array_equal(vector[schema.PLAYER_ACTIVE], [1, 1, 1, 1])
        np.testing.assert_array_equal(vector[schema.STREET], [1, 0, 0, 0])
        self.assertGreaterEqual(float(vector.min()), 0.0)
        self.assertLessEqual(float(vector.max()), 1.0)

    def test_spaces_match_environment_contract(self) -> None:
        agent = self.env.agent_selection

        self.assertEqual(
            self.env.observation_space(agent).shape,
            (schema.OBSERVATION_SIZE,),
        )
        self.assertEqual(self.env.action_space(agent).n, 5)

    def test_observation_cache_is_reused_only_until_the_next_action(self) -> None:
        """Cache nie może ukryć zmiany puli, historii ani aktualnego gracza."""
        first_agent = self.env.agent_selection
        first = self.env.observe(first_agent)
        repeated = self.env.observe(first_agent)
        self.assertIs(first, repeated)

        legal_actions = np.flatnonzero(first["action_mask"])
        self.env.step(int(legal_actions[0]))

        # Ruch czyści cały cache. Następna obserwacja musi zostać policzona ze
        # stanu po akcji, nawet gdy ponownie obserwuje ten sam gracz.
        self.assertEqual(self.env._observation_cache, {})
        next_agent = self.env.agent_selection
        refreshed = self.env.observe(next_agent)
        self.assertIsNot(refreshed, first)

    def test_step_updates_public_action_history(self) -> None:
        acting_player = self.env.agent_selection

        # CHECK_CALL jest zawsze legalne w RLCard; zależnie od sytuacji oznacza
        # ono check albo dopłatę, ale identyfikator ostatniej akcji pozostaje 1.
        self.env.step(schema.ACTION_CHECK_CALL)

        self.assertEqual(
            self.env.action_history.last_actions[acting_player],
            schema.ACTION_CHECK_CALL,
        )

    def test_short_all_in_uses_only_chips_actually_paid(self) -> None:
        # RLCard ustawia round.raised na pełną kwotę calla także wtedy, gdy
        # gracz ma krótszy stack. Encoder opiera się dlatego na in_chips.
        player = self.env.rlcard_env.game.players[1]
        player.in_chips = 13
        player.remained_chips = 0
        player.status = PlayerStatus.ALLIN
        self.env.rlcard_env.game.round.raised[1] = 29

        contributions = self.env._street_contributions()
        self.assertEqual(contributions[1], 13)

    def test_completed_hands_update_opponent_statistics(self) -> None:
        # Kolejne foldy szybko kończą rozdania bez uzależniania testu od kart.
        for agent in self.env.agent_iter(max_iter=10):
            observation, _, terminated, truncated, _ = self.env.last()
            if terminated or truncated:
                action = None
            else:
                legal_actions = np.flatnonzero(observation["action_mask"])
                action = (
                    schema.ACTION_FOLD
                    if schema.ACTION_FOLD in legal_actions
                    else int(legal_actions[0])
                )
            self.env.step(action)

        observed_hands = [
            statistics.observed_hands
            for statistics in self.env.opponent_stats.players.values()
        ]
        self.assertGreater(min(observed_hands), 0)


if __name__ == "__main__":
    unittest.main()
