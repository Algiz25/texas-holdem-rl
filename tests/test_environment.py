import unittest

import numpy as np

from environment import TexasHoldemTournament


class TexasHoldemTournamentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TexasHoldemTournament(num_players=4, starting_chips=200)
        self.env.reset(seed=7)

    def test_observation_and_action_mask_shapes(self) -> None:
        observation = self.env.observe(self.env.agent_selection)

        self.assertEqual(observation["observation"].shape, (68,))
        self.assertEqual(observation["action_mask"].shape, (5,))
        self.assertEqual(observation["observation"].dtype, np.float32)
        self.assertGreater(int(observation["action_mask"].sum()), 0)

    def test_spaces_match_environment_contract(self) -> None:
        agent = self.env.agent_selection

        self.assertEqual(self.env.observation_space(agent).shape, (68,))
        self.assertEqual(self.env.action_space(agent).n, 5)


if __name__ == "__main__":
    unittest.main()
