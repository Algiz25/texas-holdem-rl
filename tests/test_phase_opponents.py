"""Testy osobowości przeciwników w równoległych turniejach."""

import unittest

import numpy as np
from gymnasium.spaces import Discrete
from tianshou.data import Batch

from opponents import PassivePolicy
from phases import DynamicOpponentPolicy


class DynamicOpponentPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        action_space = Discrete(5)
        passive = PassivePolicy(action_space)
        self.policy = DynamicOpponentPolicy(
            action_space,
            {"passive": passive},
            {"passive": 1.0},
            seed=123,
        )

    def test_personality_is_stable_within_one_tournament(self) -> None:
        first = self.policy.personality_for("environment-a:tournament-1")
        second = self.policy.personality_for("environment-a:tournament-1")

        self.assertEqual(first, second)
        self.assertEqual(len(self.policy._personality_by_tournament), 1)

    def test_parallel_tournaments_are_tracked_independently(self) -> None:
        batch = Batch(
            obs=Batch(
                action_mask=np.array(
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                    ],
                    dtype=np.int8,
                )
            ),
            info=Batch(
                tournament_id=np.array(
                    ["environment-a:tournament-1", "environment-b:tournament-7"]
                )
            ),
        )

        result = self.policy(batch)

        np.testing.assert_array_equal(result.act, np.array([1, 1]))
        self.assertEqual(len(self.policy._personality_by_tournament), 2)


if __name__ == "__main__":
    unittest.main()
