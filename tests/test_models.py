import unittest

import numpy as np
import torch

from models import Critic, MaskedActor
from observation import schema


class PokerModelsTest(unittest.TestCase):
    def test_actor_output_and_action_mask(self) -> None:
        actor = MaskedActor(
            state_shape=schema.OBSERVATION_SIZE,
            action_shape=schema.NUM_ACTIONS,
        )
        observation = {
            "observation": np.zeros(
                (1, schema.OBSERVATION_SIZE),
                dtype=np.float32,
            ),
            "action_mask": np.array([[1, 1, 0, 1, 0]], dtype=np.int8),
        }

        logits, _ = actor(observation)

        self.assertEqual(tuple(logits.shape), (1, 5))
        self.assertLess(float(logits[0, 2].detach()), -1e8)
        self.assertLess(float(logits[0, 4].detach()), -1e8)

    def test_critic_returns_one_value_per_observation(self) -> None:
        critic = Critic(state_shape=schema.OBSERVATION_SIZE)
        values = critic(
            np.zeros((3, schema.OBSERVATION_SIZE), dtype=np.float32)
        )

        self.assertEqual(tuple(values.shape), (3, 1))
        self.assertTrue(torch.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
