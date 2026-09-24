import unittest

from observation import schema


class ObservationSchemaTest(unittest.TestCase):
    def test_observation_size(self) -> None:
        self.assertEqual(schema.OBSERVATION_SIZE, 222)

    def test_schema_has_no_gaps_or_overlaps(self) -> None:
        current_index = 0

        for _, block in schema.SCHEMA_BLOCKS:
            self.assertEqual(block.start, current_index)
            self.assertGreater(block.stop, block.start)
            current_index = block.stop

        self.assertEqual(current_index, schema.OBSERVATION_SIZE)

    def test_last_action_blocks(self) -> None:
        self.assertEqual(schema.last_action_slice(0), slice(138, 143))
        self.assertEqual(schema.last_action_slice(1), slice(143, 148))
        self.assertEqual(schema.last_action_slice(2), slice(148, 153))
        self.assertEqual(schema.last_action_slice(3), slice(153, 158))

    def test_street_summary_blocks(self) -> None:
        self.assertEqual(
            schema.street_summary_slice(schema.STREET_PREFLOP),
            slice(158, 165),
        )
        self.assertEqual(
            schema.street_summary_slice(schema.STREET_FLOP),
            slice(165, 172),
        )
        self.assertEqual(
            schema.street_summary_slice(schema.STREET_TURN),
            slice(172, 179),
        )
        self.assertEqual(
            schema.street_summary_slice(schema.STREET_RIVER),
            slice(179, 186),
        )

    def test_opponent_stat_blocks(self) -> None:
        self.assertEqual(schema.opponent_stats_slice(0), slice(186, 192))
        self.assertEqual(schema.opponent_stats_slice(1), slice(192, 198))
        self.assertEqual(schema.opponent_stats_slice(2), slice(198, 204))

    def test_invalid_indices_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            schema.last_action_slice(4)

        with self.assertRaises(ValueError):
            schema.street_summary_slice(-1)

        with self.assertRaises(ValueError):
            schema.opponent_stats_slice(3)


if __name__ == "__main__":
    unittest.main()