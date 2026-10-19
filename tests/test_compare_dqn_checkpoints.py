"""Testy wznawialnego porównania checkpointów bez rozgrywania turniejów."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.compare_dqn_checkpoints import (
    DEFAULT_STEPS,
    bootstrap_suite,
    report_is_complete,
)
from evaluation.evaluator import EVALUATION_SUITES


class DQNCheckpointComparisonTests(unittest.TestCase):
    def test_default_comparison_includes_strong_early_checkpoints(self) -> None:
        self.assertEqual(DEFAULT_STEPS[:2], (750_000, 1_250_000))
        self.assertEqual(len(DEFAULT_STEPS), 7)

    def test_only_complete_report_is_skipped_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            complete_suite = {
                "tournaments": 2,
                "tournament_samples": [{"seed": 1}, {"seed": 2}],
            }
            report_path.write_text(
                json.dumps({suite: complete_suite for suite in EVALUATION_SUITES}),
                encoding="utf-8",
            )
            self.assertTrue(report_is_complete(report_path, tournaments=2))

            # Brak jednej próbki oznacza raport niepełny i wymusza powtórzenie
            # całego checkpointu zamiast użycia częściowego wyniku.
            incomplete = json.loads(report_path.read_text(encoding="utf-8"))
            incomplete["random"]["tournament_samples"].pop()
            report_path.write_text(json.dumps(incomplete), encoding="utf-8")
            self.assertFalse(report_is_complete(report_path, tournaments=2))

    def test_bootstrap_uses_chip_delta_per_hand_and_resolved_wins(self) -> None:
        result = {
            "tournament_samples": [
                {
                    "chip_delta": 20.0,
                    "hands": 10,
                    "won": True,
                    "resolved": True,
                },
                {
                    "chip_delta": 20.0,
                    "hands": 10,
                    "won": False,
                    "resolved": False,
                },
            ]
        }
        bb_samples, win_samples = bootstrap_suite(
            result,
            rng=np.random.default_rng(123),
            repetitions=20,
        )
        # 20 żetonów / 10 rozdań / BB=2 × 100 = 100 bb/100.
        np.testing.assert_allclose(bb_samples, 100.0)
        # Nierozstrzygnięty turniej nie jest automatycznie traktowany jako porażka.
        np.testing.assert_allclose(win_samples, 1.0)


if __name__ == "__main__":
    unittest.main()
