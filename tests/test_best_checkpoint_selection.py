"""Regresje wyboru najlepszego modelu podczas walidacji."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from training.base_trainer import BasePokerTrainer


class _FakeEvaluator:
    score = 999.0

    def __init__(self, **kwargs) -> None:
        pass

    def evaluate(self, **kwargs):
        return {
            "random": SimpleNamespace(bb_per_100=self.score),
            "passive": SimpleNamespace(bb_per_100=self.score),
            "mixed": SimpleNamespace(bb_per_100=self.score),
            "phase1_mix": SimpleNamespace(bb_per_100=self.score),
        }


class BestCheckpointSelectionTests(unittest.TestCase):
    def _trainer_without_environments(self, checkpoint_dir: Path) -> BasePokerTrainer:
        trainer = BasePokerTrainer.__new__(BasePokerTrainer)
        trainer.checkpoint_dir = checkpoint_dir
        trainer.evaluation_report_dir = checkpoint_dir / "evaluations"
        trainer.evaluation_report_dir.mkdir(parents=True)
        trainer.evaluator_class = _FakeEvaluator
        trainer.training_phase = 1
        trainer.algo_name = "dqn"
        trainer.tensorboard_writer = None
        trainer.best_validation_score = float("-inf")
        trainer._existing_checkpoints_preserved = False
        trainer.evaluation_interval_steps = 250_000
        trainer.next_evaluation_step = 250_000
        trainer.device = "cpu"
        return trainer

    def test_untrained_baseline_is_not_saved_as_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._trainer_without_environments(Path(temporary_directory))
            policy = torch.nn.Linear(2, 2)

            trainer.run_initial_evaluation(learner_policy=policy)

            self.assertTrue((trainer.checkpoint_dir / "latest.pth").exists())
            self.assertFalse((trainer.checkpoint_dir / "best.pth").exists())
            self.assertEqual(trainer.best_validation_score, float("-inf"))

    def test_first_trained_validation_creates_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._trainer_without_environments(Path(temporary_directory))
            policy = torch.nn.Linear(2, 2)
            trainer.run_initial_evaluation(learner_policy=policy)

            _FakeEvaluator.score = 123.0
            trainer.run_periodic_evaluation(
                env_step=250_000,
                learner_policy=policy,
            )

            self.assertTrue((trainer.checkpoint_dir / "best.pth").exists())
            self.assertEqual(trainer.best_validation_score, 123.0)
            self.assertTrue((trainer.checkpoint_dir / "candidate_1.pth").exists())
            self.assertTrue((trainer.checkpoint_dir / "top_candidates.json").exists())

    def test_validation_score_uses_training_opponent_weights(self) -> None:
        score = BasePokerTrainer._phase_one_validation_score(
            {
                "random": SimpleNamespace(bb_per_100=100.0),
                "passive": SimpleNamespace(bb_per_100=200.0),
                "mixed": SimpleNamespace(bb_per_100=300.0),
            }
        )
        self.assertAlmostEqual(score, 175.0)

    def test_phase_two_score_balances_baseline_and_old_bots(self) -> None:
        """Self-play nie może wybrać modelu, który zapomniał starych botów."""
        trainer = BasePokerTrainer.__new__(BasePokerTrainer)
        trainer.training_phase = 2
        results = {
            "random": SimpleNamespace(bb_per_100=100.0),
            "passive": SimpleNamespace(bb_per_100=200.0),
            "mixed": SimpleNamespace(bb_per_100=300.0),
            "baseline": SimpleNamespace(bb_per_100=25.0),
        }
        self.assertAlmostEqual(trainer._validation_score(results), 100.0)

    def test_only_three_best_candidates_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            trainer = self._trainer_without_environments(Path(temporary_directory))
            policy = torch.nn.Linear(2, 2)
            trainer.run_initial_evaluation(learner_policy=policy)

            for index, score in enumerate((100.0, 400.0, 200.0, 300.0), start=1):
                _FakeEvaluator.score = score
                trainer.run_periodic_evaluation(
                    env_step=index * 250_000,
                    learner_policy=policy,
                )

            candidates = json.loads(
                (trainer.checkpoint_dir / "top_candidates.json").read_text()
            )
            self.assertEqual([row["score"] for row in candidates], [400.0, 300.0, 200.0])
            self.assertTrue((trainer.checkpoint_dir / "candidate_3.pth").exists())


if __name__ == "__main__":
    unittest.main()
