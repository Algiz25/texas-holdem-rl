"""Stable project paths independent of the current working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DQN_CHECKPOINT_DIR = CHECKPOINT_DIR / "dqn"
PPO_CHECKPOINT_DIR = CHECKPOINT_DIR / "ppo"


def ensure_output_directories() -> None:
    """Create local output directories used by training and evaluation."""
    for path in (DQN_CHECKPOINT_DIR, PPO_CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
