"""Stable project paths independent of the current working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DQN_CHECKPOINT_DIR = CHECKPOINT_DIR / "dqn"
PPO_CHECKPOINT_DIR = CHECKPOINT_DIR / "ppo"


def evaluation_dir(algo_name: str) -> Path:
    """Return the directory containing reports for one algorithm."""
    checkpoint_dirs = {
        "dqn": DQN_CHECKPOINT_DIR,
        "ppo": PPO_CHECKPOINT_DIR,
    }
    try:
        return checkpoint_dirs[algo_name] / "evaluations"
    except KeyError as error:
        raise ValueError(f"Unknown algorithm: {algo_name}") from error


def ensure_output_directories() -> None:
    """Create local output directories used by training and evaluation."""
    for path in (DQN_CHECKPOINT_DIR, PPO_CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)
        evaluation_dir(path.name).mkdir(parents=True, exist_ok=True)
