# AGENTS.md

## Project overview

This repository trains multi-agent reinforcement-learning policies for a four-player no-limit Texas Hold'em tournament. It uses RLCard for poker mechanics, PettingZoo for the multi-agent environment, Tianshou for DQN/PPO training, and PyTorch for neural networks.

The current implemented observation vector has 68 values and the action space has five discrete actions: fold, check/call, raise half-pot, raise pot, and all-in. Discussions or documentation about larger observation vectors are design proposals until the environment, models, evaluators, configurations, and tests are updated together.

## Repository layout

- `checkpoints/`: local `.pth` model weights, split by algorithm.
- `runs/`: local logs and experiment metrics; automatic run logging is not implemented yet.
- `src/environment.py`: RLCard/PettingZoo tournament environment and observation construction.
- `src/config.py`: environment, DQN, PPO, and evaluation parameters.
- `src/models.py`: actor, critic, and policy wrappers.
- `src/opponents.py`: random, heuristic, and frozen opponents.
- `src/train_dqn.py` and `src/train_ppo.py`: training entry points.
- `src/evaluator.py`, `src/evaluator_dqn.py`, and `src/evaluator_ppo.py`: model evaluation.
- `tests/`: environment and model tests.

## Environment setup

Use Python 3.12 and the project-local virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Do not add generated virtual-environment contents to Git.

## Validation

Run these checks after code changes:

```bash
.venv/bin/python -m compileall -q src tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

For import and configuration changes, also verify:

```bash
PYTHONPATH=src .venv/bin/python -c "import config"
```

Do not start a full DQN or PPO training run merely as a validation step. Training is long-running and writes checkpoints. Import the modules and test small components instead unless a training run is explicitly requested.

## Running the project

```bash
python src/train_dqn.py
python src/train_ppo.py
python src/evaluator_dqn.py
python src/evaluator_ppo.py
```

Training settings belong in `src/config.py`, not as new hard-coded constants in training modules.

## Coding conventions

- Keep imports consistent with the flat `src/` layout.
- Keep environment mechanics and observation encoding in `environment.py`.
- Keep neural-network definitions in `models.py`.
- Keep training orchestration separate from evaluation.
- Use paths from `paths.py` for checkpoints and run output so behavior is independent of the current working directory.
- When changing observation size or layout, update the environment space, shared configuration, DQN/PPO model construction, evaluators, heuristic policies that use fixed indices, and tests in the same change.
- Keep the five-action ordering stable unless every consumer and saved-model compatibility concern is handled explicitly.
- Prefer small, named helpers over adding more unrelated code to `opponents.py`.

## Checkpoints and generated files

- Existing model weights are user data. Preserve them unless the user explicitly asks to remove or replace them.
- DQN weights live under `checkpoints/dqn/`; PPO weights live under `checkpoints/ppo/`.
- `.pth`, `.pt`, `.ckpt`, `runs/`, and logs are intentionally ignored by Git.
- Do not claim that ignored checkpoints are committed or portable with the repository.
- A policy checkpoint becomes incompatible when the observation size or network architecture changes. Call this out clearly before making such a change.

## Testing expectations

- Add or update tests whenever observation layout, legal-action masking, environment transitions, or network input/output shapes change.
- Use deterministic seeds where practical.
- Avoid tests that launch subprocess vector environments or lengthy tournaments unless they target that behavior specifically.
- Preserve the existing checkpoint-loading smoke test behavior when reorganizing paths or evaluator code.

## Known considerations

- The observation redesign discussed for richer cards, betting history, opponent statistics, and derived poker features is not implemented yet.
- The `runs/` directory is reserved for metrics, logs, configuration snapshots, and TensorBoard data, but training does not write those artifacts yet.
- The current device selection uses CUDA when available and otherwise CPU; Apple MPS support has not been integrated into training.
