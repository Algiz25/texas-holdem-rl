# Texas Hold'em RL

Projekt środowiska turniejowego Texas Hold'em oraz agentów reinforcement learning opartych na DQN i PPO.

## Instalacja

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Alternatywnie można utworzyć środowisko Conda:

```bash
conda env create -f environment.yml
conda activate texas-holdem-rl
pip install -e .
```

## Struktura

```text
checkpoints/                lokalne checkpointy modeli
runs/                       logi i wyniki eksperymentów
src/texas_holdem_rl/
  config.py                 parametry środowiska, treningu i ewaluacji
  environment.py            środowisko PettingZoo/RLCard
  models.py                 sieci Actor i Critic
  opponents.py              losowi, heurystyczni i zamrożeni przeciwnicy
  training/                 trening DQN i PPO
  evaluation/               ewaluacja modeli
tests/                      testy środowiska i modeli
```

## Trening

Parametry środowiska, DQN i PPO zmienia się w `src/texas_holdem_rl/config.py`.

```bash
python -m texas_holdem_rl.training.dqn
python -m texas_holdem_rl.training.ppo
```

Checkpointy są zapisywane niezależnie od katalogu uruchomienia:

```text
checkpoints/dqn/best.pth
checkpoints/dqn/final.pth
checkpoints/ppo/best.pth
checkpoints/ppo/final.pth
```

## Ewaluacja

```bash
python -m texas_holdem_rl.evaluation.dqn
python -m texas_holdem_rl.evaluation.ppo
```

## Testy

```bash
python -m unittest discover -s tests
```
