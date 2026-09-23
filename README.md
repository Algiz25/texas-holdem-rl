# Texas Hold'em RL

Projekt środowiska turniejowego Texas Hold'em oraz agentów reinforcement learning opartych na DQN i PPO.

## Instalacja

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Struktura

```text
checkpoints/                lokalne checkpointy modeli
src/
  config.py                 parametry środowiska, treningu i ewaluacji
  environment.py            środowisko PettingZoo/RLCard
  models.py                 sieci Actor i Critic
  opponents.py              losowi, heurystyczni i zamrożeni przeciwnicy
  train_dqn.py              trening DQN
  train_ppo.py              trening PPO
  evaluator.py              wspólna logika ewaluacji
  evaluator_dqn.py          ewaluacja DQN
  evaluator_ppo.py          ewaluacja PPO
tests/                      testy środowiska i modeli
```

## Trening

Parametry środowiska, DQN i PPO zmienia się w `src/config.py`.

```bash
python src/train_dqn.py
python src/train_ppo.py
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
python src/evaluator_dqn.py
python src/evaluator_ppo.py
```

## Testy

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
