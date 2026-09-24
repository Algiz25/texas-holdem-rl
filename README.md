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
  training/
    base_trainer.py         wspólna logika treningu
    train_dqn.py            trening DQN
    train_ppo.py            trening PPO
  evaluation/
    evaluator.py            wspólna logika ewaluacji
    evaluator_dqn.py        ewaluacja DQN
    evaluator_ppo.py        ewaluacja PPO
tests/                      testy środowiska i modeli
```

## Trening

Parametry środowiska, DQN i PPO zmienia się w `src/config.py`.

Pierwsza faza DQN jest skonfigurowana pod lokalny trening na MacBooku Air M2:

- 1 000 000 akcji treningowych (100 epok po 10 000 akcji),
- dodatkowe 25 000 losowych akcji rozgrzewających replay buffer,
- 8 równoległych procesów treningowych, 1 proces krótkiego testu technicznego
  i 1 wątek obliczeniowy PyTorch,
- przeciwnicy losowani na turniej w proporcji 50% Random, 40% Check/Call,
  10% Mixed,
- epsilon malejący z 1.0 do 0.1 przez 720 000 akcji.

Pełna walidacja odbywa się co 100 000 akcji na czterech zestawach
przeciwników. Pojedynczy mecz trwa do końca turnieju albo do 100 rozdań, więc
główna metryka `bb/100` nie zależy od tego, jak długo pasywni gracze utrzymują
się przy stole. Po zakończeniu najlepszy i ostatni model rozgrywają po 200
meczów przeciwko każdemu zestawowi. Cztery zestawy są liczone równolegle w
osobnych procesach.

Osiem środowisk celowo wykorzystuje wszystkie rdzenie komputera. Podczas
treningu system może reagować wolno, dlatego najlepiej nie wykonywać w tym
czasie innych obciążających zadań.

Silnik RLCard jest ponownie wykorzystywany pomiędzy rozdaniami i tworzony od
nowa dopiero po zmianie liczby aktywnych graczy. Nie zmienia to zasad gry ani
obserwacji, a usuwa koszt konstruowania oraz seedowania całego silnika przy
każdej kolejnej ręce.

```bash
PYTHONPATH=src .venv/bin/python src/training/train_dqn.py
PYTHONPATH=src .venv/bin/python src/training/train_ppo.py
```

Nowy trening DQN zapisuje także `training_state_latest.pth` i
`training_state_final.pth`. Zawierają wagi, sieć docelową, optymalizator oraz
liczniki potrzebne do kontynuacji. Replay buffer nie jest zapisywany, dlatego
po wznowieniu skrypt ponownie zbiera 25 000 akcji rozgrzewkowych.

Kontynuacja o kolejny milion akcji z nowego stanu treningowego:

```bash
PYTHONPATH=src .venv/bin/python src/training/train_dqn.py \
  --resume checkpoints/dqn/training_state_final.pth \
  --actions 1000000
```

Obecny model, zapisany przed dodaniem pełnego stanu treningowego, można
kontynuować z samych wag. Trzeba wtedy jawnie podać wykonany wcześniej krok:

```bash
PYTHONPATH=src .venv/bin/python src/training/train_dqn.py \
  --resume checkpoints/dqn/final.pth \
  --start-step 1000000 \
  --actions 1000000
```

Skrypt blokuje równoczesne uruchomienie drugiego treningu DQN w tym samym
katalogu checkpointów.

Checkpointy są zapisywane niezależnie od katalogu uruchomienia:

```text
checkpoints/dqn/best.pth
checkpoints/dqn/latest.pth
checkpoints/dqn/final.pth
checkpoints/ppo/best.pth
checkpoints/ppo/final.pth
```

## Ewaluacja

```bash
PYTHONPATH=src python src/evaluation/evaluator_dqn.py
PYTHONPATH=src python src/evaluation/evaluator_ppo.py
```

## Testy

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
