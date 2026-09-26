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
checkpoints/                lokalne checkpointy modeli i raporty ewaluacji
logs/                       lokalne logi TensorBoard i terminala
scripts/
  run_overnight_dqn.sh      jeden długi, bezpiecznie zatrzymywany trening
src/
  config.py                 parametry środowiska, treningu i ewaluacji
  environment.py            środowisko PettingZoo/RLCard
  models.py                 sieci Actor i Critic
  opponents.py              losowi, heurystyczni i zamrożeni przeciwnicy
  training/
    base_trainer.py         wspólna logika treningu
    dqn_environment.py      jednoagentowy widok decyzji DQN
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

- 250 000 decyzji DQN (25 epok po 10 000 decyzji), co odpowiada w przybliżeniu
  dawnemu milionowi ruchów całego czteroosobowego stołu,
- dodatkowe 25 000 legalnych decyzji rozgrzewających replay buffer,
- 8 równoległych procesów treningowych, 1 proces krótkiego testu technicznego
  i 1 wątek obliczeniowy PyTorch,
- przeciwnicy losowani na turniej w proporcji 50% Random, 40% Check/Call,
  10% Mixed,
- epsilon malejący z 1.0 do 0.1 przez 180 000 decyzji ucznia,
- jedna aktualizacja gradientu na cztery nowe decyzje (`update ratio = 0.25`).

Trening DQN korzysta z jednoagentowej nakładki. Po akcji ucznia środowisko
samodzielnie rozgrywa ruchy botów i zwraca sterowanie dopiero przy kolejnej
decyzji ucznia. Dzięki temu replay buffer nigdy nie łączy akcji `player_0` z
prywatną obserwacją następnego przeciwnika. Jeden krok DQN oznacza teraz jedną
decyzję ucznia, a nie jeden dowolny ruch przy stole.

Pełna walidacja DQN odbywa się co 25 000 decyzji na czterech zestawach
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

Każdy trening DQN otrzymuje nazwę i seed. Nazwa oddziela jego modele, raporty
ewaluacji oraz wykresy od pozostałych eksperymentów. Przykład pojedynczego
runu:

```bash
PYTHONPATH=src .venv/bin/python src/training/train_dqn.py \
  --run-name baseline_seed_11001 \
  --seed 11001
```

Nowy trening DQN zapisuje także `training_state_latest.pth` i
`training_state_final.pth`. Zawierają wagi, sieć docelową, optymalizator oraz
liczniki potrzebne do kontynuacji. Replay buffer nie jest zapisywany, dlatego
po wznowieniu skrypt ponownie zbiera 25 000 decyzji rozgrzewkowych.

Kontynuacja o kolejne 250 000 decyzji z nowego stanu treningowego:

```bash
PYTHONPATH=src .venv/bin/python src/training/train_dqn.py \
  --run-name baseline_seed_11001 \
  --seed 11001 \
  --resume checkpoints/dqn/baseline_seed_11001/training_state_final.pth \
  --decisions 250000
```

Stany zapisane przez dawny wieloagentowy kolektor są celowo odrzucane, bo
zawierają model uczony na przejściach pomiędzy perspektywami różnych graczy.
Pierwszy trening po tej poprawce należy rozpocząć od zera. Stare checkpointy
można zachować wyłącznie jako materiał porównawczy w ewaluacji.

Skrypt blokuje równoczesne uruchomienie drugiego treningu DQN w tym samym
katalogu projektu.

Checkpointy każdego DQN są zapisywane w osobnym katalogu:

```text
checkpoints/dqn/<run-name>/best.pth
checkpoints/dqn/<run-name>/latest.pth
checkpoints/dqn/<run-name>/final.pth
checkpoints/dqn/<run-name>/training_state_step_000100000.pth
checkpoints/dqn/<run-name>/training_state_step_000200000.pth
checkpoints/dqn/<run-name>/training_state_final.pth
checkpoints/ppo/best.pth
checkpoints/ppo/final.pth
```

Pliki `training_state_step_*` zachowują pełny stan DQN co 100 000 decyzji
(10 epok). Można ich użyć do wznowienia albo jako punktu startowego kolejnej
fazy. Pliki `step_*`, `best.pth`, `latest.pth` i `final.pth` zawierają wagi
przeznaczone do porównywania modeli. Replay buffer nie jest zapisywany.

## Trening nocny DQN

Skrypt nocny prowadzi jeden ciągły trening `overnight_long_seed_11001`. Dzięki
temu ten sam model oraz replay buffer rozwijają się przez całą noc, zamiast
kilka razy zaczynać od zera. Limit 100 milionów decyzji jest wyłącznie
zabezpieczeniem technicznym — po około dziewięciu godzinach trening należy
zatrzymać ręcznie.

Długi run korzysta z harmonogramu epsilon `1.0 → 0.1` przez pierwsze 700 000
decyzji i wykonuje pełną ewaluację co 250 000 decyzji. Ogranicza to czas
poświęcony na testy, ale pozostawia regularne, porównywalne punkty kontrolne.

Przed startem skrypt uruchamia testy. Sam włącza też `caffeinate`, więc macOS
nie uśpi komputera ani nie wygasi ekranu do końca pracy. Należy pozostawić
MacBooka podłączonego do zasilania i uruchomić:

```bash
./scripts/run_overnight_dqn.sh
```

Rano należy przejść do terminala z treningiem i jeden raz nacisnąć `Ctrl+C`.
Skrypt przechwytuje przerwanie, zapisuje `interrupted.pth`, aktualizuje
`latest.pth` oraz tworzy pełny `training_state_latest.pth`. Celowo nie uruchamia
wtedy długiej ewaluacji końcowej, aby komputer został zwolniony od razu.

Jeżeli trening ma być kontynuowany kolejnej nocy, ponowne wykonanie tej samej
komendy wznawia model od `training_state_latest.pth`. Ponieważ replay buffer
nie jest częścią stanu, przed dalszym uczeniem ponownie wykonywany jest warm-up
25 000 decyzji.

Postęp, loss, epsilon i wyniki pokerowe można oglądać w TensorBoard:

```bash
.venv/bin/tensorboard --logdir logs/dqn
```

Następnie należy otworzyć `http://localhost:6006`. Surowy zapis terminala z
każdego modelu trafia do `logs/dqn/overnight_<data>/`.

## Ewaluacja

```bash
PYTHONPATH=src python src/evaluation/evaluator_dqn.py
PYTHONPATH=src python src/evaluation/evaluator_ppo.py
```

## Testy

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
