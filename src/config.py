from observation.schema import OBSERVATION_SIZE


# ZMIENNE ŚRODOWISKOWE
# póki co jedyna dostępna faza to 1
TRAINING_PHASE = 1
ACTION_SPACE = 5    # nie zmieniać
STARTING_CHIPS = 200
NUM_PLAYERS = 4     # nie zmieniać

# Faza 1 losuje osobowość każdego z trzech przeciwników niezależnie na
# początku turnieju. Brak wpisu "aggressive" jest celowy: na rozgrzewce model
# ma najpierw nauczyć się podstaw przeciwko łatwiejszemu stołowi.
PHASE1_OPPONENT_WEIGHTS = {
    "random": 0.50,
    "passive": 0.40,
    "mixed": 0.10,
}

# ZMIENNE TRENINGOWE

# Krótki test techniczny jest wykonywany przez trenera po każdej epoce. Ma
# wykrywać awarie, a nie wybierać najlepszy model (10 turniejów to zbyt mała
# próba, aby wynik pokerowy był wiarygodny).
EVAL_SMOKE_TOURNAMENTS = 10

# Pełna, porównywalna walidacja uruchamia się co 100 tys. akcji środowiska.
# Każdy checkpoint gra osobno z trzema zestawami botów i z mieszanką fazy 1.
EVAL_INTERVAL_STEPS = 100_000
EVAL_TOURNAMENTS_PER_SUITE = 50

# Test końcowy korzysta z większej próby i innego zakresu seedów niż
# walidacja. 200 turniejów na zestaw ogranicza wariancję, ale nie wydłuża
# nadmiernie pierwszego lokalnego treningu na bezwentylatorowym MacBooku.
FINAL_EVAL_TOURNAMENTS_PER_SUITE = 200
EVAL_VALIDATION_SEED = 20_260
EVAL_FINAL_SEED = 90_260

# RLCard używa blindów 1/2. Jawna wartość big blinda pozwala raportować
# standardową pokerową metrykę bb/100.
BIG_BLIND = 2

# ZMIENNE TRENINGOWE DQN
DQN_LEARNING_RATE = 1e-4
DQN_GAMMA = 0.99
DQN_TARGET_NET_UPDATE = 5000 # TODO: sprawdzić czy to dobra ilość

# epsilony
DQN_EPS_MAX = 1.0
DQN_RAND_PHASE_EPS_MIN = 0.1
# Po 720 tys. akcji epsilon osiąga 0.1 i pozostaje na tym poziomie do końca.
DQN_PHASE1_EPS_DECAY_STEPS = 720_000

DQN_OTHER_PHASE_EPS_MIN = 0.02
DQN_OTHER_PHASE_EPS_MAX = 0.2
DQN_OTHER_PHASE_EPS_DECAY = 0.5

DQN_OPONENT_EPS = 0.05

# bufor
DQN_BUFFER_SIZE = 500_000
DQN_BUFFER_WARMUP = 25_000
DQN_BATCH_SIZE = 64

# MacBook Air M2 ma cztery rdzenie wydajnościowe i cztery energooszczędne.
# Osiem procesów środowiska dało najwyższą łączną przepustowość w benchmarku.
DQN_NUM_TRAIN_ENVS = 8
DQN_NUM_TEST_ENVS = 8

# Mała sieć DQN z batchem 64 działa na tym komputerze szybciej na jednym
# wątku CPU niż na wielu wątkach albo przez Apple MPS.
TORCH_NUM_THREADS = 1
TORCH_NUM_INTEROP_THREADS = 1

DQN_MAX_EPOCHS = 100
DQN_STEPS_PER_EPOCH = 10_000
# Kolektor przeplata 1000 nowych akcji z aktualizacjami sieci. Wartość dzieli
# 10 000 bez reszty, więc trener kończy dokładnie na milionie, bez nadmiarowych
# kroków wynikających z domyślnego bloku Tianshou (2048).
DQN_COLLECTION_STEPS = 1_000

# ZMIENNE TRENINGOWE PPO
PPO_LEARNING_RATE = 3e-4 # TODO: sprawdzić czy to dobra ilość
PPO_GAMMA = 0.99

# bufor
PPO_BUFFER_SIZE = 2048 # TODO: sprawdzić czy to dobry rozmiar (powinno być PPO_STEPS_PER_EPOCH/2)
PPO_BATCH_SIZE = 256

# środowiska
PPO_NUM_TRAIN_ENVS = 2
PPO_NUM_TEST_ENVS = 1
PPO_MAX_EPOCHS = 100
PPO_STEPS_PER_EPOCH = 4096 # TODO: sprawdzić czy nie za mało

# EWALUACJA
MAX_STEPS_PER_TOURNAMENT = 1000 # ile akcji mogą podjąć w testowym turnieju (żeby nie trwały w nieskończoność)
