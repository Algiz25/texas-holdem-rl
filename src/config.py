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

# Rozkład akcji osobowości Mixed jest wspólny dla treningu i ewaluacji.
# Jedno źródło zapobiega sytuacji, w której bot o tej samej nazwie zachowuje
# się inaczej podczas zbierania doświadczeń i podczas pomiaru checkpointu.
MIXED_ACTION_WEIGHTS = (0.20, 0.45, 0.18, 0.12, 0.05)

# ZMIENNE TRENINGOWE

# Krótki test techniczny jest wykonywany przez trenera po każdej epoce. Jeden
# turniej wystarcza do wykrycia awarii połączenia model–środowisko. Wynik tego
# testu nie służy do oceny jakości; rzetelna ewaluacja używa osobnego interwału.
EVAL_SMOKE_TOURNAMENTS = 1

# Domyślny interwał zachowujemy dla wieloagentowego PPO. DQN nadpisuje go
# interwałem liczonym w swoich decyzjach, zdefiniowanym niżej.
EVAL_INTERVAL_STEPS = 100_000
EVAL_TOURNAMENTS_PER_SUITE = 50

# Test końcowy korzysta z większej próby i innego zakresu seedów niż
# walidacja. 200 turniejów na zestaw ogranicza wariancję, ale nie wydłuża
# nadmiernie pierwszego lokalnego treningu na bezwentylatorowym MacBooku.
FINAL_EVAL_TOURNAMENTS_PER_SUITE = 200
EVAL_VALIDATION_SEED = 20_260
EVAL_FINAL_SEED = 90_260

# Cztery niezależne zestawy przeciwników są liczone jednocześnie. Każdy proces
# ma własne środowisko i kopię małej sieci, więc nie współdzieli zmiennego stanu
# turnieju. Jeden wątek PyTorch na proces zapobiega nadmiernemu wykorzystaniu
# rdzeni przez cztery małe operacje inferencji.
EVAL_NUM_WORKERS = 4
EVAL_WORKER_TORCH_THREADS = 1

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
# Po przejściu na jednoagentowy kolektor jeden krok oznacza decyzję ucznia.
# 180 tys. decyzji odpowiada w przybliżeniu dawnym 720 tys. ruchów całego
# czteroosobowego stołu. Pozostałe 70 tys. decyzji utrzymuje epsilon 0.1.
DQN_PHASE1_EPS_DECAY_STEPS = 180_000

DQN_OTHER_PHASE_EPS_MIN = 0.02
DQN_OTHER_PHASE_EPS_MAX = 0.2
DQN_OTHER_PHASE_EPS_DECAY = 0.5

# bufor
DQN_BUFFER_SIZE = 500_000
DQN_BUFFER_WARMUP = 25_000

# Dla macbooka
# DQN_BATCH_SIZE = 64

# Dla Borian komputer
DQN_BATCH_SIZE = 256

# Jedna aktualizacja gradientu przypada na cztery nowe decyzje ucznia.
# Zmniejsza to wielokrotne trenowanie na tych samych rekordach i pozwala
# zebrać więcej różnorodnych rozdań w ciągu jednej nocy.
DQN_UPDATE_RATIO = 0.25

# MacBook Air M2 ma cztery rdzenie wydajnościowe i cztery energooszczędne.
# Osiem procesów środowiska dało najwyższą łączną przepustowość w benchmarku.
# DQN_NUM_TRAIN_ENVS = 8

# Dla Borian komputer
DQN_NUM_TRAIN_ENVS = 4
# Test techniczny rozgrywa tylko jeden turniej, więc jeden proces testowy jest
# wystarczający. Pozostałe siedem procesów przez większość treningu było
# bezczynnych i jedynie zajmowało pamięć oraz zasoby systemowe.
DQN_NUM_TEST_ENVS = 1

# Mała sieć DQN z batchem 64 działa na tym komputerze szybciej na jednym
# wątku CPU niż na wielu wątkach albo przez Apple MPS.
TORCH_NUM_THREADS = 1
TORCH_NUM_INTEROP_THREADS = 1

# 250 tys. decyzji ucznia daje zbliżoną liczbę ruchów stołu do poprzedniego
# eksperymentu liczącego milion akcji wszystkich czterech graczy. Dzięki temu
# pierwszy poprawiony run można uczciwie porównać czasowo z poprzednim.
DQN_MAX_EPOCHS = 25
DQN_STEPS_PER_EPOCH = 10_000
# Kolektor przeplata 1000 nowych decyzji ucznia z aktualizacjami sieci.
# Wartość dzieli 10 000 bez reszty, więc nie powstają nadmiarowe kroki.
DQN_COLLECTION_STEPS = 1_000

# Walidacja co 25 tys. decyzji zachowuje dziesięć punktów kontrolnych podczas
# fazy porównywalnej z dawnym milionem ruchów całego stołu.
DQN_EVAL_INTERVAL_DECISIONS = 25_000

# Oprócz nadpisywanego stanu awaryjnego zachowujemy pełny stan co 10 epok.
# Te pliki zawierają również target network i optymalizator, dlatego mogą być
# bezpiecznym początkiem kolejnej fazy albo wznowienia eksperymentu.
DQN_FULL_STATE_INTERVAL_DECISIONS = 100_000

#TODO: WAŻNE jeśli trenowałbtś na macbooku to musisz tu pozmieniać rzeczy

# ZMIENNE TRENINGOWE PPO
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.02
PPO_EPS_CLIP = 0.2


# bufor
PPO_BUFFER_SIZE = 16384
PPO_BATCH_SIZE = 2048

# Dla macbooka
PPO_NUM_TRAIN_ENVS = 8

# Dla Borian komputer
PPO_NUM_TRAIN_ENVS = 4
PPO_NUM_TEST_ENVS = 1

# 61 epok po 4096 kroków daje ~250 000 decyzji ucznia (podobnie jak w DQN)
PPO_MAX_EPOCHS = 305
PPO_STEPS_PER_EPOCH = 16384
PPO_REPEAT_PER_COLLECT = 4

PPO_EVAL_INTERVAL_DECISIONS = 25_000
PPO_FULL_STATE_INTERVAL_DECISIONS = 100_000
# EWALUACJA

# Jeden „mecz” ewaluacyjny trwa najwyżej 100 rozdań. To ważne zwłaszcza dla
# botów Check/Call: potrafią grać bardzo długo i poprzedni limit 1000 akcji
# ucinał większość turniejów w przypadkowym momencie. Stała liczba rozdań daje
# porównywalną próbkę do głównej metryki bb/100.
EVAL_MAX_HANDS_PER_MATCH = 100

# Osobny, wysoki bezpiecznik chroni przed błędem środowiska powodującym
# nieskończoną pętlę. Osiągnięcie tego limitu jest raportowane jako awaria,
# w przeciwieństwie do planowego zakończenia po 100 rozdaniach.
EVAL_MAX_ACTIONS_PER_MATCH = 5_000
