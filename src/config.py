from observation.schema import OBSERVATION_SIZE


# ZMIENNE ŚRODOWISKOWE
# póki co jedyna dostępna faza to 1
TRAINING_PHASE = 1
ACTION_SPACE = 5    # nie zmieniać
STARTING_CHIPS = 200
NUM_PLAYERS = 4     # nie zmieniać

TORCH_NUM_THREADS = 1
TORCH_NUM_INTEROP_THREADS = 1

# Faza 1 losuje osobowość każdego z trzech przeciwników niezależnie na
# początku turnieju. Brak wpisu "aggressive" jest celowy: na rozgrzewce model
# ma najpierw nauczyć się podstaw przeciwko łatwiejszemu stołowi.
PHASE1_OPPONENT_WEIGHTS = {
    "random": 0.50,
    "passive": 0.40,
    "mixed": 0.10,
}

PHASE2_OPPONENT_WEIGHTS = {
    "historical_self": 0.40,
    "latest_self": 0.30,
    "mixed": 0.20,
    "passive": 0.10,
}
PHASE2_MAX_HISTORICAL_MODELS = 100

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
EVAL_TOURNAMENTS_PER_SUITE = 200

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
DQN_LEARNING_RATE = 5e-5
DQN_GAMMA = 0.99
DQN_TARGET_NET_UPDATE = 10000 # TODO: sprawdzić czy to dobra ilość
DQN_HUBER_LOSS_DELTA = 1.0

# epsilony
DQN_EPS_MAX = 1.0
DQN_PHASE1_EPS_MIN = 0.1
DQN_PHASE1_EPS_TAU = 50_000 

# bufor
DQN_BUFFER_SIZE = 500_000
DQN_BUFFER_WARMUP = 10_000
DQN_BATCH_SIZE = 512
DQN_UPDATE_RATIO = 0.5
DQN_NUM_TRAIN_ENVS = 4
DQN_NUM_TEST_ENVS = 1

DQN_MAX_EPOCHS = 25
DQN_STEPS_PER_EPOCH = 10_000
DQN_COLLECTION_STEPS = 1_000

DQN_EVAL_INTERVAL_DECISIONS = 50_000
DQN_FULL_STATE_INTERVAL_DECISIONS = 100_000

# ZMIENNE TRENINGOWE PPO
if TRAINING_PHASE == 1:
    PPO_LEARNING_RATE = 3e-4
    PPO_ENT_COEF = 0.02
else:
    # W Fazie 2 (Self-Play) zmniejszamy LR i entropię dla większej stabilności
    PPO_LEARNING_RATE = 1e-4
    PPO_ENT_COEF = 0.01

PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_VF_COEF = 0.5
PPO_EPS_CLIP = 0.2


# bufor
PPO_BUFFER_SIZE = 16384
PPO_BATCH_SIZE = 2048

# Dla macbooka
# PPO_NUM_TRAIN_ENVS = 8

# Dla Borian komputer
PPO_NUM_TRAIN_ENVS = 4
PPO_NUM_TEST_ENVS = 1

# 61 epok po 4096 kroków daje ~250 000 decyzji ucznia (podobnie jak w DQN)

if TRAINING_PHASE == 1:
    PPO_MAX_EPOCHS = 611
else:
    PPO_MAX_EPOCHS = 1221

PPO_STEPS_PER_EPOCH = 16384
PPO_REPEAT_PER_COLLECT = 4

PPO_EVAL_INTERVAL_DECISIONS = 25_000
PPO_FULL_STATE_INTERVAL_DECISIONS = 100_000


# ZMIENNE TRENINGOWE SAC
SAC_ACTOR_LR = 1e-4
SAC_CRITIC_LR = 1e-4
SAC_ALPHA_LR = 3e-4
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_AUTO_ALPHA = True
SAC_ALPHA = 0.05  # Używane jako wartość początkowa jeśli AUTO_ALPHA=True, lub stała jeśli False
SAC_ALPHA_LR = 3e-4
SAC_TARGET_ENTROPY_RATIO = 0.15

SAC_BUFFER_SIZE = 500_000
SAC_BUFFER_WARMUP = 10_000
SAC_BATCH_SIZE = 1024
SAC_UPDATE_RATIO = 0.5

SAC_NUM_TRAIN_ENVS = 4
SAC_NUM_TEST_ENVS = 1

SAC_MAX_EPOCHS = 50
SAC_STEPS_PER_EPOCH = 10_000
SAC_COLLECTION_STEPS = 1_000

SAC_EVAL_INTERVAL_DECISIONS = 10_000
SAC_FULL_STATE_INTERVAL_DECISIONS = 100_000


# IQN
IQN_LEARNING_RATE = 5e-5
IQN_GAMMA = 0.99
IQN_TARGET_NET_UPDATE = 5000
IQN_SAMPLE_SIZE = 32
IQN_ONLINE_SAMPLE_SIZE = 32
IQN_TARGET_SAMPLE_SIZE = 32
IQN_NUM_COSINES = 64

# epsilony
IQN_EPS_MAX = 1.0
IQN_PHASE1_EPS_MIN = 0.1
IQN_PHASE1_EPS_TAU = 500_000 
IQN_N_STEP = 3

# bufor
IQN_BUFFER_SIZE = 500_000
IQN_BUFFER_WARMUP = 10_000
IQN_BATCH_SIZE = 512
IQN_UPDATE_RATIO = 0.5
IQN_NUM_TRAIN_ENVS = 4
IQN_NUM_TEST_ENVS = 1

IQN_MAX_EPOCHS = 25
IQN_STEPS_PER_EPOCH = 10_000
IQN_COLLECTION_STEPS = 1_000

IQN_EVAL_INTERVAL_DECISIONS = 50_000
IQN_FULL_STATE_INTERVAL_DECISIONS = 100_000

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
