from observation.schema import OBSERVATION_SIZE


# ZMIENNE ŚRODOWISKOWE
# póki co jedyna dostępna faza to 1
TRAINING_PHASE = 2
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

# DQN korzysta z własnych profili przeciwników. Nie zmieniamy powyższych wag,
# ponieważ są używane przez treningi PPO, SAC oraz IQN dodane na mainie.
DQN_PHASE1_OPPONENT_WEIGHTS = {
    "random": 0.45,
    "passive": 0.35,
    "mixed": 0.20,
}
DQN_PHASE2_OPPONENT_WEIGHTS = {
    "historical_self": 0.40,
    "latest_self": 0.10,
    "mixed": 0.25,
    "random": 0.15,
    "passive": 0.10,
}
DQN_PHASE2_PPO_STYLE_OPPONENT_WEIGHTS = {
    "historical_self": 0.40,
    "latest_self": 0.30,
    "mixed": 0.20,
    "random": 0.00,
    "passive": 0.10,
}
DQN_PHASE2_MAX_HISTORICAL_MODELS = 6
DQN_PHASE2_PPO_STYLE_MAX_HISTORICAL_MODELS = 20

# Rozkład akcji osobowości Mixed jest wspólny dla treningu i ewaluacji.
# Jedno źródło zapobiega sytuacji, w której bot o tej samej nazwie zachowuje
# się inaczej podczas zbierania doświadczeń i podczas pomiaru checkpointu.
MIXED_ACTION_WEIGHTS = (0.20, 0.45, 0.18, 0.12, 0.05)
AGGRESSIVE_ACTION_WEIGHTS = (0.05, 0.15, 0.30, 0.40, 0.10)

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
# Faza 1 zaczyna od większego kroku, a następnie go zmniejsza. Pozwala to
# szybko nauczyć się podstaw bez ciągłego nadpisywania dobrej polityki pod
# koniec długiego treningu.
DQN_LEARNING_RATE = 1e-4
DQN_PHASE1_LR_AFTER_500K = 5e-5
DQN_PHASE1_LR_AFTER_2M = 2e-5
DQN_PHASE1_LR_FIRST_BOUNDARY = 500_000
DQN_PHASE1_LR_SECOND_BOUNDARY = 2_000_000
DQN_GAMMA = 0.99
DQN_TARGET_NET_UPDATE = 5_000
DQN_HUBER_LOSS_DELTA = 1.0
DQN_PHASE1_PLACEMENT_REWARD_WEIGHT = 0.0

# epsilony
DQN_EPS_MAX = 1.0
DQN_RAND_PHASE_EPS_MIN = 0.15
DQN_PHASE1_EPS_TAU_STEPS = 1_500_000

# Faza 2 startuje z wytrenowanego modelu, więc nie wraca do pełnej losowości.
DQN_PHASE2_EPS_MAX = 0.20
DQN_PHASE2_EPS_MIN = 0.05
DQN_PHASE2_EPS_TAU_STEPS = 1_500_000
DQN_PHASE2_LEARNING_RATE = 2e-5
DQN_PHASE2_LEARNING_RATE_AFTER_1M = 1e-5
DQN_PHASE2_LR_BOUNDARY = 1_000_000
DQN_PHASE2_UPDATE_RATIO = 0.10
DQN_PHASE2_BUFFER_WARMUP = 50_000
DQN_PHASE2_HISTORY_INTERVAL_DECISIONS = 500_000
DQN_PHASE2_EVAL_INTERVAL_DECISIONS = 250_000
DQN_PHASE2_PLACEMENT_REWARD_WEIGHT = 0.0

# Alternatywny profil self-play odtwarza proporcje ligi użyte w udanej fazie
# drugiej PPO. Jest wybierany jawnie flagą CLI i nie zmienia ustawień PPO.
DQN_PHASE2_PPO_STYLE_BUFFER_SIZE = 500_000
DQN_PHASE2_PPO_STYLE_UPDATE_RATIO = 0.15
DQN_PHASE2_PPO_STYLE_LEARNING_RATE = 5e-5
DQN_PHASE2_PPO_STYLE_LEARNING_RATE_AFTER_BOUNDARY = 2e-5
DQN_PHASE2_PPO_STYLE_LR_BOUNDARY = 5_000_000
DQN_PHASE2_PPO_STYLE_HISTORY_INTERVAL_DECISIONS = 100_000
DQN_PHASE2_PPO_STYLE_OPPONENT_EPSILON = 0.05

# bufor
DQN_BUFFER_SIZE = 1_000_000
DQN_BUFFER_WARMUP = 25_000
DQN_BATCH_SIZE = 64
DQN_UPDATE_RATIO = 0.25
DQN_PHASE1_UPDATE_RATIO_AFTER_500K = 0.15
DQN_PHASE1_UPDATE_RATIO_AFTER_2M = 0.10
DQN_PHASE1_UPDATE_FIRST_BOUNDARY = 500_000
DQN_PHASE1_UPDATE_SECOND_BOUNDARY = 2_000_000
DQN_TENSORBOARD_UPDATE_INTERVAL = 1
DQN_NUM_TRAIN_ENVS = 8
DQN_NUM_TEST_ENVS = 1

DQN_MAX_EPOCHS = 25
DQN_STEPS_PER_EPOCH = 10_000
DQN_COLLECTION_STEPS = 1_000

DQN_EVAL_INTERVAL_DECISIONS = 25_000
DQN_FULL_STATE_INTERVAL_DECISIONS = 250_000

# DQN używa większych prób niż domyślne testy innych algorytmów. Walidacja
# wybiera checkpoint, a końcowe 1000 turniejów daje węższe przedziały ufności.
DQN_EVAL_TOURNAMENTS_PER_SUITE = 300
DQN_FINAL_EVAL_TOURNAMENTS_PER_SUITE = 1_000

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
SAC_GAMMA = 0.99
SAC_TAU = 0.005
SAC_AUTO_ALPHA = True
SAC_ALPHA = 0.05  # Wartość startowa
SAC_ALPHA_LR = 3e-4

SAC_BATCH_SIZE = 512
SAC_NUM_TRAIN_ENVS = 4
SAC_NUM_TEST_ENVS = 1

SAC_MAX_EPOCHS = 50
SAC_STEPS_PER_EPOCH = 10_000
SAC_COLLECTION_STEPS = 1_000

SAC_EVAL_INTERVAL_DECISIONS = 50_000
SAC_FULL_STATE_INTERVAL_DECISIONS = 100_000

if TRAINING_PHASE == 1:
    SAC_ACTOR_LR = 1e-4
    SAC_CRITIC_LR = 1e-4
    SAC_TARGET_ENTROPY_RATIO = 0.15
    SAC_UPDATE_RATIO = 0.5
    SAC_BUFFER_SIZE = 500_000
    SAC_BUFFER_WARMUP = 10_000
else:
    SAC_ACTOR_LR = 5e-5
    SAC_CRITIC_LR = 5e-5
    SAC_TARGET_ENTROPY_RATIO = 0.25
    SAC_UPDATE_RATIO = 0.25
    SAC_BUFFER_SIZE = 1_000_000
    SAC_BUFFER_WARMUP = 10_000


# IQN
IQN_GAMMA = 0.99
IQN_TARGET_NET_UPDATE = 5000
IQN_SAMPLE_SIZE = 32
IQN_ONLINE_SAMPLE_SIZE = 16
IQN_TARGET_SAMPLE_SIZE = 16
IQN_NUM_COSINES = 64
IQN_N_STEP = 3
IQN_BATCH_SIZE = 256
IQN_NUM_TRAIN_ENVS = 4
IQN_NUM_TEST_ENVS = 1
IQN_MAX_EPOCHS = 25
IQN_STEPS_PER_EPOCH = 10_000
IQN_COLLECTION_STEPS = 1_000
IQN_EVAL_INTERVAL_DECISIONS = 50_000
IQN_FULL_STATE_INTERVAL_DECISIONS = 100_000

if TRAINING_PHASE == 1:
    IQN_LEARNING_RATE = 5e-5
    IQN_BUFFER_SIZE = 500_000
    IQN_BUFFER_WARMUP = 10_000
    IQN_UPDATE_RATIO = 0.5
    
    # epsilony - Faza 1
    IQN_EPS_MAX = 1.0
    IQN_EPS_MIN = 0.1
    IQN_EPS_TAU = 500_000 
else:
    # Faza 2 (Self-Play)
    IQN_LEARNING_RATE = 1e-5
    IQN_BUFFER_SIZE = 1_000_000
    IQN_BUFFER_WARMUP = 10_000
    IQN_UPDATE_RATIO = 0.25
    
    # epsilony - Faza 2
    IQN_EPS_MAX = 0.15
    IQN_EPS_MIN = 0.05
    IQN_EPS_TAU = 1_000_000

# EWALUACJA

if TRAINING_PHASE == 1:
    EVAL_SUITES = ("random", "passive", "mixed", "phase1_mix")
else:
    EVAL_SUITES = ("phase1_mix", "baseline")

EVAL_MAX_HANDS_PER_MATCH = 100
EVAL_MAX_ACTIONS_PER_MATCH = 5_000
