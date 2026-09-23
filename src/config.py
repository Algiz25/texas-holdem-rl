# ZMIENNE ŚRODOWISKOWE
# póki co jedyna dostępna faza to 1
TRAINING_PHASE = 1
ACTION_SPACE = 5    # nie zmieniać
OBSERVATION_SIZE = 68
STARTING_CHIPS = 200
NUM_PLAYERS = 4     # nie zmieniać

# ZMIENNE TRENINGOWE

# co ile epok przeciwnicy zamieniają się na nowszych - # TODO: może warto zwiększyć
# jest to też co ile epok wypisywane są statystyki dla prostowy
OPPONENT_UPDATE_INTERVAL = 10
NUM_TOURNAMENTS_PER_EVAL = 10   # liczba turnieji rozegranych do zyskania statystyk

# ZMIENNE TRENINGOWE DQN
DQN_LEARNING_RATE = 1e-4
DQN_GAMMA = 0.99
DQN_TARGET_NET_UPDATE = 5000 # TODO: sprawdzić czy to dobra ilość

# epsilony
DQN_EPS_MAX = 1.0
DQN_RAND_PHASE_EPS_MIN = 0.1
DQN_RAND_PHASE_EPS_DECAY = 0.8

DQN_OTHER_PHASE_EPS_MIN = 0.02
DQN_OTHER_PHASE_EPS_MAX = 0.2
DQN_OTHER_PHASE_EPS_DECAY = 0.5

DQN_OPONENT_EPS = 0.05

# bufor
DQN_BUFFER_SIZE = 500_000
DQN_BUFFER_WARMUP = 10_000  #niezbyt ważne
DQN_BATCH_SIZE = 64

# środowiska
DQN_NUM_TRAIN_ENVS = 2
DQN_NUM_TEST_ENVS = 1
DQN_MAX_EPOCHS = 100        # DŁUGOŚĆ TRENINGU
DQN_STEPS_PER_EPOCH = 10_000

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



