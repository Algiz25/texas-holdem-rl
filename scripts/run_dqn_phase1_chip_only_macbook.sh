#!/bin/zsh

# Stabilny nocny trening pierwszej fazy DQN dla MacBooka Air M2 z 16 GB RAM.
# Agent uczy się wyłącznie na zmianach żetonów. Wysoki limit jest bezpiecznikiem:
# rano użytkownik zatrzymuje proces jednym Ctrl+C, a trener zapisuje pełny stan.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

# Utrzymujemy aktywny system, ekran i dysk. Flaga -s działa przy podłączonym
# zasilaczu, dlatego całonocny trening należy wykonywać na ładowarce.
if [[ "${DQN_PHASE1_CHIP_ONLY_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_PHASE1_CHIP_ONLY_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

PYTHON=".venv/bin/python"
# Pierwszy start zakończył się jeszcze przed aktualizacją wag z powodu błędu
# integracji harmonogramu LR z Tianshou. Osobna nazwa zachowuje jego raporty
# diagnostyczne, ale nie miesza ich z czystym, ponowionym eksperymentem.
RUN_NAME="dqn_phase1_chip_only_stable_seed_14001_retry"
SEED=14001
TOTAL_DECISIONS=100000000
EPSILON_TAU_DECISIONS=1500000
EVALUATION_INTERVAL=250000
PLACEMENT_REWARD_WEIGHT=0.0

# Testy zatrzymują start przed wielogodzinnym eksperymentem, jeżeli zmieniły
# się obserwacje, nagroda, maskowanie akcji albo harmonogramy DQN.
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -q

RUN_DIR="checkpoints/dqn/${RUN_NAME}"
LATEST_STATE="${RUN_DIR}/training_state_latest.pth"
FINAL_STATE="${RUN_DIR}/training_state_final.pth"
DECISIONS="$TOTAL_DECISIONS"
EXTRA_ARGS=()

if [[ -f "$FINAL_STATE" ]]; then
    echo "[GOTOWE] ${RUN_NAME} osiągnął techniczny limit ${TOTAL_DECISIONS} decyzji."
    exit 0
fi

# Po przerwaniu kolejny start kontynuuje wagi, target network, optymalizator i
# liczniki harmonogramów. Replay buffer nie jest zapisywany, aby checkpoint nie
# miał kilku GB; przed dalszym uczeniem zostanie ponownie rozgrzany.
if [[ -f "$LATEST_STATE" ]]; then
    COMPLETED="$($PYTHON -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' "$LATEST_STATE")"
    DECISIONS=$((TOTAL_DECISIONS - COMPLETED))
    DECISIONS=$((DECISIONS / 10000 * 10000))
    if (( DECISIONS <= 0 )); then
        echo "[GOTOWE] Stan osiągnął techniczny limit ${TOTAL_DECISIONS} decyzji."
        exit 0
    fi
    EXTRA_ARGS=(--resume "$LATEST_STATE")
    echo "[WZNOWIENIE] ${RUN_NAME}: ${COMPLETED}/${TOTAL_DECISIONS} decyzji."
fi

echo "[START] ${RUN_NAME}: do ${DECISIONS} nowych decyzji DQN."
echo "[NAGRODA] Tylko zmiana żetonów; premia za miejsce jest wyłączona."
echo "[PAMIĘĆ] Replay buffer 1 000 000 przejść, 8 środowisk, 1 wątek PyTorch."
echo "[HARMONOGRAM] LR 1e-4→5e-5→2e-5; update ratio 0.25→0.15→0.10."
echo "[STOP] Rano naciśnij Ctrl+C jeden raz. Stan zostanie zapisany automatycznie."

PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --decisions "$DECISIONS" \
    --epsilon-tau-decisions "$EPSILON_TAU_DECISIONS" \
    --evaluation-interval "$EVALUATION_INTERVAL" \
    --placement-reward-weight "$PLACEMENT_REWARD_WEIGHT" \
    "${EXTRA_ARGS[@]}"

echo "[GOTOWE] Modele i raporty: ${RUN_DIR}"
echo "[WYKRESY] .venv/bin/tensorboard --logdir logs/dqn"
