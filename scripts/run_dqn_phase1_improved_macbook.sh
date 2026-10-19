#!/bin/zsh

# Kontrolowana, pięciomilionowa faza 1 DQN. Skrypt uruchamia jeden model od
# zera, zapisuje checkpoint co 250 tys. decyzji i korzysta z terminalnej
# nagrody za miejsce razem z nagrodą żetonową.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

# Utrzymujemy komputer, ekran i dysk aktywne. Flaga -s wymaga podłączonego
# zasilacza, dlatego dłuższy trening należy wykonywać na ładowarce.
if [[ "${DQN_PHASE1_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_PHASE1_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

PYTHON=".venv/bin/python"
RUN_NAME="dqn_phase1_huber_placement_seed_13001"
SEED=13001
TOTAL_DECISIONS=5000000
EPSILON_TAU_DECISIONS=700000
EVALUATION_INTERVAL=250000
PLACEMENT_REWARD_WEIGHT=1.0

# Testy trwają kilka sekund i zatrzymują start, jeżeli środowisko, nagroda lub
# maskowanie legalnych akcji zostały przypadkowo uszkodzone.
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -q

RUN_DIR="checkpoints/dqn/${RUN_NAME}"
LATEST_STATE="${RUN_DIR}/training_state_latest.pth"
FINAL_STATE="${RUN_DIR}/training_state_final.pth"
DECISIONS="$TOTAL_DECISIONS"
EXTRA_ARGS=()

if [[ -f "$FINAL_STATE" ]]; then
    echo "[GOTOWE] ${RUN_NAME} ukończył już ${TOTAL_DECISIONS} decyzji."
    exit 0
fi

# Po Ctrl+C kolejny start kontynuuje ten sam eksperyment. Replay buffer jest
# ponownie rozgrzewany, ale sieć, target network i optymalizator są zachowane.
if [[ -f "$LATEST_STATE" ]]; then
    COMPLETED="$($PYTHON -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' "$LATEST_STATE")"
    DECISIONS=$((TOTAL_DECISIONS - COMPLETED))
    DECISIONS=$((DECISIONS / 10000 * 10000))
    if (( DECISIONS <= 0 )); then
        echo "[GOTOWE] Stan osiągnął limit ${TOTAL_DECISIONS} decyzji."
        exit 0
    fi
    EXTRA_ARGS=(--resume "$LATEST_STATE")
    echo "[WZNOWIENIE] ${RUN_NAME}: ${COMPLETED}/${TOTAL_DECISIONS} decyzji."
fi

echo "[START] ${RUN_NAME}: ${DECISIONS} nowych decyzji DQN."
echo "[STOP] Ctrl+C zapisze model i pełny stan do późniejszego wznowienia."
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --decisions "$DECISIONS" \
    --epsilon-tau-decisions "$EPSILON_TAU_DECISIONS" \
    --evaluation-interval "$EVALUATION_INTERVAL" \
    --placement-reward-weight "$PLACEMENT_REWARD_WEIGHT" \
    "${EXTRA_ARGS[@]}"

echo "[GOTOWE] Model i raporty: ${RUN_DIR}"
echo "[WYKRESY] .venv/bin/tensorboard --logdir logs/dqn"
