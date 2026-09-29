#!/bin/zsh

# Jeden długi run zachowuje replay buffer przez całą noc. Dzięki temu nie
# tracimy czasu na wielokrotny warm-up ani na rozpoczynanie nauki od zera.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

# Ponowne uruchomienie pod caffeinate utrzymuje aktywny system, ekran i dysk
# przez cały trening. Flaga `-s` działa przy podłączonym zasilaniu, dlatego
# nocny trening powinien odbywać się z ładowarką. Zmienna chroni przed pętlą.
if [[ "${DQN_OVERNIGHT_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_OVERNIGHT_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

PYTHON=".venv/bin/python"
RUN_NAME="overnight_long_seed_11001"
SEED=11001
# Wysoki limit jest tylko zabezpieczeniem. Rano użytkownik zatrzymuje trening
# jednym Ctrl+C; trener zapisuje wtedy wagi i pełny stan do wznowienia.
# Limit ma być wyraźnie wyższy od tego, co komputer może policzyć przez
# dziewięć godzin. Dzięki temu trening trwa aż do porannego Ctrl+C.
TOTAL_DECISIONS=100000000
EPSILON_DECAY_DECISIONS=700000
EVALUATION_INTERVAL=250000
NIGHT_ID="$(date +%Y%m%d_%H%M%S)"
TERMINAL_LOG_DIR="logs/dqn/overnight_${NIGHT_ID}"
mkdir -p "$TERMINAL_LOG_DIR"

# Zanim komputer zostanie bez nadzoru, szybkie testy wykrywają uszkodzoną
# instalację lub regresję środowiska. Pełny trening rusza tylko po ich sukcesie.
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -q

RUN_DIR="checkpoints/dqn/${RUN_NAME}"
FINAL_STATE="${RUN_DIR}/training_state_final.pth"
LATEST_STATE="${RUN_DIR}/training_state_latest.pth"
TERMINAL_LOG="${TERMINAL_LOG_DIR}/${RUN_NAME}.log"

if [[ -f "$FINAL_STATE" ]]; then
    echo "[GOTOWE] ${RUN_NAME} osiągnął już limit ${TOTAL_DECISIONS} decyzji."
    exit 0
fi

EXTRA_ARGS=()
DECISIONS="$TOTAL_DECISIONS"
if [[ -f "$LATEST_STATE" ]]; then
    COMPLETED="$($PYTHON -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' "$LATEST_STATE")"
    DECISIONS=$((TOTAL_DECISIONS - COMPLETED))
    if (( DECISIONS <= 0 )); then
        echo "[BŁĄD] Stan ma ${COMPLETED} decyzji, ale brak pliku końcowego."
        exit 1
    fi
    EXTRA_ARGS=(--resume "$LATEST_STATE")
    # Trener zapisuje epoki po 10 000 decyzji. Po ręcznym Ctrl+C licznik może
    # kończyć się na granicy krótszej kolekcji 1000, dlatego pozostały limit
    # zaokrąglamy w dół do pełnych epok. Maksymalna różnica to 9999 decyzji
    # przy technicznym limicie stu milionów.
    DECISIONS=$((DECISIONS / 10000 * 10000))
    if (( DECISIONS <= 0 )); then
        echo "[GOTOWE] ${RUN_NAME} jest już przy technicznym limicie."
        exit 0
    fi
    echo "[WZNOWIENIE] ${RUN_NAME}: ${COMPLETED}/${TOTAL_DECISIONS} decyzji."
fi

echo "[START] Jeden długi trening: ${RUN_NAME}."
echo "[STOP] Rano naciśnij Ctrl+C jeden raz. Stan zostanie zapisany automatycznie."
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --decisions "$DECISIONS" \
    --epsilon-decay-decisions "$EPSILON_DECAY_DECISIONS" \
    --evaluation-interval "$EVALUATION_INTERVAL" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$TERMINAL_LOG"

echo "Trening zakończony albo bezpiecznie zatrzymany."
echo "Model: ${RUN_DIR}"
echo "TensorBoard: .venv/bin/tensorboard --logdir logs/dqn"
