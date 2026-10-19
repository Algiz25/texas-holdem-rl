#!/bin/zsh

# Dłuższa faza 2 DQN inspirowana udanym eksperymentem PPO kolegi. Profil
# zwiększa udział self-play do 70%, zapisuje snapshoty co 100 tys. decyzji,
# utrzymuje większą ligę i dodaje nagrodę turniejową. Sam plik niczego nie
# uruchamia — trening zaczyna się dopiero po ręcznym wywołaniu skryptu.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
PARENT_DIR="${PROJECT_DIR:h}"
cd "$PROJECT_DIR"

# Zapobiegaj uśpieniu MacBooka podczas treningu. Ctrl+C nadal bezpiecznie
# zatrzymuje proces i zapisuje pełny stan umożliwiający późniejsze wznowienie.
if [[ "${DQN_SELFPLAY_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_SELFPLAY_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x "${PARENT_DIR}/texas-holdem-rl/.venv/bin/python" ]]; then
    PYTHON="${PARENT_DIR}/texas-holdem-rl/.venv/bin/python"
else
    echo "[BŁĄD] Nie znaleziono Pythona w .venv. Najpierw utwórz środowisko."
    exit 1
fi

BASE_MODEL="${PROJECT_DIR}/checkpoints/dqn/REAL_BEST/phase_2/step_010500000.pth"
if [[ ! -f "$BASE_MODEL" ]]; then
    echo "[BŁĄD] Brakuje modelu bazowego: $BASE_MODEL"
    exit 1
fi

RUN_NAME="dqn_phase2_ppo_style_seed_16001"
SEED=16001

# 20 mln decyzji odpowiada skali udanej fazy 2 PPO. Na tym MacBooku będzie to
# więcej niż jedna noc; run można przerwać Ctrl+C i wznowić tą samą komendą.
TOTAL_DECISIONS=20000000
EVALUATION_INTERVAL=250000
EPSILON_TAU_DECISIONS=8000000

RUN_DIR="checkpoints/dqn/${RUN_NAME}"
FINAL_STATE="${RUN_DIR}/training_state_final.pth"
LATEST_STATE="${RUN_DIR}/training_state_latest.pth"
LOG_DIR="logs/dqn/selfplay_terminal"
mkdir -p "$LOG_DIR"
TERMINAL_LOG="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Są to wyłącznie testy kodu; nie rozgrywają pełnej ewaluacji ani treningu.
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -q

if [[ -f "$FINAL_STATE" ]]; then
    echo "[GOTOWE] ${RUN_NAME} osiągnął ${TOTAL_DECISIONS} decyzji."
    exit 0
fi

EXTRA_ARGS=()
DECISIONS="$TOTAL_DECISIONS"
if [[ -f "$LATEST_STATE" ]]; then
    COMPLETED="$($PYTHON -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["completed_env_steps"]))' "$LATEST_STATE")"
    DECISIONS=$((TOTAL_DECISIONS - COMPLETED))
    DECISIONS=$((DECISIONS / 10000 * 10000))
    if (( DECISIONS <= 0 )); then
        echo "[GOTOWE] Stan zawiera już ${COMPLETED} decyzji."
        exit 0
    fi
    EXTRA_ARGS=(--resume "$LATEST_STATE")
    echo "[WZNOWIENIE] ${RUN_NAME}: ${COMPLETED}/${TOTAL_DECISIONS} decyzji."
fi

echo "[START] DQN self-play ppo_style: ${RUN_NAME}."
echo "[LIGA] 70% self-play, do 20 modeli, snapshot co 100 000 decyzji."
echo "[NAGRODA] Żetony + premia turniejowa −3/−1/+1/+3."
echo "[STOP] Ctrl+C jeden raz zapisze stan do późniejszego wznowienia."
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
    --phase 2 \
    --self-play-profile ppo_style \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --decisions "$DECISIONS" \
    --epsilon-tau-decisions "$EPSILON_TAU_DECISIONS" \
    --evaluation-interval "$EVALUATION_INTERVAL" \
    --placement-reward-weight 1 \
    --base-model "$BASE_MODEL" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$TERMINAL_LOG"

echo "Trening zakończony albo bezpiecznie zatrzymany."
echo "Modele i raporty: ${RUN_DIR}"
echo "TensorBoard: ${PYTHON} -m tensorboard.main --logdir logs/dqn"
