#!/bin/zsh

# Kontrolowana faza 2 DQN. Startuje od najlepszego checkpointu fazy 1,
# zachowuje osobny replay buffer i katalog wyników, a przeciwników dobiera
# z małej ligi zamrożonych modeli oraz prostych botów. Skrypt sam nie jest
# uruchamiany przez repozytorium — trening zaczyna się dopiero po wywołaniu go
# przez użytkownika.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
PARENT_DIR="${PROJECT_DIR:h}"
cd "$PROJECT_DIR"

# Utrzymuj MacBooka aktywnego podczas długiego treningu. Flaga -s wymaga
# podłączonego zasilacza; -d i -i zapobiegają wygaszeniu ekranu oraz uśpieniu.
if [[ "${DQN_SELFPLAY_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_SELFPLAY_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

# Nowy worktree nie kopiuje wirtualnego środowiska. Najpierw używamy lokalnego,
# a jeśli go nie ma, bezpiecznie korzystamy z istniejącego środowiska projektu.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x "${PARENT_DIR}/texas-holdem-rl/.venv/bin/python" ]]; then
    PYTHON="${PARENT_DIR}/texas-holdem-rl/.venv/bin/python"
else
    echo "[BŁĄD] Nie znaleziono Pythona w .venv. Najpierw utwórz środowisko."
    exit 1
fi

# Najlepszy DQN jest już śledzony na mainie. Skrypt używa go bez kopiowania
# kolejnego pliku do pull requesta; kolejne snapshoty ligi powstaną dopiero w
# lokalnym katalogu użytkownika po ręcznym uruchomieniu treningu.
BASE_MODEL="${PROJECT_DIR}/checkpoints/dqn/REAL_BEST/phase_2/step_010500000.pth"
if [[ ! -f "$BASE_MODEL" ]]; then
    echo "[BŁĄD] Brakuje modelu bazowego: $BASE_MODEL"
    exit 1
fi

RUN_NAME="dqn_phase2_selfplay_seed_15001"
SEED=15001

# Sześć milionów nowych decyzji odpowiada mniej więcej dziewięciogodzinnemu
# treningowi na tym MacBooku (trzykrotność wcześniejszego, około 3-godzinnego
# planu na 2 mln decyzji). To nadal limit decyzji, a nie sztywny limit czasu,
# więc rzeczywisty czas zależy od szybkości ewaluacji i obciążenia komputera.
# Ctrl+C można nacisnąć wcześniej; pełny stan zostanie zapisany do wznowienia.
TOTAL_DECISIONS=6000000
EVALUATION_INTERVAL=250000
# Wydłużamy stałą czasową epsilon proporcjonalnie do całego treningu, aby agent
# nie zakończył eksploracji już po pierwszej ćwiartce nocnego self-play.
EPSILON_TAU_DECISIONS=4500000

RUN_DIR="checkpoints/dqn/${RUN_NAME}"
FINAL_STATE="${RUN_DIR}/training_state_final.pth"
LATEST_STATE="${RUN_DIR}/training_state_latest.pth"
LOG_DIR="logs/dqn/selfplay_terminal"
mkdir -p "$LOG_DIR"
TERMINAL_LOG="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Przed pozostawieniem komputera bez nadzoru uruchamiamy wszystkie testy.
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
        echo "[GOTOWE] Stan fazy 2 zawiera już ${COMPLETED} decyzji."
        exit 0
    fi
    EXTRA_ARGS=(--resume "$LATEST_STATE")
    echo "[WZNOWIENIE] ${RUN_NAME}: ${COMPLETED}/${TOTAL_DECISIONS} decyzji."
fi

echo "[START] Faza 2 DQN self-play: ${RUN_NAME}."
echo "[BAZA] ${BASE_MODEL}"
echo "[STOP] Naciśnij Ctrl+C jeden raz; wagi i stan do wznowienia zostaną zapisane."
PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
    --phase 2 \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --decisions "$DECISIONS" \
    --epsilon-tau-decisions "$EPSILON_TAU_DECISIONS" \
    --evaluation-interval "$EVALUATION_INTERVAL" \
    --placement-reward-weight 0 \
    --base-model "$BASE_MODEL" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$TERMINAL_LOG"

echo "Trening zakończony albo bezpiecznie zatrzymany."
echo "Modele i raporty: ${RUN_DIR}"
echo "TensorBoard: ${PYTHON} -m tensorboard.main --logdir logs/dqn"
