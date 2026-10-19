#!/bin/zsh

# Dokładna ewaluacja siedmiu kandydatów z fazy 1. Każdy model rozgrywa po 1000
# turniejów przeciwko czterem zestawom botów, zawsze na tych samych seedach.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

# Ewaluacja może potrwać kilkadziesiąt minut. Utrzymujemy aktywny komputer i
# ekran; test należy wykonywać przy podłączonym zasilaczu.
if [[ "${DQN_FINAL_COMPARISON_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_FINAL_COMPARISON_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

PYTHON=".venv/bin/python"

# Krótkie testy techniczne chronią przed kosztowną ewaluacją uszkodzonego kodu.
PYTHONPATH=src "$PYTHON" -m unittest discover -s tests -q

echo "[START] 7 checkpointów × 4 zestawy × 1000 turniejów = 28 000 turniejów."
echo "[WZNOWIENIE] Ponowny start automatycznie pominie ukończone checkpointy."
echo "[STOP] Możesz użyć Ctrl+C; rozpoczęty checkpoint zostanie powtórzony."

PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" \
    -m evaluation.compare_dqn_checkpoints \
    --run-name dqn_phase1_chip_only_stable_seed_14001_retry \
    --steps 750000 1250000 10250000 10500000 10750000 11500000 13250000 \
    --tournaments 1000 \
    --seed-base 90260 \
    --bootstrap-repetitions 5000
