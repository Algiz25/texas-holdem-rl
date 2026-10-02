#!/bin/zsh

# Porównanie dwóch funkcji nagrody na identycznym seedzie i identycznych
# hiperparametrach. Skrypt nie scala modeli: każdy wariant ma własny katalog
# checkpointów, log TensorBoard i raporty ewaluacji.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

# Laptop musi pozostać aktywny podczas obu przebiegów. Wymagane jest zasilanie,
# ponieważ `-s` utrzymuje system aktywny tylko po podłączeniu ładowarki.
if [[ "${DQN_REWARD_COMPARISON_CAFFEINATED:-0}" != "1" ]]; then
    exec env DQN_REWARD_COMPARISON_CAFFEINATED=1 caffeinate -dims "$0" "$@"
fi

PYTHON=".venv/bin/python"
SEED=12001
DECISIONS=1000000
EPSILON_TAU=700000
EVALUATION_INTERVAL=250000

run_variant() {
    local run_name="$1"
    local placement_reward_weight="$2"

    echo "[START] ${run_name}: placement reward weight ${placement_reward_weight}."
    PYTHONUNBUFFERED=1 PYTHONPATH=src "$PYTHON" src/training/train_dqn.py \
        --run-name "$run_name" \
        --seed "$SEED" \
        --decisions "$DECISIONS" \
        --epsilon-tau-decisions "$EPSILON_TAU" \
        --evaluation-interval "$EVALUATION_INTERVAL" \
        --placement-reward-weight "$placement_reward_weight"
}

# Wariant A odpowiada metryce bb/100: nagroda to wyłącznie zmiana żetonów.
run_variant "reward_compare_chip_only_seed_${SEED}" 0.0

# Wariant B odtwarza dawną premię turniejową. Jedyną różnicą względem A jest
# ta jedna liczba, więc późniejszy wynik można przypisać funkcji nagrody.
run_variant "reward_compare_chip_plus_placement_seed_${SEED}" 1.0

echo "[GOTOWE] Porównaj raporty evaluations_v3.csv w obu katalogach checkpointów."
