import os
import sys
from pathlib import Path

# Dodajemy folder 'src' do ścieżki, żeby Python widział nasze moduły
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from evaluation.evaluator_iqn import IQNEvaluator

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================
# Ścieżka do Twojego Zwycięskiego Modelu z Fazy 2:
MODEL_PATH = Path(r"checkpoints\iqn\non_local\steps_p2_42\step_004700000.pth")

# Ścieżka do modelu z Fazy 1 (potrzebna, by móc zagrać przeciwko 'baseline'):
BASELINE_MODEL = Path(r"checkpoints\iqn\non_local\steps_p1_21372137\step_001600000.pth")

# Folder docelowy na ogromny raport końcowy:
REPORT_DIR = Path(r"checkpoints\iqn\non_local\steps_p2_42\final_master_eval")
SUITES = ("baseline","phase1_mix")

# ==========================================
# USTAWIENIA TESTU
# ==========================================
NUM_TOURNAMENTS = 5000
SEED_BASE = 123456  # Nowy, unikalny seed
MODEL_STEP = 1600000 # Tylko do oznaczenia w pliku CSV/JSON

def main():
    if not MODEL_PATH.exists():
        print(f"[BŁĄD] Nie znaleziono modelu głównego: {MODEL_PATH}")
        return
        
    print(f"{'='*60}")
    print(f"ROZPOCZYNAM WIELKĄ EWALUACJĘ OSTATECZNĄ")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Ilość turniejów: {NUM_TOURNAMENTS} (Może to potrwać dłuższą chwilę!)")
    print(f"{'='*60}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = IQNEvaluator(
        num_tournaments=NUM_TOURNAMENTS,
        model_path=MODEL_PATH,
        baseline_model_path=BASELINE_MODEL,
        training_phase=2,
        report_dir=REPORT_DIR
    )

    # Uruchamiamy test. Używamy "baseline" oraz "phase1_mix", by mieć
    # materiał do wygenerowania heatmap dla obu typów przeciwników.
    evaluator.evaluate(
        suites=SUITES,
        stage="final_master_5000",
        step=MODEL_STEP,
        seed_base=SEED_BASE,
        save_report=True,
        learner_mode="model"
    )

    print(f"\n[ZAKOŃCZONO] Sukces! Plik JSON dla heatmap znajduje się w:")
    print(f"{REPORT_DIR}")

if __name__ == "__main__":
    main()