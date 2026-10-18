import os
import sys
from pathlib import Path

# Dodajemy folder 'src' do ścieżki, żeby Python widział nasze moduły
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from evaluation.evaluator_iqn import IQNEvaluator

# ==========================================
# KONFIGURACJA ŚCIEŻEK (DO SPRAWDZENIA)
# ==========================================
# Folder, w którym znajdują się Twoje modele z Fazy 2:
RUN_DIR = Path(r"checkpoints\iqn\non_local\steps_p2_42") 

# Ścieżka do najlepszego modelu z Fazy 1 (nasz Baseline):
BASELINE_MODEL = Path(r"checkpoints\iqn\REAL_BEST\phase_1\seed_001600000\step_001600000.pth")

# Folder, do którego trafią wyniki tego wielkiego testu:
REPORT_DIR = RUN_DIR / "evaluations_top10"

SUITES = ("phase1_mix", "passive", "baseline")

# ==========================================
# USTAWIENIA TESTU
# ==========================================
NUM_TOURNAMENTS = 500
SEED_BASE = 99999  # Nowy, niezależny seed dla gwarancji rzetelności

# Naszych 10 wytypowanych kandydatów:
TOP_10_STEPS = [0, 100000, 200000, 300000, 
                400000, 500000, 600000, 
                700000, 800000, 900000, 
                1000000, 1100000, 1200000, 
                1300000, 1400000, 1500000, 
                1600000, 1700000, 1800000, 
                1900000, 2000000, 2100000, 
                2200000, 2300000, 2400000, 
                2500000, 2600000, 2700000, 
                2800000, 2900000, 3000000, 
                3100000, 3200000, 3300000, 
                3400000, 3500000, 3600000, 
                3700000, 3800000, 3900000, 
                4000000, 4100000, 4200000, 
                4300000, 4400000, 4500000, 
                4600000, 4700000, 4800000, 
                4900000, 5000000, 5100000, 
                5200000]

def main():
    print(f"Szukam modeli w: {RUN_DIR}")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    for step in TOP_10_STEPS:
        model_file = f"step_{step:09d}.pth"
        model_path = RUN_DIR / model_file
        
        if not model_path.exists():
            print(f"\n[POMINIĘTO] Nie znaleziono modelu: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"ROZPOCZYNAM EWALUACJĘ MODELU: {model_file}")
        print(f"Rozgrywanie {NUM_TOURNAMENTS} turniejów...")
        print(f"{'='*60}")
        
        evaluator = IQNEvaluator(
            num_tournaments=NUM_TOURNAMENTS,
            model_path=model_path,
            baseline_model_path=BASELINE_MODEL,
            training_phase=2,
            report_dir=REPORT_DIR
        )
        
        # Testujemy tylko na 2 najważniejszych zestawach dla Fazy 2:
        evaluator.evaluate(
            suites=SUITES,
            stage="post_eval_top10",
            step=step,
            seed_base=SEED_BASE,
            save_report=True,
            learner_mode="model"
        )
        
    print(f"\n[ZAKOŃCZONO] Raport ze wszystkich 10 modeli znajdziesz w folderze: {REPORT_DIR}")

if __name__ == "__main__":
    main()