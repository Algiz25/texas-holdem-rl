import os
import sys
import re
from pathlib import Path

# Dodanie folderu src do ścieżki Pythona
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
from paths import PPO_CHECKPOINT_DIR
from evaluation.evaluator_ppo import PPOEvaluator
import evaluation.evaluator as eval_module
from evaluation.evaluator import BasePokerEvaluator

# =====================================================================
# TRIK PRZYSPIESZAJĄCY: Obejście blokady jednowątkowej dla jednego zestawu
# =====================================================================
# 1. Rejestrujemy wirtualne zestawy bazowe jako "legalne"
fake_suites = ("baseline_part_1", "baseline_part_2", "baseline_part_3", "baseline_part_4")
eval_module.EVALUATION_SUITES = tuple(list(eval_module.EVALUATION_SUITES) + list(fake_suites))

# 2. Tłumaczymy środowisku, że wirtualny zestaw to w rzeczywistości "baseline"
original_opponent_action = BasePokerEvaluator._opponent_action

@classmethod
def patched_opponent_action(cls, opponent_type, *args, **kwargs):
    if opponent_type.startswith("baseline_part"):
        opponent_type = "baseline"
    return original_opponent_action.__func__(cls, opponent_type, *args, **kwargs)

BasePokerEvaluator._opponent_action = patched_opponent_action
# =====================================================================


def main():
    run_name = "phase2_selfplay_seed_11001"
    checkpoint_dir = PPO_CHECKPOINT_DIR / run_name
    baseline_model_path = PPO_CHECKPOINT_DIR / "overnight_ppo_seed_11001" / "best.pth"
    
    report_dir = checkpoint_dir / "evaluations_post"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir.exists():
        print(f"[BŁĄD] Katalog {checkpoint_dir} nie istnieje.")
        return

    # Zbieranie modeli (jeśli chcesz badać np. co drugi checkpoint, zmień na .glob("step_*.pth"))[::2]
    step_files = sorted(checkpoint_dir.glob("step_*.pth")) 
    
    other_files = []
    for name in ["best.pth", "final.pth"]:
        p = checkpoint_dir / name
        if p.exists():
            other_files.append(p)
            
    all_models = step_files + other_files

    if not all_models:
        print("[BŁĄD] Nie znaleziono żadnych plików .pth do ewaluacji.")
        return

    print(f"Znaleziono {len(all_models)} modeli do ewaluacji w '{run_name}'.")

    for i, model_path in enumerate(all_models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(all_models)}] Ewaluacja: {model_path.name} (4 procesy x 250 turniejów = 1000)")
        print(f"{'='*60}")
        
        match = re.search(r"step_(\d+)", model_path.name)
        step = int(match.group(1)) if match else 0
        
        stage = "post_eval"
        if model_path.name == "best.pth": stage = "post_eval_best"
        elif model_path.name == "final.pth": stage = "post_eval_final"

        # UWAGA: Ustawiamy tu 250 turniejów, bo odpalamy 4 zestawy naraz! (250 * 4 = 1000)
        evaluator = PPOEvaluator(
            num_tournaments=250, 
            model_path=model_path,
            baseline_model_path=baseline_model_path if baseline_model_path.exists() else None,
            training_phase=2,
            report_dir=report_dir
        )
        
        # Przekazujemy 4 wirtualne zestawy, aby wymusić rozbicie na 4 procesy CPU
        evaluator.evaluate(
            suites=fake_suites, 
            stage=stage,
            step=step,
            save_report=True,
            learner_mode="model"
        )
        
    print("\n[SUKCES] Ewaluacja wszystkich checkpointów zakończona!")

if __name__ == "__main__":
    main()