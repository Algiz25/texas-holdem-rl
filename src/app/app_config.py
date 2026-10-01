"""Konfiguracja dla pliku generującego 1 turniej i zapisującego go do pliku .json"""

from pathlib import Path

# Root do wczytywania modeli z folderu checkpoints
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dqn_model_path = PROJECT_ROOT / "checkpoints" / "dqn" / "REAL_BEST" / "phase_2" / "step_010500000.pth"
ppo_model_path = PROJECT_ROOT / "checkpoints" / "ppo" / "REAL_BEST" / "phase_2" / "step_014991360.pth"
sac_model_path = PROJECT_ROOT / "checkpoints" / "sac" / "REAL_BEST" / "phase_2" / "step_002600000.pth"
iqn_model_path = PROJECT_ROOT / "checkpoints" / "iqn" / "REAL_BEST" / "phase_2" / "step_001150000.pth"


# Tablica decydująca jacy gracze siadają przy stole
# typ: dqn, ppo, iqn, sac, mixed, aggressive, passive, random
# path: potrzebny dla dqn, ppo, sac, iqn (lokalizacja pliku z wagami modelu)
# name: nazwa gracza wyświetlana w UI
# seed: można podać seed dla mixed
TABLE_CONFIG = {
        "player_0": {"type": "dqn", "path": str(dqn_model_path), "name": "DQN"},
        "player_1": {"type": "ppo", "path": str(ppo_model_path), "name": "PPO"},
        "player_2": {"type": "sac", "path": str(sac_model_path), "name": "SAC"},
        "player_3": {"type": "iqn", "path": str(iqn_model_path), "name": "IQN"}
    }

OUTPUT_LOCATION = PROJECT_ROOT / "src" / "app" / "replay_files"
OUPUT_FILE_NAME = "tournament_history" + ".json"
