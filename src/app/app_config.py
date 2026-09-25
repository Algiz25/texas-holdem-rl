"""Konfiguracja dla pliku generującego 1 turniej i zapisującego go do pliku .json"""

from pathlib import Path

# Root do wczytywania modeli z folderu checkpoints
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
dqn_model_path = PROJECT_ROOT / "checkpoints" / "dqn" / "best.pth"
ppo_model_path = PROJECT_ROOT / "checkpoints" / "ppo" / "best.pth"

# Tablica decydująca jacy gracze siadają przy stole
# typ: dqn, ppo, mixed, aggressive, passive, random
# path: potrzebny tylko dla dqn oraz ppo (lokalizacja pliku z wagami modelu)
# name: nazwa gracza wyświetlana w UI
# seed: można podać seed dla mixed
TABLE_CONFIG = {
        "player_0": {"type": "dqn", "path": str(dqn_model_path), "name": "DQN"},
        "player_1": {"type": "ppo", "path": str(ppo_model_path), "name": "PPO"},
        "player_2": {"type": "mixed", "seed": 621, "name": "Miesiany miesiany"},
        "player_3": {"type": "passive", "name": "Pasywny"}
    }

OUTPUT_LOCATION = PROJECT_ROOT / "src" / "app" / "replay_files"
OUPUT_FILE_NAME = "tournament_history" + ".json"
