import os
import json
import glob
import pandas as pd

# =======================================================
# USTAW SWOJE ŚCIEŻKI PONIŻEJ
# =======================================================

# Ścieżka do folderu, w którym znajdują się pliki .json:
INPUT_DIR = r".\checkpoints\ppo\overnight_ppo_seed_11001\evaluations"

# Ścieżka i nazwa docelowego pliku CSV:
OUTPUT_FILE = r".\checkpoints\ppo\overnight_ppo_seed_11001\evaluations\evaluations_v3.csv"

# =======================================================


def flatten_evaluation_data(filepath):
    """
    Wczytuje plik JSON z ewaluacją i spłaszcza go do płaskiego słownika.
    """
    flattened_rows = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[BŁĄD] Nie można odczytać pliku: {filepath}")
            return []

    for opponent_name, metrics in data.items():
        row = {}
        
        # 1. Podstawowe metryki
        for key, value in metrics.items():
            if not isinstance(value, dict):
                row[key] = value
                
        # 2. Rozpakowanie action_rates
        for action, rate in metrics.get('action_rates', {}).items():
            row[f"action_rate_{action}"] = rate
            
        # 3. Rozpakowanie actions_by_street
        for street, actions in metrics.get('actions_by_street', {}).items():
            for action, count in actions.items():
                row[f"street_{street}_{action}"] = count
                
        # 4. Rozpakowanie actions_by_hand_category
        for hand, actions in metrics.get('actions_by_hand_category', {}).items():
            for action, count in actions.items():
                row[f"hand_{hand}_{action}"] = count
                
        flattened_rows.append(row)
        
    return flattened_rows


def main():
    print(f"Szukam plików .json w folderze:\n{INPUT_DIR}\n")
    
    if not os.path.exists(INPUT_DIR):
        print(f"[BŁĄD] Podany folder wejściowy nie istnieje: {INPUT_DIR}")
        return

    # Utworzenie folderu dla pliku wyjściowego, jeśli nie istnieje
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Utworzono brakujący folder docelowy: {output_dir}")

    # Szukanie plików JSON
    search_pattern = os.path.join(INPUT_DIR, "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print("[OSTRZEŻENIE] Nie znaleziono żadnych plików .json!")
        return

    all_data = []
    
    for file in json_files:
        rows = flatten_evaluation_data(file)
        all_data.extend(rows)
        
    # Zapis do CSV
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Wypełnienie pustych miejsc zerami (brak akcji = 0)
        df = df.fillna(0)
        
        # Sortowanie dla lepszej czytelności
        if 'step' in df.columns and 'stage' in df.columns:
            df = df.sort_values(by=['stage', 'step', 'opponent_suite'])
            
        try:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
            print(f"[SUKCES] Zapisano dane do pliku:\n{OUTPUT_FILE}")
            print(f"-> Przetworzono plików JSON: {len(json_files)}")
            print(f"-> Wynikowy CSV ma {len(df)} wierszy i {len(df.columns)} kolumn.")
        except PermissionError:
            print(f"\n[BŁĄD] Brak uprawnień do zapisu. Upewnij się, że plik {OUTPUT_FILE} nie jest otwarty np. w Excelu.")
    else:
        print("Brak poprawnych danych do zapisania.")

if __name__ == "__main__":
    main()