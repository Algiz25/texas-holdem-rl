import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================
# TUTAJ WPISZ ŚCIEŻKĘ DO NOWEGO PLIKU .json:
JSON_PATH = r"checkpoints\iqn\non_local\validation_step_000500000.json" 
OUTPUT_PDF = r"raports\IQN_heatmaps.pdf"

# Wybierz, z jakim przeciwnikiem chcesz przeanalizować grę
OPPONENT = "phase1_mix" 

ACTIONS_ORDER = ['fold', 'check', 'call', 'raise_half', 'raise_pot', 'all_in']
STREETS_ORDER = ['preflop', 'flop', 'turn', 'river']

# Hierarchia układów pokerowych od najsłabszego do najsilniejszego
HANDS_ORDER = [
    'high_card', 
    'pair', 
    'two_pair', 
    'three_of_a_kind', 
    'straight', 
    'flush', 
    'full_house', 
    'four_of_a_kind', 
    'straight_flush'
]

sns.set_theme(style="whitegrid")

def plot_combined_street_heatmaps(json_path, output_pdf, opponent_suite):
    if not os.path.exists(json_path):
        print(f"[BŁĄD] Nie znaleziono pliku JSON: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if opponent_suite not in data:
        print(f"[BŁĄD] W pliku JSON nie znaleziono danych dla przeciwnika: '{opponent_suite}'")
        return

    advanced_data = data[opponent_suite].get('actions_by_street_and_category', {})
    
    if not advanced_data:
        print("[BŁĄD] Plik JSON nie zawiera zaawansowanych danych z ulic/układów.")
        return

    print(f"Generowanie złączonego raportu PDF ({output_pdf}) dla przeciwnika: {opponent_suite}...")
    
    # Tworzymy dużą planszę 2x2
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    # Przekształcamy wewnętrzne nazwy układów na ładne tytuły z zachowaniem kolejności
    ordered_labels = [h.replace('_', ' ').title() for h in HANDS_ORDER]

    for idx, street in enumerate(STREETS_ORDER):
        ax = axes[idx]
        street_data = advanced_data.get(street, {})
        
        if not street_data:
            ax.set_title(f"{street.upper()} - Brak danych", fontsize=16, color='red')
            ax.axis('off')
            continue

        # Tworzymy wstępny DataFrame
        categories = list(street_data.keys())
        matrix = pd.DataFrame(index=[c.replace('_', ' ').title() for c in categories], columns=ACTIONS_ORDER)
        
        for i, cat in enumerate(categories):
            for act in ACTIONS_ORDER:
                matrix.iloc[i][act] = street_data[cat].get(act, 0)
        
        matrix = matrix.astype(float)
        
        # Filtrujemy wiersze z samymi zerami (np. brak karety na riverze)
        matrix = matrix[(matrix.T != 0).any()]
        
        if matrix.empty:
            ax.set_title(f"{street.upper()} - Brak rozkładu", fontsize=16)
            ax.axis('off')
            continue
            
        # Obliczamy procenty w wierszach
        matrix_perc = matrix.div(matrix.sum(axis=1), axis=0).fillna(0) * 100
        
        # SORTOWANIE WIERSSZY OD NAJSŁABSZEGO DO NAJSILNIEJSZEGO
        # Wybieramy tylko te układy, które faktycznie wystąpiły na tej ulicy, ale w prawidłowej kolejności
        current_ordered_labels = [label for label in ordered_labels if label in matrix_perc.index]
        matrix_perc = matrix_perc.reindex(current_ordered_labels)

        # Rysujemy Heatmapę (cmap='coolwarm', sztywne granice 0-100% dla spójności kolorów)
        sns.heatmap(matrix_perc, annot=True, fmt=".1f", cmap="BuGn", vmin=0, vmax=100, 
                    cbar_kws={'label': '% Częstość Akcji'}, ax=ax, linewidths=.5, annot_kws={"size": 11})
        
        ax.set_title(f"{street.upper()}", fontsize=18, fontweight='bold')
        ax.set_xlabel("Podjęta Akcja", fontsize=14)
        ax.set_ylabel("Ułożony Układ", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12, rotation=0)

    # Główny tytuł dla całego slajdu
    fig.suptitle(f"Strategia Rozgrywania Układów w Fazach Licytacji\n(Przeciwnik: {opponent_suite})", 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    # Dostosowanie marginesu górnego, żeby główny tytuł się zmieścił
    plt.subplots_adjust(top=0.90) 
    
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"[SUKCES] Zapisano plik: {output_pdf}")

if __name__ == "__main__":
    plot_combined_street_heatmaps(JSON_PATH, OUTPUT_PDF, OPPONENT)