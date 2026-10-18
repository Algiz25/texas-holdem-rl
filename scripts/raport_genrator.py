import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import os
import re

# ==========================================
# KONFIGURACJA ŚCIEŻEK
# ==========================================
# Podaj jeden plik CSV (niezależnie czy z Fazy 1, Fazy 2 czy Top10)
CSV_PATH = r"checkpoints\iqn\non_local\steps_p1_21372137\evaluations_top10\evaluations_v2.csv"
OUTPUT_PDF = r"raports\IQN_Phase1_21372137_precise.pdf"

# ==========================================
# WYBÓR NAJLEPSZEGO MODELU
# ==========================================
# Krok, w którym znajduje się Twój wybrany mistrz (np. 4700000)
# Ustaw na None, jeśli nie chcesz rysować linii.
BEST_MODEL_STEP = 1600000

# Kolory akcji dla zaawansowanych wykresów
ACTION_COLORS = {
    'fold': '#d62728',         # Czerwony
    'check': '#7f7f7f',        # Szary
    'call': '#1f77b4',         # Niebieski
    'raise_half': '#2ca02c',   # Jasny zielony
    'raise_pot': '#005a00',    # Ciemny zielony
    'all_in': '#9467bd'        # Fioletowy
}
ACTIONS_ORDER = ['fold', 'check', 'call', 'raise_half', 'raise_pot', 'all_in']

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def format_xaxis(ax):
    """Formatuje oś X na czytelne miliony/tysiące kroków"""
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1e-6:g}M' if x >= 1e6 else f'{x*1e-3:g}K'))
    ax.set_xlabel("Decyzje Ucznia (Kroki)")

def load_and_aggregate_data():
    if not os.path.exists(CSV_PATH):
        print(f"[BŁĄD] Nie znaleziono pliku: {CSV_PATH}")
        return pd.DataFrame()
    
    df = pd.read_csv(CSV_PATH)
    
    # Wykluczamy początkowy, nietrenowany/losowy model (krok 0), aby nie psuł skali wykresów.
    df = df[~df['stage'].isin(['baseline_untrained', 'baseline_random'])].copy()
    
    if df.empty:
        return df

    # Usunięcie ewentualnego suffixu _part_X 
    df['opponent_suite'] = df['opponent_suite'].apply(lambda x: re.sub(r'_part_\d+$', '', str(x)))
    
    agg_dict = {}
    for col in df.columns:
        if col in ['stage', 'step', 'opponent_suite', 'checkpoint', 'seed_base']:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Wskaźniki i metryki ułamkowe -> ŚREDNIA
            if 'rate' in col or col in ['bb_per_100', 'average_finish', 'vpip', 'pfr', 'chip_delta_ci95', 'mean_chip_delta_per_tournament', 'mean_chip_delta_per_hand']:
                agg_dict[col] = 'mean'
            # Liczniki i wartości absolutne -> SUMA
            else:
                agg_dict[col] = 'sum'
                
    # Grupowanie po kroku i przeciwniku
    df_agg = df.groupby(['stage', 'step', 'opponent_suite']).agg(agg_dict).reset_index()
    return df_agg

def create_title_page(pdf, df):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    max_step = df['step'].max() if not df.empty else 0
    filename = os.path.basename(CSV_PATH)
    
    best_step_info = f"\nWybrany najlepszy model: Krok {BEST_MODEL_STEP:,}\n" if BEST_MODEL_STEP else ""
    
    text = (
        "RAPORT Z EWALUACJI MODELU RL - TEXAS HOLD'EM\n\n"
        f"Plik źródłowy: {filename}\n"
        f"Maksymalny badany krok: {max_step:,}\n"
        f"{best_step_info}\n"
        "Wykresy przedstawiają ewolucję metryk (bb/100, Win Rate, VPIP, PFR)\n"
        "względem różnych przeciwników zawartych w logach.\n\n"
        "Jeśli plik zawiera rozszerzone dane, raport uwzględni\n"
        "również analizę zagrań na poszczególnych ulicach."
    )
    
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16, fontweight='bold', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

def plot_metric(pdf, df, metric_col, title, ylabel, hline=None, is_percentage=False):
    """Tworzy jeden duży wykres na stronę PDF dla danej metryki"""
    if df.empty or metric_col not in df.columns:
        return
        
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Rysowanie linii dla każdego przeciwnika
    sns.lineplot(data=df, x='step', y=metric_col, hue='opponent_suite', marker='o', ax=ax, linewidth=2.5, palette="tab10")
    
    # Linia odniesienia (np. zero dla bb/100)
    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', alpha=0.6)
        
    # PIONOWA LINIA NAJLEPSZEGO MODELU
    if BEST_MODEL_STEP is not None and BEST_MODEL_STEP in df['step'].values:
        best_label = f"Najlepszy Model ({BEST_MODEL_STEP/1e6:g}M)"
        ax.axvline(x=BEST_MODEL_STEP, color='#9467bd', linestyle='--', linewidth=2.5, zorder=0, label=best_label)
        
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    
    if is_percentage: 
        ax.set_ylim(0, 1)
        
    format_xaxis(ax)
    
    # Ustawienie legendy po prawej stronie poza wykresem
    ax.legend(title="Legenda", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_street_evolution(pdf, df):
    if df.empty: return
    
    streets = ['preflop', 'flop', 'turn', 'river']
    
    has_street_data = any(f'street_{s}_fold' in df.columns for s in streets)
    if not has_street_data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    df_step = df.groupby('step').sum(numeric_only=True).reset_index()

    for idx, street in enumerate(streets):
        ax = axes[idx]
        cols = [f'street_{street}_{act}' for act in ACTIONS_ORDER]
        valid_cols = [c for c in cols if c in df_step.columns]
        
        if not valid_cols:
            ax.set_title(f"{street.upper()} - Brak Danych", fontsize=14)
            ax.axis('off')
            continue
            
        act_labels = [c.replace(f'street_{street}_', '') for c in valid_cols]
        colors = [ACTION_COLORS[act] for act in act_labels]
        
        data = df_step[valid_cols].copy()
        data_sum = data.sum(axis=1)
        data_perc = data.div(data_sum, axis=0).fillna(0) * 100
        
        ax.stackplot(df_step['step'], *[data_perc[col] for col in valid_cols], 
                     labels=act_labels, colors=colors, alpha=0.85)

        # PIONOWA LINIA NAJLEPSZEGO MODELU NA STACKPLOTACH
        if BEST_MODEL_STEP is not None and BEST_MODEL_STEP in df_step['step'].values:
            ax.axvline(x=BEST_MODEL_STEP, color='white', linestyle='--', linewidth=3, alpha=0.9)
            ax.axvline(x=BEST_MODEL_STEP, color='black', linestyle=':', linewidth=2, alpha=0.8, label=f'Best ({BEST_MODEL_STEP/1e6:g}M)')

        ax.set_title(f"Dystrybucja Akcji: {street.upper()}", fontsize=14, fontweight='bold')
        ax.set_ylabel("% Podjętych Decyzji")
        ax.set_ylim(0, 100)
        format_xaxis(ax)
        
        if idx == 1: 
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), title="Akcja")

    fig.suptitle("Ewolucja Zagrań na Poszczególnych Ulicach (Zagregowane)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    pdf.savefig(fig)
    plt.close(fig)

def plot_hand_categories_final_model(pdf, df):
    if df.empty: return
    
    has_hand_data = any('hand_pair_fold' in c for c in df.columns)
    if not has_hand_data: return

    # Jeżeli wybraliśmy najlepszy krok, rysujemy dla niego. Jak nie, dla ostatniego.
    target_step = BEST_MODEL_STEP if BEST_MODEL_STEP and BEST_MODEL_STEP in df['step'].values else df['step'].max()
    
    df_final = df[df['step'] == target_step].sum(numeric_only=True)
    
    hand_categories = ['high_card', 'pair', 'two_pair', 'three_of_a_kind', 'straight', 'flush', 'full_house', 'four_of_a_kind']
    plot_data = []
    
    for cat in hand_categories:
        row = {'Hand': cat.replace('_', ' ').title()}
        total_actions = 0
        
        for act in ACTIONS_ORDER:
            col_name = f'hand_{cat}_{act}'
            val = df_final[col_name] if col_name in df_final.index else 0
            row[act] = val
            total_actions += val
            
        if total_actions > 0:
            for act in ACTIONS_ORDER:
                row[act] = (row[act] / total_actions) * 100
        else:
            for act in ACTIONS_ORDER:
                row[act] = 0
                
        plot_data.append(row)
        
    df_plot = pd.DataFrame(plot_data)
    df_plot.set_index('Hand', inplace=True)
    df_plot = df_plot[(df_plot.T != 0).any()]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = [ACTION_COLORS[act] for act in ACTIONS_ORDER]
    
    df_plot[ACTIONS_ORDER].plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.75, alpha=0.9)
    
    title_suffix = "Najlepszego" if target_step == BEST_MODEL_STEP else "Końcowego"
    ax.set_title(f"Rozgrywanie Układów przez Model z kroku {target_step/1e6:g}M", fontsize=16, fontweight='bold')
    ax.set_ylabel("Odsetek podjętych akcji [%]", fontsize=12)
    ax.set_xlabel("Ułożony układ u agenta", fontsize=12)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.legend(title="Akcja", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def generate_report():
    print(f"Wczytywanie i agregowanie danych z pliku: {CSV_PATH}...")
    
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    
    df = load_and_aggregate_data()
    
    if df.empty:
        print("[BŁĄD] Nie udało się wygenerować raportu. Dane są puste.")
        return
        
    print(f"Generowanie Uniwersalnego Raportu PDF ({OUTPUT_PDF})...")
    with PdfPages(OUTPUT_PDF) as pdf:
        create_title_page(pdf, df)
        
        plot_metric(pdf, df, 'bb_per_100', "Zysk (Win Rate) w bb/100", "bb/100", hline=0)
        plot_metric(pdf, df, 'tournament_win_rate', "Odsetek Wygranych Turniejów", "Win Rate [%]", hline=0.25, is_percentage=True)
        plot_metric(pdf, df, 'vpip', "Styl Gry: VPIP (Wejścia do rozdania)", "VPIP [%]", is_percentage=True)
        plot_metric(pdf, df, 'pfr', "Styl Gry: PFR (Agresja przed Flopem)", "PFR [%]", is_percentage=True)
        
        plot_street_evolution(pdf, df)
        plot_hand_categories_final_model(pdf, df)
            
    print(f"\n[SUKCES] Twój raport został wygenerowany pomyślnie!")

if __name__ == "__main__":
    generate_report()