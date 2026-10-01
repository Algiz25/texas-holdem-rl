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
CSV_V2_PATH = r"checkpoints\ppo\phase2_selfplay_seed_11001\evaluations\evaluations_v2.csv"
CSV_V3_PATH = r"checkpoints\ppo\phase2_selfplay_seed_11001\evaluations_post\evaluations_v3.csv"
OUTPUT_PDF = "PPO_TexasHoldem_Master_Report.pdf"

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

def load_v2_data():
    if not os.path.exists(CSV_V2_PATH):
        print(f"[OSTRZEŻENIE] Nie znaleziono {CSV_V2_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(CSV_V2_PATH)
    # Wykluczamy początkowy, nietrenowany/losowy model, aby nie zaburzał skali wykresów
    return df[df['stage'].isin(['validation', 'post_eval'])].copy()

def load_and_aggregate_v3():
    if not os.path.exists(CSV_V3_PATH):
        print(f"[OSTRZEŻENIE] Nie znaleziono {CSV_V3_PATH}")
        return pd.DataFrame()
    
    df = pd.read_csv(CSV_V3_PATH)
    df = df[df['stage'].isin(['validation', 'post_eval'])].copy()
    
    # Usunięcie suffixu _part_X (łączenie procesów)
    df['base_suite'] = df['opponent_suite'].apply(lambda x: re.sub(r'_part_\d+$', '', str(x)))
    
    agg_dict = {}
    for col in df.columns:
        if col in ['stage', 'step', 'base_suite', 'opponent_suite', 'checkpoint', 'seed_base']:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Wskaźniki i metryki ułamkowe -> ŚREDNIA
            if 'rate' in col or col in ['bb_per_100', 'average_finish', 'vpip', 'pfr', 'chip_delta_ci95', 'mean_chip_delta_per_tournament', 'mean_chip_delta_per_hand']:
                agg_dict[col] = 'mean'
            # Liczniki i wartości absolutne -> SUMA
            else:
                agg_dict[col] = 'sum'
                
    df_agg = df.groupby(['stage', 'step', 'base_suite']).agg(agg_dict).reset_index()
    df_agg.rename(columns={'base_suite': 'opponent_suite'}, inplace=True)
    return df_agg

def create_title_page(pdf, df_v2, df_v3):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    max_step = max(df_v2['step'].max() if not df_v2.empty else 0, df_v3['step'].max() if not df_v3.empty else 0)
    
    text = (
        "MASTER RAPORT Z TRENINGU PPO - TEXAS HOLD'EM\n"
        "(Agent RL vs Środowisko Wieloagentowe)\n\n"
        f"Całkowity czas treningu (decyzje ucznia): {max_step:,}\n\n"
        "Górne wykresy: Gra przeciwko statycznym botom (Random, Passive, Mixed)\n"
        "Dolne wykresy: Gra przeciwko puli Baseline (Faza 2 Self-Play)\n\n"
        "Analiza taktyczna zawiera zachowania na ulicach oraz\n"
        "rozgrywanie układów pokerowych przez końcowy model."
    )
    
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=16, fontweight='bold', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

def plot_separated_metric(pdf, df_v2, df_v3, metric_col, title, ylabel, hline=None, is_percentage=False):
    """Tworzy dwa wykresy (góra/dół) dla wybranej metryki (np. bb/100)"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    
    # WYKRES 1: Proste Boty (v2)
    if not df_v2.empty and metric_col in df_v2.columns:
        sns.lineplot(data=df_v2, x='step', y=metric_col, hue='opponent_suite', marker='o', ax=axes[0], linewidth=2.5, palette="tab10")
        if hline is not None:
            axes[0].axhline(hline, color='red', linestyle='--', alpha=0.6)
        axes[0].set_title(f"{title} - Przeciwko Prostym Botom", fontsize=14, fontweight='bold')
        axes[0].set_ylabel(ylabel)
        if is_percentage: axes[0].set_ylim(0, 1)
        format_xaxis(axes[0])
        axes[0].legend(title="Typ Bota", bbox_to_anchor=(1.01, 1), loc='upper left')

    # WYKRES 2: Baseline / Self-Play (v3)
    if not df_v3.empty and metric_col in df_v3.columns:
        sns.lineplot(data=df_v3, x='step', y=metric_col, hue='opponent_suite', marker='s', ax=axes[1], linewidth=2.5, palette=["#9467bd"])
        if hline is not None:
            axes[1].axhline(hline, color='red', linestyle='--', alpha=0.6)
        axes[1].set_title(f"{title} - Przeciwko Baseline (Faza 2: Self-Play)", fontsize=14, fontweight='bold')
        axes[1].set_ylabel(ylabel)
        if is_percentage: axes[1].set_ylim(0, 1)
        format_xaxis(axes[1])
        axes[1].legend(title="Typ Przeciwnika", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_street_evolution(pdf, df_v3):
    if df_v3.empty: return
    
    streets = ['preflop', 'flop', 'turn', 'river']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, street in enumerate(streets):
        ax = axes[idx]
        cols = [f'street_{street}_{act}' for act in ACTIONS_ORDER]
        valid_cols = [c for c in cols if c in df_v3.columns]
        act_labels = [c.replace(f'street_{street}_', '') for c in valid_cols]
        colors = [ACTION_COLORS[act] for act in act_labels]
        
        data = df_v3[valid_cols].copy()
        data_sum = data.sum(axis=1)
        data_perc = data.div(data_sum, axis=0).fillna(0) * 100
        
        ax.stackplot(df_v3['step'], *[data_perc[col] for col in valid_cols], 
                     labels=act_labels, colors=colors, alpha=0.85)
        
        ax.set_title(f"Dystrybucja Akcji Baseline: {street.upper()}", fontsize=14, fontweight='bold')
        ax.set_ylabel("% Podjętych Decyzji")
        ax.set_ylim(0, 100)
        format_xaxis(ax)
        
        if idx == 1: 
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), title="Akcja")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def plot_hand_categories_final_model(pdf, df_v3):
    if df_v3.empty: return
    
    final_step = df_v3['step'].max()
    # Sumujemy po wszystkich przeciwnikach dla końcowego modelu
    df_final = df_v3[df_v3['step'] == final_step].sum(numeric_only=True)
    
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
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [ACTION_COLORS[act] for act in ACTIONS_ORDER]
    
    df_plot[ACTIONS_ORDER].plot(kind='bar', stacked=True, color=colors, ax=ax, width=0.75, alpha=0.9)
    
    ax.set_title(f"Zestawienie Baseline: Rozgrywanie układów przez końcowy model (Krok {final_step:,})", fontsize=16, fontweight='bold')
    ax.set_ylabel("Odsetek podjętych akcji [%]")
    ax.set_xlabel("Ułożony układ u agenta")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Akcja", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def generate_report():
    print("Wczytywanie danych z ewaluacji V2 (Proste Boty)...")
    df_v2 = load_v2_data()
    
    print("Wczytywanie i agregowanie danych z ewaluacji V3 (Baseline / Self-Play)...")
    df_v3 = load_and_aggregate_v3()
    
    if df_v2.empty and df_v3.empty:
        print("BŁĄD: Nie udało się wczytać żadnych danych.")
        return
        
    print("Generowanie Master Raportu PDF...")
    with PdfPages(OUTPUT_PDF) as pdf:
        # 1. Strona tytułowa
        create_title_page(pdf, df_v2, df_v3)
        
        # 2. Główna metryka: bb/100
        plot_separated_metric(pdf, df_v2, df_v3, 'bb_per_100', "Krzywa Uczenia: Zysk w bb/100", "Win Rate [bb/100]", hline=0)
        
        # 3. Win Rate w Turniejach
        plot_separated_metric(pdf, df_v2, df_v3, 'tournament_win_rate', "Odsetek Wygranych Turniejów", "Win Rate", hline=0.25, is_percentage=True)
        
        # 4. Styl gry - VPIP
        plot_separated_metric(pdf, df_v2, df_v3, 'vpip', "Styl Gry: VPIP (Wejścia do rozdania)", "VPIP", is_percentage=True)
        
        # 5. Styl gry - PFR
        plot_separated_metric(pdf, df_v2, df_v3, 'pfr', "Styl Gry: PFR (Agresja przed Flopem)", "PFR", is_percentage=True)
        
        # 6. Ewolucja Ulic i Kategoryzacja rąk (V3 Baseline)
        if not df_v3.empty:
            plot_street_evolution(pdf, df_v3)
            plot_hand_categories_final_model(pdf, df_v3)
            
    print(f"\n[SUKCES] Twój raport został zapisany jako: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_report()