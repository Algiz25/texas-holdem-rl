import streamlit as st
import json
from pathlib import Path

# Ustawienia strony
st.set_page_config(layout="wide", page_title="Poker RL Replay", page_icon="🃏")

# --- 1. FUNKCJE POMOCNICZE (Karty i Żetony) ---
def fmt(val):
    """Zamienia 10.0 -> 10, ale zostawia ułamki przy podziale nieparzystej puli (np. 10.5)."""
    val = round(float(val), 2)
    return int(val) if val.is_integer() else val

def get_card_html(card_str):
    """Zamienia kod (np. 'HQ') na wizualną kartę HTML."""
    if not card_str:
        return ""
    
    suits = {'H': ('♥', 'red'), 'D': ('♦', 'red'), 'S': ('♠', 'black'), 'C': ('♣', 'black')}
    suit_char = card_str[0]
    rank_char = "10" if card_str[1] == "T" else card_str[1]
    
    symbol, color = suits.get(suit_char, ('?', 'black'))
    
    return f"""
    <div style="
        display: inline-block; width: 50px; height: 75px; 
        background: white; border-radius: 5px; border: 1px solid #ccc;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3); margin: 0 3px;
        text-align: center; line-height: 75px; font-size: 24px;
        font-weight: bold; color: {color}; font-family: sans-serif;
        position: relative;">
        <span style="position: absolute; top: 5px; left: 5px; line-height: 1; font-size: 15px;">{rank_char}</span>
        {symbol}
    </div>
    """

def get_back_card_html():
    """Karta zakryta."""
    return """
    <div style="
        display: inline-block; width: 50px; height: 75px; 
        background: repeating-linear-gradient(45deg, #b71c1c, #b71c1c 5px, #ffffff 5px, #ffffff 10px);
        border-radius: 5px; border: 2px solid white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3); margin: 0 3px;">
    </div>
    """

def make_bet_chip_div(val):
    """Tworzy wygląd pomarańczowego żetonu zakładu."""
    return f"""
    <div style="
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: #e65100; color: white; width: 42px; height: 42px; border-radius: 50%;
        border: 2px dashed white; box-shadow: 0 4px 8px rgba(0,0,0,0.5); font-weight: bold; font-size: 12px;
        text-shadow: 1px 1px 1px black;">
        {fmt(val)}
    </div>
    """

def make_pot_chip_div(val):
    """Tworzy wygląd dużego złotego żetonu Puli na środku stołu."""
    return f"""
    <div style="
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        background: #FFD700; border: 4px dashed #B8860B; border-radius: 50%;
        width: 75px; height: 75px; box-shadow: 0 5px 15px rgba(0,0,0,0.6);
        color: black; font-weight: bold; font-size: 16px;">
        <span style="font-size: 10px; margin-bottom: -2px;">PULA</span>
        {fmt(val)}
    </div>
    """

def make_win_chip_div(val):
    """Tworzy wygląd mniejszego złotego żetonu wygranej lecącego z puli do stacka."""
    return f"""
    <div style="
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: #FFD700; color: black; width: 46px; height: 46px; border-radius: 50%;
        border: 3px dashed #B8860B; box-shadow: 0 0 12px #FFD700, 0 4px 8px rgba(0,0,0,0.6);
        font-weight: bold; font-size: 12px;">
        +{fmt(val)}
    </div>
    """

def make_stack_chip_div(val):
    """Tworzy wygląd niebieskiego żetonu Stacka gracza."""
    return f"""
    <div style="
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        background: #1565C0; color: white; width: 55px; height: 55px; border-radius: 50%;
        border: 3px dashed white; box-shadow: 0 4px 10px rgba(0,0,0,0.6); font-weight: bold; font-size: 13px;
        text-shadow: 1px 1px 1px black;">
        <span style="font-size: 9px; opacity: 0.9; margin-bottom: -2px;">STACK</span>
        {fmt(val)}
    </div>
    """

# --- 2. ŁADOWANIE DANYCH I WYBÓR PLIKU ---
st.sidebar.title("Panel Sterowania")

# Inicjalizacja zmiennych w pamięci sesji
if 'hand_idx' not in st.session_state:
    st.session_state.hand_idx = 0
if 'step_idx' not in st.session_state:
    st.session_state.step_idx = 0

def reset_step():
    st.session_state.step_idx = 0

def reset_all():
    """Resetuje rozdanie i krok po wgraniu nowego pliku JSON."""
    st.session_state.hand_idx = 0
    st.session_state.step_idx = 0

uploaded_file = st.sidebar.file_uploader(
    "📂 Wgraj historię turnieju (.json):", 
    type=["json"], 
    on_change=reset_all
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = BASE_DIR / "replay_files" / "tournament_history.json"

@st.cache_data
def load_local_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

if uploaded_file is not None:
    try:
        data = json.load(uploaded_file)
        st.sidebar.success(f"Załadowano: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Błąd odczytu pliku JSON: {e}")
        st.stop()
else:
    try:
        data = load_local_data(DEFAULT_JSON_PATH)
        st.sidebar.caption("Używany domyślny plik lokalny: `tournament_history.json`")
    except FileNotFoundError:
        st.info("👈 Wgraj plik `.json` z historią turnieju w panelu bocznym, aby uruchomić stół.")
        st.stop()

hands = data.get("hands", [])
if not hands:
    st.warning("⚠️ Wybrany plik JSON nie zawiera żadnych zapisanych rozdań (brak listy `hands`).")
    st.stop()

player_names = data.get("tournament_config", {}).get("players", {})
def get_p_name(p_id):
    return player_names.get(p_id, p_id)

st.sidebar.divider()

# --- 3. KONTROLKI ODTWARZACZA (Pasek Boczny) ---
max_hand = len(hands) - 1
if st.session_state.hand_idx > max_hand:
    st.session_state.hand_idx = 0

def prev_hand():
    if st.session_state.hand_idx > 0:
        st.session_state.hand_idx -= 1
        reset_step()

def next_hand():
    if st.session_state.hand_idx < max_hand:
        st.session_state.hand_idx += 1
        reset_step()

if max_hand > 0:
    st.sidebar.slider(
        "Wybierz rozdanie:", 
        0, max_hand, 
        key="hand_idx", 
        on_change=reset_step
    )
else:
    st.session_state.hand_idx = 0
    st.sidebar.markdown("**Rozdanie:** 1 z 1 *(tylko jedno w pliku)*")

h_col1, h_col2 = st.sidebar.columns(2)
with h_col1:
    st.button("🡰 Poprzedni", on_click=prev_hand, disabled=(st.session_state.hand_idx == 0), use_container_width=True, key="btn_prev_hand")
with h_col2:
    st.button("Następny 🡲", on_click=next_hand, disabled=(st.session_state.hand_idx == max_hand), use_container_width=True, key="btn_next_hand")

st.sidebar.divider()

current_hand = hands[st.session_state.hand_idx]
events = current_hand.get("events", [])

# --- KOREKTA BŁĘDÓW SILNIKA RL W DANYCH ---
active_in_hand = current_hand["active_players"].copy()
is_heads_up = len(active_in_hand) == 2

first_decision_ev = next((ev for ev in events if ev.get("action_str") not in ["Small Blind", "Big Blind"] and ev.get("action") != -1), None)
first_actor = first_decision_ev["player"] if first_decision_ev else None

sb_player = None
bb_player = None
dealer_player = None

if is_heads_up and first_actor:
    dealer_player = first_actor
    sb_player = first_actor
    bb_player = active_in_hand[0] if active_in_hand[1] == first_actor else active_in_hand[1]
    
    for ev in events:
        if ev.get("action_str") in ["Small Blind", "Big Blind"]:
            ev["action_str"] = "Small Blind" if ev["player"] == sb_player else "Big Blind"
            if ev.get("amount", 0) == 0:
                ev["amount"] = 1.0 if ev["action_str"] == "Small Blind" else 2.0
else:
    sb_player = next((ev["player"] for ev in events if ev.get("action_str") == "Small Blind"), None)
    bb_player = next((ev["player"] for ev in events if ev.get("action_str") == "Big Blind"), None)
    
    if not sb_player and first_actor:
        utg_idx = active_in_hand.index(first_actor)
        bb_player = active_in_hand[(utg_idx - 1) % len(active_in_hand)]
        sb_player = active_in_hand[(utg_idx - 2) % len(active_in_hand)]
        
    if sb_player and sb_player in active_in_hand:
        sb_idx = active_in_hand.index(sb_player)
        dealer_player = active_in_hand[(sb_idx - 1) % len(active_in_hand)]

# --- BUDOWANIE OSI CZASU (TIMELINE) Z KROKAMI NOWYCH FAZ I SHOWDOWNU ---
timeline = []
temp_board_len = 0
street_names = {3: "FLOP", 4: "TURN", 5: "RIVER"}

for ev in events:
    ev_board = ev.get("board", [])
    if len(ev_board) > temp_board_len:
        s_name = street_names.get(len(ev_board), "NOWA FAZA")
        timeline.append({
            "type": "STREET_START",
            "street_name": s_name,
            "board": ev_board
        })
        temp_board_len = len(ev_board)
        
    timeline.append({
        "type": "ACTION",
        "ev": ev,
        "board": ev_board
    })

showdown_data = current_hand.get("showdown", {})
showdown_players = showdown_data.get("players", [])
real_showdown = len(showdown_players) > 1
final_board = showdown_data.get("final_board", [])

if real_showdown and len(final_board) > temp_board_len:
    timeline.append({
        "type": "STREET_START",
        "street_name": "KARTY WSPÓLNE (ALL-IN)",
        "board": final_board
    })

timeline.append({
    "type": "SHOWDOWN",
    "board": final_board if real_showdown else (events[-1].get("board", []) if events else [])
})

max_step = len(timeline) - 1
if st.session_state.step_idx > max_step:
    st.session_state.step_idx = max_step

# --- KONTROLKI AKCJI (KROKU) ---
def prev_step():
    if st.session_state.step_idx > 0:
        st.session_state.step_idx -= 1

def next_step():
    if st.session_state.step_idx < max_step:
        st.session_state.step_idx += 1

if max_step > 0:
    st.sidebar.slider(
        "Przewiń akcję:", 
        0, max_step, 
        key="step_idx"
    )
else:
    st.session_state.step_idx = 0
    st.sidebar.markdown("**Krok:** 0 z 0")

s_col1, s_col2 = st.sidebar.columns(2)
with s_col1:
    st.button("🡰 Poprzedni", on_click=prev_step, disabled=(st.session_state.step_idx == 0), use_container_width=True, key="btn_prev_step")
with s_col2:
    st.button("Następny 🡲", on_click=next_step, disabled=(st.session_state.step_idx == max_step), use_container_width=True, key="btn_next_step")

hand_idx = st.session_state.hand_idx
step_idx = st.session_state.step_idx


# --- 4. OBLICZANIE STANU I POZYCJI (State) ---
central_pot = 0.0
prev_central_pot = 0.0

stacks = current_hand["initial_tournament_chips"].copy()
pre_win_stacks = stacks.copy()
won_amounts = {p: 0.0 for p in ["player_0", "player_1", "player_2", "player_3"]}
current_bets = {p: 0.0 for p in ["player_0", "player_1", "player_2", "player_3"]}
collected_bets = {p: 0.0 for p in ["player_0", "player_1", "player_2", "player_3"]}
last_actions = {p: "" for p in ["player_0", "player_1", "player_2", "player_3"]}

folded_players = set()
all_in_players = set()

if events and events[0].get("action_str") not in ["Small Blind", "Big Blind"]:
    if sb_player and bb_player:
        current_bets[sb_player] += 1.0
        current_bets[bb_player] += 2.0
        stacks[sb_player] -= 1.0
        stacks[bb_player] -= 2.0
        last_actions[sb_player] = "Small Blind"
        last_actions[bb_player] = "Big Blind"

for i in range(step_idx + 1):
    t_step = timeline[i]
    t_type = t_step["type"]
    
    if t_type == "ACTION":
        ev = t_step["ev"]
        p_id = ev["player"]
        amt = ev.get("amount", 0.0)
        
        if ev.get("action_str") == "FOLD":
            folded_players.add(p_id)
            
        stacks[p_id] = round(stacks[p_id] - amt, 2)
        current_bets[p_id] = round(current_bets[p_id] + amt, 2)
        last_actions[p_id] = ev.get("action_str", "")
        
        if ev.get("action_str") == "ALL IN" or stacks[p_id] <= 0:
            all_in_players.add(p_id)
            
    elif t_type == "STREET_START":
        if i == step_idx:
            prev_central_pot = central_pot
            collected_bets = current_bets.copy()
            
        central_pot = round(central_pot + sum(current_bets.values()), 2)
        current_bets = {p: 0.0 for p in current_bets}
        last_actions = {p: "" for p in last_actions}
        
    elif t_type == "SHOWDOWN":
        # Zapamiętujemy stan stacków tuż przed wypłatą puli dla animacji
        pre_win_stacks = stacks.copy()
        total_pot_before_win = round(central_pot + sum(current_bets.values()), 2)
        
        stacks = current_hand["final_tournament_chips"].copy()
        for p in ["player_0", "player_1", "player_2", "player_3"]:
            diff = round(stacks.get(p, 0.0) - pre_win_stacks.get(p, 0.0), 2)
            if diff > 0:
                won_amounts[p] = diff
                
        current_bets = {p: 0.0 for p in current_bets}
        central_pot = 0.0

current_t_step = timeline[step_idx]
board = current_t_step["board"]
is_street_start = (current_t_step["type"] == "STREET_START")
is_showdown_step = (current_t_step["type"] == "SHOWDOWN")
is_action_step = (current_t_step["type"] == "ACTION")

# Wyznaczanie zwycięzców (z sekcji showdown lub na podstawie przyrostu żetonów w final_tournament_chips)
winners = showdown_data.get("winners", []) if is_showdown_step else []
if is_showdown_step and not winners:
    winners = [p for p, amt in won_amounts.items() if amt > 0]

showdown_hands = showdown_data.get("hands", {}) if (is_showdown_step and real_showdown) else {}
current_event = current_t_step["ev"] if is_action_step else {"player": None, "action_str": "", "amount": 0}


# --- 5. RENDEROWANIE STOŁU HTML/CSS + ANIMACJE ---
st.markdown("<h2 style='text-align: center;'>Stół Pokerowy</h2>", unsafe_allow_html=True)

player_positions = {
    "player_0": "bottom: -130px; left: 50%; transform: translateX(-50%);", 
    "player_1": "top: 50%; left: -110px; transform: translateY(-50%);",    
    "player_2": "top: -130px; left: 50%; transform: translateX(-50%);",    
    "player_3": "top: 50%; right: -110px; transform: translateY(-50%);"    
}

button_positions = {
    "player_0": "bottom: 75px; left: 50%; transform: translateX(-50%);",
    "player_1": "top: 50%; left: 85px; transform: translateY(-50%);",
    "player_2": "top: 75px; left: 50%; transform: translateX(-50%);",
    "player_3": "top: 50%; right: 85px; transform: translateY(-50%);"
}

bet_chip_positions = {
    "player_0": "bottom: 75px; left: calc(50% + 45px); transform: translateX(-50%);",
    "player_1": "top: calc(50% + 45px); left: 85px; transform: translateY(-50%);",
    "player_2": "top: 75px; left: calc(50% - 45px); transform: translateX(-50%);",
    "player_3": "top: calc(50% - 45px); right: 85px; transform: translateY(-50%);"
}

stack_chip_positions = {
    "player_0": "bottom: 25px; left: calc(50% - 110px); transform: translateX(-50%);",
    "player_1": "top: calc(50% - 100px); left: 25px; transform: translateY(-50%);",
    "player_2": "top: 25px; left: calc(50% + 110px); transform: translateX(-50%);",
    "player_3": "top: calc(50% + 100px); right: 25px; transform: translateY(-50%);"
}

anim_id = f"h{hand_idx}_s{step_idx}"
anim_duration = "0.45s"
win_anim_duration = "0.55s"

# Klatki kluczowe CSS:
# 1) fly_player_X      - lot ze stacka do zakładu
# 2) to_pot_player_X   - lot z zakładu do centralnej puli
# 3) to_stack_player_X - lot wygranej z centralnej puli (top: 35%, left: 50%) do stacka zwycięzcy
keyframes_css = f"""
<style>
@keyframes fly_player_0_{anim_id} {{
    0%   {{ bottom: 25px; left: calc(50% - 110px); transform: translateX(-50%) scale(1.1); opacity: 1; }}
    90%  {{ bottom: 75px; left: calc(50% + 45px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    100% {{ bottom: 75px; left: calc(50% + 45px); transform: translateX(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes fly_player_1_{anim_id} {{
    0%   {{ top: calc(50% - 100px); left: 25px; transform: translateY(-50%) scale(1.1); opacity: 1; }}
    90%  {{ top: calc(50% + 45px); left: 85px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: calc(50% + 45px); left: 85px; transform: translateY(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes fly_player_2_{anim_id} {{
    0%   {{ top: 25px; left: calc(50% + 110px); transform: translateX(-50%) scale(1.1); opacity: 1; }}
    90%  {{ top: 75px; left: calc(50% - 45px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: 75px; left: calc(50% - 45px); transform: translateX(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes fly_player_3_{anim_id} {{
    0%   {{ top: calc(50% + 100px); right: 25px; transform: translateY(-50%) scale(1.1); opacity: 1; }}
    90%  {{ top: calc(50% - 45px); right: 85px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: calc(50% - 45px); right: 85px; transform: translateY(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}

@keyframes to_pot_player_0_{anim_id} {{
    0%   {{ bottom: 75px; left: calc(50% + 45px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    90%  {{ bottom: calc(65% - 21px); left: 50%; transform: translateX(-50%) scale(0.85); opacity: 1; }}
    100% {{ bottom: calc(65% - 21px); left: 50%; transform: translateX(-50%) scale(0.85); opacity: 0; visibility: hidden; }}
}}
@keyframes to_pot_player_1_{anim_id} {{
    0%   {{ top: calc(50% + 45px); left: 85px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    90%  {{ top: 35%; left: calc(50% - 21px); transform: translateY(-50%) scale(0.85); opacity: 1; }}
    100% {{ top: 35%; left: calc(50% - 21px); transform: translateY(-50%) scale(0.85); opacity: 0; visibility: hidden; }}
}}
@keyframes to_pot_player_2_{anim_id} {{
    0%   {{ top: 75px; left: calc(50% - 45px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    90%  {{ top: calc(35% - 21px); left: 50%; transform: translateX(-50%) scale(0.85); opacity: 1; }}
    100% {{ top: calc(35% - 21px); left: 50%; transform: translateX(-50%) scale(0.85); opacity: 0; visibility: hidden; }}
}}
@keyframes to_pot_player_3_{anim_id} {{
    0%   {{ top: calc(50% - 45px); right: 85px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    90%  {{ top: 35%; right: calc(50% - 21px); transform: translateY(-50%) scale(0.85); opacity: 1; }}
    100% {{ top: 35%; right: calc(50% - 21px); transform: translateY(-50%) scale(0.85); opacity: 0; visibility: hidden; }}
}}

@keyframes to_stack_player_0_{anim_id} {{
    0%   {{ bottom: calc(65% - 23px); left: 50%; transform: translateX(-50%) scale(1.15); opacity: 1; }}
    90%  {{ bottom: 30px; left: calc(50% - 110px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    100% {{ bottom: 30px; left: calc(50% - 110px); transform: translateX(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes to_stack_player_1_{anim_id} {{
    0%   {{ top: 35%; left: calc(50% - 23px); transform: translateY(-50%) scale(1.15); opacity: 1; }}
    90%  {{ top: calc(50% - 100px); left: 30px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: calc(50% - 100px); left: 30px; transform: translateY(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes to_stack_player_2_{anim_id} {{
    0%   {{ top: calc(35% - 23px); left: 50%; transform: translateX(-50%) scale(1.15); opacity: 1; }}
    90%  {{ top: 30px; left: calc(50% + 110px); transform: translateX(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: 30px; left: calc(50% + 110px); transform: translateX(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}
@keyframes to_stack_player_3_{anim_id} {{
    0%   {{ top: 35%; right: calc(50% - 23px); transform: translateY(-50%) scale(1.15); opacity: 1; }}
    90%  {{ top: calc(50% + 100px); right: 30px; transform: translateY(-50%) scale(1.0); opacity: 1; }}
    100% {{ top: calc(50% + 100px); right: 30px; transform: translateY(-50%) scale(1.0); opacity: 0; visibility: hidden; }}
}}

@keyframes show_after_fly_{anim_id} {{
    0%   {{ opacity: 0; }}
    99%  {{ opacity: 0; }}
    100% {{ opacity: 1; }}
}}
@keyframes hide_after_fly_{anim_id} {{
    0%   {{ opacity: 1; }}
    99%  {{ opacity: 1; }}
    100% {{ opacity: 0; visibility: hidden; }}
}}
</style>
"""

any_bets_flying_to_pot = is_street_start and (sum(collected_bets.values()) > 0)

if any_bets_flying_to_pot:
    pot_html = f"""
    <div style="position: absolute; top: 35%; left: 50%; transform: translate(-50%, -50%); z-index: 5;
                animation: hide_after_fly_{anim_id} {anim_duration} linear forwards;">
        {make_pot_chip_div(prev_central_pot)}
    </div>
    <div style="position: absolute; top: 35%; left: 50%; transform: translate(-50%, -50%); z-index: 6; opacity: 0;
                animation: show_after_fly_{anim_id} {anim_duration} linear forwards;">
        {make_pot_chip_div(central_pot)}
    </div>
    """
else:
    pot_html = f"""
    <div style="position: absolute; top: 35%; left: 50%; transform: translate(-50%, -50%); z-index: 5;">
        {make_pot_chip_div(central_pot)}
    </div>
    """

table_html = keyframes_css + f"""
<div style="position: relative; width: 100%; max-width: 800px; height: 500px; 
            background: radial-gradient(circle, #2E7D32, #1B5E20); 
            border: 15px solid #5D4037; border-radius: 250px; 
            box-shadow: inset 0 0 50px rgba(0,0,0,0.6), 0 10px 20px rgba(0,0,0,0.5); 
            margin: 130px auto 150px auto; font-family: sans-serif;">

    <!-- Pula na środku stołu -->
    {pot_html}

    <!-- Karty Wspólne -->
    <div style="position: absolute; top: 55%; left: 50%; transform: translate(-50%, -50%); display: flex;">
        {"".join([get_card_html(c) for c in board])}
    </div>
"""

base_btn_style = "display: inline-flex; justify-content: center; align-items: center; border-radius: 50%; width: 34px; height: 34px; font-weight: bold; margin: 0 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.6);"

active_actor = current_event["player"] if is_action_step else None
active_amt = current_event.get("amount", 0) if is_action_step else 0

for p_id in ["player_0", "player_1", "player_2", "player_3"]:
    
    # 1. Buttony (D, SB, BB)
    btn_html = ""
    if p_id == dealer_player:
        btn_html += f"<span style='{base_btn_style} background: white; color: black; font-size: 15px; border: 2px solid #333;'>D</span>"
    if p_id == sb_player and not is_heads_up:
        btn_html += f"<span style='{base_btn_style} background: #1976D2; color: white; font-size: 13px; border: 2px solid white;'>SB</span>"
    if p_id == bb_player:
        btn_html += f"<span style='{base_btn_style} background: #F57C00; color: white; font-size: 13px; border: 2px solid white;'>BB</span>"
    
    if btn_html:
        table_html += f"<div style='position: absolute; {button_positions[p_id]} z-index: 5; display: flex;'>{btn_html}</div>"

    # 2A. Animacja w kroku STREET_START: Zakłady lecą z pozycji Bet do Puli głównej
    if is_street_start and collected_bets[p_id] > 0:
        table_html += f"""
        <div style="position: absolute; z-index: 15; text-align: center;
                    animation: to_pot_{p_id}_{anim_id} {anim_duration} cubic-bezier(0.2, 0.8, 0.2, 1) forwards;">
            {make_bet_chip_div(collected_bets[p_id])}
        </div>
        """

    # 2B. Żeton Zakładu (Bet) + Animacja przesuwania ze Stacka do Bet w kroku ACTION
    bet_val = current_bets[p_id]
    is_betting_now = (is_action_step and p_id == active_actor and active_amt > 0)
    
    if is_betting_now:
        prev_bet = round(bet_val - active_amt, 2)
        
        if prev_bet > 0:
            table_html += f"""
            <div style="position: absolute; {bet_chip_positions[p_id]} z-index: 8; text-align: center;
                        animation: hide_after_fly_{anim_id} {anim_duration} linear forwards;">
                {make_bet_chip_div(prev_bet)}
            </div>
            """
            
        table_html += f"""
        <div style="position: absolute; z-index: 15; text-align: center;
                    animation: fly_{p_id}_{anim_id} {anim_duration} cubic-bezier(0.2, 0.8, 0.2, 1) forwards;">
            {make_bet_chip_div(active_amt)}
        </div>
        """
        
        table_html += f"""
        <div style="position: absolute; {bet_chip_positions[p_id]} z-index: 9; text-align: center; opacity: 0;
                    animation: show_after_fly_{anim_id} {anim_duration} linear forwards;">
            {make_bet_chip_div(bet_val)}
        </div>
        """
    else:
        if bet_val > 0:
            table_html += f"<div style='position: absolute; {bet_chip_positions[p_id]} z-index: 8; text-align: center;'>{make_bet_chip_div(bet_val)}</div>"

    # 3. Żeton Stacka na stole + Animacja przesuwania wygranej z Puli do Stacka w kroku SHOWDOWN
    is_winning_chips_now = (is_showdown_step and p_id in winners and won_amounts[p_id] > 0)
    
    if is_winning_chips_now:
        # A) Stary stack (sprzed dodania puli) widoczny przez czas lotu złotego żetonu (0.55s)
        table_html += f"""
        <div style="position: absolute; {stack_chip_positions[p_id]} z-index: 4;
                    animation: hide_after_fly_{anim_id} {win_anim_duration} linear forwards;">
            {make_stack_chip_div(pre_win_stacks[p_id])}
        </div>
        """
        
        # B) Mniejszy złoty żeton lecący z centralnej Puli prosto do Stacka wygranego
        table_html += f"""
        <div style="position: absolute; z-index: 16; text-align: center;
                    animation: to_stack_{p_id}_{anim_id} {win_anim_duration} cubic-bezier(0.2, 0.8, 0.2, 1) forwards;">
            {make_win_chip_div(won_amounts[p_id])}
        </div>
        """
        
        # C) Zaktualizowany stack wygranego (pojawia się dokładnie w momencie dolecania złotego żetonu)
        table_html += f"""
        <div style="position: absolute; {stack_chip_positions[p_id]} z-index: 5; opacity: 0;
                    animation: show_after_fly_{anim_id} {win_anim_duration} linear forwards;">
            {make_stack_chip_div(stacks[p_id])}
        </div>
        """
    else:
        if stacks[p_id] >= 0:
            table_html += f"<div style='position: absolute; {stack_chip_positions[p_id]} z-index: 4;'>{make_stack_chip_div(stacks[p_id])}</div>"


# 4. Renderowanie kart i paneli graczy
for p_id in ["player_0", "player_1", "player_2", "player_3"]:
    pos_css = player_positions[p_id]
    all_in_badge = ""
    winner_badge = ""
    
    is_winner = is_showdown_step and (p_id in winners)
    
    if p_id in folded_players:
        cards_html = "<span style='color: white; font-weight: bold; background: #d32f2f; padding: 5px 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);'>FOLD</span>"
    elif p_id in current_hand["active_players"]:
        c1, c2 = current_hand["hole_cards"][p_id]
        cards_html = f"{get_card_html(c1)}{get_card_html(c2)}"
        if p_id in all_in_players and not is_showdown_step:
            all_in_badge = "<div style='color: white; font-weight: bold; background: #d32f2f; padding: 2px 8px; border-radius: 5px; margin-top: 5px; font-size: 13px; box-shadow: 2px 2px 5px rgba(0,0,0,0.5); text-shadow: 1px 1px 1px black;'>ALL IN</div>"
        if is_winner:
            winner_badge = "<div style='color: white; font-weight: bold; background: #1976D2; padding: 3px 10px; border-radius: 5px; margin-top: 5px; font-size: 13px; box-shadow: 0 0 12px #2196F3; border: 1px solid #90CAF9;'>🏆 WYGRANA</div>"
    else:
        cards_html = ""

    is_active = is_action_step and (current_event["player"] == p_id)
    
    if is_winner:
        border_color = "#2196F3"
        bg_color = "rgba(13, 71, 161, 0.9)"
        box_shadow = "0 0 20px #2196F3"
    elif is_active:
        border_color = "#FFD700"
        bg_color = "rgba(0,0,0,0.8)"
        box_shadow = "0 4px 8px rgba(0,0,0,0.4)"
    else:
        border_color = "#ccc"
        bg_color = "rgba(0,0,0,0.5)"
        box_shadow = "0 4px 8px rgba(0,0,0,0.4)"
    
    if is_showdown_step:
        if real_showdown and p_id in showdown_hands:
            cat = showdown_hands[p_id].get("category", "")
            text_color = "#90CAF9" if is_winner else "#bbb"
            action_html = f"<div style='color: {text_color}; font-size: 13px; font-weight: bold; margin-top: 3px;'>🃏 {cat}</div>"
        elif is_winner and not real_showdown:
            action_html = "<div style='color: #90CAF9; font-size: 12px; font-weight: bold; margin-top: 3px;'>Wygrana przez pas</div>"
        else:
            action_html = ""
    else:
        action_html = f"<div style='color: #4CAF50; font-size: 13px; font-weight: bold; margin-top: 3px;'>{last_actions[p_id]}</div>" if (last_actions[p_id] and p_id not in folded_players) else ""
    
    stack_html = f"""
    <div style="background: {bg_color}; color: white; padding: 8px 15px; 
                border-radius: 8px; border: 2px solid {border_color}; min-width: 140px; box-shadow: {box_shadow};">
        <b>{get_p_name(p_id)}</b>
        {action_html}
    </div>
    """

    table_html += f"""
    <div style="position: absolute; {pos_css} text-align: center; z-index: 10;">
        <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 8px;">
            <div style="display: flex; align-items: center;">{cards_html}</div>
            {all_in_badge}
            {winner_badge}
        </div>
        {stack_html}
    </div>
    """

table_html += "</div>"
table_html_minified = table_html.replace('\n', '')
st.markdown(table_html_minified, unsafe_allow_html=True)

# --- 6. LOG AKCJI (Na dole pod stołem) ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    if is_showdown_step:
        st.subheader("🏁 Podsumowanie rozdania")
        st.info("Rozdanie zakończone. Pule zostały przesunięte do stacków zwycięzców.")
    elif is_street_start:
        st.subheader(f"🃏 Nowa faza: {current_t_step['street_name']} (Krok {step_idx})")
        st.info(f"Wyłożono karty wspólne. Zakłady z poprzedniej rundy trafiają do puli (Łącznie: **{fmt(central_pot)}**).")
    else:
        st.subheader(f"Akcja (Krok {step_idx})")
        st.info(f"**{get_p_name(current_event['player'])}** wykonuje: **{current_event['action_str']}**")
        if current_event['amount'] > 0:
            st.write(f"Wartość włożona do puli: **{fmt(current_event['amount'])}**")

with col2:
    if is_showdown_step:
        st.subheader("🏆 Wynik Rozdania")
        for winner in winners:
            win_amt = fmt(won_amounts.get(winner, 0.0))
            if real_showdown and winner in showdown_hands:
                hand_type = showdown_hands[winner]["category"]
                st.success(f"Wygrywa **{winner}** (+{win_amt}) z układem: **{hand_type}**")
            else:
                st.success(f"Wygrywa **{winner}** (+{win_amt}) — przeciwnicy spasowali")