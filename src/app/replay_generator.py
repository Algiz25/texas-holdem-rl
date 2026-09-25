"""Config dla tego pliku to app_config. Rozgrywa 1 turniej z wybranymi graczami i zapisuje jego przebieg do pliku .json"""

import json
import time
import torch
import numpy as np
from tianshou.data import Batch
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from torch.distributions import Categorical

import config
import app.app_config
from environment import TexasHoldemTournament
from models import MaskedActor, Critic, CPUActionActorPolicy
from opponents import PassivePolicy, AggressivePolicy, SeededMixedPolicy
from observation.derived import hand_category
from rlcard.games.limitholdem import PlayerStatus

ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK / CALL",
    2: "RAISE HALF POT",
    3: "RAISE POT",
    4: "ALL IN"
}

CATEGORY_NAMES = {
    0: "High Card", 1: "Pair", 2: "Two Pair", 3: "Three of a kind",
    4: "Straight", 5: "Flush", 6: "Full House", 7: "Four of a kind", 8: "Straight Flush"
}

def load_agent(agent_cfg: dict, env, device="CPU"):
    """Inicjalizuje odpowiednią politykę w zależności od zadanego typu bota."""

    agent_type = agent_cfg["type"]
    seed = agent_cfg.get("seed", None)
    action_space = env.action_space("player_0")
    observation_space = env.observation_space("player_0")

    if agent_type == "dqn":
        net = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(device)
        policy = DiscreteQLearningPolicy(
            model=net,
            action_space=action_space,
            observation_space=observation_space,
            eps_inference=0.0
        )
        policy.load_state_dict(torch.load(agent_cfg.get("path", None), map_location=device, weights_only=True))
        policy.eval()
        return policy
    elif agent_type == "ppo":
        actor = MaskedActor(state_shape=config.OBSERVATION_SIZE, action_shape=config.ACTION_SPACE).to(device)
        critic = Critic(state_shape=config.OBSERVATION_SIZE).to(device)
        def dist_fn(logits): return Categorical(logits=logits)
        policy = CPUActionActorPolicy(
            actor=actor, dist_fn=dist_fn, action_space=action_space,
            observation_space=observation_space, action_scaling=False
        )
        policy.load_state_dict(torch.load(agent_cfg.get("path", None), map_location=device, weights_only=True))
        policy.eval()
        return policy
    elif agent_type == "passive": return PassivePolicy(action_space=action_space)
    elif agent_type == "aggressive": return AggressivePolicy(action_space=action_space)
    elif agent_type == "mixed": return SeededMixedPolicy(action_space=action_space, seed=seed)
    elif agent_type == "random": return "random"
    else: raise ValueError(f"Nieznany typ agenta: {agent_type}")


def record_tournament(seats_config, output_file="tournament_history.json"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Przygotowywanie turnieju na urządzeniu: {device}")

    env = TexasHoldemTournament(num_players=config.NUM_PLAYERS, starting_chips=config.STARTING_CHIPS, debug=False)
    
    players = {}
    for seat, cfg in seats_config.items():
        players[seat] = load_agent(cfg, env, device)

    history = {
        "tournament_config": {
            "starting_chips": config.STARTING_CHIPS,
            "players": {k: v.get("name", v["type"]) for k, v in seats_config.items()}
        },
        "hands": []
    }

    env.reset()
    
    # ---------------------------------------------------------
    # PRZECHWYTYWANIE SHOWDOWNU (Foldy + Uzupełnianie Boardu)
    # ---------------------------------------------------------
    original_showdown = env._get_showdown_result
    showdown_data = {}

    def wrapped_showdown():
        showdown_players, winners = original_showdown()
        game = env.rlcard_env.game
        
        # Odrębna obsługa zwycięstwa przez fold (winners jest puste)
        if not winners:
            contenders = [p for p in game.players if p.status in (PlayerStatus.ALIVE, PlayerStatus.ALLIN)]
            if len(contenders) == 1:
                idx = game.players.index(contenders[0])
                winner_name = env.active_agents[idx]
                winners = (winner_name,)
                showdown_players = (winner_name,)
                
        # NAPRAWA ALL-IN: Jeśli do showdownu wchodzi >1 gracz, stół MUSI mieć 5 kart
        if len(showdown_players) > 1:
            while len(game.public_cards) < 5:
                game.public_cards.append(game.dealer.deal_card())
                
        showdown_data["players"] = list(showdown_players)
        showdown_data["winners"] = list(winners)
        
        board = [c.get_index() for c in game.public_cards]
        showdown_data["final_board"] = board
        
        hands_eval = {}
        for p_name in showdown_players:
            idx = env.active_agents.index(p_name)
            p_cards = [c.get_index() for c in env.rlcard_env.game.players[idx].hand]
            cat_idx = hand_category(p_cards, board)
            hands_eval[p_name] = {
                "cards": p_cards,
                "category": CATEGORY_NAMES.get(cat_idx, "Unknown")
            }
        showdown_data["hands"] = hands_eval
        
        return showdown_players, winners
        
    env._get_showdown_result = wrapped_showdown
    
    def get_card_str(card):
        return card.get_index() if hasattr(card, 'get_index') else str(card)

    current_hand_data = None
    hand_count = 0
    step_count = 0
    
    # Flaga pomagająca wykryć nowe rozdanie dla wyliczania blindów
    hand_started_flag = [True]
    original_start_hand = env._start_new_hand
    def wrapped_start_hand():
        original_start_hand()
        hand_started_flag[0] = True
    env._start_new_hand = wrapped_start_hand

    print("Rozgrywanie turnieju w tle...")
    for agent_id in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            env.step(None)
            continue
            
        # ---------------------------------------------------------
        # INICJALIZACJA NOWEGO ROZDANIA
        # ---------------------------------------------------------
        if current_hand_data is None or hand_started_flag[0]:
            if current_hand_data is not None:
                current_hand_data["final_tournament_chips"] = {k: float(v) for k, v in env.tournament_chips.items()}
                history["hands"].append(current_hand_data)
                
            hand_started_flag[0] = False
            hand_count += 1
            blinds_logged = False
            
            # Snaphot sald w celu wyłapania pobranych blindów
            initial_chips_snapshot = {k: float(v) for k, v in env.tournament_chips.items()}
            
            hole_cards = {}
            for idx, p_name in enumerate(env.active_agents):
                player_obj = env.rlcard_env.game.players[idx]
                hole_cards[p_name] = [get_card_str(c) for c in player_obj.hand]
                
            current_hand_data = {
                "hand_number": hand_count,
                "active_players": env.active_agents.copy(),
                "initial_tournament_chips": initial_chips_snapshot,
                "hole_cards": hole_cards,
                "events": [],
                "showdown": {}
            }

        # ---------------------------------------------------------
        # LOGOWANIE BLINDÓW (Zaraz przed pierwszą akcją)
        # ---------------------------------------------------------
        if not blinds_logged:
            detected_blinds = []
            
            # Najpierw zbieramy wszystkie fizyczne wpłaty
            for p_name in env.active_agents:
                idx = env.active_agents.index(p_name)
                p_obj = env.rlcard_env.game.players[idx]
                diff = initial_chips_snapshot[p_name] - p_obj.remained_chips
                if diff > 0:
                    detected_blinds.append({
                        "player": p_name,
                        "amount": float(diff)
                    })
            
            # Sortujemy wpłaty rosnąco (najpierw Small Blind, potem Big Blind)
            detected_blinds.sort(key=lambda x: x["amount"])
            
            # Wpisujemy posortowane ciemne do historii
            pot_acc = 0.0
            for i, b_data in enumerate(detected_blinds):
                current_hand_data["events"].append({
                    "player": b_data["player"],
                    "action": -1,
                    "action_str": "Small Blind" if i == 0 else "Big Blind",
                    "amount": b_data["amount"],
                    "pot_before_action": pot_acc,
                    "board": [],
                    "q_values": {}
                })
                pot_acc += b_data["amount"]
                
            blinds_logged = True

        policy = players[agent_id]
        action_mask = observation['action_mask']
        legal_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
        
        # q_values = {}
        if policy == "random":
            action = int(np.random.choice(legal_actions))
        else:
            obs_vec = observation['observation']
            batch = Batch(
                obs=Batch(
                    observation=np.expand_dims(obs_vec, axis=0),
                    action_mask=np.expand_dims(action_mask, axis=0)
                ),
                info={}
            )
            result = policy(batch)
            action = int(result.act[0])
            
            # Weryfikacja bezpieczeństwa (awaryjny FOLD lub random)
            if action_mask[action] == 0:
                action = 0 if action_mask[0] == 1 else int(np.random.choice(legal_actions))
                
            # # Wyciąganie Q-Values / Logits dla analityki RL
            # if hasattr(result, 'logits'):
            #     logits = result.logits[0].detach().cpu().numpy() if torch.is_tensor(result.logits) else result.logits[0]
            #     for a_idx, is_legal in enumerate(action_mask):
            #         if is_legal:
            #             q_values[ACTION_NAMES[a_idx]] = float(logits[a_idx])

        board = [get_card_str(c) for c in env.rlcard_env.game.public_cards]
        pot = float(sum(p.in_chips for p in env.rlcard_env.game.players))

        # ---------------------------------------------------------
        # WYKONANIE KROKU I ODCZYT SALDA (Dokładna kwota zakładu)
        # ---------------------------------------------------------
        active_idx = env.active_agents.index(agent_id)
        player_obj = env.rlcard_env.game.players[active_idx]
        chips_before = player_obj.remained_chips
        
        event = {
            "player": agent_id,
            "action": action,
            "action_str": ACTION_NAMES[action],
            "amount": 0.0, 
            "pot_before_action": pot,
            "board": board
            # "q_values": q_values
        }
        current_hand_data["events"].append(event)
        
        env.step(action)
        step_count += 1
        
        # Referencja zachowuje stan po akcji
        event["amount"] = float(chips_before - player_obj.remained_chips)
        
        # Zapis ewentualnego Showdownu
        if showdown_data:
            current_hand_data["showdown"] = showdown_data.copy()
            showdown_data.clear()
        
        if step_count > config.MAX_STEPS_PER_TOURNAMENT:
            print("Ostrzeżenie: Przekroczono limit kroków, przerywam wcześnie.")
            break

    # Zamknięcie ostatniego rozdania w JSON-ie
    if current_hand_data is not None:
        current_hand_data["final_tournament_chips"] = {k: float(v) for k, v in env.tournament_chips.items()}
        history["hands"].append(current_hand_data)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)
        
    print(f"Sukces! Pełen przebieg turnieju zapisano do: {output_file}")


if __name__ == "__main__":
    record_tournament(app.app_config.TABLE_CONFIG, output_file=app.app_config.OUTPUT_LOCATION / app.app_config.OUPUT_FILE_NAME)