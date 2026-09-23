import sys
import numpy as np
import torch
from tianshou.data import Batch
from .. import config
from ..environment import TexasHoldemTournament

action_mapping = {
    0: "FOLD", 1: "CHECK/CALL", 2: "RAISE HALF", 3: "RAISE POT", 4: "ALL IN"
}

class BasePokerEvaluator:
    def __init__(self, num_tournaments=config.NUM_TOURNAMENTS_PER_EVAL, model_path='model.pth', training_phase="RANDOM"):
        self.num_tournaments = num_tournaments
        self.model_path = model_path
        self.training_phase = training_phase
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.env = TexasHoldemTournament(num_players=config.NUM_PLAYERS, starting_chips=config.STARTING_CHIPS, debug=False)
        
    def load_policy(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate(self):
        policy = self.load_policy()
        if policy is None:
            return

        wins = 0
        stats = {
            'total_actions': 0, 'folds': 0, 'calls': 0, 'raises': 0,
            'preflop_opportunities': 0, 'vpip_actions': 0, 'pfr_actions': 0,
        }

        print(f"\nRozpoczynam ewaluację (Faza: {self.training_phase}) na {self.device}...")
        
        for t in range(self.num_tournaments):
            self.env.reset()
            step_count = 0
            tournament_actions = []
            
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = self.env.last()
                
                if termination or truncation:
                    self.env.step(None)
                    continue
                    
                obs = observation['observation']
                mask = observation['action_mask']
                legal_actions = [i for i, valid in enumerate(mask) if valid == 1]
                
                batch = Batch(
                    obs=Batch(
                        observation=np.expand_dims(obs, axis=0),
                        action_mask=np.expand_dims(mask, axis=0)
                    ),
                    info={}
                )
                
                # --- LOGIKA WYBORU AKCJI (ŚRODOWISKO ZALEŻNE OD FAZY) ---
                # przeciwnicy będą grać tym samym mózgiem w self i advanced
                is_learner = (agent == "player_0")
                is_opponent_self = (agent in ["player_1", "player_2", "player_3"] and self.training_phase == "SELF") or \
                                   (agent in ["player_1", "player_3"] and self.training_phase == "ADVANCED")
                
                if is_learner or is_opponent_self:
                    result = policy(batch)
                    action = int(result.act[0])
                    # Zabezpieczenie przed błędem wczesnego PPO (niedozwolona akcja)
                    if mask[action] == 0:
                        action = 0 if mask[0] == 1 else np.random.choice(legal_actions)
                else:
                    # Pozostali agenci grają losowo
                    action = int(np.random.choice(legal_actions))
                
                # --- AKTUALIZACJA STATYSTYK (tylko dla ucznia) ---
                if is_learner:
                    stats['total_actions'] += 1
                    if action == 0:
                        stats['folds'] += 1
                    elif action == 1:
                        stats['calls'] += 1
                    elif action in [2, 3, 4]:
                        stats['raises'] += 1
                        
                    is_preflop = (obs[55] == 1.0)
                    if is_preflop:
                        stats['preflop_opportunities'] += 1
                        if action != 0:
                            stats['vpip_actions'] += 1
                        if action in [2, 3, 4]:
                            stats['pfr_actions'] += 1
                    
                    if step_count < 15:
                        tournament_actions.append(action_mapping[action])

                self.env.step(action)
                step_count += 1
                
                if step_count > config.MAX_STEPS_PER_TOURNAMENT:
                    print(f"\n[Turniej {t+1}/{self.num_tournaments}] PRZERWANY! Limit {config.MAX_STEPS_PER_TOURNAMENT} kroków.")
                    break 
                    
            sys.stdout.write(f"\rZakończono turniej {t+1}/{self.num_tournaments} (Kroki: {step_count})")
            sys.stdout.flush()
            
            if t == 0:
                print(f"\n[Podgląd] Pierwsze decyzje badanego gracza:\n -> {', '.join(tournament_actions)}\n")

            winner = max(self.env.tournament_chips, key=self.env.tournament_chips.get)
            if winner == "player_0":
                wins += 1
        
        self._print_results(wins, stats)

    def _print_results(self, wins, stats):
        win_rate = (wins / self.num_tournaments) * 100
        print("\n\n=== WYNIKI EWALUACJI ===")
        print(f"Suma podjętych decyzji gracza 0: {stats['total_actions']}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        if stats['total_actions'] > 0:
            print("\n--- DYSTRYBUCJA AKCJI ---")
            print(f"Fold Rate:  {stats['folds'] / stats['total_actions'] * 100:.1f}%")
            print(f"Call Rate:  {stats['calls'] / stats['total_actions'] * 100:.1f}%")
            print(f"Raise Rate: {stats['raises'] / stats['total_actions'] * 100:.1f}%")

        if stats['preflop_opportunities'] > 0:
            print("\n--- ZAAWANSOWANE STATYSTYKI POKEROWE ---")
            vpip = stats['vpip_actions'] / stats['preflop_opportunities'] * 100
            pfr = stats['pfr_actions'] / stats['preflop_opportunities'] * 100
            print(f"VPIP (Voluntarily Put in Pot): {vpip:.1f}%")
            print(f"PFR (Pre-Flop Raise):          {pfr:.1f}%")
