import sys
import numpy as np
import torch
from tianshou.data import Batch
from pettingzoo_tournament import TexasHoldemTournament

action_mapping = {
    0: "FOLD", 1: "CHECK/CALL", 2: "RAISE HALF", 3: "RAISE POT", 4: "ALL IN"
}

class BasePokerEvaluator:
    def __init__(self, num_tournaments=10, model_path='model.pth'):
        self.num_tournaments = num_tournaments
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Inicjalizacja środowiska z wymiarami z plików DQN i PPO[cite: 1, 2]
        self.env = TexasHoldemTournament(num_players=4, starting_chips=200, debug=False)
        
    def load_policy(self):
        """
        Metoda do nadpisania. Powinna załadować wagi i zwrócić
        obiekt polityki, który przyjmuje 'batch' i zwraca akcję.
        """
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

        # Zabezpieczenie przed nieskończonym foldowaniem[cite: 1, 2]
        max_steps_per_tournament = 1000 
        print(f"\nRozpoczynam ewaluację na {self.device}...")
        
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
                
                batch = Batch(
                    obs=Batch(
                        observation=np.expand_dims(obs, axis=0),
                        action_mask=np.expand_dims(mask, axis=0)
                    ),
                    info={}
                )
                
                # AGENT + 3 losowych graczy
                if agent == "player_0":
                    result = policy(batch)
                    action = int(result.act[0])
                else:
                    # Dla pozostałych graczy wybieramy losową dozwoloną akcję
                    legal_actions = [i for i, valid in enumerate(mask) if valid == 1]
                    action = int(np.random.choice(legal_actions))
                
                # --- AKTUALIZACJA STATYSTYK ---[cite: 1, 2]
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
                
                if step_count > max_steps_per_tournament:
                    print(f"\n[Turniej {t+1}/{self.num_tournaments}] PRZERWANY! Limit {max_steps_per_tournament} kroków.")
                    break 
                    
            sys.stdout.write(f"\rZakończono turniej {t+1}/{self.num_tournaments} (Kroki: {step_count})")
            sys.stdout.flush()
            
            # Podgląd pierwszych akcji w turnieju[cite: 1, 2]
            if t == 0:
                print(f"\n[Podgląd] Pierwsze 15 decyzji agentów w turnieju 1:\n -> {', '.join(tournament_actions)}\n")

            winner = max(self.env.tournament_chips, key=self.env.tournament_chips.get)
            if winner == "player_0":
                wins += 1
        
        self._print_results(wins, stats)

    def _print_results(self, wins, stats):
        win_rate = (wins / self.num_tournaments) * 100
        print("\n\n=== WYNIKI EWALUACJI ===")
        print(f"Suma wszystkich podjętych decyzji: {stats['total_actions']}")
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


