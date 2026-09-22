import rlcard
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from rlcard.games.limitholdem import PlayerStatus
from gymnasium.spaces import Box, Discrete
import random
import config

action_mapping = {
    0: "fold",
    1: "check_call",
    2: "raise_half_pot",
    3: "raise_pot",
    4: "all_in"
}

class TexasHoldemTournament(AECEnv):
    def __init__(self, num_players=config.NUM_PLAYERS, starting_chips=config.STARTING_CHIPS, debug=False):
        super().__init__()
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.debug = debug
        
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents = self.possible_agents[:]

        self.observation_size = config.OBSERVATION_SIZE # Wielkość wekotra obserwacji
        self.action_spaces = {agent: Discrete(config.ACTION_SPACE) for agent in self.possible_agents} 
        self.observation_spaces = {
            agent: Box(low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32)  
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        # Turniejowe żetony przechowujemy pod nazwami agentów, co ułatwi odczyt po bankructwach
        self.tournament_chips = {agent: self.starting_chips for agent in self.possible_agents}
        self.dealer_idx = 0
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._start_new_hand()

    def _start_new_hand(self):
        # Aktywni agenci to tacy, którzy są w środowisku i nie mają flagi terminations
        self.active_agents = [a for a in self.agents if not self.terminations.get(a, False)]
        num_active = len(self.active_agents)
        
        if num_active <= 1:
            return

        # Upewniamy się, że wskaźnik krupiera mieści się w puli pozostałych przy stole graczy
        self.dealer_idx = self.dealer_idx % num_active
        
        self.rlcard_env = rlcard.make(
            'no-limit-holdem', 
            config={
                'game_num_players': num_active, 
                'chips_for_each': self.starting_chips, 
                'dealer_id': self.dealer_idx,
            }
        )
        _, rlcard_player_id = self.rlcard_env.reset()

        if self.debug:
            print("\nNEW HAND")
        
        # Odejmujemy wpłacone w resecie blindy od turniejowego salda aktywnych graczy
        for i, player in enumerate(self.rlcard_env.game.players):
            agent_name = self.active_agents[i]
            actual_chips = self.tournament_chips[agent_name]
            
            # Weryfikacja, czy gracz ma wystarczająco żetonów na blinda
            if actual_chips <= player.in_chips:
                player.in_chips = actual_chips
                player.remained_chips = 0
                player.status = PlayerStatus.ALLIN
            else:
                player.remained_chips = actual_chips - player.in_chips
                
            # Aktualizacja tablicy stawek w rundzie (w przypadku obciętego blinda)
            self.rlcard_env.game.round.raised[i] = player.in_chips
            if self.debug:
                print(f"{agent_name} żetony: {player.remained_chips}, na kupce: {player.in_chips}")


        # Wskazujemy pierwszego gracza w nowym rozdaniu używając nazwy zmapowanej z RLCard
        self.agent_selection = self.active_agents[rlcard_player_id]

    def observe(self, agent):
        # Jeśli agent zbankrutował (został wykluczony ze start_new_hand), zwracamy pustą maskę
        if self.terminations.get(agent, False) or agent not in self.active_agents:
            return {
                "observation": np.zeros(self.observation_size, dtype=np.float32),
                "action_mask": np.zeros(config.ACTION_SPACE, dtype=np.int8)
            }

        # Pobieramy stan na podstawie indeksu w aktywnym rozdaniu
        rlcard_idx = self.active_agents.index(agent)
        state = self.rlcard_env.get_state(rlcard_idx)
        
        action_mask = np.zeros(config.ACTION_SPACE, dtype=np.int8)
        for action_id in state['legal_actions']:
            action_mask[action_id] = 1

        total_chips_in_play = self.starting_chips * self.num_players

        obs = np.zeros(self.observation_size, dtype=np.float32)
        raw_obs = state['raw_obs']

        # 0-51: Karty (kopiujemy z domyślnego wektora RLCard)
        obs[0:52] = state['obs'][0:52]
        
        # 52: Nasze żetony na kupce (znormalizowane)
        obs[52] = state['obs'][52] / total_chips_in_play

        # 53: Maksymalny zakład na stole (znormalizowane)
        obs[53] = state['obs'][53] / total_chips_in_play
        
        # 54: Ile brakuje do sprawdzenia (To Call) (znormalizowane)
        obs[54] = (state['obs'][53] - state['obs'][52]) / total_chips_in_play

        # 55: Całkowita pula (znormalizowane)
        obs[55] = float(raw_obs['pot']) / total_chips_in_play
        
        # 55-58: Faza gry (One-Hot)
        stage_val = raw_obs['stage'].value
        if stage_val <= 3:
            obs[56 + stage_val] = 1.0

        # 59-62: Stacki aktywnych graczy ułożone relatywnie
        stakes = raw_obs['stakes']
        current = raw_obs['current_player']
        num_active = len(self.active_agents)
        for i in range(num_active):
            obs[60 + i] = float(stakes[(current + i) % num_active]) / total_chips_in_play

        # 64-67: Pozycja gracza względem Dealera (One-Hot do 4 graczy)
        relative_position = (current - self.dealer_idx) % num_active
        obs[64 + relative_position] = 1.0
            
        return {
            "observation": obs,
            "action_mask": action_mask
        }

    def step(self, action):
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(self.agent_selection, False):
            if self.debug:
                print(f'{self.agent_selection} is dead and will be removed')
            self._was_dead_step(action)
            return

        # --- ZABEZPIECZENIE PRZED BUGIEM RLCARD ---
        current_obs = self.observe(self.agent_selection)
        if current_obs["action_mask"][action] == 0:
            # Jeśli kolektor losowy wybierze złą akcję, wymuszamy FOLD (0) 
            # (lub 1 dla CHECK_CALL)
            action = 0 
        # ------------------------------------------

        self._clear_rewards()
        if self.debug:
            print(f"{self.agent_selection} wykona akcje: {action_mapping[action]}")

        # Tłumaczymy wybór agenta AEC na ruch na planszy RLCard
        rlcard_idx = self.active_agents.index(self.agent_selection)
        _, next_rlcard_id = self.rlcard_env.step(action)
        
        if self.rlcard_env.is_over():
            if self.debug:
                print("Hand is over")
            payoffs = self.rlcard_env.get_payoffs()
            if self.debug:
                print(f"Wynik rozdania (dla aktywnych): {payoffs}")

            # nagrody po każdym rozdaniu
            for i, payoff in enumerate(payoffs):
                agent_name = self.active_agents[i]
                self.tournament_chips[agent_name] += payoff
                self.rewards[agent_name] = float(payoff) / self.starting_chips # znormalizowana nagroda
                
            # Weryfikacja bankructw po rozliczeniu żetonów
            active_count = 0
            for agent in self.active_agents:
                if self.tournament_chips[agent] <= 0:
                    self.terminations[agent] = True
                else:
                    active_count += 1
                    
            if active_count <= 1:
                # Ostatni na polu bitwy, turniej zakończony
                for agent in self.agents:
                    self.terminations[agent] = True
                    # self.rewards[agent] = float(self.tournament_chips[agent] - self.starting_chips) # nagroda na koniec turnieju - może warto dodać większą za wygranie?
                # Ustawiamy usuwanie ostatniego agenta
                self.agent_selection = self._deads_step_first()
            else:
                self.dealer_idx += 1
                self._start_new_hand()
                # Wymusza wejście AECEnv w funkcję _was_dead_step podczas następnego iterowania 
                # co bezproblemowo zaktualizuje kolejkę do następnego gracza 
                self.agent_selection = self._deads_step_first()
        else:
            self.agent_selection = self.active_agents[next_rlcard_id]

        self._accumulate_rewards()


# === SKRYPT TESTOWY ===
# if __name__ == "__main__":
#     debug = False
#     env = TexasHoldemTournament(num_players=4, starting_chips=200, debug=debug)
#     env.reset()

#     if debug:
#         print("=== START TURNIEJU ===")
    
#     for agent in env.agent_iter():
#         if debug:
#             print(f"Tura {agent}")
#         observation, reward, termination, truncation, info = env.last()
        
#         if termination or truncation:
#             action = None
#         else:
#             action_mask = observation["action_mask"]
#             legal_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
#             action = random.choice(legal_actions)
            
#         env.step(action)

#     if debug:
#         print("\n=== KONIEC TURNIEJU ===")
#         print("Końcowe salda graczy:")
#         for agent, chips in env.tournament_chips.items():
#             print(f"{agent}: {chips} żetonów")