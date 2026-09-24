import random
import uuid

import numpy as np
import rlcard
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.limitholdem.utils import compare_hands

import config
from observation.actions import ActionHistory
from observation.opponent_stats import OpponentStatsTracker
from observation.state import (
    PlayerSnapshot,
    calculate_to_call,
    encode_base_observation,
)

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
        # Identyfikator instancji odróżnia równoległe środowiska. Polityki
        # przeciwników łączą go z numerem resetu, aby zachować wylosowany styl
        # przez cały jeden turniej, bez mieszania ośmiu procesów treningowych.
        self._environment_id = uuid.uuid4().hex
        self._tournament_number = 0
        
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
        # Seed jest opcjonalny: trening pozostaje losowy, natomiast ewaluacja
        # może odtwarzać te same rozdania dla kolejnych checkpointów.
        self._tournament_seed = seed
        self._hand_number = 0
        self._tournament_number += 1
        self.tournament_id = f"{self._environment_id}:{self._tournament_number}"
        self.agents = self.possible_agents[:]
        # Turniejowe żetony przechowujemy pod nazwami agentów, co ułatwi odczyt po bankructwach
        self.tournament_chips = {agent: self.starting_chips for agent in self.possible_agents}
        # Statystyki przeciwników obowiązują przez cały turniej, dlatego zerujemy
        # je przy resecie turnieju, a nie przy każdym nowym rozdaniu.
        self.opponent_stats = OpponentStatsTracker(self.possible_agents)
        self.dealer_idx = 0
        self.completed_hands = 0
        self.hand_wins = {agent: 0 for agent in self.possible_agents}
        self.finishing_positions = {}
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {
            agent: {"tournament_id": self.tournament_id}
            for agent in self.agents
        }
        
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
        if self._tournament_seed is not None:
            # Każde rozdanie dostaje inne, ale powtarzalne ziarno. Samo
            # utworzenie nowego środowiska RLCard resetowałoby generator.
            self.rlcard_env.seed(self._tournament_seed + self._hand_number)
        self._hand_number += 1
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

        # RLCard przechowuje żądaną wysokość calla, nawet gdy krótki stack nie
        # był w stanie wpłacić całej kwoty. Własna baza ulicy pozwala wyliczać
        # rzeczywiście pobrane żetony z player.in_chips.
        self.tracked_street = self.rlcard_env.game.stage.value
        self.street_start_contributions = [0.0] * num_active

        # Historia akcji dotyczy wyłącznie bieżącego rozdania. Używa stałych
        # nazw wszystkich czterech miejsc, również gdy ktoś już zbankrutował.
        self.action_history = ActionHistory(self.possible_agents)
        self.opponent_stats.start_hand(self.active_agents)

        # Wskazujemy pierwszego gracza w nowym rozdaniu używając nazwy zmapowanej z RLCard
        self.agent_selection = self.active_agents[rlcard_player_id]

    def _street_contributions(self):
        """Zwróć faktyczne, a nie żądane przez RLCard, wpłaty na ulicy."""
        current_street = self.rlcard_env.game.stage.value
        players = self.rlcard_env.game.players

        if current_street != self.tracked_street:
            self.tracked_street = current_street
            self.street_start_contributions = [
                float(player.in_chips) for player in players
            ]

        return [
            float(player.in_chips) - self.street_start_contributions[index]
            for index, player in enumerate(players)
        ]

    def observe(self, agent):
        # Jeśli agent zbankrutował (został wykluczony ze start_new_hand), zwracamy pustą maskę
        if self.terminations.get(agent, False) or agent not in self.active_agents:
            return {
                "observation": np.zeros(self.observation_size, dtype=np.float32),
                "action_mask": np.zeros(config.ACTION_SPACE, dtype=np.int8)
            }

        # RLCard numeruje wyłącznie graczy obecnych w aktualnym rozdaniu.
        # Zamieniamy ten indeks z powrotem na stałe nazwy player_0...player_3.
        rlcard_idx = self.active_agents.index(agent)
        state = self.rlcard_env.get_state(rlcard_idx)
        raw_obs = state["raw_obs"]
        
        action_mask = np.zeros(config.ACTION_SPACE, dtype=np.int8)
        for action_id in state['legal_actions']:
            action_mask[action_id] = 1

        # Zaczynamy od czterech stałych miejsc. Zbankrutowani gracze pozostają
        # widoczni jako nieaktywni, dzięki czemu znaczenie indeksów się nie zmienia.
        player_states = {
            player_name: PlayerSnapshot(
                stack=0,
                street_contribution=0,
                hand_contribution=0,
                active=False,
            )
            for player_name in self.possible_agents
        }

        street_contributions = self._street_contributions()
        for index, player_name in enumerate(self.active_agents):
            player = self.rlcard_env.game.players[index]
            player_states[player_name] = PlayerSnapshot(
                stack=float(player.remained_chips),
                street_contribution=street_contributions[index],
                hand_contribution=float(player.in_chips),
                active=True,
                folded=player.status == PlayerStatus.FOLDED,
                all_in=player.status == PlayerStatus.ALLIN,
            )

        own_player = self.rlcard_env.game.players[rlcard_idx]
        to_call = calculate_to_call(
            highest_street_contribution=max(street_contributions),
            own_street_contribution=street_contributions[rlcard_idx],
            own_stack=own_player.remained_chips,
        )

        observation = encode_base_observation(
            observer=agent,
            seats=self.possible_agents,
            player_states=player_states,
            button=self.active_agents[self.dealer_idx],
            street=raw_obs["stage"].value,
            pot=float(raw_obs["pot"]),
            to_call=to_call,
            chip_scale=self.starting_chips * self.num_players,
            own_cards=raw_obs["hand"],
            board_cards=raw_obs["public_cards"],
            action_history=self.action_history,
            opponent_stats=self.opponent_stats,
        )
            
        return {
            "observation": observation,
            "action_mask": action_mask
        }

    def _get_showdown_result(self):
        """Zwróć uczestników i zwycięzców publicznie widocznego showdownu."""
        game = self.rlcard_env.game
        contenders = [
            index
            for index, player in enumerate(game.players)
            if player.status in (PlayerStatus.ALIVE, PlayerStatus.ALLIN)
        ]

        # Jedyny pozostały gracz wygrywa przez foldy, a nie przez showdown.
        if len(contenders) < 2 or len(game.public_cards) != 5:
            return (), ()

        hands = [
            (
                [card.get_index() for card in player.hand + game.public_cards]
                if index in contenders
                else None
            )
            for index, player in enumerate(game.players)
        ]
        winner_mask = compare_hands(hands)

        showdown_players = tuple(
            self.active_agents[index] for index in contenders
        )
        winners = tuple(
            self.active_agents[index]
            for index, won in enumerate(winner_mask)
            if won
        )
        return showdown_players, winners

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

        # Kontekst zapisujemy przed wykonaniem ruchu. Po rlcard_env.step()
        # biblioteka może już przejść na następną ulicę i wyzerować wkłady.
        player = self.rlcard_env.game.players[rlcard_idx]
        street = self.rlcard_env.game.stage.value
        street_contributions = self._street_contributions()
        highest_contribution = max(street_contributions)
        to_call = calculate_to_call(
            highest_street_contribution=highest_contribution,
            own_street_contribution=street_contributions[rlcard_idx],
            own_stack=player.remained_chips,
        )
        all_in_increases_bet = (
            action == 4
            and street_contributions[rlcard_idx] + player.remained_chips
            > highest_contribution
        )

        # Blind nie jest raisem. Fold-to-raise otrzymuje okazję dopiero wtedy,
        # gdy na tej ulicy zapisano prawdziwą agresywną akcję.
        facing_raise = (
            to_call > 0
            and self.action_history.raise_counts[street] > 0
        )
        self.action_history.record(
            self.agent_selection,
            action,
            street,
            to_call=to_call,
            all_in_increases_bet=all_in_increases_bet,
        )
        self.opponent_stats.record_action(
            self.agent_selection,
            action,
            street,
            to_call=to_call,
            facing_raise=facing_raise,
            all_in_increases_bet=all_in_increases_bet,
        )

        _, next_rlcard_id = self.rlcard_env.step(action)
        
        if self.rlcard_env.is_over():
            if self.debug:
                print("Hand is over")
            # Zatwierdzamy statystyki dopiero po zakończeniu rozdania. Dzięki
            # temu niedokończone VPIP/PFR nie trafiają do bieżącej obserwacji.
            showdown_players, winners = self._get_showdown_result()
            self.opponent_stats.finish_hand(
                showdown_players=showdown_players,
                winners=winners,
            )

            payoffs = self.rlcard_env.get_payoffs()
            self.completed_hands += 1
            for index, payoff in enumerate(payoffs):
                if payoff > 0:
                    self.hand_wins[self.active_agents[index]] += 1
            if self.debug:
                print(f"Wynik rozdania (dla aktywnych): {payoffs}")

            # nagrody po każdym rozdaniu
            for i, payoff in enumerate(payoffs):
                agent_name = self.active_agents[i]
                self.tournament_chips[agent_name] += payoff
                self.rewards[agent_name] = float(payoff) / self.starting_chips # znormalizowana nagroda
                
            # Weryfikacja bankructw po rozliczeniu żetonów
            active_before = len(self.active_agents)
            active_count = 0
            newly_eliminated = []
            for agent in self.active_agents:
                if self.tournament_chips[agent] <= 0:
                    self.terminations[agent] = True
                    newly_eliminated.append(agent)
                else:
                    active_count += 1

            if newly_eliminated:
                # Gracze odpadający w tym samym rozdaniu zajmują ex aequo
                # średnią z przypadających im miejsc.
                tied_position = (active_count + 1 + active_before) / 2
                for agent in newly_eliminated:
                    self.finishing_positions[agent] = tied_position
                    
            if active_count <= 1:
                # Ostatni na polu bitwy, turniej zakończony
                winner = max(self.tournament_chips, key=self.tournament_chips.get)
                self.finishing_positions[winner] = 1.0
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
