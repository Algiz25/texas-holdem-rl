import torch
import torch.nn as nn
from tianshou.data import Batch

import config

class MaskedActor(nn.Module):
    """Sieć Actora: zwraca logity dla dozwolonych akcji, korzystając z powiększonej architektury dwugałęziowej."""
    def __init__(
        self,
        state_shape=config.OBSERVATION_SIZE,
        action_shape=config.ACTION_SPACE,
    ):
        super().__init__()
        
        # Gałąź 1: Przetwarzanie kart (własne 52 + stół 52 = 104)
        self.cards_branch = nn.Sequential(
            nn.Linear(104, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Gałąź 2: Przetwarzanie metadanych (stacki, oddsy, statystyki, historia)
        meta_size = state_shape - 104
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Głowa decyzyjna: łączy cechy kart (256) i metadanych (256) = 512
        self.decision_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape)
        )

    def forward(self, obs, state=None, info={}):
        action_mask = None
        
        # Ekstrakcja z obiektu Batch z Tianshou
        if isinstance(obs, Batch):
            if hasattr(obs, 'observation') and hasattr(obs, 'action_mask'):
                observation = obs.observation
                action_mask = obs.action_mask
            elif hasattr(obs, 'obs') and hasattr(obs, 'mask'):
                observation = obs.obs
                action_mask = obs.mask
            else:
                observation = obs
        # Ekstrakcja z klasycznego słownika
        elif isinstance(obs, dict):
            if "observation" in obs:
                observation = obs["observation"]
            else:
                observation = obs
            if "action_mask" in obs:
                action_mask = obs["action_mask"]
        else:
            observation = obs

        # Konwersja na tensor
        device = next(self.parameters()).device
        observation = torch.as_tensor(observation, dtype=torch.float32).to(device)
        
        # Rozdzielenie wejścia na karty i metadane
        cards_input = observation[..., :104]
        meta_input = observation[..., 104:]
        
        # Przepuszczenie przez gałęzie
        cards_features = self.cards_branch(cards_input)
        meta_features = self.meta_branch(meta_input)
        
        # Połączenie cech (konkatenacja na ostatnim wymiarze)
        combined_features = torch.cat([cards_features, meta_features], dim=-1)
        
        # Wyliczenie logitów akcji
        logits = self.decision_head(combined_features)
        
        # Maskowanie niedozwolonych akcji
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)
            logits = torch.where(action_mask, logits, torch.tensor(-1e9, device=device))
            
        return logits, state

class Critic(nn.Module):
    """Sieć Critica: ocenia stan gry korzystając z powiększonej architektury dwugałęziowej."""
    def __init__(self, state_shape=config.OBSERVATION_SIZE):
        super().__init__()
        
        self.cards_branch = nn.Sequential(
            nn.Linear(104, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        meta_size = state_shape - 104
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, state=None, info={}):
        # Ekstrakcja z obiektu Batch z Tianshou
        if isinstance(obs, Batch):
            if hasattr(obs, 'observation'):
                observation = obs.observation
            elif hasattr(obs, 'obs'):
                observation = obs.obs
            else:
                observation = obs
        # Ekstrakcja z klasycznego słownika
        elif isinstance(obs, dict):
            if "observation" in obs:
                observation = obs["observation"]
            else:
                observation = obs
        else:
            observation = obs

        # Konwersja na tensor
        device = next(self.parameters()).device
        observation = torch.as_tensor(observation, dtype=torch.float32).to(device)
        
        # Rozdzielenie wejścia
        cards_input = observation[..., :104]
        meta_input = observation[..., 104:]
        
        # Przepuszczenie przez gałęzie
        cards_features = self.cards_branch(cards_input)
        meta_features = self.meta_branch(meta_input)
        
        # Połączenie cech
        combined_features = torch.cat([cards_features, meta_features], dim=-1)
        
        # Zwracamy pojedynczą wartość (Value)
        return self.value_head(combined_features)