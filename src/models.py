import torch
import torch.nn as nn
from tianshou.data import Batch
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy

class MaskedActor(nn.Module):
    """Sieć Actora: zwraca logity dla dozwolonych akcji"""
    def __init__(self, state_shape=68, action_shape=5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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
        device = next(self.model.parameters()).device
        observation = torch.as_tensor(observation, dtype=torch.float32)
        observation = observation.to(device)
        
        logits = self.model(observation)
        
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool).to(device)
            logits = torch.where(action_mask, logits, torch.tensor(-1e9, device=device))
            
        return logits, state

class Critic(nn.Module):
    """Sieć Critica: zwraca pojedynczą wartość (Value) przewidującą sumę nagród z danego stanu"""
    def __init__(self, state_shape=68):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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
        observation = torch.as_tensor(observation, dtype=torch.float32)
        device = next(self.model.parameters()).device
        observation = observation.to(device)
        
        # ZMIANA: Zwracamy wyłącznie sam tensor, bez `state`
        return self.model(observation)

class CPUActionActorPolicy(ProbabilisticActorPolicy):
    """
    Nakładka na politykę, która po wyliczeniu akcji przez sieć na GPU,
    zrzuca wynikowy tensor do formatu NumPy na CPU.
    Zapobiega to błędom przy łączeniu (Batch.cat) z agentami losowymi.
    """
    def forward(self, batch, state=None, **kwargs):
        out = super().forward(batch, state, **kwargs)
        if hasattr(out, 'act') and isinstance(out.act, torch.Tensor):
            out.act = out.act.cpu().numpy()
        return out