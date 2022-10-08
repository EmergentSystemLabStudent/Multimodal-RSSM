import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal
from typing import Dict


class RewardModel(nn.Module):
    def __init__(self, h_size: int, s_size: int, hidden_size: int, activation='relu'):
        # p(r_t | h_t, s_t)
        super().__init__()
        self.act_fn = getattr(F, activation)
        self.fc1 = nn.Linear(s_size + h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        features_shape = reward.size()[1:]
        reward = reward.reshape(T, B, *features_shape)
        scale = torch.ones_like(reward)
        return {'loc': reward, 'scale': scale}
    
    def get_log_prob(self, h_t, s_t, r_t):
        loc_and_scale = self.forward(h_t, s_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(r_t)
        return log_prob