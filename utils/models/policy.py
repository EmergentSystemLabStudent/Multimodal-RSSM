import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal
from typing import Dict

from utils.models.encoder import build_Encoder

class ValueModel(nn.Module):
    def __init__(self, belief_size: int, state_size: int, hidden_size: int, activation_function: str =      'relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        # reshape input tensors
        (T, B), features_shape = h_t.size()[:2], h_t.size()[2:]
        h_t = h_t.reshape(T*B, *features_shape)

        (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
        s_t = s_t.reshape(T*B, *features_shape)

        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        loc = self.fc4(hidden).squeeze(dim=1)
        features_shape = loc.size()[1:]
        loc = loc.reshape(T, B, *features_shape)
        scale = torch.ones_like(loc)
        return {'loc': loc, 'scale': scale}

    def get_log_prob(self, h_t, s_t, r_t):
        loc_and_scale = self.forward(h_t, s_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(r_t)
        return log_prob


class Pie(nn.Module):
    def __init__(self, 
                 belief_size: int, 
                 state_size: int, 
                 hidden_size: int, 
                 action_size: int, 
                 dist: str = 'tanh_normal',
                 activation_function: str = 'elu', 
                 min_std: float = 1e-4, 
                 init_std: float = 5, 
                 mean_scale: float = 5
                ):
        super().__init__()
        self.belief_size = belief_size
        self.state_size = state_size
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2*action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        raw_init_std = torch.log(torch.exp(torch.tensor(
            self._init_std, dtype=torch.float32)) - 1)
        if (not h_t.shape[-1] == self.belief_size) or (not s_t.shape[-1] == self.state_size):
            raise NotImplementedError("state shape is not match. h_t: {}, s_t: {}".format(h_t.shape[-1], s_t.shape[-1]))
        x = torch.cat([h_t, s_t], dim=1)
        hidden = self.act_fn(self.fc1(x))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * \
            torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return {'loc': action_mean, 'scale': action_std}

    def sample(self, h_t, s_t, sample_shape=torch.Size([])):
        loc_and_scale = self.forward(h_t, s_t)
        action = Normal(loc_and_scale['loc'], loc_and_scale['scale']).rsample(sample_shape=sample_shape)
        return action
    
    def get_log_prob(self, h_t, s_t, a_t):
        loc_and_scale = self.forward(h_t, s_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(a_t)
        return log_prob

class ActorModel(nn.Module):
    def __init__(self, 
                 belief_size: int, 
                 state_size: int, 
                 hidden_size: int, 
                 action_size: int, 
                 dist: str = 'tanh_normal',
                 activation_function: str = 'elu', 
                 min_std: float = 1e-4, 
                 init_std: float = 5, 
                 mean_scale: float = 5
                ):
        super().__init__()
        self.pie = Pie(belief_size, state_size, hidden_size, action_size, dist=dist,
                       activation_function=activation_function, min_std=min_std, init_std=init_std, mean_scale=mean_scale)
        self.modules = self.pie.modules

    def get_action(self, belief: torch.Tensor, state: torch.Tensor, det: bool = False) -> torch.Tensor:
        if det:
            # get mode
            actions = self.pie.sample(belief, state, sample_shape=[100])  # (100, 2450, 6)
            actions = torch.tanh(actions)
            batch_size = actions.size(1)
            feature_size = actions.size(2)
            logprob = self.pie.get_log_prob(belief, state, actions)  # (100, 2450, 6)
            logprob -= torch.log(1 - actions.pow(2) + 1e-6)
            logprob = logprob.sum(dim=-1)
            indices = torch.argmax(logprob, dim=0).reshape(
                1, batch_size, 1).expand(1, batch_size, feature_size)
            return torch.gather(actions, 0, indices).squeeze(0)

        else:
            return torch.tanh(self.pie.sample(belief, state))

    def forward(self, h_t: torch.Tensor, s_t: torch.Tensor) -> Dict:
        return self.get_action(h_t, s_t)

class Pie_emb(nn.Module):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 action_size: int, 
                 dist: str = 'tanh_normal',
                 activation_function: str = 'elu', 
                 min_std: float = 1e-4, 
                 init_std: float = 5, 
                 mean_scale: float = 5
                ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 2*action_size)
        self.modules = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def forward(self, x_t: torch.Tensor) -> Dict:
        raw_init_std = torch.log(torch.exp(torch.tensor(
            self._init_std, dtype=torch.float32)) - 1)
        hidden = self.act_fn(self.fc1(x_t))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        hidden = self.act_fn(self.fc4(hidden))
        action = self.fc5(hidden).squeeze(dim=1)

        action_mean, action_std_dev = torch.chunk(action, 2, dim=1)
        action_mean = self._mean_scale * \
            torch.tanh(action_mean / self._mean_scale)
        action_std = F.softplus(action_std_dev + raw_init_std) + self._min_std
        return {'loc': action_mean, 'scale': action_std}

    def sample(self, x_t, sample_shape=torch.Size([])):
        loc_and_scale = self.forward(x_t)
        action = Normal(loc_and_scale['loc'], loc_and_scale['scale']).rsample(sample_shape=sample_shape)
        return action
    
    def get_log_prob(self, x_t, a_t):
        loc_and_scale = self.forward(x_t)
        dist = Normal(loc_and_scale['loc'], loc_and_scale['scale'])
        log_prob = dist.log_prob(a_t)
        return log_prob
    
class ActorModel_Enc(nn.Module):
    def __init__(self, 
                 name,
                 observation_shapes,
                 embedding_size,
                 normalization,
                 hidden_size: int, 
                 action_size: int, 
                 dist: str = 'tanh_normal',
                 activation_function: str = 'elu', 
                 min_std: float = 1e-4, 
                 init_std: float = 5, 
                 mean_scale: float = 5
                ):
        super().__init__()
        self.encoder = build_Encoder(name=name,
                                    observation_shapes=observation_shapes,
                                    embedding_size=embedding_size,
                                    activation_function=activation_function,
                                    normalization=normalization,
                                    )
        self.pie = Pie_emb(embedding_size["fusion"], hidden_size, action_size, dist=dist,
                       activation_function=activation_function["dense"], min_std=min_std, init_std=init_std, mean_scale=mean_scale)
        self.modules = self.encoder.modules + self.pie.modules

    def get_action(self, o_t: torch.Tensor, det: bool = False) -> torch.Tensor:
        x_t = self.encoder(o_t)
        if det:
            # get mode
            actions = self.pie.sample(x_t, sample_shape=[100])  # (100, 2450, 6)
            actions = torch.tanh(actions)
            batch_size = actions.size(1)
            feature_size = actions.size(2)
            logprob = self.pie.get_log_prob(x_t, actions)  # (100, 2450, 6)
            logprob -= torch.log(1 - actions.pow(2) + 1e-6)
            logprob = logprob.sum(dim=-1)
            indices = torch.argmax(logprob, dim=0).reshape(
                1, batch_size, 1).expand(1, batch_size, feature_size)
            return torch.gather(actions, 0, indices).squeeze(0)

        else:
            return torch.tanh(self.pie.sample(x_t))

    def forward(self, o_t: torch.Tensor) -> Dict:
        return self.get_action(o_t)


    def get_log_prob(self, o_t, a_t):
        x_t = self.encoder(o_t)
        return self.pie.get_log_prob(x_t, a_t)

# class Discriminator(nn.Module):
#     def __init__(self, s_size: int, action_size: int, hidden_size: int, activation='relu'):
#         # D(s_t, a_t)
#         super().__init__()
#         self.act_fn = getattr(F, activation)
#         self.fc1 = nn.Linear(s_size + action_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.modules = [self.fc1, self.fc2, self.fc3]

#     def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> Dict:
#         # reshape input tensors
#         (T, B), features_shape = s_t.size()[:2], s_t.size()[2:]
#         s_t = s_t.reshape(T*B, *features_shape)

#         (T, B), features_shape = a_t.size()[:2], a_t.size()[2:]
#         a_t = a_t.reshape(T*B, *features_shape)

#         x = torch.cat([s_t, a_t], dim=1)
#         hidden = self.act_fn(self.fc1(x))
#         hidden = self.act_fn(self.fc2(hidden))
#         x = self.fc3(hidden).squeeze(dim=1)
        
#         features_shape = x.size()[1:]
#         x = x.reshape(T, B, *features_shape)
        
#         return x
