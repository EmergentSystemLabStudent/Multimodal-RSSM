import itertools

from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions

from torch.distributions import Normal
from typing import Dict


def bottle_tupele(f, x_tuple, var_name: str = '', kwargs={}):
    x_tuple = list(x_tuple.values())[0]
    x_size = x_tuple.size()
    x = x_tuple.reshape(x_size[0] * x_size[1], *x_size[2:])

    y = f(x, **kwargs)
    if var_name != '':
        y = y[var_name]
    y_size = y.size()
    output = y.reshape(x_size[0], x_size[1], *y_size[1:])
    return output

def bottle_tupele_multimodal(f, x_tuples, var_name: str = '', kwargs={}):
    xs_size = []
    xs = dict()
    for name in x_tuples.keys():
        x_size = x_tuples[name].size()
        x = x_tuples[name].reshape(x_size[0] * x_size[1], *x_size[2:])

        xs_size.append(x_size)
        xs[name] = x

    y = f(xs, **kwargs)
    if var_name != '':
        y = y[var_name]
    output = dict()
    for name in y.keys():
        if torch.is_tensor(y[name]):
            y_size = y[name].size()
            output[name] = y[name].reshape(xs_size[0][0], xs_size[0][1], *y_size[1:])
        else:
            output[name] = dict()
            for k in y[name].keys():
                y_size = y[name][k].size()
                output[name][k] = y[name][k].reshape(xs_size[0][0], xs_size[0][1], *y_size[1:])
    return output

def poe(mu, scale):
    # precision of i-th Gaussian expert at point x
    T = 1. / scale
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_scale = 1. / torch.sum(T, dim=0)
    return pd_mu, pd_scale

def get_poe_state(expert_means,
                  expert_std_devs
                  ):
    experts_loc = []
    experts_scale = []
    for name in expert_means.keys():
        experts_loc.append(expert_means[name])
        experts_scale.append(expert_std_devs[name])

    experts_loc = torch.stack(experts_loc)
    experts_scale = torch.stack(experts_scale)

    posterior_means, posterior_std_devs = poe(experts_loc, experts_scale)
    posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
    return posterior_states, posterior_means, posterior_std_devs

def calc_subset_states(expert_means,
                       expert_std_devs,
                       ):
    expert_keys = list(expert_means.keys())
    expert_keys.remove("prior_expert")
    prior_expert_means = expert_means["prior_expert"]
    prior_expert_std_devs = expert_std_devs["prior_expert"]
    
    subset_means = []
    subset_std_devs = []

    for n in range(len(expert_keys)+1):
        combination = list(itertools.combinations(expert_keys, n))
        for experts in combination:
            means = [prior_expert_means]
            std_devs = [prior_expert_std_devs]
            for expert in experts:
                means.append(expert_means[expert])
                std_devs.append(expert_std_devs[expert])
            expert_loc = torch.stack(means)
            expert_scale = torch.stack(std_devs)
            subset_mean, subset_std_dev = poe(expert_loc, expert_scale)
            subset_means.append(subset_mean)
            subset_std_devs.append(subset_std_dev)
    return subset_means, subset_std_devs

def get_mopoe_state(expert_means,
                    expert_std_devs
                    ):
    subset_means, subset_std_devs = calc_subset_states(expert_means, expert_std_devs)

    num_components = len(subset_means)
    num_samples = subset_means[0].shape[-1]
    w_modalities = (1/float(num_components))*torch.ones(num_components).to(subset_means[0].device)
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k-1])
        if k == w_modalities.shape[0]-1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    posterior_means = torch.cat([subset_means[k][:, :, idx_start[k]:idx_end[k]] for k in range(w_modalities.shape[0])], dim=-1)
    posterior_std_devs = torch.cat([subset_std_devs[k][:, :, idx_start[k]:idx_end[k]] for k in range(w_modalities.shape[0])], dim=-1)
    posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
    return posterior_states, posterior_means, posterior_std_devs

class StochasticStateModel(nn.Module):
    """p(s_t | h_t)"""

    def __init__(self, h_size: int, hidden_size: int, activation: nn.Module, s_size: int, min_std_dev: float):
        super().__init__()
        self.fc1 = nn.Linear(h_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.activation = activation
        self.min_std_dev = min_std_dev

    def forward(self, h_t) -> Dict:
        hidden = self.activation(self.fc1(h_t))
        loc, scale = torch.chunk(
            self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def sample(self, h_t):
        loc_and_scale = self.forward(h_t)
        states = Normal(loc_and_scale['loc'], loc_and_scale['scale']).rsample()
        return states

class ObsEncoder(nn.Module):
    """s_t ~ p(s_t | h_t, o_t)"""

    def __init__(self, h_size: int, s_size: int, activation: nn.Module, embedding_size: int, hidden_size: int, min_std_dev: float):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(
            h_size + embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.min_std_dev = min_std_dev
        self.modules = [self.fc1, self.fc2]

    def get_loc_and_scale(self, h_t: torch.Tensor, obs_emb: torch.Tensor) -> Dict:
        return self.forward(h_t, obs_emb)

    def forward(self, h_t: torch.Tensor, o_t: torch.Tensor) -> Dict:
        hidden = self.activation(self.fc1(torch.cat([h_t, o_t], dim=1)))
        loc, scale = torch.chunk(self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def sample(self, h_t, o_t):
        loc_and_scale = self.forward(h_t, o_t)
        states = Normal(loc_and_scale['loc'], loc_and_scale['scale']).rsample()
        return states

class DummyObsEncoder(nn.Module):
    def __init__(self):
        super().__init__()

class MultimodalObsEncoder:
    """s_t ~ p(s_t | h_t, o_t)"""

    def __init__(self, expert_dist, h_size: int, s_size: int, activation: nn.Module, embedding_sizes: Dict, hidden_size: int, min_std_dev: float, device):
        self.expert_dist = expert_dist
        self.modules = []
        
        self.obs_encoder = dict()
        self.obs_encoder["prior_expert"] = StochasticStateModel(h_size=h_size, hidden_size=hidden_size, activation=activation, s_size=s_size, min_std_dev=min_std_dev).to(device)
        self.modules += [self.obs_encoder["prior_expert"]]
        for name in embedding_sizes.keys():
            if self.expert_dist == "q(st|ht,ot)":
                self.obs_encoder[name] = ObsEncoder(h_size=h_size, s_size=s_size, activation=activation, embedding_size=embedding_sizes[name], hidden_size=hidden_size, min_std_dev=min_std_dev).to(device)
                self.modules += self.obs_encoder[name].modules
            elif self.expert_dist == "q(st|ot)":
                self.obs_encoder[name] = DummyObsEncoder()

    def get_loc_and_scale(self, h_t: torch.Tensor, obs_emb: Dict, t: int) -> Dict:
        locs_and_scales = dict()
        for name in self.obs_encoder.keys():
            if name == "prior_expert":
                loc_and_scale = self.obs_encoder[name](h_t=h_t)
            else:
                if self.expert_dist == "q(st|ht,ot)":
                    loc_and_scale = self.obs_encoder[name](h_t=h_t, o_t=obs_emb[name][t])
                elif self.expert_dist == "q(st|ot)":
                    loc_and_scale = dict(loc=obs_emb[name]["loc"][t], scale=obs_emb[name]["scale"][t])
            locs_and_scales[name] = loc_and_scale
        return locs_and_scales

    def get_state_dict(self):
        state_dict = dict()
        for name in self.obs_encoder.keys():
            state_dict[name] = self.obs_encoder[name].state_dict()
        return state_dict

    def _load_state_dict(self, state_dict):
        for name in state_dict.keys():
            self.obs_encoder[name].load_state_dict(state_dict[name])

    def get_model_params(self):
        model_params = list()
        for model in self.obs_encoder.values():
            model_params += list(model.parameters())
        return model_params

    def eval(self):
        for name in self.obs_encoder.keys():
            self.obs_encoder[name].eval()
    
    def train(self):
        for name in self.obs_encoder.keys():
            self.obs_encoder[name].train()

class ObsEncoder_without_ht(nn.Module):
    """s_t ~ p(s_t | o_t)"""

    def __init__(self, s_size: int, activation: nn.Module, embedding_size: int, hidden_size: int, min_std_dev: float):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(
            embedding_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * s_size)
        self.min_std_dev = min_std_dev
        self.modules = [self.fc1, self.fc2]

    def forward(self, o_t: torch.Tensor) -> Dict:
        hidden = self.activation(self.fc1(o_t))
        loc, scale = torch.chunk(self.fc2(hidden), 2, dim=1)
        scale = F.softplus(scale) + self.min_std_dev
        return {"loc": loc, "scale": scale}

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

    def sample(self, o_t):
        loc_and_scale = self.forward(o_t)
        states = Normal(loc_and_scale['loc'], loc_and_scale['scale']).rsample()
        return states

class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size: int, embedding_size: int, activation_function: str = 'relu'):
        super().__init__()
        self.embedding_size = embedding_size
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)
        self.modules = [self.fc1, self.fc2, self.fc3]

    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.act_fn(self.fc3(hidden))
        return hidden

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class ImageEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size: int, activation_function: str = 'relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 32, 4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 128, 4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, 4, stride=2),
                                        nn.ReLU(),
            )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 32, 4, stride=2, bias=False),
                                        nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                        nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                        nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                        nn.BatchNorm2d(256, affine=True, track_running_stats=True),
                                        nn.ReLU(),
            )
        else:
            raise NotImplementedError
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv, self.fc]


    def forward(self, observation: torch.Tensor):
        hidden = self.conv(observation)
        hidden = hidden.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        if not self.embedding_size == 1024:
            hidden = self.act_fn(self.fc(hidden))

        return hidden

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class ImageEncoder_84(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size: int, activation_function: str = 'relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 32, 4, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 5, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 128, 5, stride=2),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, 6, stride=2),
                                        nn.ReLU(),
            )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 32, 4, stride=2, bias=False),
                                        nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(32, 64, 5, stride=2, bias=False),
                                        nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 128, 5, stride=2, bias=False),
                                        nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                        nn.ReLU(),
                                        nn.Conv2d(128, 256, 6, stride=2, bias=False),
                                        nn.BatchNorm2d(256, affine=True, track_running_stats=True),
                                        nn.ReLU(),
            )
        else:
            raise NotImplementedError
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv, self.fc]

    def forward(self, observation: torch.Tensor):
        hidden = self.conv(observation)
        hidden = hidden.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        if not self.embedding_size == 1024:
            hidden = self.act_fn(self.fc(hidden))
        return hidden

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class ImageEncoder_128(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size: int, activation_function: str = 'relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 16, 4, stride=2, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, 4, stride=2, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, 4, stride=2, bias=True),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 256, 4, stride=2, bias=True),
                                  nn.ReLU(),
        )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 16, 4, stride=2, bias=False),
                                    nn.BatchNorm2d(16, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                    nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                    nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                    nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                    nn.BatchNorm2d(256, affine=True, track_running_stats=True),
                                    nn.ReLU(),
            )
        elif normalization == "InstanceNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 16, 4, stride=2, bias=False),
                                    nn.InstanceNorm2d(16, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                    nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                    nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                    nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                    nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
                                    nn.ReLU(),
            )
        elif normalization == "GroupNorm":
            num_groups = 4
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 16, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=16, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=32, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=64, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=128, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=256, affine=True),
                                      nn.ReLU(),
            )
        else:
            raise NotImplementedError
            
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv, self.fc]


    def forward(self, observation: torch.Tensor):
        hidden = self.conv(observation)

        hidden = hidden.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        if not self.embedding_size == 1024:
            hidden = self.act_fn(self.fc(hidden))
        return hidden

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class ImageEncoder_256(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size: int, activation_function: str = 'relu', image_dim=3, normalization=None):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        if normalization == None:
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 8, 4, stride=2, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(8, 16, 4, stride=2, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 4, stride=2, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, 4, stride=2, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 128, 4, stride=2, bias=True),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 256, 4, stride=2, bias=True),
                                    nn.ReLU(),
            )
        elif normalization == "BatchNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 8, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(8, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 16, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(16, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(32, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                      nn.BatchNorm2d(256, affine=True, track_running_stats=True),
                                      nn.ReLU(),
            )
        elif normalization == "InstanceNorm":
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 8, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(8, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 16, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(16, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(32, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                      nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
                                      nn.ReLU(),
            )
        elif normalization == "GroupNorm":
            num_groups = 4
            self.conv = nn.Sequential(nn.Conv2d(image_dim, 8, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=8, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 16, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=16, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=32, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=64, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 128, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=128, affine=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, 4, stride=2, bias=False),
                                      nn.GroupNorm(num_groups=num_groups, num_channels=256, affine=True),
                                      nn.ReLU(),
            )
        else:
            raise NotImplementedError
            
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.modules = [self.conv, self.fc]


    def forward(self, observation: torch.Tensor):
        hidden = self.conv(observation)
        hidden = hidden.reshape(-1, 1024)
        # Identity if embedding size is 1024 else linear projection
        if not self.embedding_size == 1024:
            hidden = self.act_fn(self.fc(hidden))
        return hidden

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

class SoundEncoder(nn.Module):
    def __init__(self, embbed_size = 250):
        super(SoundEncoder, self).__init__()
        self.embbed_size = embbed_size
        self.conv = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4), bias=False),
                                  nn.BatchNorm2d(64, affine=True, track_running_stats=True),
                                  nn.GLU(dim=1),
                                  nn.Conv2d(32, 128, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
                                  nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                  nn.GLU(dim=1),
                                  nn.Conv2d(64, 256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
                                  nn.BatchNorm2d(256, affine=True, track_running_stats=True),
                                  nn.GLU(dim=1),
                                  nn.Conv2d(128, 128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False),
                                  nn.BatchNorm2d(128, affine=True, track_running_stats=True),
                                  nn.GLU(dim=1),
                                  nn.Conv2d(64, 10, kernel_size=(5, 5), stride=(3, 1), padding=(1, 2), bias=False),
                                  nn.BatchNorm2d(10, affine=True, track_running_stats=True),
                                  nn.GLU(dim=1))
        if embbed_size == 250:
            self.modules = [self.conv]
        else:
            self.fc = nn.Linear(250, self.embbed_size)
            self.modules = [self.conv, self.fc]

    def forward(self, spec: torch.Tensor):
        T = spec.size()[0]
        spec = spec.unsqueeze(1)
        z = self.conv(spec)
        z = z.reshape(T, -1)
        if self.embbed_size != 250:
            z = self.fc(z)
        return z

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

# inspired by https://github.com/SamuelBroughton/StarGAN-Voice-Conversion-2/blob/master/model.py
class SoundEncoder_v2(nn.Module):
    def __init__(self, embbed_size = 250, channels_base=128):
        super(SoundEncoder_v2, self).__init__()
        self.embbed_size = embbed_size
        self.conversion_channels = int(channels_base*64)
        # Down-sampling layers
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels_base, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=int(channels_base/2), out_channels=int(channels_base*2), kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=int(channels_base*2), affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=channels_base, out_channels=int(channels_base*4), kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=int(channels_base*4), affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=int(channels_base*2), out_channels=int(channels_base*4), kernel_size=(3, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(num_features=int(channels_base*4), affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )

        # # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=self.conversion_channels,
                      out_channels=int(embbed_size/2),
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=int(embbed_size/2), affine=True),
            nn.GLU(dim=1),
        )
        
        self.modules = [self.down_sample_1, self.down_sample_2, self.down_sample_3, self.down_sample_4, self.down_conversion]

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)
        
        x = x.contiguous().view(-1, self.conversion_channels, 4)

        x = self.down_conversion(x)
        x = x.contiguous().view(-1, self.embbed_size)
        return x

    def get_state_dict(self):
        return self.state_dict()

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)

    def get_model_params(self):
        return list(self.parameters())

def build_ImageEncoder(observation_shape, visual_embedding_size, cnn_activation_function, normalization=None):
    image_size = observation_shape[1:]
    image_dim = observation_shape[0]
    if image_size == [256,256]:
        encoder = ImageEncoder_256(visual_embedding_size, cnn_activation_function, image_dim=image_dim, normalization=normalization)
    elif image_size == [128,128]:
        encoder = ImageEncoder_128(visual_embedding_size, cnn_activation_function, image_dim=image_dim, normalization=normalization)
    elif image_size == [84,84]:
        encoder = ImageEncoder_84(visual_embedding_size, cnn_activation_function, image_dim=image_dim, normalization=normalization)
    elif image_size == [64,64]:
        encoder = ImageEncoder(visual_embedding_size, cnn_activation_function, image_dim=image_dim, normalization=normalization)
    return encoder

def build_Encoder(name, observation_shapes, embedding_size, activation_function, normalization=None):
    observation_shape = observation_shapes[name]
    if "image" in name:
        encoder = build_ImageEncoder(observation_shape, embedding_size["image"], activation_function["cnn"], normalization=normalization)
    elif"sound" in name:
        encoder = SoundEncoder_v2(embbed_size=embedding_size["sound"])
    else:
        encoder = SymbolicEncoder(observation_shape[0], embedding_size["other"], activation_function["dense"])
    return encoder

class MultimodalEncoder:
    __constants__ = ['embedding_size']

    def __init__(self, 
                observation_names_enc,
                observation_shapes,
                embedding_size, 
                activation_function,
                normalization=None,
                device=torch.device("cpu")):
        self.observation_names_enc = observation_names_enc

        self.encoders = dict()
        self.modules = []
        for name in self.observation_names_enc:
            self.encoders[name] = build_Encoder(name, observation_shapes, embedding_size, activation_function, normalization).to(device)
            self.modules += self.encoders[name].modules
        
    def get_obs(self, observations, name):
        if name in observations.keys():
            return observations[name]
        elif (name == "observation") and ("image" in observations.keys()):
            return observations["image"]
        elif (name == "image") and ("observation" in observations.keys()):
            return observations["observation"]
        else:
            print("{} is missing in {}".format(name, observations.keys()))
            raise NotImplementedError

    def __call__(self, observations):
        return self.forward(observations)

    def forward(self, observations):
        hiddens = dict()
        for name in self.encoders.keys():
            _obs = self.get_obs(observations, name)
            hiddens[name] = self.encoders[name](_obs)
        return hiddens
    
    def get_state_dict(self):
        encoder_state_dict = dict()
        for name in self.encoders.keys():
            encoder_state_dict[name] = self.encoders[name].state_dict()
        return encoder_state_dict

    def _load_state_dict(self, state_dict):
        for name in state_dict.keys():
            if name == "main":
                self.load_state_dict(state_dict["main"])
            else:
                self.encoders[name].load_state_dict(state_dict[name])

    def get_model_params(self):
        model_params = list()
        for model in self.encoders.values():
            model_params += list(model.parameters())
        return model_params
    
    def eval(self):
        for name in self.encoders.keys():
            self.encoders[name].eval()

    def train(self):
        for name in self.encoders.keys():
            self.encoders[name].train()

class Mixer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation_function,
                 ):
        super(Mixer, self).__init__()
        self.act_fn = getattr(F, activation_function["fusion"])
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, hiddens):
        hidden = []
        for name in hiddens.keys():
            hidden.append(hiddens[name])
        hidden = torch.cat(hidden, dim=-1)
        hidden = self.act_fn(self.fc(hidden))
        return hidden

class MultimodalEncoderNN:
    def __init__(self, 
                observation_names_enc,
                observation_shapes,
                embedding_size, 
                activation_function,
                normalization=None,
                device=torch.device("cpu")):
        self.multimodal_encoder = MultimodalEncoder(observation_names_enc,
                                                    observation_shapes,
                                                    embedding_size,
                                                    activation_function,
                                                    normalization,
                                                    device)
        total_embedding_size = 0
        for obs_name in self.multimodal_encoder.encoders.keys():
            total_embedding_size += self.multimodal_encoder.encoders[obs_name].embedding_size
        self.mixer = Mixer(total_embedding_size, embedding_size["fusion"], activation_function["fusion"])
        self.modules = [self.fc] + self.multimodal_encoder.modules

    def __call__(self, observations):
        return self.forward(observations)

    def forward(self, observations):
        hiddens = self.multimodal_encoder(observations)
        hidden = self.mixer(hiddens)
        return hidden

    def get_state_dict(self):
        state_dict = dict()
        state_dict["multimodal_encoder"] = self.multimodal_encoder.get_state_dict()
        state_dict["mixer"] = self.mixer.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict):
        self.multimodal_encoder._load_state_dict(state_dict["multimodal_encoder"])
        self.mixer.load_state_dict(state_dict["mixer"])

    def get_model_params(self):
        model_params = list()
        model_params += self.multimodal_encoder.get_model_params()
        model_params += list(self.mixer.parameters())
        return model_params
    
    def eval(self):
        self.multimodal_encoder.eval()
        self.mixer.eval()

    def train(self):
        self.multimodal_encoder.train()
        self.mixer.train()

class MultimodalStochasticEncoder:
    __constants__ = ['embedding_size']

    def __init__(self, 
                observation_names_rec,
                observation_shapes,
                embedding_size: int, 
                state_size: int,
                hidden_size: int, 
                activation_function,
                normalization=None,
                min_std_dev=0.1,
                device=torch.device("cpu")):
        self.observation_names_rec = observation_names_rec
        self.embedding_size = embedding_size

        self.encoders = dict()
        self.obs_encoder = dict()
        self.modules = []
        for name in self.observation_names_rec:
            self.encoders[name] = build_Encoder(name, observation_shapes, embedding_size, activation_function, normalization).to(device)
            if "image" in name:
                _embedding_size = embedding_size["image"]
            elif "sound" in name:
                _embedding_size = embedding_size["sound"]
            else:
                _embedding_size = embedding_size["other"]
            self.obs_encoder[name] = ObsEncoder_without_ht(s_size=state_size, activation=getattr(F, activation_function["dense"]), embedding_size=_embedding_size, hidden_size=hidden_size, min_std_dev=min_std_dev).to(device)
            self.modules += self.encoders[name].modules
            self.modules += self.obs_encoder[name].modules
        
    def get_obs(self, observations, name):
        if name in observations.keys():
            return observations[name]
        elif (name == "observation") and ("image" in observations.keys()):
            return observations["image"]
        elif (name == "image") and ("observation" in observations.keys()):
            return observations["observation"]
        else:
            print("{} is missing".format(name))
            raise NotImplementedError

    def __call__(self, observations):
        return self.forward(observations)

    def forward(self, observations):
        locs_and_scales = dict()
        for name in self.encoders.keys():
            _obs = self.get_obs(observations, name)
            hid = self.encoders[name](_obs)
            loc_and_scale = self.obs_encoder[name](hid)
            locs_and_scales[name] = loc_and_scale
        return locs_and_scales
    
    def get_state_dict(self):
        state_dict = dict()
        encoder_state_dict = dict()
        for name in self.encoders.keys():
            encoder_state_dict[name] = self.encoders[name].state_dict()
        state_dict["encoder"] = encoder_state_dict

        obsencoder_state_dict = dict()
        for name in self.obs_encoder.keys():
            obsencoder_state_dict[name] = self.obs_encoder[name].state_dict()
        state_dict["obsencoder"] = obsencoder_state_dict

        return state_dict

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict["main"])
        for name in state_dict["encoder"].keys():      
            self.encoders[name].load_state_dict(state_dict["encoder"][name])
        for name in state_dict["obsencoder"].keys():      
            self.obs_encoder[name].load_state_dict(state_dict["obsencoder"][name])

    def get_model_params(self):
        model_params = []
        for model in self.encoders.values():
            model_params += list(model.parameters())
        for model in self.obs_encoder.values():
            model_params += list(model.parameters())
        return model_params

    def eval(self):
        for name in self.encoders.keys():
            self.encoders[name].eval()
            self.obs_encoder[name].eval()

    def train(self):
        for name in self.encoders.keys():
            self.encoders[name].train()
            self.obs_encoder[name].train()