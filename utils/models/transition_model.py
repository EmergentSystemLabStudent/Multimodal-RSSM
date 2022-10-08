from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions import Normal

from utils.models.encoder import ObsEncoder, MultimodalObsEncoder, StochasticStateModel, get_poe_state, get_mopoe_state

class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, 
                 belief_size, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 embedding_size, 
                 activation_function='relu', 
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(
            state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)

        self.stochastic_state_model = StochasticStateModel(
            h_size=belief_size, s_size=state_size, hidden_size=hidden_size, activation=self.act_fn, min_std_dev=self.min_std_dev)

        self.obs_encoder = ObsEncoder(
            h_size=belief_size, s_size=state_size, activation=self.act_fn, embedding_size=embedding_size["fusion"], hidden_size=hidden_size, min_std_dev=self.min_std_dev)

        self.modules = [self.fc_embed_state_action,
                        self.stochastic_state_model,
                        self.obs_encoder,
                        self.rnn
                        ]
        # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
        # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
        # t :  0  1  2  3  4  5
        # o :    -X--X--X--X--X-
        # a : -X--X--X--X--X-
        # n : -X--X--X--X--X-
        # pb: -X-
        # ps: -X-
        # b : -x--X--X--X--X--X-
        # s : -x--X--X--X--X--X-

    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor, obs_emb: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None, det=False) -> List[torch.Tensor]:
        '''
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
            [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(
                0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if obs_emb is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * \
                nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(
                torch.cat([_state, actions[t]], dim=1)))
            # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            # s_t ~ p(s_t | h_t) (Stochastic State Model)
            
            loc_and_scale = self.stochastic_state_model(h_t=beliefs[t + 1])
            prior_means[t + 1], prior_std_devs[t + 1] = loc_and_scale['loc'], loc_and_scale['scale']
            if det:
                prior_states[t + 1] = loc_and_scale['loc']
            else:
                prior_states[t + 1] = self.stochastic_state_model.sample(beliefs[t + 1])

            if obs_emb is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # s_t ~ q(s_t | h_t, o_t) (Observation Model)
                t_ = t - 1  # Use t_ to deal with different time indexing for observations

                loc_and_scale = self.obs_encoder.get_loc_and_scale(h_t=beliefs[t + 1], obs_emb=obs_emb[t])

                posterior_mean = loc_and_scale["loc"]
                posterior_std_dev = loc_and_scale["scale"]

                posterior_means[t + 1] = posterior_mean
                posterior_std_devs[t + 1] = posterior_std_dev
                if det:
                    posterior_states[t + 1] = posterior_mean
                else:
                    posterior_states[t + 1] = Normal(posterior_mean, posterior_std_dev).rsample()

        # Return new hidden states
        _beliefs = torch.stack(beliefs[1:], dim=0)
        _prior_states = torch.stack(prior_states[1:], dim=0)
        _prior_means = torch.stack(prior_means[1:], dim=0)
        _prior_std_devs = torch.stack(prior_std_devs[1:], dim=0)
        hidden = [_beliefs, _prior_states, _prior_means, _prior_std_devs]
        if obs_emb is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(
                posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
            hidden += [None, None]
        return hidden

    def get_state_dict(self):
        state_dict = dict(main=self.state_dict())
        state_dict["obs_encoder"] = self.obs_encoder.get_state_dict()
        return state_dict

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict["main"])
        self.obs_encoder._load_state_dict(state_dict["obs_encoder"])

    def get_model_params(self):
        model_params = list(self.parameters())
        model_params += self.obs_encoder.get_model_params()
        return model_params

    def _train(self):
        self.train()
        self.obs_encoder.train()
    
    def _eval(self):
        self.eval()
        self.obs_encoder.eval()


class MultimodalTransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, 
                 belief_size, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 observation_names_enc,
                 embedding_size, 
                 activation_function='relu', 
                 min_std_dev=0.1,
                 device=torch.device("cpu"),
                 fusion_method="MoPoE",
                 expert_dist="q(st|ht,ot)",
                 ):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(
            state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.observation_names_enc = observation_names_enc

        self.stochastic_state_model = StochasticStateModel(
            h_size=belief_size, s_size=state_size, hidden_size=hidden_size, activation=self.act_fn, min_std_dev=self.min_std_dev)

        self.modules = [self.fc_embed_state_action,
                        self.stochastic_state_model,
                        self.rnn
                        ]
        
        embedding_sizes = dict()
        for name in self.observation_names_enc:
            if "image" in name:
                _embedding_size = embedding_size["image"]
            elif "sound" in name:
                _embedding_size = embedding_size["sound"]
            else:
                _embedding_size = embedding_size["other"]
            embedding_sizes[name] = _embedding_size
        self.obs_encoder = MultimodalObsEncoder(expert_dist=expert_dist, h_size=belief_size, s_size=state_size, activation=self.act_fn, embedding_sizes=embedding_sizes, hidden_size=hidden_size, min_std_dev=self.min_std_dev, device=device)
        self.modules += self.obs_encoder.modules
        
        self.fusion_method = fusion_method
        if self.fusion_method == "MoPoE":
            self._get_posterior_states = get_mopoe_state
        else:
            self._get_posterior_states = get_poe_state

        # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
        # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
        # t :  0  1  2  3  4  5
        # o :    -X--X--X--X--X-
        # a : -X--X--X--X--X-
        # n : -X--X--X--X--X-
        # pb: -X-
        # ps: -X-
        # b : -x--X--X--X--X--X-
        # s : -x--X--X--X--X--X-

    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor, observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None, det=False) -> List[torch.Tensor]:
        '''
        generate a sequence of data

        Input: init_belief, init_state:  torch.Size([50, 200]) torch.Size([50, 30])
        Output: beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs
                    torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
            [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(
                0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        if observations is not None:
            experts_means = dict()
            experts_std_devs = dict()
            
            experts_means["prior_expert"] = [torch.empty(0)] * T
            experts_std_devs["prior_expert"] = [torch.empty(0)] * T
            for name in observations.keys():
                experts_means[name] = [torch.empty(0)] * T
                experts_std_devs[name] = [torch.empty(0)] * T

        # Loop over time sequence
        for t in range(T - 1):
            # Select appropriate previous state
            _state = prior_states[t] if observations is None else posterior_states[t]
            _state = _state if nonterminals is None else _state * \
                nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(
                torch.cat([_state, actions[t]], dim=1)))
            # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])

            # Compute state prior by applying transition dynamics
            # s_t ~ p(s_t | h_t) (Stochastic State Model)
            
            loc_and_scale = self.stochastic_state_model(h_t=beliefs[t + 1])
            prior_means[t + 1], prior_std_devs[t + 1] = loc_and_scale['loc'], loc_and_scale['scale']
            if det:
                prior_states[t + 1] = loc_and_scale['loc']
            else:
                prior_states[t + 1] = self.stochastic_state_model.sample(beliefs[t + 1])

            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                # s_t ~ q(s_t | h_t, o_t) (Observation Model)
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                loc_and_scale = self.obs_encoder.get_loc_and_scale(h_t=beliefs[t + 1], obs_emb=observations, t=t_+1)
                for name in loc_and_scale.keys():
                    experts_means[name][t + 1] = loc_and_scale[name]['loc']
                    experts_std_devs[name][t + 1] = loc_and_scale[name]['scale']

                experts_loc = dict()
                experts_scale = dict()

                for name in loc_and_scale.keys():
                    experts_loc[name] = loc_and_scale[name]['loc'].unsqueeze(0)
                    experts_scale[name] = loc_and_scale[name]['scale'].unsqueeze(0)

                posterior_state, posterior_mean, posterior_std_dev = self._get_posterior_states(experts_loc, experts_scale)
                
                posterior_means[t + 1] = posterior_mean.squeeze(0)
                posterior_std_devs[t + 1] = posterior_std_dev.squeeze(0)
                if det:
                    posterior_states[t + 1] = posterior_mean.squeeze(0)
                else:
                    posterior_states[t + 1] = posterior_state.squeeze(0)

        # Return new hidden states
        _beliefs = torch.stack(beliefs[1:], dim=0)
        _prior_states = torch.stack(prior_states[1:], dim=0)
        _prior_means = torch.stack(prior_means[1:], dim=0)
        _prior_std_devs = torch.stack(prior_std_devs[1:], dim=0)
        hidden = [_beliefs, _prior_states, _prior_means, _prior_std_devs]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(
                posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
            for name in experts_means.keys():
                experts_means[name] = torch.stack(experts_means[name][1:], dim=0)
                experts_std_devs[name] = torch.stack(experts_std_devs[name][1:], dim=0)
            hidden += [experts_means, experts_std_devs]
        return hidden

    def get_state_dict(self):
        state_dict = dict(main=self.state_dict())
        state_dict["obs_encoder"] = self.obs_encoder.get_state_dict()
        return state_dict

    def _load_state_dict(self, state_dict):
        self.load_state_dict(state_dict["main"])
        self.obs_encoder._load_state_dict(state_dict["obs_encoder"])

    def get_model_params(self):
        model_params = list(self.parameters())
        model_params += self.obs_encoder.get_model_params()
        return model_params

    def _train(self):
        self.train()
        self.obs_encoder.train()
    
    def _eval(self):
        self.eval()
        self.obs_encoder.eval()
