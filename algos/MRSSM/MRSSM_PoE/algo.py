import torch
from torch.distributions import Normal
from torch.nn import functional as F

import wandb

from utils.models.encoder import MultimodalEncoder, MultimodalStochasticEncoder, get_poe_state
from utils.models.observation_model import MultimodalObservationModel
from utils.models.reward_model import RewardModel
from utils.models.transition_model import MultimodalTransitionModel

from algos.MRSSM.base.algo import MRSSM_base

class MRSSM_PoE(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (PoE)")

    def _init_models(self, device):
        self.transition_model = MultimodalTransitionModel(belief_size=self.cfg.rssm.belief_size, 
                                                    state_size=self.cfg.rssm.state_size,
                                                    action_size=self.cfg.env.action_size, 
                                                    hidden_size=self.cfg.rssm.hidden_size, 
                                                    observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                    embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                    device=device,
                                                    fusion_method=self.cfg.rssm.multimodal_params.fusion_method,
                                                    expert_dist=self.cfg.rssm.multimodal_params.expert_dist,
                                                    ).to(device=device)

        self.reward_model = RewardModel(h_size=self.cfg.rssm.belief_size, s_size=self.cfg.rssm.state_size, hidden_size=self.cfg.rssm.hidden_size,
                                activation=self.cfg.rssm.activation_function.dense).to(device=device)
        
        self.observation_model = MultimodalObservationModel(observation_names_rec=self.cfg.rssm.observation_names_rec,
                                                            observation_shapes=self.cfg.env.observation_shapes,
                                                            embedding_size=dict(self.cfg.rssm.embedding_size),
                                                            belief_size=self.cfg.rssm.belief_size,
                                                            state_size=self.cfg.rssm.state_size,
                                                            hidden_size=self.cfg.rssm.hidden_size,
                                                            activation_function=dict(self.cfg.rssm.activation_function),
                                                            normalization=self.cfg.rssm.normalization,
                                                            device=device)
        
        if self.cfg.rssm.multimodal_params.expert_dist == "q(st|ht,ot)":
            self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                observation_shapes=self.cfg.env.observation_shapes,
                                                embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                activation_function=dict(self.cfg.rssm.activation_function),
                                                normalization=self.cfg.rssm.normalization,
                                                device=device
                                                )
        elif self.cfg.rssm.multimodal_params.expert_dist == "q(st|ot)":
            self.encoder = MultimodalStochasticEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                                        observation_shapes=self.cfg.env.observation_shapes,
                                                        embedding_size=dict(self.cfg.rssm.embedding_size), 
                                                        state_size=self.cfg.rssm.state_size,
                                                        hidden_size=self.cfg.rssm.hidden_size,
                                                        activation_function=dict(self.cfg.rssm.activation_function),
                                                        normalization=self.cfg.rssm.normalization,
                                                        device=device
                                                        )

    def _get_posterior_states(self,
                              states,
                              ):
        expert_means = states["expert_means"]
        expert_std_devs = states["expert_std_devs"]
        return get_poe_state(expert_means, expert_std_devs)

        # experts_loc = []
        # experts_scale = []
        # for name in expert_means.keys():
        #     experts_loc.append(expert_means[name])
        #     experts_scale.append(expert_std_devs[name])

        # experts_loc = torch.stack(experts_loc)
        # experts_scale = torch.stack(experts_scale)

        # posterior_means, posterior_std_devs = poe(experts_loc, experts_scale)
        # posterior_states = Normal(posterior_means, posterior_std_devs).rsample()
        # return posterior_states, posterior_means, posterior_std_devs
