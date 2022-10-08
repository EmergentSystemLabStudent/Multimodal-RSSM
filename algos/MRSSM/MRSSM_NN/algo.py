from utils.models.encoder import MultimodalEncoder
from utils.models.observation_model import MultimodalObservationModel
from utils.models.reward_model import RewardModel
from utils.models.transition_model import MultimodalTransitionModel

from algos.MRSSM.base.algo import MRSSM_base

class MRSSM_NN(MRSSM_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Multimodal RSSM (NN)")

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
        self.encoder = MultimodalEncoder(observation_names_enc=self.cfg.rssm.observation_names_enc,
                                            observation_shapes=self.cfg.env.observation_shapes,
                                            embedding_size=dict(self.cfg.rssm.embedding_size), 
                                            activation_function=dict(self.cfg.rssm.activation_function),
                                            normalization=self.cfg.rssm.normalization,
                                            device=device
                                            )
