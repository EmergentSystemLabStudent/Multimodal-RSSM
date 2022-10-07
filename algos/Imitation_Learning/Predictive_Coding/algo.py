import torch

from algos.Imitation_Learning.base.algo import IL_base
from algos.base.predictor import PredictiveCoder
from algos.base.processing import preprocess_pose_seq, postprocess_pose

class Controller:
    def __init__(self, 
                 cfg, 
                 device=torch.device("cpu"), 
                 horizon=1) -> None:
        print("Controller of Predictive Coding")
        self.cfg = cfg
        self.device = device
        self.model = IL(self.cfg, self.device)
        self.horizon = horizon
        T = horizon + 2
        self.beliefs = [torch.empty(0)] * T
        self.prior_states = [torch.empty(0)] * T
        self.posterior_states = [torch.empty(0)] * T

        self.next_init_state = None
        self.beliefs[0] = torch.zeros(1, self.cfg.model.belief_size, device=self.device)
        self.posterior_states[0] = torch.zeros(1, self.cfg.model.state_size, device=self.device)

        self.pose_prev = torch.zeros(1, *self.cfg.env.observation_shapes["pose_quat"], device=self.device)

    def __call__(self, observations, dummy_action):
        return self.get_action(observations)

    def load_model(self, model_path):
        self.model.load_model(model_path)

    def get_action(self, observations, det=True):
        observations = preprocess_pose_seq(observations, self.pose_prev)
        self.pose_prev = observations["pose_quat"]

        # make dummy action
        dummy_action = torch.zeros((1, 1, self.cfg.env.action_size), dtype=torch.float32, device=self.device)
        
        if self.next_init_state is not None:
            self.beliefs[0], self.posterior_states[0] = self.next_init_state
        _state = self.posterior_states[0]

        obs_emb = self.model.rssm.encoder(observations).unsqueeze(0)
        beliefs, _, _, _, posterior_states, posterior_mean, _ = self.model.rssm.transition_model(_state, dummy_action, self.beliefs[0], obs_emb, None, det)
        
        self.beliefs[1] = beliefs[0]
        if det:
            self.posterior_states[1] = posterior_mean[0]
        else:
            self.posterior_states[1] = posterior_states[0]

        self.next_init_state = (self.beliefs[1], self.posterior_states[1])

        action = self.model.planner.get_action(belief=self.beliefs[1], state=self.posterior_states[1], det=det)

        pose_pred = postprocess_pose(self.cfg.model.predictive_target, action)
        print(pose_pred)
        if "d_pose_quat" in self.cfg.model.predictive_target:
            pose_pred = observations["pose_quat"] + pose_pred
        print(pose_pred)
        return pose_pred

class IL(IL_base):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        print("Predictive Coding")
        
        self.planner = PredictiveCoder(self.rssm, cfg.env.action_size, cfg.model.predictive_target)
        

    def optimize_loss(self, 
                      actions, 
                      states, 
                      step):
        pass

    def load_model(self, model_path):
        model_dicts = self.load_model_dicts(model_path)
        self.rssm.load_state_dict(model_dicts["rssm"])

    def get_state_dict(self):
        state_dict = {'rssm': self.rssm.get_state_dict(),
                     }
        return state_dict
